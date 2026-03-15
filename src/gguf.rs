//! Native GGUF runtime for executing llama.cpp workloads
//!
//! Downloads GGUF model files, loads them with llama_cpp, and streams
//! token-by-token output back through NATS.

use anyhow::{Context, Result};
use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use std::sync::Arc;
use tokio::sync::{watch, RwLock};
use tracing::info;

use crate::config::AgentConfig;
use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Execute a GGUF/llmcpp workload
pub async fn execute_gguf_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    _config: &AgentConfig,
    job: &AssignJob,
    cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let model_url = job
        .model_url
        .as_ref()
        .context("GGUF workload missing model_url")?;

    let context_size = job.model_context_size.unwrap_or(2048);
    let temperature = job.model_temperature.unwrap_or(0.7);
    let max_tokens = job
        .input
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;

    info!(
        "Executing GGUF workload: {} (ctx={}, temp={}, max_tokens={})",
        model_url, context_size, temperature, max_tokens
    );

    // Download model via cache
    let model_cache = {
        let st = state.read().await;
        st.model_cache()
            .context("Model cache not initialized")?
            .clone()
    };

    let model_path = model_cache
        .download_model(model_url, job.model_hash.as_deref())
        .await
        .with_context(|| format!("Failed to download GGUF model: {}", model_url))?;

    // Check cancellation
    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // Extract prompt from input
    let prompt = extract_prompt(&job.input)?;

    // Load model and generate in spawn_blocking, stream tokens via channel
    let model_path_str = model_path.to_string_lossy().to_string();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);

    let cancel = cancel_rx.clone();
    let generate_handle = tokio::task::spawn_blocking(move || {
        // Load model
        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&model_path_str, params)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {:?}", e))?;

        // Create session with context size
        let mut session_params = SessionParams::default();
        session_params.n_ctx = context_size;

        let mut session = model
            .create_session(session_params)
            .map_err(|e| anyhow::anyhow!("Failed to create llama session: {:?}", e))?;

        // Feed prompt
        session
            .advance_context(&prompt)
            .map_err(|e| anyhow::anyhow!("Failed to feed prompt: {:?}", e))?;

        // Configure sampler with temperature
        let sampler = StandardSampler::new_softmax(
            vec![
                SamplerStage::RepetitionPenalty {
                    repetition_penalty: 1.1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    last_n: 64,
                },
                SamplerStage::TopP(0.9),
                SamplerStage::Temperature(temperature),
            ],
            1, // min_keep
        );

        // Generate tokens
        let completions = session
            .start_completing_with(sampler, max_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to start completion: {:?}", e))?;
        let mut token_count: u64 = 0;
        let mut output = String::new();

        for token_str in completions.into_strings() {
            let token_str = token_str.to_string();
            // Check cancellation
            if *cancel.borrow() {
                break;
            }

            output.push_str(&token_str);
            token_count += 1;

            // Send token to async side; if receiver dropped, stop
            if tx.blocking_send(token_str).is_err() {
                break;
            }
        }

        Ok::<(String, u64), anyhow::Error>((output, token_count))
    });

    // Stream tokens as they arrive
    let mut seq: u64 = 0;
    while let Some(token) = rx.recv().await {
        nats.publish_output(job_id, seq, &token, false).await?;
        seq += 1;
    }

    // Wait for generation to complete
    let result = generate_handle.await?;

    match result {
        Ok((_output, token_count)) => {
            let final_event = serde_json::json!({
                "type": "done",
                "usage": { "completion_tokens": token_count }
            });
            nats.publish_output(job_id, seq, &final_event.to_string(), true)
                .await?;
            nats.publish_status(job_id, "succeeded", None).await?;
            info!("GGUF job {} completed: {} tokens", job_id, token_count);
        }
        Err(e) => {
            nats.publish_status(job_id, "failed", Some(format!("GGUF error: {}", e)))
                .await?;
        }
    }

    Ok(())
}

/// Extract prompt string from job input JSON
pub fn extract_prompt(input: &serde_json::Value) -> Result<String> {
    // Try "prompt" field first
    if let Some(prompt) = input.get("prompt").and_then(|v| v.as_str()) {
        return Ok(prompt.to_string());
    }

    // Try "messages" array (OpenAI chat format)
    if let Some(messages) = input.get("messages").and_then(|v| v.as_array()) {
        let mut prompt = String::new();
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            match role {
                "system" => {
                    prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", content));
                }
                "user" => {
                    prompt.push_str(&format!("[INST] {} [/INST]", content));
                }
                "assistant" => {
                    prompt.push_str(&format!("{} </s>", content));
                }
                _ => {
                    prompt.push_str(&format!("[INST] {} [/INST]", content));
                }
            }
        }
        return Ok(prompt);
    }

    // Try "text" field
    if let Some(text) = input.get("text").and_then(|v| v.as_str()) {
        return Ok(text.to_string());
    }

    anyhow::bail!("GGUF workload requires 'prompt', 'messages', or 'text' in input")
}
