//! Native GGUF runtime for executing llama.cpp workloads
//!
//! Downloads GGUF model files, loads them with llama-cpp-2, and streams
//! token-by-token output back through NATS.

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
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
    let model_path_clone = model_path.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);

    let cancel = cancel_rx.clone();
    let generate_handle = tokio::task::spawn_blocking(move || {
        // Initialize backend
        let backend = LlamaBackend::init()
            .map_err(|e| anyhow::anyhow!("Failed to init llama backend: {:?}", e))?;

        // Load model
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path_clone, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {:?}", e))?;

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(context_size));
        let mut ctx = model
            .new_context(&backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create llama context: {:?}", e))?;

        // Tokenize prompt
        let tokens = model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {:?}", e))?;

        // Process prompt in batches
        let n_ctx = ctx.n_ctx() as usize;
        let tokens = if tokens.len() > n_ctx.saturating_sub(4) {
            tokens[..n_ctx.saturating_sub(4)].to_vec()
        } else {
            tokens
        };

        let mut batch = LlamaBatch::new(512, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(token, i as i32, &[0], is_last)
                .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {:?}", e))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Failed to decode prompt: {:?}", e))?;

        // Set up sampler chain
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ]);
        let mut sampler = sampler;

        // Generate tokens
        let mut n_cur = tokens.len();
        let mut token_count: u64 = 0;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let eos = model.token_eos();

        for _ in 0..max_tokens {
            // Check cancellation
            if *cancel.borrow() {
                break;
            }

            // Sample next token
            let new_token = sampler.sample(&ctx, -1);
            sampler.accept(new_token);

            // Check for end of generation
            if new_token == eos {
                break;
            }

            // Convert token to string piece
            let piece = model
                .token_to_piece(new_token, &mut decoder, true, None)
                .unwrap_or_default();

            if !piece.is_empty() {
                token_count += 1;
                if tx.blocking_send(piece).is_err() {
                    break;
                }
            }

            // Prepare next batch
            batch.clear();
            batch.add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {:?}", e))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Failed to decode token: {:?}", e))?;
        }

        Ok::<u64, anyhow::Error>(token_count)
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
        Ok(token_count) => {
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
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", content));
                }
                "user" => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", content));
                }
                "assistant" => {
                    prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", content));
                }
                _ => {
                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
            }
        }
        // Add assistant opening
        prompt.push_str("<|im_start|>assistant\n");
        return Ok(prompt);
    }

    // Try "text" field
    if let Some(text) = input.get("text").and_then(|v| v.as_str()) {
        return Ok(text.to_string());
    }

    anyhow::bail!("GGUF workload requires 'prompt', 'messages', or 'text' in input")
}
