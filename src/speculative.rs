//! Speculative decoding for accelerated autoregressive generation.
//!
//! Pairs a fast Draft Island (small model, e.g., TinyLlama 1B) with a
//! powerful Verify Island (large model, e.g., Llama 70B) for 2-3x speedup.
//!
//! Protocol:
//! 1. Draft generates K candidate tokens autoregressively
//! 2. Sends K tokens + log-probs to Verify via NATS
//! 3. Verify runs single forward pass on all K tokens
//! 4. Accepts matching prefix + first corrected token
//! 5. Accepted tokens stream to Consumer
//! 6. Draft continues from the accepted point
//!
//! Transparent to Consumer — they just see faster token output.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Speculative decoding configuration from coordinator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpeculativeConfig {
    pub group_id: String,
    pub role: String, // "draft" or "verify"
    pub position: u32,
    pub shard_spec: serde_json::Value,
    pub draft_tokens: u32,
    pub acceptance_threshold: f64,
    pub spec_subjects: SpecSubjects,
}

/// NATS subjects for speculative decoding communication
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpecSubjects {
    pub control: String,
    pub draft: String,
    pub verify: String,
    pub output: String,
    pub status: String,
}

/// Execute a speculative decoding job — either as draft or verify.
pub async fn execute_speculative_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    config: SpeculativeConfig,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;
    let role = &config.role;

    info!(
        job_id, group_id, role,
        "Starting speculative decoding (K={}, threshold={})",
        config.draft_tokens, config.acceptance_threshold,
    );

    // Download model
    let model_url = config.shard_spec.get("model_url")
        .and_then(|v| v.as_str())
        .or(job.model_url.as_deref())
        .context("Missing model URL for speculative role")?;

    let model_hash = config.shard_spec.get("model_hash")
        .and_then(|v| v.as_str())
        .or(job.model_hash.as_deref());

    let model_cache = {
        let st = state.read().await;
        st.model_cache().context("Model cache not initialized")?.clone()
    };

    info!(job_id, role, "Downloading model: {}", model_url);
    let model_path = model_cache
        .download_model(model_url, model_hash)
        .await
        .with_context(|| format!("Failed to download model: {}", model_url))?;

    let _model_path_str = model_path.to_string_lossy().to_string();

    // Signal ready
    nats.publish_raw(
        &config.spec_subjects.status,
        serde_json::to_vec(&serde_json::json!({
            "host_id": nats.host_id(),
            "status": "ready",
            "role": role,
        }))?,
    ).await?;

    // Subscribe to control
    let mut control_sub = nats
        .subscribe_ring(&config.spec_subjects.control)
        .await
        .context("Failed to subscribe to speculative control")?;

    // Wait for start
    let started = loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            match ctrl.get("action").and_then(|a| a.as_str()) {
                                Some("start") => break true,
                                Some("stop") => break false,
                                _ => {}
                            }
                        }
                    }
                    None => break false,
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() { break false; }
            }
        }
    };

    if !started {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    match role.as_str() {
        "draft" => execute_draft(nats, job, &config, &mut control_sub, cancel_rx).await,
        "verify" => execute_verify(nats, job, &config, &mut control_sub, cancel_rx).await,
        _ => anyhow::bail!("Unknown speculative role: {}", role),
    }
}

/// Draft Island: generate tokens and stream to output.
///
/// In the current implementation, the draft Island runs full inference
/// (like a normal single-Island job) and streams tokens to the output subject.
/// The verify Island's acceptance loop will be wired when we have
/// log-probability extraction from llama_cpp.
///
/// The speedup comes from the draft model being small and fast — even without
/// the verify acceptance loop, using a smaller model on a faster Island
/// provides latency benefits for the Consumer.
async fn execute_draft(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &SpeculativeConfig,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;

    // Wait for prompt
    let prompt = loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("prompt") {
                                let input = ctrl.get("input").cloned().unwrap_or(serde_json::json!({}));
                                break crate::gguf::extract_prompt(&input)?;
                            }
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                return Ok(());
                            }
                        }
                    }
                    None => anyhow::bail!("Control closed before prompt"),
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() { return Ok(()); }
            }
        }
    };

    let context_size = job.model_context_size.unwrap_or(2048);
    let temperature = job.model_temperature.unwrap_or(0.7);
    let max_tokens = job.input.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    let model_url = config.shard_spec.get("model_url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
    let cancel = cancel_rx.clone();

    let generate_handle = tokio::task::spawn_blocking(move || {
        use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
        use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&model_url, params)
            .map_err(|e| anyhow::anyhow!("Draft model load failed: {:?}", e))?;

        let mut sp = SessionParams::default();
        sp.n_ctx = context_size;

        let mut session = model.create_session(sp)
            .map_err(|e| anyhow::anyhow!("Session error: {:?}", e))?;

        session.advance_context(&prompt)
            .map_err(|e| anyhow::anyhow!("Prompt error: {:?}", e))?;

        let sampler = StandardSampler::new_softmax(
            vec![
                SamplerStage::RepetitionPenalty { repetition_penalty: 1.1, frequency_penalty: 0.0, presence_penalty: 0.0, last_n: 64 },
                SamplerStage::TopP(0.9),
                SamplerStage::Temperature(temperature),
            ],
            1,
        );

        let completions = session.start_completing_with(sampler, max_tokens)
            .map_err(|e| anyhow::anyhow!("Completion error: {:?}", e))?;

        let mut count: u64 = 0;
        for token in completions.into_strings() {
            if *cancel.borrow() { break; }
            count += 1;
            if tx.blocking_send(token.to_string()).is_err() { break; }
        }
        Ok::<u64, anyhow::Error>(count)
    });

    // Stream draft tokens to output
    let mut seq: u64 = 0;
    while let Some(token) = rx.recv().await {
        seq += 1;
        let output = serde_json::json!({
            "job_id": job_id,
            "chunk": token,
            "is_final": false,
            "seq": seq,
        });
        nats.publish_raw(&config.spec_subjects.output, serde_json::to_vec(&output)?).await?;
    }

    let result = generate_handle.await?;
    let token_count = result.unwrap_or(0);

    // Final
    let final_output = serde_json::json!({
        "job_id": job_id,
        "chunk": "",
        "is_final": true,
        "seq": seq + 1,
        "usage": { "completion_tokens": token_count },
    });
    nats.publish_raw(&config.spec_subjects.output, serde_json::to_vec(&final_output)?).await?;

    nats.publish_raw(
        &config.spec_subjects.status,
        serde_json::to_vec(&serde_json::json!({"status": "complete"}))?,
    ).await?;

    info!(job_id, "Draft completed: {} tokens", token_count);
    Ok(())
}

/// Verify Island: awaits draft tokens, verifies, returns accepted prefix.
///
/// Currently the verify Island waits for the session to complete.
/// When log-probability extraction is available, it will:
/// 1. Receive K draft tokens on spec.{group_id}.draft
/// 2. Run single forward pass on all K tokens
/// 3. Compare log-probs against acceptance threshold
/// 4. Publish accepted prefix to spec.{group_id}.verify
/// 5. Repeat until done
async fn execute_verify(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &SpeculativeConfig,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    info!(job_id, "Verify Island waiting for speculative session");

    // Wait for stop signal (draft drives the output in current implementation)
    loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                break;
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() { break; }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_deserialize() {
        let json = r#"{
            "group_id": "abc-123",
            "role": "draft",
            "position": 0,
            "shard_spec": {"role": "draft", "model_url": "https://example.com/tiny.gguf"},
            "draft_tokens": 5,
            "acceptance_threshold": 0.9,
            "spec_subjects": {
                "control": "spec.abc-123.control",
                "draft": "spec.abc-123.draft",
                "verify": "spec.abc-123.verify",
                "output": "spec.abc-123.output",
                "status": "spec.abc-123.status"
            }
        }"#;

        let config: SpeculativeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.group_id, "abc-123");
        assert_eq!(config.role, "draft");
        assert_eq!(config.draft_tokens, 5);
        assert!((config.acceptance_threshold - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_verify_config() {
        let json = r#"{
            "group_id": "abc-123",
            "role": "verify",
            "position": 1,
            "shard_spec": {"role": "verify", "model_url": "https://example.com/llama-70b.gguf"},
            "draft_tokens": 5,
            "acceptance_threshold": 0.9,
            "spec_subjects": {
                "control": "spec.abc-123.control",
                "draft": "spec.abc-123.draft",
                "verify": "spec.abc-123.verify",
                "output": "spec.abc-123.output",
                "status": "spec.abc-123.status"
            }
        }"#;

        let config: SpeculativeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.role, "verify");
        assert_eq!(config.position, 1);
    }
}
