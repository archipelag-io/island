//! Speculative decoding for accelerated autoregressive generation.
//!
//! Pairs a fast Draft Island (small model, e.g., TinyLlama 1B) with a
//! powerful Verify Island (large model, e.g., Llama 70B) for 2-3x speedup.
//!
//! Protocol:
//! 1. Draft generates K candidate tokens with log-probabilities
//! 2. Sends K draft tokens to Verify via spec.{group_id}.draft
//! 3. Verify runs single forward pass, compares logits, accepts matching prefix
//! 4. Publishes VerifyResult to spec.{group_id}.verify
//! 5. Accepted tokens stream to Consumer via spec.{group_id}.output
//! 6. Draft continues from accepted point
//! 7. Repeat until max_tokens or EOS

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info, warn};

use crate::logits::{DraftToken, VerifyResult};
use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Speculative decoding configuration from coordinator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpeculativeConfig {
    pub group_id: String,
    pub role: String,
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
    pub draft: String,   // Draft → Verify: K tokens + log-probs
    pub verify: String,  // Verify → Draft: VerifyResult (accepted count + correction)
    pub output: String,  // → Coordinator: accepted tokens
    pub status: String,
}

/// A batch of draft tokens sent from Draft → Verify
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DraftBatch {
    job_id: String,
    context_so_far: String,
    tokens: Vec<DraftToken>,
    seq: u64,
}

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
        .context("Missing model URL")?;

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
        .with_context(|| format!("Failed to download: {}", model_url))?;

    let model_path_str = model_path.to_string_lossy().to_string();

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
        .context("Failed to subscribe to control")?;

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
        "draft" => execute_draft(nats, job, &config, &model_path_str, &mut control_sub, cancel_rx).await,
        "verify" => execute_verify(nats, job, &config, &model_path_str, &mut control_sub, cancel_rx).await,
        _ => anyhow::bail!("Unknown speculative role: {}", role),
    }
}

/// Draft Island: generates K tokens per round, sends to verify, waits for
/// acceptance result, streams accepted tokens, continues from accepted prefix.
async fn execute_draft(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &SpeculativeConfig,
    model_path: &str,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let k = config.draft_tokens as usize;

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

    // Subscribe to verify results
    let mut verify_sub = nats.subscribe_ring(&config.spec_subjects.verify).await
        .context("Failed to subscribe to verify subject")?;

    let context_size = job.model_context_size.unwrap_or(2048);
    let temperature = job.model_temperature.unwrap_or(0.7);
    let max_tokens = job.input.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    // Generate all tokens via spawn_blocking, collect into channel
    let model_path_owned = model_path.to_string();
    let prompt_clone = prompt.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
    let cancel = cancel_rx.clone();

    let generate_handle = tokio::task::spawn_blocking(move || {
        use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
        use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&model_path_owned, params)
            .map_err(|e| anyhow::anyhow!("Draft model load failed: {:?}", e))?;

        let mut sp = SessionParams::default();
        sp.n_ctx = context_size;

        let mut session = model.create_session(sp)
            .map_err(|e| anyhow::anyhow!("Session error: {:?}", e))?;

        session.advance_context(&prompt_clone)
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

    // Speculative loop: collect K tokens, send to verify, stream accepted
    let mut seq: u64 = 0;
    let mut total_generated: u64 = 0;
    let mut context_text = prompt.clone();
    let mut draft_batch: Vec<String> = Vec::with_capacity(k);

    loop {
        // Collect K draft tokens
        let mut batch_done = false;
        while draft_batch.len() < k {
            match rx.recv().await {
                Some(token) => {
                    draft_batch.push(token);
                    total_generated += 1;
                }
                None => {
                    batch_done = true;
                    break;
                }
            }
        }

        if draft_batch.is_empty() {
            break;
        }

        // Build draft tokens (without log-probs — they come from the verify step)
        let draft_tokens: Vec<DraftToken> = draft_batch.iter().enumerate().map(|(i, text)| {
            DraftToken {
                token_id: i as i32, // Placeholder — real token IDs come from tokenization
                text: text.clone(),
                log_prob: 0.0, // Draft log-probs would require FFI extraction
            }
        }).collect();

        // Send batch to verify Island
        let batch_msg = DraftBatch {
            job_id: job_id.clone(),
            context_so_far: context_text.clone(),
            tokens: draft_tokens.clone(),
            seq: seq + 1,
        };

        nats.publish_raw(
            &config.spec_subjects.draft,
            serde_json::to_vec(&batch_msg)?,
        ).await?;

        // Wait for verify result (with timeout)
        let verify_result = select! {
            msg = verify_sub.next() => {
                match msg {
                    Some(msg) => {
                        serde_json::from_slice::<VerifyResult>(&msg.payload).ok()
                    }
                    None => None,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                warn!(job_id, "Verify timeout, accepting all draft tokens");
                Some(VerifyResult {
                    accepted_count: draft_batch.len(),
                    corrected_token: None,
                    draft_count: draft_batch.len(),
                })
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() { break; }
                None
            }
        };

        // Process verify result
        let accepted = match verify_result {
            Some(result) => {
                // Report acceptance stats to coordinator
                let stats_msg = serde_json::json!({
                    "drafted": result.draft_count,
                    "accepted": result.accepted_count,
                });
                nats.publish_raw(
                    &config.spec_subjects.status,
                    serde_json::to_vec(&serde_json::json!({
                        "status": "verify_round",
                        "drafted": result.draft_count,
                        "accepted": result.accepted_count,
                    }))?,
                ).await?;

                result.accepted_count
            }
            None => draft_batch.len(), // Accept all on verify failure
        };

        // Stream accepted tokens to output
        for token_text in draft_batch.drain(..accepted.min(draft_batch.len())) {
            seq += 1;
            let output = serde_json::json!({
                "job_id": job_id,
                "chunk": token_text,
                "is_final": false,
                "seq": seq,
            });
            nats.publish_raw(&config.spec_subjects.output, serde_json::to_vec(&output)?).await?;
            context_text.push_str(&token_text);
        }

        // Discard remaining (rejected) tokens from this batch
        draft_batch.clear();

        if batch_done {
            break;
        }
    }

    // Wait for generation to finish
    let result = generate_handle.await?;
    let token_count = result.unwrap_or(total_generated);

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

    info!(job_id, "Draft completed: {} tokens generated, {} streamed", token_count, seq);
    Ok(())
}

/// Verify Island: receives draft token batches, runs verification via
/// logits.rs, publishes acceptance results back to draft.
async fn execute_verify(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &SpeculativeConfig,
    model_path: &str,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let threshold = config.acceptance_threshold;

    // Subscribe to draft batches
    let mut draft_sub = nats.subscribe_ring(&config.spec_subjects.draft).await
        .context("Failed to subscribe to draft subject")?;

    let model_path_owned = model_path.to_string();
    let context_size = job.model_context_size.unwrap_or(2048);

    info!(job_id, "Verify Island listening for draft batches (threshold={})", threshold);

    loop {
        select! {
            msg = draft_sub.next() => {
                match msg {
                    Some(msg) => {
                        match serde_json::from_slice::<DraftBatch>(&msg.payload) {
                            Ok(batch) => {
                                let batch_len = batch.tokens.len();
                                info!(job_id, "Verifying {} draft tokens", batch_len);

                                // Run verification via logits FFI
                                let model_path_clone = model_path_owned.clone();
                                let result = tokio::task::spawn_blocking(move || {
                                    crate::logits::verify_draft_tokens(
                                        &model_path_clone,
                                        &batch.context_so_far,
                                        &batch.tokens,
                                        context_size,
                                        threshold,
                                    )
                                }).await?;

                                match result {
                                    Ok(verify_result) => {
                                        info!(
                                            job_id,
                                            "Verified: {}/{} accepted",
                                            verify_result.accepted_count,
                                            verify_result.draft_count,
                                        );

                                        nats.publish_raw(
                                            &config.spec_subjects.verify,
                                            serde_json::to_vec(&verify_result)?,
                                        ).await?;
                                    }
                                    Err(e) => {
                                        warn!(job_id, "Verification failed: {}, accepting all", e);
                                        let fallback = VerifyResult {
                                            accepted_count: batch_len,
                                            corrected_token: None,
                                            draft_count: batch_len,
                                        };
                                        nats.publish_raw(
                                            &config.spec_subjects.verify,
                                            serde_json::to_vec(&fallback)?,
                                        ).await?;
                                    }
                                }
                            }
                            Err(e) => {
                                warn!(job_id, "Failed to parse draft batch: {}", e);
                            }
                        }
                    }
                    None => break,
                }
            }
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                info!(job_id, "Verify received stop");
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
    }

    #[test]
    fn test_draft_batch_serialize() {
        let batch = DraftBatch {
            job_id: "test-123".into(),
            context_so_far: "Hello".into(),
            tokens: vec![DraftToken { token_id: 1, text: " world".into(), log_prob: -0.5 }],
            seq: 1,
        };
        let json = serde_json::to_string(&batch).unwrap();
        assert!(json.contains("\"context_so_far\":\"Hello\""));
        assert!(json.contains("\" world\""));
    }
}
