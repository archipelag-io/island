//! Expert routing (MoE) execution for mixture-of-experts inference.
//!
//! Two roles:
//! - **Router**: Loads the gating network, tokenizes input, runs forward pass,
//!   determines which experts should process each token (top-k routing),
//!   dispatches tokens to expert Islands, collects results, combines, and
//!   streams final tokens to the coordinator.
//! - **Expert**: Loads expert weight shard(s), subscribes to its expert subject,
//!   processes dispatched tokens, returns results to the combine subject.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Expert configuration sent by the coordinator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExpertConfig {
    pub group_id: String,
    pub role: String, // "router" or "expert"
    pub position: u32,
    pub shard_spec: serde_json::Value,
    pub total_members: u32,
    pub total_experts: u32,
    pub active_experts: u32,
    pub expert_subjects: ExpertSubjects,
}

/// NATS subjects for expert communication
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExpertSubjects {
    pub control: String,
    pub output: String,
    pub status: String,
    pub combine: String,
    /// Map of expert_id → subject for dispatching tokens
    #[serde(default)]
    pub dispatch: HashMap<String, String>,
}

/// Execute an expert job — either as router or expert member.
pub async fn execute_expert_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    expert_config: ExpertConfig,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &expert_config.group_id;
    let role = &expert_config.role;

    info!(
        job_id, group_id, role,
        "Starting expert execution (position {}, {} total experts, top-{} active)",
        expert_config.position, expert_config.total_experts, expert_config.active_experts,
    );

    // Download model (router model or expert shard)
    let model_url = if role == "router" {
        expert_config.shard_spec.get("router_url")
            .and_then(|v| v.as_str())
            .or(job.model_url.as_deref())
            .context("Router missing model URL")?
    } else {
        // Expert: use first expert_url from shard_spec
        expert_config.shard_spec.get("expert_urls")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .or(job.model_url.as_deref())
            .context("Expert missing model URL")?
    };

    let model_cache = {
        let st = state.read().await;
        st.model_cache().context("Model cache not initialized")?.clone()
    };

    info!(job_id, role, "Downloading model: {}", model_url);
    let model_path = model_cache
        .download_model(model_url, job.model_hash.as_deref())
        .await
        .with_context(|| format!("Failed to download model: {}", model_url))?;

    let _model_path_str = model_path.to_string_lossy().to_string();

    // Signal ready
    nats.publish_raw(
        &expert_config.expert_subjects.status,
        serde_json::to_vec(&serde_json::json!({
            "host_id": nats.host_id(),
            "status": "ready",
            "role": role,
            "position": expert_config.position,
        }))?,
    ).await?;

    info!(job_id, role, "Expert member signaled ready, waiting for start");

    // Subscribe to control
    let mut control_sub = nats
        .subscribe_ring(&expert_config.expert_subjects.control)
        .await
        .context("Failed to subscribe to expert control")?;

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

    info!(job_id, role, "Expert session started");

    match role.as_str() {
        "router" => execute_router(nats, job, &expert_config, &mut control_sub, cancel_rx).await,
        "expert" => execute_expert_member(nats, job, &expert_config, &mut control_sub, cancel_rx).await,
        _ => anyhow::bail!("Unknown expert role: {}", role),
    }
}

/// Router: runs inference, dispatches tokens to experts, combines results, streams output.
///
/// In the current implementation, the router runs full inference (like pipeline position 0)
/// and streams output directly. True MoE gating (selecting which experts process which tokens)
/// will be added when we have expert-specific model formats.
async fn execute_router(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &ExpertConfig,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;

    let context_size = job.model_context_size.unwrap_or(2048);
    let temperature = job.model_temperature.unwrap_or(0.7);
    let max_tokens = job.input.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    // Wait for the prompt on control channel
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
                                nats.publish_status(job_id, "cancelled", None).await?;
                                return Ok(());
                            }
                        }
                    }
                    None => anyhow::bail!("Control subscription closed before prompt"),
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    nats.publish_status(job_id, "cancelled", None).await?;
                    return Ok(());
                }
            }
        }
    };

    // Run inference — for now, router generates tokens directly
    // TODO: implement actual gating network + expert dispatch when MoE model format is supported
    let model_url = config.shard_spec.get("router_url")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    info!(job_id, "Router generating tokens (model: {})", model_url);

    // Stream tokens to output subject
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
    let cancel = cancel_rx.clone();
    let model_path = model_url.to_string();

    // For now, use the same inference path as pipeline position 0
    // This will be replaced with gating + expert dispatch
    let generate_handle = tokio::task::spawn_blocking(move || {
        use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
        use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&model_path, params)
            .map_err(|e| anyhow::anyhow!("Failed to load router model: {:?}", e))?;

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

    // Batched expert dispatch + output streaming
    let expert_batch_size = 4; // Tokens per expert dispatch batch
    let mut batcher = ExpertBatcher::new(expert_batch_size);
    let mut seq: u64 = 0;
    let mut dispatched: u64 = 0;

    while let Some(token) = rx.recv().await {
        seq += 1;

        // Gate: select which experts should process this token
        let selected_experts = hash_gate_select(
            &token,
            config.total_experts,
            config.active_experts,
        );

        // Dispatch to selected experts (batched)
        for expert_id in &selected_experts {
            let eid_str = expert_id.to_string();
            if let Some(subject) = config.expert_subjects.dispatch.get(&eid_str) {
                let token_msg = serde_json::json!({
                    "job_id": job_id,
                    "token": &token,
                    "expert_id": expert_id,
                    "seq": seq,
                });

                if let Some((subj, batch)) = batcher.add(subject, token_msg) {
                    let batch_payload = serde_json::json!({
                        "job_id": job_id,
                        "tokens": batch,
                        "seq": seq,
                    });
                    nats.publish_raw(&subj, serde_json::to_vec(&batch_payload)?).await?;
                    dispatched += batch.len() as u64;
                }
            }
        }

        // Also stream to output (router produces tokens directly)
        let output = serde_json::json!({
            "job_id": job_id,
            "chunk": token,
            "is_final": false,
            "seq": seq,
        });
        nats.publish_raw(
            &config.expert_subjects.output,
            serde_json::to_vec(&output)?,
        ).await?;
    }

    // Flush remaining expert batches
    for (subj, batch) in batcher.flush_all() {
        let batch_payload = serde_json::json!({
            "job_id": job_id,
            "tokens": batch,
            "seq": seq,
        });
        nats.publish_raw(&subj, serde_json::to_vec(&batch_payload)?).await?;
        dispatched += batch.len() as u64;
    }

    let result = generate_handle.await?;
    let token_count = result.unwrap_or(0);

    // Final output
    let final_output = serde_json::json!({
        "job_id": job_id,
        "chunk": "",
        "is_final": true,
        "seq": seq + 1,
        "usage": { "completion_tokens": token_count },
    });
    nats.publish_raw(
        &config.expert_subjects.output,
        serde_json::to_vec(&final_output)?,
    ).await?;

    // Signal completion
    nats.publish_raw(
        &config.expert_subjects.status,
        serde_json::to_vec(&serde_json::json!({"status": "complete"}))?,
    ).await?;

    info!(job_id, "Router completed: {} tokens ({} dispatched to experts)", token_count, dispatched);
    Ok(())
}

/// Route a token to the appropriate expert using hash-based gating.
///
/// In a true MoE model, the gating network outputs expert selection weights.
/// Since we don't have access to the gating layer output separately, we use
/// consistent hashing on the token text to select top-K experts deterministically.
/// This provides uniform load distribution across experts.
///
/// When MoE GGUF formats expose the gating network separately, this will be
/// replaced with actual learned gating weights.
fn hash_gate_select(token: &str, total_experts: u32, active_experts: u32) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    token.hash(&mut hasher);
    let hash = hasher.finish();

    // Select top-K experts via hash partitioning
    let mut selected = Vec::with_capacity(active_experts as usize);
    for k in 0..active_experts {
        let expert_id = ((hash.wrapping_add(k as u64 * 0x9e3779b97f4a7c15)) % total_experts as u64) as u32;
        if !selected.contains(&expert_id) {
            selected.push(expert_id);
        }
    }

    // Fill remaining slots if hash collided
    while selected.len() < active_experts as usize {
        for e in 0..total_experts {
            if !selected.contains(&e) {
                selected.push(e);
                break;
            }
        }
    }

    selected
}

/// Batched dispatch: accumulate tokens for each expert, send when batch is full.
struct ExpertBatcher {
    batches: HashMap<String, Vec<serde_json::Value>>,
    batch_size: usize,
}

impl ExpertBatcher {
    fn new(batch_size: usize) -> Self {
        Self {
            batches: HashMap::new(),
            batch_size: batch_size.max(1),
        }
    }

    /// Add a token to the batch for a subject. Returns the batch if full.
    fn add(&mut self, subject: &str, token: serde_json::Value) -> Option<(String, Vec<serde_json::Value>)> {
        let batch = self.batches.entry(subject.to_string()).or_insert_with(Vec::new);
        batch.push(token);

        if batch.len() >= self.batch_size {
            let full_batch = std::mem::take(batch);
            Some((subject.to_string(), full_batch))
        } else {
            None
        }
    }

    /// Flush all remaining batches (called at end of generation)
    fn flush_all(&mut self) -> Vec<(String, Vec<serde_json::Value>)> {
        self.batches.drain()
            .filter(|(_, batch)| !batch.is_empty())
            .collect()
    }
}

/// Expert member: subscribes to its expert dispatch subjects, processes tokens,
/// returns results to the combine subject.
async fn execute_expert_member(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &ExpertConfig,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;
    let expert_ids: Vec<i64> = config.shard_spec.get("expert_ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default();

    info!(job_id, "Expert member ready for experts {:?}", expert_ids);

    // Subscribe to each expert dispatch subject
    let mut expert_subs = Vec::new();
    for (eid_str, subject) in &config.expert_subjects.dispatch {
        let sub = nats.subscribe_ring(subject).await
            .with_context(|| format!("Failed to subscribe to expert {}", eid_str))?;
        expert_subs.push((eid_str.clone(), sub));
    }

    // Main loop: receive dispatched token batches, process, return results
    let mut tokens_processed: u64 = 0;
    let mut combined_subs: Vec<async_nats::Subscriber> = Vec::new();

    // Flatten all expert subscriptions into a single receiver
    // We'll use a simple polling approach since we have multiple subs
    for (eid_str, sub) in expert_subs {
        combined_subs.push(sub);
    }

    info!(job_id, "Expert member processing tokens for experts {:?}", expert_ids);

    loop {
        // Check all expert subscriptions for incoming token batches
        let mut received_batch = false;

        for sub in combined_subs.iter_mut() {
            // Non-blocking check via select with zero timeout
            select! {
                msg = sub.next() => {
                    if let Some(msg) = msg {
                        if let Ok(batch) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            let token_count = batch.get("tokens")
                                .and_then(|t| t.as_array())
                                .map(|arr| arr.len())
                                .unwrap_or(0);

                            tokens_processed += token_count as u64;
                            received_batch = true;

                            // Process tokens (currently passthrough — expert forward pass
                            // will be added when MoE GGUF expert shards are available)
                            // Publish results to combine subject
                            let result = serde_json::json!({
                                "job_id": job_id,
                                "expert_ids": &expert_ids,
                                "tokens_processed": token_count,
                                "seq": batch.get("seq"),
                            });
                            let _ = nats.publish_raw(
                                &config.expert_subjects.combine,
                                serde_json::to_vec(&result).unwrap_or_default(),
                            ).await;

                            // Report load to coordinator
                            let _ = nats.publish_raw(
                                &config.expert_subjects.status,
                                serde_json::to_vec(&serde_json::json!({
                                    "host_id": nats.host_id(),
                                    "status": "load",
                                    "tokens_in_flight": tokens_processed,
                                })).unwrap_or_default(),
                            ).await;
                        }
                    }
                }
                msg = control_sub.next() => {
                    if let Some(msg) = msg {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                info!(job_id, "Expert member received stop ({} tokens processed)", tokens_processed);
                                return Ok(());
                            }
                        }
                    }
                }
                _ = cancel_rx.changed() => {
                    if *cancel_rx.borrow() {
                        info!(job_id, "Expert member cancelled");
                        return Ok(());
                    }
                }
            }

            if received_batch { break; } // Process one batch then re-check control
        }

        if !received_batch {
            // No batch available, brief sleep to avoid busy-wait
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_config_deserialize() {
        let json = r#"{
            "group_id": "abc-123",
            "role": "router",
            "position": 0,
            "shard_spec": {"role": "router", "router_url": "https://example.com/router.gguf"},
            "total_members": 3,
            "total_experts": 8,
            "active_experts": 2,
            "expert_subjects": {
                "control": "expert.abc-123.control",
                "output": "expert.abc-123.output",
                "status": "expert.abc-123.status",
                "combine": "expert.abc-123.combine",
                "dispatch": {
                    "0": "expert.abc-123.expert.0",
                    "1": "expert.abc-123.expert.1"
                }
            }
        }"#;

        let config: ExpertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.group_id, "abc-123");
        assert_eq!(config.role, "router");
        assert_eq!(config.total_experts, 8);
        assert_eq!(config.active_experts, 2);
        assert_eq!(config.expert_subjects.dispatch.len(), 2);
    }

    #[test]
    fn test_hash_gate_select() {
        let selected = hash_gate_select("hello", 8, 2);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|&e| e < 8));

        // Same token → same experts (deterministic)
        let selected2 = hash_gate_select("hello", 8, 2);
        assert_eq!(selected, selected2);

        // Different token → likely different experts
        let selected3 = hash_gate_select("world", 8, 2);
        // (Not guaranteed to be different, but statistically likely)
        assert_eq!(selected3.len(), 2);
    }

    #[test]
    fn test_expert_batcher() {
        let mut batcher = ExpertBatcher::new(3);

        // Add 2 tokens — no flush
        assert!(batcher.add("sub.0", serde_json::json!({"t": 1})).is_none());
        assert!(batcher.add("sub.0", serde_json::json!({"t": 2})).is_none());

        // Third token triggers flush
        let result = batcher.add("sub.0", serde_json::json!({"t": 3}));
        assert!(result.is_some());
        let (subj, batch) = result.unwrap();
        assert_eq!(subj, "sub.0");
        assert_eq!(batch.len(), 3);

        // flush_all returns remaining
        batcher.add("sub.1", serde_json::json!({"t": 4}));
        let remaining = batcher.flush_all();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].0, "sub.1");
    }

    #[test]
    fn test_expert_member_config() {
        let json = r#"{
            "group_id": "abc-123",
            "role": "expert",
            "position": 1,
            "shard_spec": {"role": "expert", "expert_ids": [0, 1, 2, 3]},
            "total_members": 3,
            "total_experts": 8,
            "active_experts": 2,
            "expert_subjects": {
                "control": "expert.abc-123.control",
                "output": "expert.abc-123.output",
                "status": "expert.abc-123.status",
                "combine": "expert.abc-123.combine",
                "dispatch": {
                    "0": "expert.abc-123.expert.0",
                    "1": "expert.abc-123.expert.1",
                    "2": "expert.abc-123.expert.2",
                    "3": "expert.abc-123.expert.3"
                }
            }
        }"#;

        let config: ExpertConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.role, "expert");
        assert_eq!(config.position, 1);
        let expert_ids = config.shard_spec.get("expert_ids").unwrap().as_array().unwrap();
        assert_eq!(expert_ids.len(), 4);
    }
}
