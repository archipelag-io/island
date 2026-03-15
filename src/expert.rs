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

    let mut seq: u64 = 0;
    while let Some(token) = rx.recv().await {
        seq += 1;
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

    info!(job_id, "Router completed: {} tokens", token_count);
    Ok(())
}

/// Expert member: subscribes to its expert dispatch subjects, processes tokens,
/// returns results. Currently a passthrough — true expert weight loading will
/// be added when MoE-specific GGUF formats are supported.
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

    // Main loop: receive dispatched tokens, process, return results
    // For now, experts are idle since the router generates directly.
    // When MoE dispatch is implemented, this loop will:
    // 1. Receive token batches from router
    // 2. Run expert forward pass on the tokens
    // 3. Publish results to combine subject
    loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            match ctrl.get("action").and_then(|a| a.as_str()) {
                                Some("stop") => {
                                    info!(job_id, "Expert member received stop");
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(job_id, "Expert member cancelled");
                    break;
                }
            }
        }
    }

    Ok(())
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
