//! Pipeline execution for sharded model inference.
//!
//! When a model is too large for a single Island, it can be split across
//! multiple Islands in a pipeline. Each Island holds a subset of layers,
//! and activations flow sequentially through the chain.
//!
//! Activation wire format:
//! - 12-byte header: magic "ARCP" (4), dtype (2), ndims (2), shape dims (4)
//! - Job ID (36 bytes, UUID string)
//! - Raw activation data (f16 bytes)

use anyhow::{Context, Result};
use futures_util::StreamExt;
use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Magic bytes for activation header
const ACTIVATION_MAGIC: &[u8; 4] = b"ARCP";

/// Default number of tokens to batch before sending an activation message.
/// Higher values reduce NATS overhead but add latency. 1 = no batching (stream per-token).
const DEFAULT_MICROBATCH_SIZE: usize = 1;

/// Pipeline configuration sent by the coordinator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineConfig {
    pub group_id: String,
    pub position: u32,
    pub shard_spec: ShardSpec,
    pub total_members: u32,
    pub ring_subjects: RingSubjects,
}

/// Shard specification — which layers this Island is responsible for
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardSpec {
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default)]
    pub shard_url: Option<String>,
}

/// NATS subjects for ring communication
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RingSubjects {
    pub activate: String,
    pub control: String,
    pub output: String,
    pub status: String,
    #[serde(default)]
    pub next_activate: Option<String>,
}

/// Header prepended to activation data on the wire
#[derive(Debug, Clone)]
pub struct ActivationHeader {
    pub dtype: u16,    // 0 = f16, 1 = f32, 2 = bf16
    pub ndims: u16,    // number of dimensions
    pub dim0: u32,     // first dimension (e.g., hidden_size)
}

impl ActivationHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12);
        buf.extend_from_slice(ACTIVATION_MAGIC);
        buf.extend_from_slice(&self.dtype.to_le_bytes());
        buf.extend_from_slice(&self.ndims.to_le_bytes());
        buf.extend_from_slice(&self.dim0.to_le_bytes());
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            anyhow::bail!("Activation header too short: {} bytes", data.len());
        }
        if &data[0..4] != ACTIVATION_MAGIC {
            anyhow::bail!("Invalid activation magic bytes");
        }
        Ok(Self {
            dtype: u16::from_le_bytes([data[4], data[5]]),
            ndims: u16::from_le_bytes([data[6], data[7]]),
            dim0: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
        })
    }
}

/// Execute a pipeline job — this Island runs its shard of the model.
///
/// For the current implementation, because llama_cpp does not expose a
/// per-layer forward pass API, each Island in the pipeline loads the full
/// shard GGUF and runs complete inference on its portion. The first Island
/// receives the prompt and generates tokens; intermediate and last Islands
/// in the pipeline also run inference on their shard. Activations between
/// Islands carry the generated token stream (not raw tensors) until a
/// native layer-split API is available.
///
/// Flow:
/// 1. Download model shard via cache system
/// 2. Signal "ready" to coordinator
/// 3. Subscribe to activation and control subjects
/// 4. Wait for "start" on control
/// 5. Position 0: load model, run inference, stream tokens to output/next
/// 6. Other positions: receive tokens, forward to next or output
pub async fn execute_pipeline_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    pipeline_config: PipelineConfig,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &pipeline_config.group_id;
    let position = pipeline_config.position;
    let is_first = position == 0;
    let is_last = pipeline_config.ring_subjects.next_activate.is_none();

    info!(
        job_id,
        group_id,
        position,
        is_first,
        is_last,
        "Starting pipeline execution (layers {}-{})",
        pipeline_config.shard_spec.layer_start,
        pipeline_config.shard_spec.layer_end,
    );

    // Determine model URL: use shard-specific URL if available, else fall back to job model_url
    let model_url = pipeline_config
        .shard_spec
        .shard_url
        .as_deref()
        .or(job.model_url.as_deref())
        .context("Pipeline job missing both shard_url and model_url")?;

    let model_hash = job.model_hash.as_deref();

    // Download model shard via cache system
    let model_cache = {
        let st = state.read().await;
        st.model_cache()
            .context("Model cache not initialized")?
            .clone()
    };

    info!(job_id, position, "Downloading model shard: {}", model_url);
    let model_path = model_cache
        .download_model(model_url, model_hash)
        .await
        .with_context(|| format!("Failed to download shard: {}", model_url))?;

    let model_path_str = model_path.to_string_lossy().to_string();
    info!(job_id, position, "Model shard cached at: {}", model_path_str);

    // Signal ready to coordinator
    nats.publish_ring_status(
        group_id,
        &serde_json::json!({
            "host_id": nats.host_id(),
            "status": "ready",
            "position": position,
        }),
    )
    .await?;

    info!(job_id, "Pipeline member signaled ready, waiting for start");

    // Subscribe to activation and control subjects
    let mut activate_sub = nats
        .subscribe_ring(&pipeline_config.ring_subjects.activate)
        .await
        .context("Failed to subscribe to activation subject")?;

    let mut control_sub = nats
        .subscribe_ring(&pipeline_config.ring_subjects.control)
        .await
        .context("Failed to subscribe to control subject")?;

    // Wait for start signal
    let started = loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("start") {
                                break true;
                            }
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                info!(job_id, "Received stop before start, aborting");
                                break false;
                            }
                        }
                    }
                    None => {
                        error!(job_id, "Control subscription closed");
                        break false;
                    }
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(job_id, "Pipeline cancelled before start");
                    break false;
                }
            }
        }
    };

    if !started {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    info!(job_id, position, "Pipeline started");

    if is_first {
        // Position 0: run inference on the shard and stream tokens
        execute_first_position(
            nats,
            job,
            &pipeline_config,
            &model_path_str,
            &mut control_sub,
            cancel_rx,
        )
        .await
    } else {
        // Other positions: receive activations, forward or output
        execute_relay_position(
            nats,
            job,
            &pipeline_config,
            &model_path_str,
            &mut activate_sub,
            &mut control_sub,
            cancel_rx,
        )
        .await
    }
}

/// Position 0: load model, run inference, stream tokens to next position or output.
async fn execute_first_position(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &PipelineConfig,
    model_path: &str,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;
    let is_last = config.ring_subjects.next_activate.is_none();

    let context_size = job.model_context_size.unwrap_or(2048);
    let temperature = job.model_temperature.unwrap_or(0.7);
    let max_tokens = job
        .input
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;

    let prompt = crate::gguf::extract_prompt(&job.input)?;

    // Load model and generate in spawn_blocking, stream tokens via channel
    let model_path_owned = model_path.to_string();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
    let cancel = cancel_rx.clone();

    let generate_handle = tokio::task::spawn_blocking(move || {
        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&model_path_owned, params)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF shard: {:?}", e))?;

        let mut session_params = SessionParams::default();
        session_params.n_ctx = context_size;

        let mut session = model
            .create_session(session_params)
            .map_err(|e| anyhow::anyhow!("Failed to create session: {:?}", e))?;

        session
            .advance_context(&prompt)
            .map_err(|e| anyhow::anyhow!("Failed to feed prompt: {:?}", e))?;

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
            1,
        );

        let completions = session
            .start_completing_with(sampler, max_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to start completion: {:?}", e))?;

        let mut token_count: u64 = 0;
        for token_str in completions.into_strings() {
            let token_str = token_str.to_string();
            if *cancel.borrow() {
                break;
            }
            token_count += 1;
            if tx.blocking_send(token_str).is_err() {
                break;
            }
        }

        Ok::<u64, anyhow::Error>(token_count)
    });

    // Microbatch size: how many tokens to collect before sending an activation message.
    // Higher values reduce NATS message overhead but add per-batch latency.
    let microbatch_size = job
        .input
        .get("microbatch_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MICROBATCH_SIZE as u64) as usize;

    // Stream tokens — either to next position (as activation) or directly to ring output
    let mut seq: u64 = 0;
    let mut cancelled = false;
    let mut token_batch = Vec::with_capacity(microbatch_size.max(1));

    loop {
        select! {
            token = rx.recv() => {
                match token {
                    Some(token_str) => {
                        token_batch.push(token_str);

                        // Flush when batch is full
                        if token_batch.len() >= microbatch_size {
                            seq += 1;
                            let batch_text: String = token_batch.drain(..).collect();

                            if is_last {
                                let output = serde_json::json!({
                                    "job_id": job_id,
                                    "chunk": batch_text,
                                    "is_final": false,
                                    "seq": seq,
                                });
                                nats.publish_ring_output(group_id, &output).await?;
                            } else if let Some(ref next_subject) = config.ring_subjects.next_activate {
                                let activation = serde_json::json!({
                                    "job_id": job_id,
                                    "token": batch_text,
                                    "seq": seq,
                                });
                                let payload = serde_json::to_vec(&activation)?;
                                nats.publish_raw(next_subject, payload).await?;
                            }
                        }
                    }
                    None => {
                        // Channel closed — flush remaining tokens
                        if !token_batch.is_empty() {
                            seq += 1;
                            let batch_text: String = token_batch.drain(..).collect();
                            if is_last {
                                let output = serde_json::json!({
                                    "job_id": job_id,
                                    "chunk": batch_text,
                                    "is_final": false,
                                    "seq": seq,
                                });
                                nats.publish_ring_output(group_id, &output).await?;
                            } else if let Some(ref next_subject) = config.ring_subjects.next_activate {
                                let activation = serde_json::json!({
                                    "job_id": job_id,
                                    "token": batch_text,
                                    "seq": seq,
                                });
                                let payload = serde_json::to_vec(&activation)?;
                                nats.publish_raw(next_subject, payload).await?;
                            }
                        }
                        break;
                    }
                }
            }
            msg = control_sub.next() => {
                if let Some(msg) = msg {
                    if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                        if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                            info!(job_id, "Received stop during generation");
                            cancelled = true;
                            break;
                        }
                    }
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    cancelled = true;
                    break;
                }
            }
        }
    }

    if cancelled {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // Wait for generation thread
    let result = generate_handle.await?;
    let token_count = result.unwrap_or(0);

    // Send final output marker
    if is_last {
        let final_output = serde_json::json!({
            "job_id": job_id,
            "chunk": "",
            "is_final": true,
            "seq": seq + 1,
            "usage": { "completion_tokens": token_count },
        });
        nats.publish_ring_output(group_id, &final_output).await?;
    } else if let Some(ref next_subject) = config.ring_subjects.next_activate {
        let final_activation = serde_json::json!({
            "job_id": job_id,
            "token": "",
            "seq": seq + 1,
            "is_final": true,
        });
        let payload = serde_json::to_vec(&final_activation)?;
        nats.publish_raw(next_subject, payload).await?;
    }

    // Signal completion
    nats.publish_ring_status(
        group_id,
        &serde_json::json!({"status": "complete"}),
    )
    .await?;

    info!(job_id, "Pipeline position 0 completed: {} tokens", token_count);
    Ok(())
}

/// Non-first positions: receive activations (tokens) and forward to next or output.
///
/// Until llama_cpp exposes per-layer APIs, relay positions pass tokens through.
/// In future, each position will run its layer range on the received activations.
async fn execute_relay_position(
    nats: &NatsAgent,
    job: &AssignJob,
    config: &PipelineConfig,
    _model_path: &str,
    activate_sub: &mut async_nats::Subscriber,
    control_sub: &mut async_nats::Subscriber,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;
    let group_id = &config.group_id;
    let position = config.position;
    let is_last = config.ring_subjects.next_activate.is_none();

    info!(job_id, position, "Relay position entering main loop");

    loop {
        select! {
            msg = activate_sub.next() => {
                match msg {
                    Some(msg) => {
                        // Parse the activation (currently a JSON token message)
                        let activation: serde_json::Value = serde_json::from_slice(&msg.payload)
                            .unwrap_or_else(|_| serde_json::json!({"token": "", "is_final": true}));

                        let token = activation.get("token").and_then(|t| t.as_str()).unwrap_or("");
                        let is_final = activation.get("is_final").and_then(|f| f.as_bool()).unwrap_or(false);
                        let seq = activation.get("seq").and_then(|s| s.as_u64()).unwrap_or(0);

                        // TODO: When per-layer API is available, run forward pass here:
                        // let output_activation = model.forward_layers(input_activation, layer_start, layer_end);

                        if is_last {
                            // Publish to ring output for the coordinator
                            let output = serde_json::json!({
                                "job_id": job_id,
                                "chunk": token,
                                "is_final": is_final,
                                "seq": seq,
                            });
                            nats.publish_ring_output(group_id, &output).await?;

                            if is_final {
                                nats.publish_ring_status(
                                    group_id,
                                    &serde_json::json!({"status": "complete"}),
                                ).await?;
                                info!(job_id, position, "Relay position published final output");
                                break;
                            }
                        } else if let Some(ref next_subject) = config.ring_subjects.next_activate {
                            // Forward to next position
                            nats.publish_raw(next_subject, msg.payload.to_vec()).await?;

                            if is_final {
                                info!(job_id, position, "Relay position forwarded final activation");
                                break;
                            }
                        }
                    }
                    None => {
                        error!(job_id, "Activation subscription closed");
                        nats.publish_ring_status(
                            group_id,
                            &serde_json::json!({
                                "host_id": nats.host_id(),
                                "status": "failed",
                                "error": "activation subscription closed",
                            }),
                        ).await?;
                        break;
                    }
                }
            }
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            if ctrl.get("action").and_then(|a| a.as_str()) == Some("stop") {
                                info!(job_id, "Received stop signal, aborting relay");
                                break;
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(job_id, "Pipeline relay cancelled");
                    nats.publish_status(job_id, "cancelled", None).await?;
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
    fn test_activation_header_roundtrip() {
        let header = ActivationHeader {
            dtype: 0,  // f16
            ndims: 1,
            dim0: 4096,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..4], ACTIVATION_MAGIC);

        let parsed = ActivationHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.dtype, 0);
        assert_eq!(parsed.ndims, 1);
        assert_eq!(parsed.dim0, 4096);
    }

    #[test]
    fn test_activation_header_too_short() {
        let result = ActivationHeader::from_bytes(&[0u8; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_activation_header_bad_magic() {
        let mut data = [0u8; 12];
        data[0..4].copy_from_slice(b"XXXX");
        let result = ActivationHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_config_deserialize() {
        let json = r#"{
            "group_id": "abc-123",
            "position": 0,
            "shard_spec": {
                "layer_start": 0,
                "layer_end": 15,
                "shard_url": "https://example.com/shard-0.gguf"
            },
            "total_members": 2,
            "ring_subjects": {
                "activate": "ring.abc-123.activate.0",
                "control": "ring.abc-123.control",
                "output": "ring.abc-123.output",
                "status": "ring.abc-123.status",
                "next_activate": "ring.abc-123.activate.1"
            }
        }"#;

        let config: PipelineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.group_id, "abc-123");
        assert_eq!(config.position, 0);
        assert_eq!(config.shard_spec.layer_start, 0);
        assert_eq!(config.shard_spec.layer_end, 15);
        assert!(config.ring_subjects.next_activate.is_some());
    }

    #[test]
    fn test_pipeline_config_last_position() {
        let json = r#"{
            "group_id": "abc-123",
            "position": 1,
            "shard_spec": {"layer_start": 16, "layer_end": 31},
            "total_members": 2,
            "ring_subjects": {
                "activate": "ring.abc-123.activate.1",
                "control": "ring.abc-123.control",
                "output": "ring.abc-123.output",
                "status": "ring.abc-123.status",
                "next_activate": null
            }
        }"#;

        let config: PipelineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.position, 1);
        assert!(config.ring_subjects.next_activate.is_none());
    }
}
