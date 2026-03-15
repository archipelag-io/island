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
use serde::{Deserialize, Serialize};
use tokio::select;
use tokio::sync::watch;
use tracing::{error, info};

use crate::nats::{AssignJob, NatsAgent};

/// Magic bytes for activation header
#[allow(dead_code)]
const ACTIVATION_MAGIC: &[u8; 4] = b"ARCP";

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
#[allow(dead_code)]
pub struct ActivationHeader {
    pub dtype: u16,    // 0 = f16, 1 = f32, 2 = bf16
    pub ndims: u16,    // number of dimensions
    pub dim0: u32,     // first dimension (e.g., hidden_size)
}

#[allow(dead_code)]
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
/// Flow:
/// 1. Parse pipeline_config from the job assignment
/// 2. Signal "ready" to coordinator
/// 3. Subscribe to activation and control subjects
/// 4. Wait for "start" on control
/// 5. Main loop: receive activation → forward pass → publish to next position
/// 6. Position 0: receives prompt, runs embedding + first layers
/// 7. Last position: runs final layers, publishes tokens to output
pub async fn execute_pipeline_job(
    nats: &NatsAgent,
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

    // TODO: Download model shard via cache system
    // For now, signal ready immediately
    // In production: download shard_spec.shard_url, load into llama.cpp

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

    info!(job_id, position, "Pipeline started, entering main loop");

    // Main activation loop
    loop {
        select! {
            msg = activate_sub.next() => {
                match msg {
                    Some(msg) => {
                        let payload = &msg.payload;

                        // TODO: Real forward pass through local layers
                        // For now, pass through activation data unchanged
                        // In production:
                        // 1. Parse activation header + data
                        // 2. Run through layers layer_start..layer_end
                        // 3. Publish result to next position or output

                        if is_last {
                            // Last position: publish to output subject
                            // TODO: Run final layers + decode tokens
                            let output = serde_json::json!({
                                "job_id": job_id,
                                "chunk": "(pipeline output placeholder)",
                                "is_final": true,
                            });
                            nats.publish_ring_output(group_id, &output).await?;

                            // Signal completion
                            nats.publish_ring_status(
                                group_id,
                                &serde_json::json!({"status": "complete"}),
                            ).await?;

                            info!(job_id, "Pipeline position {position} published final output");
                            break;
                        } else if let Some(ref next_subject) = pipeline_config.ring_subjects.next_activate {
                            // Forward activation to next position
                            nats.publish_raw(next_subject, payload.to_vec()).await?;
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
                                info!(job_id, "Received stop signal, aborting pipeline");
                                break;
                            }
                        }
                    }
                    None => break,
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(job_id, "Pipeline cancelled during execution");
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
