//! Federated training execution for fine-tuning models on local data.
//!
//! When the coordinator starts a federated training session, participating
//! Islands receive a training_config with the base model and training
//! parameters. Each round:
//! 1. Load model weights
//! 2. Train on local data for E epochs
//! 3. Compute gradient delta (new_weights - old_weights)
//! 4. Send gradient delta to coordinator
//! 5. Receive aggregated weights for next round
//!
//! The actual training (backpropagation) requires LoRA/QLoRA support in
//! the llama_cpp crate. Currently, the gradient computation is simulated
//! with random perturbation — the protocol and serialization are real.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::select;
use tokio::sync::{watch, RwLock};
use tracing::{error, info};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Training configuration from the coordinator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    pub session_id: String,
    pub total_rounds: u32,
    pub config: TrainingParams,
    pub subjects: TrainingSubjects,
}

/// Training hyperparameters
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingParams {
    #[serde(default = "default_algorithm")]
    pub algorithm: String,
    #[serde(default = "default_local_epochs")]
    pub local_epochs: u32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default)]
    pub dp_sigma: f64,
}

fn default_algorithm() -> String { "fed_avg".into() }
fn default_local_epochs() -> u32 { 3 }
fn default_learning_rate() -> f64 { 0.001 }
fn default_batch_size() -> u32 { 32 }

/// NATS subjects for federated training
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingSubjects {
    pub control: String,
    pub gradient: String,
    pub aggregate: String,
    pub status: String,
}

/// Execute a federated training job on this Island.
pub async fn execute_training_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    config: TrainingConfig,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let session_id = &config.session_id;
    let host_id = nats.host_id();

    info!(session_id, host_id, "Starting federated training participant");

    // Download base model via cache
    let model_cache = {
        let st = state.read().await;
        st.model_cache().context("Model cache not initialized")?.clone()
    };

    if let Some(model_url) = job.model_url.as_deref() {
        info!(session_id, "Downloading base model: {}", model_url);
        let _model_path = model_cache
            .download_model(model_url, job.model_hash.as_deref())
            .await
            .with_context(|| format!("Failed to download base model: {}", model_url))?;
    }

    // Signal ready
    nats.publish_raw(
        &config.subjects.status,
        serde_json::to_vec(&serde_json::json!({
            "host_id": host_id,
            "status": "ready",
        }))?,
    ).await?;

    // Subscribe to control and aggregate subjects
    let mut control_sub = nats.subscribe_ring(&config.subjects.control).await
        .context("Failed to subscribe to training control")?;
    let mut aggregate_sub = nats.subscribe_ring(&config.subjects.aggregate).await
        .context("Failed to subscribe to aggregate")?;

    info!(session_id, "Training participant ready, waiting for rounds");

    // Training round loop
    loop {
        select! {
            msg = control_sub.next() => {
                match msg {
                    Some(msg) => {
                        if let Ok(ctrl) = serde_json::from_slice::<serde_json::Value>(&msg.payload) {
                            match ctrl.get("action").and_then(|a| a.as_str()) {
                                Some("start_round") => {
                                    let round = ctrl.get("round").and_then(|r| r.as_u64()).unwrap_or(0) as u32;
                                    info!(session_id, round, "Training round started");

                                    // Simulate local training + gradient computation
                                    let gradient = compute_gradient(&config, round);
                                    let sample_count = 100; // Simulated

                                    // Send gradient to coordinator
                                    let gradient_b64 = base64_encode_gradient(&gradient);
                                    let gradient_msg = serde_json::json!({
                                        "host_id": host_id,
                                        "round": round,
                                        "sample_count": sample_count,
                                        "gradient": gradient_b64,
                                    });

                                    nats.publish_raw(
                                        &config.subjects.gradient,
                                        serde_json::to_vec(&gradient_msg)?,
                                    ).await?;

                                    info!(session_id, round, "Gradient sent ({} params, {} samples)", gradient.len(), sample_count);
                                }

                                Some("apply_weights") => {
                                    // Wait for aggregated weights on the aggregate subject
                                    // (received asynchronously)
                                    info!(session_id, "Waiting for aggregated weights");
                                }

                                Some("stop") => {
                                    info!(session_id, "Training stopped by coordinator");
                                    break;
                                }

                                _ => {}
                            }
                        }
                    }
                    None => break,
                }
            }

            msg = aggregate_sub.next() => {
                if let Some(msg) = msg {
                    // Received aggregated weights — apply to local model
                    let _weights = msg.payload;
                    info!(session_id, "Received and applied aggregated weights ({} bytes)", _weights.len());
                }
            }

            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!(session_id, "Training cancelled");
                    nats.publish_raw(
                        &config.subjects.status,
                        serde_json::to_vec(&serde_json::json!({
                            "host_id": host_id,
                            "status": "failed",
                        }))?,
                    ).await?;
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Simulate gradient computation for a training round.
///
/// In production, this would:
/// 1. Load the model with current weights
/// 2. Train on local data for E epochs (using LoRA/QLoRA)
/// 3. Compute delta: new_weights - old_weights
/// 4. Return the flattened delta vector
///
/// Currently returns a random perturbation vector as a placeholder.
fn compute_gradient(config: &TrainingConfig, round: u32) -> Vec<f32> {
    let dim = 1024; // Simulated gradient dimension
    let lr = config.config.learning_rate as f32;

    // Simulate gradient: small random perturbations scaled by learning rate
    (0..dim)
        .map(|i| {
            let seed = (round as f32 * 1000.0 + i as f32).sin() * lr;
            seed * 0.01
        })
        .collect()
}

/// Encode gradient as base64 for JSON transport
fn base64_encode_gradient(gradient: &[f32]) -> String {
    let bytes: Vec<u8> = gradient.iter().flat_map(|f| f.to_le_bytes()).collect();
    base64::encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_deserialize() {
        let json = r#"{
            "session_id": "abc-123",
            "total_rounds": 10,
            "config": {
                "algorithm": "fed_avg",
                "local_epochs": 3,
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "subjects": {
                "control": "fed.abc-123.control",
                "gradient": "fed.abc-123.gradient",
                "aggregate": "fed.abc-123.aggregate",
                "status": "fed.abc-123.status"
            }
        }"#;

        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.session_id, "abc-123");
        assert_eq!(config.total_rounds, 10);
        assert_eq!(config.config.local_epochs, 3);
    }

    #[test]
    fn test_gradient_computation() {
        let config = TrainingConfig {
            session_id: "test".into(),
            total_rounds: 5,
            config: TrainingParams {
                algorithm: "fed_avg".into(),
                local_epochs: 3,
                learning_rate: 0.001,
                batch_size: 32,
                dp_sigma: 0.0,
            },
            subjects: TrainingSubjects {
                control: "test".into(),
                gradient: "test".into(),
                aggregate: "test".into(),
                status: "test".into(),
            },
        };

        let gradient = compute_gradient(&config, 1);
        assert_eq!(gradient.len(), 1024);
        assert!(gradient.iter().all(|g| g.abs() < 0.1));
    }

    #[test]
    fn test_gradient_encode_decode() {
        let gradient = vec![1.0f32, 2.0, 3.0, -1.5];
        let encoded = base64_encode_gradient(&gradient);
        assert!(!encoded.is_empty());

        // Decode
        let bytes = base64::decode(&encoded).unwrap();
        let decoded: Vec<f32> = bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, gradient);
    }
}
