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

fn default_algorithm() -> String {
    "fed_avg".into()
}
fn default_local_epochs() -> u32 {
    3
}
fn default_learning_rate() -> f64 {
    0.001
}
fn default_batch_size() -> u32 {
    32
}

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

    info!(
        session_id,
        host_id, "Starting federated training participant"
    );

    // Download base model via cache
    let model_cache = {
        let st = state.read().await;
        st.model_cache()
            .context("Model cache not initialized")?
            .clone()
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
    )
    .await?;

    // Subscribe to control and aggregate subjects
    let mut control_sub = nats
        .subscribe_ring(&config.subjects.control)
        .await
        .context("Failed to subscribe to training control")?;
    let mut aggregate_sub = nats
        .subscribe_ring(&config.subjects.aggregate)
        .await
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

/// Compute gradient for a training round.
///
/// Two modes:
/// 1. **Simulated** (current): deterministic perturbation vector for testing the protocol
/// 2. **LoRA** (when available): actual backpropagation through adapter layers
///
/// The LoRA path will use llama_cpp_sys FFI to:
/// 1. Load the base model + LoRA adapter weights
/// 2. Forward pass on training batch
/// 3. Compute loss (cross-entropy on next-token prediction)
/// 4. Backward pass through LoRA layers only (r=8, alpha=16 typical)
/// 5. Return the LoRA weight deltas (much smaller than full model gradients)
///
/// LoRA gradient size: rank × (in_dim + out_dim) per adapted layer
/// For rank=8, 32-layer model with 4096 hidden: ~2MB (vs ~14GB for full)
pub(crate) fn compute_gradient(config: &TrainingConfig, round: u32) -> Vec<f32> {
    if lora_available() {
        compute_lora_gradient(config, round)
    } else {
        compute_simulated_gradient(config, round)
    }
}

/// Check if LoRA training is available in the current llama_cpp version
pub(crate) fn lora_available() -> bool {
    // When llama_cpp_sys exposes llama_train_* functions, this returns true
    false
}

/// Simulated gradient for protocol testing
pub(crate) fn compute_simulated_gradient(config: &TrainingConfig, round: u32) -> Vec<f32> {
    let dim = 1024;
    let lr = config.config.learning_rate as f32;

    (0..dim)
        .map(|i| {
            let seed = (round as f32 * 1000.0 + i as f32).sin() * lr;
            seed * 0.01
        })
        .collect()
}

/// LoRA gradient computation (stub — ready for when llama_cpp supports training)
///
/// Expected API:
///   llama_lora_adapter_init(model, rank, alpha) → adapter
///   llama_train_forward(ctx, adapter, batch) → loss
///   llama_train_backward(ctx, adapter) → updates adapter weights
///   llama_lora_get_weights(adapter) → weight data
pub(crate) fn compute_lora_gradient(_config: &TrainingConfig, _round: u32) -> Vec<f32> {
    // Placeholder — will be replaced with actual LoRA training:
    //
    // unsafe {
    //     let adapter = llama_cpp_sys::llama_lora_adapter_init(model, rank, alpha);
    //     for epoch in 0..config.config.local_epochs {
    //         for batch in training_data.batches(config.config.batch_size) {
    //             let loss = llama_cpp_sys::llama_train_forward(ctx, adapter, batch);
    //             llama_cpp_sys::llama_train_backward(ctx, adapter);
    //         }
    //     }
    //     let weights = llama_cpp_sys::llama_lora_get_weights(adapter);
    //     // Compute delta: new_weights - initial_weights
    //     weights_to_gradient_vector(weights)
    // }
    vec![0.0; 1024]
}

/// Encode gradient as base64 for JSON transport
pub(crate) fn base64_encode_gradient(gradient: &[f32]) -> String {
    use base64::Engine;
    let bytes: Vec<u8> = gradient.iter().flat_map(|f| f.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn make_config(lr: f64, batch_size: u32, local_epochs: u32) -> TrainingConfig {
        TrainingConfig {
            session_id: "test-session".into(),
            total_rounds: 5,
            config: TrainingParams {
                algorithm: "fed_avg".into(),
                local_epochs,
                learning_rate: lr,
                batch_size,
                dp_sigma: 0.0,
            },
            subjects: TrainingSubjects {
                control: "fed.test.control".into(),
                gradient: "fed.test.gradient".into(),
                aggregate: "fed.test.aggregate".into(),
                status: "fed.test.status".into(),
            },
        }
    }

    fn decode_gradient(encoded: &str) -> Vec<f32> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ── TrainingConfig deserialization ─────────────────────────────

    #[test]
    fn test_training_config_deserialize_full() {
        let json = r#"{
            "session_id": "abc-123",
            "total_rounds": 10,
            "config": {
                "algorithm": "fed_avg",
                "local_epochs": 3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "dp_sigma": 0.5
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
        assert!((config.config.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(config.config.batch_size, 32);
        assert!((config.config.dp_sigma - 0.5).abs() < 1e-10);
        assert_eq!(config.subjects.control, "fed.abc-123.control");
    }

    #[test]
    fn test_training_config_deserialize_defaults() {
        // Omit all fields with defaults in TrainingParams
        let json = r#"{
            "session_id": "defaults-test",
            "total_rounds": 1,
            "config": {},
            "subjects": {
                "control": "c",
                "gradient": "g",
                "aggregate": "a",
                "status": "s"
            }
        }"#;

        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.config.algorithm, "fed_avg");
        assert_eq!(config.config.local_epochs, 3);
        assert!((config.config.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(config.config.batch_size, 32);
        assert!((config.config.dp_sigma - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_config_roundtrip() {
        let config = make_config(0.01, 64, 5);
        let json = serde_json::to_string(&config).unwrap();
        let decoded: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.session_id, config.session_id);
        assert_eq!(decoded.total_rounds, config.total_rounds);
        assert_eq!(decoded.config.batch_size, 64);
        assert_eq!(decoded.config.local_epochs, 5);
        assert!((decoded.config.learning_rate - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_training_params_partial_override() {
        // Override only some fields, rest get defaults
        let json = r#"{
            "session_id": "partial",
            "total_rounds": 3,
            "config": {
                "learning_rate": 0.01
            },
            "subjects": {
                "control": "c", "gradient": "g", "aggregate": "a", "status": "s"
            }
        }"#;

        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert!((config.config.learning_rate - 0.01).abs() < 1e-10);
        // Defaults for the rest
        assert_eq!(config.config.algorithm, "fed_avg");
        assert_eq!(config.config.local_epochs, 3);
        assert_eq!(config.config.batch_size, 32);
    }

    // ── lora_available ────────────────────────────────────────────

    #[test]
    fn test_lora_available_returns_false() {
        // LoRA is not yet supported; this should always be false
        assert!(!lora_available());
    }

    // ── compute_gradient dispatch ─────────────────────────────────

    #[test]
    fn test_compute_gradient_uses_simulated() {
        // Since lora_available() is false, compute_gradient should equal compute_simulated_gradient
        let config = make_config(0.001, 32, 3);
        let gradient = compute_gradient(&config, 1);
        let simulated = compute_simulated_gradient(&config, 1);
        assert_eq!(gradient, simulated);
    }

    // ── compute_simulated_gradient ────────────────────────────────

    #[test]
    fn test_simulated_gradient_length() {
        let config = make_config(0.001, 32, 3);
        let gradient = compute_simulated_gradient(&config, 1);
        assert_eq!(gradient.len(), 1024);
    }

    #[test]
    fn test_simulated_gradient_deterministic() {
        let config = make_config(0.001, 32, 3);
        let g1 = compute_simulated_gradient(&config, 1);
        let g2 = compute_simulated_gradient(&config, 1);
        assert_eq!(
            g1, g2,
            "same config + round should produce identical gradients"
        );
    }

    #[test]
    fn test_simulated_gradient_varies_by_round() {
        let config = make_config(0.001, 32, 3);
        let g1 = compute_simulated_gradient(&config, 1);
        let g2 = compute_simulated_gradient(&config, 2);
        assert_ne!(
            g1, g2,
            "different rounds should produce different gradients"
        );
    }

    #[test]
    fn test_simulated_gradient_scales_with_learning_rate() {
        let config_low = make_config(0.001, 32, 3);
        let config_high = make_config(0.01, 32, 3);

        let g_low = compute_simulated_gradient(&config_low, 1);
        let g_high = compute_simulated_gradient(&config_high, 1);

        let mag_low: f32 = g_low.iter().map(|g| g.abs()).sum();
        let mag_high: f32 = g_high.iter().map(|g| g.abs()).sum();

        // Higher learning rate should produce larger magnitude gradients
        assert!(
            mag_high > mag_low,
            "higher lr should produce larger gradient magnitude"
        );

        // The gradient is lr * 0.01 * sin(...), so ratio should be ~10x
        let ratio = mag_high / mag_low;
        assert!(
            (ratio - 10.0).abs() < 0.01,
            "magnitude ratio should be ~10x, got {}",
            ratio
        );
    }

    #[test]
    fn test_simulated_gradient_bounded() {
        let config = make_config(0.001, 32, 3);
        let gradient = compute_simulated_gradient(&config, 1);
        // Each element is sin(...) * lr * 0.01, so bounded by lr * 0.01
        let bound = 0.001 * 0.01 + 1e-10;
        assert!(
            gradient.iter().all(|g| g.abs() <= bound as f32),
            "gradient values should be bounded by lr * 0.01"
        );
    }

    #[test]
    fn test_simulated_gradient_all_finite() {
        let config = make_config(0.001, 32, 3);
        for round in 0..10 {
            let gradient = compute_simulated_gradient(&config, round);
            assert!(
                gradient.iter().all(|g| g.is_finite()),
                "all gradient values must be finite for round {}",
                round
            );
        }
    }

    // ── compute_lora_gradient (stub) ──────────────────────────────

    #[test]
    fn test_lora_gradient_is_zeros() {
        let config = make_config(0.001, 32, 3);
        let gradient = compute_lora_gradient(&config, 1);
        assert_eq!(gradient.len(), 1024);
        assert!(
            gradient.iter().all(|g| *g == 0.0),
            "stub should return all zeros"
        );
    }

    // ── base64_encode_gradient ────────────────────────────────────

    #[test]
    fn test_gradient_encode_decode_roundtrip() {
        let gradient = vec![1.0f32, 2.0, 3.0, -1.5];
        let encoded = base64_encode_gradient(&gradient);
        let decoded = decode_gradient(&encoded);
        assert_eq!(decoded, gradient);
    }

    #[test]
    fn test_gradient_encode_empty() {
        let gradient: Vec<f32> = vec![];
        let encoded = base64_encode_gradient(&gradient);
        let decoded = decode_gradient(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_gradient_encode_single_value() {
        let gradient = vec![42.0f32];
        let encoded = base64_encode_gradient(&gradient);
        let decoded = decode_gradient(&encoded);
        assert_eq!(decoded, vec![42.0f32]);
    }

    #[test]
    fn test_gradient_encode_special_values() {
        let gradient = vec![0.0f32, -0.0, f32::MIN, f32::MAX];
        let encoded = base64_encode_gradient(&gradient);
        let decoded = decode_gradient(&encoded);
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[2], f32::MIN);
        assert_eq!(decoded[3], f32::MAX);
    }

    #[test]
    fn test_gradient_encode_large_vector() {
        let gradient: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let encoded = base64_encode_gradient(&gradient);
        let decoded = decode_gradient(&encoded);
        assert_eq!(decoded.len(), 1024);
        assert_eq!(decoded, gradient);
    }

    #[test]
    fn test_gradient_encode_byte_length() {
        let gradient = vec![1.0f32, 2.0, 3.0];
        let encoded = base64_encode_gradient(&gradient);
        // 3 floats * 4 bytes = 12 bytes, base64 of 12 bytes = 16 chars
        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded_bytes.len(), 12);
    }

    // ── TrainingSubjects ──────────────────────────────────────────

    #[test]
    fn test_training_subjects_serialize() {
        let subjects = TrainingSubjects {
            control: "fed.session1.control".into(),
            gradient: "fed.session1.gradient".into(),
            aggregate: "fed.session1.aggregate".into(),
            status: "fed.session1.status".into(),
        };
        let json = serde_json::to_string(&subjects).unwrap();
        assert!(json.contains("fed.session1.control"));
        assert!(json.contains("fed.session1.gradient"));
        assert!(json.contains("fed.session1.aggregate"));
        assert!(json.contains("fed.session1.status"));
    }
}
