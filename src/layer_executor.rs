//! Per-layer forward pass abstraction for pipeline parallelism.
//!
//! Defines a `LayerExecutor` trait that encapsulates running a subset of
//! model layers on input data (either a prompt string or an activation tensor).
//!
//! Two implementations:
//! - `FullModelExecutor`: Loads the full shard GGUF and runs complete inference.
//!   This is the current default — position 0 generates tokens, relay positions
//!   pass through. Works today with llama_cpp 0.3.2.
//!
//! - `LayerRangeExecutor` (stubbed): When llama_cpp exposes `forward_layers()`,
//!   this will run only layers `start..end` on an activation tensor and produce
//!   the next activation. This is the true pipeline parallelism path.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// The output of a layer executor — either tokens (from full inference)
/// or raw activation data (from per-layer forward).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerOutput {
    /// Token string output (from full model inference at position 0)
    Token { text: String, seq: u64 },
    /// Raw activation tensor (from per-layer forward — future)
    Activation { data: Vec<u8>, seq: u64, dtype: u16, dim: u32 },
    /// Final output marker
    Done { total_tokens: u64 },
}

/// Trait for executing a subset of model layers.
///
/// Implementations handle loading the model/shard and running forward passes.
/// The pipeline orchestrator calls `execute` and receives a stream of `LayerOutput`.
pub trait LayerExecutor: Send + Sync {
    /// Execute the model layers on the given input.
    ///
    /// For position 0: `input` is a prompt string, output is Token variants.
    /// For relay positions (future): `input` is activation bytes, output is Activation variants.
    fn execute(
        &self,
        input: &[u8],
        cancel: &std::sync::atomic::AtomicBool,
        output_tx: std::sync::mpsc::Sender<LayerOutput>,
    ) -> Result<()>;

    /// Layer range this executor covers
    fn layer_range(&self) -> (u32, u32);

    /// Whether this is a full model or a layer range
    fn is_full_model(&self) -> bool;
}

/// Full model executor — loads entire GGUF shard and runs complete inference.
/// This is the current implementation for position 0 in the pipeline.
pub struct FullModelExecutor {
    pub model_path: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub context_size: u32,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl LayerExecutor for FullModelExecutor {
    fn execute(
        &self,
        input: &[u8],
        cancel: &std::sync::atomic::AtomicBool,
        output_tx: std::sync::mpsc::Sender<LayerOutput>,
    ) -> Result<()> {
        use llama_cpp::standard_sampler::{SamplerStage, StandardSampler};
        use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

        let prompt = std::str::from_utf8(input)
            .map_err(|e| anyhow::anyhow!("Invalid prompt encoding: {}", e))?;

        let params = LlamaParams::default();
        let model = LlamaModel::load_from_file(&self.model_path, params)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

        let mut session_params = SessionParams::default();
        session_params.n_ctx = self.context_size;

        let mut session = model
            .create_session(session_params)
            .map_err(|e| anyhow::anyhow!("Failed to create session: {:?}", e))?;

        session
            .advance_context(prompt)
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
                SamplerStage::Temperature(self.temperature),
            ],
            1,
        );

        let completions = session
            .start_completing_with(sampler, self.max_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to start completion: {:?}", e))?;

        let mut seq: u64 = 0;
        for token_str in completions.into_strings() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            seq += 1;
            let output = LayerOutput::Token {
                text: token_str.to_string(),
                seq,
            };
            if output_tx.send(output).is_err() {
                break;
            }
        }

        let _ = output_tx.send(LayerOutput::Done { total_tokens: seq });
        Ok(())
    }

    fn layer_range(&self) -> (u32, u32) {
        (self.layer_start, self.layer_end)
    }

    fn is_full_model(&self) -> bool {
        true
    }
}

/// Stubbed layer-range executor for future per-layer forward pass.
///
/// When llama_cpp exposes `model.forward_layers(input_tensor, start, end)`,
/// this executor will:
/// 1. Load the shard GGUF containing only layers start..end
/// 2. Accept raw activation bytes as input
/// 3. Run forward pass through the local layers
/// 4. Output the resulting activation tensor
///
/// This is NOT yet functional — it serves as the interface contract for
/// when the upstream API becomes available.
pub struct LayerRangeExecutor {
    pub model_path: String,
    pub layer_start: u32,
    pub layer_end: u32,
}

impl LayerExecutor for LayerRangeExecutor {
    fn execute(
        &self,
        _input: &[u8],
        _cancel: &std::sync::atomic::AtomicBool,
        _output_tx: std::sync::mpsc::Sender<LayerOutput>,
    ) -> Result<()> {
        anyhow::bail!(
            "LayerRangeExecutor is not yet implemented. \
             Waiting for llama_cpp to expose per-layer forward pass API. \
             Use FullModelExecutor (position 0) in the meantime."
        )
    }

    fn layer_range(&self) -> (u32, u32) {
        (self.layer_start, self.layer_end)
    }

    fn is_full_model(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_output_serialization() {
        let token = LayerOutput::Token {
            text: "hello".into(),
            seq: 1,
        };
        let json = serde_json::to_string(&token).unwrap();
        assert!(json.contains("\"type\":\"Token\""));
        assert!(json.contains("\"text\":\"hello\""));

        let done = LayerOutput::Done { total_tokens: 42 };
        let json = serde_json::to_string(&done).unwrap();
        assert!(json.contains("\"type\":\"Done\""));
        assert!(json.contains("\"total_tokens\":42"));
    }

    #[test]
    fn test_full_model_executor_layer_range() {
        let exec = FullModelExecutor {
            model_path: "/tmp/test.gguf".into(),
            layer_start: 0,
            layer_end: 15,
            context_size: 2048,
            temperature: 0.7,
            max_tokens: 100,
        };
        assert_eq!(exec.layer_range(), (0, 15));
        assert!(exec.is_full_model());
    }

    #[test]
    fn test_layer_range_executor_not_implemented() {
        let exec = LayerRangeExecutor {
            model_path: "/tmp/test.gguf".into(),
            layer_start: 16,
            layer_end: 31,
        };
        assert_eq!(exec.layer_range(), (16, 31));
        assert!(!exec.is_full_model());

        let cancel = std::sync::atomic::AtomicBool::new(false);
        let (tx, _rx) = std::sync::mpsc::channel();
        let result = exec.execute(b"test", &cancel, tx);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}
