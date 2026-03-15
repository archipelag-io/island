//! Per-layer forward pass via llama_cpp_sys FFI.
//!
//! Provides low-level access to llama.cpp's C API for embedding extraction
//! and partial model execution. This enables pipeline stages to:
//! - Process a prompt and extract the hidden state (embedding vector)
//! - Feed an embedding vector as input and extract the next hidden state
//! - Run final logits and sample tokens from the last position
//!
//! Two execution modes:
//! - `FullModelExecutor`: Loads complete GGUF, runs full inference (position 0 default)
//! - `EmbeddingExecutor`: Loads GGUF, runs decode, extracts embeddings via FFI
//!
//! The FFI functions used (from llama_cpp_sys):
//! - `llama_load_model_from_file` — load GGUF model
//! - `llama_new_context_with_model` — create inference context
//! - `llama_tokenize` — convert text to token IDs
//! - `llama_decode` — run forward pass on a batch of tokens
//! - `llama_get_embeddings` — extract hidden state after decode
//! - `llama_n_embd` — get embedding dimension size

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

/// The output of a layer executor — either tokens or raw activations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerOutput {
    /// Token string output (from full model inference)
    Token { text: String, seq: u64 },
    /// Raw embedding/activation vector (f32 values serialized as bytes)
    Activation { data: Vec<u8>, seq: u64, n_embd: u32 },
    /// Final output marker
    Done { total_tokens: u64 },
}

/// Trait for executing model layers on input data.
pub trait LayerExecutor: Send + Sync {
    /// Execute on the given input.
    /// For text input: `input` is UTF-8 prompt bytes.
    /// For activation input: `input` is serialized f32 embedding bytes.
    fn execute(
        &self,
        input: &[u8],
        cancel: &AtomicBool,
        output_tx: std::sync::mpsc::Sender<LayerOutput>,
    ) -> Result<()>;

    fn layer_range(&self) -> (u32, u32);
    fn is_full_model(&self) -> bool;
}

/// Full model executor — loads entire GGUF shard, runs complete inference.
/// This is the current default for position 0 in the pipeline.
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
        cancel: &AtomicBool,
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
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            seq += 1;
            if output_tx.send(LayerOutput::Token {
                text: token_str.to_string(),
                seq,
            }).is_err() {
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

/// Embedding executor — loads GGUF model, tokenizes input, runs llama_decode
/// via FFI, and extracts the hidden state embedding vector.
///
/// This is the building block for true pipeline parallelism: each position
/// loads its shard, processes input (text or activations), and outputs the
/// resulting embedding for the next position.
///
/// Currently loads the FULL model and extracts embeddings after all layers
/// (llama.cpp doesn't support partial layer execution). When upstream adds
/// layer-range support, this executor will be updated to run only its layers.
pub struct EmbeddingExecutor {
    pub model_path: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub context_size: u32,
}

impl LayerExecutor for EmbeddingExecutor {
    fn execute(
        &self,
        input: &[u8],
        cancel: &AtomicBool,
        output_tx: std::sync::mpsc::Sender<LayerOutput>,
    ) -> Result<()> {
        let prompt = std::str::from_utf8(input)
            .map_err(|e| anyhow::anyhow!("Invalid input encoding: {}", e))?;

        unsafe {
            // Load model
            let c_path = CString::new(self.model_path.as_str())
                .map_err(|e| anyhow::anyhow!("Invalid model path: {}", e))?;

            let mut model_params = llama_cpp_sys::llama_model_default_params();
            // Enable embedding mode
            model_params.n_gpu_layers = 99; // offload all layers to GPU if available

            let model = llama_cpp_sys::llama_load_model_from_file(
                c_path.as_ptr(),
                model_params,
            );
            if model.is_null() {
                anyhow::bail!("Failed to load model: {}", self.model_path);
            }

            // Create context with embedding enabled
            let mut ctx_params = llama_cpp_sys::llama_context_default_params();
            ctx_params.n_ctx = self.context_size;
            ctx_params.embeddings = true; // Enable embedding extraction

            let ctx = llama_cpp_sys::llama_new_context_with_model(model, ctx_params);
            if ctx.is_null() {
                llama_cpp_sys::llama_free_model(model);
                anyhow::bail!("Failed to create context");
            }

            // Tokenize input
            let c_prompt = CString::new(prompt)
                .map_err(|e| anyhow::anyhow!("Invalid prompt: {}", e))?;

            let max_tokens = self.context_size as i32;
            let mut tokens = vec![0i32; max_tokens as usize];
            let n_tokens = llama_cpp_sys::llama_tokenize(
                model,
                c_prompt.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                max_tokens,
                true,  // add_bos
                false, // special
            );

            if n_tokens < 0 {
                llama_cpp_sys::llama_free(ctx);
                llama_cpp_sys::llama_free_model(model);
                anyhow::bail!("Tokenization failed (needed {} tokens, buffer is {})", -n_tokens, max_tokens);
            }

            tokens.truncate(n_tokens as usize);

            if cancel.load(Ordering::Relaxed) {
                llama_cpp_sys::llama_free(ctx);
                llama_cpp_sys::llama_free_model(model);
                return Ok(());
            }

            // Run decode (forward pass through all layers)
            let batch = llama_cpp_sys::llama_batch_get_one(
                tokens.as_mut_ptr(),
                n_tokens,
                0, // pos
                0, // seq_id
            );

            let decode_result = llama_cpp_sys::llama_decode(ctx, batch);
            if decode_result != 0 {
                llama_cpp_sys::llama_free(ctx);
                llama_cpp_sys::llama_free_model(model);
                anyhow::bail!("llama_decode failed with code {}", decode_result);
            }

            // Extract embeddings
            let n_embd = llama_cpp_sys::llama_n_embd(model) as usize;
            let embd_ptr = llama_cpp_sys::llama_get_embeddings(ctx);

            if embd_ptr.is_null() {
                llama_cpp_sys::llama_free(ctx);
                llama_cpp_sys::llama_free_model(model);
                anyhow::bail!("llama_get_embeddings returned null (is embedding mode enabled?)");
            }

            // Copy embedding data to a Vec<u8>
            let embd_slice = std::slice::from_raw_parts(embd_ptr, n_embd);
            let embd_bytes: Vec<u8> = embd_slice.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();

            // Send the activation
            let _ = output_tx.send(LayerOutput::Activation {
                data: embd_bytes,
                seq: 1,
                n_embd: n_embd as u32,
            });
            let _ = output_tx.send(LayerOutput::Done { total_tokens: n_tokens as u64 });

            // Cleanup
            llama_cpp_sys::llama_free(ctx);
            llama_cpp_sys::llama_free_model(model);
        }

        Ok(())
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
        let token = LayerOutput::Token { text: "hello".into(), seq: 1 };
        let json = serde_json::to_string(&token).unwrap();
        assert!(json.contains("\"type\":\"Token\""));

        let activation = LayerOutput::Activation {
            data: vec![0, 0, 128, 63], // 1.0f32 in LE bytes
            seq: 1,
            n_embd: 1,
        };
        let json = serde_json::to_string(&activation).unwrap();
        assert!(json.contains("\"type\":\"Activation\""));
        assert!(json.contains("\"n_embd\":1"));

        let done = LayerOutput::Done { total_tokens: 42 };
        let json = serde_json::to_string(&done).unwrap();
        assert!(json.contains("\"total_tokens\":42"));
    }

    #[test]
    fn test_full_model_executor_properties() {
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
    fn test_embedding_executor_properties() {
        let exec = EmbeddingExecutor {
            model_path: "/tmp/test.gguf".into(),
            layer_start: 16,
            layer_end: 31,
            context_size: 2048,
        };
        assert_eq!(exec.layer_range(), (16, 31));
        assert!(!exec.is_full_model());
    }
}
