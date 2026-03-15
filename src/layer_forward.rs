//! Per-layer forward pass integration for when llama.cpp exposes a native API.
//!
//! This module defines the interface and integration code for running a
//! forward pass through only a subset of transformer layers. It will be
//! activated when the upstream llama.cpp adds `llama_decode_layers()`.
//!
//! ## Upstream Contribution Plan
//!
//! The following C API additions are needed in llama.cpp:
//!
//! ```c
//! // Run forward pass through layers [start, end) only.
//! // Requires the KV cache to contain the output of layer (start-1).
//! // After this call, llama_get_logits() returns the output of layer (end).
//! LLAMA_API int32_t llama_decode_layers(
//!     struct llama_context * ctx,
//!     struct llama_batch     batch,
//!     int32_t                layer_start,  // inclusive
//!     int32_t                layer_end     // exclusive
//! );
//!
//! // Get the hidden state (embedding) after the last decoded layer.
//! // Size: n_embd floats. Returns NULL if no decode has been run.
//! LLAMA_API float * llama_get_hidden_state(struct llama_context * ctx);
//!
//! // Set the hidden state as input for the next llama_decode_layers() call.
//! // This replaces the normal token embedding lookup.
//! LLAMA_API int32_t llama_set_hidden_state(
//!     struct llama_context * ctx,
//!     const float          * data,
//!     int32_t                n_embd
//! );
//! ```
//!
//! ## Integration (when available)
//!
//! The Rust side would call these via llama_cpp_sys FFI:
//!
//! ```rust,ignore
//! // Position N (not first, not last):
//! unsafe {
//!     // 1. Set hidden state from previous position's output
//!     llama_cpp_sys::llama_set_hidden_state(ctx, activation_data.as_ptr(), n_embd);
//!
//!     // 2. Run only our layer range
//!     let result = llama_cpp_sys::llama_decode_layers(ctx, batch, layer_start, layer_end);
//!
//!     // 3. Extract hidden state for next position
//!     let hidden = llama_cpp_sys::llama_get_hidden_state(ctx);
//!     let output = std::slice::from_raw_parts(hidden, n_embd as usize);
//!
//!     // 4. Publish to next position
//!     nats.publish_raw(next_subject, output_bytes).await?;
//! }
//! ```

use anyhow::Result;

/// Check if the current llama_cpp_sys version supports per-layer forward.
///
/// Returns false for llama_cpp_sys 0.3.2 (no layer API).
/// Will return true when the upstream API is available.
pub fn supports_layer_forward() -> bool {
    // When the API is available, this will check for the symbol:
    // unsafe { llama_cpp_sys::llama_decode_layers as *const () != std::ptr::null() }
    false
}

/// Proposed activation format for inter-position communication.
///
/// When per-layer forward is available, each position sends the hidden
/// state (not tokens) to the next position. This is the format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerActivation {
    /// The hidden state vector (n_embd f32 values, serialized as bytes)
    pub hidden_state: Vec<u8>,
    /// Embedding dimension
    pub n_embd: u32,
    /// Which layer produced this activation
    pub from_layer: u32,
    /// Sequence position (for KV cache alignment)
    pub seq_pos: u32,
    /// Job ID for correlation
    pub job_id: String,
}

impl LayerActivation {
    /// Serialize to bytes for NATS transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(Into::into)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(Into::into)
    }

    /// Extract the hidden state as f32 slice
    pub fn as_f32_slice(&self) -> &[f32] {
        // Safety: hidden_state was created from f32 values
        let ptr = self.hidden_state.as_ptr() as *const f32;
        let len = self.hidden_state.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_layer_forward() {
        // Currently false — will be true when upstream API lands
        assert!(!supports_layer_forward());
    }

    #[test]
    fn test_layer_activation_roundtrip() {
        // Create an activation with 4 f32 values
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let activation = LayerActivation {
            hidden_state: bytes,
            n_embd: 4,
            from_layer: 15,
            seq_pos: 0,
            job_id: "test-123".into(),
        };

        let serialized = activation.to_bytes().unwrap();
        let deserialized = LayerActivation::from_bytes(&serialized).unwrap();

        assert_eq!(deserialized.n_embd, 4);
        assert_eq!(deserialized.from_layer, 15);
        assert_eq!(deserialized.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
