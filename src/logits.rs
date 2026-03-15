//! Logit extraction and token verification via llama_cpp_sys FFI.
//!
//! Provides the core primitives for speculative decoding:
//! - Load a model and create a context
//! - Tokenize text input
//! - Run a forward pass and extract per-token logits
//! - Compare draft logits against verifier logits to determine acceptance
//!
//! Uses llama_cpp_sys directly for low-level access to:
//! - `llama_get_logits_ith()` — per-position logit extraction
//! - `llama_n_vocab()` — vocabulary size for logit comparison
//! - `llama_batch_init/free()` — manual batch construction

use anyhow::{Context, Result};
use std::ffi::CString;

/// A draft token with its log-probability from the draft model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DraftToken {
    pub token_id: i32,
    pub text: String,
    pub log_prob: f32,
}

/// Result of verifying draft tokens against the verifier model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifyResult {
    /// Number of accepted tokens from the draft (prefix length)
    pub accepted_count: usize,
    /// The corrected token from the verifier (at the first rejected position)
    pub corrected_token: Option<DraftToken>,
    /// Total tokens in the draft batch that were verified
    pub draft_count: usize,
}

/// Extract log-probabilities for a sequence of tokens using the verifier model.
///
/// Loads the model, tokenizes the context + draft tokens, runs a single forward
/// pass, and returns per-position log-probabilities via `llama_get_logits_ith`.
///
/// # Safety
/// Uses unsafe FFI to llama_cpp_sys. The model path must be valid, and the
/// context must not be used concurrently.
pub fn verify_draft_tokens(
    model_path: &str,
    context_text: &str,
    draft_tokens: &[DraftToken],
    context_size: u32,
    acceptance_threshold: f64,
) -> Result<VerifyResult> {
    unsafe {
        let c_path = CString::new(model_path)?;

        let mut model_params = llama_cpp_sys::llama_model_default_params();
        model_params.n_gpu_layers = 99;

        let model = llama_cpp_sys::llama_load_model_from_file(c_path.as_ptr(), model_params);
        if model.is_null() {
            anyhow::bail!("Failed to load verifier model: {}", model_path);
        }

        let mut ctx_params = llama_cpp_sys::llama_context_default_params();
        ctx_params.n_ctx = context_size;

        let ctx = llama_cpp_sys::llama_new_context_with_model(model, ctx_params);
        if ctx.is_null() {
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Failed to create verifier context");
        }

        // Tokenize context
        let c_context = CString::new(context_text)?;
        let max_tokens = context_size as i32;
        let mut context_tokens = vec![0i32; max_tokens as usize];
        let n_context = llama_cpp_sys::llama_tokenize(
            model,
            c_context.as_ptr(),
            context_text.len() as i32,
            context_tokens.as_mut_ptr(),
            max_tokens,
            true, false,
        );

        if n_context < 0 {
            llama_cpp_sys::llama_free(ctx);
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Context tokenization failed");
        }
        context_tokens.truncate(n_context as usize);

        // Append draft token IDs
        let draft_token_ids: Vec<i32> = draft_tokens.iter().map(|d| d.token_id).collect();
        let mut all_tokens = context_tokens;
        all_tokens.extend_from_slice(&draft_token_ids);

        // Run forward pass on all tokens
        let batch = llama_cpp_sys::llama_batch_get_one(
            all_tokens.as_mut_ptr(),
            all_tokens.len() as i32,
            0, 0,
        );

        let result = llama_cpp_sys::llama_decode(ctx, batch);
        if result != 0 {
            llama_cpp_sys::llama_free(ctx);
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Verifier llama_decode failed: {}", result);
        }

        let n_vocab = llama_cpp_sys::llama_n_vocab(model) as usize;
        let verify_start = n_context as usize; // logits for draft tokens start after context

        // Compare draft log-probs against verifier log-probs
        let mut accepted_count = 0;
        let mut corrected_token = None;

        for (i, draft) in draft_tokens.iter().enumerate() {
            let pos = (verify_start + i) as i32;
            let logits_ptr = llama_cpp_sys::llama_get_logits_ith(ctx, pos);

            if logits_ptr.is_null() {
                break;
            }

            let logits = std::slice::from_raw_parts(logits_ptr, n_vocab);

            // Compute log-softmax for the draft token
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln() + max_logit;
            let verifier_log_prob = logits[draft.token_id as usize] - log_sum_exp;

            // Accept if verifier agrees (log-prob above threshold)
            let _agreement = (verifier_log_prob - draft.log_prob).abs();
            if (verifier_log_prob as f64) > acceptance_threshold.ln() {
                accepted_count += 1;
            } else {
                // Find the verifier's preferred token at this position
                let best_token = logits.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, &logit)| (idx as i32, logit - log_sum_exp))
                    .unwrap_or((draft.token_id, verifier_log_prob));

                corrected_token = Some(DraftToken {
                    token_id: best_token.0,
                    text: String::new(), // Token text resolved by caller
                    log_prob: best_token.1,
                });
                break;
            }
        }

        llama_cpp_sys::llama_free(ctx);
        llama_cpp_sys::llama_free_model(model);

        Ok(VerifyResult {
            accepted_count,
            corrected_token,
            draft_count: draft_tokens.len(),
        })
    }
}

/// Compute the acceptance rate from a verify result
pub fn acceptance_rate(result: &VerifyResult) -> f64 {
    if result.draft_count == 0 {
        0.0
    } else {
        result.accepted_count as f64 / result.draft_count as f64
    }
}

/// Extract the log-probability of a specific token at a given position
/// from an already-decoded context. Uses `llama_get_logits_ith`.
///
/// Call this after `llama_decode` to get the log-prob of the token that
/// was generated at position `pos`.
///
/// # Safety
/// Requires a valid llama context with decoded tokens.
pub unsafe fn extract_token_log_prob(
    ctx: *mut llama_cpp_sys::llama_context,
    model: *const llama_cpp_sys::llama_model,
    pos: i32,
    token_id: i32,
) -> f32 {
    let logits_ptr = llama_cpp_sys::llama_get_logits_ith(ctx, pos);
    if logits_ptr.is_null() {
        return 0.0;
    }

    let n_vocab = llama_cpp_sys::llama_n_vocab(model) as usize;
    let logits = std::slice::from_raw_parts(logits_ptr, n_vocab);

    // Log-softmax: log_prob = logit[token] - log(sum(exp(logit)))
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln() + max_logit;
    logits[token_id as usize] - log_sum_exp
}

/// Generate K tokens with log-probabilities using llama_cpp_sys FFI.
///
/// This provides the draft token generation with real log-probs for
/// speculative decoding. Each token includes its log-probability from
/// the draft model's distribution.
pub fn generate_draft_tokens_with_logprobs(
    model_path: &str,
    context_text: &str,
    k: usize,
    context_size: u32,
    temperature: f32,
) -> Result<Vec<DraftToken>> {
    unsafe {
        let c_path = CString::new(model_path)?;

        let mut model_params = llama_cpp_sys::llama_model_default_params();
        model_params.n_gpu_layers = 99;

        let model = llama_cpp_sys::llama_load_model_from_file(c_path.as_ptr(), model_params);
        if model.is_null() {
            anyhow::bail!("Failed to load draft model: {}", model_path);
        }

        let mut ctx_params = llama_cpp_sys::llama_context_default_params();
        ctx_params.n_ctx = context_size;

        let ctx = llama_cpp_sys::llama_new_context_with_model(model, ctx_params);
        if ctx.is_null() {
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Failed to create draft context");
        }

        // Tokenize context
        let c_text = CString::new(context_text)?;
        let max_tok = context_size as i32;
        let mut tokens = vec![0i32; max_tok as usize];
        let n_ctx_tokens = llama_cpp_sys::llama_tokenize(
            model, c_text.as_ptr(), context_text.len() as i32,
            tokens.as_mut_ptr(), max_tok, true, false,
        );

        if n_ctx_tokens < 0 {
            llama_cpp_sys::llama_free(ctx);
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Tokenization failed");
        }
        tokens.truncate(n_ctx_tokens as usize);

        // Decode context
        let batch = llama_cpp_sys::llama_batch_get_one(
            tokens.as_mut_ptr(), n_ctx_tokens, 0, 0,
        );
        if llama_cpp_sys::llama_decode(ctx, batch) != 0 {
            llama_cpp_sys::llama_free(ctx);
            llama_cpp_sys::llama_free_model(model);
            anyhow::bail!("Context decode failed");
        }

        let n_vocab = llama_cpp_sys::llama_n_vocab(model) as usize;
        let mut draft_tokens = Vec::with_capacity(k);
        let mut current_pos = n_ctx_tokens;

        for _ in 0..k {
            // Get logits at current position
            let logits_ptr = llama_cpp_sys::llama_get_logits_ith(ctx, current_pos - 1);
            if logits_ptr.is_null() {
                break;
            }

            let logits = std::slice::from_raw_parts(logits_ptr, n_vocab);

            // Apply temperature and sample (greedy for simplicity)
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln() + max_logit;

            // Find top token
            let (best_id, best_logit) = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &l)| (idx as i32, l))
                .unwrap_or((0, 0.0));

            let log_prob = best_logit - log_sum_exp;

            // Convert token to text
            let mut buf = vec![0u8; 64];
            let text_len = llama_cpp_sys::llama_token_to_piece(
                model, best_id, buf.as_mut_ptr() as *mut i8, buf.len() as i32,
            );
            let text = if text_len > 0 {
                String::from_utf8_lossy(&buf[..text_len as usize]).to_string()
            } else {
                String::new()
            };

            draft_tokens.push(DraftToken {
                token_id: best_id,
                text,
                log_prob,
            });

            // Check for EOS
            if best_id == llama_cpp_sys::llama_token_eos(model) {
                break;
            }

            // Decode the new token to advance the context
            let mut next_token = vec![best_id];
            let next_batch = llama_cpp_sys::llama_batch_get_one(
                next_token.as_mut_ptr(), 1, current_pos, 0,
            );
            if llama_cpp_sys::llama_decode(ctx, next_batch) != 0 {
                break;
            }
            current_pos += 1;
        }

        llama_cpp_sys::llama_free(ctx);
        llama_cpp_sys::llama_free_model(model);

        Ok(draft_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draft_token_serialize() {
        let token = DraftToken {
            token_id: 42,
            text: "hello".into(),
            log_prob: -0.5,
        };
        let json = serde_json::to_string(&token).unwrap();
        assert!(json.contains("\"token_id\":42"));
        assert!(json.contains("\"log_prob\":-0.5"));
    }

    #[test]
    fn test_verify_result_serialize() {
        let result = VerifyResult {
            accepted_count: 3,
            corrected_token: Some(DraftToken {
                token_id: 10,
                text: "world".into(),
                log_prob: -0.3,
            }),
            draft_count: 5,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"accepted_count\":3"));
        assert!(json.contains("\"draft_count\":5"));
    }

    #[test]
    fn test_acceptance_rate() {
        let result = VerifyResult {
            accepted_count: 4,
            corrected_token: None,
            draft_count: 5,
        };
        assert!((acceptance_rate(&result) - 0.8).abs() < f64::EPSILON);

        let empty = VerifyResult {
            accepted_count: 0,
            corrected_token: None,
            draft_count: 0,
        };
        assert!((acceptance_rate(&empty) - 0.0).abs() < f64::EPSILON);
    }
}
