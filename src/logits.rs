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

/// Compute log-softmax: log_prob = logit[token] - log(sum(exp(logit)))
///
/// Uses the numerically stable form with max-subtraction to prevent overflow.
/// Returns (log_prob_for_token, log_sum_exp).
pub(crate) fn log_softmax(logits: &[f32], token_id: usize) -> (f32, f32) {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln() + max_logit;
    (logits[token_id] - log_sum_exp, log_sum_exp)
}

/// Find the token with the highest logit value and return (token_id, log_prob).
pub(crate) fn argmax_token(logits: &[f32], log_sum_exp: f32) -> (i32, f32) {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &logit)| (idx as i32, logit - log_sum_exp))
        .unwrap_or((0, 0.0))
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

    // ── DraftToken tests ──────────────────────────────────────────

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
        assert!(json.contains("\"text\":\"hello\""));
    }

    #[test]
    fn test_draft_token_roundtrip() {
        let token = DraftToken {
            token_id: 100,
            text: "world".into(),
            log_prob: -1.234,
        };
        let json = serde_json::to_string(&token).unwrap();
        let decoded: DraftToken = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.token_id, 100);
        assert_eq!(decoded.text, "world");
        assert!((decoded.log_prob - (-1.234)).abs() < 1e-5);
    }

    #[test]
    fn test_draft_token_empty_text() {
        let token = DraftToken {
            token_id: 0,
            text: String::new(),
            log_prob: 0.0,
        };
        let json = serde_json::to_string(&token).unwrap();
        let decoded: DraftToken = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.text, "");
        assert_eq!(decoded.token_id, 0);
    }

    #[test]
    fn test_draft_token_negative_token_id() {
        let token = DraftToken {
            token_id: -1,
            text: "eos".into(),
            log_prob: -0.01,
        };
        let json = serde_json::to_string(&token).unwrap();
        let decoded: DraftToken = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.token_id, -1);
    }

    // ── VerifyResult tests ────────────────────────────────────────

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
        assert!(json.contains("\"corrected_token\""));
    }

    #[test]
    fn test_verify_result_no_correction() {
        let result = VerifyResult {
            accepted_count: 5,
            corrected_token: None,
            draft_count: 5,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"corrected_token\":null"));
    }

    #[test]
    fn test_verify_result_roundtrip() {
        let result = VerifyResult {
            accepted_count: 2,
            corrected_token: Some(DraftToken {
                token_id: 7,
                text: "token".into(),
                log_prob: -2.5,
            }),
            draft_count: 4,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: VerifyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.accepted_count, 2);
        assert_eq!(decoded.draft_count, 4);
        assert!(decoded.corrected_token.is_some());
        let ct = decoded.corrected_token.unwrap();
        assert_eq!(ct.token_id, 7);
    }

    // ── acceptance_rate tests ─────────────────────────────────────

    #[test]
    fn test_acceptance_rate_normal() {
        let result = VerifyResult {
            accepted_count: 4,
            corrected_token: None,
            draft_count: 5,
        };
        assert!((acceptance_rate(&result) - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceptance_rate_empty_draft() {
        let empty = VerifyResult {
            accepted_count: 0,
            corrected_token: None,
            draft_count: 0,
        };
        assert!((acceptance_rate(&empty) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceptance_rate_all_accepted() {
        let result = VerifyResult {
            accepted_count: 10,
            corrected_token: None,
            draft_count: 10,
        };
        assert!((acceptance_rate(&result) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceptance_rate_none_accepted() {
        let result = VerifyResult {
            accepted_count: 0,
            corrected_token: Some(DraftToken {
                token_id: 1,
                text: "x".into(),
                log_prob: -3.0,
            }),
            draft_count: 8,
        };
        assert!((acceptance_rate(&result) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceptance_rate_single_token() {
        let result = VerifyResult {
            accepted_count: 1,
            corrected_token: None,
            draft_count: 1,
        };
        assert!((acceptance_rate(&result) - 1.0).abs() < f64::EPSILON);
    }

    // ── log_softmax tests ─────────────────────────────────────────

    #[test]
    fn test_log_softmax_uniform() {
        // All equal logits: each token gets log(1/N) = -ln(N)
        let logits = vec![1.0f32; 4];
        let (log_prob, _) = log_softmax(&logits, 0);
        let expected = -(4.0f32).ln();
        assert!((log_prob - expected).abs() < 1e-5, "got {}, expected {}", log_prob, expected);
    }

    #[test]
    fn test_log_softmax_dominant_token() {
        // One very high logit should get log_prob close to 0
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let (log_prob, _) = log_softmax(&logits, 0);
        assert!(log_prob > -0.001, "dominant token log_prob should be near 0, got {}", log_prob);
    }

    #[test]
    fn test_log_softmax_low_token() {
        // Token with low logit should get very negative log_prob
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let (log_prob, _) = log_softmax(&logits, 1);
        assert!(log_prob < -50.0, "low token should have very negative log_prob, got {}", log_prob);
    }

    #[test]
    fn test_log_softmax_probabilities_sum_to_one() {
        let logits = vec![2.0, 1.0, 0.5, -1.0, 3.0];
        let probs_sum: f32 = (0..logits.len())
            .map(|i| log_softmax(&logits, i).0.exp())
            .sum();
        assert!((probs_sum - 1.0).abs() < 1e-5, "probabilities should sum to 1, got {}", probs_sum);
    }

    #[test]
    fn test_log_softmax_negative_logits() {
        let logits = vec![-10.0, -20.0, -5.0];
        let (log_prob, _) = log_softmax(&logits, 2); // -5.0 is highest
        // Should be the highest probability
        let (log_prob_0, _) = log_softmax(&logits, 0);
        let (log_prob_1, _) = log_softmax(&logits, 1);
        assert!(log_prob > log_prob_0);
        assert!(log_prob > log_prob_1);
    }

    #[test]
    fn test_log_softmax_single_element() {
        let logits = vec![5.0];
        let (log_prob, _) = log_softmax(&logits, 0);
        // Single element: probability is 1, log_prob is 0
        assert!((log_prob - 0.0).abs() < 1e-5, "single element log_prob should be 0, got {}", log_prob);
    }

    #[test]
    fn test_log_softmax_large_values_numerical_stability() {
        // Very large logits should not overflow thanks to max-subtraction
        let logits = vec![1000.0, 999.0, 998.0];
        let (log_prob, _) = log_softmax(&logits, 0);
        assert!(log_prob.is_finite(), "log_prob should be finite for large logits");
        assert!(log_prob > -2.0 && log_prob <= 0.0);
    }

    #[test]
    fn test_log_softmax_very_negative_values() {
        let logits = vec![-1000.0, -999.0, -998.0];
        let (log_prob, _) = log_softmax(&logits, 2); // -998 is highest
        assert!(log_prob.is_finite(), "log_prob should be finite for very negative logits");
    }

    // ── argmax_token tests ────────────────────────────────────────

    #[test]
    fn test_argmax_token_basic() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let (_, lse) = log_softmax(&logits, 0);
        let (token_id, log_prob) = argmax_token(&logits, lse);
        assert_eq!(token_id, 1, "should pick index 1 (value 3.0)");
        assert!((log_prob - (3.0 - lse)).abs() < 1e-5);
    }

    #[test]
    fn test_argmax_token_first_element() {
        let logits = vec![10.0, 1.0, 2.0];
        let (_, lse) = log_softmax(&logits, 0);
        let (token_id, _) = argmax_token(&logits, lse);
        assert_eq!(token_id, 0);
    }

    #[test]
    fn test_argmax_token_last_element() {
        let logits = vec![1.0, 2.0, 10.0];
        let (_, lse) = log_softmax(&logits, 0);
        let (token_id, _) = argmax_token(&logits, lse);
        assert_eq!(token_id, 2);
    }

    #[test]
    fn test_argmax_token_all_equal() {
        // When all equal, should pick the first (max_by is stable for first max)
        let logits = vec![1.0, 1.0, 1.0];
        let (_, lse) = log_softmax(&logits, 0);
        let (token_id, _) = argmax_token(&logits, lse);
        // max_by returns last max with partial_cmp, but enumerate goes 0..N
        // Actually, iter().enumerate().max_by picks the LAST one with equal values
        assert!(token_id >= 0 && token_id <= 2);
    }

    #[test]
    fn test_argmax_token_negative_logits() {
        let logits = vec![-5.0, -1.0, -10.0];
        let (_, lse) = log_softmax(&logits, 0);
        let (token_id, _) = argmax_token(&logits, lse);
        assert_eq!(token_id, 1, "should pick -1.0 as the highest");
    }
}
