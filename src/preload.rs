//! Startup preloading and demand-based warmup for ML models
//!
//! At startup, the Island selects a starter set of models to preload based
//! on its hardware capabilities. As jobs execute, the model cache keeps
//! recently-used models warm. The preloader also tracks demand patterns
//! and proactively caches popular models.
//!
//! ## Hardware tiers
//!
//! | Tier   | RAM      | GPU           | Starter models                              |
//! |--------|----------|---------------|---------------------------------------------|
//! | Tiny   | ≤2 GB    | —             | WASM only, no model preloads                |
//! | Small  | 2–8 GB   | —             | qwen3.5-0.8b (GGUF)                        |
//! | Medium | 8–16 GB  | —             | + mistral-7b, sentiment, embeddings (ONNX)  |
//! | GPU    | 8+ GB    | ≤8 GB VRAM    | + whisper, detect (ONNX)                    |
//! | Large  | 16+ GB   | 12+ GB VRAM   | + stable-diffusion, flux1-schnell           |
//! | XL     | 24+ GB   | 24+ GB VRAM   | + large GGUF, video gen                     |

use std::sync::Arc;
use tracing::{info, warn};

use crate::config::PreloadConfig;
use crate::model_cache::ModelCache;

/// A model to preload at startup
#[derive(Debug, Clone)]
#[allow(dead_code)] // runtime field used by future preload filtering
pub struct PreloadEntry {
    /// HuggingFace URI (hf://repo_id or hf://repo_id:filename)
    pub uri: String,
    /// Human-readable name for logging
    pub name: String,
    /// Minimum RAM in MB required to preload this model
    pub min_ram_mb: u32,
    /// Minimum VRAM in MB required (0 = CPU-only)
    pub min_vram_mb: u32,
    /// Runtime type this model belongs to
    pub runtime: &'static str,
}

/// Select starter models based on hardware capabilities
pub fn select_starter_models(
    ram_mb: u32,
    vram_mb: Option<u32>,
    config: &PreloadConfig,
) -> Vec<PreloadEntry> {
    if !config.enabled {
        info!("Model preloading disabled");
        return Vec::new();
    }

    // If explicit models configured, use those
    if !config.models.is_empty() {
        return config
            .models
            .iter()
            .map(|uri| PreloadEntry {
                uri: uri.clone(),
                name: uri.clone(),
                min_ram_mb: 0,
                min_vram_mb: 0,
                runtime: "llmcpp",
            })
            .collect();
    }

    let vram = vram_mb.unwrap_or(0);
    let mut models = Vec::new();

    // Tiny tier: ≤2 GB RAM — WASM only
    if ram_mb <= 2048 {
        info!(
            "Hardware tier: tiny ({}MB RAM) — WASM only, no model preloads",
            ram_mb
        );
        return models;
    }

    // Small tier: 2-8 GB RAM — smallest public LLM
    #[cfg(feature = "gguf")]
    models.push(PreloadEntry {
        uri: "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
        name: "TinyLlama 1.1B Chat (GGUF, 670MB)".into(),
        min_ram_mb: 2048,
        min_vram_mb: 0,
        runtime: "llmcpp",
    });

    // Medium tier: 8+ GB RAM — mid-size LLM + basic ONNX models
    if ram_mb >= 8192 {
        #[cfg(feature = "gguf")]
        models.push(PreloadEntry {
            uri: "hf://TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
            name: "Mistral 7B Instruct (GGUF, 4.4GB)".into(),
            min_ram_mb: 8192,
            min_vram_mb: 0,
            runtime: "llmcpp",
        });

        #[cfg(feature = "onnx")]
        {
            models.push(PreloadEntry {
                uri: "hf://distilbert-base-uncased-finetuned-sst-2-english:model.onnx".into(),
                name: "DistilBERT Sentiment (ONNX, 268MB)".into(),
                min_ram_mb: 4096,
                min_vram_mb: 0,
                runtime: "onnx",
            });
            models.push(PreloadEntry {
                uri: "hf://sentence-transformers/all-MiniLM-L6-v2:onnx/model.onnx".into(),
                name: "MiniLM-L6 Embeddings (ONNX, 90MB)".into(),
                min_ram_mb: 4096,
                min_vram_mb: 0,
                runtime: "onnx",
            });
        }
    }

    // GPU tier: has a GPU with ≤8 GB VRAM
    if vram >= 2048 {
        #[cfg(feature = "onnx")]
        {
            models.push(PreloadEntry {
                uri: "hf://openai/whisper-base".into(),
                name: "Whisper Base (ONNX)".into(),
                min_ram_mb: 2048,
                min_vram_mb: 1024,
                runtime: "onnx",
            });
            models.push(PreloadEntry {
                uri: "hf://ultralytics/yolov8n".into(),
                name: "YOLOv8 Detection (ONNX)".into(),
                min_ram_mb: 1024,
                min_vram_mb: 512,
                runtime: "onnx",
            });
        }
    }

    // Large tier: 16+ GB RAM, 12+ GB VRAM — diffusers
    if ram_mb >= 16384 && vram >= 8000 {
        #[cfg(feature = "diffusers")]
        models.push(PreloadEntry {
            uri: "hf://black-forest-labs/FLUX.1-schnell".into(),
            name: "FLUX.1-schnell (Diffusers)".into(),
            min_ram_mb: 16384,
            min_vram_mb: 8000,
            runtime: "diffusers",
        });
    }

    // XL tier: 24+ GB VRAM — large models
    if vram >= 24000 {
        #[cfg(feature = "gguf")]
        models.push(PreloadEntry {
            uri: "hf://unsloth/Qwen3.5-27B-GGUF".into(),
            name: "Qwen3.5-27B (GGUF)".into(),
            min_ram_mb: 24000,
            min_vram_mb: 18000,
            runtime: "llmcpp",
        });
    }

    // Filter by actual hardware
    models.retain(|m| ram_mb >= m.min_ram_mb && vram >= m.min_vram_mb);

    info!(
        "Hardware tier: {}MB RAM, {}MB VRAM — {} models selected for preload",
        ram_mb,
        vram,
        models.len()
    );
    for m in &models {
        info!("  Preload: {}", m.name);
    }

    models
}

/// Preload selected models into the cache (background, non-blocking)
pub async fn preload_models(cache: &Arc<ModelCache>, models: Vec<PreloadEntry>) {
    if models.is_empty() {
        return;
    }

    info!("Preloading {} models...", models.len());

    for model in &models {
        info!("Preloading: {}", model.name);
        match cache.download_model(&model.uri, None).await {
            Ok(path) => {
                info!("Preloaded: {} → {}", model.name, path.display());
            }
            Err(e) => {
                warn!("Failed to preload {} (non-fatal): {}", model.name, e);
            }
        }
    }

    info!("Model preloading complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PreloadConfig {
        PreloadConfig {
            enabled: true,
            models: Vec::new(),
        }
    }

    #[test]
    fn test_tiny_tier_no_models() {
        let models = select_starter_models(1024, None, &default_config());
        assert!(models.is_empty());
    }

    #[test]
    fn test_disabled_returns_empty() {
        let config = PreloadConfig {
            enabled: false,
            models: Vec::new(),
        };
        let models = select_starter_models(32000, Some(24000), &config);
        assert!(models.is_empty());
    }

    #[test]
    fn test_explicit_models_override() {
        let config = PreloadConfig {
            enabled: true,
            models: vec!["hf://my/model".into()],
        };
        let models = select_starter_models(2048, None, &config);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].uri, "hf://my/model");
    }

    #[test]
    fn test_small_tier_gets_small_model() {
        let models = select_starter_models(4096, None, &default_config());
        // Should have at least one model (TinyLlama) if gguf feature is on
        #[cfg(feature = "gguf")]
        assert!(models.iter().any(|m| m.uri.contains("TinyLlama")));
    }

    #[test]
    fn test_medium_tier_gets_more_models() {
        let models = select_starter_models(16384, None, &default_config());
        #[cfg(feature = "gguf")]
        assert!(models.iter().any(|m| m.uri.contains("Mistral-7B")));
    }

    #[test]
    fn test_vram_filter_applied() {
        // No GPU — shouldn't get GPU-requiring models
        let models = select_starter_models(16384, None, &default_config());
        assert!(models.iter().all(|m| m.min_vram_mb == 0));
    }
}
