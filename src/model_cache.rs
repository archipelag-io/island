//! Model cache for downloaded ML models (GGUF, ONNX, diffusers)
//!
//! Downloads models from URLs or HuggingFace repo IDs, verifies SHA256
//! hashes, and manages disk usage with LRU eviction.
//!
//! ## URL schemes
//!
//! - `hf://repo_id` — resolves to HuggingFace Hub, auto-discovers model file
//! - `hf://repo_id:filename` — resolves to a specific file in the HF repo
//! - `https://...` — direct download URL

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::ModelCacheConfig;

/// Metadata for a cached model file
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct CachedModel {
    path: PathBuf,
    size_bytes: u64,
    last_used: SystemTime,
    url: String,
}

/// Model cache manager
#[allow(dead_code)]
pub struct ModelCache {
    cache_dir: PathBuf,
    max_cache_bytes: u64,
    entries: RwLock<HashMap<String, CachedModel>>,
}

#[allow(dead_code)]
impl ModelCache {
    /// Create a new model cache
    pub fn new(config: &ModelCacheConfig) -> Result<Self> {
        let cache_dir = match &config.cache_dir {
            Some(dir) => PathBuf::from(dir),
            None => {
                let home = dirs::home_dir().context("Cannot determine home directory")?;
                home.join(".island").join("model-cache")
            }
        };

        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Failed to create model cache dir: {}", cache_dir.display()))?;

        let max_cache_bytes = config.max_cache_gb * 1024 * 1024 * 1024;

        let cache = Self {
            cache_dir,
            max_cache_bytes,
            entries: RwLock::new(HashMap::new()),
        };

        Ok(cache)
    }

    /// Initialize the cache by scanning existing files
    pub async fn init(&self) -> Result<()> {
        let mut entries = self.entries.write().await;

        if let Ok(read_dir) = std::fs::read_dir(&self.cache_dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(metadata) = std::fs::metadata(&path) {
                        if let Some(model_file) = find_model_file(&path) {
                            let size = std::fs::metadata(&model_file)
                                .map(|m| m.len())
                                .unwrap_or(0);
                            let key = path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            entries.insert(
                                key,
                                CachedModel {
                                    path: model_file,
                                    size_bytes: size,
                                    last_used: metadata
                                        .modified()
                                        .unwrap_or(SystemTime::UNIX_EPOCH),
                                    url: String::new(),
                                },
                            );
                        }
                    }
                }
            }
        }

        info!(
            "Model cache initialized: {} models in {}",
            entries.len(),
            self.cache_dir.display()
        );
        Ok(())
    }

    /// Download a model if not cached, verify hash, return local path.
    ///
    /// Accepts either a direct URL (`https://...`) or a HuggingFace URI
    /// (`hf://repo_id` or `hf://repo_id:filename`).
    pub async fn download_model(
        &self,
        uri: &str,
        expected_hash: Option<&str>,
    ) -> Result<PathBuf> {
        let resolved = resolve_uri(uri).await?;
        let download_url = &resolved.url;
        let display_name = &resolved.display_name;

        let cache_key = url_to_cache_key(download_url);
        let model_dir = self.cache_dir.join(&cache_key);

        // Check if already cached
        {
            let mut entries = self.entries.write().await;
            if let Some(entry) = entries.get_mut(&cache_key) {
                if entry.path.exists() {
                    entry.last_used = SystemTime::now();
                    let _ = filetime::set_file_mtime(
                        &model_dir,
                        filetime::FileTime::now(),
                    );
                    info!("Model cache hit: {}", display_name);
                    return Ok(entry.path.clone());
                }
            }
        }

        info!("Downloading model: {} ({})", display_name, download_url);

        // Ensure we have space
        self.evict_if_needed(0).await?;

        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("Failed to create model dir: {}", model_dir.display()))?;

        let filename = &resolved.filename;
        let model_path = model_dir.join(filename);
        let tmp_path = model_dir.join(format!("{}.tmp", filename));

        // Streaming download
        let client = reqwest::Client::new();
        let response = client
            .get(download_url)
            .send()
            .await
            .with_context(|| format!("Failed to download model from {}", download_url))?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Model download failed: HTTP {} from {}",
                response.status(),
                download_url
            );
        }

        let total_size = response.content_length();
        if let Some(size) = total_size {
            info!("Model size: {:.1} MB", size as f64 / 1_048_576.0);
        }

        let mut file = tokio::fs::File::create(&tmp_path)
            .await
            .with_context(|| format!("Failed to create temp file: {}", tmp_path.display()))?;

        let mut hasher = Sha256::new();
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading download stream")?;
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;

            // Log progress every 100MB
            if downloaded % (100 * 1024 * 1024) < chunk.len() as u64 {
                if let Some(total) = total_size {
                    info!(
                        "Download progress: {:.0}%",
                        downloaded as f64 / total as f64 * 100.0
                    );
                }
            }
        }

        file.flush().await?;
        drop(file);

        // Verify hash if provided
        let computed_hash = format!("sha256:{}", hex::encode(hasher.finalize()));
        if let Some(expected) = expected_hash {
            if !expected.is_empty() && expected != "sha256:placeholder" && computed_hash != expected {
                let _ = tokio::fs::remove_file(&tmp_path).await;
                anyhow::bail!(
                    "Model hash mismatch for {}: expected {}, got {}",
                    display_name,
                    expected,
                    computed_hash
                );
            }
        }

        // Rename tmp to final
        tokio::fs::rename(&tmp_path, &model_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to rename {} to {}",
                    tmp_path.display(),
                    model_path.display()
                )
            })?;

        info!("Model downloaded: {} ({:.1} MB)", display_name, downloaded as f64 / 1_048_576.0);

        // Register in cache
        {
            let mut entries = self.entries.write().await;
            entries.insert(
                cache_key,
                CachedModel {
                    path: model_path.clone(),
                    size_bytes: downloaded,
                    last_used: SystemTime::now(),
                    url: uri.to_string(),
                },
            );
        }

        Ok(model_path)
    }

    /// Evict least-recently-used models until we're under the size limit
    async fn evict_if_needed(&self, additional_bytes: u64) -> Result<()> {
        let mut entries = self.entries.write().await;

        let total_size: u64 = entries.values().map(|e| e.size_bytes).sum();
        if total_size + additional_bytes <= self.max_cache_bytes {
            return Ok(());
        }

        let target = self.max_cache_bytes.saturating_sub(additional_bytes);
        let mut by_time: Vec<_> = entries
            .iter()
            .map(|(k, v)| (k.clone(), v.last_used, v.size_bytes))
            .collect();
        by_time.sort_by_key(|(_, t, _)| *t);

        let mut current_size = total_size;
        for (key, _, size) in by_time {
            if current_size <= target {
                break;
            }
            if let Some(entry) = entries.remove(&key) {
                let dir = entry.path.parent().unwrap_or(Path::new(""));
                if dir.starts_with(&self.cache_dir) {
                    let _ = std::fs::remove_dir_all(dir);
                    info!("Evicted cached model: {}", entry.url);
                }
                current_size = current_size.saturating_sub(size);
            }
        }

        Ok(())
    }
}

// ============================================================================
// HuggingFace URI resolution
// ============================================================================

/// Resolved download information
struct ResolvedUri {
    /// Direct download URL
    url: String,
    /// Human-readable name for logging
    display_name: String,
    /// Filename to save as
    filename: String,
}

/// Resolve a URI to a direct download URL.
///
/// Supported schemes:
/// - `hf://repo_id` — auto-discover the main model file via HF Hub API
/// - `hf://repo_id:filename` — specific file in the repo
/// - `https://...` — pass through as-is
async fn resolve_uri(uri: &str) -> Result<ResolvedUri> {
    if let Some(hf_ref) = uri.strip_prefix("hf://") {
        resolve_huggingface(hf_ref).await
    } else {
        // Direct URL
        let filename = uri
            .rsplit('/')
            .next()
            .unwrap_or("model.bin")
            .to_string();
        Ok(ResolvedUri {
            url: uri.to_string(),
            display_name: filename.clone(),
            filename,
        })
    }
}

/// Resolve a HuggingFace reference to a download URL.
///
/// Format: `repo_id` or `repo_id:filename`
/// Examples:
///   - `distilbert-base-uncased-finetuned-sst-2-english`
///   - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf`
///   - `runwayml/stable-diffusion-v1-5`
async fn resolve_huggingface(hf_ref: &str) -> Result<ResolvedUri> {
    let (repo_id, explicit_filename) = match hf_ref.split_once(':') {
        Some((repo, file)) => (repo, Some(file)),
        None => (hf_ref, None),
    };

    if let Some(filename) = explicit_filename {
        // Explicit filename — construct direct download URL
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );
        Ok(ResolvedUri {
            display_name: format!("hf://{}/{}", repo_id, filename),
            filename: filename.to_string(),
            url,
        })
    } else {
        // No filename — query the HF Hub API to find the model file
        let api_url = format!("https://huggingface.co/api/models/{}", repo_id);
        info!("Querying HuggingFace Hub: {}", api_url);

        let client = reqwest::Client::new();
        let response = client
            .get(&api_url)
            .header("User-Agent", "archipelag-island/0.4")
            .send()
            .await
            .with_context(|| format!("Failed to query HF Hub for {}", repo_id))?;

        if !response.status().is_success() {
            anyhow::bail!(
                "HuggingFace API returned HTTP {} for repo {}",
                response.status(),
                repo_id
            );
        }

        let body: serde_json::Value = response.json().await
            .context("Failed to parse HF Hub API response")?;

        // Extract siblings (file list) from the API response
        let filename = find_model_filename_from_api(&body, repo_id)?;

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        Ok(ResolvedUri {
            display_name: format!("hf://{}/{}", repo_id, filename),
            filename,
            url,
        })
    }
}

/// Discover the primary model file from a HuggingFace API response.
///
/// Looks for files in priority order:
/// 1. `.gguf` files (GGUF models)
/// 2. `model.onnx` or `*.onnx` (ONNX models)
/// 3. `model.safetensors` or `diffusion_pytorch_model.safetensors` (diffusers/transformers)
/// 4. `pytorch_model.bin` (legacy PyTorch)
fn find_model_filename_from_api(body: &serde_json::Value, repo_id: &str) -> Result<String> {
    let siblings = body
        .get("siblings")
        .and_then(|s| s.as_array())
        .context("HF API response missing 'siblings' file list")?;

    let filenames: Vec<&str> = siblings
        .iter()
        .filter_map(|s| s.get("rfilename").and_then(|f| f.as_str()))
        .collect();

    if filenames.is_empty() {
        anyhow::bail!("HF repo {} has no files", repo_id);
    }

    // Priority 1: GGUF files (prefer Q4_K_M quantization)
    let gguf_files: Vec<&&str> = filenames.iter().filter(|f| f.ends_with(".gguf")).collect();
    if !gguf_files.is_empty() {
        // Prefer Q4_K_M, then Q4_K_S, then any
        for pattern in &["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|f| f.contains(pattern)) {
                return Ok(f.to_string());
            }
        }
        return Ok(gguf_files[0].to_string());
    }

    // Priority 2: ONNX model
    if let Some(f) = filenames.iter().find(|f| **f == "model.onnx") {
        return Ok(f.to_string());
    }
    if let Some(f) = filenames.iter().find(|f| **f == "onnx/model.onnx") {
        return Ok(f.to_string());
    }
    if let Some(f) = filenames.iter().find(|f| f.ends_with(".onnx")) {
        return Ok(f.to_string());
    }

    // Priority 3: Safetensors (diffusers / transformers)
    if let Some(f) = filenames.iter().find(|f| **f == "model.safetensors") {
        return Ok(f.to_string());
    }
    if let Some(f) = filenames.iter().find(|f| **f == "diffusion_pytorch_model.safetensors") {
        return Ok(f.to_string());
    }

    // Priority 4: Legacy PyTorch
    if let Some(f) = filenames.iter().find(|f| **f == "pytorch_model.bin") {
        return Ok(f.to_string());
    }

    // Fallback: first non-config, non-tokenizer file that looks like a model
    let model_extensions = [".bin", ".safetensors", ".onnx", ".gguf", ".pt"];
    if let Some(f) = filenames.iter().find(|f| {
        model_extensions.iter().any(|ext| f.ends_with(ext))
    }) {
        return Ok(f.to_string());
    }

    anyhow::bail!(
        "Could not find a model file in HF repo {}. Files: {:?}",
        repo_id,
        &filenames[..filenames.len().min(10)]
    )
}

// ============================================================================
// Utilities
// ============================================================================

/// Hash a URL to a cache directory key
#[allow(dead_code)]
fn url_to_cache_key(url: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    hex::encode(hasher.finalize())
}

/// Find a model file inside a cache directory
#[allow(dead_code)]
fn find_model_file(dir: &Path) -> Option<PathBuf> {
    if let Ok(read_dir) = std::fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_file() && !path.extension().map_or(false, |e| e == "tmp") {
                return Some(path);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_to_cache_key() {
        let key = url_to_cache_key("https://example.com/model.gguf");
        assert_eq!(key.len(), 64); // SHA256 hex
        assert_eq!(key, url_to_cache_key("https://example.com/model.gguf"));
        assert_ne!(key, url_to_cache_key("https://example.com/other.gguf"));
    }

    #[test]
    fn test_default_config() {
        let config = ModelCacheConfig::default();
        assert_eq!(config.max_cache_gb, 20);
        assert!(config.cache_dir.is_none());
    }

    #[tokio::test]
    async fn test_model_cache_init() {
        let dir = tempfile::tempdir().unwrap();
        let config = ModelCacheConfig {
            max_cache_gb: 1,
            cache_dir: Some(dir.path().to_string_lossy().to_string()),
        };
        let cache = ModelCache::new(&config).unwrap();
        cache.init().await.unwrap();
    }

    #[tokio::test]
    async fn test_resolve_direct_url() {
        let resolved = resolve_uri("https://example.com/models/bert.onnx").await.unwrap();
        assert_eq!(resolved.url, "https://example.com/models/bert.onnx");
        assert_eq!(resolved.filename, "bert.onnx");
    }

    #[tokio::test]
    async fn test_resolve_hf_with_filename() {
        let resolved = resolve_uri("hf://TheBloke/Mistral-7B-GGUF:mistral-7b.Q4_K_M.gguf").await.unwrap();
        assert_eq!(
            resolved.url,
            "https://huggingface.co/TheBloke/Mistral-7B-GGUF/resolve/main/mistral-7b.Q4_K_M.gguf"
        );
        assert_eq!(resolved.filename, "mistral-7b.Q4_K_M.gguf");
    }

    #[test]
    fn test_find_model_filename_gguf() {
        let body = serde_json::json!({
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "config.json"},
                {"rfilename": "mistral-7b.Q4_K_M.gguf"},
                {"rfilename": "mistral-7b.Q8_0.gguf"}
            ]
        });
        let filename = find_model_filename_from_api(&body, "test/repo").unwrap();
        assert_eq!(filename, "mistral-7b.Q4_K_M.gguf");
    }

    #[test]
    fn test_find_model_filename_onnx() {
        let body = serde_json::json!({
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.onnx"}
            ]
        });
        let filename = find_model_filename_from_api(&body, "test/repo").unwrap();
        assert_eq!(filename, "model.onnx");
    }

    #[test]
    fn test_find_model_filename_safetensors() {
        let body = serde_json::json!({
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "model.safetensors"}
            ]
        });
        let filename = find_model_filename_from_api(&body, "test/repo").unwrap();
        assert_eq!(filename, "model.safetensors");
    }

    #[test]
    fn test_find_model_filename_prefers_q4_k_m() {
        let body = serde_json::json!({
            "siblings": [
                {"rfilename": "model.Q8_0.gguf"},
                {"rfilename": "model.Q4_K_M.gguf"},
                {"rfilename": "model.Q5_K_M.gguf"}
            ]
        });
        let filename = find_model_filename_from_api(&body, "test/repo").unwrap();
        assert_eq!(filename, "model.Q4_K_M.gguf");
    }

    #[test]
    fn test_find_model_filename_no_model_files() {
        let body = serde_json::json!({
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "config.json"}
            ]
        });
        assert!(find_model_filename_from_api(&body, "test/repo").is_err());
    }
}
