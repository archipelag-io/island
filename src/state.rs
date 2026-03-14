//! Local state management for the Island software
//!
//! Handles:
//! - Pairing state persistence (whether this Island is paired)
//! - WASM module caching (download and cache modules)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Default state directory (relative to home)
const STATE_DIR: &str = ".archipelag";
/// State file name
const STATE_FILE: &str = "state.json";
/// WASM cache directory name
const WASM_CACHE_DIR: &str = "wasm-cache";

/// Persistent state for the Island software
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct AgentState {
    /// Whether this Island has been paired to an account
    pub paired: bool,
    /// Account ID this Island is paired to (if any)
    pub account_id: Option<String>,
    /// Timestamp when pairing occurred
    pub paired_at: Option<String>,
}

/// State manager for the Island software
pub struct StateManager {
    state_dir: PathBuf,
    state: AgentState,
}

impl StateManager {
    /// Create a new state manager, loading existing state if available
    pub async fn new() -> Result<Self> {
        let state_dir = get_state_dir()?;

        // Ensure directories exist
        fs::create_dir_all(&state_dir)
            .await
            .context("Failed to create state directory")?;

        let cache_dir = state_dir.join(WASM_CACHE_DIR);
        fs::create_dir_all(&cache_dir)
            .await
            .context("Failed to create WASM cache directory")?;

        // Load existing state or use default
        let state_file = state_dir.join(STATE_FILE);
        let state = if state_file.exists() {
            match fs::read_to_string(&state_file).await {
                Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
                Err(e) => {
                    warn!("Failed to read state file: {}", e);
                    AgentState::default()
                }
            }
        } else {
            AgentState::default()
        };

        debug!("State loaded: paired={}", state.paired);

        Ok(Self { state_dir, state })
    }

    /// Check if this Island is already paired
    pub fn is_paired(&self) -> bool {
        self.paired
    }

    /// Mark this Island as paired
    pub async fn set_paired(&mut self, account_id: Option<String>) -> Result<()> {
        self.state.paired = true;
        self.state.account_id = account_id;
        self.state.paired_at = Some(chrono_now());
        self.save().await
    }

    /// Save state to disk
    async fn save(&self) -> Result<()> {
        let state_file = self.state_dir.join(STATE_FILE);
        let contents =
            serde_json::to_string_pretty(&self.state).context("Failed to serialize state")?;
        fs::write(&state_file, contents)
            .await
            .context("Failed to write state file")?;
        debug!("State saved to {:?}", state_file);
        Ok(())
    }

    /// Get the path to a cached WASM module, downloading if necessary
    ///
    /// If the module is already cached and the hash matches, returns the cached path.
    /// Otherwise, downloads the module and caches it.
    pub async fn get_wasm_module(&self, url: &str, expected_hash: Option<&str>) -> Result<PathBuf> {
        let cache_dir = self.state_dir.join(WASM_CACHE_DIR);

        // Generate cache filename from URL hash
        let url_hash = hash_string(url);
        let cache_name = format!("{}.wasm", &url_hash[..16]);
        let cache_path = cache_dir.join(&cache_name);

        // Check if cached version exists and is valid
        if cache_path.exists() {
            if let Some(hash) = expected_hash {
                // Verify cached file hash
                match verify_file_hash(&cache_path, hash).await {
                    Ok(true) => {
                        info!("Using cached WASM module: {}", cache_name);
                        return Ok(cache_path);
                    }
                    Ok(false) => {
                        info!("Cached WASM hash mismatch, re-downloading");
                    }
                    Err(e) => {
                        warn!("Failed to verify cached WASM: {}", e);
                    }
                }
            } else {
                // No hash to verify, use cached version
                info!(
                    "Using cached WASM module (no hash verification): {}",
                    cache_name
                );
                return Ok(cache_path);
            }
        }

        // Download the module
        info!("Downloading WASM module from: {}", url);
        let bytes = download_file(url).await?;

        // Verify hash if provided
        if let Some(hash) = expected_hash {
            let actual_hash = hash_bytes(&bytes);
            let expected = normalize_hash(hash);
            if actual_hash != expected {
                anyhow::bail!(
                    "Downloaded WASM hash mismatch: expected {}, got {}",
                    expected,
                    actual_hash
                );
            }
            info!(
                "WASM hash verified: {}...{}",
                &actual_hash[..8],
                &actual_hash[56..]
            );
        }

        // Save to cache
        let mut file = fs::File::create(&cache_path)
            .await
            .context("Failed to create cache file")?;
        file.write_all(&bytes)
            .await
            .context("Failed to write cache file")?;
        file.flush().await?;

        info!("WASM module cached: {}", cache_name);
        Ok(cache_path)
    }
}

// Expose paired state via Deref for convenience
impl std::ops::Deref for StateManager {
    type Target = AgentState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

/// Get the state directory path
fn get_state_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(STATE_DIR))
}

/// Hash a string using SHA256 and return hex
fn hash_string(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex::encode(hasher.finalize())
}

/// Hash bytes using SHA256 and return hex
fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Normalize a hash (remove sha256: prefix if present, lowercase)
fn normalize_hash(hash: &str) -> String {
    let h = hash.strip_prefix("sha256:").unwrap_or(hash);
    h.to_lowercase()
}

/// Verify a file's SHA256 hash
async fn verify_file_hash(path: &Path, expected_hash: &str) -> Result<bool> {
    let bytes = fs::read(path).await?;
    let actual = hash_bytes(&bytes);
    let expected = normalize_hash(expected_hash);
    Ok(actual == expected)
}

/// Download a file from a URL
async fn download_file(url: &str) -> Result<Vec<u8>> {
    let response = reqwest::get(url).await.context("Failed to download file")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let bytes = response
        .bytes()
        .await
        .context("Failed to read response body")?;

    info!("Downloaded {} bytes", bytes.len());
    Ok(bytes.to_vec())
}

/// Get current timestamp as ISO string (simple implementation without chrono)
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_string() {
        let hash = hash_string("test");
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_normalize_hash() {
        assert_eq!(normalize_hash("sha256:ABC123"), "abc123");
        assert_eq!(normalize_hash("ABC123"), "abc123");
    }

    #[test]
    fn test_hash_string_deterministic() {
        let hash1 = hash_string("hello");
        let hash2 = hash_string("hello");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_string_different_inputs() {
        let hash1 = hash_string("hello");
        let hash2 = hash_string("world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_bytes() {
        let hash = hash_bytes(b"test data");
        assert_eq!(hash.len(), 64); // SHA256 hex = 64 chars
    }

    #[test]
    fn test_hash_bytes_matches_known_sha256() {
        // SHA256 of empty string
        let hash = hash_bytes(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_normalize_hash_preserves_lowercase() {
        assert_eq!(normalize_hash("abc123"), "abc123");
    }

    #[test]
    fn test_normalize_hash_strips_prefix_and_lowercases() {
        assert_eq!(normalize_hash("sha256:DEADBEEF"), "deadbeef");
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert!(!state.paired);
        assert!(state.account_id.is_none());
        assert!(state.paired_at.is_none());
    }

    #[test]
    fn test_agent_state_serialization() {
        let state = AgentState {
            paired: true,
            account_id: Some("user-123".to_string()),
            paired_at: Some("1234567890".to_string()),
        };
        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("\"paired\":true"));
        assert!(json.contains("\"account_id\":\"user-123\""));

        // Roundtrip
        let deserialized: AgentState = serde_json::from_str(&json).unwrap();
        assert!(deserialized.paired);
        assert_eq!(deserialized.account_id, Some("user-123".to_string()));
    }

    #[tokio::test]
    async fn test_verify_file_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wasm");
        std::fs::write(&path, b"test content").unwrap();

        let expected_hash = hash_bytes(b"test content");
        assert!(verify_file_hash(&path, &expected_hash).await.unwrap());

        // Wrong hash should return false
        assert!(!verify_file_hash(&path, "sha256:wrong").await.unwrap());
    }

    #[tokio::test]
    async fn test_verify_file_hash_with_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wasm");
        std::fs::write(&path, b"hello").unwrap();

        let expected_hash = format!("sha256:{}", hash_bytes(b"hello"));
        assert!(verify_file_hash(&path, &expected_hash).await.unwrap());
    }

    #[test]
    fn test_chrono_now_returns_numeric() {
        let now = chrono_now();
        assert!(now.parse::<u64>().is_ok(), "Should be a numeric timestamp");
    }
}
