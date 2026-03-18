//! Container and model caching for cold-start optimization
//!
//! This module provides functionality to reduce container startup times by:
//! - Pre-pulling container images on startup
//! - Tracking which images are cached locally
//! - Managing a pool of warm (recently-used) workloads
//! - Reporting cache status in heartbeats

use anyhow::{Context, Result};
use bollard::image::ListImagesOptions;
use bollard::Docker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Cache configuration
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct CacheConfig {
    /// Enable pre-pulling of popular images on startup
    #[serde(default)]
    pub enable_preload: bool,

    /// Images to pre-pull on startup
    #[serde(default)]
    pub preload_images: Vec<String>,

    /// Maximum number of images to keep cached
    #[serde(default = "default_max_cached_images")]
    pub max_cached_images: usize,

    /// TTL for warm container tracking (seconds)
    #[serde(default = "default_warm_ttl_seconds")]
    pub warm_ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_preload: false,
            preload_images: Vec::new(),
            max_cached_images: default_max_cached_images(),
            warm_ttl_seconds: default_warm_ttl_seconds(),
        }
    }
}

fn default_max_cached_images() -> usize {
    20
}

fn default_warm_ttl_seconds() -> u64 {
    3600 // 1 hour
}

/// Information about a cached image
#[derive(Debug, Clone, Serialize)]
pub struct CachedImage {
    /// Full image name with tag
    pub image: String,
    /// Image digest (sha256:...)
    pub digest: Option<String>,
    /// Size in bytes
    pub size_bytes: i64,
    /// When the image was last used
    #[serde(skip)]
    pub last_used: Instant,
    /// Number of times this image has been used
    pub use_count: u64,
}

/// Warm container tracking info
#[derive(Debug, Clone, Serialize)]
pub struct WarmWorkload {
    /// Workload ID
    pub workload_id: String,
    /// Container image
    pub image: String,
    /// When this workload was last run
    #[serde(skip)]
    pub last_run: Instant,
    /// Total runs of this workload
    pub run_count: u64,
}

/// Cache statistics for heartbeat reporting
#[derive(Debug, Clone, Serialize, Default)]
pub struct CacheStats {
    /// Number of cached images
    pub cached_image_count: usize,
    /// Total size of cached images in MB
    pub cached_size_mb: u64,
    /// Number of warm workloads (recently used)
    pub warm_workload_count: usize,
    /// List of warm workload IDs (most recent first)
    pub warm_workload_ids: Vec<String>,
}

/// Cache manager for container images and warm workloads
pub struct CacheManager {
    docker: Option<Docker>,
    config: CacheConfig,
    /// Track cached images
    cached_images: Arc<RwLock<HashMap<String, CachedImage>>>,
    /// Track warm (recently-used) workloads
    warm_workloads: Arc<RwLock<HashMap<String, WarmWorkload>>>,
}

#[allow(dead_code)]
impl CacheManager {
    /// Create a new cache manager
    pub fn new(docker: Docker, config: CacheConfig) -> Self {
        Self {
            docker: Some(docker),
            config,
            cached_images: Arc::new(RwLock::new(HashMap::new())),
            warm_workloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a cache manager without Docker (WASM-only mode)
    pub fn new_without_docker(config: CacheConfig) -> Self {
        Self {
            docker: None,
            config,
            cached_images: Arc::new(RwLock::new(HashMap::new())),
            warm_workloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the cache by scanning local Docker images
    pub async fn init(&self) -> Result<()> {
        if self.docker.is_none() {
            info!("Container cache disabled (no Docker)");
            return Ok(());
        }
        info!("Initializing container cache...");

        // Scan existing images
        self.refresh_image_cache().await?;

        // Pre-pull configured images
        if self.config.enable_preload && !self.config.preload_images.is_empty() {
            info!(
                "Pre-pulling {} configured images...",
                self.config.preload_images.len()
            );
            for image in &self.config.preload_images {
                match self.ensure_image(image).await {
                    Ok(_) => info!("Pre-pulled image: {}", image),
                    Err(e) => warn!("Failed to pre-pull image {}: {}", image, e),
                }
            }
        }

        let stats = self.get_stats().await;
        info!(
            "Cache initialized: {} images cached ({} MB)",
            stats.cached_image_count, stats.cached_size_mb
        );

        Ok(())
    }

    /// Refresh the cache of locally available images
    pub async fn refresh_image_cache(&self) -> Result<()> {
        let docker = match &self.docker {
            Some(d) => d,
            None => return Ok(()),
        };

        let options = ListImagesOptions::<String> {
            all: false,
            ..Default::default()
        };

        let images = docker
            .list_images(Some(options))
            .await
            .context("Failed to list Docker images")?;

        let mut cache = self.cached_images.write().await;
        cache.clear();

        for image in images {
            // Get the repo tags as the image names
            for tag in &image.repo_tags {
                if tag != "<none>:<none>" {
                    let cached = CachedImage {
                        image: tag.clone(),
                        digest: Some(image.id.clone()),
                        size_bytes: image.size,
                        last_used: Instant::now(),
                        use_count: 0,
                    };
                    cache.insert(tag.clone(), cached);
                }
            }
        }

        debug!("Refreshed image cache: {} images", cache.len());
        Ok(())
    }

    /// Check if an image is cached locally
    pub async fn is_image_cached(&self, image: &str) -> bool {
        let cache = self.cached_images.read().await;
        cache.contains_key(image)
    }

    /// Get cached image info
    pub async fn get_cached_image(&self, image: &str) -> Option<CachedImage> {
        let cache = self.cached_images.read().await;
        cache.get(image).cloned()
    }

    /// Ensure an image is available locally, pulling if necessary
    pub async fn ensure_image(&self, image: &str) -> Result<bool> {
        let docker = self.docker.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Docker is not available. Cannot pull image: {}", image)
        })?;

        // Check if already cached
        if self.is_image_cached(image).await {
            // Update last_used time
            let mut cache = self.cached_images.write().await;
            if let Some(cached) = cache.get_mut(image) {
                cached.last_used = Instant::now();
                cached.use_count += 1;
            }
            debug!("Image {} already cached", image);
            return Ok(false); // false = didn't need to pull
        }

        // Pull the image
        info!("Pulling image: {}", image);
        let start = Instant::now();

        use bollard::image::CreateImageOptions;
        use futures_util::StreamExt;

        let options = CreateImageOptions {
            from_image: image,
            ..Default::default()
        };

        let mut stream = docker.create_image(Some(options), None, None);

        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        debug!("Pull progress: {}", status);
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Failed to pull image {}: {}", image, e));
                }
            }
        }

        let elapsed = start.elapsed();
        info!("Pulled image {} in {:.2}s", image, elapsed.as_secs_f64());

        // Refresh cache to pick up the new image
        self.refresh_image_cache().await?;

        Ok(true) // true = needed to pull
    }

    /// Record that a workload was just executed
    pub async fn record_workload_run(&self, workload_id: &str, image: &str) {
        let mut warm = self.warm_workloads.write().await;

        if let Some(entry) = warm.get_mut(workload_id) {
            entry.last_run = Instant::now();
            entry.run_count += 1;
        } else {
            warm.insert(
                workload_id.to_string(),
                WarmWorkload {
                    workload_id: workload_id.to_string(),
                    image: image.to_string(),
                    last_run: Instant::now(),
                    run_count: 1,
                },
            );
        }

        // Also update image cache
        let mut cache = self.cached_images.write().await;
        if let Some(cached) = cache.get_mut(image) {
            cached.last_used = Instant::now();
            cached.use_count += 1;
        }
    }

    /// Check if a workload is "warm" (recently used)
    pub async fn is_workload_warm(&self, workload_id: &str) -> bool {
        let warm = self.warm_workloads.read().await;
        if let Some(entry) = warm.get(workload_id) {
            let ttl = Duration::from_secs(self.config.warm_ttl_seconds);
            entry.last_run.elapsed() < ttl
        } else {
            false
        }
    }

    /// Get list of warm workload IDs
    pub async fn get_warm_workloads(&self) -> Vec<String> {
        let warm = self.warm_workloads.read().await;
        let ttl = Duration::from_secs(self.config.warm_ttl_seconds);

        let mut warm_list: Vec<_> = warm
            .values()
            .filter(|w| w.last_run.elapsed() < ttl)
            .collect();

        // Sort by most recent first
        warm_list.sort_by(|a, b| a.last_run.elapsed().cmp(&b.last_run.elapsed()));

        warm_list.iter().map(|w| w.workload_id.clone()).collect()
    }

    /// Clean up stale cache entries
    pub async fn cleanup_stale(&self) {
        let ttl = Duration::from_secs(self.config.warm_ttl_seconds);

        // Clean up stale warm workloads
        let mut warm = self.warm_workloads.write().await;
        warm.retain(|_, w| w.last_run.elapsed() < ttl);

        debug!("Cleaned up warm workloads, {} remaining", warm.len());
    }

    /// Evict least-recently-used images when max_cached_images is exceeded.
    ///
    /// Removes entries from the tracking map only. Docker image pruning is a
    /// separate concern (unsafe to do automatically during job execution).
    pub async fn evict_lru(&self) -> usize {
        let stats = self.get_stats().await;
        if stats.cached_image_count <= self.config.max_cached_images {
            return 0;
        }
        let excess = stats.cached_image_count - self.config.max_cached_images;
        let mut cache = self.cached_images.write().await;
        // Sort by last_used ascending (oldest first), remove oldest
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.last_used))
            .collect();
        entries.sort_by_key(|(_, t)| *t);
        let to_remove: Vec<String> = entries.into_iter().take(excess).map(|(k, _)| k).collect();
        for key in &to_remove {
            cache.remove(key);
        }
        to_remove.len()
    }

    /// Get cache statistics for heartbeat reporting
    pub async fn get_stats(&self) -> CacheStats {
        let cache = self.cached_images.read().await;
        let warm = self.warm_workloads.read().await;

        let total_size_bytes: i64 = cache.values().map(|c| c.size_bytes).sum();
        let ttl = Duration::from_secs(self.config.warm_ttl_seconds);

        let mut warm_list: Vec<_> = warm
            .values()
            .filter(|w| w.last_run.elapsed() < ttl)
            .collect();
        warm_list.sort_by(|a, b| a.last_run.elapsed().cmp(&b.last_run.elapsed()));

        CacheStats {
            cached_image_count: cache.len(),
            cached_size_mb: (total_size_bytes / (1024 * 1024)) as u64,
            warm_workload_count: warm_list.len(),
            warm_workload_ids: warm_list.iter().map(|w| w.workload_id.clone()).collect(),
        }
    }

    /// Check if this Island should be preferred for a workload based on cache state
    pub async fn get_warmth_score(&self, workload_id: &str, image: &str) -> u32 {
        let mut score = 0;

        // Check if workload was recently run (+50 points)
        if self.is_workload_warm(workload_id).await {
            score += 50;
        }

        // Check if image is cached (+30 points)
        if self.is_image_cached(image).await {
            score += 30;

            // Additional points based on use count
            if let Some(cached) = self.get_cached_image(image).await {
                score += cached.use_count.min(20) as u32; // Up to 20 more points
            }
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert!(!config.enable_preload);
        assert!(config.preload_images.is_empty());
        assert_eq!(config.max_cached_images, 20);
        assert_eq!(config.warm_ttl_seconds, 3600);
    }

    #[test]
    fn test_cache_config_deserialization() {
        let json = r#"{
            "enable_preload": true,
            "preload_images": ["image1:latest", "image2:v1"],
            "max_cached_images": 50,
            "warm_ttl_seconds": 7200
        }"#;
        let config: CacheConfig = serde_json::from_str(json).unwrap();
        assert!(config.enable_preload);
        assert_eq!(config.preload_images.len(), 2);
        assert_eq!(config.max_cached_images, 50);
        assert_eq!(config.warm_ttl_seconds, 7200);
    }

    #[test]
    fn test_cache_config_partial_deserialization() {
        // Only specifying some fields should use defaults for the rest
        let json = r#"{"enable_preload": true}"#;
        let config: CacheConfig = serde_json::from_str(json).unwrap();
        assert!(config.enable_preload);
        assert!(config.preload_images.is_empty()); // default
        assert_eq!(config.max_cached_images, 20); // default
        assert_eq!(config.warm_ttl_seconds, 3600); // default
    }

    #[test]
    fn test_cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.cached_image_count, 0);
        assert_eq!(stats.cached_size_mb, 0);
        assert_eq!(stats.warm_workload_count, 0);
        assert!(stats.warm_workload_ids.is_empty());
    }

    #[test]
    fn test_cache_stats_serialization() {
        let stats = CacheStats {
            cached_image_count: 5,
            cached_size_mb: 2048,
            warm_workload_count: 2,
            warm_workload_ids: vec!["wl-1".to_string(), "wl-2".to_string()],
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"cached_image_count\":5"));
        assert!(json.contains("\"cached_size_mb\":2048"));
    }

    #[test]
    fn test_cached_image_serialization() {
        let image = CachedImage {
            image: "nginx:latest".to_string(),
            digest: Some("sha256:abc123".to_string()),
            size_bytes: 1073741824, // 1GB
            last_used: Instant::now(),
            use_count: 42,
        };
        let json = serde_json::to_string(&image).unwrap();
        assert!(json.contains("\"image\":\"nginx:latest\""));
        assert!(json.contains("\"use_count\":42"));
        // last_used should be skipped (has #[serde(skip)])
        assert!(!json.contains("last_used"));
    }

    #[test]
    fn test_warm_workload_serialization() {
        let wl = WarmWorkload {
            workload_id: "wl-123".to_string(),
            image: "myimage:v1".to_string(),
            last_run: Instant::now(),
            run_count: 10,
        };
        let json = serde_json::to_string(&wl).unwrap();
        assert!(json.contains("\"workload_id\":\"wl-123\""));
        assert!(json.contains("\"run_count\":10"));
        // last_run should be skipped
        assert!(!json.contains("last_run"));
    }
}
