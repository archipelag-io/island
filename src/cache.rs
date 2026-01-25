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
#[derive(Debug, Deserialize, Clone, Default)]
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
    docker: Docker,
    config: CacheConfig,
    /// Track cached images
    cached_images: Arc<RwLock<HashMap<String, CachedImage>>>,
    /// Track warm (recently-used) workloads
    warm_workloads: Arc<RwLock<HashMap<String, WarmWorkload>>>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(docker: Docker, config: CacheConfig) -> Self {
        Self {
            docker,
            config,
            cached_images: Arc::new(RwLock::new(HashMap::new())),
            warm_workloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the cache by scanning local Docker images
    pub async fn init(&self) -> Result<()> {
        info!("Initializing container cache...");

        // Scan existing images
        self.refresh_image_cache().await?;

        // Pre-pull configured images
        if self.config.enable_preload && !self.config.preload_images.is_empty() {
            info!("Pre-pulling {} configured images...", self.config.preload_images.len());
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
        let options = ListImagesOptions::<String> {
            all: false,
            ..Default::default()
        };

        let images = self
            .docker
            .list_images(Some(options))
            .await
            .context("Failed to list Docker images")?;

        let mut cache = self.cached_images.write().await;
        cache.clear();

        for image in images {
            // Get the first repo tag as the image name
            if let Some(repo_tags) = &image.repo_tags {
                for tag in repo_tags {
                    if tag != "<none>:<none>" {
                        let cached = CachedImage {
                            image: tag.clone(),
                            digest: image.id.clone(),
                            size_bytes: image.size,
                            last_used: Instant::now(),
                            use_count: 0,
                        };
                        cache.insert(tag.clone(), cached);
                    }
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

        let mut stream = self.docker.create_image(Some(options), None, None);

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

    /// Check if this host should be preferred for a workload based on cache state
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
                score += (cached.use_count.min(20) as u32) * 1; // Up to 20 more points
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
}
