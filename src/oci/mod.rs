//! OCI container runtime support (Docker-free).
//!
//! Pulls OCI images directly from registries, unpacks layers into a rootfs,
//! and runs containers using a bundled OCI runtime (crun).
//!
//! This enables Linux hosts to run container workloads without Docker installed.

pub mod pull;
pub mod runtime;
pub mod unpack;

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::docker::ContainerOutput;

/// Directory where OCI bundles are stored
const BUNDLES_DIR: &str = "bundles";
/// Directory where pulled image layers are cached
const LAYERS_DIR: &str = "layers";

/// OCI container manager — alternative to Docker for Linux hosts
pub struct OciManager {
    /// Base directory for OCI data (~/.archipelag/oci/)
    base_dir: PathBuf,
    /// Path to the crun binary
    runtime_path: PathBuf,
}

impl OciManager {
    /// Create a new OCI manager, locating the crun binary
    pub fn new(data_dir: &Path) -> Result<Self> {
        let base_dir = data_dir.join("oci");
        std::fs::create_dir_all(base_dir.join(BUNDLES_DIR))?;
        std::fs::create_dir_all(base_dir.join(LAYERS_DIR))?;

        let runtime_path = Self::find_runtime(data_dir)?;
        info!("OCI runtime: {}", runtime_path.display());

        Ok(Self {
            base_dir,
            runtime_path,
        })
    }

    /// Find the crun binary — check bundled location first, then PATH
    fn find_runtime(data_dir: &Path) -> Result<PathBuf> {
        // 1. Check bundled location (~/.archipelag/bin/crun)
        let bundled = data_dir.join("bin").join("crun");
        if bundled.exists() {
            debug!("Using bundled crun: {}", bundled.display());
            return Ok(bundled);
        }

        // 2. Check system PATH
        if let Ok(output) = std::process::Command::new("which")
            .arg("crun")
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    debug!("Using system crun: {}", path);
                    return Ok(PathBuf::from(path));
                }
            }
        }

        // 3. Also accept runc as fallback
        if let Ok(output) = std::process::Command::new("which")
            .arg("runc")
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    warn!("crun not found, falling back to runc: {}", path);
                    return Ok(PathBuf::from(path));
                }
            }
        }

        anyhow::bail!(
            "No OCI runtime found. Install crun or place it at {}",
            bundled.display()
        )
    }

    /// Pull an image and prepare a bundle for execution
    pub async fn prepare_bundle(
        &self,
        image: &str,
        input: &str,
        config: &BundleConfig,
    ) -> Result<PathBuf> {
        let bundle_id = format!("job-{}", uuid::Uuid::new_v4());
        let bundle_dir = self.base_dir.join(BUNDLES_DIR).join(&bundle_id);
        let rootfs_dir = bundle_dir.join("rootfs");
        std::fs::create_dir_all(&rootfs_dir)?;

        // Pull and unpack the image
        let layers_cache = self.base_dir.join(LAYERS_DIR);
        pull::pull_image(image, &layers_cache).await?;
        unpack::unpack_image(&layers_cache, image, &rootfs_dir)?;

        // Write input to a file the container can read
        let input_path = bundle_dir.join("input.json");
        std::fs::write(&input_path, input)?;

        // Generate OCI runtime config
        runtime::generate_config(&bundle_dir, config)?;

        info!("Bundle prepared: {}", bundle_dir.display());
        Ok(bundle_dir)
    }

    /// Run a container from a prepared bundle
    pub async fn run_container(
        &self,
        bundle_dir: &Path,
        timeout_secs: u64,
        output_tx: mpsc::Sender<ContainerOutput>,
    ) -> Result<i64> {
        let container_id = bundle_dir
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let exit_code = runtime::run(
            &self.runtime_path,
            &container_id,
            bundle_dir,
            timeout_secs,
            output_tx,
        )
        .await?;

        // Clean up bundle
        if let Err(e) = std::fs::remove_dir_all(bundle_dir) {
            warn!("Failed to clean up bundle: {}", e);
        }

        Ok(exit_code)
    }

    /// Pull, prepare, and run a container in one step
    pub async fn execute(
        &self,
        image: &str,
        input: &str,
        config: &BundleConfig,
        timeout_secs: u64,
        output_tx: mpsc::Sender<ContainerOutput>,
    ) -> Result<i64> {
        let bundle_dir = self
            .prepare_bundle(image, input, config)
            .await
            .context("Failed to prepare OCI bundle")?;

        self.run_container(&bundle_dir, timeout_secs, output_tx)
            .await
    }

    /// Check if the OCI runtime is available
    pub fn is_available(&self) -> bool {
        self.runtime_path.exists()
            || std::process::Command::new(&self.runtime_path)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }
}

/// Configuration for an OCI container bundle
pub struct BundleConfig {
    pub memory_bytes: Option<i64>,
    pub cpu_quota: Option<i64>,
    pub read_only_rootfs: bool,
    pub network_disabled: bool,
    pub tmpfs_size_mb: u64,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            memory_bytes: Some(1024 * 1024 * 1024), // 1GB
            cpu_quota: None,
            read_only_rootfs: true,
            network_disabled: true,
            tmpfs_size_mb: 256,
        }
    }
}
