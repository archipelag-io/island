//! Configuration loading and management

use anyhow::{Context, Result};
use config::{Config, File};
use serde::Deserialize;

pub use crate::cache::CacheConfig;
pub use crate::security::SigningConfig;

/// Agent configuration
#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    /// Island ID (generated on first run if not set)
    pub host_id: Option<String>,

    /// Island settings
    #[serde(default)]
    pub host: HostConfig,

    /// Coordinator settings
    pub coordinator: CoordinatorConfig,

    /// Docker settings (reserved for future custom socket config)
    #[allow(dead_code)]
    #[serde(default)]
    pub docker: DockerConfig,

    /// Workload settings
    #[serde(default)]
    pub workload: WorkloadConfig,

    /// Cache settings for cold-start optimization
    #[serde(default)]
    pub cache: CacheConfig,

    /// Signature verification settings
    #[serde(default)]
    pub signing: SigningConfig,

    /// Registry allowlist settings
    #[serde(default)]
    pub registry: RegistryConfig,

    /// Model cache settings (for GGUF, ONNX, diffusers models)
    #[serde(default)]
    pub model_cache: ModelCacheConfig,

    /// Model preloading settings
    #[serde(default)]
    pub preload: PreloadConfig,
}

/// Model preloading configuration
#[derive(Debug, Deserialize, Clone)]
pub struct PreloadConfig {
    /// Enable automatic model preloading at startup (default: true)
    #[serde(default = "default_preload_enabled")]
    pub enabled: bool,

    /// Explicit list of model URIs to preload (overrides auto-selection).
    /// If empty, auto-selects based on hardware capabilities.
    /// Example: ["hf://Qwen/Qwen3.5-0.8B-GGUF", "hf://openai/whisper-base"]
    #[serde(default)]
    pub models: Vec<String>,
}

fn default_preload_enabled() -> bool {
    true
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            models: Vec::new(),
        }
    }
}

/// Model cache configuration for downloaded ML models
#[derive(Debug, Deserialize, Clone)]
pub struct ModelCacheConfig {
    /// Maximum cache size in GB (default: 20)
    #[serde(default = "default_max_model_cache_gb")]
    pub max_cache_gb: u64,

    /// Cache directory (default: ~/.island/model-cache)
    #[serde(default)]
    pub cache_dir: Option<String>,

    /// HuggingFace API token for gated models (optional).
    /// Set via config or HF_TOKEN environment variable.
    #[serde(default)]
    pub hf_token: Option<String>,
}

fn default_max_model_cache_gb() -> u64 {
    20
}

impl Default for ModelCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_gb: default_max_model_cache_gb(),
            cache_dir: None,
            hf_token: None,
        }
    }
}

/// Registry allowlist configuration
#[derive(Debug, Deserialize, Clone)]
pub struct RegistryConfig {
    /// Enable registry allowlist enforcement (default: true)
    #[serde(default = "default_registry_enabled")]
    pub enabled: bool,

    /// Allowed registry prefixes (e.g., "ghcr.io/archipelag-io")
    /// If empty, uses built-in defaults
    #[serde(default)]
    pub allowed: Vec<String>,

    /// Require images to have a pinned digest (sha256:...)
    #[serde(default)]
    pub require_digest: bool,
}

fn default_registry_enabled() -> bool {
    true
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed: vec![],
            require_digest: false,
        }
    }
}

/// Island configuration
#[derive(Debug, Deserialize, Clone, Default)]
pub struct HostConfig {
    /// Geographic region (e.g., "us-west-2", "eu-central-1")
    pub region: Option<String>,

    /// Human-readable name for this Island
    #[allow(dead_code)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CoordinatorConfig {
    /// NATS server URL
    pub nats_url: String,
}

/// Docker configuration (reserved for future use)
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone, Default)]
pub struct DockerConfig {
    /// Docker socket path (default: unix:///var/run/docker.sock)
    pub socket: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WorkloadConfig {
    /// Default container image for LLM chat
    #[serde(default = "default_llm_chat_image")]
    pub llm_chat_image: String,

    /// GPU device IDs to use (e.g., ["0"] or ["0", "1"])
    pub gpu_devices: Option<Vec<String>>,

    /// Resource limits for container workloads
    #[serde(default)]
    pub resource_limits: ResourceLimits,
}

/// Resource limits for container workloads
#[derive(Debug, Deserialize, Clone)]
pub struct ResourceLimits {
    /// Memory limit in MB (default: 8192 = 8GB)
    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,

    /// Enable read-only root filesystem (default: true)
    #[serde(default = "default_read_only_rootfs")]
    pub read_only_rootfs: bool,

    /// Size of tmpfs mount at /tmp in MB (default: 256)
    /// Only used when read_only_rootfs is true
    #[serde(default = "default_tmpfs_size_mb")]
    pub tmpfs_size_mb: u64,

    /// CPU quota as percentage (e.g., 200 = 2 cores, 50 = half core)
    /// None = no limit
    pub cpu_percent: Option<u64>,

    /// Disable network access for containers (default: true)
    /// When true, containers run with network_mode: "none"
    #[serde(default = "default_network_disabled")]
    pub network_disabled: bool,
}

fn default_memory_mb() -> u64 {
    8192 // 8GB
}

fn default_read_only_rootfs() -> bool {
    true
}

fn default_tmpfs_size_mb() -> u64 {
    256
}

fn default_network_disabled() -> bool {
    true
}

fn default_llm_chat_image() -> String {
    "ghcr.io/archipelag-io/llm-chat:latest".to_string()
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            llm_chat_image: default_llm_chat_image(),
            gpu_devices: None,
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_mb: default_memory_mb(),
            read_only_rootfs: default_read_only_rootfs(),
            tmpfs_size_mb: default_tmpfs_size_mb(),
            cpu_percent: None,
            network_disabled: default_network_disabled(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            host_id: None,
            host: HostConfig::default(),
            coordinator: CoordinatorConfig {
                nats_url: "nats://localhost:4222".to_string(),
            },
            docker: DockerConfig { socket: None },
            workload: WorkloadConfig {
                // Use mock image by default for development
                llm_chat_image: "archipelag-llm-chat-mock:latest".to_string(),
                // No GPU needed for mock
                gpu_devices: None,
                resource_limits: ResourceLimits::default(),
            },
            cache: CacheConfig::default(),
            signing: SigningConfig::default(),
            registry: RegistryConfig::default(),
            model_cache: ModelCacheConfig::default(),
            preload: PreloadConfig::default(),
        }
    }
}

/// Load configuration from file
pub fn load(path: &str) -> Result<AgentConfig> {
    // Check if config file exists; if not, use defaults
    if !std::path::Path::new(path).exists() {
        tracing::warn!("Config file not found at {}, using defaults", path);
        return Ok(AgentConfig::default());
    }

    let config = Config::builder()
        .add_source(File::with_name(path).required(true))
        .build()
        .context("Failed to build configuration")?;

    config
        .try_deserialize()
        .context("Failed to parse configuration file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AgentConfig::default();
        assert!(config.host_id.is_none());
        assert_eq!(config.coordinator.nats_url, "nats://localhost:4222");
        assert!(config.docker.socket.is_none());
        assert_eq!(
            config.workload.llm_chat_image,
            "archipelag-llm-chat-mock:latest"
        );
        assert!(config.workload.gpu_devices.is_none());
    }

    #[test]
    fn test_default_resource_limits() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.memory_mb, 8192);
        assert!(limits.read_only_rootfs);
        assert_eq!(limits.tmpfs_size_mb, 256);
        assert!(limits.cpu_percent.is_none());
        assert!(limits.network_disabled);
    }

    #[test]
    fn test_load_nonexistent_file_returns_defaults() {
        let config = load("/nonexistent/path/config").unwrap();
        assert_eq!(config.coordinator.nats_url, "nats://localhost:4222");
    }

    #[test]
    fn test_host_config_defaults() {
        let host = HostConfig::default();
        assert!(host.region.is_none());
        assert!(host.name.is_none());
    }
}
