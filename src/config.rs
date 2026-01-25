//! Configuration loading and management

use anyhow::{Context, Result};
use config::{Config, File};
use serde::Deserialize;

/// Agent configuration
#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    /// Host ID (generated on first run if not set)
    pub host_id: Option<String>,

    /// Host settings
    #[serde(default)]
    pub host: HostConfig,

    /// Coordinator settings
    pub coordinator: CoordinatorConfig,

    /// Docker settings (reserved for future custom socket config)
    #[allow(dead_code)]
    pub docker: DockerConfig,

    /// Workload settings
    pub workload: WorkloadConfig,
}

/// Host configuration
#[derive(Debug, Deserialize, Clone, Default)]
pub struct HostConfig {
    /// Geographic region (e.g., "us-west-2", "eu-central-1")
    pub region: Option<String>,

    /// Human-readable name for this host
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
#[derive(Debug, Deserialize, Clone)]
pub struct DockerConfig {
    /// Docker socket path (default: unix:///var/run/docker.sock)
    pub socket: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WorkloadConfig {
    /// Default container image for LLM chat
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

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_mb: default_memory_mb(),
            read_only_rootfs: default_read_only_rootfs(),
            tmpfs_size_mb: default_tmpfs_size_mb(),
            cpu_percent: None,
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
        }
    }
}

/// Load configuration from file
pub fn load(path: &str) -> Result<AgentConfig> {
    let config = Config::builder()
        .add_source(File::with_name(path).required(false))
        .build()
        .context("Failed to build configuration")?;

    // If no config file exists, use defaults
    config
        .try_deserialize()
        .or_else(|_| Ok(AgentConfig::default()))
}
