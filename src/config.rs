//! Configuration loading and management

use anyhow::{Context, Result};
use config::{Config, File};
use serde::Deserialize;

/// Agent configuration
#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    /// Host ID (generated on first run if not set)
    pub host_id: Option<String>,

    /// Coordinator settings
    pub coordinator: CoordinatorConfig,

    /// Docker settings
    pub docker: DockerConfig,

    /// Workload settings
    pub workload: WorkloadConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CoordinatorConfig {
    /// NATS server URL
    pub nats_url: String,
}

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
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            host_id: None,
            coordinator: CoordinatorConfig {
                nats_url: "nats://localhost:4222".to_string(),
            },
            docker: DockerConfig { socket: None },
            workload: WorkloadConfig {
                llm_chat_image: "llm-chat:latest".to_string(),
                gpu_devices: Some(vec!["0".to_string()]),
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
