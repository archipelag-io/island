//! NATS client for coordinator communication
//!
//! Handles connection to NATS, job subscriptions, and message publishing.

use anyhow::{Context, Result};
use async_nats::{Client, ConnectOptions, Message, Subscriber};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

/// NATS subject patterns
pub mod subjects {
    pub fn jobs(host_id: &str) -> String {
        format!("host.{}.jobs", host_id)
    }

    pub fn status(host_id: &str) -> String {
        format!("host.{}.status", host_id)
    }

    pub fn output(host_id: &str) -> String {
        format!("host.{}.output", host_id)
    }

    pub fn heartbeat(host_id: &str) -> String {
        format!("host.{}.heartbeat", host_id)
    }

    pub const REGISTRATION: &str = "coordinator.hosts.register";
}

/// Host capabilities reported during registration
#[derive(Debug, Clone, Serialize)]
pub struct HostCapabilities {
    pub gpu_model: Option<String>,
    pub gpu_vram_mb: Option<u32>,
    pub cpu_cores: u32,
    pub ram_mb: u32,
    pub region: Option<String>,
}

/// Host registration message
#[derive(Debug, Serialize)]
pub struct RegisterHost {
    pub host_id: String,
    pub capabilities: HostCapabilities,
    pub version: String,
}

/// Heartbeat message
#[derive(Debug, Serialize)]
pub struct Heartbeat {
    pub host_id: String,
    pub status: String,
    pub active_jobs: u32,
    pub timestamp: i64,
}

/// Job assignment from coordinator
#[derive(Debug, Deserialize)]
pub struct AssignJob {
    pub job_id: String,
    pub workload_id: String,
    pub input: serde_json::Value,
    pub lease_expires: i64,
    /// Runtime type: "container" or "wasm"
    #[serde(default = "default_runtime_type")]
    pub runtime_type: String,
    /// For container workloads
    pub container_image: Option<String>,
    /// For WASM workloads
    pub wasm_url: Option<String>,
    pub wasm_hash: Option<String>,
}

fn default_runtime_type() -> String {
    "container".to_string()
}

/// Job status update
#[derive(Debug, Serialize)]
pub struct JobStatus {
    pub job_id: String,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub timestamp: i64,
}

/// Job output chunk
#[derive(Debug, Serialize)]
pub struct JobOutput {
    pub job_id: String,
    pub seq: u64,
    pub chunk: String,
    pub is_final: bool,
}

/// NATS connection wrapper with agent-specific functionality
#[derive(Clone)]
pub struct NatsAgent {
    client: Client,
    host_id: String,
}

impl NatsAgent {
    /// Connect to NATS server
    pub async fn connect(nats_url: &str, host_id: String) -> Result<Self> {
        let options = ConnectOptions::new()
            .name(&format!("archipelag-agent-{}", &host_id[..8]))
            .retry_on_initial_connect()
            .connection_timeout(Duration::from_secs(10))
            .ping_interval(Duration::from_secs(10))
            .max_reconnects(None); // Reconnect forever

        let client = options
            .connect(nats_url)
            .await
            .context("Failed to connect to NATS")?;

        info!("Connected to NATS at {}", nats_url);

        Ok(Self { client, host_id })
    }

    /// Register this host with the coordinator
    pub async fn register(&self, capabilities: HostCapabilities) -> Result<()> {
        let msg = RegisterHost {
            host_id: self.host_id.clone(),
            capabilities,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize registration")?;

        self.client
            .publish(subjects::REGISTRATION, payload.into())
            .await
            .context("Failed to publish registration")?;

        info!("Registered host {} with coordinator", self.host_id);
        Ok(())
    }

    /// Subscribe to job assignments for this host
    pub async fn subscribe_jobs(&self) -> Result<Subscriber> {
        let subject = subjects::jobs(&self.host_id);
        let subscriber = self
            .client
            .subscribe(subject.clone())
            .await
            .context("Failed to subscribe to jobs")?;

        info!("Subscribed to job assignments on {}", subject);
        Ok(subscriber)
    }

    /// Send heartbeat
    pub async fn send_heartbeat(&self, active_jobs: u32) -> Result<()> {
        let msg = Heartbeat {
            host_id: self.host_id.clone(),
            status: "online".to_string(),
            active_jobs,
            timestamp: chrono_timestamp(),
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize heartbeat")?;

        self.client
            .publish(subjects::heartbeat(&self.host_id), payload.into())
            .await
            .context("Failed to publish heartbeat")?;

        debug!("Sent heartbeat");
        Ok(())
    }

    /// Publish job status update
    pub async fn publish_status(&self, job_id: &str, state: &str, error: Option<String>) -> Result<()> {
        let msg = JobStatus {
            job_id: job_id.to_string(),
            state: state.to_string(),
            error,
            timestamp: chrono_timestamp(),
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize status")?;

        self.client
            .publish(subjects::status(&self.host_id), payload.into())
            .await
            .context("Failed to publish status")?;

        debug!("Published status: job={} state={}", job_id, state);
        Ok(())
    }

    /// Publish job output chunk
    pub async fn publish_output(&self, job_id: &str, seq: u64, chunk: &str, is_final: bool) -> Result<()> {
        let msg = JobOutput {
            job_id: job_id.to_string(),
            seq,
            chunk: chunk.to_string(),
            is_final,
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize output")?;

        self.client
            .publish(subjects::output(&self.host_id), payload.into())
            .await
            .context("Failed to publish output")?;

        Ok(())
    }

    /// Get the host ID
    pub fn host_id(&self) -> &str {
        &self.host_id
    }
}

/// Parse a job assignment message
pub fn parse_job_assignment(msg: &Message) -> Result<AssignJob> {
    serde_json::from_slice(&msg.payload).context("Failed to parse job assignment")
}

/// Get current timestamp in milliseconds
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
