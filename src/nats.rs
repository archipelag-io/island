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

    pub fn cancel(host_id: &str) -> String {
        format!("host.{}.cancel", host_id)
    }

    pub fn lease(host_id: &str) -> String {
        format!("host.{}.lease", host_id)
    }

    pub const REGISTRATION: &str = "coordinator.hosts.register";
    pub const PAIRING: &str = "coordinator.hosts.pairing";
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

/// Heartbeat message (basic)
#[derive(Debug, Serialize)]
pub struct Heartbeat {
    pub host_id: String,
    pub status: String,
    pub active_jobs: u32,
    pub timestamp: i64,
}

/// Enhanced heartbeat message with detailed metrics
#[derive(Debug, Serialize)]
pub struct EnhancedHeartbeat {
    pub host_id: String,
    pub status: String,
    pub active_jobs: u32,
    pub timestamp: i64,
    pub agent_version: String,
    /// System-wide metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemMetricsSnapshot>,
    /// GPU metrics (one per GPU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<GpuMetricsSnapshot>>,
    /// Metrics for currently active jobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_job_metrics: Option<Vec<ActiveJobMetrics>>,
    /// Cache statistics for cold-start optimization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMetricsSnapshot>,
}

/// Cache metrics snapshot for heartbeat
#[derive(Debug, Serialize)]
pub struct CacheMetricsSnapshot {
    /// Number of cached container images
    pub cached_image_count: usize,
    /// Total size of cached images in MB
    pub cached_size_mb: u64,
    /// Number of warm workloads (recently used)
    pub warm_workload_count: usize,
    /// List of warm workload IDs
    pub warm_workload_ids: Vec<String>,
}

/// System metrics snapshot for heartbeat
#[derive(Debug, Serialize)]
pub struct SystemMetricsSnapshot {
    pub cpu_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub disk_used_gb: u64,
    pub disk_total_gb: u64,
}

/// GPU metrics snapshot for heartbeat
#[derive(Debug, Serialize)]
pub struct GpuMetricsSnapshot {
    pub index: u32,
    pub utilization_percent: u32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_c: u32,
    pub power_draw_w: f32,
}

/// Active job metrics for heartbeat
#[derive(Debug, Serialize)]
pub struct ActiveJobMetrics {
    pub job_id: String,
    pub job_type: String,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_generated: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_mb: Option<u64>,
}

/// Job assignment from coordinator
#[derive(Debug, Deserialize, Clone)]
pub struct AssignJob {
    pub job_id: String,
    /// Workload ID (for cache tracking)
    pub workload_id: Option<String>,
    pub input: serde_json::Value,
    #[allow(dead_code)]
    pub lease_expires: i64,
    /// Runtime type: "container" or "wasm"
    #[serde(default = "default_runtime_type")]
    pub runtime_type: String,
    /// For container workloads
    pub container_image: Option<String>,
    /// Expected digest of the container image (sha256:...)
    /// If provided, the agent will verify the image digest before execution
    pub image_digest: Option<String>,
    /// For WASM workloads
    pub wasm_url: Option<String>,
    /// Expected hash of the WASM module
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

/// Job output chunk (for streaming text)
#[derive(Debug, Serialize)]
pub struct JobOutput {
    pub job_id: String,
    pub seq: u64,
    pub chunk: String,
    pub is_final: bool,
}

/// Job output with image data
#[derive(Debug, Serialize)]
pub struct JobImageOutput {
    pub job_id: String,
    pub image_data: String,  // base64 encoded
    pub format: String,      // "png", "jpeg", etc.
    pub width: u32,
    pub height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Job progress update
#[derive(Debug, Serialize)]
pub struct JobProgress {
    pub job_id: String,
    pub step: u32,
    pub total: u32,
}

/// Cancel job request from coordinator
#[derive(Debug, Deserialize)]
pub struct CancelJob {
    pub job_id: String,
}

/// Lease renewal request to coordinator
#[derive(Debug, Serialize)]
pub struct LeaseRenewal {
    pub job_id: String,
    pub extend_seconds: u64,
}

/// Pairing request message
#[derive(Debug, Serialize)]
pub struct PairingRequest {
    pub host_id: String,
}

/// Pairing response from coordinator
#[derive(Debug, Deserialize)]
pub struct PairingResponse {
    pub success: bool,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub expires_in_seconds: Option<u64>,
    #[serde(default)]
    pub pair_url: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
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

    /// Subscribe to cancel requests for this host
    pub async fn subscribe_cancel(&self) -> Result<Subscriber> {
        let subject = subjects::cancel(&self.host_id);
        let subscriber = self
            .client
            .subscribe(subject.clone())
            .await
            .context("Failed to subscribe to cancel")?;

        info!("Subscribed to cancel requests on {}", subject);
        Ok(subscriber)
    }

    /// Send heartbeat (basic)
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

    /// Send enhanced heartbeat with detailed metrics
    pub async fn send_enhanced_heartbeat(
        &self,
        active_jobs: u32,
        system: Option<SystemMetricsSnapshot>,
        gpus: Option<Vec<GpuMetricsSnapshot>>,
        active_job_metrics: Option<Vec<ActiveJobMetrics>>,
        cache: Option<CacheMetricsSnapshot>,
    ) -> Result<()> {
        let msg = EnhancedHeartbeat {
            host_id: self.host_id.clone(),
            status: "online".to_string(),
            active_jobs,
            timestamp: chrono_timestamp(),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
            system,
            gpus,
            active_job_metrics,
            cache,
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize enhanced heartbeat")?;

        self.client
            .publish(subjects::heartbeat(&self.host_id), payload.into())
            .await
            .context("Failed to publish enhanced heartbeat")?;

        debug!("Sent enhanced heartbeat with metrics");
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

    /// Publish job output chunk (for text streaming)
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

    /// Publish job progress update
    pub async fn publish_progress(&self, job_id: &str, step: u32, total: u32) -> Result<()> {
        let msg = JobProgress {
            job_id: job_id.to_string(),
            step,
            total,
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize progress")?;

        // Use a separate subject for progress updates
        let subject = format!("host.{}.progress", self.host_id);
        self.client
            .publish(subject, payload.into())
            .await
            .context("Failed to publish progress")?;

        Ok(())
    }

    /// Publish image output
    pub async fn publish_image(&self, job_id: &str, image_data: &str, format: &str, width: u32, height: u32, seed: Option<u64>) -> Result<()> {
        let msg = JobImageOutput {
            job_id: job_id.to_string(),
            image_data: image_data.to_string(),
            format: format.to_string(),
            width,
            height,
            seed,
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize image output")?;

        // Use a separate subject for image outputs
        let subject = format!("host.{}.image", self.host_id);
        self.client
            .publish(subject, payload.into())
            .await
            .context("Failed to publish image")?;

        Ok(())
    }

    /// Get the host ID
    #[allow(dead_code)]
    pub fn host_id(&self) -> &str {
        &self.host_id
    }

    /// Renew lease for a running job
    pub async fn renew_lease(&self, job_id: &str, extend_seconds: u64) -> Result<()> {
        let msg = LeaseRenewal {
            job_id: job_id.to_string(),
            extend_seconds,
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize lease renewal")?;

        self.client
            .publish(subjects::lease(&self.host_id), payload.into())
            .await
            .context("Failed to publish lease renewal")?;

        debug!("Renewed lease for job {} by {} seconds", job_id, extend_seconds);
        Ok(())
    }

    /// Request a pairing code from the coordinator
    pub async fn request_pairing(&self) -> Result<PairingResponse> {
        let msg = PairingRequest {
            host_id: self.host_id.clone(),
        };

        let payload = serde_json::to_vec(&msg).context("Failed to serialize pairing request")?;

        // Send request and wait for response (with 10 second timeout)
        let response = self
            .client
            .request(subjects::PAIRING, payload.into())
            .await
            .context("Failed to send pairing request")?;

        let pairing_response: PairingResponse =
            serde_json::from_slice(&response.payload).context("Failed to parse pairing response")?;

        Ok(pairing_response)
    }
}

/// Parse a job assignment message
pub fn parse_job_assignment(msg: &Message) -> Result<AssignJob> {
    match serde_json::from_slice(&msg.payload) {
        Ok(job) => Ok(job),
        Err(e) => {
            // Log the actual payload for debugging
            let payload_str = String::from_utf8_lossy(&msg.payload);
            tracing::error!("Failed to parse job assignment: {} - Payload: {}", e, payload_str);
            Err(e).context("Failed to parse job assignment")
        }
    }
}

/// Get current timestamp in milliseconds
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
