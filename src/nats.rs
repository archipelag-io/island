//! NATS client for coordinator communication
//!
//! Handles connection to NATS, job subscriptions, and message publishing.

use anyhow::{Context, Result};
use async_nats::jetstream;
use async_nats::{Client, ConnectOptions, Message, Subscriber};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

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

/// Island capabilities reported during registration
#[derive(Debug, Clone, Serialize)]
pub struct HostCapabilities {
    pub gpu_model: Option<String>,
    pub gpu_vram_mb: Option<u32>,
    pub cpu_cores: u32,
    pub ram_mb: u32,
    pub region: Option<String>,
}

/// Island registration message
#[derive(Debug, Serialize)]
pub struct RegisterHost {
    pub host_id: String,
    pub capabilities: HostCapabilities,
    pub version: String,
}

/// Heartbeat message (basic — retained as fallback; enhanced heartbeat is preferred)
#[allow(dead_code)]
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
    /// Performance estimates for workload fit scoring
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance_estimates: Option<PerformanceEstimates>,
}

/// Performance estimates computed from hardware specs.
/// Used by the coordinator for workload-to-Island fit scoring.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceEstimates {
    /// GPU memory bandwidth in GB/s (looked up from GPU model)
    pub gpu_bandwidth_gb_s: Option<f32>,
    /// Estimated LLM tokens/second for a 7B Q4 model
    pub estimated_llm_tok_s: Option<f32>,
    /// Max concurrent containers (based on CPU/RAM)
    pub max_concurrent_containers: Option<u32>,
    /// Wasmtime linear memory limit in MB (if WASM runtime supported)
    pub wasm_memory_limit_mb: Option<u32>,
    /// Supported runtime types
    pub supported_runtimes: Vec<String>,
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
    /// Workload ID (for cache tracking) — accepts both string and integer from coordinator
    #[serde(default, deserialize_with = "deserialize_string_or_int")]
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
    /// For GGUF/llmcpp and diffusers workloads
    #[allow(dead_code)]
    pub model_url: Option<String>,
    #[allow(dead_code)]
    pub model_hash: Option<String>,
    #[allow(dead_code)]
    pub model_context_size: Option<u32>,
    #[allow(dead_code)]
    pub model_temperature: Option<f32>,
    /// For ONNX workloads
    #[allow(dead_code)]
    pub onnx_model_url: Option<String>,
    #[allow(dead_code)]
    pub onnx_model_hash: Option<String>,
    #[allow(dead_code)]
    pub onnx_task_type: Option<String>,
    /// Sandbox tier for trust-level-based resource limits
    /// Values: "restricted", "standard", "elevated"
    pub sandbox_tier: Option<String>,
    /// Pipeline configuration (present when this is a pipeline shard job)
    #[serde(default)]
    pub pipeline_config: Option<serde_json::Value>,
}

fn default_runtime_type() -> String {
    "container".to_string()
}

/// Deserialize a field that may be a string or an integer (coordinator sends workload_id as int)
fn deserialize_string_or_int<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrInt;

    impl<'de> de::Visitor<'de> for StringOrInt {
        type Value = Option<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string, integer, or null")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }

        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
    }

    deserializer.deserialize_any(StringOrInt)
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
    pub image_data: String, // base64 encoded
    pub format: String,     // "png", "jpeg", etc.
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
        let mut options = ConnectOptions::new()
            .name(format!("island-{}", &host_id[..8]))
            .retry_on_initial_connect()
            .connection_timeout(Duration::from_secs(10))
            .ping_interval(Duration::from_secs(10))
            .max_reconnects(None); // Reconnect forever

        // Parse credentials from URL (e.g., tls://user:pass@host:port)
        let connect_url = if let Some(at_pos) = nats_url.find('@') {
            let scheme_end = nats_url.find("://").map(|p| p + 3).unwrap_or(0);
            let userinfo = &nats_url[scheme_end..at_pos];
            if let Some(colon) = userinfo.find(':') {
                let user = &userinfo[..colon];
                let pass = &userinfo[colon + 1..];
                options = options.user_and_password(user.to_string(), pass.to_string());
            }
            // Reconstruct URL without credentials
            let scheme = &nats_url[..scheme_end];
            let host_part = &nats_url[at_pos + 1..];
            format!("{}{}", scheme, host_part)
        } else {
            nats_url.to_string()
        };

        let client = options
            .connect(&connect_url)
            .await
            .context("Failed to connect to NATS")?;

        // Log URL without credentials
        info!("Connected to NATS at {}", connect_url);

        Ok(Self { client, host_id })
    }

    /// Register this Island with the coordinator
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

        info!("Registered Island {} with coordinator", self.host_id);
        Ok(())
    }

    /// Subscribe to job assignments for this Island.
    ///
    /// Tries JetStream pull consumer first (stream JOBS, consumer host-{id}).
    /// Falls back to core NATS subscription if the stream doesn't exist.
    pub async fn subscribe_jobs(&self) -> Result<JobSubscription> {
        match self.try_jetstream_subscribe().await {
            Ok(js_sub) => {
                info!(
                    "Subscribed to job assignments via JetStream (host-{})",
                    &self.host_id[..8]
                );
                Ok(js_sub)
            }
            Err(e) => {
                warn!(
                    "JetStream subscribe failed ({}), falling back to core NATS",
                    e
                );
                let subject = subjects::jobs(&self.host_id);
                let subscriber = self
                    .client
                    .subscribe(subject.clone())
                    .await
                    .context("Failed to subscribe to jobs")?;

                info!("Subscribed to job assignments on {} (core NATS)", subject);
                Ok(JobSubscription::Core(subscriber))
            }
        }
    }

    /// Attempt to set up a JetStream pull consumer for durable job delivery on this Island.
    async fn try_jetstream_subscribe(&self) -> Result<JobSubscription> {
        let js = jetstream::new(self.client.clone());

        // Check if the JOBS stream exists
        let stream = js
            .get_stream("JOBS")
            .await
            .context("JOBS stream not found")?;

        let consumer_name = format!("host-{}", self.host_id);
        let filter_subject = subjects::jobs(&self.host_id);

        // Get or create the durable consumer for this Island
        let consumer = match stream.get_consumer(&consumer_name).await {
            Ok(consumer) => consumer,
            Err(_) => {
                // Create durable pull consumer for this Island
                let config = jetstream::consumer::pull::Config {
                    durable_name: Some(consumer_name.clone()),
                    ack_policy: jetstream::consumer::AckPolicy::Explicit,
                    filter_subject: filter_subject.clone(),
                    max_deliver: 5,
                    ack_wait: Duration::from_secs(60),
                    ..Default::default()
                };

                stream
                    .create_consumer(config)
                    .await
                    .context("Failed to create JetStream consumer")?
            }
        };

        let messages = consumer
            .messages()
            .await
            .context("Failed to get JetStream message stream")?;

        Ok(JobSubscription::JetStream(Box::new(messages)))
    }

    /// Subscribe to job assignments via core NATS (used for recovery)
    pub async fn subscribe_jobs_core(&self) -> Result<Subscriber> {
        let subject = subjects::jobs(&self.host_id);
        let subscriber = self
            .client
            .subscribe(subject.clone())
            .await
            .context("Failed to subscribe to jobs")?;

        info!("Subscribed to job assignments on {}", subject);
        Ok(subscriber)
    }

    /// Subscribe to cancel requests for this Island
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

    /// Send heartbeat (basic — retained as fallback; enhanced heartbeat is preferred)
    #[allow(dead_code)]
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
        performance_estimates: Option<PerformanceEstimates>,
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
            performance_estimates,
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
    pub async fn publish_status(
        &self,
        job_id: &str,
        state: &str,
        error: Option<String>,
    ) -> Result<()> {
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
    pub async fn publish_output(
        &self,
        job_id: &str,
        seq: u64,
        chunk: &str,
        is_final: bool,
    ) -> Result<()> {
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
    pub async fn publish_image(
        &self,
        job_id: &str,
        image_data: &str,
        format: &str,
        width: u32,
        height: u32,
        seed: Option<u64>,
    ) -> Result<()> {
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

    /// Get the Island's host ID
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

        debug!(
            "Renewed lease for job {} by {} seconds",
            job_id, extend_seconds
        );
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

        let pairing_response: PairingResponse = serde_json::from_slice(&response.payload)
            .context("Failed to parse pairing response")?;

        Ok(pairing_response)
    }

    // ========================================================================
    // Pipeline ring methods
    // ========================================================================

    /// Subscribe to a ring subject (activation or control)
    pub async fn subscribe_ring(&self, subject: &str) -> Result<Subscriber> {
        self.client
            .subscribe(subject.to_string())
            .await
            .context(format!("Failed to subscribe to ring subject: {}", subject))
    }

    /// Publish a status message on the ring status subject
    pub async fn publish_ring_status(
        &self,
        group_id: &str,
        msg: &serde_json::Value,
    ) -> Result<()> {
        let subject = format!("ring.{}.status", group_id);
        let payload = serde_json::to_vec(msg).context("Failed to serialize ring status")?;
        self.client
            .publish(subject, payload.into())
            .await
            .context("Failed to publish ring status")?;
        Ok(())
    }

    /// Publish output from the last position in a ring
    pub async fn publish_ring_output(
        &self,
        group_id: &str,
        msg: &serde_json::Value,
    ) -> Result<()> {
        let subject = format!("ring.{}.output", group_id);
        let payload = serde_json::to_vec(msg).context("Failed to serialize ring output")?;
        self.client
            .publish(subject, payload.into())
            .await
            .context("Failed to publish ring output")?;
        Ok(())
    }

    /// Publish raw bytes to a subject (for activation forwarding)
    pub async fn publish_raw(&self, subject: &str, data: Vec<u8>) -> Result<()> {
        self.client
            .publish(subject.to_string(), data.into())
            .await
            .context("Failed to publish raw data")?;
        Ok(())
    }
}

/// Abstraction over core NATS and JetStream subscriptions for job delivery.
///
/// When using JetStream, messages are acked after successful job spawn
/// (not after completion — that would be too late for lease-based delivery).
pub enum JobSubscription {
    /// Core NATS subscription (fire-and-forget)
    Core(Subscriber),
    /// JetStream pull consumer (at-least-once delivery with explicit ack)
    JetStream(Box<jetstream::consumer::pull::Stream>),
}

impl JobSubscription {
    /// Get the next job assignment message.
    ///
    /// For JetStream messages, acks the message immediately on receipt
    /// (ack-on-spawn, not ack-on-completion).
    pub async fn next(&mut self) -> Option<Message> {
        match self {
            JobSubscription::Core(sub) => sub.next().await,
            JobSubscription::JetStream(stream) => {
                loop {
                    match stream.next().await {
                        Some(Ok(jetstream_msg)) => {
                            // Ack immediately — the coordinator stream uses workqueue
                            // retention, so the message is removed on ack.
                            // We ack on spawn, not on completion, because the lease
                            // mechanism handles delivery guarantees after this point.
                            let inner = jetstream_msg.message.clone();
                            if let Err(e) = jetstream_msg.ack().await {
                                warn!("Failed to ack JetStream message: {}", e);
                            }
                            return Some(inner);
                        }
                        Some(Err(e)) => {
                            warn!("JetStream message error: {}", e);
                            continue;
                        }
                        None => return None,
                    }
                }
            }
        }
    }
}

/// Parse a job assignment message
pub fn parse_job_assignment(msg: &Message) -> Result<AssignJob> {
    match serde_json::from_slice(&msg.payload) {
        Ok(job) => Ok(job),
        Err(e) => {
            // Log the actual payload for debugging
            let payload_str = String::from_utf8_lossy(&msg.payload);
            tracing::error!(
                "Failed to parse job assignment: {} - Payload: {}",
                e,
                payload_str
            );
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
