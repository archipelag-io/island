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

    pub fn preload(host_id: &str) -> String {
        format!("host.{}.preload", host_id)
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
    /// Round-trip time to NATS server in milliseconds (measured via request/reply)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nats_rtt_ms: Option<f32>,
    /// Public IP:port as discovered via STUN (for QUIC direct transport)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_addr: Option<String>,
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
#[allow(dead_code)] // Protocol fields used by future features
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
    /// Expert configuration (present when this is an MoE expert/router job)
    #[serde(default)]
    pub expert_config: Option<serde_json::Value>,
    /// Speculative decoding configuration (present when this is a draft/verify job)
    #[serde(default)]
    pub speculative_config: Option<serde_json::Value>,
    /// Federated training configuration (present when this is a training participant job)
    #[serde(default)]
    pub training_config: Option<serde_json::Value>,
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

/// Preload recommendation from coordinator (demand-driven)
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct PreloadRecommendation {
    #[serde(rename = "type")]
    pub msg_type: Option<String>,
    pub workload_slug: String,
    pub model_url: Option<String>,
    pub model_hash: Option<String>,
    pub runtime_type: String,
    pub estimated_earnings_per_job: Option<String>,
    pub queued_demand: Option<u32>,
    pub demand_score: Option<u32>,
    pub priority: Option<String>,
}

/// Response from the coordinator recommendations API
#[derive(Debug, Deserialize)]
pub struct RecommendationsResponse {
    pub recommendations: Vec<PreloadRecommendation>,
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
        let (connect_url, credentials) = parse_nats_url(nats_url);
        if let Some((user, pass)) = credentials {
            options = options.user_and_password(user, pass);
        }

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

    /// Subscribe to preload recommendations for this Island
    pub async fn subscribe_preload(&self) -> Result<Subscriber> {
        let subject = subjects::preload(&self.host_id);
        let subscriber = self
            .client
            .subscribe(subject.clone())
            .await
            .context("Failed to subscribe to preload")?;

        info!("Subscribed to preload recommendations: {}", subject);
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

    /// Get a reference to the underlying NATS client (for probe replies etc.)
    pub fn client(&self) -> &async_nats::Client {
        &self.client
    }

    /// Measure round-trip time to the NATS server in milliseconds.
    /// Uses a request/reply on a per-host ping subject. Returns None on timeout or error.
    pub async fn measure_rtt(&self) -> Option<f32> {
        let subject = format!("host.{}.ping", self.host_id);
        let start = std::time::Instant::now();

        // Use a short timeout — we just want RTT, not a meaningful response
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.client.request(subject, "ping".into()),
        )
        .await
        {
            Ok(Ok(_)) => {
                let rtt = start.elapsed().as_secs_f32() * 1000.0;
                Some(rtt)
            }
            _ => {
                // NATS request/reply may not have a responder — fall back to flush timing
                let start2 = std::time::Instant::now();
                if self.client.flush().await.is_ok() {
                    let rtt = start2.elapsed().as_secs_f32() * 1000.0;
                    Some(rtt)
                } else {
                    None
                }
            }
        }
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
    // Peer-to-peer RTT probing
    // ========================================================================

    /// Subscribe to incoming probe requests from other Islands
    pub async fn subscribe_probes(&self) -> Result<Subscriber> {
        let subject = format!("host.{}.probe", self.host_id);
        self.client
            .subscribe(subject)
            .await
            .context("Failed to subscribe to probes")
    }

    /// Measure round-trip time to a specific peer Island via NATS request/reply.
    /// Returns RTT in milliseconds, or None on timeout.
    #[allow(dead_code)]
    pub async fn probe_peer(&self, peer_host_id: &str) -> Option<f32> {
        let subject = format!("host.{}.probe", peer_host_id);
        let start = std::time::Instant::now();

        match tokio::time::timeout(
            std::time::Duration::from_secs(3),
            self.client.request(subject, "probe".into()),
        )
        .await
        {
            Ok(Ok(_)) => {
                let rtt = start.elapsed().as_secs_f32() * 1000.0;
                Some(rtt)
            }
            _ => None,
        }
    }

    // ========================================================================
    // Pipeline ring methods
    // ========================================================================

    /// Subscribe to a ring subject (activation or control)
    #[allow(dead_code)] // Pipeline ring feature not yet wired
    pub async fn subscribe_ring(&self, subject: &str) -> Result<Subscriber> {
        self.client
            .subscribe(subject.to_string())
            .await
            .context(format!("Failed to subscribe to ring subject: {}", subject))
    }

    /// Publish a status message on the ring status subject
    #[allow(dead_code)] // Pipeline ring feature not yet wired
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
    #[allow(dead_code)] // Pipeline ring feature not yet wired
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
    #[allow(dead_code)] // Pipeline ring feature not yet wired
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

/// Parse a NATS URL, extracting credentials if present.
///
/// Returns `(clean_url, Option<(user, password)>)` where `clean_url` has
/// credentials stripped (safe for logging / connecting).
///
/// Supports URLs like:
/// - `nats://host:4222`            → no credentials
/// - `tls://user:pass@host:4222`   → credentials extracted
/// - `host:4222`                   → no scheme, no credentials
pub fn parse_nats_url(nats_url: &str) -> (String, Option<(String, String)>) {
    if let Some(at_pos) = nats_url.find('@') {
        let scheme_end = nats_url.find("://").map(|p| p + 3).unwrap_or(0);
        let userinfo = &nats_url[scheme_end..at_pos];
        let credentials = if let Some(colon) = userinfo.find(':') {
            let user = &userinfo[..colon];
            let pass = &userinfo[colon + 1..];
            Some((user.to_string(), pass.to_string()))
        } else {
            None
        };
        // Reconstruct URL without credentials
        let scheme = &nats_url[..scheme_end];
        let host_part = &nats_url[at_pos + 1..];
        (format!("{}{}", scheme, host_part), credentials)
    } else {
        (nats_url.to_string(), None)
    }
}

/// Get current timestamp in milliseconds
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========================================================================
    // 1. Subject pattern generation
    // ========================================================================

    #[test]
    fn test_subject_jobs() {
        assert_eq!(subjects::jobs("abc-123"), "host.abc-123.jobs");
    }

    #[test]
    fn test_subject_status() {
        assert_eq!(subjects::status("host-1"), "host.host-1.status");
    }

    #[test]
    fn test_subject_output() {
        assert_eq!(subjects::output("host-1"), "host.host-1.output");
    }

    #[test]
    fn test_subject_heartbeat() {
        assert_eq!(subjects::heartbeat("id-42"), "host.id-42.heartbeat");
    }

    #[test]
    fn test_subject_cancel() {
        assert_eq!(subjects::cancel("h"), "host.h.cancel");
    }

    #[test]
    fn test_subject_lease() {
        assert_eq!(subjects::lease("host-1"), "host.host-1.lease");
    }

    #[test]
    fn test_subject_preload() {
        assert_eq!(subjects::preload("host-1"), "host.host-1.preload");
    }

    #[test]
    fn test_subject_constants() {
        assert_eq!(subjects::REGISTRATION, "coordinator.hosts.register");
        assert_eq!(subjects::PAIRING, "coordinator.hosts.pairing");
    }

    // ========================================================================
    // 2. AssignJob deserialization
    // ========================================================================

    fn full_assign_job_json() -> serde_json::Value {
        json!({
            "job_id": "job-abc-123",
            "workload_id": "42",
            "input": {"prompt": "Hello"},
            "lease_expires": 1710000000000_i64,
            "runtime_type": "container",
            "container_image": "registry.example.com/llm-chat:latest",
            "image_digest": "sha256:abcdef1234567890",
            "wasm_url": null,
            "wasm_hash": null,
            "model_url": null,
            "model_hash": null,
            "model_context_size": null,
            "model_temperature": null,
            "onnx_model_url": null,
            "onnx_model_hash": null,
            "onnx_task_type": null,
            "sandbox_tier": "standard"
        })
    }

    #[test]
    fn test_assign_job_all_fields() {
        let json_str = serde_json::to_string(&full_assign_job_json()).unwrap();
        let job: AssignJob = serde_json::from_str(&json_str).unwrap();
        assert_eq!(job.job_id, "job-abc-123");
        assert_eq!(job.workload_id, Some("42".to_string()));
        assert_eq!(job.runtime_type, "container");
        assert_eq!(
            job.container_image,
            Some("registry.example.com/llm-chat:latest".to_string())
        );
        assert_eq!(
            job.image_digest,
            Some("sha256:abcdef1234567890".to_string())
        );
        assert_eq!(job.sandbox_tier, Some("standard".to_string()));
        assert_eq!(job.lease_expires, 1710000000000);
    }

    #[test]
    fn test_assign_job_workload_id_as_string() {
        let j = json!({
            "job_id": "j1",
            "workload_id": "my-workload",
            "input": {},
            "lease_expires": 0
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.workload_id, Some("my-workload".to_string()));
    }

    #[test]
    fn test_assign_job_workload_id_as_integer() {
        let j = json!({
            "job_id": "j1",
            "workload_id": 99,
            "input": {},
            "lease_expires": 0
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.workload_id, Some("99".to_string()));
    }

    #[test]
    fn test_assign_job_workload_id_null() {
        let j = json!({
            "job_id": "j1",
            "workload_id": null,
            "input": {},
            "lease_expires": 0
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.workload_id, None);
    }

    #[test]
    fn test_assign_job_workload_id_absent() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.workload_id, None);
    }

    #[test]
    fn test_assign_job_missing_optional_fields() {
        let j = json!({
            "job_id": "j1",
            "input": {"prompt": "test"},
            "lease_expires": 100
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.container_image, None);
        assert_eq!(job.image_digest, None);
        assert_eq!(job.wasm_url, None);
        assert_eq!(job.wasm_hash, None);
        assert_eq!(job.model_url, None);
        assert_eq!(job.model_hash, None);
        assert_eq!(job.model_context_size, None);
        assert_eq!(job.model_temperature, None);
        assert_eq!(job.onnx_model_url, None);
        assert_eq!(job.onnx_model_hash, None);
        assert_eq!(job.onnx_task_type, None);
        assert_eq!(job.sandbox_tier, None);
        assert_eq!(job.pipeline_config, None);
        assert_eq!(job.expert_config, None);
        assert_eq!(job.speculative_config, None);
        assert_eq!(job.training_config, None);
    }

    #[test]
    fn test_assign_job_default_runtime_type() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.runtime_type, "container");
    }

    #[test]
    fn test_assign_job_runtime_type_wasm() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0,
            "runtime_type": "wasm",
            "wasm_url": "https://example.com/module.wasm",
            "wasm_hash": "sha256:abc"
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.runtime_type, "wasm");
        assert_eq!(
            job.wasm_url,
            Some("https://example.com/module.wasm".to_string())
        );
    }

    #[test]
    fn test_assign_job_runtime_type_onnx() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0,
            "runtime_type": "onnx",
            "onnx_model_url": "https://example.com/model.onnx",
            "onnx_task_type": "text-classification"
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.runtime_type, "onnx");
        assert_eq!(job.onnx_task_type, Some("text-classification".to_string()));
    }

    #[test]
    fn test_assign_job_runtime_type_llmcpp() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0,
            "runtime_type": "llmcpp",
            "model_url": "https://hf.co/model.gguf",
            "model_hash": "sha256:xyz",
            "model_context_size": 2048,
            "model_temperature": 0.7
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.runtime_type, "llmcpp");
        assert_eq!(
            job.model_url,
            Some("https://hf.co/model.gguf".to_string())
        );
        assert_eq!(job.model_context_size, Some(2048));
        assert_eq!(job.model_temperature, Some(0.7));
    }

    #[test]
    fn test_assign_job_runtime_type_diffusers() {
        let j = json!({
            "job_id": "j1",
            "input": {},
            "lease_expires": 0,
            "runtime_type": "diffusers"
        });
        let job: AssignJob = serde_json::from_value(j).unwrap();
        assert_eq!(job.runtime_type, "diffusers");
    }

    #[test]
    fn test_assign_job_malformed_json() {
        let result = serde_json::from_str::<AssignJob>("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_assign_job_missing_required_field() {
        // job_id is required
        let j = json!({
            "input": {},
            "lease_expires": 0
        });
        let result = serde_json::from_value::<AssignJob>(j);
        assert!(result.is_err());
    }

    // ========================================================================
    // 3. NATS URL credential parsing
    // ========================================================================

    #[test]
    fn test_parse_nats_url_plain() {
        let (url, creds) = parse_nats_url("nats://host:4222");
        assert_eq!(url, "nats://host:4222");
        assert!(creds.is_none());
    }

    #[test]
    fn test_parse_nats_url_with_credentials() {
        let (url, creds) = parse_nats_url("tls://user:pass@host:4222");
        assert_eq!(url, "tls://host:4222");
        let (user, pass) = creds.unwrap();
        assert_eq!(user, "user");
        assert_eq!(pass, "pass");
    }

    #[test]
    fn test_parse_nats_url_no_scheme() {
        let (url, creds) = parse_nats_url("host:4222");
        assert_eq!(url, "host:4222");
        assert!(creds.is_none());
    }

    #[test]
    fn test_parse_nats_url_credentials_special_chars() {
        let (url, creds) = parse_nats_url("nats://admin:p%40ss@sail.example.com:4222");
        assert_eq!(url, "nats://sail.example.com:4222");
        let (user, pass) = creds.unwrap();
        assert_eq!(user, "admin");
        assert_eq!(pass, "p%40ss");
    }

    // ========================================================================
    // 4. Serialization round-trip tests
    // ========================================================================

    #[test]
    fn test_register_host_serialization() {
        let msg = RegisterHost {
            host_id: "host-1".to_string(),
            capabilities: HostCapabilities {
                gpu_model: Some("RTX 4090".to_string()),
                gpu_vram_mb: Some(24576),
                cpu_cores: 16,
                ram_mb: 65536,
                region: Some("us-east".to_string()),
            },
            version: "0.1.0".to_string(),
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["host_id"], "host-1");
        assert_eq!(v["capabilities"]["gpu_model"], "RTX 4090");
        assert_eq!(v["capabilities"]["gpu_vram_mb"], 24576);
        assert_eq!(v["capabilities"]["cpu_cores"], 16);
        assert_eq!(v["capabilities"]["ram_mb"], 65536);
        assert_eq!(v["capabilities"]["region"], "us-east");
        assert_eq!(v["version"], "0.1.0");
    }

    #[test]
    fn test_heartbeat_serialization() {
        let msg = Heartbeat {
            host_id: "h1".to_string(),
            status: "online".to_string(),
            active_jobs: 3,
            timestamp: 1710000000000,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["host_id"], "h1");
        assert_eq!(v["status"], "online");
        assert_eq!(v["active_jobs"], 3);
        assert_eq!(v["timestamp"], 1710000000000_i64);
    }

    #[test]
    fn test_enhanced_heartbeat_serialization_minimal() {
        let msg = EnhancedHeartbeat {
            host_id: "h1".to_string(),
            status: "online".to_string(),
            active_jobs: 0,
            timestamp: 100,
            agent_version: "0.1.0".to_string(),
            system: None,
            gpus: None,
            active_job_metrics: None,
            cache: None,
            performance_estimates: None,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["host_id"], "h1");
        // Optional fields should be absent (skip_serializing_if = None)
        assert!(v.get("system").is_none());
        assert!(v.get("gpus").is_none());
        assert!(v.get("cache").is_none());
        assert!(v.get("performance_estimates").is_none());
    }

    #[test]
    fn test_enhanced_heartbeat_serialization_full() {
        let msg = EnhancedHeartbeat {
            host_id: "h1".to_string(),
            status: "online".to_string(),
            active_jobs: 1,
            timestamp: 100,
            agent_version: "0.2.0".to_string(),
            system: Some(SystemMetricsSnapshot {
                cpu_percent: 45.5,
                memory_used_mb: 8192,
                memory_total_mb: 32768,
                disk_used_gb: 100,
                disk_total_gb: 500,
            }),
            gpus: Some(vec![GpuMetricsSnapshot {
                index: 0,
                utilization_percent: 80,
                memory_used_mb: 12000,
                memory_total_mb: 24576,
                temperature_c: 72,
                power_draw_w: 280.5,
            }]),
            active_job_metrics: Some(vec![ActiveJobMetrics {
                job_id: "j1".to_string(),
                job_type: "llm-chat".to_string(),
                duration_ms: 5000,
                tokens_generated: Some(150),
                memory_mb: Some(4096),
                gpu_memory_mb: Some(8000),
            }]),
            cache: Some(CacheMetricsSnapshot {
                cached_image_count: 3,
                cached_size_mb: 15000,
                warm_workload_count: 2,
                warm_workload_ids: vec!["w1".to_string(), "w2".to_string()],
            }),
            performance_estimates: Some(PerformanceEstimates {
                gpu_bandwidth_gb_s: Some(1008.0),
                estimated_llm_tok_s: Some(45.0),
                max_concurrent_containers: Some(4),
                wasm_memory_limit_mb: Some(256),
                supported_runtimes: vec!["container".to_string(), "wasm".to_string()],
                nats_rtt_ms: Some(12.5),
                public_addr: Some("1.2.3.4:5678".to_string()),
            }),
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["system"]["cpu_percent"], 45.5);
        assert_eq!(v["gpus"][0]["utilization_percent"], 80);
        assert_eq!(v["active_job_metrics"][0]["tokens_generated"], 150);
        assert_eq!(v["cache"]["warm_workload_count"], 2);
        assert_eq!(v["performance_estimates"]["nats_rtt_ms"], 12.5);
    }

    #[test]
    fn test_job_status_serialization() {
        let msg = JobStatus {
            job_id: "j1".to_string(),
            state: "running".to_string(),
            error: None,
            timestamp: 100,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["job_id"], "j1");
        assert_eq!(v["state"], "running");
        // error should be absent when None
        assert!(v.get("error").is_none());
    }

    #[test]
    fn test_job_status_serialization_with_error() {
        let msg = JobStatus {
            job_id: "j1".to_string(),
            state: "failed".to_string(),
            error: Some("OOM killed".to_string()),
            timestamp: 100,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["error"], "OOM killed");
    }

    #[test]
    fn test_job_output_serialization() {
        let msg = JobOutput {
            job_id: "j1".to_string(),
            seq: 42,
            chunk: "Hello world".to_string(),
            is_final: false,
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["seq"], 42);
        assert_eq!(v["chunk"], "Hello world");
        assert_eq!(v["is_final"], false);
    }

    #[test]
    fn test_cancel_job_deserialization() {
        let j = json!({"job_id": "job-to-cancel"});
        let cancel: CancelJob = serde_json::from_value(j).unwrap();
        assert_eq!(cancel.job_id, "job-to-cancel");
    }

    #[test]
    fn test_preload_recommendation_deserialization() {
        let j = json!({
            "type": "preload",
            "workload_slug": "llm-chat",
            "model_url": "https://hf.co/model.gguf",
            "model_hash": "sha256:abc",
            "runtime_type": "llmcpp",
            "estimated_earnings_per_job": "0.05",
            "queued_demand": 10,
            "demand_score": 85,
            "priority": "high"
        });
        let rec: PreloadRecommendation = serde_json::from_value(j).unwrap();
        assert_eq!(rec.msg_type, Some("preload".to_string()));
        assert_eq!(rec.workload_slug, "llm-chat");
        assert_eq!(rec.runtime_type, "llmcpp");
        assert_eq!(rec.queued_demand, Some(10));
        assert_eq!(rec.priority, Some("high".to_string()));
    }

    #[test]
    fn test_preload_recommendation_minimal() {
        let j = json!({
            "workload_slug": "image-gen",
            "runtime_type": "diffusers"
        });
        let rec: PreloadRecommendation = serde_json::from_value(j).unwrap();
        assert_eq!(rec.workload_slug, "image-gen");
        assert_eq!(rec.msg_type, None);
        assert_eq!(rec.model_url, None);
        assert_eq!(rec.estimated_earnings_per_job, None);
    }

    #[test]
    fn test_pairing_request_serialization() {
        let msg = PairingRequest {
            host_id: "host-abc".to_string(),
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(v["host_id"], "host-abc");
    }

    #[test]
    fn test_pairing_response_deserialization_success() {
        let j = json!({
            "success": true,
            "code": "ABC-123",
            "expires_in_seconds": 300,
            "pair_url": "https://archipelag.io/pair/ABC-123"
        });
        let resp: PairingResponse = serde_json::from_value(j).unwrap();
        assert!(resp.success);
        assert_eq!(resp.code, Some("ABC-123".to_string()));
        assert_eq!(resp.expires_in_seconds, Some(300));
        assert_eq!(
            resp.pair_url,
            Some("https://archipelag.io/pair/ABC-123".to_string())
        );
        assert_eq!(resp.error, None);
    }

    #[test]
    fn test_pairing_response_deserialization_error() {
        let j = json!({
            "success": false,
            "error": "host not approved"
        });
        let resp: PairingResponse = serde_json::from_value(j).unwrap();
        assert!(!resp.success);
        assert_eq!(resp.error, Some("host not approved".to_string()));
        assert_eq!(resp.code, None);
    }
}
