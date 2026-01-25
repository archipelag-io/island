//! Main agent logic
//!
//! Coordinates NATS connection, job execution, and heartbeats.

use crate::config::AgentConfig;
use crate::docker::{self, ContainerConfig, ContainerOutput};
use crate::messages::WorkloadOutput;
use crate::nats::{self, AssignJob, CancelJob, HostCapabilities, NatsAgent};
use crate::wasm::{WasmConfig, WasmExecutor, WasmOutput};
use anyhow::{Context, Result};
use bollard::Docker;
use dashmap::DashMap;
use futures_util::StreamExt;
use rand::Rng;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use sysinfo::System;
use tokio::select;
use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

/// Maximum backoff delay for reconnection attempts (30 seconds)
const MAX_BACKOFF_SECS: u64 = 30;
/// Initial backoff delay (1 second)
const INITIAL_BACKOFF_SECS: u64 = 1;
/// Jitter range for backoff (±25%)
const JITTER_MIN: f64 = 0.75;
const JITTER_MAX: f64 = 1.25;

/// The main agent that coordinates all activity
pub struct Agent {
    config: AgentConfig,
    docker: Docker,
    nats: NatsAgent,
    active_jobs: Arc<AtomicU32>,
    shutdown: Arc<AtomicBool>,
    /// Map of job_id -> cancel sender for running jobs
    job_cancellers: Arc<DashMap<String, watch::Sender<bool>>>,
}

impl Agent {
    /// Create a new agent
    pub async fn new(config: AgentConfig, docker: Docker) -> Result<Self> {
        // Generate or use existing host ID
        let host_id = config
            .host_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        info!("Host ID: {}", host_id);

        // Connect to NATS
        let nats = NatsAgent::connect(&config.coordinator.nats_url, host_id).await?;

        Ok(Self {
            config,
            docker,
            nats,
            active_jobs: Arc::new(AtomicU32::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            job_cancellers: Arc::new(DashMap::new()),
        })
    }

    /// Run the agent
    pub async fn run(&self) -> Result<()> {
        // Detect capabilities once at startup
        let capabilities = self.detect_capabilities();

        // Initial registration
        self.register_and_setup(&capabilities).await?;

        // Request pairing if needed
        self.check_and_request_pairing().await;

        // Create channel for job completion notifications
        let (job_done_tx, mut job_done_rx) = mpsc::channel::<String>(32);

        // Subscribe to job assignments
        let mut job_subscriber = self.nats.subscribe_jobs().await?;

        // Subscribe to cancel requests
        let mut cancel_subscriber = self.nats.subscribe_cancel().await?;

        let mut consecutive_failures: u32 = 0;

        // Heartbeat interval
        let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(10));

        info!("Agent running. Waiting for jobs...");

        loop {
            select! {
                // Heartbeat tick
                _ = heartbeat_interval.tick() => {
                    let active = self.active_jobs.load(Ordering::Relaxed);
                    if let Err(e) = self.nats.send_heartbeat(active).await {
                        warn!("Failed to send heartbeat: {}", e);
                        consecutive_failures += 1;

                        // If heartbeats are failing, subscription might be stale
                        if consecutive_failures >= 3 {
                            warn!("Multiple heartbeat failures, attempting to resubscribe...");
                            match self.recover_subscription(&capabilities, &mut job_subscriber).await {
                                Ok(()) => {
                                    consecutive_failures = 0;
                                    info!("Successfully recovered subscription");
                                }
                                Err(e) => {
                                    error!("Failed to recover subscription: {}", e);
                                }
                            }
                        }
                    } else {
                        consecutive_failures = 0;
                    }
                }

                // Job assignment received (or subscription closed)
                msg = job_subscriber.next() => {
                    match msg {
                        Some(msg) => {
                            match nats::parse_job_assignment(&msg) {
                                Ok(job) => {
                                    info!("Received job assignment: {}", job.job_id);
                                    self.spawn_job(job, job_done_tx.clone());
                                }
                                Err(e) => {
                                    error!("Failed to parse job assignment: {}", e);
                                }
                            }
                        }
                        None => {
                            // Subscription closed - need to recover
                            warn!("Job subscription closed, attempting to recover...");
                            match self.recover_subscription(&capabilities, &mut job_subscriber).await {
                                Ok(()) => {
                                    info!("Successfully recovered subscription");
                                }
                                Err(e) => {
                                    error!("Failed to recover subscription after retries: {}", e);
                                    // Continue loop - will try again on next iteration
                                }
                            }
                        }
                    }
                }

                // Job completed
                Some(job_id) = job_done_rx.recv() => {
                    self.active_jobs.fetch_sub(1, Ordering::Relaxed);
                    // Clean up canceller
                    self.job_cancellers.remove(&job_id);
                    info!("Job {} completed", job_id);
                }

                // Cancel request received
                msg = cancel_subscriber.next() => {
                    if let Some(msg) = msg {
                        if let Ok(cancel) = serde_json::from_slice::<CancelJob>(&msg.payload) {
                            info!("Received cancel request for job {}", cancel.job_id);
                            // Signal cancellation to the running job
                            if let Some(sender) = self.job_cancellers.get(&cancel.job_id) {
                                let _ = sender.send(true);
                                info!("Signaled cancellation for job {}", cancel.job_id);
                            } else {
                                debug!("Job {} not found in active jobs (may have already completed)", cancel.job_id);
                            }
                        }
                    }
                }

                // Shutdown signal
                _ = tokio::signal::ctrl_c() => {
                    info!("Received shutdown signal");
                    self.shutdown.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        // Wait for active jobs to complete (with timeout)
        let active = self.active_jobs.load(Ordering::Relaxed);
        if active > 0 {
            info!("Waiting for {} active job(s) to complete...", active);
            tokio::time::sleep(Duration::from_secs(30)).await;
        }

        info!("Agent shutdown complete");
        Ok(())
    }

    /// Register with coordinator
    async fn register_and_setup(&self, capabilities: &HostCapabilities) -> Result<()> {
        self.nats.register(capabilities.clone()).await?;
        Ok(())
    }

    /// Recover subscription with exponential backoff
    async fn recover_subscription(
        &self,
        capabilities: &HostCapabilities,
        subscriber: &mut async_nats::Subscriber,
    ) -> Result<()> {
        let mut backoff = Duration::from_secs(INITIAL_BACKOFF_SECS);
        let max_backoff = Duration::from_secs(MAX_BACKOFF_SECS);
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 10;

        while attempts < MAX_ATTEMPTS {
            attempts += 1;

            // Apply jitter to prevent thundering herd
            let jitter = rand::thread_rng().gen_range(JITTER_MIN..JITTER_MAX);
            let backoff_with_jitter = backoff.mul_f64(jitter);

            info!(
                "Attempting to recover subscription (attempt {}/{}, backoff {:?})",
                attempts, MAX_ATTEMPTS, backoff_with_jitter
            );

            // Wait with jittered backoff
            tokio::time::sleep(backoff_with_jitter).await;

            // Try to re-register first (in case coordinator lost our state)
            if let Err(e) = self.nats.register(capabilities.clone()).await {
                warn!("Failed to re-register: {}", e);
                backoff = std::cmp::min(backoff * 2, max_backoff);
                continue;
            }

            // Try to resubscribe
            match self.nats.subscribe_jobs().await {
                Ok(new_subscriber) => {
                    *subscriber = new_subscriber;
                    info!("Subscription recovered successfully after {} attempts", attempts);
                    return Ok(());
                }
                Err(e) => {
                    warn!("Failed to resubscribe: {}", e);
                    backoff = std::cmp::min(backoff * 2, max_backoff);
                }
            }
        }

        anyhow::bail!(
            "Failed to recover subscription after {} attempts",
            MAX_ATTEMPTS
        )
    }

    /// Detect host capabilities
    fn detect_capabilities(&self) -> HostCapabilities {
        // Detect RAM using sysinfo
        let mut sys = System::new_all();
        sys.refresh_memory();
        let ram_mb = (sys.total_memory() / 1024 / 1024) as u32;

        // Detect GPU using nvidia-smi
        let (gpu_model, gpu_vram_mb) = detect_nvidia_gpu();

        let capabilities = HostCapabilities {
            gpu_model,
            gpu_vram_mb,
            cpu_cores: num_cpus::get() as u32,
            ram_mb,
            region: self.config.host.region.clone(),
        };

        info!(
            "Detected capabilities: {} CPU cores, {} MB RAM, GPU: {:?} ({:?} MB VRAM)",
            capabilities.cpu_cores,
            capabilities.ram_mb,
            capabilities.gpu_model,
            capabilities.gpu_vram_mb
        );

        capabilities
    }

    /// Check if host needs pairing and request a pairing code if so
    async fn check_and_request_pairing(&self) {
        // TODO: Check if host is already paired (could store in local config)
        // For now, always request pairing on startup

        match self.nats.request_pairing().await {
            Ok(response) => {
                if response.success {
                    if let Some(code) = response.code {
                        info!("========================================");
                        info!("       HOST PAIRING CODE: {}", code);
                        info!("========================================");
                        if let Some(url) = response.pair_url {
                            info!("Visit {} to pair this host", url);
                        }
                        if let Some(expires) = response.expires_in_seconds {
                            info!("Code expires in {} minutes", expires / 60);
                        }
                        info!("========================================");
                    }
                } else if let Some(error) = response.error {
                    if error.contains("already paired") {
                        info!("Host is already paired to an account");
                    } else {
                        warn!("Pairing request failed: {}", error);
                    }
                }
            }
            Err(e) => {
                // Don't fail startup if pairing request fails
                // This can happen if coordinator doesn't support pairing yet
                debug!("Could not request pairing code: {}", e);
            }
        }
    }

    /// Spawn a job execution task
    fn spawn_job(&self, job: AssignJob, done_tx: mpsc::Sender<String>) {
        self.active_jobs.fetch_add(1, Ordering::Relaxed);

        // Create cancel channel for this job
        let (cancel_tx, cancel_rx) = watch::channel(false);
        self.job_cancellers.insert(job.job_id.clone(), cancel_tx);

        let docker = self.docker.clone();
        let nats = self.nats.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let job_id = job.job_id.clone();

            if let Err(e) = execute_job(&docker, &nats, &config, job, shutdown, cancel_rx).await {
                error!("Job {} failed: {}", job_id, e);
                let _ = nats
                    .publish_status(&job_id, "failed", Some(e.to_string()))
                    .await;
            }

            let _ = done_tx.send(job_id).await;
        });
    }
}

/// Execute a single job (routes to container or WASM executor)
async fn execute_job(
    docker: &Docker,
    nats: &NatsAgent,
    config: &AgentConfig,
    job: AssignJob,
    _shutdown: Arc<AtomicBool>,
    cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    // Check if already cancelled before starting
    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // Notify started
    nats.publish_status(job_id, "started", None).await?;

    // Route based on runtime type
    match job.runtime_type.as_str() {
        "wasm" => execute_wasm_job(nats, &job, cancel_rx).await,
        "container" | _ => execute_container_job(docker, nats, config, &job, cancel_rx).await,
    }
}

/// Execute a WASM workload
async fn execute_wasm_job(nats: &NatsAgent, job: &AssignJob, mut cancel_rx: watch::Receiver<bool>) -> Result<()> {
    let job_id = &job.job_id;

    let wasm_url = job
        .wasm_url
        .as_ref()
        .context("WASM workload missing wasm_url")?;

    info!("Executing WASM workload: {}", wasm_url);

    // TODO: Download WASM module from URL and cache it
    // For now, assume wasm_url is a local path for testing
    let wasm_path = wasm_url.clone();

    // Prepare input
    let input_json = serde_json::to_string(&job.input).context("Failed to serialize job input")?;

    let wasm_config = WasmConfig {
        module_path: wasm_path,
        input: input_json,
        timeout_seconds: DEFAULT_WASM_TIMEOUT_SECS,
        expected_hash: job.wasm_hash.clone(),
        ..Default::default()
    };

    // Create WASM executor
    let executor = WasmExecutor::new()?;

    // Create channel for WASM output
    let (output_tx, mut output_rx) = mpsc::channel::<WasmOutput>(256);

    // Spawn WASM runner
    let wasm_handle = tokio::spawn(async move {
        executor.run(wasm_config, output_tx).await
    });

    // Lease renewal interval (every 30 seconds)
    let mut lease_interval = tokio::time::interval(Duration::from_secs(LEASE_RENEWAL_INTERVAL_SECS));
    lease_interval.tick().await; // Skip immediate first tick

    // Process WASM output with cancellation and lease renewal support
    let mut cancelled = false;
    let (exit_code, token_count, timed_out) = loop {
        select! {
            result = process_wasm_output(nats, job_id, &mut output_rx) => {
                break result?;
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!("WASM job {} cancelled", job_id);
                    cancelled = true;
                    wasm_handle.abort();
                    break (0, 0, false);
                }
            }
            // Lease renewal tick
            _ = lease_interval.tick() => {
                debug!("Renewing lease for WASM job {}", job_id);
                if let Err(e) = nats.renew_lease(job_id, LEASE_EXTENSION_SECS).await {
                    warn!("Failed to renew lease for WASM job {}: {}", job_id, e);
                }
            }
        }
    };

    // Wait for WASM to fully finish (if not aborted)
    if !cancelled {
        let _ = wasm_handle.await;
    }

    // Send final status
    if cancelled {
        nats.publish_status(job_id, "cancelled", None).await?;
    } else if timed_out {
        nats.publish_status(
            job_id,
            "failed",
            Some(format!("Timeout: job exceeded {}s limit", DEFAULT_WASM_TIMEOUT_SECS)),
        )
        .await?;
        warn!("WASM job {} failed: timeout", job_id);
    } else if exit_code == 0 {
        nats.publish_status(job_id, "succeeded", None).await?;
        info!("WASM job {} succeeded, generated {} tokens", job_id, token_count);
    } else {
        nats.publish_status(job_id, "failed", Some(format!("Exit code: {}", exit_code)))
            .await?;
    }

    Ok(())
}

/// Default timeout for WASM workloads (60 seconds)
const DEFAULT_WASM_TIMEOUT_SECS: u64 = 60;

/// Process WASM output stream
async fn process_wasm_output(
    nats: &NatsAgent,
    job_id: &str,
    output_rx: &mut mpsc::Receiver<WasmOutput>,
) -> Result<(i32, u32, bool)> {
    let mut seq: u64 = 0;
    let mut token_count: u32 = 0;
    let mut exit_code: i32 = 0;
    let mut streaming_started = false;
    let mut timed_out = false;

    while let Some(output) = output_rx.recv().await {
        match output {
            WasmOutput::Stdout(text) => {
                // Parse JSON lines from stdout
                for line in text.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    if let Ok(workload_output) = serde_json::from_str::<WorkloadOutput>(line) {
                        match &workload_output {
                            WorkloadOutput::Status { message } => {
                                debug!("WASM status: {}", message);
                                if !streaming_started {
                                    nats.publish_status(job_id, "streaming", None).await?;
                                    streaming_started = true;
                                }
                            }
                            WorkloadOutput::Token { content } => {
                                token_count += 1;
                                seq += 1;
                                nats.publish_output(job_id, seq, content, false).await?;
                            }
                            WorkloadOutput::Progress { step, total } => {
                                debug!("WASM progress: {}/{}", step, total);
                                nats.publish_progress(job_id, *step, *total).await?;
                            }
                            WorkloadOutput::Image { data, format, width, height } => {
                                info!("WASM image: {}x{} {}", width, height, format);
                                nats.publish_image(job_id, data, format, *width, *height, None).await?;
                            }
                            WorkloadOutput::Done { usage, seed } => {
                                debug!("WASM done: usage={:?}, seed={:?}", usage, seed);
                                nats.publish_output(job_id, seq + 1, "", true).await?;
                            }
                            WorkloadOutput::Error { message } => {
                                error!("WASM error: {}", message);
                            }
                        }
                    }
                }
            }
            WasmOutput::Stderr(text) => {
                debug!("WASM stderr: {}", text);
            }
            WasmOutput::Exit(code) => {
                exit_code = code;
                debug!("WASM exited with code: {}", code);
            }
            WasmOutput::Timeout => {
                warn!("WASM timed out after {}s", DEFAULT_WASM_TIMEOUT_SECS);
                timed_out = true;
                exit_code = -1;
            }
        }
    }

    Ok((exit_code, token_count, timed_out))
}

/// Default timeout for container workloads (5 minutes)
const DEFAULT_CONTAINER_TIMEOUT_SECS: u64 = 300;

/// Lease renewal interval (30 seconds)
const LEASE_RENEWAL_INTERVAL_SECS: u64 = 30;

/// Lease extension duration (60 seconds)
const LEASE_EXTENSION_SECS: u64 = 60;

/// Execute a container workload
async fn execute_container_job(
    docker: &Docker,
    nats: &NatsAgent,
    config: &AgentConfig,
    job: &AssignJob,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    // Use image from job assignment, fall back to config
    let image = job
        .container_image
        .clone()
        .unwrap_or_else(|| config.workload.llm_chat_image.clone());

    info!("Executing container workload: {}", image);

    // Prepare container config
    let input_json = serde_json::to_string(&job.input).context("Failed to serialize job input")?;

    // Build resource limits from config
    let limits = &config.workload.resource_limits;
    let memory_bytes = Some((limits.memory_mb * 1024 * 1024) as i64);

    // Set up tmpfs mounts if read-only rootfs is enabled
    let tmpfs_mounts = if limits.read_only_rootfs {
        let mut mounts = std::collections::HashMap::new();
        mounts.insert(
            "/tmp".to_string(),
            format!("rw,noexec,nosuid,size={}m", limits.tmpfs_size_mb),
        );
        Some(mounts)
    } else {
        None
    };

    // Convert CPU percentage to quota (100% = 100000 microseconds per period)
    let cpu_quota = limits.cpu_percent.map(|percent| (percent * 1000) as i64);

    let container_config = ContainerConfig {
        image,
        input: input_json,
        gpu_devices: config.workload.gpu_devices.clone(),
        timeout_seconds: DEFAULT_CONTAINER_TIMEOUT_SECS,
        expected_digest: job.image_digest.clone(),
        memory_bytes,
        read_only_rootfs: limits.read_only_rootfs,
        tmpfs_mounts,
        cpu_quota,
        network_disabled: limits.network_disabled,
    };

    // Create channel for container output
    let (output_tx, mut output_rx) = mpsc::channel::<ContainerOutput>(256);

    // Spawn container runner
    let docker_clone = docker.clone();
    let container_handle = tokio::spawn(async move {
        docker::run_container_streaming(&docker_clone, container_config, output_tx).await
    });

    // Track output
    let mut seq: u64 = 0;
    let mut buffer = String::new();
    let mut token_count: u32 = 0;
    let mut streaming_started = false;
    let mut failure_reason: Option<String> = None;
    let mut cancelled = false;

    // Lease renewal interval (every 30 seconds)
    let mut lease_interval = tokio::time::interval(Duration::from_secs(LEASE_RENEWAL_INTERVAL_SECS));
    lease_interval.tick().await; // Skip immediate first tick

    // Process container output and forward to NATS, with cancellation and lease renewal
    loop {
        select! {
            output = output_rx.recv() => {
                let Some(output) = output else { break };
                match output {
                    ContainerOutput::Stdout(chunk) => {
                        buffer.push_str(&chunk);

                        // Process complete JSON lines
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            if line.trim().is_empty() {
                                continue;
                            }

                            // Parse workload output
                            if let Ok(workload_output) = serde_json::from_str::<WorkloadOutput>(&line) {
                                match &workload_output {
                                    WorkloadOutput::Status { message } => {
                                        debug!("Workload status: {}", message);
                                        if message == "ready" && !streaming_started {
                                            nats.publish_status(job_id, "streaming", None).await?;
                                            streaming_started = true;
                                        }
                                    }
                                    WorkloadOutput::Token { content } => {
                                        token_count += 1;
                                        seq += 1;
                                        nats.publish_output(job_id, seq, content, false).await?;
                                    }
                                    WorkloadOutput::Progress { step, total } => {
                                        debug!("Workload progress: {}/{}", step, total);
                                        nats.publish_progress(job_id, *step, *total).await?;
                                    }
                                    WorkloadOutput::Image { data, format, width, height } => {
                                        info!("Received image: {}x{} {}", width, height, format);
                                        nats.publish_image(job_id, data, format, *width, *height, None).await?;
                                    }
                                    WorkloadOutput::Done { usage, seed } => {
                                        debug!("Workload done: usage={:?}, seed={:?}", usage, seed);
                                    }
                                    WorkloadOutput::Error { message } => {
                                        error!("Workload error: {}", message);
                                    }
                                }
                            } else {
                                debug!("Unparsed output line: {}", line);
                            }
                        }
                    }
                    ContainerOutput::Stderr(text) => {
                        debug!("Container stderr: {}", text);
                    }
                    ContainerOutput::Exit(code) => {
                        debug!("Container exited with code: {}", code);
                        break;
                    }
                    ContainerOutput::Timeout => {
                        warn!("Container timed out after {}s", DEFAULT_CONTAINER_TIMEOUT_SECS);
                        failure_reason = Some(format!(
                            "Timeout: job exceeded {}s limit",
                            DEFAULT_CONTAINER_TIMEOUT_SECS
                        ));
                        break;
                    }
                    ContainerOutput::OomKilled => {
                        error!("Container was killed due to out-of-memory");
                        failure_reason = Some("Out of memory: container exceeded memory limit".to_string());
                        break;
                    }
                    ContainerOutput::Crashed { exit_code, reason } => {
                        error!("Container crashed: {} (exit code {})", reason, exit_code);
                        failure_reason = Some(format!("Container crash: {}", reason));
                        break;
                    }
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    info!("Container job {} cancelled", job_id);
                    cancelled = true;
                    container_handle.abort();
                    break;
                }
            }
            // Lease renewal tick
            _ = lease_interval.tick() => {
                debug!("Renewing lease for job {}", job_id);
                if let Err(e) = nats.renew_lease(job_id, LEASE_EXTENSION_SECS).await {
                    warn!("Failed to renew lease for job {}: {}", job_id, e);
                    // Don't fail the job just because lease renewal failed
                    // Coordinator will handle lease expiry if needed
                }
            }
        }
    }

    // Wait for container to fully finish (if not aborted)
    if !cancelled {
        let _ = container_handle.await;
    }

    // Send final status
    if cancelled {
        nats.publish_status(job_id, "cancelled", None).await?;
    } else if let Some(reason) = failure_reason {
        nats.publish_status(job_id, "failed", Some(reason.clone()))
            .await?;
        warn!("Job {} failed: {}", job_id, reason);
    } else {
        nats.publish_output(job_id, seq + 1, "", true).await?;
        nats.publish_status(job_id, "succeeded", None).await?;
        info!("Job {} succeeded, generated {} tokens", job_id, token_count);
    }

    Ok(())
}

/// Detect NVIDIA GPU using nvidia-smi command
fn detect_nvidia_gpu() -> (Option<String>, Option<u32>) {
    // Try to run nvidia-smi to get GPU info
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.lines().next().unwrap_or("");

            // Parse "NVIDIA GeForce RTX 3080, 10240"
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                let gpu_model = Some(parts[0].to_string());
                let gpu_vram_mb = parts[1].parse::<u32>().ok();
                (gpu_model, gpu_vram_mb)
            } else {
                (None, None)
            }
        }
        Ok(_) => {
            debug!("nvidia-smi returned non-zero exit code");
            (None, None)
        }
        Err(e) => {
            debug!("nvidia-smi not available: {}", e);
            (None, None)
        }
    }
}
