//! Main agent logic
//!
//! Coordinates NATS connection, job execution, and heartbeats.

use crate::config::AgentConfig;
use crate::docker::{self, ContainerConfig, ContainerOutput};
use crate::messages::WorkloadOutput;
use crate::nats::{self, AssignJob, HostCapabilities, NatsAgent};
use crate::wasm::{WasmConfig, WasmExecutor, WasmOutput};
use anyhow::{Context, Result};
use bollard::Docker;
use futures_util::StreamExt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::select;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// The main agent that coordinates all activity
pub struct Agent {
    config: AgentConfig,
    docker: Docker,
    nats: NatsAgent,
    active_jobs: Arc<AtomicU32>,
    shutdown: Arc<AtomicBool>,
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
        })
    }

    /// Run the agent
    pub async fn run(&self) -> Result<()> {
        // Register with coordinator
        let capabilities = self.detect_capabilities();
        self.nats.register(capabilities).await?;

        // Request pairing if needed
        self.check_and_request_pairing().await;

        // Subscribe to job assignments
        let mut job_subscriber = self.nats.subscribe_jobs().await?;

        // Create channel for job completion notifications
        let (job_done_tx, mut job_done_rx) = mpsc::channel::<String>(32);

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
                    }
                }

                // Job assignment received
                Some(msg) = job_subscriber.next() => {
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

                // Job completed
                Some(job_id) = job_done_rx.recv() => {
                    self.active_jobs.fetch_sub(1, Ordering::Relaxed);
                    info!("Job {} completed", job_id);
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

    /// Detect host capabilities
    fn detect_capabilities(&self) -> HostCapabilities {
        // TODO: Actually detect GPU, CPU, RAM
        // For now, use config or defaults
        HostCapabilities {
            gpu_model: Some("NVIDIA GeForce RTX".to_string()), // Placeholder
            gpu_vram_mb: Some(8192),
            cpu_cores: num_cpus::get() as u32,
            ram_mb: 16384, // Placeholder - should use sysinfo
            region: None,
        }
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

        let docker = self.docker.clone();
        let nats = self.nats.clone();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let job_id = job.job_id.clone();

            if let Err(e) = execute_job(&docker, &nats, &config, job, shutdown).await {
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
) -> Result<()> {
    let job_id = &job.job_id;

    // Notify started
    nats.publish_status(job_id, "started", None).await?;

    // Route based on runtime type
    match job.runtime_type.as_str() {
        "wasm" => execute_wasm_job(nats, &job).await,
        "container" | _ => execute_container_job(docker, nats, config, &job).await,
    }
}

/// Execute a WASM workload
async fn execute_wasm_job(nats: &NatsAgent, job: &AssignJob) -> Result<()> {
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

    // Process WASM output
    let (exit_code, token_count, timed_out) = process_wasm_output(nats, job_id, &mut output_rx).await?;

    // Wait for WASM to fully finish
    let _exit_code = wasm_handle.await??;

    // Send final status
    if timed_out {
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

/// Execute a container workload
async fn execute_container_job(
    docker: &Docker,
    nats: &NatsAgent,
    config: &AgentConfig,
    job: &AssignJob,
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

    let container_config = ContainerConfig {
        image,
        input: input_json,
        gpu_devices: config.workload.gpu_devices.clone(),
        timeout_seconds: DEFAULT_CONTAINER_TIMEOUT_SECS,
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
    let mut timed_out = false;

    // Process container output and forward to NATS
    while let Some(output) = output_rx.recv().await {
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
                                // Publish token to NATS
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
                timed_out = true;
                break;
            }
        }
    }

    // Wait for container to fully finish
    let exit_code = container_handle.await??;

    // Send final status
    if timed_out {
        nats.publish_status(
            job_id,
            "failed",
            Some(format!("Timeout: job exceeded {}s limit", DEFAULT_CONTAINER_TIMEOUT_SECS)),
        )
        .await?;
        warn!("Job {} failed: timeout", job_id);
    } else if exit_code == 0 {
        nats.publish_output(job_id, seq + 1, "", true).await?;
        nats.publish_status(job_id, "succeeded", None).await?;
        info!("Job {} succeeded, generated {} tokens", job_id, token_count);
    } else {
        nats.publish_status(job_id, "failed", Some(format!("Exit code: {}", exit_code)))
            .await?;
    }

    Ok(())
}
