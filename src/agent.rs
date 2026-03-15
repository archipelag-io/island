//! Main agent logic
//!
//! Coordinates NATS connection, job execution, and heartbeats.

use crate::cache::CacheManager;
use crate::config::AgentConfig;
use crate::docker::{self, ContainerConfig, ContainerOutput};
use crate::messages::WorkloadOutput;
use crate::metrics::collect_system_metrics;
use crate::metrics::gpu::GpuMetricsCollector;
use crate::metrics::gpu::bandwidth_for_gpu;
use crate::nats::{
    self, AssignJob, CacheMetricsSnapshot, CancelJob, GpuMetricsSnapshot, HostCapabilities,
    JobSubscription, NatsAgent, PerformanceEstimates, SystemMetricsSnapshot,
};
use crate::security::registry::RegistryAllowlist;
use crate::security::signing::SignatureVerifier;
use crate::state::StateManager;
use crate::update::UpdateChecker;
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
use tokio::sync::{mpsc, watch, RwLock};
use tracing::{debug, error, info, instrument, warn};

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
    docker: Option<Docker>,
    nats: NatsAgent,
    state: Arc<RwLock<StateManager>>,
    cache: Arc<CacheManager>,
    /// Signature verifier for workload images
    signature_verifier: Arc<SignatureVerifier>,
    /// Registry allowlist for container image validation
    registry_allowlist: Arc<RegistryAllowlist>,
    active_jobs: Arc<AtomicU32>,
    shutdown: Arc<AtomicBool>,
    /// Map of job_id -> cancel sender for running jobs
    job_cancellers: Arc<DashMap<String, watch::Sender<bool>>>,
}

impl Agent {
    /// Create a new agent
    pub async fn new(config: AgentConfig, docker: Option<Docker>) -> Result<Self> {
        // Load persistent state
        let mut state = StateManager::new().await?;

        // Use config host_id if set, otherwise get/create a stable one from state
        let host_id = if let Some(ref id) = config.host_id {
            id.clone()
        } else {
            state.get_or_create_host_id().await?
        };

        info!("Island ID: {}", host_id);
        info!("State loaded (paired: {})", state.is_paired());

        // Initialize cache manager for cold-start optimization (requires Docker)
        let cache = match docker {
            Some(ref d) => CacheManager::new(d.clone(), config.cache.clone()),
            None => CacheManager::new_without_docker(config.cache.clone()),
        };

        // Initialize signature verifier
        let mut signature_verifier = SignatureVerifier::new(config.signing.clone());

        // Check if cosign is available
        if signature_verifier.is_enabled() {
            if SignatureVerifier::cosign_available() {
                info!("Signature verification enabled (cosign available)");

                // Try to load keys from coordinator
                if let Err(e) = signature_verifier.load_keys_from_coordinator().await {
                    warn!("Failed to load signing keys from coordinator: {}", e);
                }

                // Try to load cached keys as fallback
                if signature_verifier.key_count() == 0 {
                    if let Err(e) = signature_verifier.load_keys_from_cache().await {
                        debug!("Failed to load cached signing keys: {}", e);
                    }
                }

                info!(
                    "Signature verifier initialized with {} trusted keys (required: {})",
                    signature_verifier.key_count(),
                    signature_verifier.is_required()
                );
            } else {
                warn!("Signature verification enabled but cosign not found - verification will be skipped");
            }
        } else {
            info!("Signature verification disabled");
        }

        // Initialize registry allowlist
        let registry_allowlist = if !config.registry.enabled {
            info!("Registry allowlist disabled");
            RegistryAllowlist::disabled()
        } else if config.registry.allowed.is_empty() {
            let allowlist = RegistryAllowlist::new()
                .with_require_digest(config.registry.require_digest);
            info!(
                "Registry allowlist enabled with defaults (require_digest: {})",
                config.registry.require_digest
            );
            allowlist
        } else {
            let allowlist = RegistryAllowlist::with_registries(config.registry.allowed.clone())
                .with_require_digest(config.registry.require_digest);
            info!(
                "Registry allowlist enabled: {} registries (require_digest: {})",
                config.registry.allowed.len(),
                config.registry.require_digest
            );
            allowlist
        };

        // Initialize model cache for GGUF/ONNX/diffusers models
        let model_cache = Arc::new(
            crate::model_cache::ModelCache::new(&config.model_cache)
                .context("Failed to initialize model cache")?
        );
        model_cache.init().await?;
        state.set_model_cache(model_cache);

        // Connect to NATS
        let nats = NatsAgent::connect(&config.coordinator.nats_url, host_id).await?;

        Ok(Self {
            config,
            docker,
            nats,
            state: Arc::new(RwLock::new(state)),
            cache: Arc::new(cache),
            signature_verifier: Arc::new(signature_verifier),
            registry_allowlist: Arc::new(registry_allowlist),
            active_jobs: Arc::new(AtomicU32::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            job_cancellers: Arc::new(DashMap::new()),
        })
    }

    /// Run the agent
    pub async fn run(&self) -> Result<()> {
        // Initialize cache (pre-pull configured images)
        if let Err(e) = self.cache.init().await {
            warn!("Cache initialization failed (non-fatal): {}", e);
        }

        // Detect capabilities once at startup
        let capabilities = self.detect_capabilities();

        // Preload starter models based on hardware (background task)
        {
            let starter_models = crate::preload::select_starter_models(
                capabilities.ram_mb,
                capabilities.gpu_vram_mb,
                &self.config.preload,
            );
            if !starter_models.is_empty() {
                let state = self.state.clone();
                tokio::spawn(async move {
                    let st = state.read().await;
                    if let Some(cache) = st.model_cache() {
                        crate::preload::preload_models(cache, starter_models).await;
                    }
                });
            }
        }

        // Initial registration
        self.register_and_setup(&capabilities).await?;

        // Request pairing if needed
        self.check_and_request_pairing().await;

        // Create channel for job completion notifications
        let (job_done_tx, mut job_done_rx) = mpsc::channel::<String>(32);

        // Subscribe to job assignments (JetStream with core NATS fallback)
        let mut job_subscriber = self.nats.subscribe_jobs().await?;
        info!("Job subscription established");

        // Subscribe to cancel requests
        let mut cancel_subscriber = self.nats.subscribe_cancel().await?;

        // Subscribe to peer probe requests (for peer-to-peer RTT measurement)
        let mut probe_subscriber = self.nats.subscribe_probes().await?;

        let mut consecutive_failures: u32 = 0;

        // GPU metrics collector
        let mut gpu_collector = GpuMetricsCollector::new();

        // Cache cleanup interval (every 5 minutes)
        let mut cache_cleanup_interval = tokio::time::interval(Duration::from_secs(300));
        cache_cleanup_interval.tick().await; // consume first immediate tick

        // Update checker (30 min interval; has internal rate limiting with jitter)
        let mut update_checker = match UpdateChecker::new(&self.config, self.nats.host_id().to_string()) {
            Ok(checker) => Some(checker),
            Err(e) => {
                warn!("Failed to initialize update checker: {}", e);
                None
            }
        };
        let mut update_check_interval = tokio::time::interval(Duration::from_secs(1800));
        update_check_interval.tick().await; // consume first immediate tick

        // Heartbeat interval
        let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(10));

        info!("Agent running. Waiting for jobs...");

        loop {
            select! {
                // Heartbeat tick
                _ = heartbeat_interval.tick() => {
                    let active = self.active_jobs.load(Ordering::Relaxed);

                    let gpu_metrics = gpu_collector.collect().map(|gpus| {
                        gpus.into_iter().map(|g| GpuMetricsSnapshot {
                            index: g.index,
                            utilization_percent: g.utilization_percent,
                            memory_used_mb: g.memory_used_mb,
                            memory_total_mb: g.memory_total_mb,
                            temperature_c: g.temperature_c,
                            power_draw_w: g.power_draw_w,
                        }).collect()
                    });

                    let sys = collect_system_metrics();
                    let system_snapshot = Some(SystemMetricsSnapshot {
                        cpu_percent: sys.cpu_percent,
                        memory_used_mb: sys.memory_used_mb,
                        memory_total_mb: sys.memory_total_mb,
                        disk_used_gb: sys.disk_used_gb,
                        disk_total_gb: sys.disk_total_gb,
                    });

                    let cache_stats = self.cache.get_stats().await;
                    let cache_snapshot = Some(CacheMetricsSnapshot {
                        cached_image_count: cache_stats.cached_image_count,
                        cached_size_mb: cache_stats.cached_size_mb,
                        warm_workload_count: cache_stats.warm_workload_count,
                        warm_workload_ids: cache_stats.warm_workload_ids,
                    });

                    let mut perf_estimates = self.compute_performance_estimates(&capabilities);

                    // Measure NATS RTT for pipeline formation scoring
                    perf_estimates.nats_rtt_ms = self.nats.measure_rtt().await;

                    if let Err(e) = self.nats.send_enhanced_heartbeat(
                        active, system_snapshot, gpu_metrics, None, cache_snapshot, Some(perf_estimates)
                    ).await {
                        warn!("Failed to send heartbeat: {}", e);
                        consecutive_failures += 1;

                        // If heartbeats are failing, subscription might be stale
                        if consecutive_failures >= 3 {
                            warn!("Multiple heartbeat failures, attempting to resubscribe...");
                            let mut core_sub = match self.nats.subscribe_jobs_core().await {
                                Ok(sub) => sub,
                                Err(e) => {
                                    error!("Failed to create recovery subscriber: {}", e);
                                    continue;
                                }
                            };
                            match self.recover_subscription(&capabilities, &mut core_sub).await {
                                Ok(()) => {
                                    job_subscriber = JobSubscription::Core(core_sub);
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
                            // Subscription closed - need to recover with core NATS
                            warn!("Job subscription closed, attempting to recover...");
                            // Recover into a core NATS subscriber
                            let mut core_sub = match self.nats.subscribe_jobs_core().await {
                                Ok(sub) => sub,
                                Err(e) => {
                                    error!("Failed initial recovery: {}", e);
                                    continue;
                                }
                            };
                            match self.recover_subscription(&capabilities, &mut core_sub).await {
                                Ok(()) => {
                                    job_subscriber = JobSubscription::Core(core_sub);
                                    info!("Successfully recovered subscription (core NATS)");
                                }
                                Err(e) => {
                                    error!("Failed to recover subscription after retries: {}", e);
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

                // Peer probe request — respond immediately for RTT measurement
                msg = probe_subscriber.next() => {
                    if let Some(msg) = msg {
                        if let Some(reply) = msg.reply {
                            let _ = self.nats.client().publish(reply, "pong".into()).await;
                        }
                    }
                }

                // Cache cleanup (every 5 minutes)
                _ = cache_cleanup_interval.tick() => {
                    self.cache.cleanup_stale().await;
                    let evicted = self.cache.evict_lru().await;
                    if evicted > 0 {
                        info!(evicted_count = evicted, "Cache LRU eviction completed");
                    }
                }

                // Update check (every 30 minutes)
                _ = update_check_interval.tick() => {
                    if let Some(ref mut checker) = update_checker {
                        match checker.check_for_update().await {
                            Ok(Some(info)) => {
                                warn!(
                                    current = %checker.current_version(),
                                    latest = ?info.latest_version,
                                    critical = info.is_critical,
                                    "Update available"
                                );
                                // Log only — auto-download deferred (placeholder signing key)
                            }
                            Ok(None) => {} // No update or rate-limited
                            Err(e) => debug!(error = %e, "Update check failed"),
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

            // Try to resubscribe (core NATS for recovery)
            match self.nats.subscribe_jobs_core().await {
                Ok(new_subscriber) => {
                    *subscriber = new_subscriber;
                    info!(
                        "Subscription recovered successfully after {} attempts",
                        attempts
                    );
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

    /// Detect Island capabilities
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

    /// Compute performance estimates from hardware capabilities.
    /// These are sent with each heartbeat so the coordinator can rank Islands.
    fn compute_performance_estimates(&self, capabilities: &HostCapabilities) -> PerformanceEstimates {
        let gpu_bw = capabilities.gpu_model.as_deref().map(|model| {
            bandwidth_for_gpu(Some(model), capabilities.gpu_vram_mb)
        });

        // Estimate LLM tok/s for a reference 7B Q4_K_M model (~4.5 GB)
        let reference_model_size_gb: f32 = 4.5;
        let estimated_llm_tok_s = gpu_bw.map(|bw| {
            let efficiency = 0.55;
            let mode_factor = match capabilities.gpu_vram_mb {
                Some(vram) if vram >= (reference_model_size_gb * 1024.0) as u32 => 1.0_f32,
                Some(_) => 0.5,
                None => 0.3,
            };
            (bw / reference_model_size_gb * efficiency * mode_factor).round()
        });

        // Max concurrent containers: min(CPU / 2, RAM / 2048) — each container gets at least 2 cores and 2 GB
        let max_containers = std::cmp::min(
            capabilities.cpu_cores / 2,
            capabilities.ram_mb / 2048,
        ).max(1);

        // WASM memory: allow up to 75% of RAM for WASM linear memory
        let wasm_mem = capabilities.ram_mb * 3 / 4;

        // Supported runtimes — container and wasm are always available
        #[allow(unused_mut)]
        let mut runtimes = vec!["container".to_string(), "wasm".to_string()];
        #[cfg(feature = "onnx")]
        runtimes.push("onnx".to_string());
        #[cfg(feature = "gguf")]
        runtimes.push("llmcpp".to_string());
        #[cfg(feature = "diffusers")]
        runtimes.push("diffusers".to_string());

        let estimates = PerformanceEstimates {
            gpu_bandwidth_gb_s: gpu_bw,
            estimated_llm_tok_s,
            max_concurrent_containers: Some(max_containers),
            wasm_memory_limit_mb: Some(wasm_mem),
            supported_runtimes: runtimes,
            nats_rtt_ms: None, // Populated during heartbeat
        };

        debug!(
            "Performance estimates: bandwidth={:?} GB/s, llm_tok_s={:?}, max_containers={}, wasm_mem={}MB",
            estimates.gpu_bandwidth_gb_s,
            estimates.estimated_llm_tok_s,
            max_containers,
            wasm_mem
        );

        estimates
    }

    /// Check if this Island needs pairing and request a pairing code if so
    async fn check_and_request_pairing(&self) {
        // Check if already paired locally
        {
            let state = self.state.read().await;
            if state.is_paired() {
                info!("Island is already paired (from local state)");
                return;
            }
        }

        match self.nats.request_pairing().await {
            Ok(response) => {
                if response.success {
                    if let Some(code) = response.code {
                        info!("========================================");
                        info!("       ISLAND PAIRING CODE: {}", code);
                        info!("========================================");
                        if let Some(url) = response.pair_url {
                            info!("Visit {} to pair this Island", url);
                        }
                        if let Some(expires) = response.expires_in_seconds {
                            info!("Code expires in {} minutes", expires / 60);
                        }
                        info!("========================================");
                    }
                } else if let Some(error) = response.error {
                    if error.contains("already paired") {
                        info!("Island is already paired to an account");
                        // Mark as paired in local state
                        let mut state = self.state.write().await;
                        if let Err(e) = state.set_paired(None).await {
                            warn!("Failed to save pairing state: {}", e);
                        }
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
        let state = self.state.clone();
        let cache = self.cache.clone();
        let signature_verifier = self.signature_verifier.clone();
        let registry_allowlist = self.registry_allowlist.clone();

        tokio::spawn(async move {
            let job_id = job.job_id.clone();
            let workload_id = job.workload_id.clone();
            let image = job
                .container_image
                .clone()
                .unwrap_or_else(|| config.workload.llm_chat_image.clone());

            let result = execute_job(
                &docker,
                &nats,
                &config,
                &state,
                &cache,
                &signature_verifier,
                &registry_allowlist,
                job,
                shutdown,
                cancel_rx,
            )
            .await;

            match &result {
                Ok(()) => {
                    // Record successful workload run for cache warmth tracking
                    if let Some(ref wid) = workload_id {
                        cache.record_workload_run(wid, &image).await;
                    }
                }
                Err(e) => {
                    error!("Job {} failed: {}", job_id, e);
                    let _ = nats
                        .publish_status(&job_id, "failed", Some(e.to_string()))
                        .await;
                }
            }

            let _ = done_tx.send(job_id).await;
        });
    }
}

/// Execute a single job (routes to container or WASM executor)
#[allow(clippy::too_many_arguments)]
#[instrument(
    skip(docker, nats, config, state, cache, signature_verifier, registry_allowlist, _shutdown, cancel_rx),
    fields(
        job_id = %job.job_id,
        workload_id = ?job.workload_id,
        runtime_type = %job.runtime_type,
    )
)]
async fn execute_job(
    docker: &Option<Docker>,
    nats: &NatsAgent,
    config: &AgentConfig,
    state: &Arc<RwLock<StateManager>>,
    cache: &Arc<CacheManager>,
    signature_verifier: &Arc<SignatureVerifier>,
    registry_allowlist: &Arc<RegistryAllowlist>,
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

    // Check for pipeline config first — pipeline jobs bypass normal runtime dispatch
    #[cfg(feature = "pipeline")]
    if let Some(ref pc_value) = job.pipeline_config {
        match serde_json::from_value::<crate::pipeline::PipelineConfig>(pc_value.clone()) {
            Ok(pipeline_config) => {
                return crate::pipeline::execute_pipeline_job(nats, state, &job, pipeline_config, cancel_rx).await;
            }
            Err(e) => {
                tracing::error!("Failed to parse pipeline_config: {}, falling back to normal dispatch", e);
            }
        }
    }

    // Route based on runtime type
    match job.runtime_type.as_str() {
        "wasm" => execute_wasm_job(nats, state, &job, cancel_rx).await,
        #[cfg(feature = "onnx")]
        "onnx" => crate::onnx::execute_onnx_job(nats, state, &job, cancel_rx).await,
        #[cfg(feature = "gguf")]
        "llmcpp" => crate::gguf::execute_gguf_job(nats, state, config, &job, cancel_rx).await,
        #[cfg(feature = "diffusers")]
        "diffusers" => crate::diffusers::execute_diffusers_job(nats, state, config, &job, cancel_rx).await,
        _ => {
            let docker = docker.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Job {} requires Docker but Docker is not available. Install Docker to run container workloads.",
                    job_id
                )
            })?;
            execute_container_job(
                docker,
                nats,
                config,
                cache,
                signature_verifier,
                registry_allowlist,
                &job,
                cancel_rx,
            )
            .await
        }
    }
}

/// Execute a WASM workload
async fn execute_wasm_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let wasm_url = job
        .wasm_url
        .as_ref()
        .context("WASM workload missing wasm_url")?;

    info!("Executing WASM workload: {}", wasm_url);

    // Get WASM module path (download and cache if URL, or use directly if local path)
    let wasm_path = if wasm_url.starts_with("http://") || wasm_url.starts_with("https://") {
        // Download and cache the WASM module
        let state_guard = state.read().await;
        let cached_path = state_guard
            .get_wasm_module(wasm_url, job.wasm_hash.as_deref())
            .await
            .context("Failed to download/cache WASM module")?;
        cached_path.to_string_lossy().to_string()
    } else {
        // Assume it's a local path
        wasm_url.clone()
    };

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
    let wasm_handle = tokio::spawn(async move { executor.run(wasm_config, output_tx).await });

    // Lease renewal interval (every 30 seconds)
    let mut lease_interval =
        tokio::time::interval(Duration::from_secs(LEASE_RENEWAL_INTERVAL_SECS));
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
            Some(format!(
                "Timeout: job exceeded {}s limit",
                DEFAULT_WASM_TIMEOUT_SECS
            )),
        )
        .await?;
        warn!("WASM job {} failed: timeout", job_id);
    } else if exit_code == 0 {
        nats.publish_status(job_id, "succeeded", None).await?;
        info!(
            "WASM job {} succeeded, generated {} tokens",
            job_id, token_count
        );
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
                            WorkloadOutput::Image {
                                data,
                                format,
                                width,
                                height,
                            } => {
                                info!("WASM image: {}x{} {}", width, height, format);
                                nats.publish_image(job_id, data, format, *width, *height, None)
                                    .await?;
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
#[allow(clippy::too_many_arguments)]
async fn execute_container_job(
    docker: &Docker,
    nats: &NatsAgent,
    config: &AgentConfig,
    cache: &Arc<CacheManager>,
    signature_verifier: &Arc<SignatureVerifier>,
    registry_allowlist: &Arc<RegistryAllowlist>,
    job: &AssignJob,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    // Use image from job assignment, fall back to config
    let image = job
        .container_image
        .clone()
        .unwrap_or_else(|| config.workload.llm_chat_image.clone());

    // Enforce registry allowlist before pulling or running the image
    if let Err(e) = registry_allowlist.check(&image) {
        error!("Registry allowlist rejected image {}: {}", image, e);
        anyhow::bail!("Image not allowed: {}", e);
    }

    info!("Executing container workload: {}", image);

    // Ensure image is available (pre-pull if not cached)
    let needed_pull = cache
        .ensure_image(&image)
        .await
        .context("Failed to ensure container image")?;
    if needed_pull {
        info!("Image {} was pulled (cold start)", image);
    } else {
        debug!("Image {} was already cached (warm start)", image);
    }

    // Prepare container config
    let input_json = serde_json::to_string(&job.input).context("Failed to serialize job input")?;

    // Build resource limits from config (can be overridden by sandbox_tier)
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

    // Get sandbox tier from job assignment (trust-level-based limits)
    let sandbox_tier = job.sandbox_tier.clone();

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
        sandbox_tier,
        seccomp_profile: None, // Applied by apply_sandbox_tier()
    };

    // Create channel for container output
    let (output_tx, mut output_rx) = mpsc::channel::<ContainerOutput>(256);

    // Spawn container runner with signature verification
    let docker_clone = docker.clone();
    let verifier = Some(signature_verifier.clone());
    let container_handle = tokio::spawn(async move {
        docker::run_verified_container(&docker_clone, container_config, verifier, output_tx).await
    });

    // Track output
    let mut seq: u64 = 0;
    let mut buffer = String::new();
    let mut token_count: u32 = 0;
    let mut streaming_started = false;
    let mut failure_reason: Option<String> = None;
    let mut cancelled = false;

    // Lease renewal interval (every 30 seconds)
    let mut lease_interval =
        tokio::time::interval(Duration::from_secs(LEASE_RENEWAL_INTERVAL_SECS));
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
