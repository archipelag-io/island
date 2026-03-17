//! Docker container management
//!
//! Provides functions to run containers with streaming output,
//! timeout handling, cleanup, and signature verification.

use anyhow::{Context, Result};
use bollard::container::{
    Config, CreateContainerOptions, KillContainerOptions, LogOutput, RemoveContainerOptions,
    StartContainerOptions, WaitContainerOptions,
};
use bollard::models::{DeviceRequest, HostConfig};
use bollard::Docker;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::security::seccomp::{ProfileType, SeccompProfile};
use crate::security::signing::{SignatureResult, SignatureVerifier};

/// Connect to the Docker daemon
pub async fn connect() -> Result<Docker> {
    let docker =
        Docker::connect_with_local_defaults().context("Failed to connect to Docker daemon")?;

    // Verify connection
    let version = docker
        .version()
        .await
        .context("Failed to get Docker version")?;
    info!(
        "Docker version: {}",
        version.version.unwrap_or_else(|| "unknown".to_string())
    );

    Ok(docker)
}

/// Verify that an image's digest matches the expected digest.
///
/// The digest should be in the format "sha256:<hash>" or just "<hash>".
/// Returns Ok(()) if verification passes, Err if it fails or can't be verified.
pub async fn verify_image_digest(
    docker: &Docker,
    image: &str,
    expected_digest: &str,
) -> Result<()> {
    // Normalize expected digest (ensure it has sha256: prefix)
    let expected = if expected_digest.starts_with("sha256:") {
        expected_digest.to_string()
    } else {
        format!("sha256:{}", expected_digest)
    };

    // Inspect the image to get its digest
    let inspect = docker
        .inspect_image(image)
        .await
        .context("Failed to inspect image for digest verification")?;

    // Get the image ID (which is the digest)
    let image_id = inspect.id.unwrap_or_default();

    // Image ID format is "sha256:<hash>"
    if image_id == expected {
        info!(
            "Image digest verified: {}",
            &expected[..20.min(expected.len())]
        );
        return Ok(());
    }

    // Also check RepoDigests for images pulled with a tag
    if let Some(repo_digests) = inspect.repo_digests {
        for repo_digest in &repo_digests {
            // RepoDigests are in format "repo@sha256:<hash>"
            if let Some(digest_part) = repo_digest.split('@').nth(1) {
                if digest_part == expected {
                    info!(
                        "Image digest verified via repo digest: {}",
                        &expected[..20.min(expected.len())]
                    );
                    return Ok(());
                }
            }
        }
    }

    // Digest mismatch - this is a security violation
    warn!(
        "Image digest mismatch! Expected: {}, Got ID: {}",
        expected, image_id
    );
    anyhow::bail!(
        "Image digest verification failed: expected {}, got {}",
        expected,
        image_id
    )
}

/// Container configuration for a workload
pub struct ContainerConfig {
    pub image: String,
    pub input: String,
    pub gpu_devices: Option<Vec<String>>,
    /// Timeout in seconds (default: 300 = 5 minutes)
    pub timeout_seconds: u64,
    /// Expected image digest (sha256:...) for verification
    /// If provided, the agent will verify the pulled image matches this digest
    pub expected_digest: Option<String>,
    /// Memory limit in bytes (default: 8GB)
    pub memory_bytes: Option<i64>,
    /// Enable read-only root filesystem (default: true)
    pub read_only_rootfs: bool,
    /// Tmpfs mounts (e.g., {"/tmp": "rw,noexec,nosuid,size=256m"})
    pub tmpfs_mounts: Option<HashMap<String, String>>,
    /// CPU quota in microseconds per 100ms period (100000 = 1 CPU)
    /// None = no limit
    pub cpu_quota: Option<i64>,
    /// Disable network access for the container (default: true for security)
    /// When true, container runs with network_mode: "none"
    pub network_disabled: bool,
    /// Sandbox tier for trust-level-based resource limits
    /// Values: "restricted", "standard", "elevated"
    pub sandbox_tier: Option<String>,
    /// Seccomp profile JSON string for syscall filtering
    /// Applied via Docker SecurityOpt as "seccomp=<json>"
    pub seccomp_profile: Option<String>,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            image: String::new(),
            input: String::new(),
            gpu_devices: None,
            timeout_seconds: 300, // 5 minutes default
            expected_digest: None,
            memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB default
            read_only_rootfs: true,
            tmpfs_mounts: None,
            cpu_quota: None,
            network_disabled: true, // Secure by default: no network access
            sandbox_tier: Some("standard".to_string()),
            seccomp_profile: None,
        }
    }
}

impl ContainerConfig {
    /// Apply sandbox tier resource limits
    ///
    /// Sandbox tiers:
    /// - "restricted": 256MB RAM, 60s timeout, no network, no GPU
    /// - "standard": 1GB RAM, 300s timeout, no network
    /// - "elevated": 8GB RAM, 600s timeout, network allowed, GPU allowed
    pub fn apply_sandbox_tier(&mut self) {
        let profile_type = match self.sandbox_tier.as_deref() {
            Some("restricted") => {
                self.memory_bytes = Some(256 * 1024 * 1024); // 256MB
                self.timeout_seconds = 60;
                self.network_disabled = true;
                self.gpu_devices = None; // No GPU access
                self.cpu_quota = Some(100_000); // 1 CPU
                ProfileType::Minimal
            }
            Some("standard") => {
                self.memory_bytes = Some(1024 * 1024 * 1024); // 1GB
                self.timeout_seconds = 300;
                self.network_disabled = true;
                // GPU access if specified
                self.cpu_quota = Some(200_000); // 2 CPUs
                ProfileType::Default
            }
            Some("elevated") => {
                self.memory_bytes = Some(8 * 1024 * 1024 * 1024); // 8GB
                self.timeout_seconds = 600;
                self.network_disabled = false; // Network allowed
                // GPU access if specified
                self.cpu_quota = Some(400_000); // 4 CPUs
                // Use GPU profile if GPU devices are configured, otherwise Network
                if self.gpu_devices.is_some() {
                    ProfileType::Gpu
                } else {
                    ProfileType::Network
                }
            }
            _ => {
                // Default to standard if unknown
                debug!("Unknown sandbox tier, using standard defaults");
                ProfileType::Default
            }
        };

        // Apply seccomp profile for this tier
        match SeccompProfile::for_type(profile_type).to_json() {
            Ok(json) => {
                self.seccomp_profile = Some(json);
                debug!("Applied {:?} seccomp profile", profile_type);
            }
            Err(e) => {
                warn!("Failed to serialize seccomp profile: {}", e);
                // Continue without seccomp rather than failing the job
            }
        }
    }
}

/// Output chunk from container
#[derive(Debug)]
pub enum ContainerOutput {
    Stdout(String),
    Stderr(String),
    Exit(i64),
    /// Container was killed due to timeout
    Timeout,
    /// Container was killed due to OOM (out of memory)
    OomKilled,
    /// Container crashed with an error
    Crashed {
        exit_code: i64,
        reason: String,
    },
}

/// Run a container with full security verification and stream its output.
///
/// This function performs:
/// 1. Signature verification (if verifier is provided and enabled)
/// 2. Digest verification (if expected_digest is provided)
/// 3. Sandbox tier resource limits (if sandbox_tier is specified)
/// 4. Container execution with timeout
///
/// Returns the exit code (or -1 for timeout).
pub async fn run_verified_container(
    docker: &Docker,
    mut config: ContainerConfig,
    verifier: Option<Arc<SignatureVerifier>>,
    output_tx: mpsc::Sender<ContainerOutput>,
) -> Result<i64> {
    // Step 1: Verify signature if verifier is provided
    if let Some(ref verifier) = verifier {
        let image_ref = build_image_reference(&config.image, config.expected_digest.as_deref());

        match verifier.verify(&image_ref).await {
            Ok(SignatureResult::Valid { key_id, issuer }) => {
                info!(
                    "Signature verified for {} with key {} (issuer: {:?})",
                    config.image, key_id, issuer
                );
            }
            Ok(SignatureResult::Skipped) => {
                debug!("Signature verification skipped for {}", config.image);
            }
            Err(e) => {
                error!("Signature verification failed for {}: {}", config.image, e);
                anyhow::bail!("Signature verification failed: {}", e);
            }
        }
    }

    // Step 2: Apply sandbox tier resource limits
    if config.sandbox_tier.is_some() {
        config.apply_sandbox_tier();
        debug!(
            "Applied sandbox tier {:?}: memory={:?}MB, timeout={}s, network={}",
            config.sandbox_tier,
            config.memory_bytes.map(|b| b / 1024 / 1024),
            config.timeout_seconds,
            !config.network_disabled
        );
    }

    // Step 3: Run the container
    run_container_streaming(docker, config, output_tx).await
}

/// Build the full image reference with digest for verification
pub(crate) fn build_image_reference(image: &str, digest: Option<&str>) -> String {
    match digest {
        Some(d) if !d.is_empty() => {
            // If image already contains @sha256:, use as-is
            if image.contains('@') {
                image.to_string()
            } else {
                // Append digest
                let d = if d.starts_with("sha256:") {
                    d.to_string()
                } else {
                    format!("sha256:{}", d)
                };
                format!("{}@{}", image.split(':').next().unwrap_or(image), d)
            }
        }
        _ => image.to_string(),
    }
}

/// Run a container and stream its output through a channel.
///
/// The container will be killed if it exceeds the configured timeout.
/// Returns the exit code (or -1 for timeout).
///
/// Note: Prefer `run_verified_container` which includes signature verification.
pub async fn run_container_streaming(
    docker: &Docker,
    config: ContainerConfig,
    output_tx: mpsc::Sender<ContainerOutput>,
) -> Result<i64> {
    let container_name = format!("archipelag-job-{}", uuid::Uuid::new_v4());
    let timeout_duration = Duration::from_secs(config.timeout_seconds);

    // Verify image digest if one was specified
    if let Some(ref expected_digest) = config.expected_digest {
        verify_image_digest(docker, &config.image, expected_digest)
            .await
            .context("Image digest verification failed - refusing to execute")?;
    } else {
        debug!(
            "No expected digest provided, skipping verification for image: {}",
            config.image
        );
    }

    // Configure GPU access (only if devices are actually specified)
    let device_requests = config.gpu_devices.as_ref().and_then(|devices| {
        if devices.is_empty() {
            None
        } else {
            Some(vec![DeviceRequest {
                driver: Some("nvidia".to_string()),
                device_ids: Some(devices.clone()),
                capabilities: Some(vec![vec!["gpu".to_string()]]),
                ..Default::default()
            }])
        }
    });

    // Build seccomp security option if a profile is provided
    let security_opt = config.seccomp_profile.as_ref().map(|profile_json| {
        vec![format!("seccomp={}", profile_json)]
    });

    // Create container with resource limits
    let host_config = HostConfig {
        device_requests,
        // Memory limit
        memory: config.memory_bytes,
        // Read-only root filesystem for security
        readonly_rootfs: Some(config.read_only_rootfs),
        // Tmpfs mounts (e.g., /tmp for writable temp space when rootfs is read-only)
        tmpfs: config.tmpfs_mounts.clone(),
        // CPU quota (microseconds per 100ms period)
        cpu_quota: config.cpu_quota,
        // Network isolation: disable network access for workloads (security)
        network_mode: if config.network_disabled {
            Some("none".to_string())
        } else {
            None
        },
        // Seccomp profile for syscall filtering
        security_opt,
        ..Default::default()
    };

    if config.read_only_rootfs {
        debug!(
            "Container will use read-only rootfs{}",
            if config.tmpfs_mounts.is_some() {
                " with tmpfs mounts"
            } else {
                ""
            }
        );
    }

    if config.network_disabled {
        debug!("Container network access disabled (network_mode: none)");
    }

    if config.seccomp_profile.is_some() {
        debug!("Container seccomp profile applied");
    }

    let container_config = Config {
        image: Some(config.image.clone()),
        host_config: Some(host_config),
        open_stdin: Some(true),
        stdin_once: Some(true), // Close stdin after first detach
        attach_stdin: Some(true),
        attach_stdout: Some(true),
        attach_stderr: Some(true),
        tty: Some(false),
        ..Default::default()
    };

    let create_options = CreateContainerOptions {
        name: &container_name,
        platform: None,
    };

    debug!(
        "Creating container: {} (timeout: {}s)",
        container_name, config.timeout_seconds
    );
    let container = docker
        .create_container(Some(create_options), container_config)
        .await
        .context("Failed to create container")?;

    let container_id = container.id.clone();

    // Attach to container for stdin/stdout
    let attach_options = bollard::container::AttachContainerOptions::<String> {
        stdin: Some(true),
        stdout: Some(true),
        stderr: Some(true),
        stream: Some(true),
        ..Default::default()
    };

    let mut attached = docker
        .attach_container(&container_id, Some(attach_options))
        .await
        .context("Failed to attach to container")?;

    // Send input to stdin BEFORE starting the container.
    // This ensures the data is in the pipe buffer when the container process starts reading.
    use tokio::io::AsyncWriteExt;
    debug!("Sending input to container: {} bytes", config.input.len());
    attached
        .input
        .write_all(config.input.as_bytes())
        .await
        .context("Failed to write to container stdin")?;
    attached.input.write_all(b"\n").await?;
    attached.input.shutdown().await?;
    debug!("Input sent and stdin closed");

    // Start container (input is already in the pipe buffer)
    debug!("Starting container: {}", container_name);
    docker
        .start_container(&container_id, None::<StartContainerOptions<String>>)
        .await
        .context("Failed to start container")?;

    // Run container with timeout
    let run_result = timeout(timeout_duration, async {
        // Read output and send to channel
        while let Some(output) = attached.output.next().await {
            match output {
                Ok(LogOutput::StdOut { message }) => {
                    if let Ok(text) = String::from_utf8(message.to_vec()) {
                        let _ = output_tx.send(ContainerOutput::Stdout(text)).await;
                    }
                }
                Ok(LogOutput::StdErr { message }) => {
                    if let Ok(text) = String::from_utf8(message.to_vec()) {
                        debug!("Container stderr: {}", text);
                        let _ = output_tx.send(ContainerOutput::Stderr(text)).await;
                    }
                }
                Err(e) => {
                    debug!("Error reading container output: {}", e);
                }
                _ => {}
            }
        }

        // Wait for container to finish
        let wait_options = WaitContainerOptions {
            condition: "not-running",
        };

        let mut wait_stream = docker.wait_container(&container_id, Some(wait_options));
        if let Some(result) = wait_stream.next().await {
            match result {
                Ok(r) => r.status_code,
                Err(e) => {
                    warn!("Failed to wait for container: {}", e);
                    -1
                }
            }
        } else {
            -1
        }
    })
    .await;

    let exit_code = match run_result {
        Ok(code) => {
            // Inspect container to get detailed exit info
            let (final_code, oom_killed) = inspect_container_exit(docker, &container_id).await;
            let code = final_code.unwrap_or(code);

            if oom_killed {
                warn!("Container {} was OOM killed", container_name);
                let _ = output_tx.send(ContainerOutput::OomKilled).await;
            } else if code != 0 {
                // Non-zero exit - determine crash reason
                let reason = interpret_exit_code(code);
                warn!(
                    "Container {} crashed with exit code {}: {}",
                    container_name, code, reason
                );
                let _ = output_tx
                    .send(ContainerOutput::Crashed {
                        exit_code: code,
                        reason,
                    })
                    .await;
            } else {
                // Normal completion
                let _ = output_tx.send(ContainerOutput::Exit(code)).await;
            }
            code
        }
        Err(_) => {
            // Timeout - kill the container
            warn!(
                "Container {} exceeded timeout ({}s), killing...",
                container_name, config.timeout_seconds
            );
            let _ = output_tx.send(ContainerOutput::Timeout).await;

            // Kill the container
            if let Err(e) = kill_container(docker, &container_id).await {
                warn!("Failed to kill container: {}", e);
            }

            -1 // Return -1 for timeout
        }
    };

    // Clean up container
    let remove_options = RemoveContainerOptions {
        force: true,
        ..Default::default()
    };
    if let Err(e) = docker
        .remove_container(&container_id, Some(remove_options))
        .await
    {
        warn!("Failed to remove container: {}", e);
    }

    debug!(
        "Container {} finished with exit code {}",
        container_name, exit_code
    );

    Ok(exit_code)
}

/// Kill a running container
pub async fn kill_container(docker: &Docker, container_id: &str) -> Result<()> {
    let kill_options = KillContainerOptions { signal: "SIGKILL" };

    docker
        .kill_container(container_id, Some(kill_options))
        .await
        .context("Failed to kill container")?;

    info!("Killed container {}", container_id);
    Ok(())
}

/// Run a container with a sync callback (for backwards compatibility)
pub async fn run_container<F>(
    docker: &Docker,
    config: ContainerConfig,
    mut on_output: F,
) -> Result<i64>
where
    F: FnMut(String),
{
    let (tx, mut rx) = mpsc::channel(256);

    // Spawn the container runner
    let docker_clone = docker.clone();
    let runner =
        tokio::spawn(async move { run_container_streaming(&docker_clone, config, tx).await });

    // Process output
    while let Some(output) = rx.recv().await {
        match output {
            ContainerOutput::Stdout(text) => on_output(text),
            ContainerOutput::Stderr(_) => {}
            ContainerOutput::Exit(_)
            | ContainerOutput::Timeout
            | ContainerOutput::OomKilled
            | ContainerOutput::Crashed { .. } => break,
        }
    }

    runner.await?
}

/// Inspect a container to get detailed exit information
async fn inspect_container_exit(docker: &Docker, container_id: &str) -> (Option<i64>, bool) {
    match docker.inspect_container(container_id, None).await {
        Ok(info) => {
            let state = info.state.unwrap_or_default();
            let exit_code = state.exit_code;
            let oom_killed = state.oom_killed.unwrap_or(false);

            debug!(
                "Container exit inspection: code={:?}, oom_killed={}",
                exit_code, oom_killed
            );

            (exit_code, oom_killed)
        }
        Err(e) => {
            debug!("Failed to inspect container: {}", e);
            (None, false)
        }
    }
}

/// Interpret container exit code to human-readable reason
pub(crate) fn interpret_exit_code(code: i64) -> String {
    match code {
        1 => "General error (application failure)".to_string(),
        2 => "Misuse of shell command or incorrect arguments".to_string(),
        126 => "Command not executable (permission denied or not a binary)".to_string(),
        127 => "Command not found".to_string(),
        128 => "Invalid exit argument".to_string(),
        // Signals: 128 + signal number
        129 => "SIGHUP (hangup)".to_string(),
        130 => "SIGINT (interrupt from keyboard, Ctrl+C)".to_string(),
        131 => "SIGQUIT (quit from keyboard)".to_string(),
        132 => "SIGILL (illegal instruction)".to_string(),
        133 => "SIGTRAP (trace/breakpoint trap)".to_string(),
        134 => "SIGABRT (abort signal)".to_string(),
        135 => "SIGBUS (bus error)".to_string(),
        136 => "SIGFPE (floating-point exception)".to_string(),
        137 => "SIGKILL (killed, possibly by OOM killer or external signal)".to_string(),
        139 => "SIGSEGV (segmentation fault)".to_string(),
        141 => "SIGPIPE (broken pipe)".to_string(),
        143 => "SIGTERM (terminated)".to_string(),
        // Common application exit codes
        255 => "Exit status out of range or SSH error".to_string(),
        _ if code > 128 && code < 192 => {
            format!(
                "Killed by signal {} ({})",
                code - 128,
                signal_name(code - 128)
            )
        }
        _ => format!("Unknown error (code {})", code),
    }
}

/// Get signal name from number
pub(crate) fn signal_name(signal: i64) -> &'static str {
    match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        4 => "SIGILL",
        5 => "SIGTRAP",
        6 => "SIGABRT",
        7 => "SIGBUS",
        8 => "SIGFPE",
        9 => "SIGKILL",
        10 => "SIGUSR1",
        11 => "SIGSEGV",
        12 => "SIGUSR2",
        13 => "SIGPIPE",
        14 => "SIGALRM",
        15 => "SIGTERM",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_image_reference ──────────────────────────────────────

    #[test]
    fn build_image_reference_without_digest() {
        let result = build_image_reference("myregistry/myimage:latest", None);
        assert_eq!(result, "myregistry/myimage:latest");
    }

    #[test]
    fn build_image_reference_with_empty_digest() {
        let result = build_image_reference("myregistry/myimage:latest", Some(""));
        assert_eq!(result, "myregistry/myimage:latest");
    }

    #[test]
    fn build_image_reference_with_sha256_prefixed_digest() {
        let result = build_image_reference(
            "myregistry/myimage:latest",
            Some("sha256:abcdef1234567890"),
        );
        assert_eq!(result, "myregistry/myimage@sha256:abcdef1234567890");
    }

    #[test]
    fn build_image_reference_with_bare_hex_digest() {
        let result = build_image_reference("myregistry/myimage:latest", Some("abcdef1234567890"));
        assert_eq!(result, "myregistry/myimage@sha256:abcdef1234567890");
    }

    #[test]
    fn build_image_reference_already_contains_digest() {
        let image = "myregistry/myimage@sha256:existingdigest";
        let result = build_image_reference(image, Some("sha256:otherdigest"));
        // Should keep the original image reference unchanged
        assert_eq!(result, image);
    }

    // ── interpret_exit_code ────────────────────────────────────────

    #[test]
    fn interpret_exit_code_zero_is_unknown() {
        // Exit code 0 is not explicitly handled (it means success, not a crash)
        let result = interpret_exit_code(0);
        assert!(result.contains("Unknown error"));
    }

    #[test]
    fn interpret_exit_code_known_codes() {
        assert!(interpret_exit_code(1).contains("General error"));
        assert!(interpret_exit_code(2).contains("Misuse of shell"));
        assert!(interpret_exit_code(126).contains("not executable"));
        assert!(interpret_exit_code(127).contains("not found"));
        assert!(interpret_exit_code(128).contains("Invalid exit"));
    }

    #[test]
    fn interpret_exit_code_signal_137_sigkill() {
        let result = interpret_exit_code(137);
        assert!(result.contains("SIGKILL"));
        assert!(result.contains("OOM"));
    }

    #[test]
    fn interpret_exit_code_signal_139_sigsegv() {
        let result = interpret_exit_code(139);
        assert!(result.contains("SIGSEGV"));
    }

    #[test]
    fn interpret_exit_code_signal_143_sigterm() {
        let result = interpret_exit_code(143);
        assert!(result.contains("SIGTERM"));
    }

    #[test]
    fn interpret_exit_code_unknown_signal_range() {
        // Code 138 = 128 + 10 = SIGUSR1, falls through to the range match
        let result = interpret_exit_code(138);
        assert!(result.contains("signal 10"));
        assert!(result.contains("SIGUSR1"));
    }

    #[test]
    fn interpret_exit_code_unknown_code() {
        let result = interpret_exit_code(42);
        assert!(result.contains("Unknown error"));
        assert!(result.contains("42"));
    }

    // ── signal_name ────────────────────────────────────────────────

    #[test]
    fn signal_name_known_signals() {
        assert_eq!(signal_name(1), "SIGHUP");
        assert_eq!(signal_name(2), "SIGINT");
        assert_eq!(signal_name(9), "SIGKILL");
        assert_eq!(signal_name(11), "SIGSEGV");
        assert_eq!(signal_name(15), "SIGTERM");
    }

    #[test]
    fn signal_name_unknown_signal() {
        assert_eq!(signal_name(99), "unknown");
        assert_eq!(signal_name(0), "unknown");
    }

    // ── ContainerConfig defaults ───────────────────────────────────

    #[test]
    fn container_config_defaults_are_security_sane() {
        let config = ContainerConfig::default();

        assert!(config.network_disabled, "network should be disabled by default");
        assert!(config.read_only_rootfs, "rootfs should be read-only by default");
        assert_eq!(config.timeout_seconds, 300);
        assert_eq!(config.memory_bytes, Some(8 * 1024 * 1024 * 1024)); // 8GB
        assert!(config.gpu_devices.is_none(), "no GPU by default");
        assert!(config.expected_digest.is_none());
        assert_eq!(config.sandbox_tier, Some("standard".to_string()));
        assert!(config.seccomp_profile.is_none());
    }

    // ── apply_sandbox_tier ─────────────────────────────────────────

    #[test]
    fn apply_sandbox_tier_restricted() {
        let mut config = ContainerConfig {
            sandbox_tier: Some("restricted".to_string()),
            gpu_devices: Some(vec!["0".to_string()]),
            ..Default::default()
        };
        config.apply_sandbox_tier();

        assert_eq!(config.memory_bytes, Some(256 * 1024 * 1024));
        assert_eq!(config.timeout_seconds, 60);
        assert!(config.network_disabled);
        assert!(config.gpu_devices.is_none(), "restricted tier strips GPU");
        assert_eq!(config.cpu_quota, Some(100_000));
        assert!(config.seccomp_profile.is_some());
    }

    #[test]
    fn apply_sandbox_tier_standard() {
        let mut config = ContainerConfig {
            sandbox_tier: Some("standard".to_string()),
            ..Default::default()
        };
        config.apply_sandbox_tier();

        assert_eq!(config.memory_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(config.timeout_seconds, 300);
        assert!(config.network_disabled);
        assert_eq!(config.cpu_quota, Some(200_000));
        assert!(config.seccomp_profile.is_some());
    }

    #[test]
    fn apply_sandbox_tier_elevated_with_gpu() {
        let mut config = ContainerConfig {
            sandbox_tier: Some("elevated".to_string()),
            gpu_devices: Some(vec!["0".to_string()]),
            ..Default::default()
        };
        config.apply_sandbox_tier();

        assert_eq!(config.memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(config.timeout_seconds, 600);
        assert!(!config.network_disabled, "elevated tier allows network");
        assert_eq!(config.cpu_quota, Some(400_000));
        assert!(config.seccomp_profile.is_some());
    }

    #[test]
    fn apply_sandbox_tier_elevated_without_gpu() {
        let mut config = ContainerConfig {
            sandbox_tier: Some("elevated".to_string()),
            gpu_devices: None,
            ..Default::default()
        };
        config.apply_sandbox_tier();

        assert!(!config.network_disabled);
        assert!(config.seccomp_profile.is_some());
    }

    #[test]
    fn apply_sandbox_tier_unknown_defaults_to_standard_seccomp() {
        let mut config = ContainerConfig {
            sandbox_tier: Some("unknown_tier".to_string()),
            ..Default::default()
        };
        let original_memory = config.memory_bytes;
        let original_timeout = config.timeout_seconds;
        config.apply_sandbox_tier();

        // Unknown tier only applies the default seccomp profile; does NOT override memory/timeout
        assert_eq!(config.memory_bytes, original_memory);
        assert_eq!(config.timeout_seconds, original_timeout);
        assert!(config.seccomp_profile.is_some());
    }
}
