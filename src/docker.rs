//! Docker container management
//!
//! Provides functions to run containers with streaming output,
//! timeout handling, and cleanup.

use anyhow::{Context, Result};
use bollard::container::{
    Config, CreateContainerOptions, KillContainerOptions, LogOutput, RemoveContainerOptions,
    StartContainerOptions, WaitContainerOptions,
};
use bollard::models::{DeviceRequest, HostConfig};
use bollard::Docker;
use futures_util::StreamExt;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Connect to the Docker daemon
pub async fn connect() -> Result<Docker> {
    let docker = Docker::connect_with_local_defaults()
        .context("Failed to connect to Docker daemon")?;

    // Verify connection
    let version = docker.version().await.context("Failed to get Docker version")?;
    info!(
        "Docker version: {}",
        version.version.unwrap_or_else(|| "unknown".to_string())
    );

    Ok(docker)
}

/// Container configuration for a workload
pub struct ContainerConfig {
    pub image: String,
    pub input: String,
    pub gpu_devices: Option<Vec<String>>,
    /// Timeout in seconds (default: 300 = 5 minutes)
    pub timeout_seconds: u64,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            image: String::new(),
            input: String::new(),
            gpu_devices: None,
            timeout_seconds: 300, // 5 minutes default
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
    Crashed { exit_code: i64, reason: String },
}

/// Run a container and stream its output through a channel.
///
/// The container will be killed if it exceeds the configured timeout.
/// Returns the exit code (or -1 for timeout).
pub async fn run_container_streaming(
    docker: &Docker,
    config: ContainerConfig,
    output_tx: mpsc::Sender<ContainerOutput>,
) -> Result<i64> {
    let container_name = format!("archipelag-job-{}", uuid::Uuid::new_v4());
    let timeout_duration = Duration::from_secs(config.timeout_seconds);

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

    // Create container
    let host_config = HostConfig {
        device_requests,
        // Read-only root filesystem for security
        // read_only_rootfs: Some(true),
        // Memory limit (8GB)
        memory: Some(8 * 1024 * 1024 * 1024),
        // No network access for workloads (security)
        // network_mode: Some("none".to_string()),
        ..Default::default()
    };

    let container_config = Config {
        image: Some(config.image.clone()),
        host_config: Some(host_config),
        open_stdin: Some(true),
        stdin_once: Some(true),  // Close stdin after first detach
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

    debug!("Creating container: {} (timeout: {}s)", container_name, config.timeout_seconds);
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

    // Start container
    debug!("Starting container: {}", container_name);
    docker
        .start_container(&container_id, None::<StartContainerOptions<String>>)
        .await
        .context("Failed to start container")?;

    // Small delay to let container process start before sending input
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Send input to stdin
    use tokio::io::AsyncWriteExt;
    debug!("Sending input to container: {} bytes", config.input.len());
    attached
        .input
        .write_all(config.input.as_bytes())
        .await
        .context("Failed to write to container stdin")?;
    // Ensure newline at end for JSON parsers
    attached.input.write_all(b"\n").await?;
    attached.input.shutdown().await?;
    debug!("Input sent and stdin closed");

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
                let _ = output_tx.send(ContainerOutput::Crashed {
                    exit_code: code,
                    reason,
                }).await;
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
    let runner = tokio::spawn(async move {
        run_container_streaming(&docker_clone, config, tx).await
    });

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
fn interpret_exit_code(code: i64) -> String {
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
            format!("Killed by signal {} ({})", code - 128, signal_name(code - 128))
        }
        _ => format!("Unknown error (code {})", code),
    }
}

/// Get signal name from number
fn signal_name(signal: i64) -> &'static str {
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
