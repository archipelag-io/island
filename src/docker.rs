//! Docker container management

use anyhow::{Context, Result};
use bollard::container::{
    Config, CreateContainerOptions, LogOutput, RemoveContainerOptions, StartContainerOptions,
    WaitContainerOptions,
};
use bollard::models::{DeviceRequest, HostConfig};
use bollard::Docker;
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tracing::{debug, info};

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
}

/// Output chunk from container
#[derive(Debug)]
pub enum ContainerOutput {
    Stdout(String),
    Stderr(String),
    Exit(i64),
}

/// Run a container and stream its output through a channel
pub async fn run_container_streaming(
    docker: &Docker,
    config: ContainerConfig,
    output_tx: mpsc::Sender<ContainerOutput>,
) -> Result<i64> {
    let container_name = format!("archipelag-job-{}", uuid::Uuid::new_v4());

    // Configure GPU access
    let device_requests = config.gpu_devices.map(|devices| {
        vec![DeviceRequest {
            driver: Some("nvidia".to_string()),
            device_ids: Some(devices),
            capabilities: Some(vec![vec!["gpu".to_string()]]),
            ..Default::default()
        }]
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

    debug!("Creating container: {}", container_name);
    let container = docker
        .create_container(Some(create_options), container_config)
        .await
        .context("Failed to create container")?;

    // Attach to container for stdin/stdout
    let attach_options = bollard::container::AttachContainerOptions::<String> {
        stdin: Some(true),
        stdout: Some(true),
        stderr: Some(true),
        stream: Some(true),
        ..Default::default()
    };

    let mut attached = docker
        .attach_container(&container.id, Some(attach_options))
        .await
        .context("Failed to attach to container")?;

    // Start container
    debug!("Starting container: {}", container_name);
    docker
        .start_container(&container.id, None::<StartContainerOptions<String>>)
        .await
        .context("Failed to start container")?;

    // Send input to stdin
    use tokio::io::AsyncWriteExt;
    attached
        .input
        .write_all(config.input.as_bytes())
        .await
        .context("Failed to write to container stdin")?;
    attached.input.shutdown().await?;

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

    let mut wait_stream = docker.wait_container(&container.id, Some(wait_options));
    let exit_code = if let Some(result) = wait_stream.next().await {
        result.context("Failed to wait for container")?.status_code
    } else {
        -1
    };

    // Send exit code
    let _ = output_tx.send(ContainerOutput::Exit(exit_code)).await;

    // Clean up container
    let remove_options = RemoveContainerOptions {
        force: true,
        ..Default::default()
    };
    docker
        .remove_container(&container.id, Some(remove_options))
        .await
        .context("Failed to remove container")?;

    debug!("Container {} finished with exit code {}", container_name, exit_code);

    Ok(exit_code)
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
            ContainerOutput::Exit(_) => break,
        }
    }

    runner.await?
}
