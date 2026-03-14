//! OCI runtime wrapper (invokes crun/runc).
//!
//! Generates an OCI runtime specification (config.json) and invokes
//! the runtime binary to create and run containers.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::docker::ContainerOutput;
use super::BundleConfig;

/// Generate an OCI runtime config.json in the bundle directory
pub fn generate_config(bundle_dir: &Path, config: &BundleConfig) -> Result<()> {
    let rootfs_dir = bundle_dir.join("rootfs");

    // Build the OCI runtime spec
    let spec = serde_json::json!({
        "ociVersion": "1.0.2",
        "process": {
            "terminal": false,
            "user": { "uid": 0, "gid": 0 },
            "args": ["/bin/sh", "-c", "cat /input.json | /entrypoint 2>/proc/self/fd/2"],
            "env": [
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "TERM=xterm"
            ],
            "cwd": "/",
            "capabilities": {
                "bounding": [],
                "effective": [],
                "inheritable": [],
                "permitted": [],
                "ambient": []
            },
            "rlimits": [
                { "type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024 }
            ],
            "noNewPrivileges": true
        },
        "root": {
            "path": rootfs_dir.to_string_lossy(),
            "readonly": config.read_only_rootfs
        },
        "hostname": "archipelag",
        "mounts": build_mounts(config),
        "linux": {
            "resources": build_resources(config),
            "namespaces": build_namespaces(config),
            "maskedPaths": [
                "/proc/acpi", "/proc/asound", "/proc/kcore", "/proc/keys",
                "/proc/latency_stats", "/proc/timer_list", "/proc/timer_stats",
                "/proc/sched_debug", "/sys/firmware", "/proc/scsi"
            ],
            "readonlyPaths": [
                "/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger"
            ]
        }
    });

    let config_path = bundle_dir.join("config.json");
    let json = serde_json::to_string_pretty(&spec)?;
    std::fs::write(&config_path, &json)?;

    debug!("Generated OCI config: {}", config_path.display());
    Ok(())
}

fn build_mounts(config: &BundleConfig) -> serde_json::Value {
    let mut mounts = vec![
        serde_json::json!({
            "destination": "/proc",
            "type": "proc",
            "source": "proc"
        }),
        serde_json::json!({
            "destination": "/dev",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]
        }),
        serde_json::json!({
            "destination": "/dev/pts",
            "type": "devpts",
            "source": "devpts",
            "options": ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"]
        }),
        serde_json::json!({
            "destination": "/dev/shm",
            "type": "tmpfs",
            "source": "shm",
            "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=65536k"]
        }),
        serde_json::json!({
            "destination": "/dev/mqueue",
            "type": "mqueue",
            "source": "mqueue",
            "options": ["nosuid", "noexec", "nodev"]
        }),
        serde_json::json!({
            "destination": "/sys",
            "type": "sysfs",
            "source": "sysfs",
            "options": ["nosuid", "noexec", "nodev", "ro"]
        }),
    ];

    // Writable /tmp via tmpfs
    if config.read_only_rootfs {
        mounts.push(serde_json::json!({
            "destination": "/tmp",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": [
                "nosuid", "nodev", "mode=1777",
                format!("size={}m", config.tmpfs_size_mb)
            ]
        }));
    }

    // Bind-mount the input file
    mounts.push(serde_json::json!({
        "destination": "/input.json",
        "type": "bind",
        "source": "input.json",
        "options": ["rbind", "ro"]
    }));

    serde_json::Value::Array(mounts)
}

fn build_resources(config: &BundleConfig) -> serde_json::Value {
    let mut resources = serde_json::json!({});

    if let Some(memory_bytes) = config.memory_bytes {
        resources["memory"] = serde_json::json!({
            "limit": memory_bytes,
            "swap": memory_bytes  // No swap
        });
    }

    if let Some(cpu_quota) = config.cpu_quota {
        resources["cpu"] = serde_json::json!({
            "quota": cpu_quota,
            "period": 100000  // 100ms
        });
    }

    // Disable all devices by default
    resources["devices"] = serde_json::json!([
        { "allow": false, "access": "rwm" }
    ]);

    resources
}

fn build_namespaces(config: &BundleConfig) -> serde_json::Value {
    let mut namespaces = vec![
        serde_json::json!({ "type": "pid" }),
        serde_json::json!({ "type": "ipc" }),
        serde_json::json!({ "type": "uts" }),
        serde_json::json!({ "type": "mount" }),
        serde_json::json!({ "type": "cgroup" }),
    ];

    if config.network_disabled {
        namespaces.push(serde_json::json!({ "type": "network" }));
    }

    serde_json::Value::Array(namespaces)
}

/// Run a container using the OCI runtime (crun/runc)
pub async fn run(
    runtime_path: &Path,
    container_id: &str,
    bundle_dir: &Path,
    timeout_secs: u64,
    output_tx: mpsc::Sender<ContainerOutput>,
) -> Result<i64> {
    info!("Running container {} via {}", container_id, runtime_path.display());

    let mut child = tokio::process::Command::new(runtime_path)
        .arg("run")
        .arg("--bundle")
        .arg(bundle_dir)
        .arg(container_id)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn OCI runtime")?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let tx_stdout = output_tx.clone();
    let tx_stderr = output_tx.clone();

    // Stream stdout
    let stdout_handle = tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = tx_stdout.send(ContainerOutput::Stdout(line + "\n")).await;
        }
    });

    // Stream stderr
    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            debug!("Container stderr: {}", line);
            let _ = tx_stderr.send(ContainerOutput::Stderr(line + "\n")).await;
        }
    });

    // Wait with timeout
    let timeout = tokio::time::Duration::from_secs(timeout_secs);
    let result = tokio::time::timeout(timeout, child.wait()).await;

    match result {
        Ok(Ok(status)) => {
            let code = status.code().unwrap_or(-1) as i64;
            stdout_handle.abort();
            stderr_handle.abort();

            if code == 0 {
                let _ = output_tx.send(ContainerOutput::Exit(code)).await;
            } else {
                let _ = output_tx
                    .send(ContainerOutput::Crashed {
                        exit_code: code,
                        reason: format!("Container exited with code {}", code),
                    })
                    .await;
            }

            // Clean up the container state
            cleanup_container(runtime_path, container_id).await;

            Ok(code)
        }
        Ok(Err(e)) => {
            error!("Runtime process error: {}", e);
            stdout_handle.abort();
            stderr_handle.abort();
            cleanup_container(runtime_path, container_id).await;
            anyhow::bail!("OCI runtime failed: {}", e)
        }
        Err(_) => {
            // Timeout — kill the container
            warn!("Container {} timed out after {}s", container_id, timeout_secs);
            let _ = output_tx.send(ContainerOutput::Timeout).await;

            // Kill the process
            let _ = child.kill().await;
            stdout_handle.abort();
            stderr_handle.abort();
            cleanup_container(runtime_path, container_id).await;

            Ok(-1)
        }
    }
}

/// Clean up container state after execution
async fn cleanup_container(runtime_path: &Path, container_id: &str) {
    let _ = tokio::process::Command::new(runtime_path)
        .arg("delete")
        .arg("--force")
        .arg(container_id)
        .output()
        .await;
}
