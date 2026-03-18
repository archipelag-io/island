//! Job execution logic

use crate::config::AgentConfig;
use crate::docker::{self, ContainerConfig};
use crate::messages::{ChatInput, WorkloadOutput};
use anyhow::{Context, Result};
use bollard::Docker;
use tracing::{error, info};

/// Compute memory limit in bytes from megabytes.
pub(crate) fn compute_memory_bytes(memory_mb: u64) -> i64 {
    (memory_mb * 1024 * 1024) as i64
}

/// Build tmpfs mounts map when read-only rootfs is enabled.
/// Returns `None` when `read_only_rootfs` is `false`.
pub(crate) fn build_tmpfs_mounts(
    read_only_rootfs: bool,
    tmpfs_size_mb: u64,
) -> Option<std::collections::HashMap<String, String>> {
    if read_only_rootfs {
        let mut mounts = std::collections::HashMap::new();
        mounts.insert(
            "/tmp".to_string(),
            format!("rw,noexec,nosuid,size={}m", tmpfs_size_mb),
        );
        Some(mounts)
    } else {
        None
    }
}

/// Convert a CPU percentage to a Docker CPU quota value (microseconds per 100ms period).
/// `None` input means no limit, yielding `None` output.
pub(crate) fn compute_cpu_quota(cpu_percent: Option<u64>) -> Option<i64> {
    cpu_percent.map(|percent| (percent * 1000) as i64)
}

/// Parse complete lines from a buffer of streaming output.
/// Returns the parsed `WorkloadOutput` items and the remaining (incomplete) buffer content.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn parse_output_lines(buffer: &str) -> (Vec<std::result::Result<WorkloadOutput, String>>, String) {
    let mut results = Vec::new();
    let mut remaining = buffer.to_string();

    while let Some(newline_pos) = remaining.find('\n') {
        let line = remaining[..newline_pos].to_string();
        remaining = remaining[newline_pos + 1..].to_string();

        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<WorkloadOutput>(&line) {
            Ok(output) => results.push(Ok(output)),
            Err(_) => results.push(Err(line)),
        }
    }

    (results, remaining)
}

/// Run a test job (for development/debugging)
pub async fn run_test_job(docker: &Docker, config: &AgentConfig, prompt: &str) -> Result<()> {
    info!("Preparing test job");

    // Create input
    let input = ChatInput {
        prompt: prompt.to_string(),
        max_tokens: Some(512),
        temperature: Some(0.7),
    };

    let input_json = serde_json::to_string(&input).context("Failed to serialize input")?;

    // Configure container with resource limits from config
    let limits = &config.workload.resource_limits;
    let memory_bytes = Some(compute_memory_bytes(limits.memory_mb));
    let tmpfs_mounts = build_tmpfs_mounts(limits.read_only_rootfs, limits.tmpfs_size_mb);
    let cpu_quota = compute_cpu_quota(limits.cpu_percent);

    let container_config = ContainerConfig {
        image: config.workload.llm_chat_image.clone(),
        input: input_json,
        gpu_devices: config.workload.gpu_devices.clone(),
        timeout_seconds: 300,  // 5 minute timeout for test jobs
        expected_digest: None, // No digest verification for test jobs
        memory_bytes,
        read_only_rootfs: limits.read_only_rootfs,
        tmpfs_mounts,
        cpu_quota,
        network_disabled: limits.network_disabled,
        sandbox_tier: None, // Test jobs bypass sandbox tier limits
        seccomp_profile: None, // No seccomp for test jobs
    };

    info!("Starting container: {}", container_config.image);

    // Buffer for accumulating output lines
    let mut buffer = String::new();

    // Run container and process output
    let exit_code = docker::run_container(docker, container_config, |chunk| {
        buffer.push_str(&chunk);

        // Process complete lines
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON line
            match serde_json::from_str::<WorkloadOutput>(&line) {
                Ok(output) => match output {
                    WorkloadOutput::Status { message } => {
                        info!("Status: {}", message);
                    }
                    WorkloadOutput::Token { content } => {
                        // Print tokens without newline for streaming effect
                        print!("{}", content);
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                    WorkloadOutput::Progress { step, total } => {
                        info!("Progress: {}/{}", step, total);
                    }
                    WorkloadOutput::Image {
                        data: _,
                        format,
                        width,
                        height,
                    } => {
                        info!("Image generated: {}x{} {}", width, height, format);
                    }
                    WorkloadOutput::Done { usage, seed } => {
                        println!(); // Final newline
                        if let Some(usage) = usage {
                            info!("Done. Tokens: {}", usage.completion_tokens.unwrap_or(0));
                        } else if let Some(s) = seed {
                            info!("Done. Seed: {}", s);
                        } else {
                            info!("Done.");
                        }
                    }
                    WorkloadOutput::Error { message } => {
                        error!("Workload error: {}", message);
                    }
                },
                Err(_) => {
                    // Not valid JSON, might be raw output
                    info!("Raw output: {}", line);
                }
            }
        }
    })
    .await?;

    if exit_code != 0 {
        error!("Container exited with code {}", exit_code);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- compute_memory_bytes tests ---

    #[test]
    fn test_memory_bytes_default_8gb() {
        // 8192 MB = 8GB = 8_589_934_592 bytes
        assert_eq!(compute_memory_bytes(8192), 8_589_934_592);
    }

    #[test]
    fn test_memory_bytes_small() {
        // 256 MB
        assert_eq!(compute_memory_bytes(256), 256 * 1024 * 1024);
    }

    #[test]
    fn test_memory_bytes_zero() {
        assert_eq!(compute_memory_bytes(0), 0);
    }

    // --- build_tmpfs_mounts tests ---

    #[test]
    fn test_tmpfs_mounts_when_read_only() {
        let mounts = build_tmpfs_mounts(true, 256);
        assert!(mounts.is_some());
        let mounts = mounts.unwrap();
        assert_eq!(mounts.len(), 1);
        assert_eq!(mounts.get("/tmp").unwrap(), "rw,noexec,nosuid,size=256m");
    }

    #[test]
    fn test_tmpfs_mounts_when_writable_rootfs() {
        let mounts = build_tmpfs_mounts(false, 256);
        assert!(mounts.is_none());
    }

    #[test]
    fn test_tmpfs_mounts_custom_size() {
        let mounts = build_tmpfs_mounts(true, 1024).unwrap();
        assert_eq!(mounts.get("/tmp").unwrap(), "rw,noexec,nosuid,size=1024m");
    }

    // --- compute_cpu_quota tests ---

    #[test]
    fn test_cpu_quota_none_means_no_limit() {
        assert_eq!(compute_cpu_quota(None), None);
    }

    #[test]
    fn test_cpu_quota_100_percent_is_one_cpu() {
        // 100% = 100_000 microseconds per 100ms period
        assert_eq!(compute_cpu_quota(Some(100)), Some(100_000));
    }

    #[test]
    fn test_cpu_quota_200_percent_is_two_cpus() {
        assert_eq!(compute_cpu_quota(Some(200)), Some(200_000));
    }

    #[test]
    fn test_cpu_quota_50_percent_is_half_cpu() {
        assert_eq!(compute_cpu_quota(Some(50)), Some(50_000));
    }

    // --- parse_output_lines tests ---

    #[test]
    fn test_parse_single_json_line() {
        let buffer = "{\"type\":\"status\",\"message\":\"Loading\"}\n";
        let (results, remaining) = parse_output_lines(buffer);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
        match results[0].as_ref().unwrap() {
            WorkloadOutput::Status { message } => assert_eq!(message, "Loading"),
            _ => panic!("Expected Status"),
        }
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_multiple_lines() {
        let buffer = "{\"type\":\"token\",\"content\":\"Hi\"}\n{\"type\":\"token\",\"content\":\" there\"}\n";
        let (results, remaining) = parse_output_lines(buffer);
        assert_eq!(results.len(), 2);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_incomplete_line_kept_in_buffer() {
        let buffer = "{\"type\":\"token\",\"content\":\"Hi\"}\n{\"type\":\"tok";
        let (results, remaining) = parse_output_lines(buffer);
        assert_eq!(results.len(), 1);
        assert_eq!(remaining, "{\"type\":\"tok");
    }

    #[test]
    fn test_parse_empty_lines_skipped() {
        let buffer = "\n  \n{\"type\":\"status\",\"message\":\"ok\"}\n\n";
        let (results, remaining) = parse_output_lines(buffer);
        assert_eq!(results.len(), 1);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_invalid_json_returns_err_with_raw_line() {
        let buffer = "not json at all\n";
        let (results, remaining) = parse_output_lines(buffer);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
        assert_eq!(results[0].as_ref().unwrap_err(), "not json at all");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_no_newline_returns_everything_as_remaining() {
        let buffer = "partial data";
        let (results, remaining) = parse_output_lines(buffer);
        assert!(results.is_empty());
        assert_eq!(remaining, "partial data");
    }
}
