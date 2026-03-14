//! Metrics collection for the Island software.
//!
//! This module provides comprehensive metering for:
//! - GPU utilization (nvidia-smi)
//! - Container resource usage (Docker stats)
//! - Job performance tracking
//!
//! Metrics are collected and aggregated for inclusion in heartbeats.

pub mod container;
pub mod gpu;

use serde::{Deserialize, Serialize};

/// Aggregated system metrics for heartbeats
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// System-wide CPU usage percentage
    pub cpu_percent: f32,
    /// Memory used in MB
    pub memory_used_mb: u64,
    /// Memory total in MB
    pub memory_total_mb: u64,
    /// Disk used in GB on the primary disk
    pub disk_used_gb: u64,
    /// Disk total in GB
    pub disk_total_gb: u64,
    /// Network bytes received since boot
    pub network_rx_bytes: u64,
    /// Network bytes transmitted since boot
    pub network_tx_bytes: u64,
}

/// Job performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    /// Job ID
    pub job_id: String,
    /// Job type (e.g., "llm-chat", "image-gen")
    pub job_type: String,
    /// Start timestamp (Unix ms)
    pub started_at: u64,
    /// Duration in milliseconds (if completed)
    pub duration_ms: Option<u64>,
    /// Total tokens generated (for LLM jobs)
    pub tokens_generated: Option<u64>,
    /// Tokens per second (for LLM jobs)
    pub tokens_per_second: Option<f32>,
    /// Time to first token in ms (for streaming LLM jobs)
    pub time_to_first_token_ms: Option<u64>,
    /// Peak memory usage in MB
    pub peak_memory_mb: Option<u64>,
    /// Peak GPU memory usage in MB (if GPU job)
    pub peak_gpu_memory_mb: Option<u64>,
    /// Average GPU utilization percentage
    pub avg_gpu_utilization: Option<f32>,
    /// Exit code (0 = success)
    pub exit_code: Option<i32>,
    /// Error message if failed
    pub error: Option<String>,
}

impl JobMetrics {
    /// Create a new job metrics tracker
    pub fn new(job_id: String, job_type: String) -> Self {
        Self {
            job_id,
            job_type,
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            duration_ms: None,
            tokens_generated: None,
            tokens_per_second: None,
            time_to_first_token_ms: None,
            peak_memory_mb: None,
            peak_gpu_memory_mb: None,
            avg_gpu_utilization: None,
            exit_code: None,
            error: None,
        }
    }

    /// Mark the job as complete
    pub fn complete(&mut self, exit_code: i32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.duration_ms = Some(now.saturating_sub(self.started_at));
        self.exit_code = Some(exit_code);

        // Calculate tokens per second if we have token count
        if let (Some(tokens), Some(duration)) = (self.tokens_generated, self.duration_ms) {
            if duration > 0 {
                self.tokens_per_second = Some(tokens as f32 / (duration as f32 / 1000.0));
            }
        }
    }

    /// Mark the job as failed
    pub fn fail(&mut self, error: String) {
        self.complete(-1);
        self.error = Some(error);
    }

    /// Update peak memory usage
    pub fn update_peak_memory(&mut self, memory_mb: u64) {
        match self.peak_memory_mb {
            Some(peak) if memory_mb > peak => self.peak_memory_mb = Some(memory_mb),
            None => self.peak_memory_mb = Some(memory_mb),
            _ => {}
        }
    }

    /// Update peak GPU memory usage
    pub fn update_peak_gpu_memory(&mut self, memory_mb: u64) {
        match self.peak_gpu_memory_mb {
            Some(peak) if memory_mb > peak => self.peak_gpu_memory_mb = Some(memory_mb),
            None => self.peak_gpu_memory_mb = Some(memory_mb),
            _ => {}
        }
    }

    /// Record first token time
    pub fn record_first_token(&mut self) {
        if self.time_to_first_token_ms.is_none() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            self.time_to_first_token_ms = Some(now.saturating_sub(self.started_at));
        }
    }

    /// Increment token count
    pub fn add_tokens(&mut self, count: u64) {
        self.tokens_generated = Some(self.tokens_generated.unwrap_or(0) + count);
    }
}

/// Collect system-wide metrics using sysinfo
pub fn collect_system_metrics() -> SystemMetrics {
    use sysinfo::System;

    let mut sys = System::new();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    let cpu_percent = sys.global_cpu_usage();

    let memory_used_mb = sys.used_memory() / (1024 * 1024);
    let memory_total_mb = sys.total_memory() / (1024 * 1024);

    // Disk metrics
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    let (disk_used_gb, disk_total_gb) = disks
        .iter()
        .filter(|d| d.mount_point() == std::path::Path::new("/"))
        .map(|d| {
            let total = d.total_space() / (1024 * 1024 * 1024);
            let used = (d.total_space() - d.available_space()) / (1024 * 1024 * 1024);
            (used, total)
        })
        .next()
        .unwrap_or((0, 0));

    // Network metrics
    use sysinfo::Networks;
    let networks = Networks::new_with_refreshed_list();
    let (network_rx_bytes, network_tx_bytes) =
        networks.iter().fold((0u64, 0u64), |(rx, tx), (_, data)| {
            (rx + data.total_received(), tx + data.total_transmitted())
        });

    SystemMetrics {
        cpu_percent,
        memory_used_mb,
        memory_total_mb,
        disk_used_gb,
        disk_total_gb,
        network_rx_bytes,
        network_tx_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_metrics_new() {
        let metrics = JobMetrics::new("job-123".into(), "llm-chat".into());
        assert_eq!(metrics.job_id, "job-123");
        assert_eq!(metrics.job_type, "llm-chat");
        assert!(metrics.started_at > 0);
    }

    #[test]
    fn test_job_metrics_complete() {
        let mut metrics = JobMetrics::new("job-123".into(), "llm-chat".into());
        std::thread::sleep(std::time::Duration::from_millis(10));
        metrics.complete(0);

        assert!(metrics.duration_ms.is_some());
        assert!(metrics.duration_ms.unwrap() >= 10);
        assert_eq!(metrics.exit_code, Some(0));
    }

    #[test]
    fn test_job_metrics_tokens() {
        let mut metrics = JobMetrics::new("job-123".into(), "llm-chat".into());
        metrics.add_tokens(10);
        metrics.add_tokens(5);
        assert_eq!(metrics.tokens_generated, Some(15));
    }

    #[test]
    fn test_job_metrics_peak_memory() {
        let mut metrics = JobMetrics::new("job-123".into(), "llm-chat".into());
        metrics.update_peak_memory(100);
        metrics.update_peak_memory(50); // Should not update
        metrics.update_peak_memory(150); // Should update
        assert_eq!(metrics.peak_memory_mb, Some(150));
    }

    #[test]
    fn test_collect_system_metrics() {
        let metrics = collect_system_metrics();
        // Basic sanity checks
        assert!(metrics.memory_total_mb > 0);
        assert!(metrics.cpu_percent >= 0.0);
    }
}
