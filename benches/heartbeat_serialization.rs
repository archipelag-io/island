//! Benchmarks for EnhancedHeartbeat serialization — runs every 10 seconds.
//!
//! The heartbeat is the most frequently serialized outbound message. Measuring
//! serialization cost ensures we're not spending significant CPU budget on
//! telemetry encoding.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Re-define the types here since they're in a binary crate.
// These must stay in sync with src/nats.rs.
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct EnhancedHeartbeat {
    pub host_id: String,
    pub status: String,
    pub active_jobs: u32,
    pub timestamp: i64,
    pub agent_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<GpuMetricsSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_job_metrics: Option<Vec<ActiveJobMetrics>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asking_prices: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct SystemMetricsSnapshot {
    pub cpu_percent: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub disk_used_gb: u64,
    pub disk_total_gb: u64,
}

#[derive(Debug, Serialize)]
pub struct GpuMetricsSnapshot {
    pub index: u32,
    pub utilization_percent: u32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub temperature_c: u32,
    pub power_draw_w: f32,
}

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

#[derive(Debug, Serialize)]
pub struct CacheMetricsSnapshot {
    pub cached_image_count: usize,
    pub cached_size_mb: u64,
    pub warm_workload_count: usize,
    pub warm_workload_ids: Vec<String>,
}

fn make_minimal_heartbeat() -> EnhancedHeartbeat {
    EnhancedHeartbeat {
        host_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        status: "online".to_string(),
        active_jobs: 0,
        timestamp: 1708300800000,
        agent_version: "0.1.0".to_string(),
        system: None,
        gpus: None,
        active_job_metrics: None,
        cache: None,
        asking_prices: None,
    }
}

fn make_full_heartbeat() -> EnhancedHeartbeat {
    EnhancedHeartbeat {
        host_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        status: "online".to_string(),
        active_jobs: 3,
        timestamp: 1708300800000,
        agent_version: "0.1.0".to_string(),
        system: Some(SystemMetricsSnapshot {
            cpu_percent: 45.2,
            memory_used_mb: 12_288,
            memory_total_mb: 32_768,
            disk_used_gb: 120,
            disk_total_gb: 500,
        }),
        gpus: Some(vec![
            GpuMetricsSnapshot {
                index: 0,
                utilization_percent: 87,
                memory_used_mb: 18_432,
                memory_total_mb: 24_576,
                temperature_c: 72,
                power_draw_w: 285.5,
            },
            GpuMetricsSnapshot {
                index: 1,
                utilization_percent: 45,
                memory_used_mb: 8_192,
                memory_total_mb: 24_576,
                temperature_c: 65,
                power_draw_w: 180.3,
            },
        ]),
        active_job_metrics: Some(vec![
            ActiveJobMetrics {
                job_id: "job-001".to_string(),
                job_type: "llm-chat".to_string(),
                duration_ms: 5200,
                tokens_generated: Some(128),
                memory_mb: Some(4096),
                gpu_memory_mb: Some(8192),
            },
            ActiveJobMetrics {
                job_id: "job-002".to_string(),
                job_type: "image-gen".to_string(),
                duration_ms: 12000,
                tokens_generated: None,
                memory_mb: Some(2048),
                gpu_memory_mb: Some(10240),
            },
            ActiveJobMetrics {
                job_id: "job-003".to_string(),
                job_type: "embeddings".to_string(),
                duration_ms: 800,
                tokens_generated: None,
                memory_mb: Some(512),
                gpu_memory_mb: None,
            },
        ]),
        cache: Some(CacheMetricsSnapshot {
            cached_image_count: 12,
            cached_size_mb: 8500,
            warm_workload_count: 3,
            warm_workload_ids: vec![
                "wk-llm-chat".to_string(),
                "wk-sd-diffusers".to_string(),
                "wk-embeddings".to_string(),
            ],
        }),
        asking_prices: Some(HashMap::from([
            ("_default".to_string(), "1.0".to_string()),
            ("llm-chat".to_string(), "2.0".to_string()),
            ("image-gen".to_string(), "5.0".to_string()),
        ])),
    }
}

fn bench_minimal_heartbeat(c: &mut Criterion) {
    let heartbeat = make_minimal_heartbeat();

    c.bench_function("serialize_heartbeat_minimal", |b| {
        b.iter(|| {
            let _bytes = serde_json::to_vec(black_box(&heartbeat)).unwrap();
        })
    });
}

fn bench_full_heartbeat(c: &mut Criterion) {
    let heartbeat = make_full_heartbeat();

    c.bench_function("serialize_heartbeat_full", |b| {
        b.iter(|| {
            let _bytes = serde_json::to_vec(black_box(&heartbeat)).unwrap();
        })
    });
}

fn bench_full_heartbeat_to_string(c: &mut Criterion) {
    let heartbeat = make_full_heartbeat();

    c.bench_function("serialize_heartbeat_full_string", |b| {
        b.iter(|| {
            let _s = serde_json::to_string(black_box(&heartbeat)).unwrap();
        })
    });
}

fn bench_heartbeat_payload_size(c: &mut Criterion) {
    // Verify payload size stays reasonable (not a perf bench, but useful to track)
    let minimal = serde_json::to_vec(&make_minimal_heartbeat()).unwrap();
    let full = serde_json::to_vec(&make_full_heartbeat()).unwrap();

    println!("Minimal heartbeat payload: {} bytes", minimal.len());
    println!("Full heartbeat payload: {} bytes", full.len());

    // Bench the allocation pattern
    c.bench_function("serialize_heartbeat_preallocated", |b| {
        let heartbeat = make_full_heartbeat();
        let mut buf = Vec::with_capacity(2048);
        b.iter(|| {
            buf.clear();
            serde_json::to_writer(black_box(&mut buf), black_box(&heartbeat)).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_minimal_heartbeat,
    bench_full_heartbeat,
    bench_full_heartbeat_to_string,
    bench_heartbeat_payload_size,
);
criterion_main!(benches);
