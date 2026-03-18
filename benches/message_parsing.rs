//! Benchmarks for WorkloadOutput deserialization — the hot path during streaming.
//!
//! Every token emitted by a workload container is parsed through this path,
//! so deserialization performance directly affects streaming latency.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Re-define the types here since they're in a binary crate.
// These must stay in sync with src/messages.rs.
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum WorkloadOutput {
    Status {
        message: String,
    },
    Token {
        content: String,
    },
    Progress {
        step: u32,
        total: u32,
    },
    Image {
        data: String,
        format: String,
        width: u32,
        height: u32,
    },
    Done {
        #[serde(default)]
        usage: Option<Usage>,
        #[serde(default)]
        seed: Option<u64>,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

fn bench_token_parsing(c: &mut Criterion) {
    let token_json = r#"{"type":"token","content":"Hello"}"#;

    c.bench_function("parse_token_short", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(token_json)).unwrap();
        })
    });

    // Longer token content (more realistic for multi-byte/sentence tokens)
    let long_token = format!(
        r#"{{"type":"token","content":"{}"}}"#,
        "The quick brown fox jumps over the lazy dog. ".repeat(10)
    );

    c.bench_function("parse_token_long", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(&long_token)).unwrap();
        })
    });
}

fn bench_status_parsing(c: &mut Criterion) {
    let status_json = r#"{"type":"status","message":"Loading model..."}"#;

    c.bench_function("parse_status", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(status_json)).unwrap();
        })
    });
}

fn bench_done_parsing(c: &mut Criterion) {
    let done_simple = r#"{"type":"done"}"#;
    let done_with_usage =
        r#"{"type":"done","usage":{"prompt_tokens":128,"completion_tokens":512},"seed":42}"#;

    c.bench_function("parse_done_simple", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(done_simple)).unwrap();
        })
    });

    c.bench_function("parse_done_with_usage", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(done_with_usage)).unwrap();
        })
    });
}

fn bench_image_parsing(c: &mut Criterion) {
    // Simulate a small base64-encoded image (~1KB)
    let small_data = "A".repeat(1024);
    let small_image = format!(
        r#"{{"type":"image","data":"{}","format":"png","width":64,"height":64}}"#,
        small_data
    );

    // Simulate a realistic base64-encoded image (~500KB)
    let large_data = "A".repeat(500_000);
    let large_image = format!(
        r#"{{"type":"image","data":"{}","format":"png","width":512,"height":512}}"#,
        large_data
    );

    let mut group = c.benchmark_group("parse_image");
    group.bench_with_input(BenchmarkId::new("1KB", "small"), &small_image, |b, json| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(json)).unwrap();
        })
    });
    group.bench_with_input(
        BenchmarkId::new("500KB", "large"),
        &large_image,
        |b, json| {
            b.iter(|| {
                let _output: WorkloadOutput = serde_json::from_str(black_box(json)).unwrap();
            })
        },
    );
    group.finish();
}

fn bench_error_parsing(c: &mut Criterion) {
    let error_json =
        r#"{"type":"error","message":"OOM killed: container exceeded 4GB memory limit"}"#;

    c.bench_function("parse_error", |b| {
        b.iter(|| {
            let _output: WorkloadOutput = serde_json::from_str(black_box(error_json)).unwrap();
        })
    });
}

fn bench_from_bytes(c: &mut Criterion) {
    // In the real hot path, we receive bytes from Docker/NATS, not a str
    let token_bytes = br#"{"type":"token","content":"Hello world"}"#;

    c.bench_function("parse_token_from_bytes", |b| {
        b.iter(|| {
            let _output: WorkloadOutput =
                serde_json::from_slice(black_box(token_bytes.as_slice())).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_token_parsing,
    bench_status_parsing,
    bench_done_parsing,
    bench_image_parsing,
    bench_error_parsing,
    bench_from_bytes,
);
criterion_main!(benches);
