# Changelog

All notable changes to the Island software will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-10-18

### Added
- Chat Jobs use the model's own chat template, with OpenAI-style `tools`,
  `tool_choice` and `response_format`; tool calls are constrained by a lazy
  grammar and returned as `tool_calls` in the final event
- Streamed output carries a `delta` (content, reasoning or tool-call parts)
- Reasoning models: thinking is returned separately as `reasoning_content`
  (`enable_thinking` to turn it off)
- llama.cpp (GGUF) runtime in every native release build, including Linux ARM
- Exo multi-Island cluster runtime
- `[host.capability_overrides]` for lab and staging capability simulation
- Heartbeats advertise `supported_quantization_formats`

### Changed
- Prompts are decoded in batches and the loaded model is cached between Jobs

### Fixed
- Grammar-constrained sampling no longer accepts each token twice (crashed
  with `GGML_ASSERT(!stacks.empty())`)
- Concurrent downloads of the same model no longer corrupt each other
- The model is unloaded on shutdown, avoiding a Metal assert at exit

## [0.5.0] - 2026-03-19

### Added
- Modern GGUF runtime (llama-cpp-2 0.1.139), pipeline execution, market pricing
- PR template with Island-specific checklist items
- Issue templates (bug report, feature request, security vulnerability)

## [0.1.0] - 2026-02-19

Initial stabilization release. Compilation errors fixed, tests added, documentation rewritten.

### Added

#### Testing (Tier 1)
- 73 tests across messages, state, cache, config, security modules
- Inline `#[cfg(test)]` modules in source files
- Criterion benchmarks: message parsing (~85ns) and heartbeat serialization

#### Monitoring & Logging (Tier 2)
- `RUST_LOG` environment filter support
- `ARCHIPELAG_LOG_JSON` for JSON log output
- `#[instrument]` span on `execute_job` with job_id, workload_id, runtime_type

#### Security (Tier 3)
- Seccomp profiles mapped to sandbox tiers (restricted/standard/elevated)
- Registry allowlist enforcement in container execution path
- Container image signature verification (cosign)

#### Data Integrity (Tier 4)
- JetStream job subscription with `JobSubscription` enum (Core/JetStream with fallback)
- Ack-on-spawn for reliable job processing
- State persistence with serde serialization

#### Performance (Tier 6)
- Criterion benchmarks: `WorkloadOutput` deserialization, `EnhancedHeartbeat` serialization
- Message parsing optimized to ~85ns (from_bytes)

#### CI/CD
- GitHub Actions CI pipeline: fmt check, clippy (warnings-as-errors), test with coverage, release build
- `cargo-llvm-cov` for code coverage

### Fixed
- 6 compilation errors (cache type handling, ContainerConfig fields, Instant serialization)
- All clippy warnings resolved (`-D warnings` clean)
- Code formatting (`cargo fmt` clean)
- Dependency vulnerabilities: bytes 1.11.0 → 1.11.1, time 0.3.44 → 0.3.47

### Changed
- Cache module: `Instant` fields use `#[serde(skip)]` for proper serialization
- Enhanced heartbeat: GPU, system, and cache metrics wired into heartbeat payload
- UpdateChecker wired into main loop (30-minute check interval)
