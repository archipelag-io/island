# Archipelag.io Island

The Island software runs on contributor machines and executes workloads dispatched by the coordinator. It supports Docker containers, WASM modules, and reports hardware capabilities and metrics.

## Features

- **Container execution** — Docker-based workloads with GPU passthrough (NVIDIA)
- **WASM execution** — Wasmtime runtime with WASI, fuel metering, and timeout enforcement
- **NATS with JetStream** — Durable job subscriptions with ack-on-spawn, core NATS fallback
- **Heartbeat & metrics** — System, GPU, and cache metrics reported every 10 seconds
- **Security** — Cosign signature verification, registry allowlist, seccomp profiles per sandbox tier
- **Image caching** — LRU eviction, warm workload tracking, pre-pull on startup
- **Self-update** — Polls coordinator for new versions, Ed25519 signature verification
- **Persistent state** — Pairing status and WASM cache in `~/.archipelag/`

## Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | stable (via mise) | Build from source |
| Docker | 20.10+ | Container runtime |
| NVIDIA drivers + Container Toolkit | Optional | GPU workloads |
| NATS | 2.10+ | Coordinator communication (via `infra/`) |
| `cosign` | Optional | Container image signature verification |

## Building

```bash
# Debug build
cargo build

# Release build (LTO + stripped)
cargo build --release
# Binary: target/release/archipelag-island
```

## Configuration

Copy and edit the example config:

```bash
cp config.example.toml config.toml
```

### Configuration Reference

```toml
[host]
name = "my-host"                        # Human-readable name
region = "us-east"                       # Geographic region

[coordinator]
nats_url = "nats://localhost:4222"       # NATS server URL

[docker]
# socket = "unix:///var/run/docker.sock" # Custom Docker socket (optional)

[workload]
llm_chat_image = "archipelag-llm-chat-mock:latest"
gpu_devices = []                         # GPU device IDs: ["0"], ["0","1"]

[workload.resource_limits]
memory_mb = 8192                         # Memory limit (default: 8GB)
read_only_rootfs = true                  # Read-only filesystem
tmpfs_size_mb = 256                      # /tmp writable size
cpu_percent = 200                        # CPU quota (optional)
network_disabled = true                  # Network isolation (default: true)

[cache]
enable_preload = false                   # Pre-pull images on startup
preload_images = ["llm-chat:latest"]
max_cached_images = 20                   # LRU eviction threshold
warm_ttl_seconds = 3600                  # Warm workload tracking TTL

[signing]
enabled = true                           # Enable cosign verification
require_signature = false                # Reject unsigned images
keys_url = "http://coordinator:4000/api/v1/signing-keys"

[registry]
enabled = true                           # Enable registry allowlist
require_digest = false                   # Require image digest pinning
allowed = [                              # Allowed registries
  "ghcr.io/archipelag-io",
  "docker.io/archipelag"
]
```

## Usage

### Island Mode (production)

Connect to coordinator and accept jobs:

```bash
cargo run -- --config config.toml
```

### Test Modes

Run a single job without coordinator:

```bash
# Container test job
cargo run -- --test-job "What is the capital of France?"

# WASM test job
cargo run -- --test-wasm path/to/module.wasm --wasm-input '{"key":"value"}'
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RUST_LOG` | `info` | Log level filter (e.g., `debug`, `archipelag_island=trace`) |
| `ARCHIPELAG_LOG_JSON` | unset | Enable JSON structured logging |
| `RUST_BACKTRACE` | unset | Show panic backtraces (`1` or `full`) |

## Architecture

```
src/
├── main.rs                  # CLI entry point (clap)
├── agent.rs                 # Core orchestrator (~850 LOC)
│                            #   - Event loop (tokio::select!)
│                            #   - Job spawning & cancellation
│                            #   - Heartbeat, cache cleanup, update checks
├── config.rs                # TOML configuration loading
├── nats.rs                  # NATS/JetStream communication
│                            #   - Registration, heartbeat publishing
│                            #   - JetStream pull consumer (with core fallback)
│                            #   - JobSubscription enum
├── docker.rs                # Container management (Bollard)
│                            #   - Security: seccomp, read-only rootfs, network isolation
│                            #   - GPU passthrough, resource limits
│                            #   - Real-time output streaming
├── executor.rs              # Test job execution helper
├── messages.rs              # Workload I/O protocol (JSON lines)
├── wasm.rs                  # WASM execution (Wasmtime + WASI)
│                            #   - Fuel metering, timeout, hash verification
├── cache.rs                 # Image & WASM module caching
│                            #   - LRU eviction, warm workload tracking
├── state.rs                 # Persistent state (~/.archipelag/)
│                            #   - Pairing status, WASM cache
├── metrics/
│   ├── mod.rs               # Metrics aggregation (system, job)
│   ├── gpu.rs               # nvidia-smi GPU metrics collection
│   └── container.rs         # Docker stats API
├── security/
│   ├── mod.rs
│   ├── signing.rs           # Cosign container verification
│   ├── registry.rs          # Registry allowlist enforcement
│   ├── seccomp.rs           # Seccomp profile generation
│   └── tls.rs               # TLS certificate pinning (stubbed)
└── update/
    ├── mod.rs               # Update checker (30 min poll)
    ├── verify.rs            # Ed25519 binary verification
    ├── download.rs          # Resumable binary download
    └── restart.rs           # Graceful restart
```

**Total:** ~7,700 lines of Rust across 22 source files.

## NATS Communication

### Subjects

| Subject | Direction | Purpose |
|---------|-----------|---------|
| `coordinator.hosts.register` | Island → Coordinator | Registration with capabilities |
| `coordinator.hosts.pairing` | Island → Coordinator | Pairing request (request/reply) |
| `host.{id}.heartbeat` | Island → Coordinator | Metrics snapshot (every 10s) |
| `host.{id}.lease` | Island → Coordinator | Lease renewal |
| `host.{id}.jobs` | Coordinator → Island | Job dispatch (**JetStream**) |
| `host.{id}.status` | Island → Coordinator | Job state updates (**JetStream**) |
| `host.{id}.output` | Island → Coordinator | Output streaming (**JetStream**) |
| `host.{id}.cancel` | Coordinator → Island | Cancel running job |

### JetStream Subscription

The Island prefers JetStream pull consumers for reliable job delivery:
- Stream: `JOBS`, Consumer: `host-{host_id}`
- Ack policy: explicit (ack-on-spawn)
- Max deliver: 5, Ack wait: 60s
- Falls back to core NATS subscription if JetStream unavailable

## Security

### Sandbox Tiers

| Tier | Memory | Timeout | Network | GPU | Seccomp |
|------|--------|---------|---------|-----|---------|
| `restricted` | 256 MB | 60s | Disabled | No | Minimal |
| `standard` | 1 GB | 300s | Disabled | Yes | Default |
| `elevated` | 8 GB | 600s | Allowed | Yes | GPU/Network |

### Container Isolation

- Read-only root filesystem (tmpfs `/tmp` for writable space)
- Network mode `none` by default
- All Linux capabilities dropped
- Seccomp syscall filtering per sandbox tier
- Image digest verification
- Registry allowlist enforcement
- Cosign signature verification (optional)

## Development

```bash
mise run dev             # Debug build + run
mise run dev:release     # Release build + run
mise run test            # Run all 73 tests
mise run test:verbose    # Tests with output
mise run fmt             # Format code
mise run fmt:check       # Check formatting
mise run clippy          # Lint (warnings = errors)
mise run ci              # All CI checks (fmt + clippy + test)
mise run bench           # Run Criterion benchmarks
mise run doc             # Generate and open docs
mise run clean           # Clean build artifacts
```

## Testing

73 tests across inline `#[cfg(test)]` modules in source files:

| Module | Tests | Coverage |
|--------|-------|----------|
| `messages.rs` | Token parsing, serialization, edge cases |
| `state.rs` | Persistence, WASM cache |
| `cache.rs` | LRU eviction, warm tracking |
| `config.rs` | TOML parsing, defaults, validation |
| `security/signing.rs` | Signature verification |
| `security/registry.rs` | Allowlist, digest checking |
| `security/seccomp.rs` | Profile generation |

## Benchmarks

Criterion 0.5 benchmarks in `benches/`:

| Benchmark | What it measures |
|-----------|-----------------|
| `message_parsing` | WorkloadOutput deserialization (hot path) — token, status, done, image payloads |
| `heartbeat_serialization` | EnhancedHeartbeat serialization — minimal vs full metrics |

Run with `mise run bench` or `cargo bench`.

**Note:** Benchmark types are duplicated from `src/` (binary crate limitation). Keep `messages.rs` and `nats.rs` types in sync with benchmark files.

## Troubleshooting

### Island can't connect to NATS
- Ensure NATS is running: `curl http://localhost:8222/varz`
- Check `nats_url` in `config.toml`
- Verify JetStream is enabled: `curl http://localhost:8222/jsz`

### Docker permission denied
- Add user to docker group: `sudo usermod -aG docker $USER`
- Or configure custom socket in `[docker]` config

### GPU not detected
- Verify NVIDIA drivers: `nvidia-smi`
- Install NVIDIA Container Toolkit
- Set `gpu_devices = ["0"]` in config
- Island gracefully falls back to CPU-only if nvidia-smi unavailable

### Image pull failures
- Check registry allowlist in `[registry]` config
- For local images, use `docker.io/library/` prefix or disable registry check

### High memory usage
- Reduce `max_cached_images` in `[cache]` config
- Lower `memory_mb` in `[workload.resource_limits]`

## Related

| Resource | Path |
|----------|------|
| [Coordinator](../app/) | Elixir control plane |
| [Infrastructure](../infra/) | Docker Compose, E2E tests |
| [Architecture](docs/ARCHITECTURE.md) | Internal Island architecture doc |
| [Example Config](config.example.toml) | Full configuration reference |

## License

MIT
