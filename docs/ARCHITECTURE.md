# Island Architecture

> Internal developer documentation for the Archipelag.io Island software.

## Overview

The Island is a Rust-based daemon that runs on contributor machines, executing workloads assigned by the coordinator. It supports Docker containers and WebAssembly modules.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust (async/await) |
| Runtime | Tokio |
| Container | Bollard (Docker API) |
| WASM | Wasmtime + WASI |
| Messaging | async-nats |

## Directory Structure

```
src/
├── main.rs              # CLI entry point
├── agent.rs             # Core orchestrator (~850 lines)
├── config.rs            # Configuration (TOML)
├── docker.rs            # Container management
├── wasm.rs              # WASM execution
├── nats.rs              # Coordinator communication
├── messages.rs          # I/O protocol types
├── executor.rs          # Test job runner
├── cache.rs             # Image/module caching
├── state.rs             # Persistent state
├── metrics/             # Performance tracking
│   ├── mod.rs
│   ├── gpu.rs           # nvidia-smi integration
│   └── container.rs     # Docker stats
├── security/            # Hardening
│   ├── signing.rs       # cosign verification
│   ├── registry.rs      # Allowlists
│   ├── seccomp.rs       # Syscall filtering
│   └── tls.rs           # Certificate pinning
└── update/              # Auto-update system
    ├── verify.rs        # Ed25519 signatures
    ├── download.rs      # Binary downloads
    └── restart.rs       # Graceful restart
```

## Core Modules

### AgentCore (agent.rs)
Central orchestrator managing the full job lifecycle:
- Capability detection (CPU, RAM, GPU via nvidia-smi)
- NATS connection and registration
- Job assignment handling
- Lease renewal (every 30s)
- Heartbeat loop (every 10s)

### Docker (docker.rs)
Container management with security hardening:
- Image digest verification
- Read-only root filesystem
- Network isolation (network_mode: "none")
- Memory/CPU limits
- GPU passthrough via NVIDIA Container Toolkit

### WASM (wasm.rs)
Sandboxed WebAssembly execution:
- WASI support for stdin/stdout
- Fuel-based instruction limiting
- SHA256 hash verification
- Module caching

### NATS (nats.rs)
Coordinator communication:
- Auto-reconnect with exponential backoff
- Phoenix-compatible message format
- Job streaming protocol

## Configuration

```toml
host_id = "my-host"

[host]
region = "us-west-2"

[coordinator]
nats_url = "nats://localhost:4222"

[workload]
llm_chat_image = "llm-chat:latest"
gpu_devices = ["0"]

[workload.resource_limits]
memory_mb = 8192
read_only_rootfs = true
network_disabled = true
```

## Message Protocol

### Input (stdin)
```json
{"prompt": "Hello", "max_tokens": 100, "temperature": 0.7}
```

### Output (stdout, JSON lines)
```json
{"type": "status", "message": "loading"}
{"type": "token", "content": "Hello"}
{"type": "done", "usage": {"completion_tokens": 42}}
```

## Job Flow

1. Coordinator publishes to `host.{id}.jobs`
2. Agent receives AssignJob message
3. Sends "started" status
4. Pulls container (if needed) or loads WASM
5. Executes with resource limits
6. Streams output chunks to coordinator
7. Renews lease every 30s
8. Sends final status (succeeded/failed)

## Security Model

| Layer | Mechanism |
|-------|-----------|
| Image | SHA256 digest verification |
| Image | cosign signature verification |
| Container | Read-only filesystem |
| Container | Network isolation |
| Container | Memory/CPU limits |
| WASM | Fuel limits (instruction budget) |
| WASM | Hash verification |

## Building

```bash
# Development
mise run dev

# Release build
mise run build

# Tests
mise run test

# CI checks
mise run ci
```

## Operating Modes

```bash
# Test single job
cargo run -- --test-job "Hello world"

# Test WASM module
cargo run -- --test-wasm /path/to/module.wasm

# Production agent
cargo run -- --agent --config config.toml
```

## Dependencies

Key crates:
- `tokio` - Async runtime
- `bollard` - Docker client
- `wasmtime` + `wasmtime-wasi` - WASM runtime
- `async-nats` - NATS client
- `serde`/`serde_json` - Serialization
- `sha2` - Hash verification
- `ed25519-dalek` - Signature verification
