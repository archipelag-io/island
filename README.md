# Archipelag.io Node Agent

The node agent runs on host machines and executes workloads dispatched by the coordinator.

## Features

- Connects to Docker daemon to run containerized workloads
- Streams output in real-time
- GPU support via NVIDIA Container Toolkit
- (TODO) NATS integration for coordinator communication
- (TODO) Heartbeat and health reporting
- (TODO) GPU detection and capability reporting

## Requirements

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Docker with NVIDIA Container Toolkit (for GPU workloads)
- Linux (macOS/Windows support planned)

## Building

```bash
cargo build --release
```

The binary will be at `target/release/archipelag-agent`.

## Configuration

Create a `config.toml` file:

```toml
# Host ID (optional, auto-generated if not set)
# host_id = "my-gaming-rig"

[coordinator]
nats_url = "nats://localhost:4222"

[docker]
# Optional: custom Docker socket path
# socket = "unix:///var/run/docker.sock"

[workload]
llm_chat_image = "llm-chat:latest"
gpu_devices = ["0"]
```

## Usage

### Test Mode (Phase 0)

Run a single job without coordinator:

```bash
# First, build the llm-chat container from workload-containers repo
docker build -t llm-chat ../workload-containers/llm-chat

# Then run a test job
cargo run -- --test-job "What is the capital of France?"
```

### Agent Mode (Phase 1+)

```bash
cargo run -- --config config.toml
```

## Development

```bash
# Run with debug logging
RUST_LOG=debug cargo run -- --test-job "Hello"

# Check code
cargo check

# Run tests
cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

## Architecture

```
archipelag-agent
├── main.rs        # Entry point, CLI parsing
├── config.rs      # Configuration loading
├── docker.rs      # Docker container management
├── executor.rs    # Job execution logic
└── messages.rs    # Message types for workload I/O
```

## License

MIT
