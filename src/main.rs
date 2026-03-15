//! Archipelag.io Island
//!
//! The Island software runs on contributor machines and executes workloads
//! dispatched by the coordinator. It manages container lifecycle, streams
//! output, and reports health/status.

mod agent;
mod cache;
mod config;
#[cfg(feature = "diffusers")]
mod diffusers;
mod docker;
mod executor;
#[cfg(feature = "gguf")]
mod gguf;
mod messages;
#[allow(dead_code)]
mod metrics;
mod model_cache;
mod nats;
#[cfg(feature = "pipeline")]
#[allow(dead_code)]
mod layer_executor;
#[cfg(feature = "pipeline")]
mod pipeline;
mod preload;
#[cfg(feature = "pipeline")]
mod shard_split;
#[cfg(feature = "pipeline")]
#[allow(dead_code)]
mod transport;
#[cfg(target_os = "linux")]
mod oci;
#[cfg(feature = "onnx")]
mod onnx;
#[allow(dead_code)]
mod security;
mod state;
#[allow(dead_code)]
mod update;
mod wasm;

use anyhow::Result;
use clap::Parser;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "island")]
#[command(about = "Island software for the Archipelag.io distributed compute network")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Run a single container job for testing (bypasses NATS)
    #[arg(long)]
    test_job: Option<String>,

    /// Run a WASM module for testing
    #[arg(long)]
    test_wasm: Option<String>,

    /// JSON input for WASM test (default: {})
    #[arg(long, default_value = "{}")]
    wasm_input: String,

    /// Run in Island mode (connect to NATS and wait for jobs)
    #[arg(long)]
    agent: bool,

    /// Split a GGUF model into N shards for pipeline parallelism
    #[cfg(feature = "pipeline")]
    #[arg(long)]
    split_model: Option<String>,

    /// Number of shards to split into (default: 2, used with --split-model)
    #[cfg(feature = "pipeline")]
    #[arg(long, default_value = "2")]
    shards: usize,

    /// Output directory for shards (default: ./shards, used with --split-model)
    #[cfg(feature = "pipeline")]
    #[arg(long, default_value = "shards")]
    output_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Install rustls crypto provider for TLS connections (NATS, OCI registry)
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    // Initialize logging with RUST_LOG support (default: info)
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    if std::env::var("ARCHIPELAG_LOG_JSON").is_ok() {
        // JSON output for production log aggregation
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .init();
    } else {
        // Human-readable output for development
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .init();
    }

    let args = Args::parse();

    info!("Starting island v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = config::load(&args.config)?;
    info!("Loaded configuration from {}", args.config);

    // If split-model mode, split a GGUF file into shards and exit
    #[cfg(feature = "pipeline")]
    if let Some(model_path) = args.split_model {
        info!("Splitting model {} into {} shards", model_path, args.shards);
        let input = std::path::Path::new(&model_path);
        let output = std::path::Path::new(&args.output_dir);
        shard_split::split_model(input, args.shards, output)?;
        return Ok(());
    }

    // If WASM test mode, run a WASM module (no Docker needed)
    if let Some(wasm_path) = args.test_wasm {
        info!("Running WASM module: {}", wasm_path);
        return run_wasm_test(&wasm_path, &args.wasm_input).await;
    }

    // Connect to Docker (optional — Island can run in WASM-only mode)
    let docker = match docker::connect().await {
        Ok(d) => {
            info!("Connected to Docker daemon");
            Some(d)
        }
        Err(e) => {
            warn!("Docker not available: {}. Running in WASM-only mode.", e);
            None
        }
    };

    // If container test mode, run a single job and exit
    if let Some(prompt) = args.test_job {
        let docker = docker.ok_or_else(|| {
            anyhow::anyhow!("Docker is required for container test jobs. Install Docker and try again.")
        })?;
        info!("Running test job with prompt: {}", prompt);
        return executor::run_test_job(&docker, &config, &prompt).await;
    }

    // If agent mode, run the full Island loop
    if args.agent {
        info!("Starting Island mode");
        let agent = agent::Agent::new(config, docker).await?;
        return agent.run().await;
    }

    // Default: show help
    info!("Island ready. Options:");
    info!("  --test-job <PROMPT>   Run a container job with the given prompt");
    info!("  --test-wasm <PATH>    Run a WASM module");
    info!("  --wasm-input <JSON>   JSON input for WASM module");
    info!("  --agent               Run in Island mode (connect to NATS)");

    Ok(())
}

/// Run a WASM module for testing
async fn run_wasm_test(wasm_path: &str, input: &str) -> Result<()> {
    use tokio::sync::mpsc;
    use wasm::{WasmConfig, WasmExecutor, WasmOutput};

    let executor = WasmExecutor::new()?;

    // Validate the module first
    info!("Validating WASM module...");
    let module_info = executor.validate_module(std::path::Path::new(wasm_path))?;
    info!("  Exports: {:?}", module_info.exports);
    info!("  Has _start: {}", module_info.has_start);

    if !module_info.has_start {
        anyhow::bail!("Module must have a _start export (WASI entry point)");
    }

    // Run the module
    let config = WasmConfig {
        module_path: wasm_path.to_string(),
        input: input.to_string(),
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(32);

    info!("Executing WASM module with input: {}", input);

    let exit_code = executor.run(config, tx).await?;

    // Print output
    while let Some(output) = rx.recv().await {
        match output {
            WasmOutput::Stdout(s) => {
                for line in s.lines() {
                    println!("{}", line);
                }
            }
            WasmOutput::Stderr(s) => {
                for line in s.lines() {
                    eprintln!("stderr: {}", line);
                }
            }
            WasmOutput::Exit(code) => {
                info!("WASM exit code: {}", code);
            }
            WasmOutput::Timeout => {
                error!("WASM execution timed out");
            }
        }
    }

    info!("WASM execution complete, exit code: {}", exit_code);

    Ok(())
}
