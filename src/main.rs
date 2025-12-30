//! Archipelag.io Node Agent
//!
//! The node agent runs on host machines and executes workloads dispatched
//! by the coordinator. It manages container lifecycle, streams output,
//! and reports health/status.

mod agent;
mod config;
mod docker;
mod executor;
mod messages;
mod nats;
mod wasm;

use anyhow::Result;
use clap::Parser;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(name = "archipelag-agent")]
#[command(about = "Node agent for archipelag.io distributed compute network")]
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

    /// Run in agent mode (connect to NATS and wait for jobs)
    #[arg(long)]
    agent: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    info!("Starting archipelag-agent v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = config::load(&args.config)?;
    info!("Loaded configuration from {}", args.config);

    // Connect to Docker
    let docker = docker::connect().await?;
    info!("Connected to Docker daemon");

    // If WASM test mode, run a WASM module
    if let Some(wasm_path) = args.test_wasm {
        info!("Running WASM module: {}", wasm_path);
        return run_wasm_test(&wasm_path, &args.wasm_input).await;
    }

    // If container test mode, run a single job and exit
    if let Some(prompt) = args.test_job {
        info!("Running test job with prompt: {}", prompt);
        return executor::run_test_job(&docker, &config, &prompt).await;
    }

    // If agent mode, run the full agent loop
    if args.agent {
        info!("Starting agent mode");
        let agent = agent::Agent::new(config, docker).await?;
        return agent.run().await;
    }

    // Default: show help
    info!("Agent ready. Options:");
    info!("  --test-job <PROMPT>   Run a container job with the given prompt");
    info!("  --test-wasm <PATH>    Run a WASM module");
    info!("  --wasm-input <JSON>   JSON input for WASM module");
    info!("  --agent               Run in agent mode (connect to NATS)");

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
