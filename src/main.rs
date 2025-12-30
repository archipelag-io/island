//! Archipelag.io Node Agent
//!
//! The node agent runs on host machines and executes workloads dispatched
//! by the coordinator. It manages container lifecycle, streams output,
//! and reports health/status.

mod config;
mod docker;
mod executor;
mod messages;

use anyhow::Result;
use clap::Parser;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(name = "archipelag-agent")]
#[command(about = "Node agent for archipelag.io distributed compute network")]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Run a single job for testing (bypasses NATS)
    #[arg(long)]
    test_job: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
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

    // If test mode, run a single job and exit
    if let Some(prompt) = args.test_job {
        info!("Running test job with prompt: {}", prompt);
        return executor::run_test_job(&docker, &config, &prompt).await;
    }

    // TODO: Main agent loop
    // 1. Connect to NATS
    // 2. Register with coordinator
    // 3. Send heartbeats
    // 4. Listen for job assignments
    // 5. Execute jobs and stream output

    info!("Agent ready. Use --test-job to run a test, or implement NATS integration.");

    Ok(())
}
