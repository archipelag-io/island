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

    /// Run in agent mode (connect to NATS and wait for jobs)
    #[arg(long)]
    agent: bool,
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

    // If agent mode, run the full agent loop
    if args.agent {
        info!("Starting agent mode");
        let agent = agent::Agent::new(config, docker).await?;
        return agent.run().await;
    }

    // Default: show help
    info!("Agent ready. Options:");
    info!("  --test-job <PROMPT>  Run a test job with the given prompt");
    info!("  --agent              Run in agent mode (connect to NATS)");

    Ok(())
}
