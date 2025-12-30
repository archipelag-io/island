//! Job execution logic

use crate::config::AgentConfig;
use crate::docker::{self, ContainerConfig};
use crate::messages::{ChatInput, WorkloadOutput};
use anyhow::{Context, Result};
use bollard::Docker;
use tracing::{error, info};

/// Run a test job (for development/debugging)
pub async fn run_test_job(docker: &Docker, config: &AgentConfig, prompt: &str) -> Result<()> {
    info!("Preparing test job");

    // Create input
    let input = ChatInput {
        prompt: prompt.to_string(),
        max_tokens: Some(512),
        temperature: Some(0.7),
    };

    let input_json = serde_json::to_string(&input).context("Failed to serialize input")?;

    // Configure container
    let container_config = ContainerConfig {
        image: config.workload.llm_chat_image.clone(),
        input: input_json,
        gpu_devices: config.workload.gpu_devices.clone(),
    };

    info!("Starting container: {}", container_config.image);

    // Buffer for accumulating output lines
    let mut buffer = String::new();

    // Run container and process output
    let exit_code = docker::run_container(docker, container_config, |chunk| {
        buffer.push_str(&chunk);

        // Process complete lines
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON line
            match serde_json::from_str::<WorkloadOutput>(&line) {
                Ok(output) => match output {
                    WorkloadOutput::Status { message } => {
                        info!("Status: {}", message);
                    }
                    WorkloadOutput::Token { content } => {
                        // Print tokens without newline for streaming effect
                        print!("{}", content);
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                    WorkloadOutput::Done { usage } => {
                        println!(); // Final newline
                        if let Some(usage) = usage {
                            info!(
                                "Done. Tokens: {}",
                                usage.completion_tokens.unwrap_or(0)
                            );
                        } else {
                            info!("Done.");
                        }
                    }
                    WorkloadOutput::Error { message } => {
                        error!("Workload error: {}", message);
                    }
                },
                Err(e) => {
                    // Not valid JSON, might be raw output
                    info!("Raw output: {}", line);
                }
            }
        }
    })
    .await?;

    if exit_code != 0 {
        error!("Container exited with code {}", exit_code);
    }

    Ok(())
}
