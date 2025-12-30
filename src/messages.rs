//! Message types for workload I/O
//!
//! These match the JSON format expected by workload containers.

use serde::{Deserialize, Serialize};

/// Input to an LLM chat workload
#[derive(Debug, Serialize)]
pub struct ChatInput {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Output event from a workload
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum WorkloadOutput {
    Status { message: String },
    Token { content: String },
    Done { usage: Option<Usage> },
    Error { message: String },
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}
