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

/// Input to an image generation workload
#[derive(Debug, Serialize)]
pub struct ImageGenInput {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Output event from a workload
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum WorkloadOutput {
    /// Status message (loading, ready, etc.)
    Status { message: String },

    /// Token from LLM streaming
    Token { content: String },

    /// Progress update (for image generation)
    Progress { step: u32, total: u32 },

    /// Generated image (base64 encoded)
    Image {
        data: String,
        format: String,
        width: u32,
        height: u32,
    },

    /// Workload complete
    Done {
        #[serde(default)]
        usage: Option<Usage>,
        #[serde(default)]
        seed: Option<u64>,
    },

    /// Error occurred
    Error { message: String },
}

#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}
