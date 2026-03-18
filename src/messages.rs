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
#[allow(dead_code)]
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

/// Token usage statistics from LLM workloads
#[derive(Debug, Deserialize, Clone)]
pub struct Usage {
    #[allow(dead_code)]
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_input_serialization() {
        let input = ChatInput {
            prompt: "Hello, world!".to_string(),
            max_tokens: Some(100),
            temperature: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"prompt\":\"Hello, world!\""));
        assert!(json.contains("\"max_tokens\":100"));
        // temperature should be skipped when None
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn test_chat_input_all_fields() {
        let input = ChatInput {
            prompt: "Test".to_string(),
            max_tokens: Some(50),
            temperature: Some(0.7),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_image_gen_input_serialization() {
        let input = ImageGenInput {
            prompt: "A cat".to_string(),
            width: Some(512),
            height: Some(512),
            steps: None,
            seed: Some(42),
        };
        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"prompt\":\"A cat\""));
        assert!(json.contains("\"width\":512"));
        assert!(json.contains("\"seed\":42"));
        assert!(!json.contains("steps"));
    }

    #[test]
    fn test_workload_output_status_deserialization() {
        let json = r#"{"type":"status","message":"Loading model..."}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Status { message } => assert_eq!(message, "Loading model..."),
            _ => panic!("Expected Status variant"),
        }
    }

    #[test]
    fn test_workload_output_token_deserialization() {
        let json = r#"{"type":"token","content":"Hello"}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Token { content } => assert_eq!(content, "Hello"),
            _ => panic!("Expected Token variant"),
        }
    }

    #[test]
    fn test_workload_output_progress_deserialization() {
        let json = r#"{"type":"progress","step":5,"total":20}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Progress { step, total } => {
                assert_eq!(step, 5);
                assert_eq!(total, 20);
            }
            _ => panic!("Expected Progress variant"),
        }
    }

    #[test]
    fn test_workload_output_done_with_usage() {
        let json = r#"{"type":"done","usage":{"prompt_tokens":10,"completion_tokens":25}}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Done { usage, seed } => {
                let u = usage.unwrap();
                assert_eq!(u.prompt_tokens, Some(10));
                assert_eq!(u.completion_tokens, Some(25));
                assert!(seed.is_none());
            }
            _ => panic!("Expected Done variant"),
        }
    }

    #[test]
    fn test_workload_output_done_without_usage() {
        let json = r#"{"type":"done"}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Done { usage, seed } => {
                assert!(usage.is_none());
                assert!(seed.is_none());
            }
            _ => panic!("Expected Done variant"),
        }
    }

    #[test]
    fn test_workload_output_error_deserialization() {
        let json = r#"{"type":"error","message":"OOM killed"}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Error { message } => assert_eq!(message, "OOM killed"),
            _ => panic!("Expected Error variant"),
        }
    }

    #[test]
    fn test_workload_output_image_deserialization() {
        let json =
            r#"{"type":"image","data":"base64data","format":"png","width":512,"height":512}"#;
        let output: WorkloadOutput = serde_json::from_str(json).unwrap();
        match output {
            WorkloadOutput::Image {
                data,
                format,
                width,
                height,
            } => {
                assert_eq!(data, "base64data");
                assert_eq!(format, "png");
                assert_eq!(width, 512);
                assert_eq!(height, 512);
            }
            _ => panic!("Expected Image variant"),
        }
    }
}
