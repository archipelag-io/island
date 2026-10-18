//! Chat request parsing and output helpers for LLM runtimes.
//!
//! Turns an OpenAI-style Job input (`messages`, `tools`, `tool_choice`,
//! `response_format`, sampling fields) into what a runtime needs, and holds
//! back streamed text that might be the start of a stop sequence. Kept free of
//! llama.cpp types so it builds and tests without the `gguf` feature.

use serde_json::Value;

/// Default number of tokens to generate when the input does not say
pub const DEFAULT_MAX_TOKENS: usize = 1024;

/// An OpenAI-compatible chat request, ready for a chat template
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// The `messages` array, serialized
    pub messages_json: String,
    /// The `tools` array, serialized, when tools were offered
    pub tools_json: Option<String>,
    /// `auto`, `none` or `required`
    pub tool_choice: Option<String>,
    /// JSON schema the reply must follow (from `response_format`)
    pub json_schema: Option<String>,
    pub parallel_tool_calls: bool,
    pub enable_thinking: bool,
}

impl ChatRequest {
    pub fn has_tools(&self) -> bool {
        self.tools_json.is_some() && self.tool_choice.as_deref() != Some("none")
    }
}

/// Sampling settings for one generation
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

/// Build a chat request when the input carries a `messages` array.
///
/// Returns `None` for prompt-style inputs (`prompt` or `text`).
pub fn chat_request(input: &Value) -> Option<ChatRequest> {
    let messages = input.get("messages")?.as_array()?;
    if messages.is_empty() {
        return None;
    }

    let mut tools = input
        .get("tools")
        .and_then(Value::as_array)
        .filter(|t| !t.is_empty())
        .cloned();

    // llama.cpp only understands string tool choices. A named function
    // becomes "required" with the tool list narrowed to that function.
    let tool_choice = match input.get("tool_choice") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(obj)) => {
            let name = obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str);
            if let (Some(name), Some(list)) = (name, tools.as_mut()) {
                list.retain(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        == Some(name)
                });
            }
            Some("required".to_string())
        }
        _ => None,
    };

    let json_schema =
        input
            .get("response_format")
            .and_then(|rf| match rf.get("type").and_then(Value::as_str) {
                Some("json_schema") => rf
                    .get("json_schema")
                    .and_then(|js| js.get("schema"))
                    .map(Value::to_string),
                Some("json_object") => Some(
                    rf.get("schema")
                        .map(Value::to_string)
                        .unwrap_or_else(|| r#"{"type":"object"}"#.to_string()),
                ),
                _ => None,
            });

    Some(ChatRequest {
        messages_json: Value::Array(messages.clone()).to_string(),
        tools_json: tools
            .filter(|t| !t.is_empty())
            .map(|t| Value::Array(t).to_string()),
        tool_choice,
        json_schema,
        parallel_tool_calls: input
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        enable_thinking: input
            .get("enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

/// Read sampling settings from the input, falling back to the Cargo's
/// temperature and then to llama.cpp server defaults.
pub fn sampling_params(input: &Value, cargo_temperature: Option<f32>) -> SamplingParams {
    let temperature = input
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|t| t as f32)
        .or(cargo_temperature)
        .unwrap_or(0.7)
        .clamp(0.0, 2.0);

    let top_p = input
        .get("top_p")
        .and_then(Value::as_f64)
        .map(|p| p as f32)
        .filter(|p| *p > 0.0 && *p <= 1.0)
        .unwrap_or(0.95);

    let seed = input
        .get("seed")
        .and_then(Value::as_u64)
        .map(|s| s as u32)
        .unwrap_or_else(rand::random);

    let max_tokens = input
        .get("max_completion_tokens")
        .or_else(|| input.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_TOKENS);

    let stop = match input.get("stop") {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(list)) => list
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };

    SamplingParams {
        temperature,
        top_p,
        seed,
        max_tokens,
        stop,
    }
}

/// Collects generated text and releases it once it can no longer turn into
/// a stop sequence, so stop strings never reach the Consumer.
#[derive(Debug, Default)]
pub struct StopScanner {
    stops: Vec<String>,
    text: String,
    released: usize,
    stopped: bool,
}

impl StopScanner {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            ..Self::default()
        }
    }

    /// Add a generated piece. Returns the text that is now safe to emit.
    pub fn push(&mut self, piece: &str) -> String {
        if self.stopped {
            return String::new();
        }
        let search_from = self.released;
        self.text.push_str(piece);

        let hit = self
            .stops
            .iter()
            .filter_map(|s| {
                self.text[search_from..]
                    .find(s.as_str())
                    .map(|i| search_from + i)
            })
            .min();
        if let Some(end) = hit {
            self.text.truncate(end);
            self.stopped = true;
            return self.release_to(end);
        }

        // Hold back the longest tail that is a prefix of some stop sequence
        let mut safe = self.text.len();
        for stop in &self.stops {
            for len in (1..stop.len().min(self.text.len() - search_from + 1)).rev() {
                let start = self.text.len() - len;
                if self.text.is_char_boundary(start) && stop.starts_with(&self.text[start..]) {
                    safe = safe.min(start);
                    break;
                }
            }
        }
        self.release_to(safe)
    }

    /// Whether a stop sequence has been seen
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Release whatever is still held back at the end of generation
    pub fn finish(&mut self) -> String {
        let end = self.text.len();
        self.release_to(end)
    }

    /// All text generated so far, minus any stop sequence
    pub fn text(&self) -> &str {
        &self.text
    }

    fn release_to(&mut self, end: usize) -> String {
        if end <= self.released {
            return String::new();
        }
        let out = self.text[self.released..end].to_string();
        self.released = end;
        out
    }
}

/// OpenAI `finish_reason` for a finished generation
pub fn finish_reason(has_tool_calls: bool, hit_length_limit: bool) -> &'static str {
    if has_tool_calls {
        "tool_calls"
    } else if hit_length_limit {
        "length"
    } else {
        "stop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prompt_inputs_are_not_chat_requests() {
        assert!(chat_request(&json!({"prompt": "hi"})).is_none());
        assert!(chat_request(&json!({"messages": []})).is_none());
    }

    #[test]
    fn chat_request_passes_messages_and_tools_through() {
        let input = json!({
            "messages": [
                {"role": "user", "content": "Weather in Oslo?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "4°C"}
            ],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            "tool_choice": "auto"
        });
        let req = chat_request(&input).unwrap();
        let messages: Value = serde_json::from_str(&req.messages_json).unwrap();
        assert_eq!(messages[2]["tool_call_id"], "c1");
        assert!(req.tools_json.unwrap().contains("weather"));
        assert_eq!(req.tool_choice.as_deref(), Some("auto"));
        assert!(!req.parallel_tool_calls);
    }

    #[test]
    fn named_tool_choice_narrows_tools() {
        let input = json!({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "b"}}
        });
        let req = chat_request(&input).unwrap();
        assert_eq!(req.tool_choice.as_deref(), Some("required"));
        let tools: Value = serde_json::from_str(&req.tools_json.unwrap()).unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 1);
        assert_eq!(tools[0]["function"]["name"], "b");
    }

    #[test]
    fn tool_choice_none_disables_tools() {
        let input = json!({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "a"}}],
            "tool_choice": "none"
        });
        assert!(!chat_request(&input).unwrap().has_tools());
    }

    #[test]
    fn response_format_becomes_json_schema() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
        let input = json!({
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}
        });
        let req = chat_request(&input).unwrap();
        let parsed: Value = serde_json::from_str(&req.json_schema.unwrap()).unwrap();
        assert_eq!(parsed, schema);

        let input = json!({
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"}
        });
        assert_eq!(
            chat_request(&input).unwrap().json_schema.as_deref(),
            Some(r#"{"type":"object"}"#)
        );
    }

    #[test]
    fn sampling_prefers_request_over_cargo_defaults() {
        let p = sampling_params(
            &json!({"temperature": 0.2, "top_p": 0.5, "seed": 7, "max_tokens": 64, "stop": ["END"]}),
            Some(0.9),
        );
        assert_eq!(p.temperature, 0.2);
        assert_eq!(p.top_p, 0.5);
        assert_eq!(p.seed, 7);
        assert_eq!(p.max_tokens, 64);
        assert_eq!(p.stop, vec!["END".to_string()]);

        let p = sampling_params(&json!({"stop": "\n\n"}), Some(0.9));
        assert_eq!(p.temperature, 0.9);
        assert_eq!(p.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(p.stop, vec!["\n\n".to_string()]);
    }

    #[test]
    fn max_completion_tokens_wins() {
        let p = sampling_params(
            &json!({"max_tokens": 10, "max_completion_tokens": 20}),
            None,
        );
        assert_eq!(p.max_tokens, 20);
    }

    #[test]
    fn stop_scanner_without_stops_passes_through() {
        let mut s = StopScanner::new(vec![]);
        assert_eq!(s.push("Hello"), "Hello");
        assert_eq!(s.push(" world"), " world");
        assert_eq!(s.finish(), "");
        assert!(!s.stopped());
    }

    #[test]
    fn stop_scanner_cuts_at_stop_split_across_pieces() {
        let mut s = StopScanner::new(vec!["<END>".to_string()]);
        assert_eq!(s.push("Done <E"), "Done ");
        assert_eq!(s.push("ND> extra"), "");
        assert!(s.stopped());
        assert_eq!(s.text(), "Done ");
        assert_eq!(s.push("more"), "");
    }

    #[test]
    fn stop_scanner_releases_false_starts() {
        let mut s = StopScanner::new(vec!["<END>".to_string()]);
        assert_eq!(s.push("a <E"), "a ");
        assert_eq!(s.push("xample"), "<Example");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn stop_scanner_flushes_held_text_at_finish() {
        let mut s = StopScanner::new(vec!["###".to_string()]);
        assert_eq!(s.push("total #"), "total ");
        assert_eq!(s.finish(), "#");
    }

    #[test]
    fn stop_scanner_handles_multibyte_text() {
        let mut s = StopScanner::new(vec!["ß!".to_string()]);
        assert_eq!(s.push("Grüß"), "Grü");
        assert_eq!(s.push("e"), "ße");
        assert_eq!(s.push("ß!"), "");
        assert_eq!(s.text(), "Grüße");
    }

    #[test]
    fn finish_reasons() {
        assert_eq!(finish_reason(true, true), "tool_calls");
        assert_eq!(finish_reason(false, true), "length");
        assert_eq!(finish_reason(false, false), "stop");
    }
}
