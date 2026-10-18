//! Native GGUF runtime for executing llama.cpp workloads
//!
//! Downloads GGUF model files, loads them with llama-cpp-2, and streams
//! token-by-token output back through NATS.
//!
//! Chat inputs (`messages`) are rendered with the model's own chat template,
//! so multi-turn history, tool definitions and tool results reach the model in
//! the format it was trained on. When tools or a JSON schema are present,
//! llama.cpp supplies a grammar that keeps the output well-formed, and its
//! parser turns the raw text into OpenAI-style deltas (content, reasoning,
//! tool calls).

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, ChatTemplateResult, GrammarTriggerType, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::{watch, RwLock};
use tracing::{info, warn};

use crate::chat::{self, ChatRequest, SamplingParams, StopScanner};
use crate::config::AgentConfig;
use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Context size when the Cargo does not set one
const DEFAULT_CONTEXT_SIZE: u32 = 4096;
/// Tokens decoded per batch while reading the prompt
const PROMPT_BATCH: usize = 512;

/// llama.cpp's backend may only be initialized once per process
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// The most recently used model stays loaded, so back-to-back Jobs on the
/// same Cargo (an agent's tool loop, a chat) skip the load.
static LOADED_MODEL: Mutex<Option<(PathBuf, Arc<LlamaModel>)>> = Mutex::new(None);

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let backend = LlamaBackend::init()
        .map_err(|e| anyhow::anyhow!("Failed to init llama backend: {:?}", e))?;
    Ok(BACKEND.get_or_init(|| backend))
}

fn load_model(path: &Path) -> Result<Arc<LlamaModel>> {
    let mut slot = LOADED_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((loaded_path, model)) = slot.as_ref() {
        if loaded_path == path {
            return Ok(model.clone());
        }
    }
    // Release the previous model before loading the next one
    *slot = None;
    let model = LlamaModel::load_from_file(backend()?, path, &LlamaModelParams::default())
        .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {:?}", e))?;
    let model = Arc::new(model);
    *slot = Some((path.to_path_buf(), model.clone()));
    Ok(model)
}

/// Drop the cached model. Call before exit: llama.cpp's Metal backend
/// aborts if a model is still alive when the process tears it down.
pub fn unload_model() {
    let mut slot = LOADED_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    *slot = None;
}

/// What the generation thread sends back to the async side
enum Event {
    /// Visible answer text
    Text(String),
    /// A structured delta (reasoning, tool calls) as OpenAI JSON
    Delta(Value),
}

/// Result of a finished generation
struct Generation {
    prompt_tokens: usize,
    completion_tokens: u64,
    finish_reason: &'static str,
    /// Parsed final assistant message (chat inputs only)
    message: Option<Value>,
}

/// Final output chunk. A struct rather than `json!` so `"type":"done"`
/// stays first: serde_json sorts map keys, and the web chat finds the
/// marker by its `{"type":"done"` prefix.
#[derive(serde::Serialize)]
struct DoneEvent<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    finish_reason: &'static str,
    usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<&'a Value>,
}

#[derive(serde::Serialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

fn done_event(gen: &Generation) -> String {
    let prompt_tokens = gen.prompt_tokens as u64;
    let event = DoneEvent {
        kind: "done",
        finish_reason: gen.finish_reason,
        usage: Usage {
            prompt_tokens,
            completion_tokens: gen.completion_tokens,
            total_tokens: prompt_tokens + gen.completion_tokens,
        },
        message: gen.message.as_ref(),
    };
    serde_json::to_string(&event).unwrap_or_else(|_| r#"{"type":"done"}"#.to_string())
}

/// Execute a GGUF/llmcpp workload
pub async fn execute_gguf_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    _config: &AgentConfig,
    job: &AssignJob,
    cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let model_url = job
        .model_url
        .as_ref()
        .context("GGUF workload missing model_url")?;

    let context_size = job.model_context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
    let sampling = chat::sampling_params(&job.input, job.model_temperature);
    let chat_request = chat::chat_request(&job.input);

    info!(
        "Executing GGUF workload: {} (ctx={}, temp={}, max_tokens={}, chat={}, tools={})",
        model_url,
        context_size,
        sampling.temperature,
        sampling.max_tokens,
        chat_request.is_some(),
        chat_request.as_ref().is_some_and(ChatRequest::has_tools)
    );

    // Download model via cache
    let model_cache = {
        let st = state.read().await;
        st.model_cache()
            .context("Model cache not initialized")?
            .clone()
    };

    let model_path = model_cache
        .download_model(model_url, job.model_hash.as_deref())
        .await
        .with_context(|| format!("Failed to download GGUF model: {}", model_url))?;

    // Check cancellation
    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    let prompt = match chat_request {
        Some(_) => None,
        None => Some(extract_prompt(&job.input)?),
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Event>(256);
    let cancel = cancel_rx.clone();
    let generate_handle = tokio::task::spawn_blocking(move || {
        generate(
            &model_path,
            context_size,
            chat_request,
            prompt,
            sampling,
            &tx,
            &cancel,
        )
    });

    // Stream output as it arrives. Text goes in `chunk` as before; structured
    // deltas ride alongside with an empty chunk.
    let mut seq: u64 = 0;
    while let Some(event) = rx.recv().await {
        match event {
            Event::Text(text) => nats.publish_output(job_id, seq, &text, false).await?,
            Event::Delta(delta) => nats.publish_output_delta(job_id, seq, delta, false).await?,
        }
        seq += 1;
    }

    match generate_handle.await? {
        Ok(gen) => {
            nats.publish_output(job_id, seq, &done_event(&gen), true)
                .await?;
            if *cancel_rx.borrow() {
                nats.publish_status(job_id, "cancelled", None).await?;
            } else {
                nats.publish_status(job_id, "succeeded", None).await?;
            }
            info!(
                "GGUF job {} completed: {} tokens ({})",
                job_id, gen.completion_tokens, gen.finish_reason
            );
        }
        Err(e) => {
            nats.publish_status(job_id, "failed", Some(format!("GGUF error: {}", e)))
                .await?;
        }
    }

    Ok(())
}

/// Render the chat template; `None` when the model ships without one
fn render_chat(model: &LlamaModel, request: &ChatRequest) -> Result<Option<ChatTemplateResult>> {
    let template = match model.chat_template(None) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let params = OpenAIChatTemplateParams {
        messages_json: &request.messages_json,
        tools_json: if request.has_tools() {
            request.tools_json.as_deref()
        } else {
            None
        },
        tool_choice: request.tool_choice.as_deref(),
        json_schema: request.json_schema.as_deref(),
        grammar: None,
        reasoning_format: Some("auto"),
        chat_template_kwargs: None,
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: request.parallel_tool_calls,
        enable_thinking: request.enable_thinking,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: request.has_tools(),
    };
    let rendered = model
        .apply_chat_template_oaicompat(&template, &params)
        .map_err(|e| anyhow::anyhow!("Failed to apply chat template: {:?}", e))?;
    Ok(Some(rendered))
}

/// Grammar sampler for tool calls / JSON output, mirroring llama-server:
/// lazy grammars only engage once the model starts a tool call.
fn grammar_sampler(
    model: &LlamaModel,
    rendered: &ChatTemplateResult,
) -> Result<Option<LlamaSampler>> {
    let Some(grammar) = rendered.grammar.as_deref() else {
        return Ok(None);
    };
    if !rendered.grammar_lazy {
        let sampler = LlamaSampler::grammar(model, grammar, "root")
            .map_err(|e| anyhow::anyhow!("Invalid grammar: {:?}", e))?;
        return Ok(Some(sampler));
    }

    // Same trigger handling as llama.cpp's common/sampling.cpp
    let mut patterns = Vec::new();
    let mut trigger_tokens = Vec::new();
    for trigger in &rendered.grammar_triggers {
        match trigger.trigger_type {
            GrammarTriggerType::Token => trigger_tokens.extend(trigger.token),
            GrammarTriggerType::Word => patterns.push(regex_escape(&trigger.value)),
            GrammarTriggerType::Pattern => patterns.push(trigger.value.clone()),
            GrammarTriggerType::PatternFull => patterns.push(anchor_pattern(&trigger.value)),
        }
    }
    let sampler =
        LlamaSampler::grammar_lazy_patterns(model, grammar, "root", &patterns, &trigger_tokens)
            .map_err(|e| anyhow::anyhow!("Invalid lazy grammar: {:?}", e))?;
    Ok(Some(sampler))
}

/// Anchor a full-match trigger pattern at both ends
fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let start = if pattern.starts_with('^') { "" } else { "^" };
    let end = if pattern.ends_with('$') { "" } else { "$" };
    format!("{start}{pattern}{end}")
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Special tokens the response parser needs to see (e.g. `<tool_call>`)
fn preserved_token_ids(model: &LlamaModel, preserved: &[String]) -> HashSet<LlamaToken> {
    preserved
        .iter()
        .filter_map(|text| match model.str_to_token(text, AddBos::Never) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        })
        .collect()
}

fn generate(
    model_path: &Path,
    context_size: u32,
    chat_request: Option<ChatRequest>,
    prompt: Option<String>,
    sampling: SamplingParams,
    tx: &tokio::sync::mpsc::Sender<Event>,
    cancel: &watch::Receiver<bool>,
) -> Result<Generation> {
    let model = load_model(model_path)?;
    let context_size = context_size.min(model.n_ctx_train().max(512));

    // Render the prompt
    let rendered = match &chat_request {
        Some(request) => render_chat(&model, request)?,
        None => None,
    };
    let prompt_text = match (&rendered, &chat_request, prompt) {
        (Some(r), _, _) => r.prompt.clone(),
        (None, Some(_), _) => {
            warn!("Model has no chat template; falling back to ChatML");
            let messages: Value =
                serde_json::from_str(&chat_request.as_ref().unwrap().messages_json)?;
            extract_prompt(&json!({ "messages": messages }))?
        }
        (None, None, Some(p)) => p,
        (None, None, None) => anyhow::bail!("No prompt to generate from"),
    };

    let mut stops = sampling.stop.clone();
    let mut keep_special = HashSet::new();
    let mut grammar = None;
    let mut parser = None;
    if let Some(r) = &rendered {
        stops.extend(r.additional_stops.iter().cloned());
        keep_special = preserved_token_ids(&model, &r.preserved_tokens);
        grammar = grammar_sampler(&model, r)?;
        parser = Some(
            r.streaming_state_oaicompat()
                .map_err(|e| anyhow::anyhow!("Failed to init response parser: {:?}", e))?,
        );
    }

    // Tokenize and check the prompt fits, leaving room for the answer
    let tokens = model
        .str_to_token(&prompt_text, AddBos::Always)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {:?}", e))?;
    let n_ctx = context_size as usize;
    if tokens.len() + 16 > n_ctx {
        anyhow::bail!(
            "Prompt is {} tokens, but the context holds {}; shorten the conversation",
            tokens.len(),
            n_ctx
        );
    }
    let max_tokens = sampling.max_tokens.min(n_ctx - tokens.len());

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(context_size))
        .with_n_batch(PROMPT_BATCH as u32);
    let mut ctx = model
        .new_context(backend()?, ctx_params)
        .map_err(|e| anyhow::anyhow!("Failed to create llama context: {:?}", e))?;

    // Read the prompt in batches; only the last token needs logits
    let mut batch = LlamaBatch::new(PROMPT_BATCH, 1);
    for (chunk_index, chunk) in tokens.chunks(PROMPT_BATCH).enumerate() {
        batch.clear();
        for (i, &token) in chunk.iter().enumerate() {
            let pos = chunk_index * PROMPT_BATCH + i;
            batch
                .add(token, pos as i32, &[0], pos == tokens.len() - 1)
                .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {:?}", e))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Failed to decode prompt: {:?}", e))?;
        if *cancel.borrow() {
            break;
        }
    }

    // Sampler chain: grammar first so constrained output is always valid
    let mut chain: Vec<LlamaSampler> = grammar.into_iter().collect();
    if sampling.temperature <= 0.0 {
        chain.push(LlamaSampler::greedy());
    } else {
        chain.extend([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(sampling.top_p, 1),
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(sampling.temperature),
            LlamaSampler::dist(sampling.seed),
        ]);
    }
    let mut sampler = LlamaSampler::chain_simple(chain);

    let mut scanner = StopScanner::new(stops);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut completion_tokens: u64 = 0;
    let mut hit_limit = true;

    let emit = |text: String, parser: &mut Option<_>, is_partial: bool| -> bool {
        match parser {
            Some(p) => {
                match llama_cpp_2::openai::ChatParseStateOaicompat::update(p, &text, is_partial) {
                    Ok(deltas) => deltas.iter().all(|d| send_delta(tx, d)),
                    Err(e) => {
                        warn!("Response parser failed, sending raw text: {:?}", e);
                        text.is_empty() || tx.blocking_send(Event::Text(text)).is_ok()
                    }
                }
            }
            None => text.is_empty() || tx.blocking_send(Event::Text(text)).is_ok(),
        }
    };

    for n_cur in tokens.len()..tokens.len() + max_tokens {
        if *cancel.borrow() {
            hit_limit = false;
            break;
        }

        // sample() also records the token with every sampler (grammar
        // included); accepting it again would feed the grammar twice
        let token = sampler.sample(&ctx, -1);

        if model.is_eog_token(token) {
            hit_limit = false;
            break;
        }
        completion_tokens += 1;

        let piece = model
            .token_to_piece(token, &mut decoder, keep_special.contains(&token), None)
            .unwrap_or_default();
        let ready = scanner.push(&piece);
        if !ready.is_empty() && !emit(ready, &mut parser, true) {
            break;
        }
        if scanner.stopped() {
            hit_limit = false;
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[0], true)
            .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {:?}", e))?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Failed to decode token: {:?}", e))?;
    }

    // Flush held-back text and let the parser close any open structure
    let rest = scanner.finish();
    emit(rest, &mut parser, false);

    let message = match &rendered {
        Some(r) => r
            .parse_response_oaicompat(scanner.text(), false)
            .ok()
            .and_then(|m| serde_json::from_str::<Value>(&m).ok()),
        None => None,
    };
    let has_tool_calls = message
        .as_ref()
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());

    Ok(Generation {
        prompt_tokens: tokens.len(),
        completion_tokens,
        finish_reason: chat::finish_reason(has_tool_calls, hit_limit),
        message,
    })
}

/// Forward one parser delta: answer text as plain text, anything else
/// (reasoning, tool calls) as structured JSON. Returns false if the
/// receiver has gone away.
fn send_delta(tx: &tokio::sync::mpsc::Sender<Event>, delta_json: &str) -> bool {
    let Ok(Value::Object(mut delta)) = serde_json::from_str::<Value>(delta_json) else {
        return true;
    };
    if let Some(Value::String(text)) = delta.remove("content") {
        if !text.is_empty() && tx.blocking_send(Event::Text(text)).is_err() {
            return false;
        }
    }
    delta.retain(|_, v| !v.is_null());
    if delta.is_empty() {
        return true;
    }
    tx.blocking_send(Event::Delta(Value::Object(delta))).is_ok()
}

/// Extract prompt string from job input JSON
pub fn extract_prompt(input: &serde_json::Value) -> Result<String> {
    // Try "prompt" field first
    if let Some(prompt) = input.get("prompt").and_then(|v| v.as_str()) {
        return Ok(prompt.to_string());
    }

    // Try "messages" array (OpenAI chat format). Used only for models that
    // ship without a chat template, and by the distributed runtimes.
    if let Some(messages) = input.get("messages").and_then(|v| v.as_array()) {
        let mut prompt = String::new();
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
            match role {
                "system" => {
                    prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", content));
                }
                "user" => {
                    prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", content));
                }
                "assistant" => {
                    prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", content));
                }
                _ => {
                    prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
            }
        }
        // Add assistant opening
        prompt.push_str("<|im_start|>assistant\n");
        return Ok(prompt);
    }

    // Try "text" field
    if let Some(text) = input.get("text").and_then(|v| v.as_str()) {
        return Ok(text.to_string());
    }

    anyhow::bail!("GGUF workload requires 'prompt', 'messages', or 'text' in input")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a generation against a real model. Set ISLAND_TEST_GGUF to a
    /// tool-capable chat model (e.g. Qwen3-0.6B-Q8_0.gguf) and run with
    /// `cargo test --features gguf -- --ignored`.
    fn run_model(input: Value) -> Option<(Generation, Vec<Event>)> {
        let path = PathBuf::from(std::env::var("ISLAND_TEST_GGUF").ok()?);
        let (tx, mut rx) = tokio::sync::mpsc::channel(100_000);
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let sampling = chat::sampling_params(&input, None);
        let request = chat::chat_request(&input);
        let prompt = if request.is_none() {
            Some(extract_prompt(&input).unwrap())
        } else {
            None
        };
        let gen = generate(&path, 4096, request, prompt, sampling, &tx, &cancel_rx).unwrap();
        unload_model();
        drop(tx);
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        Some((gen, events))
    }

    fn text_of(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Text(t) => Some(t.as_str()),
                Event::Delta(_) => None,
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn model_answers_multi_turn_chat() {
        let Some((gen, events)) = run_model(json!({
            "messages": [
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Nice to meet you, Ada."},
                {"role": "user", "content": "What is my name?"}
            ],
            "enable_thinking": false,
            "temperature": 0,
            "max_tokens": 64
        })) else {
            return;
        };
        let text = text_of(&events);
        println!("answer: {text:?} finish={}", gen.finish_reason);
        assert!(text.contains("Ada"), "{text}");
        assert!(!text.contains("<|im_end|>"));
        assert_eq!(gen.finish_reason, "stop");
        assert_eq!(gen.message.unwrap()["role"], "assistant");
    }

    #[test]
    #[ignore]
    fn model_calls_a_tool_and_uses_the_result() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]);
        let Some((gen, events)) = run_model(json!({
            "messages": [{"role": "user", "content": "What's the weather in Oslo right now?"}],
            "tools": tools,
            "enable_thinking": false,
            "temperature": 0,
            "max_tokens": 256
        })) else {
            return;
        };
        let message = gen.message.expect("parsed message");
        println!("message: {message}");
        assert_eq!(gen.finish_reason, "tool_calls");
        let call = &message["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert!(args["city"].as_str().unwrap().contains("Oslo"));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Delta(d) if d.get("tool_calls").is_some())));

        // Second turn: feed the tool result back
        let call_id = call["id"].as_str().unwrap_or("call_0").to_string();
        let Some((gen, events)) = run_model(json!({
            "messages": [
                {"role": "user", "content": "What's the weather in Oslo right now?"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call_id, "content": "{\"temp_c\": 4, \"sky\": \"rain\"}"}
            ],
            "tools": tools,
            "enable_thinking": false,
            "temperature": 0,
            "max_tokens": 128
        })) else {
            return;
        };
        let text = text_of(&events);
        println!("final: {text:?}");
        assert_eq!(gen.finish_reason, "stop");
        assert!(text.contains('4'), "{text}");
    }

    #[test]
    #[ignore]
    fn model_reasoning_is_separated_from_the_answer() {
        let Some((gen, events)) = run_model(json!({
            "messages": [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
            "temperature": 0,
            "max_tokens": 2048
        })) else {
            return;
        };
        let text = text_of(&events);
        println!(
            "answer: {text:?} finish={} message={:?}",
            gen.finish_reason, gen.message
        );
        assert!(!text.contains("<think>"), "{text}");
        assert!(text.contains("42"), "{text}");
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Delta(d) if d.get("reasoning_content").is_some())));
        assert!(gen.message.unwrap().get("reasoning_content").is_some());
    }

    #[test]
    #[ignore]
    fn model_honours_stop_sequences_and_json_schema() {
        let Some((gen, events)) = run_model(json!({
            "messages": [{"role": "user", "content": "Count from 1 to 10 separated by spaces."}],
            "enable_thinking": false,
            "temperature": 0,
            "stop": [" 5"],
            "max_tokens": 64
        })) else {
            return;
        };
        let text = text_of(&events);
        println!("stopped: {text:?}");
        assert!(text.contains('4') && !text.contains('5'), "{text}");
        assert_eq!(gen.finish_reason, "stop");

        let Some((_, events)) = run_model(json!({
            "messages": [{"role": "user", "content": "Give me a person called Ada aged 36."}],
            "enable_thinking": false,
            "temperature": 0,
            "max_tokens": 128,
            "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"]
            }}}
        })) else {
            return;
        };
        let text = text_of(&events);
        println!("json: {text}");
        let v: Value = serde_json::from_str(text.trim()).expect("valid JSON");
        assert!(v["age"].is_i64());
    }

    #[test]
    fn regex_escape_escapes_metacharacters() {
        assert_eq!(regex_escape("<tool_call>"), "<tool_call>");
        assert_eq!(regex_escape("[TOOL_CALLS]"), "\\[TOOL_CALLS\\]");
        assert_eq!(regex_escape("a.b*c"), "a\\.b\\*c");
    }

    #[test]
    fn done_event_keeps_the_type_marker_first() {
        let gen = Generation {
            prompt_tokens: 10,
            completion_tokens: 5,
            finish_reason: "tool_calls",
            message: Some(json!({"role": "assistant", "content": "a\nb", "tool_calls": []})),
        };
        let chunk = done_event(&gen);
        assert!(chunk.starts_with(r#"{"type":"done","#), "{chunk}");
        assert!(!chunk.contains('\n'), "must stay on one line: {chunk}");
        let v: Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["finish_reason"], "tool_calls");
        assert_eq!(v["usage"]["total_tokens"], 15);
        assert_eq!(v["message"]["content"], "a\nb");
    }

    #[test]
    fn full_patterns_are_anchored_once() {
        assert_eq!(
            anchor_pattern("<tool_call>[\\s\\S]*"),
            "^<tool_call>[\\s\\S]*$"
        );
        assert_eq!(anchor_pattern("^already$"), "^already$");
        assert_eq!(anchor_pattern(""), "^$");
    }
}
