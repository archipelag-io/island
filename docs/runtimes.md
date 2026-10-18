# Native ML Runtimes

The Archipelag.io Island software includes optional native ML runtimes that execute inference directly on the Island's hardware, without requiring Docker. Each runtime is gated behind a Cargo feature flag and only compiled when explicitly enabled.

## Runtime Dispatch

When the Island receives a job via NATS, it routes execution based on the `runtime_type` field in the `AssignJob` message. The dispatch logic in `agent.rs` works as follows:

| `runtime_type` | Feature flag | Handler |
|----------------|-------------|---------|
| `"wasm"` | always available | `execute_wasm_job` |
| `"onnx"` | `onnx` | `onnx::execute_onnx_job` |
| `"llmcpp"` | `gguf` | `gguf::execute_gguf_job` |
| `"diffusers"` | `diffusers` | `diffusers::execute_diffusers_job` |
| anything else | none | Falls through to Docker container execution |

If a runtime feature is not compiled in, the `#[cfg(feature = "...")]` match arm is excluded at compile time. This means that a job with `runtime_type: "onnx"` sent to an Island built without `--features onnx` will fall through to the catch-all arm and attempt Docker container execution (which requires Docker to be available). If Docker is also unavailable, the job fails with an error.

## Building with Runtimes

By default, no native ML runtimes are included. Enable them individually or all at once:

```bash
# Individual runtimes
cargo build --release --features onnx
cargo build --release --features gguf
cargo build --release --features diffusers

# All runtimes (includes onnx, gguf, diffusers, pipeline)
cargo build --release --features all-runtimes

# Combine specific runtimes
cargo build --release --features "onnx,gguf"
```

Each feature pulls in its own dependencies:

| Feature | Key dependencies |
|---------|-----------------|
| `onnx` | `ort` (ONNX Runtime), `ndarray`, `tokenizers`, `image`, `base64` |
| `gguf` | `llama-cpp-2` (llama.cpp bindings) |
| `diffusers` | `candle-core`, `candle-transformers`, `candle-nn`, `tokenizers`, `image`, `base64`, `hf-hub` |
| `all-runtimes` | All of the above plus `pipeline` |

## ONNX Runtime

**Source:** `src/onnx.rs`

Executes ONNX model inference using the `ort` crate (bindings to Microsoft ONNX Runtime). Supports both text and vision tasks.

### Supported Task Types

| Task type | Input fields | Output |
|-----------|-------------|--------|
| `text-classification` | `{"text": "..."}` | `{label, score, scores, task}` |
| `fill-mask` | `{"text": "..."}` | Same as text-classification |
| `token-classification` | `{"text": "..."}` | Same as text-classification |
| `zero-shot-classification` | `{"text": "..."}` | Same as text-classification |
| `feature-extraction` | `{"text": "..."}` | `{embedding, dimensions, task}` |
| `question-answering` | `{"question": "...", "context": "..."}` | `{answer, score, start, end, task}` |
| `object-detection` | `{"image": "<base64>"}` | `{detections, count, task}` |
| `image-segmentation` | `{"image": "<base64>"}` | `{output_shape, num_classes, original_size, task}` |

Unrecognized task types fall back to text-classification.

### AssignJob Fields

| Field | Required | Description |
|-------|----------|-------------|
| `onnx_model_url` | yes | Model URI (`hf://repo_id`, `hf://repo_id:file`, or `https://...`) |
| `onnx_model_hash` | no | Expected SHA256 hash for verification (`sha256:...`) |
| `onnx_task_type` | no | Task type string (defaults to `"text-classification"`) |
| `input` | yes | JSON object with task-specific fields (see table above) |

### Behavior

1. Downloads the ONNX model via the model cache (see [model-cache.md](model-cache.md)).
2. For text tasks, automatically downloads `tokenizer.json` from the same HuggingFace repo.
3. Creates an ONNX Runtime session with `intra_threads` set to `min(num_cpus, 4)`.
4. Runs inference through the task-specific handler.
5. Publishes the result as a single output chunk with `is_final: true`.

### Object Detection Details

- Input images are resized to 640x640 and normalized to [0, 1].
- Pixel layout is CHW (channels-first) as expected by YOLO-style models.
- The `threshold` field in input controls detection confidence cutoff (default: 0.5).
- Returns up to 1000 detections.

## GGUF Runtime (llmcpp)

**Source:** `src/gguf.rs`, `src/chat.rs`

Executes GGUF-format language models with `llama-cpp-2` (Rust bindings to llama.cpp). Streams output back to the Consumer in real time, including tool calls and reasoning.

### AssignJob Fields

| Field | Required | Description |
|-------|----------|-------------|
| `model_url` | yes | Model URI (`hf://repo_id`, `hf://repo_id:file.gguf`, or `https://...`) |
| `model_hash` | no | Expected SHA256 hash for verification |
| `model_context_size` | no | Context window size (default: 4096, capped at the model's training context) |
| `model_temperature` | no | Cargo default temperature (default: 0.7) |
| `input` | yes | JSON with the request (see below) |

### Input Format

**Chat (`messages`)**: an OpenAI-compatible request. The messages are rendered with the model's own chat template (Jinja, from the GGUF metadata), so multi-turn history, `tool` results and assistant `tool_calls` reach the model in the format it was trained on. Supported fields:

| Field | Description |
|-------|-------------|
| `messages` | OpenAI messages, including `tool_calls` and `tool_call_id` |
| `tools` | OpenAI function definitions |
| `tool_choice` | `auto`, `none`, `required`, or `{"type":"function","function":{"name":...}}` |
| `parallel_tool_calls` | Allow several calls in one turn (default `false`) |
| `response_format` | `{"type":"json_object"}` or `{"type":"json_schema","json_schema":{"schema":...}}` |
| `enable_thinking` | Let reasoning models think first (default `true`) |
| `temperature`, `top_p`, `seed` | Sampling; `temperature: 0` is greedy |
| `max_tokens` / `max_completion_tokens` | Generation limit (default 1024) |
| `stop` | String or array of stop sequences |

When tools or a JSON schema are present, llama.cpp generates a grammar that keeps the output valid. For tools it is lazy: it only engages once the model starts a tool call, so ordinary answers are unaffected. Models without a chat template fall back to ChatML.

**Prompt (`prompt` or `text`)**: raw text passed to the model as-is, with the same sampling fields.

### Output

Each NATS output message is `{job_id, seq, chunk, is_final}`:

- Answer text arrives in `chunk`, as before.
- Reasoning and tool-call fragments arrive with an empty `chunk` and a `delta` field holding an OpenAI streaming delta, e.g. `{"reasoning_content": "..."}` or `{"tool_calls": [{"index": 0, "id": "...", "function": {"name": "get_weather", "arguments": "{\"ci"}}]}`.
- The final message (`is_final: true`) has a JSON `chunk`:

```json
{
  "type": "done",
  "finish_reason": "stop | length | tool_calls",
  "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
  "message": {"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]}
}
```

`message` is the parsed final assistant message (chat inputs only) and is the authoritative source for tool calls.

### Behavior

1. Downloads the GGUF model file via the model cache.
2. Loads the model, or reuses it if the previous Job used the same file. One model stays loaded; llama.cpp's backend is initialized once per process.
3. Renders the prompt and rejects it with a clear error if it does not fit the context.
4. Reads the prompt in 512-token batches, then samples with top-k 40, top-p, min-p 0.05, temperature and a per-Job seed, with the grammar first when one is set.
5. Stops at any end-of-generation token, a stop sequence, cancellation, or the token limit. Text that could be the start of a stop sequence is held back until it is clear it isn't.
6. Streams output through a 256-slot channel, so back-pressure applies if NATS falls behind.

### Testing with a real model

The unit tests run without a model. To exercise generation, tool calling, reasoning, stop sequences and JSON output end to end:

```bash
ISLAND_TEST_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf cargo test --features gguf gguf::tests -- --include-ignored --test-threads=1
```

## Diffusers Runtime

**Source:** `src/diffusers.rs`

Executes Stable Diffusion text-to-image pipelines using the `candle` framework (Rust ML library by Hugging Face). Currently configured for Stable Diffusion v1.5 architecture.

### AssignJob Fields

| Field | Required | Description |
|-------|----------|-------------|
| `model_url` | yes | HuggingFace repo URI (e.g., `hf://runwayml/stable-diffusion-v1-5`) |
| `input` | yes | JSON with generation parameters (see below) |

### Input Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | (required) | Text prompt for image generation |
| `negative_prompt` | string | `""` | Negative prompt for classifier-free guidance |
| `num_steps` | integer | 20 | Number of diffusion steps |
| `guidance_scale` | float | 7.5 | Classifier-free guidance scale |
| `width` | integer | 512 | Output image width in pixels |
| `height` | integer | 512 | Output image height in pixels |
| `seed` | integer | 42 | Random seed for reproducibility |

### Behavior

1. Downloads four model components from HuggingFace Hub via the `hf-hub` crate:
   - `tokenizer.json` (CLIP tokenizer)
   - `text_encoder/model.safetensors` (CLIP text encoder)
   - `vae/diffusion_pytorch_model.safetensors` (VAE decoder)
   - `unet/diffusion_pytorch_model.safetensors` (UNet denoiser)
2. Tokenizes both the prompt and negative prompt (padded to 77 tokens, CLIP max length).
3. Encodes text embeddings with the CLIP text encoder, then frees CLIP memory.
4. Initializes random latents in the latent space (height/8 x width/8, 4 channels).
5. Runs the diffusion loop for `num_steps` iterations with classifier-free guidance.
6. Decodes the final latents through the VAE.
7. Encodes the output as a PNG image.

### Output

The final output is a JSON object:

```json
{
  "type": "image",
  "format": "png",
  "width": 512,
  "height": 512,
  "data": "<base64-encoded PNG>",
  "size_bytes": 123456
}
```

### Progress Events

During execution, the runtime publishes progress events to NATS:

- **Download phase:** `{"type": "progress", "phase": "downloading", "step": 0, "total": N}`
- **Diffusion steps:** `{"type": "progress", "phase": "diffusion", "step": 1, "total": 20}` (one per step)

### Notes

- All inference currently runs on CPU (`Device::Cpu`) with `DType::F32`.
- The CLIP text encoder is dropped after encoding to free memory before loading the UNet and VAE.
- Cancellation is checked before download, after download, and at each diffusion step.

## Job Status Protocol

All runtimes use the same NATS status protocol. The coordinator expects these exact state strings:

| Status | When Sent | Coordinator Transition |
|--------|-----------|----------------------|
| `"started"` | Before inference begins | `assigned → started` |
| `"succeeded"` | After inference completes successfully | `started → succeeded` |
| `"failed"` | On any error | `started → failed` |
| `"cancelled"` | If cancel signal received | `started → cancelled` |

**Important:** The status must be `"succeeded"`, not `"completed"`. The coordinator's state machine rejects `"completed"` as an invalid transition. This was a bug fixed in commit `fff32bf`.

Output chunks are published to `host.<host_id>.output` and status updates to `host.<host_id>.status`. Both subjects are captured by JetStream streams (`JOB_OUTPUT` and `JOB_STATUS`).

## Model Resolution

All three runtimes use the model cache for downloading and caching models. The `hf://` URI scheme is the standard way to reference models. See [model-cache.md](model-cache.md) for full details on URI resolution, caching, and eviction.

## Common Gotchas

- **Build deps:** GGUF requires `cmake` and `libclang-dev` on the build machine (for llama.cpp compilation via bindgen).
- **CPU performance:** Mistral 7B on CPU generates ~1 token/second on a 12-core machine. TinyLlama 1.1B is much faster (~30 tokens/second). For demos, prefer smaller models.
- **VRAM requirements:** GGUF models run on CPU, so their `required_vram_mb` in the DB should be `NULL` (not a GPU VRAM number). Otherwise the coordinator's placement query rejects CPU-only Islands.
- **Gated HuggingFace models:** Some models (e.g., Qwen3.5) require an HF token. Set `hf_token` in `[model_cache]` config or the `HF_TOKEN` env var. The default preload list uses only public models.
