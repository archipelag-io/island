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
| `gguf` | `llama_cpp` (llama.cpp bindings) |
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

**Source:** `src/gguf.rs`

Executes GGUF-format language models using `llama_cpp` (Rust bindings to llama.cpp). Streams tokens back to the Consumer in real time.

### AssignJob Fields

| Field | Required | Description |
|-------|----------|-------------|
| `model_url` | yes | Model URI (`hf://repo_id`, `hf://repo_id:file.gguf`, or `https://...`) |
| `model_hash` | no | Expected SHA256 hash for verification |
| `model_context_size` | no | Context window size (default: 2048) |
| `model_temperature` | no | Sampling temperature (default: 0.7) |
| `input` | yes | JSON with prompt content (see below) |

### Input Format

The runtime accepts three prompt formats, checked in order:

1. **`prompt`** (string) -- Raw prompt text, passed directly to the model.
2. **`messages`** (array) -- OpenAI chat format. Converted to Llama 2 instruction format:
   - `system` messages become `[INST] <<SYS>>\n...\n<</SYS>>\n\n`
   - `user` messages become `[INST] ... [/INST]`
   - `assistant` messages become `... </s>`
3. **`text`** (string) -- Fallback, treated as raw prompt.

The `max_tokens` field inside `input` controls the maximum number of generated tokens (default: 1024).

### Behavior

1. Downloads the GGUF model file via the model cache.
2. Loads the model with `LlamaModel::load_from_file`.
3. Creates a session with the configured `context_size`.
4. Configures a sampler with repetition penalty (1.1), top-p (0.9), and the configured temperature.
5. Generates tokens in a `spawn_blocking` task, streaming each token through an `mpsc` channel.
6. Each token is published as a separate NATS output message with an incrementing sequence number.
7. On completion, publishes a final output with `{"type": "done", "usage": {"completion_tokens": N}}`.

### Streaming

Token streaming is real-time: each token is published to NATS as soon as it is generated. The async side reads from a 256-slot `mpsc` channel, so back-pressure is applied if the NATS publisher falls behind.

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

## Model Resolution

All three runtimes use the model cache for downloading and caching models. The `hf://` URI scheme is the standard way to reference models. See [model-cache.md](model-cache.md) for full details on URI resolution, caching, and eviction.
