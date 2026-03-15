# Startup Preloading

**Source:** `src/preload.rs`

When an Archipelag.io Island starts up, it can automatically preload ML models into the local cache based on the machine's hardware capabilities. This ensures that common models are ready to serve immediately when jobs arrive, avoiding cold-start download delays.

## How It Works

1. At startup, the Island detects available RAM and VRAM.
2. `select_starter_models()` chooses which models to preload based on a hardware tier system.
3. `preload_models()` downloads each selected model through the model cache (see [model-cache.md](model-cache.md)).
4. Preloading runs in the background and is non-blocking -- the Island begins accepting jobs immediately, even before preloading finishes.
5. If a preload download fails, it is logged as a warning but does not prevent the Island from operating. Models that fail to preload will be downloaded on-demand when a job requires them.

## Hardware Tiers

The preloader classifies the Island's hardware into one of six tiers and selects starter models accordingly:

| Tier | RAM | VRAM | Preloaded Models |
|------|-----|------|-----------------|
| **Tiny** | 2 GB or less | -- | None (WASM-only Island) |
| **Small** | 2--8 GB | -- | Qwen3.5-0.8B (GGUF, ~600 MB) |
| **Medium** | 8--16 GB | -- | + Mistral 7B Instruct (GGUF, ~4.4 GB), DistilBERT Sentiment (ONNX), MiniLM-L6 Embeddings (ONNX) |
| **GPU** | 8+ GB | 2--8 GB | + Whisper Base (ONNX), YOLOv8n Detection (ONNX) |
| **Large** | 16+ GB | 8--12+ GB | + FLUX.1-schnell (Diffusers) |
| **XL** | 24+ GB | 24+ GB | + Qwen3.5-27B (GGUF) |

Tiers are cumulative -- a Large-tier Island gets everything from Small, Medium, and GPU tiers as well (subject to the final hardware filter).

### Feature-Gated Selection

Model entries are conditionally compiled with `#[cfg(feature = "...")]`:

- GGUF models (Qwen, Mistral) are only included when built with `--features gguf`
- ONNX models (DistilBERT, MiniLM, Whisper, YOLOv8) are only included when built with `--features onnx`
- Diffusers models (FLUX.1-schnell) are only included when built with `--features diffusers`

An Island built without any runtime features will preload nothing regardless of hardware.

### Hardware Filter

After tier-based selection, a final filter removes any model whose minimum requirements exceed the actual hardware:

```
models.retain(|m| ram_mb >= m.min_ram_mb && vram >= m.min_vram_mb)
```

This means a machine with 12 GB RAM and no GPU will get Small + Medium tier GGUF/ONNX models, but not GPU-tier or Large-tier models.

## Configuration

Preload settings are in the `[preload]` section of `config.toml`:

```toml
[preload]
# Enable automatic model preloading at startup (default: true)
enabled = true

# Explicit list of models to preload (overrides auto-selection)
# If empty, auto-selects based on hardware capabilities
# models = ["hf://Qwen/Qwen3.5-0.8B-GGUF", "hf://openai/whisper-base"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable or disable preloading entirely |
| `models` | `Vec<String>` | `[]` (empty) | Explicit model URIs to preload |

### Auto-Selection vs. Explicit List

- **Empty `models` list (default):** The preloader auto-selects models based on the hardware tier table above.
- **Non-empty `models` list:** The preloader ignores the hardware tier system entirely and downloads exactly the listed models. No hardware filtering is applied -- you are responsible for ensuring the models fit in memory.

```toml
[preload]
enabled = true
models = [
  "hf://TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf",
  "hf://sentence-transformers/all-MiniLM-L6-v2",
]
```

### Disabling Preloading

To skip preloading entirely (for example, on a machine with limited bandwidth):

```toml
[preload]
enabled = false
```

## On-Demand Downloads

Models that are not preloaded are downloaded on-demand when a job arrives that needs them. The flow is:

1. Job arrives with a `model_url` or `onnx_model_url` field.
2. The runtime calls `model_cache.download_model(uri, hash)`.
3. If the model is already cached (from preloading or a previous job), it is returned immediately.
4. If not cached, it is downloaded, verified, and cached before inference begins.

This means preloading is purely an optimization. An Island can serve any model its compiled runtimes support, whether or not it was preloaded.

## PreloadEntry Structure

Each model to preload is described by:

| Field | Type | Description |
|-------|------|-------------|
| `uri` | `String` | HuggingFace URI (e.g., `hf://Qwen/Qwen3.5-0.8B-GGUF`) |
| `name` | `String` | Human-readable name for logging |
| `min_ram_mb` | `u32` | Minimum system RAM in MB |
| `min_vram_mb` | `u32` | Minimum GPU VRAM in MB (0 = CPU-only) |
| `runtime` | `&'static str` | Runtime type: `"llmcpp"`, `"onnx"`, or `"diffusers"` |
