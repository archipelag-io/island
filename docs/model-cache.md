# Model Cache and HuggingFace Resolution

**Source:** `src/model_cache.rs`

The model cache manages downloaded ML model files for the Archipelag.io Island's native runtimes (ONNX, GGUF, Diffusers). It handles URI resolution, streaming downloads with SHA256 verification, on-disk caching, and LRU eviction.

## URI Schemes

The cache accepts three URI formats:

### `hf://repo_id` -- Auto-discovery

Queries the HuggingFace Hub API to find the primary model file in the repository.

```
hf://distilbert-base-uncased-finetuned-sst-2-english
hf://TheBloke/Mistral-7B-Instruct-v0.2-GGUF
hf://runwayml/stable-diffusion-v1-5
```

The API call is:
```
GET https://huggingface.co/api/models/{repo_id}
User-Agent: archipelag-island/0.4
```

The response's `siblings` array (file listing) is scanned in priority order to find the main model file:

1. **GGUF files** (`.gguf`) -- Preferred quantization: `Q4_K_M` > `Q4_K_S` > `Q5_K_M` > `Q8_0` > first `.gguf`
2. **ONNX files** -- `model.onnx` > `onnx/model.onnx` > any `.onnx`
3. **Safetensors** -- `model.safetensors` > `diffusion_pytorch_model.safetensors`
4. **Legacy PyTorch** -- `pytorch_model.bin`
5. **Fallback** -- First file with a model extension (`.bin`, `.safetensors`, `.onnx`, `.gguf`, `.pt`)

If no model file is found, the download fails with an error listing the first 10 filenames in the repo.

### `hf://repo_id:filename` -- Explicit file

Downloads a specific file from the repo without querying the API.

```
hf://TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf
hf://sentence-transformers/all-MiniLM-L6-v2:model.onnx
```

Resolves directly to:
```
https://huggingface.co/{repo_id}/resolve/main/{filename}
```

### `https://...` -- Direct URL

Any HTTPS URL is passed through as-is. The filename is extracted from the last path segment.

```
https://example.com/models/my-model.onnx
```

## Cache Directory

By default, models are cached at:

```
~/.island/model-cache/
```

The directory is created automatically on first use. Each model is stored in a subdirectory named by the SHA256 hash of its download URL:

```
~/.island/model-cache/
  a3b4c5d6.../          # SHA256(download_url)
    model.Q4_K_M.gguf   # The actual model file
  e7f8a9b0.../
    model.onnx
```

## Configuration

Model cache settings are in the `[model_cache]` section of `config.toml`:

```toml
[model_cache]
# Maximum cache size in GB (default: 20)
max_cache_gb = 20

# Custom cache directory (default: ~/.island/model-cache)
# cache_dir = "/data/models"
```

The `ModelCacheConfig` struct:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_cache_gb` | `u64` | `20` | Maximum total cache size in gigabytes |
| `cache_dir` | `Option<String>` | `None` (uses `~/.island/model-cache`) | Override cache directory path |

## Download Flow

When `download_model(uri, expected_hash)` is called:

1. **Resolve URI** -- Convert `hf://` URIs to direct HTTPS download URLs.
2. **Check cache** -- Look up the URL's SHA256 hash in the in-memory entry map. If found and the file exists on disk, update the `last_used` timestamp and return immediately.
3. **Evict if needed** -- If total cache size would exceed `max_cache_gb`, remove least-recently-used entries until there is room.
4. **Stream download** -- Download the file to a `.tmp` file in the model's cache directory, computing SHA256 as bytes arrive. Progress is logged every 100 MB.
5. **Verify hash** -- If `expected_hash` is provided (and is not empty or the placeholder `sha256:placeholder`), compare the computed hash. On mismatch, delete the temp file and return an error.
6. **Finalize** -- Rename the `.tmp` file to its final name. Register the entry in the in-memory cache map.

### Hash Format

Hashes use the format `sha256:<hex-digest>`:

```
sha256:a3b4c5d6e7f8...
```

## LRU Eviction

When the total size of cached models exceeds `max_cache_bytes` (derived from `max_cache_gb`), the cache evicts models in least-recently-used order:

1. All entries are sorted by `last_used` timestamp (oldest first).
2. Entries are removed one at a time (entire subdirectory deleted) until total size is under the limit.
3. Eviction is logged: `Evicted cached model: {url}`.

The `last_used` timestamp is updated on every cache hit, both in memory and on the filesystem (via `filetime`).

## Initialization

At startup, the cache scans its directory for existing model files:

```rust
cache.init().await?;
```

This populates the in-memory entry map with any previously downloaded models, so they are available immediately without re-downloading.

## Tokenizer Auto-Download

The ONNX runtime has special handling for tokenizers. When executing a text task, it derives a tokenizer URI from the model URI:

- `hf://repo_id` or `hf://repo_id:model.onnx` becomes `hf://repo_id:tokenizer.json`
- Direct URLs use the parent directory: `https://example.com/models/model.onnx` becomes `https://example.com/models/tokenizer.json`

The tokenizer is downloaded through the same model cache, so it benefits from caching and hash verification.
