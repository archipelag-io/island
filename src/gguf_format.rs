//! GGUF binary format parser and layer-aware splitter.
//!
//! Parses GGUF v3 files to extract tensor metadata, then splits models
//! by transformer layer boundaries to produce valid sub-GGUF files that
//! can be loaded independently by llama.cpp.
//!
//! Tensor naming convention (llama/mistral/etc.):
//!   blk.{layer_id}.attn_q.weight
//!   blk.{layer_id}.attn_k.weight
//!   blk.{layer_id}.ffn_gate.weight
//!   token_embd.weight          (shared — first shard)
//!   output_norm.weight          (shared — last shard)
//!   output.weight               (shared — last shard)
//!
//! Reference: gguf.h from llama.cpp

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::info;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Parsed GGUF file header and tensor metadata (no tensor data loaded)
#[derive(Debug)]
pub struct GgufFile {
    #[allow(dead_code)]
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub kv_pairs: Vec<KvPair>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: usize,
}

#[derive(Debug, Clone)]
pub struct KvPair {
    pub key: String,
    pub value_type: u32,
    pub raw_bytes: Vec<u8>, // serialized value (for passthrough)
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<i64>,
    pub data_type: u32,
    pub data_offset: u64, // offset within the tensor data blob
    pub data_size: u64,   // computed from dims + dtype
    /// Which layer this tensor belongs to (None for shared tensors)
    pub layer_id: Option<u32>,
}

/// GGML data type sizes in bytes per element (partial — covers common quantizations)
fn ggml_type_size(dtype: u32) -> f64 {
    match dtype {
        0 => 4.0,   // F32
        1 => 2.0,   // F16
        2 => 0.5625, // Q4_0 (4.5 bits per weight average)
        3 => 0.625,  // Q4_1
        6 => 0.5625, // Q5_0
        7 => 0.625,  // Q5_1
        8 => 1.0,    // Q8_0
        9 => 1.0,    // Q8_1
        10 => 0.5625, // Q2_K
        11 => 0.625,  // Q3_K
        12 => 0.5625, // Q4_K
        13 => 0.6875, // Q5_K
        14 => 0.8125, // Q6_K
        28 => 2.0,    // BF16
        _ => 2.0,     // fallback: assume f16
    }
}

/// Compute tensor data size from dims and dtype
fn compute_tensor_size(dims: &[i64], dtype: u32) -> u64 {
    let n_elements: u64 = dims.iter().map(|&d| d as u64).product();
    (n_elements as f64 * ggml_type_size(dtype)).ceil() as u64
}

/// Extract layer ID from tensor name (e.g., "blk.5.attn_q.weight" -> Some(5))
fn extract_layer_id(name: &str) -> Option<u32> {
    if name.starts_with("blk.") {
        name.split('.').nth(1)?.parse::<u32>().ok()
    } else {
        None
    }
}

/// Determine if a tensor is shared (needed by all shards or specific ones)
fn tensor_placement(name: &str, layer_start: u32, layer_end: u32, total_layers: u32) -> bool {
    if let Some(layer_id) = extract_layer_id(name) {
        // Layer-specific tensor — include if in range
        layer_id >= layer_start && layer_id <= layer_end
    } else {
        // Shared tensor — include based on position:
        // - token_embd, rope_freqs -> first shard only
        // - output_norm, output -> last shard only
        // - Other metadata tensors -> all shards
        let is_first = layer_start == 0;
        let is_last = layer_end >= total_layers.saturating_sub(1);

        if name.starts_with("token_embd") || name.starts_with("rope_freqs") {
            is_first
        } else if name.starts_with("output_norm") || name == "output.weight" {
            is_last
        } else {
            // Unknown shared tensor — include in all shards
            true
        }
    }
}

/// Parse a GGUF file header and tensor metadata.
///
/// Does NOT load tensor data — only reads the header, KV pairs, and tensor info.
pub fn parse_gguf(path: &Path) -> Result<GgufFile> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open GGUF: {}", path.display()))?;

    // 1. Magic
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        anyhow::bail!("Not a valid GGUF file (bad magic)");
    }

    // 2. Version
    let version = read_u32(&mut file)?;
    if version != GGUF_VERSION {
        anyhow::bail!("Unsupported GGUF version: {} (expected {})", version, GGUF_VERSION);
    }

    // 3. Tensor count
    let tensor_count = read_i64(&mut file)? as u64;

    // 4. KV count
    let kv_count = read_i64(&mut file)? as u64;

    // 5. Read KV pairs
    let mut kv_pairs = Vec::with_capacity(kv_count as usize);
    let mut alignment = GGUF_DEFAULT_ALIGNMENT;

    for _ in 0..kv_count {
        let kv = read_kv_pair(&mut file)?;
        if kv.key == "general.alignment" && kv.value_type == 4 {
            // uint32
            if kv.raw_bytes.len() >= 4 {
                alignment = u32::from_le_bytes([
                    kv.raw_bytes[0], kv.raw_bytes[1],
                    kv.raw_bytes[2], kv.raw_bytes[3],
                ]) as usize;
            }
        }
        kv_pairs.push(kv);
    }

    // 6. Read tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let info = read_tensor_info(&mut file)?;
        tensors.push(info);
    }

    // Data offset is the current position, aligned
    let header_end = file.stream_position()?;
    let data_offset = align_offset(header_end, alignment);

    // Compute tensor data sizes and assign layer IDs
    for tensor in &mut tensors {
        tensor.data_size = compute_tensor_size(&tensor.dims, tensor.data_type);
        tensor.layer_id = extract_layer_id(&tensor.name);
    }

    Ok(GgufFile {
        version,
        tensor_count,
        kv_count,
        kv_pairs,
        tensors,
        data_offset,
        alignment,
    })
}

/// Split a GGUF model into N shards by layer boundaries.
///
/// Each shard is a valid GGUF file containing:
/// - All KV metadata from the original
/// - Tensors for its layer range
/// - Shared tensors (embeddings for first shard, output head for last)
pub fn split_by_layers(
    source_path: &Path,
    shard_count: usize,
    output_dir: &Path,
) -> Result<PathBuf> {
    if !(2..=16).contains(&shard_count) {
        anyhow::bail!("Shard count must be 2–16");
    }

    std::fs::create_dir_all(output_dir)?;

    let gguf = parse_gguf(source_path)?;

    // Determine total layer count from tensor names
    let max_layer = gguf.tensors.iter()
        .filter_map(|t| t.layer_id)
        .max()
        .unwrap_or(31);
    let total_layers = max_layer + 1;

    let layers_per_shard = total_layers / shard_count as u32;
    let remainder = total_layers % shard_count as u32;

    info!(
        "GGUF: {} tensors, {} layers, splitting into {} shards",
        gguf.tensor_count, total_layers, shard_count
    );

    let stem = source_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mut source_file = std::fs::File::open(source_path)?;
    let mut shard_infos = Vec::new();

    let mut layer_offset: u32 = 0;
    for i in 0..shard_count {
        let extra = if (i as u32) < remainder { 1 } else { 0 };
        let shard_layers = layers_per_shard + extra;
        let layer_start = layer_offset;
        let layer_end = layer_offset + shard_layers - 1;

        // Filter tensors for this shard
        let shard_tensors: Vec<&TensorInfo> = gguf.tensors.iter()
            .filter(|t| tensor_placement(&t.name, layer_start, layer_end, total_layers))
            .collect();

        let shard_name = format!("{}-shard-{}.gguf", stem, i);
        let shard_path = output_dir.join(&shard_name);

        let hash = write_shard_gguf(
            &shard_path,
            &gguf,
            &shard_tensors,
            &mut source_file,
            layer_start,
            shard_layers,
        )?;

        let file_size = std::fs::metadata(&shard_path)?.len();

        info!(
            "  Shard {}: layers {}-{} ({} tensors, {:.1} MB, {})",
            i, layer_start, layer_end, shard_tensors.len(),
            file_size as f64 / 1_048_576.0, &hash[..20]
        );

        shard_infos.push(ShardManifestEntry {
            index: i,
            filename: shard_name,
            hash,
            layer_start,
            layer_end,
            tensor_count: shard_tensors.len(),
        });

        layer_offset += shard_layers;
    }

    // Write manifest
    let manifest = build_manifest(stem, total_layers, shard_count, &shard_infos);
    let manifest_path = output_dir.join("shard_manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

    info!("Layer-aware manifest written to {}", manifest_path.display());
    Ok(manifest_path)
}

/// Write a single shard GGUF file with selected tensors.
///
/// Layer tensors are renumbered so that the shard is a valid standalone model:
/// `blk.16.attn_q.weight` → `blk.0.attn_q.weight` (if layer_start=16).
/// The `*.block_count` KV metadata is also updated to reflect the shard's layer count.
fn write_shard_gguf(
    path: &Path,
    source: &GgufFile,
    tensors: &[&TensorInfo],
    source_file: &mut std::fs::File,
    layer_start: u32,
    shard_layer_count: u32,
) -> Result<String> {
    let mut out = std::fs::File::create(path)?;
    let mut hasher = Sha256::new();

    // Header
    let magic_bytes = GGUF_MAGIC;
    out.write_all(magic_bytes)?;
    hasher.update(magic_bytes);

    write_u32_hashed(&mut out, &mut hasher, GGUF_VERSION)?;
    write_i64_hashed(&mut out, &mut hasher, tensors.len() as i64)?;

    // Count KV pairs (may add/modify block_count)
    let kv_pairs = rewrite_kv_pairs(&source.kv_pairs, shard_layer_count);
    write_i64_hashed(&mut out, &mut hasher, kv_pairs.len() as i64)?;

    // KV pairs (with block_count updated)
    for kv in &kv_pairs {
        write_string_hashed(&mut out, &mut hasher, &kv.key)?;
        write_u32_hashed(&mut out, &mut hasher, kv.value_type)?;
        out.write_all(&kv.raw_bytes)?;
        hasher.update(&kv.raw_bytes);
    }

    // Tensor info (with layer renumbering)
    let mut running_offset: u64 = 0;
    let mut tensor_data_offsets = Vec::new();

    for tensor in tensors {
        let renamed = renumber_tensor_name(&tensor.name, layer_start);
        write_string_hashed(&mut out, &mut hasher, &renamed)?;
        write_u32_hashed(&mut out, &mut hasher, tensor.n_dims)?;
        for &dim in &tensor.dims {
            write_i64_hashed(&mut out, &mut hasher, dim)?;
        }
        write_u32_hashed(&mut out, &mut hasher, tensor.data_type)?;

        // New offset in this shard
        let aligned = align_offset(running_offset, source.alignment);
        write_u64_hashed(&mut out, &mut hasher, aligned)?;
        tensor_data_offsets.push((tensor, aligned));
        running_offset = aligned + tensor.data_size;
    }

    // Align to data start
    let header_end = out.stream_position()?;
    let data_start = align_offset(header_end, source.alignment);
    let padding = data_start - header_end;
    let pad_bytes = vec![0u8; padding as usize];
    out.write_all(&pad_bytes)?;
    hasher.update(&pad_bytes);

    // Tensor data
    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    for (tensor, shard_offset) in &tensor_data_offsets {
        // Seek to this tensor's data in the source file
        let src_abs_offset = source.data_offset + tensor.data_offset;
        source_file.seek(SeekFrom::Start(src_abs_offset))?;

        // Pad to alignment if needed
        let current = out.stream_position()?;
        let target = data_start + shard_offset;
        if target > current {
            let pad = vec![0u8; (target - current) as usize];
            out.write_all(&pad)?;
            hasher.update(&pad);
        }

        // Copy tensor data
        let mut remaining = tensor.data_size;
        while remaining > 0 {
            let to_read = std::cmp::min(buf.len() as u64, remaining) as usize;
            let n = source_file.read(&mut buf[..to_read])?;
            if n == 0 { break; }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            remaining -= n as u64;
        }
    }

    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

/// Renumber layer tensor names so the shard is a valid standalone model.
/// `blk.16.attn_q.weight` with layer_start=16 becomes `blk.0.attn_q.weight`.
/// Non-layer tensors (token_embd, output, etc.) pass through unchanged.
fn renumber_tensor_name(name: &str, layer_start: u32) -> String {
    if let Some(layer_id) = extract_layer_id(name) {
        let new_id = layer_id.saturating_sub(layer_start);
        let prefix = format!("blk.{}.", layer_id);
        let new_prefix = format!("blk.{}.", new_id);
        name.replacen(&prefix, &new_prefix, 1)
    } else {
        name.to_string()
    }
}

/// Rewrite KV pairs to update block_count for the shard's layer count.
/// Keys ending in `.block_count` (e.g., `llama.block_count`) are rewritten.
fn rewrite_kv_pairs(source_kvs: &[KvPair], shard_layer_count: u32) -> Vec<KvPair> {
    source_kvs
        .iter()
        .map(|kv| {
            if kv.key.ends_with(".block_count") && kv.value_type == 4 {
                // uint32 — rewrite to shard layer count
                KvPair {
                    key: kv.key.clone(),
                    value_type: kv.value_type,
                    raw_bytes: shard_layer_count.to_le_bytes().to_vec(),
                }
            } else {
                kv.clone()
            }
        })
        .collect()
}

#[derive(Debug)]
struct ShardManifestEntry {
    index: usize,
    filename: String,
    hash: String,
    layer_start: u32,
    layer_end: u32,
    tensor_count: usize,
}

fn build_manifest(
    model_name: &str,
    total_layers: u32,
    shard_count: usize,
    shards: &[ShardManifestEntry],
) -> serde_json::Value {
    let shard_urls: serde_json::Map<String, serde_json::Value> = shards.iter()
        .map(|s| (s.index.to_string(), serde_json::json!(format!("https://YOUR_CDN/{}", s.filename))))
        .collect();

    let shard_hashes: serde_json::Map<String, serde_json::Value> = shards.iter()
        .map(|s| (s.index.to_string(), serde_json::json!(&s.hash)))
        .collect();

    let shard_layers: serde_json::Map<String, serde_json::Value> = shards.iter()
        .map(|s| (s.index.to_string(), serde_json::json!({
            "layer_start": s.layer_start,
            "layer_end": s.layer_end,
            "tensor_count": s.tensor_count,
        })))
        .collect();

    serde_json::json!({
        "total_layers": total_layers,
        "min_shards": 2,
        "max_shards": shard_count,
        "shard_urls": shard_urls,
        "shard_hashes": shard_hashes,
        "shard_layers": shard_layers,
        "model_name": model_name,
        "shard_count": shard_count,
        "layer_aware": true,
        "note": "Update shard_urls with actual CDN paths."
    })
}

// ============================================================================
// Binary reading helpers
// ============================================================================

fn read_u32(f: &mut std::fs::File) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i64(f: &mut std::fs::File) -> Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_u64(f: &mut std::fs::File) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(f: &mut std::fs::File) -> Result<String> {
    let len = read_u64(f)? as usize;
    if len > 1_000_000 {
        anyhow::bail!("GGUF string too long: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn read_kv_pair(f: &mut std::fs::File) -> Result<KvPair> {
    let key = read_string(f)?;
    let value_type = read_u32(f)?;
    let raw_bytes = read_kv_value(f, value_type)?;
    Ok(KvPair { key, value_type, raw_bytes })
}

fn read_kv_value(f: &mut std::fs::File, vtype: u32) -> Result<Vec<u8>> {
    match vtype {
        0 | 7 => { let mut b = [0u8; 1]; f.read_exact(&mut b)?; Ok(b.to_vec()) } // u8, bool
        1 => { let mut b = [0u8; 1]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // i8
        2 => { let mut b = [0u8; 2]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // u16
        3 => { let mut b = [0u8; 2]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // i16
        4 => { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // u32
        5 => { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // i32
        6 => { let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(b.to_vec()) }      // f32
        8 => { // string
            let s = read_string(f)?;
            let mut bytes = (s.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(s.as_bytes());
            Ok(bytes)
        }
        9 => { // array
            let arr_type = read_u32(f)?;
            let arr_len = read_u64(f)?;
            let mut bytes = arr_type.to_le_bytes().to_vec();
            bytes.extend_from_slice(&arr_len.to_le_bytes());
            for _ in 0..arr_len {
                let elem = read_kv_value(f, arr_type)?;
                bytes.extend_from_slice(&elem);
            }
            Ok(bytes)
        }
        10 => { let mut b = [0u8; 8]; f.read_exact(&mut b)?; Ok(b.to_vec()) } // u64
        11 => { let mut b = [0u8; 8]; f.read_exact(&mut b)?; Ok(b.to_vec()) } // i64
        12 => { let mut b = [0u8; 8]; f.read_exact(&mut b)?; Ok(b.to_vec()) } // f64
        _ => anyhow::bail!("Unknown GGUF KV type: {}", vtype),
    }
}

fn read_tensor_info(f: &mut std::fs::File) -> Result<TensorInfo> {
    let name = read_string(f)?;
    let n_dims = read_u32(f)?;
    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        dims.push(read_i64(f)?);
    }
    let data_type = read_u32(f)?;
    let data_offset = read_u64(f)?;

    Ok(TensorInfo {
        name,
        n_dims,
        dims,
        data_type,
        data_offset,
        data_size: 0, // computed after parsing
        layer_id: None, // assigned after parsing
    })
}

fn align_offset(offset: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    offset.div_ceil(a) * a
}

// ============================================================================
// Binary writing helpers (with hasher)
// ============================================================================

fn write_u32_hashed(f: &mut std::fs::File, h: &mut Sha256, v: u32) -> Result<()> {
    let bytes = v.to_le_bytes();
    f.write_all(&bytes)?;
    h.update(bytes);
    Ok(())
}

fn write_i64_hashed(f: &mut std::fs::File, h: &mut Sha256, v: i64) -> Result<()> {
    let bytes = v.to_le_bytes();
    f.write_all(&bytes)?;
    h.update(bytes);
    Ok(())
}

fn write_u64_hashed(f: &mut std::fs::File, h: &mut Sha256, v: u64) -> Result<()> {
    let bytes = v.to_le_bytes();
    f.write_all(&bytes)?;
    h.update(bytes);
    Ok(())
}

fn write_string_hashed(f: &mut std::fs::File, h: &mut Sha256, s: &str) -> Result<()> {
    let len_bytes = (s.len() as u64).to_le_bytes();
    f.write_all(&len_bytes)?;
    h.update(len_bytes);
    f.write_all(s.as_bytes())?;
    h.update(s.as_bytes());
    Ok(())
}

/// Identify MoE gating tensors in a parsed GGUF file.
///
/// In Mixtral-style MoE models, gating tensors have names like:
/// - `blk.{N}.ffn_gate_inp.weight` — the gating network for layer N
///
/// Returns a list of (layer_id, tensor_index) pairs for gating tensors.
pub fn find_gating_tensors(gguf: &GgufFile) -> Vec<(u32, usize)> {
    gguf.tensors.iter()
        .enumerate()
        .filter(|(_, t)| t.name.contains("ffn_gate_inp"))
        .filter_map(|(idx, t)| {
            t.layer_id.map(|lid| (lid, idx))
        })
        .collect()
}

/// Extract the raw tensor data for a specific tensor from the GGUF file.
///
/// Reads `tensor.data_size` bytes starting at `data_offset + tensor.data_offset`.
pub fn extract_tensor_data(
    source_path: &Path,
    gguf: &GgufFile,
    tensor_idx: usize,
) -> Result<Vec<u8>> {
    if tensor_idx >= gguf.tensors.len() {
        anyhow::bail!("Tensor index {} out of range ({})", tensor_idx, gguf.tensors.len());
    }

    let tensor = &gguf.tensors[tensor_idx];
    let abs_offset = gguf.data_offset + tensor.data_offset;

    let mut file = std::fs::File::open(source_path)
        .with_context(|| format!("Failed to open GGUF: {}", source_path.display()))?;

    file.seek(SeekFrom::Start(abs_offset))?;

    let mut data = vec![0u8; tensor.data_size as usize];
    file.read_exact(&mut data)?;

    Ok(data)
}

/// Extract gating weights from an MoE GGUF model.
///
/// Reads `blk.N.ffn_gate_inp.weight` tensors and returns per-layer gating data.
/// Returns None if the model doesn't contain gating tensors.
pub fn extract_gating_weights(source_path: &Path) -> Result<Option<GatingWeightData>> {
    let all = extract_all_gating_weights(source_path)?;
    Ok(all.into_iter().next())
}

/// Extract ALL per-layer gating weights from an MoE GGUF model.
///
/// Returns a `GatingWeightData` for each layer that has a gating tensor,
/// enabling per-layer expert selection with different learned weights.
pub fn extract_all_gating_weights(source_path: &Path) -> Result<Vec<GatingWeightData>> {
    let gguf = parse_gguf(source_path)?;

    let gating_tensors = find_gating_tensors(&gguf);
    if gating_tensors.is_empty() {
        return Ok(Vec::new());
    }

    let n_embd = gguf.kv_pairs.iter()
        .find(|kv| kv.key.ends_with(".embedding_length"))
        .and_then(|kv| {
            if kv.value_type == 4 && kv.raw_bytes.len() >= 4 {
                Some(u32::from_le_bytes([kv.raw_bytes[0], kv.raw_bytes[1], kv.raw_bytes[2], kv.raw_bytes[3]]))
            } else {
                None
            }
        })
        .unwrap_or(4096);

    let mut results = Vec::with_capacity(gating_tensors.len());

    for (layer_id, tensor_idx) in &gating_tensors {
        let tensor = &gguf.tensors[*tensor_idx];

        tracing::info!(
            "Found MoE gating tensor: {} (layer {}, dims {:?}, dtype {})",
            tensor.name, layer_id, tensor.dims, tensor.data_type
        );

        let raw_data = extract_tensor_data(source_path, &gguf, *tensor_idx)?;

        let (n_experts, hidden_dim) = if tensor.dims.len() == 2 {
            (tensor.dims[0] as u32, tensor.dims[1] as u32)
        } else if tensor.dims.len() == 1 {
            let n = tensor.dims[0] as u32;
            (n / n_embd, n_embd)
        } else {
            continue;
        };

        results.push(GatingWeightData {
            raw_data,
            n_experts,
            hidden_dim,
            data_type: tensor.data_type,
            layer_id: *layer_id,
        });
    }

    Ok(results)
}

/// Raw gating weight data extracted from a GGUF file
#[derive(Debug)]
pub struct GatingWeightData {
    /// Raw tensor bytes
    pub raw_data: Vec<u8>,
    /// Number of experts
    pub n_experts: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Data type (0=f32, 1=f16, etc.)
    pub data_type: u32,
    /// Which transformer layer this gating is from
    pub layer_id: u32,
}

impl GatingWeightData {
    /// Convert to f32 weight matrix (n_experts rows × hidden_dim columns).
    /// Handles f32 and f16 input formats.
    pub fn to_f32_matrix(&self) -> Vec<Vec<f32>> {
        let n_exp = self.n_experts as usize;
        let h_dim = self.hidden_dim as usize;
        let mut matrix = vec![vec![0.0f32; h_dim]; n_exp];

        match self.data_type {
            0 => {
                // F32: 4 bytes per element
                for (e, row) in matrix.iter_mut().enumerate() {
                    for (h, cell) in row.iter_mut().enumerate() {
                        let offset = (e * h_dim + h) * 4;
                        if offset + 4 <= self.raw_data.len() {
                            *cell = f32::from_le_bytes([
                                self.raw_data[offset],
                                self.raw_data[offset + 1],
                                self.raw_data[offset + 2],
                                self.raw_data[offset + 3],
                            ]);
                        }
                    }
                }
            }
            1 => {
                // F16: 2 bytes per element (convert to f32)
                for (e, row) in matrix.iter_mut().enumerate() {
                    for (h, cell) in row.iter_mut().enumerate() {
                        let offset = (e * h_dim + h) * 2;
                        if offset + 2 <= self.raw_data.len() {
                            let half = u16::from_le_bytes([self.raw_data[offset], self.raw_data[offset + 1]]);
                            *cell = half_to_f32(half);
                        }
                    }
                }
            }
            _ => {
                // Quantized formats: fall back to zero matrix (gating weights are
                // typically stored in f32 or f16, not quantized)
                tracing::warn!("Gating tensor dtype {} not supported for direct extraction", self.data_type);
            }
        }

        matrix
    }
}

/// Convert IEEE 754 half-precision (f16) to f32
fn half_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exp = ((half >> 10) & 0x1F) as u32;
    let mant = (half & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            f32::from_bits((sign << 31) | ((127 - 15 - e) << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf/NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_id() {
        assert_eq!(extract_layer_id("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_id("blk.31.ffn_gate.weight"), Some(31));
        assert_eq!(extract_layer_id("token_embd.weight"), None);
        assert_eq!(extract_layer_id("output.weight"), None);
        assert_eq!(extract_layer_id("output_norm.weight"), None);
    }

    #[test]
    fn test_tensor_placement() {
        // Layer tensors
        assert!(tensor_placement("blk.0.attn_q.weight", 0, 15, 32));
        assert!(tensor_placement("blk.15.ffn_down.weight", 0, 15, 32));
        assert!(!tensor_placement("blk.16.attn_q.weight", 0, 15, 32));
        assert!(tensor_placement("blk.16.attn_q.weight", 16, 31, 32));

        // Shared tensors — first shard
        assert!(tensor_placement("token_embd.weight", 0, 15, 32));
        assert!(!tensor_placement("token_embd.weight", 16, 31, 32));

        // Shared tensors — last shard
        assert!(tensor_placement("output.weight", 16, 31, 32));
        assert!(!tensor_placement("output.weight", 0, 15, 32));
        assert!(tensor_placement("output_norm.weight", 16, 31, 32));
    }

    #[test]
    fn test_compute_tensor_size() {
        // F32: 4 bytes per element
        assert_eq!(compute_tensor_size(&[4096, 4096], 0), 4096 * 4096 * 4);
        // F16: 2 bytes per element
        assert_eq!(compute_tensor_size(&[4096], 1), 4096 * 2);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn test_ggml_type_size() {
        assert_eq!(ggml_type_size(0), 4.0); // F32
        assert_eq!(ggml_type_size(1), 2.0); // F16
    }

    #[test]
    fn test_renumber_tensor_name() {
        // Layer tensor renumbering
        assert_eq!(renumber_tensor_name("blk.16.attn_q.weight", 16), "blk.0.attn_q.weight");
        assert_eq!(renumber_tensor_name("blk.17.ffn_gate.weight", 16), "blk.1.ffn_gate.weight");
        assert_eq!(renumber_tensor_name("blk.31.attn_output.weight", 16), "blk.15.attn_output.weight");
        // First shard — no change
        assert_eq!(renumber_tensor_name("blk.0.attn_q.weight", 0), "blk.0.attn_q.weight");
        // Non-layer tensors pass through
        assert_eq!(renumber_tensor_name("token_embd.weight", 16), "token_embd.weight");
        assert_eq!(renumber_tensor_name("output.weight", 16), "output.weight");
    }

    #[test]
    fn test_find_gating_tensors() {
        let gguf = GgufFile {
            version: 3,
            tensor_count: 3,
            kv_count: 0,
            kv_pairs: vec![],
            tensors: vec![
                TensorInfo { name: "blk.0.attn_q.weight".into(), n_dims: 2, dims: vec![4096, 4096], data_type: 0, data_offset: 0, data_size: 0, layer_id: Some(0) },
                TensorInfo { name: "blk.0.ffn_gate_inp.weight".into(), n_dims: 2, dims: vec![8, 4096], data_type: 0, data_offset: 0, data_size: 0, layer_id: Some(0) },
                TensorInfo { name: "blk.1.ffn_gate_inp.weight".into(), n_dims: 2, dims: vec![8, 4096], data_type: 0, data_offset: 0, data_size: 0, layer_id: Some(1) },
            ],
            data_offset: 0,
            alignment: 32,
        };

        let gating = find_gating_tensors(&gguf);
        assert_eq!(gating.len(), 2);
        assert_eq!(gating[0], (0, 1)); // layer 0, tensor index 1
        assert_eq!(gating[1], (1, 2)); // layer 1, tensor index 2
    }

    #[test]
    fn test_half_to_f32() {
        // 1.0 in f16 = 0x3C00
        let one = half_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-3);

        // 0.0 in f16 = 0x0000
        let zero = half_to_f32(0x0000);
        assert_eq!(zero, 0.0);

        // -1.0 in f16 = 0xBC00
        let neg_one = half_to_f32(0xBC00);
        assert!((neg_one - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_gating_weight_data_to_f32() {
        // 2 experts × 2 hidden_dim, f32
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let gwd = GatingWeightData {
            raw_data: data,
            n_experts: 2,
            hidden_dim: 2,
            data_type: 0, // f32
            layer_id: 0,
        };

        let matrix = gwd.to_f32_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![1.0, 2.0]);
        assert_eq!(matrix[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_rewrite_kv_pairs() {
        let kvs = vec![
            KvPair { key: "general.name".into(), value_type: 8, raw_bytes: vec![3, 0, 0, 0, 0, 0, 0, 0, b'f', b'o', b'o'] },
            KvPair { key: "llama.block_count".into(), value_type: 4, raw_bytes: 32u32.to_le_bytes().to_vec() },
            KvPair { key: "llama.attention.head_count".into(), value_type: 4, raw_bytes: 32u32.to_le_bytes().to_vec() },
        ];

        let rewritten = rewrite_kv_pairs(&kvs, 16);
        assert_eq!(rewritten.len(), 3);
        // block_count should be rewritten to 16
        assert_eq!(
            u32::from_le_bytes([rewritten[1].raw_bytes[0], rewritten[1].raw_bytes[1], rewritten[1].raw_bytes[2], rewritten[1].raw_bytes[3]]),
            16
        );
        // Other KVs unchanged
        assert_eq!(rewritten[0].raw_bytes, kvs[0].raw_bytes);
        assert_eq!(rewritten[2].raw_bytes, kvs[2].raw_bytes);
    }

    // ========================================================================
    // Additional tests
    // ========================================================================

    #[test]
    fn test_ggml_type_size_all_known_types() {
        // Verify all explicitly handled quantization types
        assert_eq!(ggml_type_size(2), 0.5625);  // Q4_0
        assert_eq!(ggml_type_size(3), 0.625);   // Q4_1
        assert_eq!(ggml_type_size(6), 0.5625);  // Q5_0
        assert_eq!(ggml_type_size(7), 0.625);   // Q5_1
        assert_eq!(ggml_type_size(8), 1.0);     // Q8_0
        assert_eq!(ggml_type_size(9), 1.0);     // Q8_1
        assert_eq!(ggml_type_size(10), 0.5625); // Q2_K
        assert_eq!(ggml_type_size(11), 0.625);  // Q3_K
        assert_eq!(ggml_type_size(12), 0.5625); // Q4_K
        assert_eq!(ggml_type_size(13), 0.6875); // Q5_K
        assert_eq!(ggml_type_size(14), 0.8125); // Q6_K
        assert_eq!(ggml_type_size(28), 2.0);    // BF16
    }

    #[test]
    fn test_ggml_type_size_unknown_falls_back_to_f16() {
        // Unknown types should fall back to f16 size (2.0)
        assert_eq!(ggml_type_size(99), 2.0);
        assert_eq!(ggml_type_size(255), 2.0);
        assert_eq!(ggml_type_size(15), 2.0);
    }

    #[test]
    fn test_compute_tensor_size_quantized() {
        // Q4_0 (type 2): 0.5625 bytes per element
        // 4096 elements * 0.5625 = 2304
        assert_eq!(compute_tensor_size(&[4096], 2), 2304);
    }

    #[test]
    fn test_compute_tensor_size_multidimensional() {
        // 3D tensor: 2 x 3 x 4 = 24 elements, F32 = 96 bytes
        assert_eq!(compute_tensor_size(&[2, 3, 4], 0), 96);
    }

    #[test]
    fn test_compute_tensor_size_single_element() {
        // 1 element, F32 = 4 bytes
        assert_eq!(compute_tensor_size(&[1], 0), 4);
        // 1 element, F16 = 2 bytes
        assert_eq!(compute_tensor_size(&[1], 1), 2);
    }

    #[test]
    fn test_extract_layer_id_high_layer_numbers() {
        assert_eq!(extract_layer_id("blk.127.attn_q.weight"), Some(127));
        assert_eq!(extract_layer_id("blk.999.ffn_down.weight"), Some(999));
    }

    #[test]
    fn test_extract_layer_id_malformed_names() {
        // Prefix matches but no valid number
        assert_eq!(extract_layer_id("blk.abc.attn_q.weight"), None);
        // Only "blk." with no following segment
        assert_eq!(extract_layer_id("blk."), None);
        // Empty string
        assert_eq!(extract_layer_id(""), None);
        // Similar prefix but not exact
        assert_eq!(extract_layer_id("block.5.attn_q.weight"), None);
    }

    #[test]
    fn test_tensor_placement_single_shard_gets_all_shared() {
        // When a single shard covers all layers (0..31, total 32),
        // it is both first and last, so all shared tensors are included
        assert!(tensor_placement("token_embd.weight", 0, 31, 32));
        assert!(tensor_placement("output.weight", 0, 31, 32));
        assert!(tensor_placement("output_norm.weight", 0, 31, 32));
        assert!(tensor_placement("rope_freqs.weight", 0, 31, 32));
    }

    #[test]
    fn test_tensor_placement_unknown_shared_tensor_all_shards() {
        // Unknown shared tensors should be included in every shard
        assert!(tensor_placement("some_custom_tensor.weight", 0, 15, 32));
        assert!(tensor_placement("some_custom_tensor.weight", 16, 31, 32));
    }

    #[test]
    fn test_tensor_placement_rope_freqs_first_shard_only() {
        assert!(tensor_placement("rope_freqs.weight", 0, 15, 32));
        assert!(!tensor_placement("rope_freqs.weight", 16, 31, 32));
    }

    #[test]
    fn test_tensor_placement_boundary_layer() {
        // Layer exactly at boundary is included
        assert!(tensor_placement("blk.15.attn_q.weight", 0, 15, 32));
        // Layer one past boundary is excluded
        assert!(!tensor_placement("blk.16.attn_q.weight", 0, 15, 32));
        // Layer one before start is excluded
        assert!(!tensor_placement("blk.15.attn_q.weight", 16, 31, 32));
    }

    #[test]
    fn test_renumber_tensor_name_saturating_sub() {
        // layer_start > layer_id should saturate to 0 (defensive)
        assert_eq!(renumber_tensor_name("blk.5.attn_q.weight", 10), "blk.0.attn_q.weight");
    }

    #[test]
    fn test_align_offset_alignment_one() {
        // Alignment of 1 should return the offset unchanged
        assert_eq!(align_offset(0, 1), 0);
        assert_eq!(align_offset(7, 1), 7);
        assert_eq!(align_offset(100, 1), 100);
    }

    #[test]
    fn test_align_offset_large_alignment() {
        assert_eq!(align_offset(0, 4096), 0);
        assert_eq!(align_offset(1, 4096), 4096);
        assert_eq!(align_offset(4095, 4096), 4096);
        assert_eq!(align_offset(4096, 4096), 4096);
        assert_eq!(align_offset(4097, 4096), 8192);
    }

    #[test]
    fn test_half_to_f32_positive_infinity() {
        // Positive infinity in f16 = 0x7C00
        let inf = half_to_f32(0x7C00);
        assert!(inf.is_infinite());
        assert!(inf > 0.0);
    }

    #[test]
    fn test_half_to_f32_negative_infinity() {
        // Negative infinity in f16 = 0xFC00
        let neg_inf = half_to_f32(0xFC00);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf < 0.0);
    }

    #[test]
    fn test_half_to_f32_nan() {
        // NaN in f16 = 0x7C01 (exponent all 1s, mantissa non-zero)
        let nan = half_to_f32(0x7C01);
        assert!(nan.is_nan());
    }

    #[test]
    fn test_half_to_f32_negative_zero() {
        // Negative zero in f16 = 0x8000
        let neg_zero = half_to_f32(0x8000);
        assert_eq!(neg_zero, 0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[test]
    fn test_half_to_f32_small_values() {
        // 0.5 in f16 = 0x3800
        let half_val = half_to_f32(0x3800);
        assert!((half_val - 0.5).abs() < 1e-3);

        // 2.0 in f16 = 0x4000
        let two = half_to_f32(0x4000);
        assert!((two - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_half_to_f32_denormalized() {
        // Smallest positive denormalized f16 = 0x0001
        let tiny = half_to_f32(0x0001);
        assert!(tiny > 0.0);
        assert!(tiny < 1e-4); // very small value
    }

    #[test]
    fn test_gating_weight_data_to_f32_f16_input() {
        // 1 expert x 2 hidden_dim, f16 format
        // 1.0 in f16 = 0x3C00, 2.0 in f16 = 0x4000
        let data: Vec<u8> = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];

        let gwd = GatingWeightData {
            raw_data: data,
            n_experts: 1,
            hidden_dim: 2,
            data_type: 1, // f16
            layer_id: 0,
        };

        let matrix = gwd.to_f32_matrix();
        assert_eq!(matrix.len(), 1);
        assert!((matrix[0][0] - 1.0).abs() < 1e-3);
        assert!((matrix[0][1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_gating_weight_data_to_f32_unsupported_dtype() {
        // Quantized type should produce zero matrix
        let gwd = GatingWeightData {
            raw_data: vec![0u8; 16],
            n_experts: 2,
            hidden_dim: 2,
            data_type: 8, // Q8_0
            layer_id: 0,
        };

        let matrix = gwd.to_f32_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![0.0, 0.0]);
        assert_eq!(matrix[1], vec![0.0, 0.0]);
    }

    #[test]
    fn test_find_gating_tensors_empty() {
        let gguf = GgufFile {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            kv_pairs: vec![],
            tensors: vec![],
            data_offset: 0,
            alignment: 32,
        };

        let gating = find_gating_tensors(&gguf);
        assert!(gating.is_empty());
    }

    #[test]
    fn test_find_gating_tensors_no_gating() {
        let gguf = GgufFile {
            version: 3,
            tensor_count: 2,
            kv_count: 0,
            kv_pairs: vec![],
            tensors: vec![
                TensorInfo { name: "blk.0.attn_q.weight".into(), n_dims: 2, dims: vec![4096, 4096], data_type: 0, data_offset: 0, data_size: 0, layer_id: Some(0) },
                TensorInfo { name: "blk.0.ffn_gate.weight".into(), n_dims: 2, dims: vec![4096, 4096], data_type: 0, data_offset: 0, data_size: 0, layer_id: Some(0) },
            ],
            data_offset: 0,
            alignment: 32,
        };

        // ffn_gate (not ffn_gate_inp) should NOT be detected as a gating tensor
        let gating = find_gating_tensors(&gguf);
        assert!(gating.is_empty());
    }

    #[test]
    fn test_rewrite_kv_pairs_no_block_count() {
        let kvs = vec![
            KvPair { key: "general.name".into(), value_type: 8, raw_bytes: vec![1, 0, 0, 0, 0, 0, 0, 0, b'x'] },
        ];

        let rewritten = rewrite_kv_pairs(&kvs, 8);
        assert_eq!(rewritten.len(), 1);
        assert_eq!(rewritten[0].raw_bytes, kvs[0].raw_bytes);
    }

    #[test]
    fn test_rewrite_kv_pairs_wrong_type_block_count() {
        // block_count with wrong type (string instead of uint32) should NOT be rewritten
        let kvs = vec![
            KvPair { key: "llama.block_count".into(), value_type: 8, raw_bytes: vec![2, 0, 0, 0, 0, 0, 0, 0, b'3', b'2'] },
        ];

        let rewritten = rewrite_kv_pairs(&kvs, 8);
        // Should be unchanged since type is 8 (string), not 4 (uint32)
        assert_eq!(rewritten[0].raw_bytes, kvs[0].raw_bytes);
    }

    #[test]
    fn test_build_manifest_structure() {
        let shards = vec![
            ShardManifestEntry {
                index: 0,
                filename: "model-shard-0.gguf".into(),
                hash: "sha256:abc123".into(),
                layer_start: 0,
                layer_end: 15,
                tensor_count: 100,
            },
            ShardManifestEntry {
                index: 1,
                filename: "model-shard-1.gguf".into(),
                hash: "sha256:def456".into(),
                layer_start: 16,
                layer_end: 31,
                tensor_count: 105,
            },
        ];

        let manifest = build_manifest("test-model", 32, 2, &shards);

        assert_eq!(manifest["total_layers"], 32);
        assert_eq!(manifest["shard_count"], 2);
        assert_eq!(manifest["min_shards"], 2);
        assert_eq!(manifest["max_shards"], 2);
        assert_eq!(manifest["model_name"], "test-model");
        assert_eq!(manifest["layer_aware"], true);

        // Check shard hashes
        assert_eq!(manifest["shard_hashes"]["0"], "sha256:abc123");
        assert_eq!(manifest["shard_hashes"]["1"], "sha256:def456");

        // Check shard layers
        assert_eq!(manifest["shard_layers"]["0"]["layer_start"], 0);
        assert_eq!(manifest["shard_layers"]["0"]["layer_end"], 15);
        assert_eq!(manifest["shard_layers"]["0"]["tensor_count"], 100);
        assert_eq!(manifest["shard_layers"]["1"]["layer_start"], 16);
        assert_eq!(manifest["shard_layers"]["1"]["layer_end"], 31);
    }
}
