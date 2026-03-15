//! Model shard splitting tool for pipeline parallelism.
//!
//! Takes a GGUF model file and produces N shard files + a JSON manifest
//! suitable for the `shard_manifest` field on a Cargo workload.
//!
//! Usage:
//!   island split-model --input model.gguf --shards 4 --output-dir ./shards
//!
//! This produces:
//!   ./shards/model-shard-0.gguf
//!   ./shards/model-shard-1.gguf
//!   ./shards/model-shard-2.gguf
//!   ./shards/model-shard-3.gguf
//!   ./shards/shard_manifest.json
//!
//! Note: This performs a naive byte-level split, NOT a layer-aware split.
//! For production use, a layer-aware GGUF splitter that understands the
//! tensor layout is needed. This tool is sufficient for testing the
//! pipeline infrastructure end-to-end.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tracing::info;

/// Split a GGUF model file into N equal-sized shards.
///
/// Returns the path to the generated shard_manifest.json.
pub fn split_model(input: &Path, shard_count: usize, output_dir: &Path) -> Result<PathBuf> {
    if shard_count < 2 {
        anyhow::bail!("Shard count must be at least 2");
    }
    if shard_count > 16 {
        anyhow::bail!("Shard count must be at most 16");
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    let file_size = std::fs::metadata(input)
        .with_context(|| format!("Failed to read model file: {}", input.display()))?
        .len();

    let shard_size = file_size / shard_count as u64;
    let remainder = file_size % shard_count as u64;

    info!(
        "Splitting {} ({:.1} GB) into {} shards of ~{:.1} MB each",
        input.display(),
        file_size as f64 / 1_073_741_824.0,
        shard_count,
        shard_size as f64 / 1_048_576.0,
    );

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    let mut source = std::fs::File::open(input)
        .with_context(|| format!("Failed to open model file: {}", input.display()))?;

    let mut shard_infos = Vec::with_capacity(shard_count);
    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB read buffer

    for i in 0..shard_count {
        let shard_name = format!("{}-shard-{}.gguf", stem, i);
        let shard_path = output_dir.join(&shard_name);

        // Last shard gets the remainder bytes
        let this_shard_size = if i == shard_count - 1 {
            shard_size + remainder
        } else {
            shard_size
        };

        let mut shard_file = std::fs::File::create(&shard_path)
            .with_context(|| format!("Failed to create shard file: {}", shard_path.display()))?;

        let mut hasher = Sha256::new();
        let mut bytes_written: u64 = 0;

        while bytes_written < this_shard_size {
            let to_read = std::cmp::min(
                buf.len() as u64,
                this_shard_size - bytes_written,
            ) as usize;

            let n = source.read(&mut buf[..to_read])
                .context("Failed to read from source model")?;

            if n == 0 {
                break;
            }

            shard_file.write_all(&buf[..n])
                .context("Failed to write shard data")?;
            hasher.update(&buf[..n]);
            bytes_written += n as u64;
        }

        let hash = format!("sha256:{}", hex::encode(hasher.finalize()));

        info!(
            "  Shard {}: {} ({:.1} MB, {})",
            i,
            shard_name,
            bytes_written as f64 / 1_048_576.0,
            &hash[..20],
        );

        shard_infos.push(ShardInfo {
            index: i,
            filename: shard_name,
            size_bytes: bytes_written,
            hash,
        });
    }

    // Generate manifest
    let manifest = generate_manifest(stem, shard_count, &shard_infos);
    let manifest_path = output_dir.join("shard_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, &manifest_json)?;

    info!("Manifest written to {}", manifest_path.display());
    info!("\nTo use this manifest, set it as the Cargo's shard_manifest field.");
    info!("Update shard_urls with the actual download URLs for each shard file.");

    Ok(manifest_path)
}

#[derive(Debug)]
#[allow(dead_code)]
struct ShardInfo {
    index: usize,
    filename: String,
    size_bytes: u64,
    hash: String,
}

fn generate_manifest(
    model_name: &str,
    shard_count: usize,
    shards: &[ShardInfo],
) -> serde_json::Value {
    // Estimate layers from shard count (common transformer layer counts)
    let estimated_layers = estimate_layer_count(model_name);

    let shard_urls: serde_json::Map<String, serde_json::Value> = shards
        .iter()
        .map(|s| {
            (
                s.index.to_string(),
                serde_json::json!(format!("https://YOUR_CDN/{}", s.filename)),
            )
        })
        .collect();

    let shard_hashes: serde_json::Map<String, serde_json::Value> = shards
        .iter()
        .map(|s| (s.index.to_string(), serde_json::json!(s.hash)))
        .collect();

    serde_json::json!({
        "total_layers": estimated_layers,
        "min_shards": 2,
        "max_shards": shard_count,
        "shard_urls": shard_urls,
        "shard_hashes": shard_hashes,
        "model_name": model_name,
        "shard_count": shard_count,
        "note": "Update shard_urls with actual CDN paths. total_layers is estimated — set to the real value for your model."
    })
}

/// Estimate layer count from model name heuristics
fn estimate_layer_count(model_name: &str) -> u32 {
    let name_lower = model_name.to_lowercase();
    if name_lower.contains("70b") {
        80
    } else if name_lower.contains("34b") || name_lower.contains("33b") {
        60
    } else if name_lower.contains("13b") {
        40
    } else if name_lower.contains("7b") || name_lower.contains("8b") {
        32
    } else if name_lower.contains("3b") {
        26
    } else if name_lower.contains("1b") || name_lower.contains("tiny") {
        22
    } else {
        32 // Default guess
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_layer_count() {
        assert_eq!(estimate_layer_count("llama-70b-chat"), 80);
        assert_eq!(estimate_layer_count("mistral-7b-instruct"), 32);
        assert_eq!(estimate_layer_count("codellama-13b"), 40);
        assert_eq!(estimate_layer_count("tinyllama-1.1b"), 22);
        assert_eq!(estimate_layer_count("unknown-model"), 32);
    }

    #[test]
    fn test_generate_manifest() {
        let shards = vec![
            ShardInfo {
                index: 0,
                filename: "model-shard-0.gguf".into(),
                size_bytes: 1_000_000,
                hash: "sha256:abc123".into(),
            },
            ShardInfo {
                index: 1,
                filename: "model-shard-1.gguf".into(),
                size_bytes: 1_000_000,
                hash: "sha256:def456".into(),
            },
        ];

        let manifest = generate_manifest("llama-7b", 2, &shards);
        assert_eq!(manifest["total_layers"], 32);
        assert_eq!(manifest["min_shards"], 2);
        assert_eq!(manifest["max_shards"], 2);
        assert!(manifest["shard_urls"]["0"].is_string());
        assert!(manifest["shard_urls"]["1"].is_string());
        assert_eq!(manifest["shard_hashes"]["0"], "sha256:abc123");
    }

    #[test]
    fn test_split_model_creates_shards() {
        let dir = tempfile::TempDir::new().unwrap();
        let input_path = dir.path().join("test-model.gguf");

        // Create a small test file (1KB)
        let data = vec![42u8; 1024];
        std::fs::write(&input_path, &data).unwrap();

        let output_dir = dir.path().join("shards");
        let manifest_path = split_model(&input_path, 2, &output_dir).unwrap();

        // Check shards exist
        assert!(output_dir.join("test-model-shard-0.gguf").exists());
        assert!(output_dir.join("test-model-shard-1.gguf").exists());
        assert!(manifest_path.exists());

        // Check sizes (512 bytes each for even split)
        let s0 = std::fs::metadata(output_dir.join("test-model-shard-0.gguf"))
            .unwrap()
            .len();
        let s1 = std::fs::metadata(output_dir.join("test-model-shard-1.gguf"))
            .unwrap()
            .len();
        assert_eq!(s0 + s1, 1024);

        // Check manifest
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["shard_count"], 2);
    }
}
