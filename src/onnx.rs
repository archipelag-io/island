//! Native ONNX runtime for executing ONNX model workloads
//!
//! Supports task types: text-classification, feature-extraction,
//! object-detection, image-segmentation, question-answering.
//!
//! Text tasks require a tokenizer.json alongside the model. The module
//! auto-downloads it from the same HuggingFace repo.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{watch, RwLock};
use tracing::{info, warn};

use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Execute an ONNX workload
pub async fn execute_onnx_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    job: &AssignJob,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let onnx_uri = job
        .onnx_model_url
        .as_ref()
        .context("ONNX workload missing onnx_model_url")?;

    let task_type = job
        .onnx_task_type
        .as_deref()
        .unwrap_or("text-classification");

    info!(
        "Executing ONNX workload: {} (task: {})",
        onnx_uri, task_type
    );

    // Get model cache
    let model_cache = {
        let st = state.read().await;
        st.model_cache()
            .context("Model cache not initialized")?
            .clone()
    };

    // Download ONNX model
    let model_path = model_cache
        .download_model(onnx_uri, job.onnx_model_hash.as_deref())
        .await
        .with_context(|| format!("Failed to download ONNX model: {}", onnx_uri))?;

    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // For text tasks, download tokenizer from same HF repo
    let tokenizer = if is_text_task(task_type) {
        let tokenizer_uri = derive_tokenizer_uri(onnx_uri);
        match model_cache.download_model(&tokenizer_uri, None).await {
            Ok(tok_path) => match Tokenizer::from_file(&tok_path) {
                Ok(t) => Some(t),
                Err(e) => {
                    warn!("Failed to load tokenizer from {:?}: {}", tok_path, e);
                    None
                }
            },
            Err(e) => {
                warn!("Failed to download tokenizer for {}: {}", onnx_uri, e);
                None
            }
        }
    } else {
        None
    };

    // Create ONNX session
    let model_path_clone = model_path.clone();
    let session = tokio::task::spawn_blocking(move || -> Result<Session> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create ONNX session builder: {}", e))?
            .with_intra_threads(num_cpus::get().min(4))
            .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
            .commit_from_file(&model_path_clone)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))?;
        Ok(session)
    })
    .await??;

    // Route to task-specific handler
    let mut session = session;
    let result = match task_type {
        "text-classification"
        | "fill-mask"
        | "token-classification"
        | "zero-shot-classification" => {
            run_text_classification(&mut session, &job.input, tokenizer.as_ref())
        }
        "feature-extraction" => {
            run_feature_extraction(&mut session, &job.input, tokenizer.as_ref())
        }
        "object-detection" => run_object_detection(&mut session, &job.input),
        "image-segmentation" => run_image_segmentation(&mut session, &job.input),
        "question-answering" => {
            run_question_answering(&mut session, &job.input, tokenizer.as_ref())
        }
        other => {
            warn!(
                "ONNX task '{}' — attempting text-classification fallback",
                other
            );
            run_text_classification(&mut session, &job.input, tokenizer.as_ref())
        }
    };

    match result {
        Ok(output) => {
            let output_json = serde_json::to_string(&output)?;
            nats.publish_output(job_id, 0, &output_json, true).await?;
            nats.publish_status(job_id, "succeeded", None).await?;
        }
        Err(e) => {
            nats.publish_status(job_id, "failed", Some(format!("ONNX error: {}", e)))
                .await?;
        }
    }

    Ok(())
}

fn is_text_task(task: &str) -> bool {
    matches!(
        task,
        "text-classification"
            | "fill-mask"
            | "token-classification"
            | "feature-extraction"
            | "question-answering"
            | "zero-shot-classification"
            | "text-generation"
            | "text2text-generation"
            | "summarization"
            | "translation"
    )
}

fn derive_tokenizer_uri(model_uri: &str) -> String {
    if let Some(hf_ref) = model_uri.strip_prefix("hf://") {
        let repo_id = hf_ref.split(':').next().unwrap_or(hf_ref);
        format!("hf://{}:tokenizer.json", repo_id)
    } else if let Some(base) = model_uri.rsplit_once('/') {
        format!("{}/tokenizer.json", base.0)
    } else {
        model_uri.to_string()
    }
}

/// Tokenize text → (input_ids, attention_mask) as ort Tensor values
fn tokenize_to_tensors(
    tokenizer: Option<&Tokenizer>,
    text: &str,
) -> Result<(Tensor<i64>, Tensor<i64>, usize)> {
    let tokenizer = tokenizer.context("This task requires a tokenizer but none was loaded")?;

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let seq_len = ids.len();

    let ids_tensor = Tensor::from_array(([1, seq_len], ids))
        .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;
    let mask_tensor = Tensor::from_array(([1, seq_len], mask))
        .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {}", e))?;

    Ok((ids_tensor, mask_tensor, seq_len))
}

/// Text classification
fn run_text_classification(
    session: &mut Session,
    input: &serde_json::Value,
    tokenizer: Option<&Tokenizer>,
) -> Result<serde_json::Value> {
    let text = input
        .get("text")
        .and_then(|v| v.as_str())
        .context("text-classification requires 'text' field")?;

    let (ids_tensor, mask_tensor, _seq_len) = tokenize_to_tensors(tokenizer, text)?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor
        ])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let (shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {}", e))?;

    let num_classes = *shape.last().unwrap_or(&0) as usize;
    let scores = softmax(&logits[..num_classes]);

    let (best_idx, best_score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    Ok(serde_json::json!({
        "label": format!("LABEL_{}", best_idx),
        "score": best_score,
        "scores": scores,
        "task": "text-classification"
    }))
}

/// Feature extraction (embeddings)
fn run_feature_extraction(
    session: &mut Session,
    input: &serde_json::Value,
    tokenizer: Option<&Tokenizer>,
) -> Result<serde_json::Value> {
    let text = input
        .get("text")
        .and_then(|v| v.as_str())
        .context("feature-extraction requires 'text' field")?;

    let tokenizer_ref = tokenizer.context("feature-extraction requires a tokenizer")?;
    let encoding = tokenizer_ref
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let seq_len = ids.len();

    let ids_tensor =
        Tensor::from_array(([1, seq_len], ids)).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mask_tensor =
        Tensor::from_array(([1, seq_len], mask.clone())).map_err(|e| anyhow::anyhow!("{}", e))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor
        ])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {}", e))?;

    let hidden_dim = *shape.last().unwrap_or(&0) as usize;

    // Mean pooling
    let mut embedding = vec![0.0f32; hidden_dim];
    if seq_len > 0 && hidden_dim > 0 {
        for token_idx in 0..seq_len {
            let mask_val = mask[token_idx] as f32;
            for dim in 0..hidden_dim {
                embedding[dim] += data[token_idx * hidden_dim + dim] * mask_val;
            }
        }
        let mask_sum: f32 = mask.iter().map(|&m| m as f32).sum::<f32>().max(1.0);
        for val in &mut embedding {
            *val /= mask_sum;
        }
    }

    Ok(serde_json::json!({
        "embedding": embedding,
        "dimensions": hidden_dim,
        "task": "feature-extraction"
    }))
}

/// Object detection
fn run_object_detection(
    session: &mut Session,
    input: &serde_json::Value,
) -> Result<serde_json::Value> {
    let image_b64 = input
        .get("image")
        .and_then(|v| v.as_str())
        .context("object-detection requires 'image' field (base64)")?;

    use base64::Engine;
    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(image_b64)
        .context("Failed to decode base64 image")?;

    let img = image::load_from_memory(&image_bytes).context("Failed to decode image")?;
    let resized = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mut pixels = vec![0.0f32; 3 * 640 * 640];
    for (i, pixel) in rgb.pixels().enumerate() {
        pixels[i] = pixel[0] as f32 / 255.0;
        pixels[640 * 640 + i] = pixel[1] as f32 / 255.0;
        pixels[2 * 640 * 640 + i] = pixel[2] as f32 / 255.0;
    }

    let input_tensor = Tensor::from_array(([1usize, 3, 640, 640], pixels))
        .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {}", e))?;

    let threshold = input
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    let mut detections = Vec::new();
    let stride = *shape.last().unwrap_or(&6) as usize;
    let num_dets = if shape.len() >= 2 {
        shape[shape.len() - 2] as usize
    } else {
        data.len() / stride
    };

    for i in 0..num_dets.min(1000) {
        let off = i * stride;
        if off + 5 > data.len() {
            break;
        }
        let conf = data[off + 4];
        if conf >= threshold {
            detections.push(serde_json::json!({
                "x_center": data[off], "y_center": data[off + 1],
                "width": data[off + 2], "height": data[off + 3],
                "confidence": conf,
                "class_id": data.get(off + 5).map(|&v| v as i32).unwrap_or(0),
            }));
        }
    }

    Ok(serde_json::json!({
        "detections": detections,
        "count": detections.len(),
        "task": "object-detection"
    }))
}

/// Image segmentation
fn run_image_segmentation(
    session: &mut Session,
    input: &serde_json::Value,
) -> Result<serde_json::Value> {
    let image_b64 = input
        .get("image")
        .and_then(|v| v.as_str())
        .context("image-segmentation requires 'image' field (base64)")?;

    use base64::Engine;
    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(image_b64)
        .context("Failed to decode base64 image")?;

    let img = image::load_from_memory(&image_bytes).context("Failed to decode image")?;
    let (orig_w, orig_h) = (img.width(), img.height());
    let size = 640usize;

    let resized = img.resize_exact(
        size as u32,
        size as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    let mut pixels = vec![0.0f32; 3 * size * size];
    for (i, pixel) in rgb.pixels().enumerate() {
        pixels[i] = pixel[0] as f32 / 255.0;
        pixels[size * size + i] = pixel[1] as f32 / 255.0;
        pixels[2 * size * size + i] = pixel[2] as f32 / 255.0;
    }

    let input_tensor = Tensor::from_array(([1usize, 3, size, size], pixels))
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let (shape, _data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output: {}", e))?;

    let num_classes = if shape.len() >= 2 {
        shape[1] as usize
    } else {
        1
    };

    Ok(serde_json::json!({
        "output_shape": shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        "num_classes": num_classes,
        "original_size": [orig_w, orig_h],
        "task": "image-segmentation"
    }))
}

/// Question answering
fn run_question_answering(
    session: &mut Session,
    input: &serde_json::Value,
    tokenizer: Option<&Tokenizer>,
) -> Result<serde_json::Value> {
    let question = input
        .get("question")
        .and_then(|v| v.as_str())
        .context("question-answering requires 'question' field")?;
    let context = input
        .get("context")
        .and_then(|v| v.as_str())
        .context("question-answering requires 'context' field")?;

    let tokenizer = tokenizer.context("QA requires a tokenizer")?;
    let encoding = tokenizer
        .encode((question, context), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let seq_len = ids.len();

    let ids_tensor =
        Tensor::from_array(([1, seq_len], ids)).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mask_tensor =
        Tensor::from_array(([1, seq_len], mask)).map_err(|e| anyhow::anyhow!("{}", e))?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor
        ])
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    let (_s1, start_logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let (_s2, end_logits) = outputs[1]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let start_idx = argmax(&start_logits[..seq_len]);
    let end_idx = argmax(&end_logits[..seq_len]).max(start_idx);

    let answer_ids: Vec<u32> = encoding.get_ids()[start_idx..=end_idx].to_vec();
    let answer = tokenizer.decode(&answer_ids, true).unwrap_or_default();
    let score = start_logits[start_idx] + end_logits[end_idx];

    Ok(serde_json::json!({
        "answer": answer, "score": score,
        "start": start_idx, "end": end_idx,
        "task": "question-answering"
    }))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
