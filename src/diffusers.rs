//! Native diffusers runtime for Stable Diffusion pipelines
//!
//! Uses candle-transformers for inference. Downloads model components from
//! HuggingFace Hub and runs the full text-to-image diffusion pipeline.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use hf_hub::api::sync::Api;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{watch, RwLock};
use tracing::info;

use crate::config::AgentConfig;
use crate::nats::{AssignJob, NatsAgent};
use crate::state::StateManager;

/// Model files needed for Stable Diffusion
struct ModelFiles {
    tokenizer: std::path::PathBuf,
    clip_weights: std::path::PathBuf,
    vae_weights: std::path::PathBuf,
    unet_weights: std::path::PathBuf,
}

/// Execute a diffusers workload
pub async fn execute_diffusers_job(
    nats: &NatsAgent,
    state: &Arc<RwLock<StateManager>>,
    _config: &AgentConfig,
    job: &AssignJob,
    mut cancel_rx: watch::Receiver<bool>,
) -> Result<()> {
    let job_id = &job.job_id;

    let model_uri = job
        .model_url
        .as_ref()
        .context("Diffusers workload missing model_url")?;

    info!("Executing diffusers workload: {}", model_uri);

    // Extract parameters from input
    let prompt = job
        .input
        .get("prompt")
        .and_then(|v| v.as_str())
        .context("Diffusers workload requires 'prompt' in input")?
        .to_string();

    let negative_prompt = job
        .input
        .get("negative_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let num_steps = job
        .input
        .get("num_steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;

    let guidance_scale = job
        .input
        .get("guidance_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(7.5);

    let width = job
        .input
        .get("width")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let height = job
        .input
        .get("height")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;

    let seed = job
        .input
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(42);

    info!(
        "Diffusion params: prompt='{}', steps={}, guidance={}, {}x{}, seed={}",
        prompt, num_steps, guidance_scale, width, height, seed
    );

    // Check cancellation before download
    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // Emit download progress
    nats.publish_output(
        job_id,
        0,
        &serde_json::json!({"type": "progress", "phase": "downloading", "step": 0, "total": num_steps}).to_string(),
        false,
    )
    .await?;

    // Resolve HuggingFace repo ID
    let repo_id = model_uri
        .strip_prefix("hf://")
        .unwrap_or(model_uri)
        .split(':')
        .next()
        .unwrap_or(model_uri)
        .to_string();

    // Download model files from HuggingFace (blocking)
    let repo_id_clone = repo_id.clone();
    let model_files = tokio::task::spawn_blocking(move || -> Result<ModelFiles> {
        let api = Api::new().context("Failed to create HuggingFace Hub API client")?;
        let repo = api.model(repo_id_clone);

        info!("Downloading model components from HuggingFace...");

        let tokenizer = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let clip_weights = repo
            .get("text_encoder/model.safetensors")
            .context("Failed to download CLIP weights")?;
        let vae_weights = repo
            .get("vae/diffusion_pytorch_model.safetensors")
            .context("Failed to download VAE weights")?;
        let unet_weights = repo
            .get("unet/diffusion_pytorch_model.safetensors")
            .context("Failed to download UNet weights")?;

        info!("All model components downloaded");

        Ok(ModelFiles {
            tokenizer,
            clip_weights,
            vae_weights,
            unet_weights,
        })
    })
    .await??;

    // Check cancellation after download
    if *cancel_rx.borrow() {
        nats.publish_status(job_id, "cancelled", None).await?;
        return Ok(());
    }

    // Run inference (blocking — heavy compute)
    let nats_clone = nats.clone();
    let job_id_clone = job_id.clone();
    let cancel = cancel_rx.clone();

    let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // SD v1.5 configuration
        let sd_config =
            stable_diffusion::StableDiffusionConfig::v1_5(None, Some(height), Some(width));

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&model_files.tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Tokenize prompt
        let pad_id = *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap_or(&49407);
        let max_len = 77; // CLIP max sequence length

        let tokens = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {}", e))?;
        let mut token_ids: Vec<u32> = tokens.get_ids().to_vec();
        token_ids.truncate(max_len);
        while token_ids.len() < max_len {
            token_ids.push(pad_id);
        }

        // Tokenize negative prompt for classifier-free guidance
        let uncond_tokens = tokenizer
            .encode(negative_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize negative prompt: {}", e))?;
        let mut uncond_ids: Vec<u32> = uncond_tokens.get_ids().to_vec();
        uncond_ids.truncate(max_len);
        while uncond_ids.len() < max_len {
            uncond_ids.push(pad_id);
        }

        let token_ids = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
        let uncond_ids = Tensor::new(uncond_ids.as_slice(), &device)?.unsqueeze(0)?;

        // Load CLIP text encoder
        info!("Loading CLIP text encoder...");
        let text_model = stable_diffusion::build_clip_transformer(
            &sd_config.clip,
            &model_files.clip_weights,
            &device,
            DType::F32,
        )?;

        // Encode text
        let text_embeddings = text_model.forward(&token_ids)?;
        let uncond_embeddings = text_model.forward(&uncond_ids)?;
        let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?;
        drop(text_model); // Free CLIP memory

        // Load VAE
        info!("Loading VAE...");
        let vae = sd_config.build_vae(&model_files.vae_weights, &device, dtype)?;

        // Load UNet
        info!("Loading UNet...");
        let unet = sd_config.build_unet(&model_files.unet_weights, &device, 4, false, dtype)?;

        // Create scheduler
        let mut scheduler = sd_config.build_scheduler(num_steps)?;
        let timesteps = scheduler.timesteps().to_vec();

        // Initialize random latents
        let latent_height = height / 8;
        let latent_width = width / 8;
        let mut latents =
            Tensor::randn(0f32, 1f32, (1, 4, latent_height, latent_width), &device)?;

        // Scale initial noise by scheduler's init noise sigma
        latents = (latents * scheduler.init_noise_sigma())?;

        info!("Starting diffusion ({} steps)...", num_steps);

        // Diffusion loop
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            // Check cancellation
            if *cancel.borrow() {
                anyhow::bail!("Cancelled");
            }

            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
            let latent_model_input =
                scheduler.scale_model_input(latent_model_input, timestep)?;

            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            // Classifier-free guidance
            let noise_pred = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &noise_pred[0];
            let noise_pred_text = &noise_pred[1];
            let noise_pred = (noise_pred_uncond
                + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?;

            latents = scheduler.step(&noise_pred, timestep, &latents)?;

            // Emit progress (fire-and-forget from blocking context)
            let rt = tokio::runtime::Handle::current();
            let n = nats_clone.clone();
            let jid = job_id_clone.clone();
            let step = step_idx + 1;
            let total = num_steps;
            rt.spawn(async move {
                let progress = serde_json::json!({
                    "type": "progress",
                    "phase": "diffusion",
                    "step": step,
                    "total": total
                });
                let _ = n
                    .publish_output(&jid, step as u64, &progress.to_string(), false)
                    .await;
            });
        }

        // Decode latents to image
        info!("Decoding latents to image...");
        let vae_scale = 0.18215f64;
        let scaled_latents = (&latents / vae_scale)?;
        let images = vae.decode(&scaled_latents)?;
        let images = ((images / 2.)? + 0.5)?;
        let images = (images.clamp(0f32, 1f32)? * 255.)?;

        // Extract first image and convert to RGB bytes
        // images shape: [batch, channels, height, width]
        let image = images.get(0)?; // batch dim
        let (channels, img_h, img_w) = image.dims3()?;
        let mut rgb_data = vec![0u8; img_h * img_w * 3];

        for c in 0..channels.min(3) {
            let channel = image.get(c)?; // channel dim
            let channel_data = channel.flatten_all()?.to_vec1::<f32>()?;
            for (i, &val) in channel_data.iter().enumerate() {
                rgb_data[i * 3 + c] = val.clamp(0.0, 255.0) as u8;
            }
        }

        // Encode as PNG
        let mut png_data = Vec::new();
        {
            let encoder = image::codecs::png::PngEncoder::new(&mut png_data);
            image::ImageEncoder::write_image(
                encoder,
                &rgb_data,
                img_w as u32,
                img_h as u32,
                image::ExtendedColorType::Rgb8,
            )
            .context("Failed to encode PNG")?;
        }

        Ok(png_data)
    })
    .await?;

    match result {
        Ok(png_data) => {
            let b64 = base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &png_data,
            );

            let result_event = serde_json::json!({
                "type": "image",
                "format": "png",
                "width": width,
                "height": height,
                "data": b64,
                "size_bytes": png_data.len()
            });

            nats.publish_output(
                job_id,
                (num_steps + 1) as u64,
                &result_event.to_string(),
                true,
            )
            .await?;
            nats.publish_status(job_id, "completed", None).await?;
            info!("Diffusers job {} completed: {}x{} PNG ({} bytes)", job_id, width, height, png_data.len());
        }
        Err(e) => {
            let msg = format!("{}", e);
            if msg.contains("Cancelled") {
                nats.publish_status(job_id, "cancelled", None).await?;
            } else {
                nats.publish_status(job_id, "failed", Some(format!("Diffusers error: {}", e)))
                    .await?;
            }
        }
    }

    Ok(())
}
