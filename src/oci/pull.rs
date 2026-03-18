//! OCI image pulling via the Distribution Spec.
//!
//! Downloads container image manifests and layers directly from registries
//! (Docker Hub, GHCR, etc.) without requiring a Docker daemon.

use anyhow::{Context, Result};
use oci_distribution::client::{ClientConfig, ClientProtocol};
use oci_distribution::secrets::RegistryAuth;
use oci_distribution::{Client, Reference};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};

/// Pull an OCI image's layers to the cache directory.
///
/// Layers are stored as `<cache_dir>/<image_hash>/<layer_digest>.tar.gz`.
/// If layers already exist, they are not re-downloaded.
pub async fn pull_image(image: &str, cache_dir: &Path) -> Result<()> {
    let reference: Reference = image
        .parse()
        .context(format!("Invalid image reference: {}", image))?;

    info!("Pulling image: {}", reference);

    let client_config = ClientConfig {
        protocol: ClientProtocol::Https,
        ..Default::default()
    };
    let mut client = Client::new(client_config);

    // Use anonymous auth (public images)
    let auth = RegistryAuth::Anonymous;

    // Create a cache subdir based on image name
    let image_cache = cache_dir.join(sanitize_image_name(image));
    fs::create_dir_all(&image_cache).await?;

    // Pull manifest
    let (manifest, _digest) = client
        .pull_image_manifest(&reference, &auth)
        .await
        .context("Failed to pull image manifest")?;

    debug!(
        "Image has {} layers, config: {}",
        manifest.layers.len(),
        manifest.config.digest
    );

    // Save manifest
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(image_cache.join("manifest.json"), &manifest_json).await?;

    // Pull config blob
    let config_path = image_cache.join("config.json");
    if !config_path.exists() {
        let mut config_data = Vec::new();
        client
            .pull_blob(&reference, &manifest.config, &mut config_data)
            .await
            .context("Failed to pull image config")?;
        fs::write(&config_path, &config_data).await?;
        debug!("Pulled config: {} bytes", config_data.len());
    }

    // Pull each layer
    for (i, layer) in manifest.layers.iter().enumerate() {
        let layer_filename = format!("{}.tar.gz", layer.digest.replace(':', "_"));
        let layer_path = image_cache.join(&layer_filename);

        if layer_path.exists() {
            debug!(
                "Layer {}/{} already cached: {}",
                i + 1,
                manifest.layers.len(),
                layer.digest
            );
            continue;
        }

        info!(
            "Pulling layer {}/{}: {} ({} bytes)",
            i + 1,
            manifest.layers.len(),
            &layer.digest[..19],
            layer.size
        );

        let mut layer_data = Vec::new();
        client
            .pull_blob(&reference, layer, &mut layer_data)
            .await
            .context(format!("Failed to pull layer {}", layer.digest))?;

        fs::write(&layer_path, &layer_data).await?;
        debug!("Layer saved: {}", layer_path.display());
    }

    info!("Image pulled: {} ({} layers)", image, manifest.layers.len());
    Ok(())
}

/// Sanitize an image name for use as a directory name
pub(crate) fn sanitize_image_name(image: &str) -> String {
    image.replace('/', "_").replace(':', "_").replace('@', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_image_name() {
        assert_eq!(
            sanitize_image_name("ghcr.io/archipelag-io/llm-chat:latest"),
            "ghcr.io_archipelag-io_llm-chat_latest"
        );
        assert_eq!(sanitize_image_name("alpine:3.18"), "alpine_3.18");
    }
}
