//! OCI image layer unpacking.
//!
//! Extracts tar.gz layers from a pulled image into a rootfs directory,
//! applying them in order (bottom-up) as specified by the manifest.

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::path::Path;
use tar::Archive;
use tracing::{debug, info, warn};

/// Unpack a pulled image's layers into a rootfs directory.
///
/// Layers are applied in order from the manifest (bottom-up).
/// Whiteout files (.wh.*) are handled by deleting the corresponding entry.
pub fn unpack_image(cache_dir: &Path, image: &str, rootfs_dir: &Path) -> Result<()> {
    let image_cache = cache_dir.join(super::pull::sanitize_image_name(image));

    // Read manifest to get layer order
    let manifest_path = image_cache.join("manifest.json");
    let manifest_data = fs::read_to_string(&manifest_path)
        .context("Manifest not found — pull the image first")?;

    let manifest: oci_distribution::manifest::OciImageManifest =
        serde_json::from_str(&manifest_data).context("Invalid manifest")?;

    info!("Unpacking {} layers into {}", manifest.layers.len(), rootfs_dir.display());

    for (i, layer) in manifest.layers.iter().enumerate() {
        let layer_filename = format!("{}.tar.gz", layer.digest.replace(':', "_"));
        let layer_path = image_cache.join(&layer_filename);

        if !layer_path.exists() {
            anyhow::bail!("Layer file not found: {}", layer_path.display());
        }

        debug!(
            "Unpacking layer {}/{}: {}",
            i + 1,
            manifest.layers.len(),
            &layer.digest[..19]
        );

        unpack_layer(&layer_path, rootfs_dir)
            .with_context(|| format!("Failed to unpack layer {}", layer.digest))?;
    }

    info!("Rootfs ready: {}", rootfs_dir.display());
    Ok(())
}

/// Unpack a single tar.gz layer into the rootfs
fn unpack_layer(layer_path: &Path, rootfs_dir: &Path) -> Result<()> {
    let file = File::open(layer_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    // Don't preserve permissions/ownership issues from the image
    archive.set_preserve_permissions(false);
    archive.set_preserve_mtime(false);

    for entry in archive.entries()? {
        let mut entry = match entry {
            Ok(e) => e,
            Err(e) => {
                debug!("Skipping malformed entry: {}", e);
                continue;
            }
        };

        let path = match entry.path() {
            Ok(p) => p.to_path_buf(),
            Err(_) => continue,
        };

        let path_str = path.to_string_lossy();

        // Handle OCI whiteout files
        if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
            if filename.starts_with(".wh.") {
                handle_whiteout(&path_str, rootfs_dir);
                continue;
            }
        }

        // Skip paths that could escape the rootfs
        if path_str.contains("..") {
            warn!("Skipping suspicious path: {}", path_str);
            continue;
        }

        // Extract to rootfs
        let dest = rootfs_dir.join(&path);

        // Create parent directories
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).ok();
        }

        if let Err(e) = entry.unpack_in(rootfs_dir) {
            debug!("Failed to unpack {}: {} (continuing)", path_str, e);
        }
    }

    Ok(())
}

/// Handle OCI whiteout files by deleting the corresponding entry
///
/// .wh.filename → delete filename
/// .wh..wh..opq → delete all children of the directory (opaque whiteout)
fn handle_whiteout(path_str: &str, rootfs_dir: &Path) {
    let path = std::path::Path::new(path_str);
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();

    if filename == ".wh..wh..opq" {
        // Opaque whiteout — clear the directory
        if let Some(parent) = path.parent() {
            let target = rootfs_dir.join(parent);
            if target.is_dir() {
                debug!("Opaque whiteout: clearing {}", target.display());
                if let Ok(entries) = fs::read_dir(&target) {
                    for entry in entries.flatten() {
                        let _ = fs::remove_dir_all(entry.path());
                    }
                }
            }
        }
    } else if let Some(target_name) = filename.strip_prefix(".wh.") {
        // Regular whiteout — delete the specific file
        if let Some(parent) = path.parent() {
            let target = rootfs_dir.join(parent).join(target_name);
            debug!("Whiteout: removing {}", target.display());
            if target.is_dir() {
                let _ = fs::remove_dir_all(&target);
            } else {
                let _ = fs::remove_file(&target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a tar.gz at `path` from a list of (entry_name, content) pairs.
    fn create_tar_gz(path: &std::path::Path, entries: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);

        for (name, content) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *name, &content[..])
                .unwrap();
        }
        builder.finish().unwrap();
    }

    #[test]
    fn test_unpack_layer_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let layer_path = tmp.path().join("test.tar.gz");
        let rootfs = tmp.path().join("rootfs");
        fs::create_dir_all(&rootfs).unwrap();

        create_tar_gz(&layer_path, &[("test.txt", b"hello world")]);

        unpack_layer(&layer_path, &rootfs).unwrap();

        let result = fs::read_to_string(rootfs.join("test.txt")).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_whiteout_deletes_target_file() {
        let tmp = tempfile::tempdir().unwrap();
        let rootfs = tmp.path().join("rootfs");
        fs::create_dir_all(&rootfs).unwrap();

        // Pre-create the file that should be whited-out
        fs::write(rootfs.join("old.txt"), "should be deleted").unwrap();
        assert!(rootfs.join("old.txt").exists());

        // Apply a whiteout: .wh.old.txt in the root directory
        handle_whiteout(".wh.old.txt", &rootfs);

        assert!(!rootfs.join("old.txt").exists());
    }

    #[test]
    fn test_whiteout_deletes_target_in_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let rootfs = tmp.path().join("rootfs");
        let sub = rootfs.join("etc");
        fs::create_dir_all(&sub).unwrap();

        fs::write(sub.join("config"), "old config").unwrap();
        assert!(sub.join("config").exists());

        handle_whiteout("etc/.wh.config", &rootfs);

        assert!(!sub.join("config").exists());
    }

    #[test]
    fn test_opaque_whiteout_clears_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let rootfs = tmp.path().join("rootfs");
        let sub = rootfs.join("var");
        fs::create_dir_all(&sub).unwrap();

        // Create some files in the directory
        fs::write(sub.join("a.txt"), "aaa").unwrap();
        fs::write(sub.join("b.txt"), "bbb").unwrap();

        handle_whiteout("var/.wh..wh..opq", &rootfs);

        // Directory itself still exists, but children should be gone
        assert!(sub.exists());
        assert!(fs::read_dir(&sub).unwrap().next().is_none());
    }

    #[test]
    fn test_whiteout_nonexistent_file_is_safe() {
        let tmp = tempfile::tempdir().unwrap();
        let rootfs = tmp.path().join("rootfs");
        fs::create_dir_all(&rootfs).unwrap();

        // Whiteout for a file that doesn't exist — should not panic
        handle_whiteout(".wh.ghost.txt", &rootfs);
        // No assertion needed; we just verify it doesn't panic
    }

    #[test]
    fn test_path_traversal_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let layer_path = tmp.path().join("evil.tar.gz");
        let rootfs = tmp.path().join("rootfs");
        fs::create_dir_all(&rootfs).unwrap();

        // Create a tar.gz with a path containing ".."
        create_tar_gz(&layer_path, &[("../../etc/passwd", b"evil")]);

        unpack_layer(&layer_path, &rootfs).unwrap();

        // The traversal path should have been skipped
        assert!(!rootfs.join("etc/passwd").exists());
        // Also verify nothing was written outside rootfs
        assert!(!tmp.path().join("etc").exists());
    }

    #[test]
    fn test_normal_path_extracted() {
        let tmp = tempfile::tempdir().unwrap();
        let layer_path = tmp.path().join("normal.tar.gz");
        let rootfs = tmp.path().join("rootfs");
        fs::create_dir_all(&rootfs).unwrap();

        create_tar_gz(
            &layer_path,
            &[
                ("usr/bin/app", b"binary content"),
                ("etc/config.toml", b"[settings]\nfoo = true"),
            ],
        );

        unpack_layer(&layer_path, &rootfs).unwrap();

        assert_eq!(
            fs::read_to_string(rootfs.join("usr/bin/app")).unwrap(),
            "binary content"
        );
        assert_eq!(
            fs::read_to_string(rootfs.join("etc/config.toml")).unwrap(),
            "[settings]\nfoo = true"
        );
    }
}
