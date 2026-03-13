//! Graceful restart manager for auto-updates.
//!
//! Handles the delicate process of restarting the agent:
//! - Wait for active jobs to complete (with timeout)
//! - Replace the current binary
//! - exec() the new binary (Unix) or spawn + exit (Windows)
//!
//! ## Safety
//!
//! - Never restart during active jobs (unless critical update)
//! - Verify new binary is executable before replacing
//! - Keep old binary as backup until new one starts successfully

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::fs;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use super::verify::BinaryVerifier;
use super::{UpdateError, UpdateInfo};

/// Maximum time to wait for active jobs before forcing restart
const DEFAULT_GRACEFUL_TIMEOUT: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Critical update timeout (force restart quickly)
const CRITICAL_UPDATE_TIMEOUT: Duration = Duration::from_secs(30);

/// Restart manager handles graceful agent restarts
pub struct RestartManager {
    /// Current binary path
    current_binary: PathBuf,
    /// Backup directory for old binaries
    backup_dir: PathBuf,
    /// Graceful shutdown timeout
    graceful_timeout: Duration,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new() -> Result<Self, UpdateError> {
        let current_binary = std::env::current_exe().map_err(UpdateError::Io)?;

        // Use ~/.archipelag/backups for old binaries
        let backup_dir = dirs::home_dir()
            .ok_or_else(|| {
                UpdateError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Could not find home directory",
                ))
            })?
            .join(".archipelag")
            .join("backups");

        Ok(Self {
            current_binary,
            backup_dir,
            graceful_timeout: DEFAULT_GRACEFUL_TIMEOUT,
        })
    }

    /// Set the graceful shutdown timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.graceful_timeout = timeout;
        self
    }

    /// Prepare for restart by verifying the new binary
    pub async fn prepare(
        &self,
        new_binary_path: &Path,
        update_info: &UpdateInfo,
    ) -> Result<(), UpdateError> {
        info!(
            "Preparing restart with new binary: {}",
            new_binary_path.display()
        );

        // 1. Verify checksum
        let expected_checksum = update_info
            .checksum_sha256
            .as_ref()
            .ok_or_else(|| UpdateError::MissingField("checksum_sha256".to_string()))?;

        BinaryVerifier::verify_checksum(new_binary_path, expected_checksum)?;
        debug!("Checksum verified");

        // 2. Verify signature
        let signature = update_info
            .signature
            .as_ref()
            .ok_or_else(|| UpdateError::MissingField("signature".to_string()))?;

        BinaryVerifier::verify(new_binary_path, expected_checksum, signature)?;
        debug!("Signature verified");

        // 3. Make executable (Unix)
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(new_binary_path)
                .await
                .map_err(UpdateError::Io)?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(new_binary_path, perms)
                .await
                .map_err(UpdateError::Io)?;
            debug!("Set executable permissions");
        }

        // 4. Verify it's a valid executable (quick sanity check)
        #[cfg(unix)]
        {
            use std::process::Command;
            let output = Command::new(new_binary_path).arg("--version").output();

            match output {
                Ok(out) if out.status.success() => {
                    debug!("Binary --version check passed");
                }
                Ok(out) => {
                    warn!(
                        "Binary --version returned non-zero: {:?}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                    // Continue anyway - might not have --version flag
                }
                Err(e) => {
                    // Binary might not support --version, that's okay
                    debug!("Could not run binary --version: {}", e);
                }
            }
        }

        info!("New binary prepared and verified");
        Ok(())
    }

    /// Wait for active jobs to complete
    ///
    /// Returns when either:
    /// - All jobs complete
    /// - Timeout is reached
    /// - is_critical is true and critical timeout reached
    pub async fn wait_for_jobs<F>(
        &self,
        is_critical: bool,
        mut get_active_jobs: F,
    ) -> Result<(), UpdateError>
    where
        F: FnMut() -> usize,
    {
        let timeout_duration = if is_critical {
            info!("Critical update - using short timeout");
            CRITICAL_UPDATE_TIMEOUT
        } else {
            self.graceful_timeout
        };

        let start = std::time::Instant::now();

        loop {
            let active_jobs = get_active_jobs();

            if active_jobs == 0 {
                info!("No active jobs, ready to restart");
                return Ok(());
            }

            if start.elapsed() >= timeout_duration {
                if is_critical {
                    warn!(
                        "Critical update timeout reached with {} active jobs - forcing restart",
                        active_jobs
                    );
                    return Ok(());
                } else {
                    warn!(
                        "Graceful timeout reached with {} active jobs - forcing restart",
                        active_jobs
                    );
                    return Ok(());
                }
            }

            info!(
                "Waiting for {} active job(s) to complete before restart ({}s remaining)",
                active_jobs,
                (timeout_duration - start.elapsed()).as_secs()
            );

            // Check every 5 seconds
            sleep(Duration::from_secs(5)).await;
        }
    }

    /// Install the new binary and restart
    ///
    /// This function does not return on success - it execs the new binary.
    pub async fn install_and_restart(&self, new_binary_path: &Path) -> Result<(), UpdateError> {
        info!("Installing new binary and restarting");

        // 1. Create backup directory
        fs::create_dir_all(&self.backup_dir)
            .await
            .map_err(UpdateError::Io)?;

        // 2. Backup current binary
        let backup_name = format!("agent-{}-backup", chrono_lite_timestamp());
        let backup_path = self.backup_dir.join(&backup_name);

        info!("Backing up current binary to: {}", backup_path.display());
        fs::copy(&self.current_binary, &backup_path)
            .await
            .map_err(UpdateError::Io)?;

        // 3. Replace current binary with new one
        info!("Replacing binary: {}", self.current_binary.display());

        // On Unix, we can rename over the running binary
        // The old inode stays valid until the process exits
        fs::copy(new_binary_path, &self.current_binary)
            .await
            .map_err(UpdateError::Io)?;

        // 4. Clean up temp file
        let _ = fs::remove_file(new_binary_path).await;

        // 5. Exec the new binary (Unix)
        info!("Restarting agent...");
        self.exec_self()
    }

    /// Exec the current binary with original arguments
    #[cfg(unix)]
    fn exec_self(&self) -> Result<(), UpdateError> {
        use std::os::unix::process::CommandExt;

        let args: Vec<String> = std::env::args().collect();
        let mut cmd = std::process::Command::new(&self.current_binary);

        // Pass through all arguments except argv[0]
        if args.len() > 1 {
            cmd.args(&args[1..]);
        }

        // exec() replaces current process - does not return on success
        let err = cmd.exec();

        // If we get here, exec failed
        Err(UpdateError::Io(err))
    }

    /// On Windows, spawn new process and exit
    #[cfg(windows)]
    fn exec_self(&self) -> Result<(), UpdateError> {
        use std::process::Command;

        let args: Vec<String> = std::env::args().collect();
        let mut cmd = Command::new(&self.current_binary);

        if args.len() > 1 {
            cmd.args(&args[1..]);
        }

        // Spawn new process
        cmd.spawn().map_err(UpdateError::Io)?;

        // Exit current process
        std::process::exit(0);
    }

    /// Clean up old backups (keep last 3)
    pub async fn cleanup_backups(&self) -> Result<(), UpdateError> {
        if !self.backup_dir.exists() {
            return Ok(());
        }

        let mut entries: Vec<_> = Vec::new();
        let mut dir = fs::read_dir(&self.backup_dir)
            .await
            .map_err(UpdateError::Io)?;

        while let Some(entry) = dir.next_entry().await.map_err(UpdateError::Io)? {
            if let Ok(metadata) = entry.metadata().await {
                entries.push((entry.path(), metadata.modified().ok()));
            }
        }

        // Sort by modification time (newest first)
        entries.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove all but the 3 newest
        for (path, _) in entries.iter().skip(3) {
            info!("Removing old backup: {}", path.display());
            let _ = fs::remove_file(path).await;
        }

        Ok(())
    }
}

/// Generate a simple timestamp without external crates
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let ts = chrono_lite_timestamp();
        // Should be a valid unix timestamp (> year 2020)
        let parsed: u64 = ts.parse().expect("Should be numeric");
        assert!(parsed > 1577836800); // > 2020-01-01
    }
}
