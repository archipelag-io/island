//! Trusted Execution Environment (TEE) detection and attestation.
//!
//! Detects TEE hardware capabilities on the Island and generates
//! attestation reports for the coordinator.
//!
//! Supported TEE types:
//! - Intel SGX (via /dev/sgx_enclave or /dev/isgx)
//! - AMD SEV (via /dev/sev or /dev/sev-guest)
//! - ARM TrustZone (via /dev/tee0)
//!
//! The attestation hash is a proof that the Island has legitimate TEE
//! hardware. In production, this would be a full attestation report
//! verified against the vendor's attestation service.

use serde::{Deserialize, Serialize};
use tracing::info;

/// Detected TEE type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TeeType {
    Sgx,
    Sev,
    TrustZone,
    Nitro,
    None,
}

impl TeeType {
    pub fn as_str(&self) -> &str {
        match self {
            TeeType::Sgx => "sgx",
            TeeType::Sev => "sev",
            TeeType::TrustZone => "trustzone",
            TeeType::Nitro => "nitro",
            TeeType::None => "none",
        }
    }
}

/// TEE detection result
#[derive(Debug, Clone, Serialize)]
pub struct TeeInfo {
    pub tee_type: TeeType,
    pub available: bool,
    pub attestation_hash: Option<String>,
}

/// Detect TEE hardware on this machine.
pub fn detect_tee() -> TeeInfo {
    // Check for Intel SGX
    if check_sgx() {
        let hash = generate_attestation_hash("sgx");
        info!("TEE detected: Intel SGX");
        return TeeInfo {
            tee_type: TeeType::Sgx,
            available: true,
            attestation_hash: Some(hash),
        };
    }

    // Check for AMD SEV
    if check_sev() {
        let hash = generate_attestation_hash("sev");
        info!("TEE detected: AMD SEV");
        return TeeInfo {
            tee_type: TeeType::Sev,
            available: true,
            attestation_hash: Some(hash),
        };
    }

    // Check for ARM TrustZone
    if check_trustzone() {
        let hash = generate_attestation_hash("trustzone");
        info!("TEE detected: ARM TrustZone");
        return TeeInfo {
            tee_type: TeeType::TrustZone,
            available: true,
            attestation_hash: Some(hash),
        };
    }

    // Check for AWS Nitro
    if check_nitro() {
        let hash = generate_attestation_hash("nitro");
        info!("TEE detected: AWS Nitro Enclaves");
        return TeeInfo {
            tee_type: TeeType::Nitro,
            available: true,
            attestation_hash: Some(hash),
        };
    }

    info!("No TEE hardware detected");
    TeeInfo {
        tee_type: TeeType::None,
        available: false,
        attestation_hash: None,
    }
}

/// Check for Intel SGX device
fn check_sgx() -> bool {
    std::path::Path::new("/dev/sgx_enclave").exists()
        || std::path::Path::new("/dev/isgx").exists()
}

/// Check for AMD SEV device
fn check_sev() -> bool {
    std::path::Path::new("/dev/sev").exists()
        || std::path::Path::new("/dev/sev-guest").exists()
}

/// Check for ARM TrustZone device
fn check_trustzone() -> bool {
    std::path::Path::new("/dev/tee0").exists()
}

/// Check for AWS Nitro
fn check_nitro() -> bool {
    // Nitro enclaves are detected via the vsock device
    std::path::Path::new("/dev/vsock").exists()
        && std::path::Path::new("/sys/devices/virtual/misc/nitro_enclaves").exists()
}

/// Generate an attestation hash.
///
/// In production, this would:
/// - SGX: generate a quote via AESM service
/// - SEV: generate an attestation report via /dev/sev-guest
/// - TrustZone: generate a signed certificate
/// - Nitro: generate PCR values via nsm_lib
///
/// For now: SHA256 hash of machine identity + TEE type + timestamp.
fn generate_attestation_hash(tee_type: &str) -> String {
    use sha2::{Digest, Sha256};

    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let input = format!("{}:{}:{}", tee_type, hostname, timestamp);
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_tee() {
        let info = detect_tee();
        // On most dev machines, no TEE is available
        assert!(info.tee_type == TeeType::None || info.available);
    }

    #[test]
    fn test_tee_type_serialize() {
        assert_eq!(serde_json::to_string(&TeeType::Sgx).unwrap(), "\"sgx\"");
        assert_eq!(serde_json::to_string(&TeeType::None).unwrap(), "\"none\"");
    }

    #[test]
    fn test_attestation_hash_generation() {
        let hash = generate_attestation_hash("sgx");
        assert_eq!(hash.len(), 64); // SHA256 hex
    }
}
