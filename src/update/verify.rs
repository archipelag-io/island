//! Binary verification using Ed25519 signatures and SHA256 checksums.
//!
//! ## Security Model
//!
//! - SHA256 checksum verifies binary integrity (no corruption)
//! - Ed25519 signature verifies authenticity (signed by us)
//! - Multiple public keys supported for key rotation
//!
//! ## Key Rotation
//!
//! When rotating keys:
//! 1. Add new public key to SIGNING_PUBLIC_KEYS
//! 2. Start signing with new key
//! 3. After all agents updated, remove old key

use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Embedded public keys for signature verification.
///
/// Format: (key_id, public_key_bytes)
/// Key IDs help identify which key was used in logs.
///
/// To generate a new keypair:
/// ```bash
/// openssl genpkey -algorithm ED25519 -out private.pem
/// openssl pkey -in private.pem -pubout -out public.pem
/// # Extract raw bytes from public.pem
/// ```
const SIGNING_PUBLIC_KEYS: &[(&str, &[u8; 32])] = &[
    // Primary release signing key (2026-03)
    // Generated via: openssl genpkey -algorithm ED25519
    // Private key stored in GitHub Secrets as RELEASE_SIGNING_PRIVATE_KEY
    (
        "release-2026-03",
        b"\xa9\x0c\x6a\xda\x9b\xd0\x73\x19\x1d\x6d\xbd\x9a\x88\x38\xb1\xa6\x85\x14\xa2\x5f\x35\xfc\x3f\xf5\x43\x19\x04\xd1\x5b\x36\x9b\x6d",
    ),
];

/// Binary verification errors
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Invalid signature")]
    SignatureInvalid,

    #[error("Invalid signature format: {0}")]
    SignatureFormat(String),

    #[error("Invalid public key: {0}")]
    PublicKeyInvalid(String),

    #[error("No valid signature found (tried {0} keys)")]
    NoValidSignature(usize),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hex decode error: {0}")]
    HexDecode(#[from] hex::FromHexError),
}

/// Binary verifier using Ed25519 signatures and SHA256 checksums
pub struct BinaryVerifier;

impl BinaryVerifier {
    /// Verify a downloaded binary.
    ///
    /// # Arguments
    /// * `binary_path` - Path to the downloaded binary
    /// * `expected_checksum` - Expected SHA256 checksum (hex string, 64 chars)
    /// * `signature` - Ed25519 signature (hex string, 128 chars)
    ///
    /// # Returns
    /// * `Ok(())` if verification passes
    /// * `Err(VerifyError)` if verification fails
    pub fn verify(
        binary_path: &Path,
        expected_checksum: &str,
        signature: &str,
    ) -> Result<(), VerifyError> {
        tracing::info!("Verifying binary: {}", binary_path.display());

        // 1. Verify SHA256 checksum
        let actual_checksum = Self::compute_sha256(binary_path)?;
        if actual_checksum != expected_checksum.to_lowercase() {
            return Err(VerifyError::ChecksumMismatch {
                expected: expected_checksum.to_string(),
                actual: actual_checksum,
            });
        }
        tracing::debug!("Checksum verified: {}", actual_checksum);

        // 2. Read binary contents for signature verification
        let file_contents = std::fs::read(binary_path)?;

        // 3. Decode signature from hex
        let signature_bytes = hex::decode(signature)?;
        if signature_bytes.len() != 64 {
            return Err(VerifyError::SignatureFormat(format!(
                "Expected 64 bytes, got {}",
                signature_bytes.len()
            )));
        }

        let signature =
            Signature::from_bytes(signature_bytes.as_slice().try_into().map_err(|_| {
                VerifyError::SignatureFormat("Invalid signature length".to_string())
            })?);

        // 4. Try each public key
        for (key_id, public_key_bytes) in SIGNING_PUBLIC_KEYS {
            match Self::verify_with_key(&file_contents, &signature, public_key_bytes) {
                Ok(()) => {
                    tracing::info!("Signature verified with key: {}", key_id);
                    return Ok(());
                }
                Err(e) => {
                    tracing::debug!("Key {} failed: {}", key_id, e);
                }
            }
        }

        Err(VerifyError::NoValidSignature(SIGNING_PUBLIC_KEYS.len()))
    }

    /// Compute SHA256 checksum of a file
    pub fn compute_sha256(path: &Path) -> Result<String, VerifyError> {
        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();

        let mut buffer = [0u8; 8192];
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let hash = hasher.finalize();
        Ok(hex::encode(hash))
    }

    /// Verify signature with a specific public key
    fn verify_with_key(
        data: &[u8],
        signature: &Signature,
        public_key_bytes: &[u8; 32],
    ) -> Result<(), VerifyError> {
        let verifying_key = VerifyingKey::from_bytes(public_key_bytes)
            .map_err(|e| VerifyError::PublicKeyInvalid(e.to_string()))?;

        use ed25519_dalek::Verifier;
        verifying_key
            .verify(data, signature)
            .map_err(|_| VerifyError::SignatureInvalid)
    }

    /// Verify just the checksum (for WASM modules, etc.)
    pub fn verify_checksum(path: &Path, expected_checksum: &str) -> Result<(), VerifyError> {
        let actual_checksum = Self::compute_sha256(path)?;
        if actual_checksum != expected_checksum.to_lowercase() {
            return Err(VerifyError::ChecksumMismatch {
                expected: expected_checksum.to_string(),
                actual: actual_checksum,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_sha256() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"hello world").unwrap();

        let checksum = BinaryVerifier::compute_sha256(file.path()).unwrap();

        // SHA256 of "hello world"
        assert_eq!(
            checksum,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_sign_and_verify_roundtrip() {
        use ed25519_dalek::{SigningKey, Signer};

        // Generate a test keypair
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let public_key_bytes: [u8; 32] = verifying_key.to_bytes();

        // Create a test binary
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"fake binary content for signing test")
            .unwrap();

        // Compute checksum
        let checksum = BinaryVerifier::compute_sha256(file.path()).unwrap();

        // Sign the binary content
        let file_contents = std::fs::read(file.path()).unwrap();
        let signature = signing_key.sign(&file_contents);
        let _signature_hex = hex::encode(signature.to_bytes());

        // Verify with the test key using verify_with_key directly
        let sig = Signature::from_bytes(&signature.to_bytes());
        assert!(BinaryVerifier::verify_with_key(&file_contents, &sig, &public_key_bytes).is_ok());

        // Verify checksum matches
        assert!(BinaryVerifier::verify_checksum(file.path(), &checksum).is_ok());

        // Verify wrong data fails
        let wrong_data = b"tampered content";
        assert!(BinaryVerifier::verify_with_key(wrong_data, &sig, &public_key_bytes).is_err());
    }

    #[test]
    fn test_verify_checksum() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();

        let expected = "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9";
        assert!(BinaryVerifier::verify_checksum(file.path(), expected).is_ok());

        // Wrong checksum should fail
        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(BinaryVerifier::verify_checksum(file.path(), wrong).is_err());
    }
}
