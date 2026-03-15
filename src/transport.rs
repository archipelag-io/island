//! Activation transport abstraction for pipeline parallelism.
//!
//! Provides a unified interface for sending activations between pipeline
//! positions, with pluggable backends:
//! - **NATS** (default): Uses existing NATS pub/sub. Simple, reliable, ~1-2ms overhead.
//! - **QUIC** (pipeline feature): Direct QUIC connection between Islands for ~0.1-0.5ms latency.
//!
//! The coordinator provides peer connection info in the pipeline_config.
//! Each position uses a `TransportSender` to publish and a `TransportReceiver`
//! to consume activations.

use anyhow::{Context, Result};
use async_nats::Subscriber;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::nats::NatsAgent;

/// Transport mode for activation delivery
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportMode {
    /// NATS pub/sub (default, always available)
    Nats,
    /// Direct QUIC connection between Islands
    Quic,
}

impl Default for TransportMode {
    fn default() -> Self {
        TransportMode::Nats
    }
}

/// Peer connection info provided by the coordinator for direct transport
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PeerInfo {
    /// Host ID of the peer Island
    pub host_id: String,
    /// Public address for direct connection (ip:port)
    #[serde(default)]
    pub address: Option<String>,
    /// Transport mode to use for this peer
    #[serde(default)]
    pub mode: TransportMode,
}

/// Sends activations to the next position in the pipeline
pub enum TransportSender {
    Nats { subject: String },
    Quic { tx: mpsc::Sender<Vec<u8>> },
}

impl TransportSender {
    /// Create a NATS-based sender (default)
    pub fn nats(subject: String) -> Self {
        TransportSender::Nats { subject }
    }

    /// Create a sender, choosing transport based on peer info
    pub async fn new(nats_subject: String, peer: Option<&PeerInfo>) -> Result<Self> {
        let mode = peer.map(|p| p.mode.clone()).unwrap_or_default();

        match mode {
            TransportMode::Quic => {
                let addr = peer
                    .and_then(|p| p.address.as_deref())
                    .context("QUIC transport requires peer address")?;

                match create_quic_sender(addr).await {
                    Ok(tx) => {
                        info!("QUIC transport connected to {}", addr);
                        Ok(TransportSender::Quic { tx })
                    }
                    Err(e) => {
                        warn!("QUIC transport to {} failed: {}, falling back to NATS", addr, e);
                        Ok(TransportSender::Nats { subject: nats_subject })
                    }
                }
            }
            TransportMode::Nats => Ok(TransportSender::Nats { subject: nats_subject }),
        }
    }

    /// Send activation data to the next position
    pub async fn send(&self, nats: &NatsAgent, data: Vec<u8>) -> Result<()> {
        match self {
            TransportSender::Nats { subject } => {
                nats.publish_raw(subject, data).await
            }
            TransportSender::Quic { tx } => {
                tx.send(data).await.map_err(|_| anyhow::anyhow!("QUIC send channel closed"))
            }
        }
    }

    /// Get the effective transport mode
    pub fn mode(&self) -> TransportMode {
        match self {
            TransportSender::Nats { .. } => TransportMode::Nats,
            TransportSender::Quic { .. } => TransportMode::Quic,
        }
    }
}

/// Receives activations from the previous position in the pipeline
pub enum TransportReceiver {
    Nats { sub: Subscriber },
    Quic { rx: mpsc::Receiver<Vec<u8>> },
}

impl TransportReceiver {
    /// Create a NATS-based receiver
    pub async fn nats(nats: &NatsAgent, subject: &str) -> Result<Self> {
        let sub = nats.subscribe_ring(subject).await
            .context("Failed to subscribe to activation subject")?;
        Ok(TransportReceiver::Nats { sub })
    }

    /// Create a receiver, optionally starting a QUIC listener
    pub async fn new(nats: &NatsAgent, nats_subject: &str, listen_addr: Option<&str>) -> Result<Self> {
        match listen_addr {
            Some(addr) => {
                match create_quic_listener(addr).await {
                    Ok(rx) => {
                        info!("QUIC listener bound on {}", addr);
                        Ok(TransportReceiver::Quic { rx })
                    }
                    Err(e) => {
                        warn!("QUIC listener on {} failed: {}, falling back to NATS", addr, e);
                        Self::nats(nats, nats_subject).await
                    }
                }
            }
            None => Self::nats(nats, nats_subject).await,
        }
    }

    /// Receive the next activation message. Returns None if the stream is closed.
    pub async fn recv(&mut self) -> Option<Vec<u8>> {
        match self {
            TransportReceiver::Nats { sub } => {
                sub.next().await.map(|msg| msg.payload.to_vec())
            }
            TransportReceiver::Quic { rx } => {
                rx.recv().await
            }
        }
    }
}

// ============================================================================
// QUIC transport implementation
// ============================================================================

/// Generate a self-signed TLS certificate for QUIC connections.
/// Islands use ephemeral self-signed certs — mutual TLS is not required
/// because activation data is not secret (the model weights are public).
fn generate_self_signed_cert() -> Result<(rustls::pki_types::CertificateDer<'static>, rustls::pki_types::PrivateKeyDer<'static>)> {
    let cert = rcgen::generate_simple_self_signed(vec!["island.local".to_string()])
        .context("Failed to generate self-signed cert")?;
    let cert_der = rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der())
    );
    Ok((cert_der, key_der))
}

/// Create a QUIC sender that connects to a peer Island
async fn create_quic_sender(addr: &str) -> Result<mpsc::Sender<Vec<u8>>> {
    let peer_addr: SocketAddr = addr.parse()
        .with_context(|| format!("Invalid peer address: {}", addr))?;

    // Client config: skip server cert verification (self-signed peer certs)
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .context("Failed to create QUIC client config")?
    ));

    let mut endpoint = quinn::Endpoint::client("0.0.0.0:0".parse()?)?;
    endpoint.set_default_client_config(client_config);

    let connection = endpoint
        .connect(peer_addr, "island.local")?
        .await
        .context("QUIC connection failed")?;

    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(256);

    // Spawn a task that reads from the channel and sends via QUIC
    tokio::spawn(async move {
        while let Some(data) = rx.recv().await {
            match connection.open_uni().await {
                Ok(mut stream) => {
                    let len = (data.len() as u32).to_le_bytes();
                    if stream.write_all(&len).await.is_err() { break; }
                    if stream.write_all(&data).await.is_err() { break; }
                    let _ = stream.finish();
                }
                Err(e) => {
                    warn!("QUIC stream open failed: {}", e);
                    break;
                }
            }
        }
    });

    Ok(tx)
}

/// Create a QUIC listener that accepts activation data from peer Islands
async fn create_quic_listener(addr: &str) -> Result<mpsc::Receiver<Vec<u8>>> {
    let listen_addr: SocketAddr = addr.parse()
        .with_context(|| format!("Invalid listen address: {}", addr))?;

    let (cert_der, key_der) = generate_self_signed_cert()?;

    let server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .context("Failed to create TLS server config")?;

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .context("Failed to create QUIC server config")?
    ));

    let endpoint = quinn::Endpoint::server(server_config, listen_addr)?;

    let (tx, rx) = mpsc::channel::<Vec<u8>>(256);

    // Spawn acceptor task
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                if let Ok(connection) = incoming.await {
                    while let Ok(mut stream) = connection.accept_uni().await {
                        // Read length-prefixed message
                        let mut len_buf = [0u8; 4];
                        if stream.read_exact(&mut len_buf).await.is_err() { continue; }
                        let len = u32::from_le_bytes(len_buf) as usize;
                        if len > 64 * 1024 * 1024 { continue; } // 64MB max

                        let mut data = vec![0u8; len];
                        if stream.read_exact(&mut data).await.is_err() { continue; }

                        if tx.send(data).await.is_err() { break; }
                    }
                }
            });
        }
    });

    Ok(rx)
}

/// TLS certificate verifier that accepts any certificate.
/// Used for Island-to-Island QUIC where both endpoints are ephemeral.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_mode_default() {
        assert_eq!(TransportMode::default(), TransportMode::Nats);
    }

    #[test]
    fn test_transport_mode_deserialize() {
        let mode: TransportMode = serde_json::from_str(r#""nats""#).unwrap();
        assert_eq!(mode, TransportMode::Nats);

        let mode: TransportMode = serde_json::from_str(r#""quic""#).unwrap();
        assert_eq!(mode, TransportMode::Quic);
    }

    #[test]
    fn test_peer_info_deserialize() {
        let json = r#"{
            "host_id": "abc-123",
            "address": "192.168.1.100:9000",
            "mode": "quic"
        }"#;
        let peer: PeerInfo = serde_json::from_str(json).unwrap();
        assert_eq!(peer.host_id, "abc-123");
        assert_eq!(peer.address, Some("192.168.1.100:9000".into()));
        assert_eq!(peer.mode, TransportMode::Quic);
    }

    #[test]
    fn test_nats_sender_mode() {
        let sender = TransportSender::nats("ring.test.activate.1".into());
        assert_eq!(sender.mode(), TransportMode::Nats);
    }

    #[test]
    fn test_self_signed_cert_generation() {
        let result = generate_self_signed_cert();
        assert!(result.is_ok());
    }
}
