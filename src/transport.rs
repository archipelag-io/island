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
    /// NATS pub/sub (default, always available).
    /// Acts as the TURN-equivalent relay for symmetric NAT environments.
    Nats,
    /// Direct QUIC connection between Islands
    Quic,
    /// QUIC with NATS relay fallback — tries QUIC first, falls back to NATS
    /// if direct connection fails (NAT, firewall, symmetric NAT).
    /// This is the recommended mode for production pipelines.
    /// The NATS fallback serves as a built-in TURN relay without requiring
    /// external relay infrastructure.
    QuicWithRelay,
}

/// Transport statistics for monitoring relay usage
#[derive(Debug, Clone, Default, Serialize)]
pub struct TransportStats {
    /// Number of messages sent via NATS (relay)
    pub nats_messages: u64,
    /// Number of messages sent via QUIC (direct)
    pub quic_messages: u64,
    /// Whether QUIC connection was attempted and failed (relay mode active)
    pub relay_active: bool,
    /// Reason for relay fallback (if any)
    pub relay_reason: Option<String>,
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
            TransportMode::Quic | TransportMode::QuicWithRelay => {
                let addr = peer
                    .and_then(|p| p.address.as_deref())
                    .context("QUIC transport requires peer address")?;

                match create_quic_sender(addr).await {
                    Ok(tx) => {
                        info!("QUIC transport connected to {}", addr);
                        Ok(TransportSender::Quic { tx })
                    }
                    Err(e) => {
                        if mode == TransportMode::QuicWithRelay {
                            info!("QUIC to {} failed ({}), using NATS relay", addr, e);
                        } else {
                            warn!(
                                "QUIC transport to {} failed: {}, falling back to NATS",
                                addr, e
                            );
                        }
                        Ok(TransportSender::Nats {
                            subject: nats_subject,
                        })
                    }
                }
            }
            TransportMode::Nats => Ok(TransportSender::Nats {
                subject: nats_subject,
            }),
        }
    }

    /// Send activation data to the next position
    pub async fn send(&self, nats: &NatsAgent, data: Vec<u8>) -> Result<()> {
        match self {
            TransportSender::Nats { subject } => nats.publish_raw(subject, data).await,
            TransportSender::Quic { tx } => tx
                .send(data)
                .await
                .map_err(|_| anyhow::anyhow!("QUIC send channel closed")),
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
        let sub = nats
            .subscribe_ring(subject)
            .await
            .context("Failed to subscribe to activation subject")?;
        Ok(TransportReceiver::Nats { sub })
    }

    /// Create a receiver, optionally starting a QUIC listener.
    /// Falls back to NATS if QUIC listener fails (relay mode).
    pub async fn new(
        nats: &NatsAgent,
        nats_subject: &str,
        listen_addr: Option<&str>,
    ) -> Result<Self> {
        match listen_addr {
            Some(addr) => match create_quic_listener(addr).await {
                Ok(rx) => {
                    info!("QUIC listener bound on {}", addr);
                    Ok(TransportReceiver::Quic { rx })
                }
                Err(e) => {
                    info!("QUIC listener on {} failed ({}), using NATS relay", addr, e);
                    Self::nats(nats, nats_subject).await
                }
            },
            None => Self::nats(nats, nats_subject).await,
        }
    }

    /// Receive the next activation message. Returns None if the stream is closed.
    pub async fn recv(&mut self) -> Option<Vec<u8>> {
        match self {
            TransportReceiver::Nats { sub } => sub.next().await.map(|msg| msg.payload.to_vec()),
            TransportReceiver::Quic { rx } => rx.recv().await,
        }
    }
}

// ============================================================================
// QUIC transport implementation
// ============================================================================

/// Generate a self-signed TLS certificate for QUIC connections.
/// Islands use ephemeral self-signed certs — mutual TLS is not required
/// because activation data is not secret (the model weights are public).
fn generate_self_signed_cert() -> Result<(
    rustls::pki_types::CertificateDer<'static>,
    rustls::pki_types::PrivateKeyDer<'static>,
)> {
    let cert = rcgen::generate_simple_self_signed(vec!["island.local".to_string()])
        .context("Failed to generate self-signed cert")?;
    let cert_der = rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()),
    );
    Ok((cert_der, key_der))
}

/// Create a QUIC sender that connects to a peer Island
async fn create_quic_sender(addr: &str) -> Result<mpsc::Sender<Vec<u8>>> {
    let peer_addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("Invalid peer address: {}", addr))?;

    // Client config: skip server cert verification (self-signed peer certs)
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    let client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .context("Failed to create QUIC client config")?,
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
                    if stream.write_all(&len).await.is_err() {
                        break;
                    }
                    if stream.write_all(&data).await.is_err() {
                        break;
                    }
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
    let listen_addr: SocketAddr = addr
        .parse()
        .with_context(|| format!("Invalid listen address: {}", addr))?;

    let (cert_der, key_der) = generate_self_signed_cert()?;

    let server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .context("Failed to create TLS server config")?;

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .context("Failed to create QUIC server config")?,
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
                        if stream.read_exact(&mut len_buf).await.is_err() {
                            continue;
                        }
                        let len = u32::from_le_bytes(len_buf) as usize;
                        if len > 64 * 1024 * 1024 {
                            continue;
                        } // 64MB max

                        let mut data = vec![0u8; len];
                        if stream.read_exact(&mut data).await.is_err() {
                            continue;
                        }

                        if tx.send(data).await.is_err() {
                            break;
                        }
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

    // ── TransportMode ──────────────────────────────────────────────────

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
    fn test_transport_mode_deserialize_quic_with_relay() {
        let mode: TransportMode = serde_json::from_str(r#""quic_with_relay""#).unwrap();
        assert_eq!(mode, TransportMode::QuicWithRelay);
    }

    #[test]
    fn test_transport_mode_serialize_roundtrip() {
        for mode in [
            TransportMode::Nats,
            TransportMode::Quic,
            TransportMode::QuicWithRelay,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let restored: TransportMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, restored);
        }
    }

    #[test]
    fn test_transport_mode_serialize_snake_case() {
        let json = serde_json::to_string(&TransportMode::QuicWithRelay).unwrap();
        assert_eq!(json, r#""quic_with_relay""#);

        let json = serde_json::to_string(&TransportMode::Nats).unwrap();
        assert_eq!(json, r#""nats""#);

        let json = serde_json::to_string(&TransportMode::Quic).unwrap();
        assert_eq!(json, r#""quic""#);
    }

    #[test]
    fn test_transport_mode_invalid_value() {
        let result: Result<TransportMode, _> = serde_json::from_str(r#""tcp""#);
        assert!(result.is_err());
    }

    #[test]
    fn test_transport_mode_clone() {
        let mode = TransportMode::QuicWithRelay;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    // ── PeerInfo ───────────────────────────────────────────────────────

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
    fn test_peer_info_defaults_when_optional_fields_missing() {
        let json = r#"{"host_id": "host-1"}"#;
        let peer: PeerInfo = serde_json::from_str(json).unwrap();
        assert_eq!(peer.host_id, "host-1");
        assert_eq!(peer.address, None);
        assert_eq!(peer.mode, TransportMode::Nats); // default
    }

    #[test]
    fn test_peer_info_with_ipv6_address() {
        let json = r#"{
            "host_id": "ipv6-host",
            "address": "[::1]:9000",
            "mode": "quic"
        }"#;
        let peer: PeerInfo = serde_json::from_str(json).unwrap();
        assert_eq!(peer.address, Some("[::1]:9000".into()));
    }

    #[test]
    fn test_peer_info_roundtrip() {
        let original = PeerInfo {
            host_id: "test-island-42".into(),
            address: Some("10.0.0.1:8443".into()),
            mode: TransportMode::QuicWithRelay,
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: PeerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.host_id, "test-island-42");
        assert_eq!(restored.address, Some("10.0.0.1:8443".into()));
        assert_eq!(restored.mode, TransportMode::QuicWithRelay);
    }

    #[test]
    fn test_peer_info_missing_host_id_fails() {
        let json = r#"{"address": "1.2.3.4:9000"}"#;
        let result: Result<PeerInfo, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── TransportSender ────────────────────────────────────────────────

    #[test]
    fn test_nats_sender_mode() {
        let sender = TransportSender::nats("ring.test.activate.1".into());
        assert_eq!(sender.mode(), TransportMode::Nats);
    }

    #[test]
    fn test_quic_sender_mode() {
        let (tx, _rx) = mpsc::channel(1);
        let sender = TransportSender::Quic { tx };
        assert_eq!(sender.mode(), TransportMode::Quic);
    }

    #[test]
    fn test_nats_sender_factory() {
        let subject = "ring.pipeline.activate.3".to_string();
        let sender = TransportSender::nats(subject.clone());
        match &sender {
            TransportSender::Nats { subject: s } => assert_eq!(s, &subject),
            _ => panic!("Expected Nats variant"),
        }
    }

    // ── TransportStats ─────────────────────────────────────────────────

    #[test]
    fn test_transport_stats_default() {
        let stats = TransportStats::default();
        assert_eq!(stats.nats_messages, 0);
        assert_eq!(stats.quic_messages, 0);
        assert!(!stats.relay_active);
        assert!(stats.relay_reason.is_none());
    }

    #[test]
    fn test_transport_stats_serialize() {
        let stats = TransportStats {
            nats_messages: 100,
            quic_messages: 50,
            relay_active: true,
            relay_reason: Some("symmetric NAT detected".into()),
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"nats_messages\":100"));
        assert!(json.contains("\"quic_messages\":50"));
        assert!(json.contains("\"relay_active\":true"));
        assert!(json.contains("symmetric NAT detected"));
    }

    #[test]
    fn test_transport_stats_clone() {
        let stats = TransportStats {
            nats_messages: 10,
            quic_messages: 20,
            relay_active: false,
            relay_reason: None,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.nats_messages, 10);
        assert_eq!(cloned.quic_messages, 20);
    }

    // ── Certificate generation ─────────────────────────────────────────

    #[test]
    fn test_self_signed_cert_generation() {
        let result = generate_self_signed_cert();
        assert!(result.is_ok());
    }

    #[test]
    fn test_self_signed_cert_produces_non_empty_data() {
        let (cert_der, key_der) = generate_self_signed_cert().unwrap();
        assert!(!cert_der.is_empty(), "Certificate DER should not be empty");
        match &key_der {
            rustls::pki_types::PrivateKeyDer::Pkcs8(pkcs8) => {
                assert!(
                    !pkcs8.secret_pkcs8_der().is_empty(),
                    "Private key should not be empty"
                );
            }
            _ => panic!("Expected PKCS8 key format"),
        }
    }

    #[test]
    fn test_self_signed_cert_generates_unique_keys() {
        let (_, key1) = generate_self_signed_cert().unwrap();
        let (_, key2) = generate_self_signed_cert().unwrap();
        // Two calls should produce different keys (different random material)
        let k1_bytes = match &key1 {
            rustls::pki_types::PrivateKeyDer::Pkcs8(k) => k.secret_pkcs8_der().to_vec(),
            _ => panic!("Expected PKCS8"),
        };
        let k2_bytes = match &key2 {
            rustls::pki_types::PrivateKeyDer::Pkcs8(k) => k.secret_pkcs8_der().to_vec(),
            _ => panic!("Expected PKCS8"),
        };
        assert_ne!(
            k1_bytes, k2_bytes,
            "Each cert generation should produce unique keys"
        );
    }

    // ── Address parsing (used internally by create_quic_sender/listener) ──

    #[test]
    fn test_socket_addr_parsing_ipv4() {
        let addr: Result<SocketAddr, _> = "192.168.1.100:9000".parse();
        assert!(addr.is_ok());
        let addr = addr.unwrap();
        assert_eq!(addr.port(), 9000);
    }

    #[test]
    fn test_socket_addr_parsing_ipv6() {
        let addr: Result<SocketAddr, _> = "[::1]:4433".parse();
        assert!(addr.is_ok());
        let addr = addr.unwrap();
        assert_eq!(addr.port(), 4433);
        assert!(addr.is_ipv6());
    }

    #[test]
    fn test_socket_addr_parsing_invalid() {
        let addr: Result<SocketAddr, _> = "not-an-address".parse();
        assert!(addr.is_err());
    }

    #[test]
    fn test_socket_addr_parsing_missing_port() {
        let addr: Result<SocketAddr, _> = "192.168.1.1".parse();
        assert!(addr.is_err());
    }

    // ── SkipServerVerification ─────────────────────────────────────────

    #[test]
    fn test_skip_server_verification_supported_schemes() {
        use rustls::client::danger::ServerCertVerifier;
        let verifier = SkipServerVerification;
        let schemes = verifier.supported_verify_schemes();
        assert!(!schemes.is_empty());
        assert!(schemes.contains(&rustls::SignatureScheme::ECDSA_NISTP256_SHA256));
        assert!(schemes.contains(&rustls::SignatureScheme::ED25519));
        assert!(schemes.contains(&rustls::SignatureScheme::RSA_PSS_SHA256));
        assert_eq!(schemes.len(), 6);
    }
}
