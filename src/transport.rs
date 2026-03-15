//! Activation transport abstraction for pipeline parallelism.
//!
//! Provides a unified interface for sending activations between pipeline
//! positions, with pluggable backends:
//! - **NATS** (default): Uses existing NATS pub/sub. Simple, reliable, ~1-2ms overhead.
//! - **Direct TCP** (future): Point-to-point TCP between Islands for lower latency.
//!
//! The coordinator provides peer connection info in the pipeline_config.
//! Each position uses a `TransportSender` to publish and a `TransportReceiver`
//! to consume activations.

use anyhow::{Context, Result};
use async_nats::Subscriber;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::nats::NatsAgent;

/// Transport mode for activation delivery
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportMode {
    /// NATS pub/sub (default, always available)
    Nats,
    /// Direct TCP connection between Islands (future)
    DirectTcp,
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
pub struct TransportSender {
    mode: TransportMode,
    nats_subject: String,
}

impl TransportSender {
    /// Create a sender for the given subject and transport mode
    pub fn new(nats_subject: String, peer: Option<&PeerInfo>) -> Self {
        let mode = peer
            .map(|p| p.mode.clone())
            .unwrap_or(TransportMode::Nats);

        // For now, always use NATS — direct TCP is scaffolded but not connected
        let effective_mode = match mode {
            TransportMode::DirectTcp => {
                tracing::info!("Direct TCP transport requested but not yet implemented, falling back to NATS");
                TransportMode::Nats
            }
            other => other,
        };

        Self {
            mode: effective_mode,
            nats_subject,
        }
    }

    /// Send activation data to the next position
    pub async fn send(&self, nats: &NatsAgent, data: Vec<u8>) -> Result<()> {
        match self.mode {
            TransportMode::Nats => {
                nats.publish_raw(&self.nats_subject, data).await
            }
            TransportMode::DirectTcp => {
                // Future: direct TCP send
                anyhow::bail!("Direct TCP transport not yet implemented")
            }
        }
    }

    /// Get the effective transport mode
    pub fn mode(&self) -> &TransportMode {
        &self.mode
    }
}

/// Receives activations from the previous position in the pipeline
pub struct TransportReceiver {
    mode: TransportMode,
    nats_sub: Subscriber,
}

impl TransportReceiver {
    /// Create a receiver by subscribing to the activation subject
    pub async fn new(nats: &NatsAgent, nats_subject: &str, peer: Option<&PeerInfo>) -> Result<Self> {
        let mode = peer
            .map(|p| p.mode.clone())
            .unwrap_or(TransportMode::Nats);

        let effective_mode = match mode {
            TransportMode::DirectTcp => {
                tracing::info!("Direct TCP transport requested but not yet implemented, falling back to NATS");
                TransportMode::Nats
            }
            other => other,
        };

        let nats_sub = nats
            .subscribe_ring(nats_subject)
            .await
            .context("Failed to subscribe to activation subject")?;

        Ok(Self {
            mode: effective_mode,
            nats_sub,
        })
    }

    /// Receive the next activation message. Returns None if the stream is closed.
    pub async fn recv(&mut self) -> Option<Vec<u8>> {
        match self.mode {
            TransportMode::Nats => {
                self.nats_sub.next().await.map(|msg| msg.payload.to_vec())
            }
            TransportMode::DirectTcp => {
                // Future: direct TCP recv
                None
            }
        }
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
        let json = r#""nats""#;
        let mode: TransportMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, TransportMode::Nats);

        let json = r#""direct_tcp""#;
        let mode: TransportMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, TransportMode::DirectTcp);
    }

    #[test]
    fn test_sender_falls_back_to_nats() {
        let peer = PeerInfo {
            host_id: "test".into(),
            address: Some("1.2.3.4:9000".into()),
            mode: TransportMode::DirectTcp,
        };
        let sender = TransportSender::new("ring.test.activate.1".into(), Some(&peer));
        // Should fall back to NATS since DirectTcp is not implemented
        assert_eq!(*sender.mode(), TransportMode::Nats);
    }

    #[test]
    fn test_sender_uses_nats_by_default() {
        let sender = TransportSender::new("ring.test.activate.1".into(), None);
        assert_eq!(*sender.mode(), TransportMode::Nats);
    }

    #[test]
    fn test_peer_info_deserialize() {
        let json = r#"{
            "host_id": "abc-123",
            "address": "192.168.1.100:9000",
            "mode": "direct_tcp"
        }"#;
        let peer: PeerInfo = serde_json::from_str(json).unwrap();
        assert_eq!(peer.host_id, "abc-123");
        assert_eq!(peer.address, Some("192.168.1.100:9000".into()));
        assert_eq!(peer.mode, TransportMode::DirectTcp);
    }
}
