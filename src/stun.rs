//! Lightweight STUN client for NAT traversal (RFC 5389).
//!
//! Discovers the public IP and port of this Island by sending a STUN
//! Binding Request to a public STUN server and parsing the XOR-MAPPED-ADDRESS
//! response. Used to provide peer addresses for QUIC direct transport.
//!
//! Only implements the Binding Request — no TURN, no ICE, no authentication.
//! Sufficient for discovering whether the Island is behind NAT and what its
//! public endpoint is.

use anyhow::{Context, Result};
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use tracing::{debug, info};

/// Default public STUN servers
const DEFAULT_STUN_SERVERS: &[&str] = &[
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun.cloudflare.com:3478",
];

/// STUN message type: Binding Request
const BINDING_REQUEST: u16 = 0x0001;
/// STUN message type: Binding Response (success)
const BINDING_RESPONSE: u16 = 0x0101;
/// STUN magic cookie (RFC 5389)
const MAGIC_COOKIE: u32 = 0x2112A442;
/// XOR-MAPPED-ADDRESS attribute type
const XOR_MAPPED_ADDRESS: u16 = 0x0020;
/// MAPPED-ADDRESS attribute type (fallback for older servers)
const MAPPED_ADDRESS: u16 = 0x0001;

/// Result of a STUN binding request
#[derive(Debug, Clone)]
pub struct StunResult {
    /// Public IP:port as seen by the STUN server
    pub public_addr: SocketAddr,
    /// Which STUN server responded
    pub server: String,
}

/// Discover this Island's public IP and port via STUN.
///
/// Tries multiple STUN servers in order. Returns the first successful result.
/// Timeout: 3 seconds per server.
pub async fn discover_public_addr() -> Result<StunResult> {
    for server in DEFAULT_STUN_SERVERS {
        match probe_stun_server(server).await {
            Ok(result) => {
                info!("STUN discovery: public address is {} (via {})", result.public_addr, server);
                return Ok(result);
            }
            Err(e) => {
                debug!("STUN server {} failed: {}", server, e);
                continue;
            }
        }
    }
    anyhow::bail!("All STUN servers failed — unable to discover public address")
}

/// Send a STUN Binding Request and parse the response.
async fn probe_stun_server(server: &str) -> Result<StunResult> {
    let socket = UdpSocket::bind("0.0.0.0:0").await
        .context("Failed to bind UDP socket")?;

    // Resolve server address
    let server_addr: SocketAddr = tokio::net::lookup_host(server).await
        .context("STUN DNS lookup failed")?
        .next()
        .context("No addresses for STUN server")?;

    // Build STUN Binding Request (20 bytes)
    let transaction_id: [u8; 12] = rand::random();
    let request = build_binding_request(&transaction_id);

    socket.send_to(&request, server_addr).await
        .context("Failed to send STUN request")?;

    // Wait for response (3 second timeout)
    let mut buf = [0u8; 256];
    let n = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        socket.recv(&mut buf),
    )
    .await
    .context("STUN response timeout")?
    .context("Failed to receive STUN response")?;

    parse_binding_response(&buf[..n], &transaction_id, server)
}

/// Build a STUN Binding Request message (RFC 5389 §6)
fn build_binding_request(transaction_id: &[u8; 12]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(20);
    msg.extend_from_slice(&BINDING_REQUEST.to_be_bytes());
    msg.extend_from_slice(&0u16.to_be_bytes()); // message length (no attributes)
    msg.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    msg.extend_from_slice(transaction_id);
    msg
}

/// Parse a STUN Binding Response and extract the mapped address
fn parse_binding_response(data: &[u8], expected_txn: &[u8; 12], server: &str) -> Result<StunResult> {
    if data.len() < 20 {
        anyhow::bail!("STUN response too short: {} bytes", data.len());
    }

    let msg_type = u16::from_be_bytes([data[0], data[1]]);
    if msg_type != BINDING_RESPONSE {
        anyhow::bail!("Not a STUN Binding Response: 0x{:04x}", msg_type);
    }

    let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
    let cookie = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    if cookie != MAGIC_COOKIE {
        anyhow::bail!("Invalid STUN magic cookie");
    }

    // Verify transaction ID
    if &data[8..20] != expected_txn {
        anyhow::bail!("STUN transaction ID mismatch");
    }

    // Parse attributes
    let attrs = &data[20..20 + msg_len.min(data.len() - 20)];
    let mut offset = 0;

    while offset + 4 <= attrs.len() {
        let attr_type = u16::from_be_bytes([attrs[offset], attrs[offset + 1]]);
        let attr_len = u16::from_be_bytes([attrs[offset + 2], attrs[offset + 3]]) as usize;
        let attr_data = &attrs[offset + 4..offset + 4 + attr_len.min(attrs.len() - offset - 4)];

        match attr_type {
            XOR_MAPPED_ADDRESS => {
                let addr = parse_xor_mapped_address(attr_data, expected_txn)?;
                return Ok(StunResult { public_addr: addr, server: server.to_string() });
            }
            MAPPED_ADDRESS => {
                let addr = parse_mapped_address(attr_data)?;
                return Ok(StunResult { public_addr: addr, server: server.to_string() });
            }
            _ => {} // Skip unknown attributes
        }

        // Attributes are padded to 4-byte boundaries
        offset += 4 + ((attr_len + 3) & !3);
    }

    anyhow::bail!("No MAPPED-ADDRESS in STUN response")
}

/// Parse XOR-MAPPED-ADDRESS (RFC 5389 §15.2)
fn parse_xor_mapped_address(data: &[u8], txn_id: &[u8; 12]) -> Result<SocketAddr> {
    if data.len() < 8 {
        anyhow::bail!("XOR-MAPPED-ADDRESS too short");
    }

    let family = data[1];
    let xor_port = u16::from_be_bytes([data[2], data[3]]);
    let port = xor_port ^ (MAGIC_COOKIE >> 16) as u16;

    match family {
        0x01 => {
            // IPv4
            let xor_addr = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
            let addr = xor_addr ^ MAGIC_COOKIE;
            let ip = std::net::Ipv4Addr::from(addr);
            Ok(SocketAddr::new(ip.into(), port))
        }
        0x02 => {
            // IPv6
            if data.len() < 20 {
                anyhow::bail!("XOR-MAPPED-ADDRESS IPv6 too short");
            }
            let mut xor_bytes = [0u8; 16];
            xor_bytes.copy_from_slice(&data[4..20]);
            let mut key = [0u8; 16];
            key[0..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            key[4..16].copy_from_slice(txn_id);
            for i in 0..16 {
                xor_bytes[i] ^= key[i];
            }
            let ip = std::net::Ipv6Addr::from(xor_bytes);
            Ok(SocketAddr::new(ip.into(), port))
        }
        _ => anyhow::bail!("Unknown address family: {}", family),
    }
}

/// Parse MAPPED-ADDRESS (RFC 5389 §15.1) — fallback for non-XOR servers
fn parse_mapped_address(data: &[u8]) -> Result<SocketAddr> {
    if data.len() < 8 {
        anyhow::bail!("MAPPED-ADDRESS too short");
    }
    let family = data[1];
    let port = u16::from_be_bytes([data[2], data[3]]);

    match family {
        0x01 => {
            let ip = std::net::Ipv4Addr::new(data[4], data[5], data[6], data[7]);
            Ok(SocketAddr::new(ip.into(), port))
        }
        _ => anyhow::bail!("Unsupported address family: {}", family),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_binding_request() {
        let txn = [1u8; 12];
        let req = build_binding_request(&txn);
        assert_eq!(req.len(), 20);
        assert_eq!(&req[0..2], &BINDING_REQUEST.to_be_bytes());
        assert_eq!(&req[2..4], &[0, 0]); // length = 0
        assert_eq!(&req[4..8], &MAGIC_COOKIE.to_be_bytes());
        assert_eq!(&req[8..20], &txn);
    }

    #[test]
    fn test_parse_xor_mapped_address_ipv4() {
        let txn = [0u8; 12];
        // XOR-MAPPED-ADDRESS for 198.51.100.1:3478
        // IP XOR magic: 198.51.100.1 XOR 0x2112A442
        let ip = u32::from_be_bytes([198, 51, 100, 1]);
        let xor_ip = (ip ^ MAGIC_COOKIE).to_be_bytes();
        let xor_port = (3478u16 ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes();

        let data = [
            0x00, 0x01, // reserved + IPv4
            xor_port[0], xor_port[1],
            xor_ip[0], xor_ip[1], xor_ip[2], xor_ip[3],
        ];

        let addr = parse_xor_mapped_address(&data, &txn).unwrap();
        assert_eq!(addr.ip(), std::net::IpAddr::V4(std::net::Ipv4Addr::new(198, 51, 100, 1)));
        assert_eq!(addr.port(), 3478);
    }

    #[test]
    fn test_parse_binding_response() {
        let txn = [42u8; 12];

        // Build a minimal Binding Response with XOR-MAPPED-ADDRESS
        let ip = u32::from_be_bytes([10, 0, 0, 1]);
        let xor_ip = (ip ^ MAGIC_COOKIE).to_be_bytes();
        let xor_port = (9000u16 ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes();

        let attr_value = [
            0x00, 0x01, // IPv4
            xor_port[0], xor_port[1],
            xor_ip[0], xor_ip[1], xor_ip[2], xor_ip[3],
        ];

        let attr_type = XOR_MAPPED_ADDRESS.to_be_bytes();
        let attr_len = (attr_value.len() as u16).to_be_bytes();

        let mut msg = Vec::new();
        msg.extend_from_slice(&BINDING_RESPONSE.to_be_bytes());
        let body_len = (4 + attr_value.len()) as u16;
        msg.extend_from_slice(&body_len.to_be_bytes());
        msg.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
        msg.extend_from_slice(&txn);
        msg.extend_from_slice(&attr_type);
        msg.extend_from_slice(&attr_len);
        msg.extend_from_slice(&attr_value);

        let result = parse_binding_response(&msg, &txn, "test:3478").unwrap();
        assert_eq!(result.public_addr.port(), 9000);
        assert_eq!(result.server, "test:3478");
    }
}
