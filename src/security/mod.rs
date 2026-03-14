//! Security hardening modules for the Island software.
//!
//! This module provides:
//! - TLS pinning for coordinator/NATS connections
//! - Registry allowlists for container images
//! - Seccomp profiles for container sandboxing
//! - Container signature verification (cosign)

pub mod registry;
pub mod seccomp;
pub mod signing;
pub mod tls;

pub use signing::SigningConfig;
