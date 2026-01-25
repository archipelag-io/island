//! WASM workload executor using wasmtime
//!
//! Provides sandboxed execution of WebAssembly modules with WASI support.
//! Workloads receive JSON input via stdin and produce JSON output via stdout.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};
use wasmtime::*;
use wasmtime_wasi::pipe::{MemoryInputPipe, MemoryOutputPipe};
use wasmtime_wasi::preview1::WasiP1Ctx;
use wasmtime_wasi::WasiCtxBuilder;

/// Output from WASM execution
#[derive(Debug)]
pub enum WasmOutput {
    Stdout(String),
    Stderr(String),
    Exit(i32),
    /// WASM execution timed out or exceeded fuel limit
    Timeout,
}

/// Configuration for WASM execution
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Path to the WASM module file
    pub module_path: String,
    /// JSON input to pass via stdin
    pub input: String,
    /// Maximum memory in bytes (default: 256MB) - reserved for future use
    #[allow(dead_code)]
    pub max_memory_bytes: u64,
    /// Maximum fuel (instructions) - None for unlimited
    pub max_fuel: Option<u64>,
    /// Timeout in seconds (default: 60 seconds for WASM)
    pub timeout_seconds: u64,
    /// Expected SHA256 hash of the WASM module (hex string)
    /// If provided, the module will be verified before execution
    pub expected_hash: Option<String>,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            module_path: String::new(),
            input: String::new(),
            max_memory_bytes: 256 * 1024 * 1024, // 256MB
            max_fuel: Some(100_000_000_000),     // 100 billion instructions
            timeout_seconds: 60,                  // 1 minute timeout for WASM
            expected_hash: None,
        }
    }
}

/// Verify that a WASM module's hash matches the expected hash.
///
/// The hash should be in the format "sha256:<hash>" or just "<hash>" (64 hex chars).
/// Returns Ok(()) if verification passes, Err if it fails.
pub fn verify_wasm_hash(module_bytes: &[u8], expected_hash: &str) -> Result<()> {
    // Normalize expected hash (remove sha256: prefix if present)
    let expected = if expected_hash.starts_with("sha256:") {
        &expected_hash[7..]
    } else {
        expected_hash
    };

    // Compute SHA256 hash of the module
    let mut hasher = Sha256::new();
    hasher.update(module_bytes);
    let actual_hash = hex::encode(hasher.finalize());

    if actual_hash == expected.to_lowercase() {
        info!(
            "WASM module hash verified: {}...{}",
            &actual_hash[..8],
            &actual_hash[actual_hash.len() - 8..]
        );
        Ok(())
    } else {
        warn!(
            "WASM hash mismatch! Expected: {}..., Got: {}...",
            &expected[..expected.len().min(16)],
            &actual_hash[..16]
        );
        anyhow::bail!(
            "WASM module hash verification failed: expected {}, got {}",
            expected,
            actual_hash
        )
    }
}

/// WASI context state for wasmtime preview1 modules
struct WasiState {
    wasi: WasiP1Ctx,
}

/// WASM executor that runs modules in a sandboxed environment
pub struct WasmExecutor {
    engine: Engine,
}

impl WasmExecutor {
    /// Create a new WASM executor
    pub fn new() -> Result<Self> {
        let mut config = Config::new();

        // Enable fuel consumption for resource limiting
        config.consume_fuel(true);

        // Async support for non-blocking execution
        config.async_support(false); // Use sync for simpler implementation

        // Memory limits
        config.max_wasm_stack(512 * 1024); // 512KB stack

        let engine = Engine::new(&config).context("Failed to create WASM engine")?;

        info!("WASM executor initialized");

        Ok(Self { engine })
    }

    /// Run a WASM module with the given configuration.
    ///
    /// The execution will be terminated if it exceeds the configured timeout.
    pub async fn run(
        &self,
        config: WasmConfig,
        output_tx: mpsc::Sender<WasmOutput>,
    ) -> Result<i32> {
        let engine = self.engine.clone();
        let module_path = config.module_path.clone();
        let input = config.input.clone();
        let max_fuel = config.max_fuel;
        let expected_hash = config.expected_hash.clone();
        let timeout_duration = Duration::from_secs(config.timeout_seconds);

        // Run in blocking task since wasmtime operations are sync, with timeout
        let run_result = timeout(timeout_duration, async {
            tokio::task::spawn_blocking(move || {
                Self::run_sync(&engine, &module_path, &input, max_fuel, expected_hash.as_deref())
            })
            .await
            .context("WASM execution task failed")?
        })
        .await;

        match run_result {
            Ok(Ok(result)) => {
                // Normal completion
                if !result.stdout.is_empty() {
                    let _ = output_tx.send(WasmOutput::Stdout(result.stdout)).await;
                }
                if !result.stderr.is_empty() {
                    let _ = output_tx.send(WasmOutput::Stderr(result.stderr)).await;
                }
                let _ = output_tx.send(WasmOutput::Exit(result.exit_code)).await;
                Ok(result.exit_code)
            }
            Ok(Err(e)) => {
                // Execution error
                Err(e)
            }
            Err(_) => {
                // Timeout
                warn!(
                    "WASM execution exceeded timeout ({}s)",
                    config.timeout_seconds
                );
                let _ = output_tx.send(WasmOutput::Timeout).await;
                Ok(-1)
            }
        }
    }

    /// Synchronous WASM execution for core modules (not components)
    fn run_sync(
        engine: &Engine,
        module_path: &str,
        input: &str,
        max_fuel: Option<u64>,
        expected_hash: Option<&str>,
    ) -> Result<ExecutionResult> {
        debug!("Loading WASM module: {}", module_path);

        // Read module bytes
        let module_bytes = std::fs::read(module_path)
            .with_context(|| format!("Failed to read WASM module: {}", module_path))?;

        // Verify hash if expected
        if let Some(hash) = expected_hash {
            verify_wasm_hash(&module_bytes, hash)
                .context("WASM hash verification failed - refusing to execute")?;
        } else {
            debug!("No expected hash provided, skipping verification for WASM module");
        }

        // Load the module from bytes
        let module = Module::new(engine, &module_bytes)
            .context("Failed to load WASM module")?;

        // Create pipes for stdin/stdout/stderr
        let stdin = MemoryInputPipe::new(input.as_bytes().to_vec());
        let stdout = MemoryOutputPipe::new(4096);
        let stderr = MemoryOutputPipe::new(4096);

        let stdout_clone = stdout.clone();
        let stderr_clone = stderr.clone();

        // Build WASI context
        let wasi_ctx = WasiCtxBuilder::new()
            .stdin(stdin)
            .stdout(stdout)
            .stderr(stderr)
            .build_p1();

        let state = WasiState { wasi: wasi_ctx };

        // Create store with state
        let mut store = Store::new(engine, state);

        // Set fuel limit if configured
        if let Some(fuel) = max_fuel {
            store.set_fuel(fuel).context("Failed to set fuel")?;
        }

        // Create a linker and add WASI preview1 functions
        let mut linker = Linker::new(engine);
        wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s: &mut WasiState| &mut s.wasi)
            .context("Failed to add WASI to linker")?;

        // Instantiate the module
        let instance = linker
            .instantiate(&mut store, &module)
            .context("Failed to instantiate WASM module")?;

        // Get the _start function (WASI entry point)
        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .context("Module missing _start function")?;

        // Run the module
        let exit_code = match start.call(&mut store, ()) {
            Ok(()) => 0,
            Err(e) => {
                // Check if it's a normal WASI exit
                if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
                    exit.0
                } else {
                    debug!("WASM execution error: {}", e);
                    1
                }
            }
        };

        // Drop the store to release references to the pipes
        drop(store);

        // Collect output - try_into_inner only succeeds if we have the last reference
        let stdout_bytes = stdout_clone.try_into_inner().unwrap_or_default();
        let stderr_bytes = stderr_clone.try_into_inner().unwrap_or_default();

        let stdout_str = String::from_utf8_lossy(&stdout_bytes).to_string();
        let stderr_str = String::from_utf8_lossy(&stderr_bytes).to_string();

        debug!("WASM execution complete, exit code: {}", exit_code);

        Ok(ExecutionResult {
            exit_code,
            stdout: stdout_str,
            stderr: stderr_str,
        })
    }

    /// Load and validate a WASM module without executing it
    pub fn validate_module(&self, path: &Path) -> Result<ModuleInfo> {
        let module = Module::from_file(&self.engine, path)
            .context("Failed to load WASM module")?;

        let exports: Vec<String> = module
            .exports()
            .map(|e| e.name().to_string())
            .collect();

        let imports: Vec<String> = module
            .imports()
            .map(|i| format!("{}::{}", i.module(), i.name()))
            .collect();

        let has_start = exports.iter().any(|e| e == "_start");

        Ok(ModuleInfo {
            exports,
            imports,
            has_start,
        })
    }
}

impl Default for WasmExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default WASM executor")
    }
}

/// Result of WASM execution
struct ExecutionResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

/// Information about a WASM module
#[derive(Debug)]
pub struct ModuleInfo {
    pub exports: Vec<String>,
    #[allow(dead_code)]
    pub imports: Vec<String>,
    pub has_start: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = WasmExecutor::new();
        assert!(executor.is_ok());
    }
}
