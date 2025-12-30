//! WASM workload executor using wasmtime
//!
//! Provides sandboxed execution of WebAssembly modules with WASI support.
//! Workloads receive JSON input via stdin and produce JSON output via stdout.

use anyhow::{Context, Result};
use std::path::Path;
use tokio::sync::mpsc;
use tracing::{debug, info};
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
}

/// Configuration for WASM execution
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Path to the WASM module file
    pub module_path: String,
    /// JSON input to pass via stdin
    pub input: String,
    /// Maximum memory in bytes (default: 256MB)
    pub max_memory_bytes: u64,
    /// Maximum fuel (instructions) - None for unlimited
    pub max_fuel: Option<u64>,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            module_path: String::new(),
            input: String::new(),
            max_memory_bytes: 256 * 1024 * 1024, // 256MB
            max_fuel: Some(100_000_000_000),     // 100 billion instructions
        }
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

    /// Run a WASM module with the given configuration
    pub async fn run(
        &self,
        config: WasmConfig,
        output_tx: mpsc::Sender<WasmOutput>,
    ) -> Result<i32> {
        let engine = self.engine.clone();
        let module_path = config.module_path.clone();
        let input = config.input.clone();
        let max_fuel = config.max_fuel;

        // Run in blocking task since wasmtime operations are sync
        let result = tokio::task::spawn_blocking(move || {
            Self::run_sync(&engine, &module_path, &input, max_fuel)
        })
        .await
        .context("WASM execution task failed")??;

        // Send output
        if !result.stdout.is_empty() {
            let _ = output_tx.send(WasmOutput::Stdout(result.stdout)).await;
        }
        if !result.stderr.is_empty() {
            let _ = output_tx.send(WasmOutput::Stderr(result.stderr)).await;
        }
        let _ = output_tx.send(WasmOutput::Exit(result.exit_code)).await;

        Ok(result.exit_code)
    }

    /// Synchronous WASM execution for core modules (not components)
    fn run_sync(
        engine: &Engine,
        module_path: &str,
        input: &str,
        max_fuel: Option<u64>,
    ) -> Result<ExecutionResult> {
        debug!("Loading WASM module: {}", module_path);

        // Load the module
        let module = Module::from_file(engine, module_path)
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
