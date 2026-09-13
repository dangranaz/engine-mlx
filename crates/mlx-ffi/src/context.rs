//! MLX-C FFI context management.
//!
//! Provides safe wrappers around MLX stream and array management.

use crate::*;
use anyhow::Result;

/// Safe wrapper around MLX-C stream.
#[derive(Clone)]
pub struct MlxContext {
    pub stream: mlx_stream,
}

impl MlxContext {
    pub fn new(stream: mlx_stream) -> Self {
        Self { stream }
    }

    pub fn cpu() -> Self {
        let stream = unsafe { mlx_default_cpu_stream_new() };
        Self { stream }
    }

    pub fn gpu() -> Self {
        let stream = unsafe { mlx_default_gpu_stream_new() };
        Self { stream }
    }

    /// Check return code from MLX-C. 0 = OK.
    pub fn check(val: i32, op: &str) -> Result<()> {
        if val != 0 {
            anyhow::bail!("MLX op '{}' failed with code {}", op, val);
        }
        Ok(())
    }
}