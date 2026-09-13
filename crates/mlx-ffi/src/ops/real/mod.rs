use anyhow::Result;
use crate::*;

pub struct MlxCtx {
    pub stream: mlx_stream,
}

unsafe extern "C" fn custom_error_handler(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
    if msg.is_null() {
        return;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(msg) };
    if let Ok(s) = cstr.to_str() {
        tracing::warn!(target: "engine_mlx::mlx::error", "MLX error: {s}");
    }
}

impl MlxCtx {
    pub fn new(stream: mlx_stream) -> Self {
        unsafe { mlx_set_error_handler(Some(custom_error_handler), std::ptr::null_mut(), None); }
        tracing::info!(target: "engine_mlx::mlx::ctx", "MlxCtx::new");
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

    pub fn check(val: i32, op: &str) -> Result<()> {
        if val != 0 {
            anyhow::bail!("MLX op '{}' failed with code {}", op, val);
        }
        Ok(())
    }
}

mod core;
mod blas;
mod transform;
mod slice;
mod elementwise;
mod array;
mod quant;
