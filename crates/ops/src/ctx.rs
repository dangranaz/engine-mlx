//! MLX context wrapper.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_stream;

/// Wrapper around MLX stream.
pub struct MlxCtx {
    pub stream: mlx_stream,
    _device_id: i32,
}

impl MlxCtx {
    pub fn new_cpu() -> Self {
        let stream = unsafe { crate::ffi::mlx_default_cpu_stream_new() };
        Self { stream, _device_id: 0 }
    }

    pub fn new_gpu(device_id: i32) -> Self {
        let stream = unsafe { crate::ffi::mlx_default_gpu_stream_new() };
        Self { stream, _device_id: device_id }
    }

    pub fn synchronize(&self) {
        unsafe { crate::ffi::mlx_synchronize(self.stream) };
    }
}
