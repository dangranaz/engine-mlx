#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for the MLX-C FFI stub functions in `engine_mlx_ffi`.
//!
//! Without the `mlx` feature, every FFI symbol is a stub that panics with the
//! feature-gate message. These tests assert the gate is active so a future
//! incomplete bindgen cannot ship silently.

use engine_mlx_ffi::{mlx_array, mlx_stream};

fn null_stream() -> mlx_stream {
    unsafe { std::mem::zeroed() }
}

fn null_array() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn default_cpu_stream_new_panics() {
    let _ = engine_mlx_ffi::mlx_default_cpu_stream_new();
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn default_gpu_stream_new_panics() {
    let _ = engine_mlx_ffi::mlx_default_gpu_stream_new();
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn synchronize_panics() {
    engine_mlx_ffi::mlx_synchronize(null_stream());
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn array_new_data_panics() {
    let shape = [1i32, 2, 3];
    let _ = engine_mlx_ffi::mlx_array_new_data(std::ptr::null(), shape.as_ptr(), 3, engine_mlx_ffi::mlx_dtype::MLX_FLOAT32);
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn quantized_matmul_panics() {
    let _ = engine_mlx_ffi::mlx_quantized_matmul(
        std::ptr::null_mut(),
        null_array(),
        null_array(),
        null_array(),
        null_array(),
        false,
        64,
        4,
        std::ptr::null(),
        null_stream(),
    );
}
