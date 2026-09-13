#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_ops::linear` module.
//!
//! Linear ops are stubbed and panic without the `mlx` feature / mlx-c headers.

use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn dummy_array() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn null_ctx() -> MlxCtx {
    MlxCtx::new(unsafe { std::mem::zeroed() })
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn linear_panics_without_mlx() {
    let ctx = null_ctx();
    let _ = engine_mlx_ops::linear::linear(&ctx, dummy_array(), dummy_array(), None);
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn linear_quantized_panics_without_mlx() {
    let ctx = null_ctx();
    let _ = engine_mlx_ops::linear::linear_quantized(
        &ctx,
        dummy_array(),
        dummy_array(),
        dummy_array(),
        dummy_array(),
        64,
        4,
    );
}
