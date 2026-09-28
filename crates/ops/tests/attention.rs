#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_ops::attention` module.
//!
//! The ops crate now exposes GQA building blocks (project_qkv, transpose_for_attn, etc).
//! All operations require MLX-C FFI and panic/baile without the `mlx` feature.

use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use engine_mlx_ops::quant::QuantWeights;

fn dummy_array() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn dummy_weights() -> QuantWeights {
    let a = dummy_array();
    QuantWeights::new(a, a, a)
}

fn null_ctx() -> MlxCtx {
    MlxCtx::new(unsafe { std::mem::zeroed() })
}

#[test]
#[should_panic(expected = "not implemented")]
fn project_qkv_panics_without_mlx() {
    let ctx = null_ctx();
    let _ = engine_mlx_ops::attention::project_qkv(&ctx, dummy_array(), &dummy_weights(), &dummy_weights(), &dummy_weights(), 8, 2, 128, 1).unwrap();
}

#[test]
#[should_panic(expected = "not implemented")]
fn transpose_for_attn_panics_without_mlx() {
    let ctx = null_ctx();
    let _ = engine_mlx_ops::attention::transpose_for_attn(&ctx, dummy_array(), dummy_array(), dummy_array()).unwrap();
}

#[test]
#[should_panic(expected = "not implemented")]
fn attend_and_project_panics_without_mlx() {
    let ctx = null_ctx();
    let _ = engine_mlx_ops::attention::attend_and_project(&ctx, dummy_array(), dummy_array(), dummy_array(), &dummy_weights(), 0.1, true, 8, 128).unwrap();
}
