#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_ffi::ops` (low-level FFI helper wrappers).
//!
//! Every op is stubbed and returns `Err` with a "needs mlx-c FFI" marker.

use engine_mlx_ffi::{MlxCtx, mlx_stream};

fn null_ctx() -> MlxCtx {
    MlxCtx::new(unsafe { std::mem::zeroed() })
}

#[test]
#[cfg(not(feature = "mlx"))]
#[should_panic(expected = "MLX feature not enabled")]
fn cpu_panics_without_mlx() {
    let _ = MlxCtx::cpu();
}

#[test]
#[cfg(not(feature = "mlx"))]
fn check_is_pure_logic() {
    let _ctx = null_ctx();
    engine_mlx_ffi::MlxCtx::check(0, "x").unwrap();
    let err = engine_mlx_ffi::MlxCtx::check(1, "y").unwrap_err();
    assert!(err.to_string().contains("y"));
}

#[test]
#[cfg(not(feature = "mlx"))]
fn ops_bail_with_consistent_marker() {
    let ctx = null_ctx();
    let ops = [
        "embed",
        "matmul",
        "rms_norm",
        "sdpa",
        "rope",
        "quantized_matmul",
        "dequantize_weight",
        "concatenate",
        "softmax",
        "sum_axis",
    ];
    let results = [
        ctx.embed(mlx_array_placeholder(), mlx_array_placeholder()),
        ctx.matmul(mlx_array_placeholder(), mlx_array_placeholder()),
        ctx.rms_norm(mlx_array_placeholder(), mlx_array_placeholder(), 1e-5),
        ctx.sdpa(mlx_array_placeholder(), mlx_array_placeholder(), mlx_array_placeholder(), 1.0, true),
        ctx.rope(mlx_array_placeholder(), 64, 10000.0, 0),
        ctx.quantized_matmul(
            mlx_array_placeholder(),
            mlx_array_placeholder(),
            mlx_array_placeholder(),
            mlx_array_placeholder(),
            false,
            64,
            4,
        ),
        ctx.dequantize_weight(
            mlx_array_placeholder(),
            mlx_array_placeholder(),
            mlx_array_placeholder(),
            64,
            4,
        ),
        ctx.concatenate(mlx_array_placeholder(), mlx_array_placeholder(), 0),
        ctx.softmax(mlx_array_placeholder()),
        ctx.sum_axis(mlx_array_placeholder(), 0, false),
    ];
    for (name, r) in ops.iter().zip(results.iter()) {
        let err = r.as_ref().unwrap_err();
        assert!(
            err.to_string().contains("needs mlx-c FFI"),
            "op '{name}' should bail with mlx-c marker, got: {err}"
        );
    }
}

fn mlx_array_placeholder() -> engine_mlx_ffi::mlx_array {
    unsafe { std::mem::zeroed() }
}
