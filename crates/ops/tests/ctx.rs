#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_ops::ctx` module.
//!
//! `MlxCtx` construction requires an MLX stream, which is stubbed and panics
//! without mlx-c. This locks the "feature gate" contract.

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn new_cpu_panics_without_mlx() {
    let _ = engine_mlx_ops::ctx::MlxCtx::new_cpu();
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn new_gpu_panics_without_mlx() {
    let _ = engine_mlx_ops::ctx::MlxCtx::new_gpu(0);
}
