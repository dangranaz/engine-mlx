#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_kvcache::fp8` (FP8 E4M3FN cache, simulated).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::Fp8Cache;

#[test]
fn new_is_empty() {
    let cache = Fp8Cache::new(64, 16);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = Fp8Cache::new(8, 2);
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn append_bails_without_mlx() {
    let mut cache = Fp8Cache::new(32, 8);
    let err = cache
        .append(
            &null_ctx(),
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        )
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI") || err.to_string().contains("shape"), "err={err}");
}

#[test]
fn get_bails_when_empty() {
    let cache = Fp8Cache::new(16, 4);
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("empty"), "err={err}");
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() })
}
