#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_kvcache::rotating` (sliding-window cache).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::RotatingCache;

#[test]
fn new_is_empty() {
    let cache = RotatingCache::new(128, 2, 64);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    // memory is fixed window allocation
    assert_eq!(cache.memory_bytes(), 128 * 2 * 64 * 4 * 2);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = RotatingCache::new(32, 1, 16);
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn append_bails_without_mlx() {
    let mut cache = RotatingCache::new(16, 2, 8);
    let err = cache
        .append(
            &null_ctx(),
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        )
        .unwrap_err();
    // shape not implemented in stub
    assert!(err.to_string().contains("needs mlx-c FFI") || err.to_string().contains("shape") || err.to_string().contains("empty"), "err={err}");
}

#[test]
fn get_bails_when_empty() {
    let cache = RotatingCache::new(16, 1, 8);
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("empty"), "err={err}");
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() })
}
