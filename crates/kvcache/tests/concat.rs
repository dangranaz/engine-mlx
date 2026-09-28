#![cfg(not(feature = "mlx"))] // stub-mode tests; skip under --features mlx
//! Unit tests for `engine_mlx_kvcache::concat` (raw concatenation cache).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::ConcatCache;

#[test]
fn new_is_empty() {
    let cache = ConcatCache::new();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = ConcatCache::new();
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn append_bails_without_mlx() {
    let mut cache = ConcatCache::new();
    let err = cache
        .append(
            &null_ctx(),
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        )
        .unwrap_err();
    // stub MlxCtx bails with "needs mlx-c FFI" on shape()
    assert!(err.to_string().contains("needs mlx-c FFI") || err.to_string().contains("shape"), "err={err}");
}

#[test]
fn get_bails_when_empty() {
    let cache = ConcatCache::new();
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("empty"), "err={err}");
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() })
}
