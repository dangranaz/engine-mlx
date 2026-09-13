//! Unit tests for the `engine_mlx_kvcache::KvCache` trait contract.

use engine_mlx_kvcache::KvCache;

/// A trivial in-memory cache used to exercise the trait's default method
/// (`is_empty`) and the required surface, confirming the contract compiles
/// and behaves as documented.
struct InMemoryCache {
    seq_len: usize,
}

impl KvCache for InMemoryCache {
    fn append(&mut self, _ctx: &engine_mlx_ops::MlxCtx, _k: engine_mlx_ops::ffi::mlx_array, _v: engine_mlx_ops::ffi::mlx_array) -> anyhow::Result<()> {
        self.seq_len += 1;
        Ok(())
    }

    fn get(&self, _ctx: &engine_mlx_ops::MlxCtx) -> anyhow::Result<(engine_mlx_ops::ffi::mlx_array, engine_mlx_ops::ffi::mlx_array)> {
        Ok((
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        ))
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) {
        self.seq_len = 0;
    }

    fn memory_bytes(&self) -> usize {
        self.seq_len * 16
    }
}

#[test]
fn is_empty_default_uses_len() {
    let cache = InMemoryCache { seq_len: 0 };
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn append_updates_len_and_not_empty() {
    let mut cache = InMemoryCache { seq_len: 0 };
    let ctx = engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() });
    cache
        .append(&ctx, unsafe { std::mem::zeroed() }, unsafe { std::mem::zeroed() })
        .unwrap();
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    cache.reset();
    assert!(cache.is_empty());
}
