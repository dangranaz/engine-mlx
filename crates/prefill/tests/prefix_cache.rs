//! Tests for `engine_mlx_prefill::prefix_cache` — extracted from inline `#[cfg(test)]`.

use engine_mlx_prefill::prefix_cache::PrefixCache;

#[test]
fn test_prefix_cache_lru_eviction() {
    let mut cache = PrefixCache::new(3);

    cache.store(&[1, 2, 3], 1);
    cache.store(&[4, 5, 6], 2);
    cache.store(&[7, 8, 9], 3);

    cache.lookup(&[1, 2, 3]);

    cache.store(&[10, 11, 12], 4);

    assert!(cache.lookup(&[1, 2, 3]).is_some());
    assert!(cache.lookup(&[4, 5, 6]).is_none());
    assert!(cache.lookup(&[7, 8, 9]).is_some());
}

#[test]
fn test_prefix_cache_lookup_updates_lru() {
    let mut cache = PrefixCache::new(2);

    cache.store(&[1, 2], 1);
    cache.store(&[3, 4], 2);

    cache.lookup(&[1, 2]);

    cache.store(&[5, 6], 3);

    assert!(cache.lookup(&[1, 2]).is_some());
    assert!(cache.lookup(&[3, 4]).is_none());
}
