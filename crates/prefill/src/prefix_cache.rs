//! Prefix cache — radix tree for fast prefix matching with LRU eviction.
//!
//! SHA1 of token sequences enables O(1) lookup. Stores KV state offsets
//! so repeated prefixes skip prefill entirely.

use sha1::{Digest, Sha1};
use std::collections::HashMap;

/// Cached prefix entry.
#[derive(Clone)]
pub struct CachedPrefix {
    /// Number of tokens this entry covers.
    pub token_count: usize,
    /// Opaque state handle (model-specific KV state identifier).
    pub state_id: u64,
    /// Last access timestamp (for LRU eviction).
    pub last_access: u64,
}

/// Radix-based prefix cache for KV state reuse with LRU eviction.
pub struct PrefixCache {
    /// SHA1(token_prefix) -> cached state.
    entries: HashMap<[u8; 20], CachedPrefix>,
    /// Max entries before eviction.
    max_entries: usize,
    /// Monotonic counter for LRU ordering.
    access_counter: u64,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            access_counter: 0,
        }
    }

    /// Look up the longest cached prefix of `tokens`.
    /// Returns (cached_token_count, state_id) or None.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<CachedPrefix> {
        self.access_counter += 1;
        let current_access = self.access_counter;

        // Try progressively shorter prefixes (longest match first)
        // Check at boundaries: full, 3/4, 1/2, 1/4, system-only (first 256)
        let checkpoints = [
            tokens.len(),
            tokens.len() * 3 / 4,
            tokens.len() / 2,
            tokens.len() / 4,
            256.min(tokens.len()),
        ];

        for &len in &checkpoints {
            if len == 0 { continue; }
            let hash = hash_tokens(&tokens[..len]);
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.last_access = current_access;
                tracing::debug!(target: "engine_mlx::prefill", cached = entry.token_count, "prefix cache hit");
                return Some(entry.clone());
            }
        }
        None
    }

    /// Store a prefix in the cache.
    pub fn store(&mut self, tokens: &[u32], state_id: u64) {
        self.access_counter += 1;

        // LRU eviction: remove oldest entry if at capacity
        if self.entries.len() >= self.max_entries {
            if let Some(key_to_remove) = self.find_lru_key() {
                self.entries.remove(&key_to_remove);
                tracing::debug!(target: "engine_mlx::prefill", "evicted LRU prefix cache entry");
            }
        }

        let hash = hash_tokens(tokens);
        self.entries.insert(hash, CachedPrefix {
            token_count: tokens.len(),
            state_id,
            last_access: self.access_counter,
        });
    }

    /// Find the key of the least recently used entry.
    fn find_lru_key(&self) -> Option<[u8; 20]> {
        self.entries.iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| *key)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            max_entries: self.max_entries,
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn clear(&mut self) { self.entries.clear(); self.access_counter = 0; }
}

/// Cache statistics for observability.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub max_entries: usize,
}

/// SHA1 hash of a token sequence.
fn hash_tokens(tokens: &[u32]) -> [u8; 20] {
    let mut hasher = Sha1::new();
    for &t in tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.finalize().into()
}


