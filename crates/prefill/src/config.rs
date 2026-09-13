//! Prefill pipeline configuration.

/// Pipeline configuration — controls which stages activate.
#[derive(Debug, Clone)]
pub struct PrefillConfig {
    /// Chunk size for batch prefill (tokens per forward pass).
    pub chunk_size: usize,
    /// Enable prefix cache (radix tree in RAM).
    pub prefix_cache_enabled: bool,
    /// Max entries in prefix cache.
    pub prefix_cache_max_entries: usize,
}

impl Default for PrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            prefix_cache_enabled: true,
            prefix_cache_max_entries: 64,
        }
    }
}
