//! # engine-mlx-kvcache
//!
//! KV cache implementations for MLX-C inference engine.
//!
//! ## Design
//!
//! The KV cache stores Key and Value tensors from previous decode steps
//! so that attention can attend over the full context without recomputing.
//! Different strategies trade memory for precision:
//!
//! | Strategy | Compression | Quality | Use case |
//! |----------|-------------|---------|----------|
//! | [`concat`] | None (f32) | Perfect | Short contexts, benchmarking |
//! | [`fp8`] | 2× (E4M3FN) | Near-lossless | Standard production |
//! | [`rotating`] | Windowed | Perfect (within window) | Sliding window models |
//!
//! ## Trait
//!
//! All cache backends implement [`KvCache`]:
//!
//! ```rust,ignore
//! let mut cache = ConcatCache::new();
//! cache.append(ctx, k_new, v_new)?;
//! let (full_k, full_v) = cache.get(ctx)?;
//! // full_k/full_v contain ALL accumulated K/V for SDPA
//! ```
//!
//! ## Combinability with Attention
//!
//! This crate is designed to be used by `engine_mlx_attention`. The attention
//! crate chooses which cache backend to use based on model config and context length.
//!
//! ```text
//! engine-mlx-kvcache (this)   →  storage/compression algorithms
//!        ↓
//! engine-mlx-attention        →  composes cache + SDPA into full attention
//!        ↓
//! model crate                 →  chooses attention variant + cache variant
//! ```

pub mod concat;
pub mod fp8;
pub mod rotating;

use anyhow::Result;
use engine_mlx_ffi::{MlxCtx, mlx_array};

/// KV cache backend trait.
///
/// All implementations store K/V tensors and provide retrieval for SDPA.
/// Tensors are `mlx_array` handles — they live on the MLX compute graph.
pub trait KvCache: Send {
    /// Append new K and V for the current step.
    ///
    /// `k` shape: `[1, n_kv_heads, seq_len, head_dim]` (already transposed)
    /// `v` shape: `[1, n_kv_heads, seq_len, head_dim]`
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()>;

    /// Get the full accumulated K and V for SDPA.
    ///
    /// Returns `(full_k, full_v)` covering all stored steps.
    /// Shape: `[1, n_kv_heads, total_seq, head_dim]`
    fn get(&self, ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)>;

    /// Current sequence length stored in cache.
    fn len(&self) -> usize;

    /// Whether the cache is empty.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Reset the cache (start new generation).
    fn reset(&mut self);

    /// Approximate memory usage in bytes.
    fn memory_bytes(&self) -> usize;
}

// Re-export all cache types
pub use concat::ConcatCache;
pub use fp8::Fp8Cache;
pub use rotating::RotatingCache;
