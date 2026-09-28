//! Rotating KV cache — sliding window with fixed memory.
//!
//! Keeps only the last `window_size` tokens. Older tokens are overwritten
//! in a circular buffer pattern. Memory is constant regardless of sequence length.
//!
//! - **Pros**: O(1) memory, O(1) append, perfect for sliding window attention
//! - **Cons**: Loses tokens beyond the window
//! - **Best for**: Models with sliding window attention (Qwen3 SWA layers, Mistral)
//!
//! ## Reference
//!
//! - `vendor/mlx-lm/mlx_lm/models/cache.py` → `RotatingKVCache`
//! - `vendor/rmlx/crates/rmlx-kv-quant/` → rotating cache variants

use anyhow::Result;
use engine_mlx_ffi::{MlxCtx, mlx_array};

use crate::KvCache;

/// Rotating (sliding window) KV cache.
///
/// Pre-allocates buffers of `[1, n_kv_heads, window_size, head_dim]` and
/// overwrites the oldest position on each append.
pub struct RotatingCache {
    /// Full K buffer: [1, n_kv_heads, window_size, head_dim]
    k: Option<mlx_array>,
    /// Full V buffer: [1, n_kv_heads, window_size, head_dim]
    v: Option<mlx_array>,
    /// Write position (wraps around window_size)
    write_pos: usize,
    /// Total tokens seen (may exceed window_size)
    total_len: usize,
    /// Maximum tokens stored
    window_size: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl RotatingCache {
    pub fn new(window_size: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            write_pos: 0,
            total_len: 0,
            window_size,
            n_kv_heads,
            head_dim,
        }
    }
}

impl KvCache for RotatingCache {
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()> {
        let k_shape = ctx.shape(k)?;
        let new_seq = k_shape[2] as usize;

        if self.total_len == 0 && new_seq <= self.window_size {
            // First call (prefill or first token): store directly
            self.k = Some(k);
            self.v = Some(v);
            self.write_pos = new_seq;
            self.total_len = new_seq;
            return Ok(());
        }

        // For decode (new_seq == 1), concatenate and trim if needed
        let full_k = match self.k {
            Some(prev) => ctx.concatenate(prev, k, 2)?,
            None => k,
        };
        let full_v = match self.v {
            Some(prev) => ctx.concatenate(prev, v, 2)?,
            None => v,
        };

        self.total_len += new_seq;

        // If exceeds window, slice to keep only last window_size tokens
        let current_seq = ctx.shape(full_k)?[2] as usize;
        if current_seq > self.window_size {
            let start = (current_seq - self.window_size) as i32;
            let end = current_seq as i32;
            let nkv = self.n_kv_heads as i32;
            let hd = self.head_dim as i32;

            self.k = Some(ctx.slice(full_k, &[0, 0, start, 0], &[1, nkv, end, hd], &[1, 1, 1, 1])?);
            self.v = Some(ctx.slice(full_v, &[0, 0, start, 0], &[1, nkv, end, hd], &[1, 1, 1, 1])?);
            self.write_pos = self.window_size;
        } else {
            self.k = Some(full_k);
            self.v = Some(full_v);
            self.write_pos = current_seq;
        }

        Ok(())
    }

    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => anyhow::bail!("Rotating KV cache is empty"),
        }
    }

    fn len(&self) -> usize {
        self.write_pos.min(self.total_len)
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.write_pos = 0;
        self.total_len = 0;
    }

    fn memory_bytes(&self) -> usize {
        // Fixed: window_size × n_kv_heads × head_dim × 4 bytes × 2 (K+V)
        self.window_size * self.n_kv_heads * self.head_dim * 4 * 2
    }
}
