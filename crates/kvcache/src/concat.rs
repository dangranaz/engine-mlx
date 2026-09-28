//! Concat KV cache — raw concatenation, no compression.
//!
//! The simplest and most accurate strategy. Stores K/V as full-precision
//! mlx_array and concatenates along the sequence axis on each step.
//!
//! - **Pros**: Zero quality loss, simple implementation
//! - **Cons**: O(n) memory growth, O(n) copy on each append
//! - **Best for**: Short contexts, benchmarking, debugging

use anyhow::Result;
use engine_mlx_ffi::{mlx_array, MlxCtx};

use crate::KvCache;

/// Raw concatenation cache. Stores K/V as mlx_array on the compute graph.
pub struct ConcatCache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
}

impl ConcatCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }
}

impl Default for ConcatCache {
    fn default() -> Self { Self::new() }
}

impl KvCache for ConcatCache {
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()> {
        // Get new sequence length from k shape [1, n_kv_heads, new_seq, head_dim]
        let k_shape = ctx.shape(k)?;
        let new_seq = k_shape[2] as usize;

        self.k = Some(match self.k {
            Some(prev_k) => ctx.concatenate(prev_k, k, 2)?,
            None => k,
        });
        self.v = Some(match self.v {
            Some(prev_v) => ctx.concatenate(prev_v, v, 2)?,
            None => v,
        });

        self.seq_len += new_seq;
        Ok(())
    }

    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => anyhow::bail!("KV cache is empty"),
        }
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    fn memory_bytes(&self) -> usize {
        // Approximate: we don't know exact bytes without evaluating the array
        // Estimate based on seq_len * typical dimensions
        self.seq_len * 2 * 4 // placeholder
    }
}
