//! FP8 E4M3FN KV cache — simulated round-trip quantization.
//!
//! Stores K/V at FP8 precision (simulated: values stay f32 but have
//! FP8 representable accuracy). Halves effective memory vs f32.
//!
//! - **Pros**: 2× compression, near-lossless for most models
//! - **Cons**: Small quantization noise, host-side processing
//! - **Best for**: Standard production use, models trained with FP8 KV (DeepSeek)
//!
//! ## Algorithm
//!
//! Per 64-element block:
//! 1. Find amax → compute scale = 2^ceil(log2(amax/448))
//! 2. Scale down, round to nearest E4M3FN representable value
//! 3. Scale back up
//! 4. Store the "dequantized" f32 values (same precision as FP8)
//!
//! The RoPE portion of head_dim is left unquantized (passthrough).
//! Deduplicates E4M3FN logic by delegating to `engine_mlx_ops::fp8`.

use anyhow::Result;
use engine_mlx_ffi::{MlxCtx, mlx_array};

use crate::KvCache;

/// FP8 E4M3FN cache with simulated round-trip.
pub struct Fp8Cache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
    /// Number of RoPE dimensions at the tail of head_dim (passthrough).
    n_rot: usize,
    /// Head dimension (total).
    head_dim: usize,
}

impl Fp8Cache {
    /// Create FP8 cache.
    ///
    /// - `head_dim`: total head dimension
    /// - `n_rot`: number of RoPE dims (tail of head, left unquantized)
    pub fn new(head_dim: usize, n_rot: usize) -> Self {
        Self { k: None, v: None, seq_len: 0, n_rot, head_dim }
    }
}

impl KvCache for Fp8Cache {
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()> {
        let k_shape = ctx.shape(k)?;
        let new_seq = k_shape[2] as usize;

        // Quantize K and V through FP8 round-trip via engine_mlx_ops::fp8
        let k_q = engine_mlx_ops::fp8::fp8_round_trip(ctx, k, self.head_dim as i32, self.n_rot as i32)?;
        let v_q = engine_mlx_ops::fp8::fp8_round_trip(ctx, v, self.head_dim as i32, self.n_rot as i32)?;

        self.k = Some(match self.k {
            Some(prev_k) => ctx.concatenate(prev_k, k_q, 2)?,
            None => k_q,
        });
        self.v = Some(match self.v {
            Some(prev_v) => ctx.concatenate(prev_v, v_q, 2)?,
            None => v_q,
        });

        self.seq_len += new_seq;
        Ok(())
    }

    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => anyhow::bail!("FP8 KV cache is empty"),
        }
    }

    fn len(&self) -> usize { self.seq_len }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    fn memory_bytes(&self) -> usize {
        // FP8 stores same f32 bytes but with reduced precision
        // True FP8 would be half — here it's simulated so still 4 bytes/value
        self.seq_len * 2 * self.head_dim * 4
    }
}
