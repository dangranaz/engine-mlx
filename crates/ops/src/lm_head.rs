//! Language model head — projects hidden states to vocabulary logits.
//!
//! Two variants are supported:
//!
//! * **Tied** — `lm_head.weight` reuses `embed_tokens.weight` (SmolLM3)
//! * **Untied** — separate LM head matrix, accessed via `QuantWeights` (Qwen3)
//!
//! Both path are just a single `quantized_matmul`; the difference is which
//! `QuantWeights` bundle is passed.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use crate::quant::{qmatmul, QuantWeights};

/// Project hidden state to vocabulary logits.
///
/// For **prefill** (`seq_len > 1`) this extracts only the last position's
/// hidden state before projection, matching the standard LLM eval pattern.
/// For **decode** (`seq_len == 1`) it projects the single token directly.
pub fn lm_head(
    ctx: &MlxCtx,
    x: mlx_array,
    head: &QuantWeights,
    hidden_size: i32,
    seq_len: usize,
) -> Result<mlx_array> {
    let hs = hidden_size;
    let sl = seq_len as i32;

    let x = if seq_len > 1 {
        let flat = ctx.reshape(x, &[sl, hs])?;
        let last_idx = unsafe {
            engine_mlx_ffi::mlx_array_new_data(
                [sl - 1].as_ptr() as *const std::ffi::c_void,
                [1].as_ptr(),
                1,
                engine_mlx_ffi::mlx_dtype::MLX_INT32,
            )
        };
        ctx.reshape(ctx.take(flat, last_idx)?, &[1, hs])?
    } else {
        ctx.reshape(x, &[1, hs])?
    };

    qmatmul(ctx, x, head)
}
