//! RoPE / NoPE position encoding.
//!
//! Every transformer model applies some form of positional encoding to Q and K
//* after the projection and before SDPA. The two variants supported are:
//!
//! * **RoPE** (Rotary Position Embedding) — used by Qwen3, Llama, Gemma, etc.
//! * **NoPE** (No Position Encoding) — used by SmolLM3 on ¾ of its layers.
//!
//! SmolLM3 uses a 3 : 1 NoPE/RoPE hybrid pattern, which means `apply_rope` must
//! be conditional per layer. This module provides exactly that: a single function
//! that applies RoPE when `use_rope == true` and returns Q/K unchanged otherwise.
//!
//! # Static vs dynamic offset
//!
//! * `apply_rope` – takes a plain `i32` offset. Used in the **manual** forward path.
//! * `apply_rope_dynamic` – takes an `mlx_array` offset. Used inside **`mx.compile`**
//!   closures where the shape graph must be static.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;

/// Apply RoPE to Q and K, or return them unchanged.
///
/// * `use_rope = true`  → standard RoPE with `(theta, offset)`
/// * `use_rope = false` → identity (NoPE layer)
///
/// Q/K must already be reshaped & transposed: `[1, n_heads, 1, head_dim]`.
pub fn apply_rope(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    head_dim: i32,
    theta: f32,
    offset: i32,
    use_rope: bool,
) -> Result<(mlx_array, mlx_array)> {
    if !use_rope {
        return Ok((q, k));
    }
    Ok((
        ctx.rope(q, head_dim, theta, offset)?,
        ctx.rope(k, head_dim, theta, offset)?,
    ))
}

/// Apply RoPE with a dynamic offset `mlx_array` (for `mx.compile` shapeless graphs).
///
/// Same semantics as `apply_rope` but `offset` is an `mlx_array` so the
/// compiled graph can change the offset at every step without recompilation.
pub fn apply_rope_dynamic(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    head_dim: i32,
    theta: f32,
    offset_arr: mlx_array,
    use_rope: bool,
) -> Result<(mlx_array, mlx_array)> {
    if !use_rope {
        return Ok((q, k));
    }
    Ok((
        ctx.rope_dynamic(q, head_dim, theta, offset_arr)?,
        ctx.rope_dynamic(k, head_dim, theta, offset_arr)?,
    ))
}
