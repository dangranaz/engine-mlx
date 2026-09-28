//! Token embedding — maps token IDs to hidden vectors via dequantized weight matrix.
//!
//! The standard pattern is:
//!
//! 1. **Dequantize** the 4-bit quantized `embed_tokens` weight once at init
//! 2. **Take** rows corresponding to the input IDs
//! 3. **Expand dims** to add the batch dimension: `[seq_len, hidden]` → `[1, seq_len, hidden]`

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi as ffi;
use engine_mlx_ffi::MlxCtx;

/// Dequantize the embedding weight matrix and cache it.
///
/// Call this once during `from_dir()`. The returned `mlx_array` is the
/// full-precision (f32) embedding table, ready for `take`.
pub fn dequant_embed(ctx: &MlxCtx, tokens: mlx_array, scales: mlx_array, biases: mlx_array) -> Result<mlx_array> {
    let e = ctx.dequantize_weight(tokens, scales, biases, 64, 4)?;
    ctx.evaluate(e)?;
    Ok(e)
}

/// Embed a sequence of token IDs via `take` from the (dequantized) table.
///
/// Returns `[1, seq_len, hidden_size]`.
pub fn embed_seq(ctx: &MlxCtx, embed_dequant: mlx_array, ids: &[u32]) -> Result<mlx_array> {
    let i32_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let idx = unsafe {
        ffi::mlx_array_new_data(
            i32_ids.as_ptr() as *const std::ffi::c_void,
            [ids.len() as i32].as_ptr(),
            1,
            engine_mlx_ffi::mlx_dtype::MLX_INT32,
        )
    };
    let rows = ctx.take(embed_dequant, idx)?;
    ctx.expand_dims(rows, 0)
}
