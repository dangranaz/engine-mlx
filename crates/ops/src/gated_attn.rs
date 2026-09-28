//! Gated Output Attention — Q projection produces queries + output gate.
//!
//! Used by Qwen3.5/Qwen3Next full attention layers where:
//!   q_proj output = num_heads × head_dim × 2 (queries || gate)
//!   output = SDPA(q, k, v) * sigmoid(gate)
//!
//! Reusable by any model with `attn_output_gate = true`.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;

use crate::quant::QuantWeights;

/// Project Q with output gate: q_proj output is 2× normal, split into queries + gate.
///
/// Returns (queries, gate) where:
/// - queries: [batch, seq, num_heads, head_dim] (ready for transpose)
/// - gate: [batch, seq, num_heads * head_dim] (applied after SDPA, before o_proj)
pub fn project_q_gated(
    ctx: &MlxCtx,
    x: mlx_array,
    q_weights: &QuantWeights,
    num_heads: i32,
    head_dim: i32,
    seq_len: i32,
) -> Result<(mlx_array, mlx_array)> {
    // q_proj output: [seq, num_heads * head_dim * 2]
    let q_full = crate::quant::qmatmul(ctx, x, q_weights)?;

    // Reshape to [1, seq, num_heads, head_dim * 2]
    let reshaped = ctx.reshape(q_full, &[1, seq_len, num_heads, head_dim * 2])?;

    // Split along last dim: queries [1,seq,nh,hd] and gate [1,seq,nh,hd]
    let queries = ctx.slice(reshaped, &[0, 0, 0, 0], &[1, seq_len, num_heads, head_dim], &[1, 1, 1, 1])?;
    let gate_4d = ctx.slice(reshaped, &[0, 0, 0, head_dim], &[1, seq_len, num_heads, head_dim * 2], &[1, 1, 1, 1])?;

    // Gate reshaped to [1, seq, num_heads * head_dim] for element-wise multiply after SDPA
    let gate = ctx.reshape(gate_4d, &[1, seq_len, num_heads * head_dim])?;

    Ok((queries, gate))
}

/// Apply gated attention output: multiply SDPA result by sigmoid(gate), then o_proj.
///
/// - attn_output: [1, seq, num_heads * head_dim] (after SDPA transpose+reshape)
/// - gate: [1, seq, num_heads * head_dim]
/// - o_weights: output projection weights
pub fn apply_output_gate(
    ctx: &MlxCtx,
    attn_output: mlx_array,
    gate: mlx_array,
    o_weights: &QuantWeights,
) -> Result<mlx_array> {
    let gate_sigmoid = ctx.sigmoid(gate)?;
    let gated = ctx.multiply(attn_output, gate_sigmoid)?;
    crate::quant::qmatmul(ctx, gated, o_weights)
}
