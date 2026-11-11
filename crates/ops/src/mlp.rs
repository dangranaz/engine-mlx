//! SwiGLU MLP — the standard feed-forward block in every supported model.
//!
//! Every decoder-only model (Qwen3, SmolLM3, Llama, Gemma, etc.) uses
//! the same SwiGLU MLP pattern:
//!
//! ```text
//! hidden = SiLU(x @ gate_proj) * (x @ up_proj)
//! out    = hidden @ down_proj
//! ```
//!
//! All three projections use 4-bit quantized weights with `group_size=64`.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use crate::quant::{qmatmul, QuantWeights};

/// SwiGLU feed-forward block.
///
/// * `gate_proj` — projection for the SiLU gate
/// * `up_proj` — projection for the "up" branch
/// * `down_proj` — output projection
pub fn swiglu_mlp(
    ctx: &MlxCtx,
    x: mlx_array,
    gate_proj: &QuantWeights,
    up_proj: &QuantWeights,
    down_proj: &QuantWeights,
) -> Result<mlx_array> {
    let gate = qmatmul(ctx, x, gate_proj)?;
    let up = qmatmul(ctx, x, up_proj)?;
    let gate = ctx.silu(gate)?;
    let hidden = ctx.multiply(gate, up)?;
    qmatmul(ctx, hidden, down_proj)
}
