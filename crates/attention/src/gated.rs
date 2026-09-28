//! Gated Output Attention — Q projection produces queries + output gate.
//!
//! Used by Qwen3.5/Qwen3Next full attention layers where:
//!   q_proj output = num_heads × head_dim × 2 (queries || gate)
//!   output = SDPA(q, k, v) * sigmoid(gate)
//!
//! The gate modulates the attention output before the O projection,
//! giving the model learned control over information flow.
//!
//! ## Difference from GQA
//!
//! - Q projection is 2× wider (produces queries + gate)
//! - After SDPA: output = attn_out * sigmoid(gate)
//! - Then O projection as normal

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use engine_mlx_kvcache::KvCache;
use engine_mlx_ops::quant::QuantWeights;

use crate::gqa::GqaConfig;

/// Weights for gated attention layer.
pub struct GatedAttnWeights {
    /// Q projection: output is 2× (queries + gate concatenated).
    pub q_proj: QuantWeights,
    pub k_proj: QuantWeights,
    pub v_proj: QuantWeights,
    pub o_proj: QuantWeights,
    /// Optional QK-norm.
    pub q_norm: Option<mlx_array>,
    pub k_norm: Option<mlx_array>,
}

/// Gated attention forward pass.
///
/// Same as GQA but Q projection produces queries + sigmoid gate.
/// Output = SDPA(q, k, v) * sigmoid(gate) → O projection.
pub fn gated_attention<C: KvCache>(
    ctx: &MlxCtx,
    x: mlx_array,
    weights: &GatedAttnWeights,
    cache: &mut C,
    config: &GqaConfig,
    offset: i32,
) -> Result<mlx_array> {
    let nh = config.n_heads;
    let nkv = config.n_kv_heads;
    let hd = config.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    let x_shape = ctx.shape(x)?;
    let seq_len = x_shape[1];
    let is_prefill = seq_len > 1;

    // 1. Q projection with gate (2× output)
    let (queries, gate) = engine_mlx_ops::gated_attn::project_q_gated(
        ctx, x, &weights.q_proj, nh, hd, seq_len,
    )?;

    // 2. K, V projection + reshape
    let k = engine_mlx_ops::quant::qmatmul(ctx, x, &weights.k_proj)?;
    let v = engine_mlx_ops::quant::qmatmul(ctx, x, &weights.v_proj)?;
    let k = ctx.reshape(k, &[1, seq_len, nkv, hd])?;
    let v = ctx.reshape(v, &[1, seq_len, nkv, hd])?;

    // 3. Optional QK-norm
    let (queries, k) = engine_mlx_ops::attention::optional_qk_norm(
        ctx, queries, k, weights.q_norm, weights.k_norm, config.norm_eps,
    )?;

    // 4. Transpose
    let (q, k, v) = engine_mlx_ops::attention::transpose_for_attn(ctx, queries, k, v)?;

    // 5. RoPE
    let (q, k) = engine_mlx_ops::rope::apply_rope(
        ctx, q, k, hd, config.rope_theta, offset, config.use_rope,
    )?;

    // 6. KV cache update
    cache.append(ctx, k, v)?;
    let (full_k, full_v) = cache.get(ctx)?;

    // 7. SDPA
    let attn = ctx.sdpa(q, full_k, full_v, scale, is_prefill)?;
    let attn = ctx.transpose_axes(attn, &[0, 2, 1, 3])?;
    let attn = ctx.reshape(attn, &[1, seq_len, nh * hd])?;

    // 8. Apply gate: output = attn * sigmoid(gate)
    engine_mlx_ops::gated_attn::apply_output_gate(ctx, attn, gate, &weights.o_proj)
}
