//! Mixture-of-Experts (MoE) routing and computation.
//!
//! Generic implementation that supports:
//! - Configurable number of experts and top-k routing
//! - Optional expert bias (additive bias to gate scores before softmax)
//! - Optional top-k probability normalization
//! - SwiGLU MLP per expert (gate_proj/up_proj/down_proj)
//! - Quantized expert weights (4-bit or 8-bit)
//!
//! ## Usage
//!
//! ```rust,ignore
//! let moe_cfg = MoeConfig {
//!     num_experts: 32,
//!     top_k: 4,
//!     hidden_size: 2048,
//!     norm_topk_prob: true,
//!     gate_bits: 8,
//!     gate_group_size: 64,
//! };
//!
//! let output = moe_forward(
//!     ctx, x, &moe_cfg,
//!     &gate_weights, expert_bias,
//!     &expert_w1, &expert_w2, &expert_w3,
//! )?;
//! ```
//!
//! ## Supported models
//!
//! | Model | Experts | Top-K | Gate quant |
//! |-------|---------|-------|------------|
//! | LFM2.5-8B-A1B | 32 | 4 | 8-bit |
//! | Gemma4-26B-A4B | 128 | 8 | varies |
//! | GPT-OSS-20B | 32 | 4 | varies |
//! | Qwen3.5-35B-A3B | varies | varies | varies |

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use crate::quant::{qmatmul, QuantWeights};

/// MoE configuration — model-agnostic.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts.
    pub num_experts: i32,
    /// Number of experts selected per token (top-k).
    pub top_k: i32,
    /// Hidden size (used for output shape).
    pub hidden_size: i32,
    /// Whether to normalize top-k probabilities to sum to 1.
    pub norm_topk_prob: bool,
    /// Gate weight quantization bits (typically 8 for LFM, 4 for others).
    pub gate_bits: i32,
    /// Gate weight quantization group size.
    pub gate_group_size: i32,
    /// Expert weight quantization bits (default 4).
    pub expert_bits: i32,
    /// Expert weight quantization group size (default 64, 32 for mxfp4).
    pub expert_group_size: i32,
    /// Expert quantization mode: "affine" or "mxfp4".
    pub expert_mode: &'static str,
}

/// Expert weights bundle — all experts packed in 3D tensors.
///
/// Each tensor has shape `[num_experts, dim_out, dim_in_packed]` for quantized,
/// or `[num_experts, dim_out, dim_in]` for non-quantized.
///
/// - `w1` (gate_proj): [num_experts, intermediate, hidden_packed]
/// - `w3` (up_proj):   [num_experts, intermediate, hidden_packed]
/// - `w2` (down_proj): [num_experts, hidden, intermediate_packed]
#[derive(Clone)]
pub struct MoeExpertWeights {
    pub w1_w: mlx_array,
    pub w1_s: mlx_array,
    pub w1_b: mlx_array,
    pub w2_w: mlx_array,
    pub w2_s: mlx_array,
    pub w2_b: mlx_array,
    pub w3_w: mlx_array,
    pub w3_s: mlx_array,
    pub w3_b: mlx_array,
}

/// Gate weights for routing.
#[derive(Clone)]
pub struct MoeGateWeights {
    pub w: mlx_array,
    pub s: mlx_array,
    pub b: mlx_array,
    /// Optional additive bias applied to gate scores before softmax.
    pub expert_bias: Option<mlx_array>,
}

/// MoE forward pass: route input through top-k experts.
///
/// `x` should be `[1, hidden_size]` (single token) or `[seq_len, hidden_size]` (flattened batch).
/// For batch > 1, routing is done per-token independently.
///
/// Returns the weighted combination of expert outputs, same shape as `x`.
pub fn moe_forward(
    ctx: &MlxCtx,
    x: mlx_array,
    cfg: &MoeConfig,
    gate: &MoeGateWeights,
    experts: &MoeExpertWeights,
) -> Result<mlx_array> {
    let hidden = cfg.hidden_size;

    // 1. Compute gate scores: x @ gate_weight → [1, num_experts]
    let gate_qw = QuantWeights::with_quant(gate.w, gate.s, gate.b, cfg.gate_bits, cfg.gate_group_size);
    let gate_scores = qmatmul(ctx, x, &gate_qw)?;

    // 2. Add expert bias if present
    let gate_scores = if let Some(bias) = gate.expert_bias {
        ctx.add(gate_scores, bias)?
    } else {
        gate_scores
    };

    // 3. Softmax → probabilities
    let gate_probs = ctx.softmax_axis(gate_scores, -1)?;

    // 4. Top-k on CPU (small array, cheaper than GPU sort)
    let scores_f32 = ctx.to_vec_f32(gate_probs)?;
    let (top_k_indices, top_k_weights) = topk_cpu(&scores_f32, cfg.top_k as usize, cfg.norm_topk_prob);

    // 5. Route through selected experts
    let w1_shape = ctx.shape(experts.w1_w)?;
    let s1_shape = ctx.shape(experts.w1_s)?;
    let w3_shape = ctx.shape(experts.w3_w)?;
    let s3_shape = ctx.shape(experts.w3_s)?;
    let w2_shape = ctx.shape(experts.w2_w)?;
    let s2_shape = ctx.shape(experts.w2_s)?;

    let mut output = ctx.zeros(&[1, hidden])?;

    for (rank, &expert_idx) in top_k_indices.iter().enumerate() {
        let ei = expert_idx as i32;
        let idx = ctx.new_array_i32(&[ei])?;

        // Extract per-expert weights from packed 3D tensors
        let w1 = extract_expert_weights(ctx, experts.w1_w, experts.w1_s, experts.w1_b, idx, &w1_shape, &s1_shape, cfg.expert_bits, cfg.expert_group_size, cfg.expert_mode)?;
        let w3 = extract_expert_weights(ctx, experts.w3_w, experts.w3_s, experts.w3_b, idx, &w3_shape, &s3_shape, cfg.expert_bits, cfg.expert_group_size, cfg.expert_mode)?;
        let w2 = extract_expert_weights(ctx, experts.w2_w, experts.w2_s, experts.w2_b, idx, &w2_shape, &s2_shape, cfg.expert_bits, cfg.expert_group_size, cfg.expert_mode)?;

        // SwiGLU MLP: silu(x @ w1) * (x @ w3) → @ w2
        let expert_out = crate::mlp::swiglu_mlp(ctx, x, &w1, &w3, &w2)?;

        // For MXFP4, add the per-output bias from w2 (down_proj bias)
        let expert_out = if cfg.expert_mode == "mxfp4" {
            ctx.add(expert_out, w2.biases)?
        } else {
            expert_out
        };

        // Weighted contribution
        let weight_scalar = ctx.new_array(&[top_k_weights[rank]])?;
        let weighted = ctx.multiply(expert_out, weight_scalar)?;
        output = ctx.add(output, weighted)?;
    }

    Ok(output)
}

/// MoE forward for a sequence of tokens (batch routing).
///
/// `x` is `[seq_len, hidden_size]`. Routes each token independently.
/// More expensive than single-token but needed for prefill.
///
/// Returns `[seq_len, hidden_size]`.
pub fn moe_forward_batch(
    ctx: &MlxCtx,
    x: mlx_array,
    cfg: &MoeConfig,
    gate: &MoeGateWeights,
    experts: &MoeExpertWeights,
) -> Result<mlx_array> {
    let x_shape = ctx.shape(x)?;
    let seq_len = x_shape[0];

    if seq_len == 1 {
        return moe_forward(ctx, x, cfg, gate, experts);
    }

    // Process each token independently
    let hidden = cfg.hidden_size;
    let mut outputs: Vec<mlx_array> = Vec::with_capacity(seq_len as usize);

    for t in 0..seq_len {
        let x_t = ctx.slice(x, &[t, 0], &[t + 1, hidden], &[1, 1])?;
        let out_t = moe_forward(ctx, x_t, cfg, gate, experts)?;
        outputs.push(out_t);
    }

    // Concatenate along seq dimension
    if outputs.len() == 1 {
        return Ok(outputs[0]);
    }
    let mut acc = outputs[0];
    for i in 1..outputs.len() {
        acc = ctx.concatenate(acc, outputs[i], 0)?;
    }
    Ok(acc)
}

/// Extract a single expert's weights from packed 3D tensors.
///
/// `packed_w` shape: [num_experts, dim_out, dim_in_packed]
/// Returns QuantWeights for that expert: w=[dim_out, dim_in_packed], s/b=[dim_out, groups]
fn extract_expert_weights(
    ctx: &MlxCtx,
    packed_w: mlx_array,
    packed_s: mlx_array,
    packed_b: mlx_array,
    idx: mlx_array,
    w_shape: &[i32],
    s_shape: &[i32],
    bits: i32,
    group_size: i32,
    mode: &'static str,
) -> Result<QuantWeights> {
    // take(packed, idx) → [1, dim1, dim2] → reshape to [dim1, dim2]
    let w_3d = ctx.take(packed_w, idx)?;
    let w_2d = ctx.reshape(w_3d, &[w_shape[1], w_shape[2]])?;

    let s_3d = ctx.take(packed_s, idx)?;
    let s_2d = ctx.reshape(s_3d, &[s_shape[1], s_shape[2]])?;

    let b_3d = ctx.take(packed_b, idx)?;
    // Bias shape depends on mode:
    // - affine: [dim_out, groups] (same as scales)
    // - mxfp4: [dim_out] (per-output additive bias)
    let b_2d = if mode == "mxfp4" {
        // For mxfp4, bias is [num_experts, dim_out] → take → [1, dim_out] → [dim_out]
        ctx.reshape(b_3d, &[w_shape[1]])?
    } else {
        ctx.reshape(b_3d, &[s_shape[1], s_shape[2]])?
    };

    Ok(QuantWeights {
        weight: w_2d,
        scales: s_2d,
        biases: b_2d,
        bits,
        group_size,
        mode,
    })
}

/// CPU top-k selection with optional normalization.
///
/// Returns (indices, weights) sorted by score descending.
fn topk_cpu(scores: &[f32], k: usize, normalize: bool) -> (Vec<usize>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_indices: Vec<usize> = indexed.iter().take(k).map(|&(i, _)| i).collect();
    let mut top_weights: Vec<f32> = top_indices.iter().map(|&i| scores[i]).collect();

    if normalize {
        let sum: f32 = top_weights.iter().sum();
        if sum > 1e-20 {
            top_weights.iter_mut().for_each(|w| *w /= sum);
        }
    }

    (top_indices, top_weights)
}
