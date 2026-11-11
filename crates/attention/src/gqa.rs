//! GQA (Grouped-Query Attention) — the standard attention for decoder-only models.
//!
//! Full pipeline:
//! 1. QKV projection (quantized matmul)
//! 2. Reshape to multi-head
//! 3. Optional QK-norm (per-head RMS norm)
//! 4. Transpose for attention axis order
//! 5. RoPE position encoding
//! 6. KV cache update
//! 7. SDPA (scaled dot-product attention)
//! 8. Output projection
//!
//! Generic over `KvCache` — the cache backend is injected by the caller.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let output = gqa_attention(ctx, x, &weights, &mut cache, &config)?;
//! ```

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use engine_mlx_kvcache::KvCache;
use engine_mlx_ops::quant::QuantWeights;

/// GQA attention configuration.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of KV heads (< n_heads for grouped-query).
    pub n_kv_heads: i32,
    /// Dimension per head.
    pub head_dim: i32,
    /// RoPE base theta.
    pub rope_theta: f32,
    /// RMS norm epsilon (for QK-norm).
    pub norm_eps: f32,
    /// Whether to apply RoPE (false for NoPE layers like SmolLM3).
    pub use_rope: bool,
}

/// Weights for one GQA attention layer.
pub struct GqaWeights {
    pub q_proj: QuantWeights,
    pub k_proj: QuantWeights,
    pub v_proj: QuantWeights,
    pub o_proj: QuantWeights,
    /// Optional per-head QK-norm weights (Qwen3).
    pub q_norm: Option<mlx_array>,
    pub k_norm: Option<mlx_array>,
}

/// Full GQA attention forward pass, generic over cache backend.
///
/// - `x`: input hidden state `[1, seq_len, hidden_size]`
/// - `cache`: mutable KV cache (any implementation of `KvCache`)
/// - `offset`: position offset for RoPE (cumulative tokens generated)
///
/// Returns: `[1, seq_len, hidden_size]` — attention output
pub fn gqa_attention<C: KvCache>(
    ctx: &MlxCtx,
    x: mlx_array,
    weights: &GqaWeights,
    cache: &mut C,
    config: &GqaConfig,
    offset: i32,
) -> Result<mlx_array> {
    let nh = config.n_heads;
    let nkv = config.n_kv_heads;
    let hd = config.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    // Get sequence length from input shape
    let x_shape = ctx.shape(x)?;
    let seq_len = x_shape[1];
    let is_prefill = seq_len > 1;

    // 1. QKV projection + reshape
    let (q, k, v) = engine_mlx_ops::attention::project_qkv(
        ctx, x, &weights.q_proj, &weights.k_proj, &weights.v_proj,
        nh, nkv, hd, seq_len,
    )?;

    // 2. Optional QK-norm
    let (q, k) = engine_mlx_ops::attention::optional_qk_norm(
        ctx, q, k, weights.q_norm, weights.k_norm, config.norm_eps,
    )?;

    // 3. Transpose: [1, seq, heads, dim] → [1, heads, seq, dim]
    let (q, k, v) = engine_mlx_ops::attention::transpose_for_attn(ctx, q, k, v)?;

    // 4. RoPE
    let (q, k) = engine_mlx_ops::rope::apply_rope(
        ctx, q, k, hd, config.rope_theta, offset, config.use_rope,
    )?;

    // 5. KV cache update
    cache.append(ctx, k, v)?;
    let (full_k, full_v) = cache.get(ctx)?;

    // 6. SDPA + output projection
    engine_mlx_ops::attention::attend_and_project(
        ctx, q, full_k, full_v, &weights.o_proj, scale, is_prefill, nh, hd,
    )
}
