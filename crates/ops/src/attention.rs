//! GQA (Grouped-Query Attention) building blocks.
//!
//! Every decoder-only model here uses the same GQA pattern:
//!
//! 1. **QKV projection** — quantized_matmul for Q, K, V separately
//! 2. **Reshape** — `[1, seq_len, hidden]` → `[1, seq_len, n_heads, head_dim]`
//! 3. **QK-norm** (optional) — per-head RMS norm on Q and K
//! 4. **Transpose** — `[1, seq_len, head, dim]` → `[1, head, seq_len, dim]`
//! 5. **Position encoding** — RoPE or NoPE (see `rope` module)
//! 6. **KV cache update** — raw concatenation or quantized backend
//! 7. **SDPA** — `mlx.sdpa(q, k, v, scale, is_prefill)`
//! 8. **Transpose back** — `[1, head, seq, dim]` → `[1, seq, head, dim]`
//! 9. **Output projection** — quantized_matmul with O weight
//!
//! The functions in this module are designed to be composed in model crates,
//! not as a monolithic "do everything" function. This keeps each step testable
//! and allows model crates to insert model-specific logic between them
//! (e.g., SmolLM3's conditional NoPE/RoPE, Qwen3's sliding window slice).

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi as ffi;
use engine_mlx_ffi::MlxCtx;
use crate::kv_traits::{KvCacheBackend, TensorRef};

use crate::quant::{qmatmul, QuantWeights};

/// Project Q, K, V from input `x` via quantized matmuls + reshape.
///
/// Returns `(q, k, v)` each shaped `[1, seq_len, n_heads, head_dim]` (before transpose).
pub fn project_qkv(
    ctx: &MlxCtx,
    x: mlx_array,
    q_proj: &QuantWeights,
    k_proj: &QuantWeights,
    v_proj: &QuantWeights,
    nh: i32,
    nkv: i32,
    hd: i32,
    seq_len: i32,
) -> Result<(mlx_array, mlx_array, mlx_array)> {
    let q = qmatmul(ctx, x, q_proj)?;
    let k = qmatmul(ctx, x, k_proj)?;
    let v = qmatmul(ctx, x, v_proj)?;

    let q = ctx.reshape(q, &[1, seq_len, nh, hd])?;
    let k = ctx.reshape(k, &[1, seq_len, nkv, hd])?;
    let v = ctx.reshape(v, &[1, seq_len, nkv, hd])?;

    Ok((q, k, v))
}

/// Apply per-head QK-norm (RMS norm) — no-op if both norms are `None`.
///
/// Qwen3 stores `q_norm.weight` and `k_norm.weight` per layer. Other models
/// (SmolLM3, Llama) do not have QK-norm — pass `None` for both.
pub fn optional_qk_norm(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    q_norm: Option<mlx_array>,
    k_norm: Option<mlx_array>,
    eps: f32,
) -> Result<(mlx_array, mlx_array)> {
    let q = if let Some(qn) = q_norm {
        ctx.rms_norm(q, qn, eps)?
    } else {
        q
    };
    let k = if let Some(kn) = k_norm {
        ctx.rms_norm(k, kn, eps)?
    } else {
        k
    };
    Ok((q, k))
}

/// Transpose `[1, seq_len, n_heads, head_dim]` → `[1, n_heads, seq_len, head_dim]`.
///
/// This is the standard attention axis order expected by `mlx.sdpa`.
pub fn transpose_for_attn(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    v: mlx_array,
) -> Result<(mlx_array, mlx_array, mlx_array)> {
    Ok((
        ctx.transpose_axes(q, &[0, 2, 1, 3])?,
        ctx.transpose_axes(k, &[0, 2, 1, 3])?,
        ctx.transpose_axes(v, &[0, 2, 1, 3])?,
    ))
}

/// Update KV cache via raw concatenation on the MLX graph.
///
/// On the first call (cache is `None`) this returns `(k, v)` directly.
/// On subsequent calls it concatenates along the sequence dimension (axis 2).
pub fn kv_cache_concat(
    ctx: &MlxCtx,
    cache_k: Option<mlx_array>,
    cache_v: Option<mlx_array>,
    k: mlx_array,
    v: mlx_array,
) -> Result<(mlx_array, mlx_array)> {
    match (cache_k, cache_v) {
        (Some(ck), Some(cv)) => {
            let fk = ctx.concatenate(ck, k, 2)?;
            let fv = ctx.concatenate(cv, v, 2)?;
            Ok((fk, fv))
        }
        _ => Ok((k, v)),
    }
}

/// Update KV cache via a quantized backend (PlanarQuant, etc.).
///
/// This is the **all-quant** path: K/V are encoded on append and decoded
/// to f32 on read, so the SDPA still sees full-precision values but the
/// stored cache is compressed.
pub fn kv_cache_quant<B: KvCacheBackend + ?Sized>(
    backend: &mut B,
    ctx: &MlxCtx,
    k: mlx_array,
    v: mlx_array,
    nkv: i32,
    hd: i32,
) -> Result<(mlx_array, mlx_array)> {
    let k_host = ctx.to_vec_f32(k)?;
    let v_host = ctx.to_vec_f32(v)?;
    backend.append(TensorRef::F32(k_host), TensorRef::F32(v_host))?;

    let (dec_k, dec_v) = backend.decode_to_f32()?;
    let k_per_step = (nkv * hd) as usize;
    let total_seq = dec_k.len() / k_per_step;

    let fk = unsafe {
        ffi::mlx_array_new_data(
            dec_k.as_ptr() as *const std::ffi::c_void,
            [1, nkv, total_seq as i32, hd].as_ptr(),
            4,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
        )
    };
    let fv = unsafe {
        ffi::mlx_array_new_data(
            dec_v.as_ptr() as *const std::ffi::c_void,
            [1, nkv, total_seq as i32, hd].as_ptr(),
            4,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
        )
    };
    Ok((fk, fv))
}

/// Run SDPA and project the output.
///
/// Steps:
/// 1. `ctx.sdpa(q, k, v, scale, is_prefill)` — flash attention or vanilla
/// 2. Transpose back to `[1, seq_len, n_heads, head_dim]`
/// 3. Reshape to `[1, seq_len, nh * hd]`
/// 4. Quantized matmul with O projection
pub fn attend_and_project(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    v: mlx_array,
    o_proj: &QuantWeights,
    scale: f32,
    is_prefill: bool,
    nh: i32,
    hd: i32,
) -> Result<mlx_array> {
    let attn = ctx.sdpa(q, k, v, scale, is_prefill)?;
    let attn = ctx.transpose_axes(attn, &[0, 2, 1, 3])?;
    let attn = ctx.reshape(attn, &[1, -1, nh * hd])?;
    qmatmul(ctx, attn, o_proj)
}
