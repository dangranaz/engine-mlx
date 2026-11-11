//! Gated DeltaNet (GDN) — linear recurrent attention.
//!
//! Used by Qwen3.5 (75% of layers), Kimi-Linear, etc.
//! O(1) memory per step — no KV cache growth.
//!
//! Full forward pass lives in `gdn_forward.rs` (split per RULES 300-line limit).
//!
//! ## Recurrent step (per timestep)
//!
//! ```text
//! g = exp(-exp(A_log) * softplus(a + dt_bias))  // decay
//! state = state * g                              // forget
//! kv_mem = (state * k).sum(axis=-1)              // recall
//! delta = (v - kv_mem) * sigmoid(b)              // write signal
//! state = state + k.unsqueeze(-2) * delta.unsqueeze(-1)  // update
//! y = (state * q).sum(axis=-1)                   // read
//! ```
//!
//! State: [B, Hv, Dv, Dk] — fixed size regardless of sequence length.

use anyhow::Result;
use engine_mlx_ffi::{mlx_array, MlxCtx};

use crate::quant::QuantWeights;

/// GDN configuration extracted from model config.
pub struct GdnConfig {
    pub num_key_heads: i32,
    pub num_value_heads: i32,
    pub key_head_dim: i32,
    pub value_head_dim: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
    pub use_metal_kernel: bool,
}

/// Persistent state for a single GDN layer.
pub struct GdnState {
    /// Recurrent state: [1, Hv, Dv, Dk]
    pub h: Option<mlx_array>,
    /// Conv1d buffer: [1, kernel-1, conv_dim] — for causal conv during decode
    pub conv_buf: Option<mlx_array>,
}

impl GdnState {
    pub fn new() -> Self { Self { h: None, conv_buf: None } }
    pub fn reset(&mut self) { self.h = None; self.conv_buf = None; }
}

/// Compute decay gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
///
/// - a: [1, T, Hv]
/// - A_log: [Hv]
/// - dt_bias: [Hv]
/// Returns: g [1, T, Hv]
pub fn compute_gate(
    ctx: &MlxCtx,
    a: mlx_array,
    a_log: mlx_array,
    dt_bias: mlx_array,
) -> Result<mlx_array> {
    // Cast A_log to fp32 for precision (matches Python: A_log.astype(mx.float32))
    let a_log_f32 = cast_f32(a_log, ctx)?;
    let a_biased = ctx.add(a, dt_bias)?;
    let sp = ctx.softplus(a_biased)?;
    let a_exp = ctx.exp(a_log_f32)?;
    let neg_prod = ctx.negative(ctx.multiply(a_exp, sp)?)?;
    ctx.exp(neg_prod)
}

/// Single GDN recurrent step.
///
/// # Arguments
/// * `q` — query `[1, Hv, Dk]`
/// * `k` — key `[1, Hk, Dk]` (will be expanded to Hv heads if needed)
/// * `v` — value `[1, Hv, Dv]`
/// * `g` — decay gate `[1, Hv]`
/// * `beta` — write gate `[1, Hv]`
/// * `state` — recurrent state `[1, Hv, Dv, Dk]`
/// * `hk`, `hv`, `dk`, `dv` — head counts and dims
///
/// Returns: `(y [1, 1, Hv, Dv], new_state [1, Hv, Dv, Dk])`
pub fn gdn_step(
    ctx: &MlxCtx,
    q: mlx_array,
    k: mlx_array,
    v: mlx_array,
    g: mlx_array,
    beta: mlx_array,
    state: mlx_array,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
) -> Result<(mlx_array, mlx_array)> {
    let repeat = hv / hk;

    // Decay: state * g[1, Hv, 1, 1]
    let g_4d = ctx.reshape(g, &[1, hv, 1, 1])?;
    let state = ctx.multiply(state, g_4d)?;

    // Expand k from [1, hk, dk] → [1, hv, dk] (repeat-interleave: each head repeated)
    let k_expanded = if repeat > 1 {
        let k3 = ctx.reshape(k, &[1, hk, dk])?;
        // Repeat each head: [1, hk, dk] → [1, hk, repeat, dk] → [1, hv, dk]
        let k4 = ctx.reshape(k3, &[1, hk, 1, dk])?;
        let k_rep = ctx.tile(k4, &[1, 1, repeat, 1])?;
        ctx.reshape(k_rep, &[1, hv, dk])?
    } else {
        ctx.reshape(k, &[1, hv, dk])?
    };

    // kv_mem = (state * k_expanded).sum(-1)
    let k_sq = ctx.reshape(k_expanded, &[1, hv, 1, dk])?;
    let state_k = ctx.multiply(state, k_sq)?;
    let kv_mem = ctx.sum_axis(state_k, -1, false)?;

    // delta = (v - kv_mem) * beta
    let v_sq = ctx.reshape(v, &[1, hv, dv])?;
    let diff = ctx.subtract(v_sq, kv_mem)?;
    let beta_3d = ctx.reshape(beta, &[1, hv, 1])?;
    let delta = ctx.multiply(diff, beta_3d)?;

    // state += outer(k, delta)
    let delta_4d = ctx.reshape(delta, &[1, hv, dv, 1])?;
    let k_update = ctx.multiply(k_sq, delta_4d)?;
    let new_state = ctx.add(state, k_update)?;

    // Expand q from [1, hk, dk] → [1, hv, dk]
    let q_expanded = if repeat > 1 {
        let q3 = ctx.reshape(q, &[1, hk, dk])?;
        let q4 = ctx.reshape(q3, &[1, hk, 1, dk])?;
        let q_rep = ctx.tile(q4, &[1, 1, repeat, 1])?;
        ctx.reshape(q_rep, &[1, hv, dk])?
    } else {
        ctx.reshape(q, &[1, hv, dk])?
    };

    // y = (new_state * q_expanded).sum(-1)
    let q_sq = ctx.reshape(q_expanded, &[1, hv, 1, dk])?;
    let state_q = ctx.multiply(new_state, q_sq)?;
    let y = ctx.sum_axis(state_q, -1, false)?;
    let y_4d = ctx.reshape(y, &[1, 1, hv, dv])?;

    // Cast y back to fp16 (matches Python: y.astype(q.dtype))
    let y_4d = {
        let mut out: mlx_array = unsafe { std::mem::zeroed() };
        unsafe { engine_mlx_ffi::mlx_astype(&mut out, y_4d, engine_mlx_ffi::mlx_dtype::MLX_FLOAT16, ctx.stream); }
        out
    };

    // Keep state in f32 to prevent precision loss across recurrent steps
    let new_state = cast_f32(new_state, ctx)?;

    Ok((y_4d, new_state))
}

/// Cast array to f32 to keep recurrent state in full precision.
pub(crate) fn cast_f32(arr: mlx_array, ctx: &MlxCtx) -> Result<mlx_array> {
    let mut out: mlx_array = unsafe { std::mem::zeroed() };
    unsafe {
        engine_mlx_ffi::mlx_astype(
            &mut out, arr,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
            ctx.stream,
        );
    }
    if unsafe { std::mem::transmute::<mlx_array, *mut std::ffi::c_void>(out).is_null() } {
        return Ok(arr);
    }
    Ok(out)
}
