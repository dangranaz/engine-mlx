//! GDN forward pass — full layer computation (split from `gdn.rs` per RULES).

use anyhow::{Context, Result};
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;

use crate::gdn::{GdnConfig, GdnState, cast_f32};
use crate::quant::QuantWeights;

/// RMS norm without learnable weight: uses a weight of all-ones.
///
/// Computes `x / sqrt(mean(x^2) + eps)` where mean is over the last dimension.
/// This is equivalent to RMSNorm with a ones weight — used in QK-norm paths.
fn rms_norm_no_weight(ctx: &MlxCtx, x: mlx_array) -> Result<mlx_array> {
    let shape = ctx.shape(x)?;
    let last_dim = *shape.last().context("rms_norm_no_weight: empty shape")?;
    // Create ones weight [last_dim] in f16 (matches norm_w dtype)
    let ones_data: Vec<u16> = vec![0x3C00u16; last_dim as usize]; // 1.0 in fp16
    let ones = unsafe {
        engine_mlx_ffi::mlx_array_new_data(
            ones_data.as_ptr() as *const std::ffi::c_void,
            [last_dim].as_ptr(), 1,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT16,
        )
    };
    if unsafe { std::mem::transmute::<mlx_array, *mut std::ffi::c_void>(ones).is_null() } {
        return Err(crate::error::OpsError::Gdn(crate::error::GdnError::ArrayCreation {
            context: "rms_norm_no_weight: failed to create ones".to_string(),
        }).into());
    }
    ctx.rms_norm(x, ones, 1e-6)
}

/// Full GDN layer forward pass.
///
/// Handles both prefill (sequential scan over T steps) and decode (single step).
///
/// # Arguments
/// * `ctx` — MLX context
/// * `x` — input activations `[1, seq_len, hidden]`
/// * `cfg` — GDN configuration
/// * `state` — mutable GDN recurrent state (updated in place)
/// * `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` — quantized input projections
/// * `a_log` / `dt_bias` — decay gate parameters
/// * `conv1d_w` — depthwise conv1d weights
/// * `norm_w` — RMSNorm weight for output
/// * `out_proj` — quantized output projection
/// * `seq_len` — sequence length
pub fn gdn_forward(
    ctx: &MlxCtx,
    x: mlx_array,
    cfg: &GdnConfig,
    state: &mut GdnState,
    in_proj_qkv: &QuantWeights,
    in_proj_z: &QuantWeights,
    in_proj_a: &QuantWeights,
    in_proj_b: &QuantWeights,
    a_log: mlx_array,
    dt_bias: mlx_array,
    conv1d_w: mlx_array,
    norm_w: mlx_array,
    out_proj: &QuantWeights,
    seq_len: usize,
) -> Result<mlx_array> {
    let hk = cfg.num_key_heads;
    let hv = cfg.num_value_heads;
    let dk = cfg.key_head_dim;
    let dv = cfg.value_head_dim;
    let sl = seq_len as i32;
    let ks = cfg.conv_kernel_size;

    tracing::trace!(target: "engine_mlx::gdn", sl, hk, hv, dk, dv, ks, "gdn_forward start");

    // Project inputs — 4 separate matmuls (batching doesn't help here because
    // quantized_matmul is already a single kernel, and the split overhead negates gains)
    let conv_dim = hk * dk * 2 + hv * dv;
    let qkv = crate::quant::qmatmul(ctx, x, in_proj_qkv)?;
    let z = crate::quant::qmatmul(ctx, x, in_proj_z)?;
    let a_raw = crate::quant::qmatmul(ctx, x, in_proj_a)?;
    let b_raw = crate::quant::qmatmul(ctx, x, in_proj_b)?;

    tracing::trace!(target: "engine_mlx::gdn", "projections done");

    // Causal depthwise conv1d with state buffer
    let qkv_3d = ctx.reshape(qkv, &[1, sl, conv_dim])?;

    let n_keep = ks - 1;
    let conv_input = if let Some(buf) = state.conv_buf {
        ctx.concatenate(buf, qkv_3d, 1)?
    } else {
        let pad = ctx.zeros(&[1, n_keep, conv_dim])?;
        ctx.concatenate(pad, qkv_3d, 1)?
    };

    // Save conv buffer
    let total_len = n_keep + sl;
    let _buf_start = total_len - n_keep;
    state.conv_buf = Some(ctx.slice(
        conv_input, &[0, total_len - n_keep, 0], &[1, total_len, conv_dim], &[1, 1, 1],
    )?);

    // Depthwise conv1d — use native mlx_conv1d (handles kernel flip correctly)
    // Weight shape from file: [conv_dim, kernel_size, 1] — already in MLX conv1d format
    // Input: [1, total_len, conv_dim] → need [1, conv_dim, total_len] for conv1d (NCL format)
    // Actually MLX conv1d expects: input [N, L, C_in], weight [C_out, K, C_in/groups]
    // With groups=conv_dim (depthwise): input [1, total_len, conv_dim], weight [conv_dim, ks, 1]
    let conv_out = ctx.conv1d(conv_input, conv1d_w, 1, 0, 1, conv_dim)?;
    // Output: [1, total_len - ks + 1, conv_dim] = [1, sl, conv_dim]

    let conv_out = ctx.silu(conv_out)?;

    // Split into q, k, v
    let key_dim = hk * dk;
    let conv_2d = ctx.reshape(conv_out, &[sl, conv_dim])?;
    let q_flat = ctx.slice(conv_2d, &[0, 0], &[sl, key_dim], &[1, 1])?;
    let k_flat = ctx.slice(conv_2d, &[0, key_dim], &[sl, key_dim * 2], &[1, 1])?;
    let v_flat = ctx.slice(conv_2d, &[0, key_dim * 2], &[sl, conv_dim], &[1, 1])?;

    // Reshape to head format — q/k keep hk heads, v has hv heads
    let q = ctx.reshape(q_flat, &[1, sl, hk, dk])?;
    let k = ctx.reshape(k_flat, &[1, sl, hk, dk])?;
    let v = ctx.reshape(v_flat, &[1, sl, hv, dv])?;

    // Expand Hk→Hv if needed
    let repeat_factor = hv / hk;
    let (q, k) = if repeat_factor > 1 {
        let q5 = ctx.reshape(q, &[1, sl, hk, 1, dk])?;
        let q_rep = ctx.tile(q5, &[1, 1, 1, repeat_factor, 1])?;
        let q_exp = ctx.reshape(q_rep, &[1, sl, hv, dk])?;
        let k5 = ctx.reshape(k, &[1, sl, hk, 1, dk])?;
        let k_rep = ctx.tile(k5, &[1, 1, 1, repeat_factor, 1])?;
        let k_exp = ctx.reshape(k_rep, &[1, sl, hv, dk])?;
        (q_exp, k_exp)
    } else {
        (q, k)
    };

    // QK norm — CRITICAL for correct output
    // rms_norm(q) * (1/dk), rms_norm(k) * (1/√dk)
    let dk_f = dk as f32;
    let q_scale_val = 1.0f32 / dk_f;
    let k_scale_val = 1.0f32 / dk_f.sqrt();
    let q_scale = unsafe {
        engine_mlx_ffi::mlx_array_new_data(
            [q_scale_val].as_ptr() as *const std::ffi::c_void,
            [1].as_ptr(), 1,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
        )
    };
    let k_scale = unsafe {
        engine_mlx_ffi::mlx_array_new_data(
            [k_scale_val].as_ptr() as *const std::ffi::c_void,
            [1].as_ptr(), 1,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
        )
    };
    let q = ctx.multiply(rms_norm_no_weight(ctx, q)?, q_scale)?;
    let k = ctx.multiply(rms_norm_no_weight(ctx, k)?, k_scale)?;

    // Compute gates
    let a_3d = ctx.reshape(a_raw, &[1, sl, hv])?;
    let b_3d = ctx.reshape(b_raw, &[1, sl, hv])?;
    let g = crate::gdn::compute_gate(ctx, a_3d, a_log, dt_bias)?;
    let beta = ctx.sigmoid(b_3d)?;

    tracing::trace!(target: "engine_mlx::gdn", "gates computed, starting recurrent scan");

    // Init state
    if state.h.is_none() {
        state.h = Some(ctx.zeros_f32(&[1, hv, dv, dk])?);
    }
    let mut h = state.h.context("GDN state uninitialized after init")?;

    // Sequential scan
    let mut outputs: Vec<mlx_array> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let ti = t as i32;
        // After GQA expand, q and k have hv heads (not hk!)
        let q_t = ctx.slice(q, &[0, ti, 0, 0], &[1, ti+1, hv, dk], &[1,1,1,1])?;
        let k_t = ctx.slice(k, &[0, ti, 0, 0], &[1, ti+1, hv, dk], &[1,1,1,1])?;
        let v_t = ctx.slice(v, &[0, ti, 0, 0], &[1, ti+1, hv, dv], &[1,1,1,1])?;
        let g_t = ctx.slice(g, &[0, ti, 0], &[1, ti+1, hv], &[1,1,1])?;
        let beta_t = ctx.slice(beta, &[0, ti, 0], &[1, ti+1, hv], &[1,1,1])?;

        let (y_t, new_h) = if cfg.use_metal_kernel && seq_len == 1 {
            crate::gdn_metal::gdn_step_metal(ctx, q_t, k_t, v_t, g_t, beta_t, h, hv, hv, dk, dv)?
        } else {
            crate::gdn::gdn_step(ctx, q_t, k_t, v_t, g_t, beta_t, h, hv, hv, dk, dv)?
        };
        h = new_h;
        outputs.push(y_t);
    }
    state.h = Some(h);

    tracing::trace!(target: "engine_mlx::gdn", "recurrent scan done, output projection");

    // Concatenate outputs
    let out = if outputs.len() == 1 {
        outputs[0]
    } else {
        let mut acc = outputs[0];
        for i in 1..outputs.len() {
            acc = ctx.concatenate(acc, outputs[i], 1)?;
        }
        acc
    };

    // RMSNormGated: norm(out, weight) * silu(z)
    let z_4d = ctx.reshape(z, &[1, sl, hv, dv])?;
    // Precise SwiGLU: cast to f32 before multiply for numerical stability
    let z_act = ctx.silu(z_4d)?;
    let z_act_f32 = cast_f32(z_act, ctx)?;
    let out_normed = ctx.rms_norm(out, norm_w, cfg.rms_norm_eps)?;
    let out_normed_f32 = cast_f32(out_normed, ctx)?;
    let gated_out = ctx.multiply(out_normed_f32, z_act_f32)?;

    let flat = ctx.reshape(gated_out, &[sl, hv * dv])?;
    crate::quant::qmatmul(ctx, flat, out_proj)
}
