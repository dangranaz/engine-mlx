//! Fused Metal kernel for Gated DeltaNet (GDN) recurrent step.
//!
//! Fuses the 13 separate MLX operations of `gdn_step` into a single Metal
//! kernel dispatch, eliminating intermediate array allocations.
//!
//! The MLX fast kernel API auto-generates the function signature from
//! input/output names and template args. We only provide the body.

use anyhow::Result;
use engine_mlx_ffi as ffi;
use engine_mlx_ffi::MlxCtx;
use std::ffi::CString;

/// Kernel body — Hv, Dv, Dk are template int args injected by MLX.
/// Thread attributes are auto-added to the function signature.
const GDN_STEP_BODY: &str = r#"
    uint dv = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;

    if (dv >= (uint)Dv || h >= (uint)Hv) return;

    const uint state_base = h * Dv * Dk + dv * Dk;
    const uint k_base = h * Dk;  // q/k already expanded to Hv by caller

    // Decay: state_in *= g[h]
    float g_val = g[h];

    float kv_mem = 0.0;
    for (uint d = 0; d < (uint)Dk; d++) {
        float s = state[state_base + d] * g_val;
        kv_mem += s * k[k_base + d];
        state_out[state_base + d] = s;
    }

    // delta = (v - kv_mem) * beta (already sigmoid'd by caller)
    float delta = (v[h * Dv + dv] - kv_mem) * beta[h];

    // Update state: state_out += k * delta
    for (uint d = 0; d < (uint)Dk; d++) {
        state_out[state_base + d] += k[k_base + d] * delta;
    }

    // y = sum(state_out * q)
    float y_val = 0.0;
    for (uint d = 0; d < (uint)Dk; d++) {
        y_val += state_out[state_base + d] * q[k_base + d];
    }
    y_out[h * Dv + dv] = y_val;
"#;

/// Fused Metal kernel GDN recurrent step.
///
/// Fuses 13 MLX operations into a single Metal dispatch. Only used when
/// `cfg.use_metal_kernel && seq_len == 1` (decode step). Falls back to
/// the pure-Rust `gdn_step` for prefill or when the kernel fails.
pub fn gdn_step_metal(
    ctx: &MlxCtx,
    q: ffi::mlx_array,
    k: ffi::mlx_array,
    v: ffi::mlx_array,
    g: ffi::mlx_array,
    beta: ffi::mlx_array,
    state: ffi::mlx_array,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
) -> Result<(ffi::mlx_array, ffi::mlx_array)> {
    // Expand q/k from hk→hv for the kernel
    let repeat = hv / hk;
    let (q_exp, k_exp) = if repeat > 1 {
        let q3 = ctx.reshape(q, &[1, hk, 1, dk])?;
        let q_t = ctx.tile(q3, &[1, 1, repeat, 1])?;
        let q_e = ctx.reshape(q_t, &[hv, dk])?;
        let k3 = ctx.reshape(k, &[1, hk, 1, dk])?;
        let k_t = ctx.tile(k3, &[1, 1, repeat, 1])?;
        let k_e = ctx.reshape(k_t, &[hv, dk])?;
        (q_e, k_e)
    } else {
        (ctx.reshape(q, &[hv, dk])?, ctx.reshape(k, &[hv, dk])?)
    };

    // Pass inputs — keep 2D shapes where possible for correct buffer indexing
    let q_flat = ctx.reshape(q_exp, &[hv, dk])?;
    let k_flat = ctx.reshape(k_exp, &[hv, dk])?;
    let v_flat = ctx.reshape(v, &[hv, dv])?;
    let g_flat = ctx.reshape(g, &[hv])?;
    let beta_flat = ctx.reshape(beta, &[hv])?;
    let state_flat = ctx.reshape(state, &[hv, dv, dk])?; // f32, 3D

    let name = CString::new("gdn_step_fused")?;
    let input_names = {
        let v = unsafe { ffi::mlx_vector_string_new() };
        for n in &["q", "k", "v", "g", "beta", "state"] {
            let s = CString::new(*n).map_err(|e| crate::error::OpsError::Gdn(crate::error::GdnError::NulInName { name: n, source: e }))?;
            unsafe { ffi::mlx_vector_string_append_value(v, s.as_ptr()); }
        }
        v
    };
    let output_names = {
        let v = unsafe { ffi::mlx_vector_string_new() };
        for n in &["y_out", "state_out"] {
            let s = CString::new(*n).map_err(|e| crate::error::OpsError::Gdn(crate::error::GdnError::NulInName { name: n, source: e }))?;
            unsafe { ffi::mlx_vector_string_append_value(v, s.as_ptr()); }
        }
        v
    };
    let header = CString::new("")?;
    let source = CString::new(GDN_STEP_BODY)?;

    let kernel = unsafe {
        ffi::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            false,
            false,
        )
    };

    if unsafe { std::mem::transmute::<ffi::mlx_fast_metal_kernel, *mut std::ffi::c_void>(kernel).is_null() } {
        return Err(crate::error::OpsError::Gdn(crate::error::GdnError::MetalKernelCreate).into());
    }

    let config = unsafe { ffi::mlx_fast_metal_kernel_config_new() };

    let y_out_shape = [hv, dv];
    let state_out_shape = [hv, dv, dk];
    unsafe {
        ffi::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_out_shape.as_ptr(),
            y_out_shape.len(),
            ffi::mlx_dtype::MLX_FLOAT32,
        );
        ffi::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_out_shape.as_ptr(),
            state_out_shape.len(),
            ffi::mlx_dtype::MLX_FLOAT32,
        );

        let dv_groups = dv; // grid = total threads, not threadgroups
        ffi::mlx_fast_metal_kernel_config_set_grid(config, dv_groups, hv, 1);
        ffi::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        // Template int args for compile-time constants
        ffi::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"Hv".as_ptr(), hv);
        ffi::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"Dv".as_ptr(), dv);
        ffi::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"Dk".as_ptr(), dk);
    }

    let inputs = {
        let arrs = [q_flat, k_flat, v_flat, g_flat, beta_flat, state_flat];
        unsafe { ffi::mlx_vector_array_new_data(arrs.as_ptr(), arrs.len()) }
    };

    let mut outputs: ffi::mlx_vector_array = unsafe { std::mem::zeroed() };

    let status = unsafe {
        ffi::mlx_fast_metal_kernel_apply(&mut outputs, kernel, inputs, config, ctx.stream)
    };

    unsafe {
        ffi::mlx_fast_metal_kernel_free(kernel);
        ffi::mlx_fast_metal_kernel_config_free(config);
        ffi::mlx_vector_array_free(inputs);
    }

    if status != 0 {
        return Err(crate::error::OpsError::Gdn(crate::error::GdnError::MetalKernelFailed { status }).into());
    }

    let mut y_result: ffi::mlx_array = unsafe { std::mem::zeroed() };
    let mut state_result: ffi::mlx_array = unsafe { std::mem::zeroed() };
    unsafe {
        ffi::mlx_vector_array_get(&mut y_result, outputs, 0);
        ffi::mlx_vector_array_get(&mut state_result, outputs, 1);
        ffi::mlx_vector_array_free(outputs);
    }

    let y_4d = ctx.reshape(y_result, &[1, 1, hv, dv])?;
    let state_4d = ctx.reshape(state_result, &[1, hv, dv, dk])?;
    ctx.evaluate(state_4d)?;

    Ok((y_4d, state_4d))
}

fn cast_f32(arr: ffi::mlx_array, ctx: &MlxCtx) -> anyhow::Result<ffi::mlx_array> {
    let mut out: ffi::mlx_array = unsafe { std::mem::zeroed() };
    unsafe { ffi::mlx_astype(&mut out, arr, ffi::mlx_dtype::MLX_FLOAT32, ctx.stream); }
    if unsafe { std::mem::transmute::<ffi::mlx_array, *mut std::ffi::c_void>(out).is_null() } { return Ok(arr); }
    Ok(out)
}
