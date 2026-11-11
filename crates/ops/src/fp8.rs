//! FP8 E4M3FN quantization for KV cache.
//!
//! Simulated round-trip: quantize f32 → E4M3FN → f32.
//! Values stay f32 in memory but have FP8 precision, matching training semantics.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi as ffi;
use engine_mlx_ffi::MlxCtx;

// E4M3FN representable values: 127 entries (0..126).
// exp_scale[exp] = 2^(exp-7) for exp in 0..15
const EXP_SCALE: [f32; 16] = [
    0.0, 0.015625, 0.03125, 0.0625,
    0.125, 0.25, 0.5, 1.0,
    2.0, 4.0, 8.0, 16.0,
    32.0, 64.0, 128.0, 256.0,
];

/// Get the FP8 E4M3FN representable value for index i (0..126).
#[inline]
fn e4m3fn_value(i: u32) -> f32 {
    let exp = ((i >> 3) & 0x0f) as usize;
    let mant = (i & 0x07) as f32;
    if exp == 0 {
        mant * 0.001953125 // subnormal: mant * 2^(-9)
    } else {
        (1.0 + mant * 0.125) * EXP_SCALE[exp]
    }
}

/// Quantize-dequantize round-trip through E4M3FN with round-to-nearest-even.
#[inline]
pub fn e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);

    // Binary search for largest representable value <= ax
    let mut lo: i32 = 0;
    let mut hi: i32 = 126;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3fn_value(mid as u32) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    // Tie-breaking: prefer even index on ties (round-to-nearest-even)
    let mut best = lo as u32;
    if best < 126 {
        let best_diff = (ax - e4m3fn_value(best)).abs();
        let next_diff = (ax - e4m3fn_value(best + 1)).abs();
        if next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
        {
            best += 1;
        }
    }

    sign * e4m3fn_value(best)
}

/// Apply FP8 E4M3FN round-trip to a slice: quantize then dequantize.
///
/// Non-RoPE portion [0, n_nope): FP8 round-trip
/// RoPE portion [n_nope, head_dim): passthrough
pub fn fp8_round_trip_slice(data: &mut [f32], head_dim: usize, n_rot: usize) {
    tracing::trace!(target: "engine_mlx::fp8", function = "fp8_round_trip_slice", head_dim, n_rot, "ENTER");
    let n_nope = head_dim - n_rot;
    // Process in 64-element blocks (matching Metal kernel)
    for chunk in data.chunks_exact_mut(head_dim) {
        for off in (0..n_nope).step_by(64) {
            let end = (off + 64).min(n_nope);
            // Find amax
            let amax = chunk[off..end]
                .iter()
                .map(|v| v.abs())
                .fold(1.0e-4f32, f32::max);
            let scale = 2.0f32.powf((amax / 448.0).log2().ceil());
            let inv_scale = 1.0 / scale;
            // Quantize-dequantize
            for v in chunk[off..end].iter_mut() {
                let q = e4m3fn_dequant((*v * inv_scale).clamp(-448.0, 448.0));
                *v = q * scale;
            }
        }
        // RoPE portion unchanged (tail of head_dim)
    }
}

/// Apply FP8 E4M3FN round-trip to an MLX array.
///
/// The array is reshaped to [-1, head_dim], each row is processed independently.
/// Returns a new MLX array with the same shape.
pub fn fp8_round_trip(
    ctx: &MlxCtx,
    x: mlx_array,
    head_dim: i32,
    n_rot: i32,
) -> Result<mlx_array> {
    tracing::debug!(target: "engine_mlx::fp8", function = "fp8_round_trip", head_dim, n_rot, "ENTER");
    let hd = head_dim as usize;
    let nr = n_rot as usize;
    let n_nope = hd - nr;

    // Get total elements
    let shape = ctx.shape(x)?;
    let total: usize = shape.iter().map(|&s| s as usize).product();
    let n_rows = total / hd;

    // Extract data
    let mut data = ctx.to_vec_f32(x)?;

    // Process row by row
    for row in 0..n_rows {
        let base = row * hd;
        // Process non-RoPE portion in 64-element blocks
        for off in (0..n_nope).step_by(64) {
            let end = (off + 64).min(n_nope);
            let slice = &mut data[base + off..base + end];
            let amax = slice.iter().map(|v| v.abs()).fold(1.0e-4f32, f32::max);
            let scale = 2.0f32.powf((amax / 448.0).log2().ceil());
            let inv_scale = 1.0 / scale;
            for v in slice.iter_mut() {
                let q = e4m3fn_dequant((*v * inv_scale).clamp(-448.0, 448.0));
                *v = q * scale;
            }
        }
        // RoPE portion unchanged
    }

    // Create new MLX array with same shape
    let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    let out = unsafe {
        ffi::mlx_array_new_data(
            data.as_ptr() as *const std::ffi::c_void,
            shape_i32.as_ptr(),
            shape.len() as i32,
            engine_mlx_ffi::mlx_dtype::MLX_FLOAT32,
        )
    };
    tracing::debug!(target: "engine_mlx::fp8", function = "fp8_round_trip", n_rows, "EXIT");
    Ok(out)
}

/// Apply FP8 E4M3FN to a K or V cache row (stored as f32, FP8-simulated).
/// Used by the cache path to quantize before storage.
pub fn fp8_quantize_cache(data: &mut [f32], head_dim: usize, n_rot: usize) {
    fp8_round_trip_slice(data, head_dim, n_rot);
}


