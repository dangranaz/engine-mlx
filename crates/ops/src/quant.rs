//! Quantized weight group — bundles weight/scales/biases for quantized_matmul.
//!
//! Every model crate stores projection weights as three parallel mlx arrays.
//! `QuantWeights` groups them together so ops accept one struct instead of
//! three separate parameters.

use engine_mlx_ffi::mlx_array;

/// A quantized linear layer (weight + scales + biases + quantization params).
///
/// Each weight carries its own quantization parameters (bits, group_size)
/// to support mixed-precision models (OptiQ-style).
///
/// When `group_size == 0`, the weight is stored as plain f16 (dequantized)
/// and `qmatmul` dispatches to a regular matmul.
#[derive(Clone)]
pub struct QuantWeights {
    pub weight: mlx_array,
    pub scales: mlx_array,
    pub biases: mlx_array,
    pub bits: i32,
    pub group_size: i32,
    /// Quantization mode: "affine" (default) or "mxfp4".
    pub mode: &'static str,
}

impl QuantWeights {
    /// Create quantized weights with default params (4-bit, group 64, affine mode).
    pub fn new(weight: mlx_array, scales: mlx_array, biases: mlx_array) -> Self {
        Self { weight, scales, biases, bits: 4, group_size: 64, mode: "affine" }
    }

    /// Create quantized weights with explicit bit width and group size.
    pub fn with_quant(weight: mlx_array, scales: mlx_array, biases: mlx_array, bits: i32, group_size: i32) -> Self {
        Self { weight, scales, biases, bits, group_size, mode: "affine" }
    }

    /// Create MXFP4 quantized weights. Scales are U8, biases is unused (pass dummy).
    pub fn mxfp4(weight: mlx_array, scales: mlx_array, biases: mlx_array, group_size: i32) -> Self {
        Self { weight, scales, biases, bits: 4, group_size, mode: "mxfp4" }
    }

    /// Create a dequantized (plain f16) weight — dispatches to regular matmul.
    pub fn dequantized(weight: mlx_array) -> Self {
        Self { weight, scales: weight, biases: weight, bits: 0, group_size: 0, mode: "affine" }
    }

    /// Returns true if the weight is stored in quantized format.
    pub fn is_quantized(&self) -> bool { self.group_size > 0 }
}

/// Helper: matmul with a `QuantWeights` bundle.
///
/// Dispatches to `quantized_matmul` or regular `matmul` based on
/// whether the weight is quantized (`group_size > 0`) or dequantized.
#[inline]
pub fn qmatmul(
    ctx: &engine_mlx_ffi::MlxCtx,
    x: mlx_array,
    w: &QuantWeights,
) -> anyhow::Result<mlx_array> {
    if w.is_quantized() {
        if w.mode == "affine" {
            ctx.quantized_matmul(x, w.weight, w.scales, w.biases, true, w.group_size, w.bits)
        } else {
            // For non-affine modes (mxfp4), pass a null/zeroed array as biases
            // since mxfp4 does not use group biases.
            let null_biases: mlx_array = unsafe { std::mem::zeroed() };
            ctx.quantized_matmul_mode(x, w.weight, w.scales, null_biases, true, w.group_size, w.bits, w.mode)
        }
    } else {
        // Weight stored as [out, in] (quantized layout); transpose to [in, out] for plain matmul.
        let weight = ctx.transpose(w.weight)?;
        ctx.matmul(x, weight)
    }
}
