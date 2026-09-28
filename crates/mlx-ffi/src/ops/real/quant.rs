use anyhow::Result;
use crate::*;
use std::ffi::CString;

impl super::MlxCtx {
    /// Quantize KV tensor using native MLX `mlx_quantize` (affine, group-based).
    /// Returns (data, scales, biases) — single fused kernel.
    pub fn quantize_kv_native(
        &self,
        x: mlx_array,
        group_size: i32,
        bits: i32,
    ) -> Result<(mlx_array, mlx_array, mlx_array)> {
        let mut res: mlx_vector_array = unsafe { std::mem::zeroed() };
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: bits, has_value: true };
        let mode = CString::new("affine").map_err(|e| anyhow::anyhow!("mode NUL: {e}"))?;
        let global_scale: mlx_array = unsafe { std::mem::zeroed() };
        Self::check(
            unsafe { mlx_quantize(&mut res, x, gs, b, mode.as_ptr(), global_scale, self.stream) },
            "quantize_kv_native",
        )?;
        let n = unsafe { mlx_vector_array_size(res) };
        if n != 3 {
            unsafe { mlx_vector_array_free(res); }
            anyhow::bail!("mlx_quantize returned {n} arrays, expected 3");
        }
        let mut data: mlx_array = unsafe { std::mem::zeroed() };
        let mut scales: mlx_array = unsafe { std::mem::zeroed() };
        let mut biases: mlx_array = unsafe { std::mem::zeroed() };
        unsafe {
            mlx_vector_array_get(&mut data, res, 0);
            mlx_vector_array_get(&mut scales, res, 1);
            mlx_vector_array_get(&mut biases, res, 2);
            mlx_vector_array_free(res);
        }
        Ok((data, scales, biases))
    }

    /// Dequantize KV tensor using native MLX `mlx_dequantize`.
    pub fn dequantize_kv_native(
        &self,
        data: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        group_size: i32,
        bits: i32,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: bits, has_value: true };
        let mode = CString::new("affine").map_err(|e| anyhow::anyhow!("mode NUL: {e}"))?;
        let global_scale: mlx_array = unsafe { std::mem::zeroed() };
        let dtype = mlx_optional_dtype_ { value: mlx_dtype_::MLX_BFLOAT16, has_value: true };
        Self::check(
            unsafe {
                mlx_dequantize(
                    &mut out, data, scales, biases, gs, b, mode.as_ptr(), global_scale, dtype, self.stream,
                )
            },
            "dequantize_kv_native",
        )?;
        Ok(out)
    }
}
