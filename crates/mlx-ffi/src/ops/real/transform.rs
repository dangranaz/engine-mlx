use anyhow::Result;
use crate::*;
use std::ffi::CString;

impl super::MlxCtx {
    /// SDPA with explicit mask array.
    /// mask shape: [1, 1, q_len, kv_len] — additive mask (0 = attend, -inf = ignore)
    pub fn sdpa_masked(
        &self, q: mlx_array, k: mlx_array, v: mlx_array,
        scale: f32, mask: mlx_array,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        // mask_mode must be "array" for the explicit additive `mask` to be used;
        // an empty mode ("") makes MLX ignore the mask entirely (attends to all
        // positions, including zero-padded future slots in the static KV buffer).
        Self::check(unsafe {
            mlx_fast_scaled_dot_product_attention(
                &mut out, q, k, v, scale,
                c"array".as_ptr(), mask, std::mem::zeroed(),
                self.stream,
            )
        }, "sdpa_masked")?;
        Ok(out)
    }

    pub fn tile(&self, x: mlx_array, reps: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_tile(&mut out, x, reps.as_ptr(), reps.len(), self.stream) }, "tile")?;
        Ok(out)
    }

    pub fn astype(&self, x: mlx_array, dtype: mlx_dtype_) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_astype(&mut out, x, dtype, self.stream)
        }, "astype")?;
        Ok(out)
    }

    pub fn dequantize_weight(
        &self, w: mlx_array, scales: mlx_array, biases: mlx_array,
        group_size: i32, bits: i32,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: bits, has_value: true };
        // BF16 to match layernorm weights (avoids F16+BF16->F32 promotion in rms_norm,
        // which doubled activation bandwidth). See dtype_trace test.
        let dtype = mlx_optional_dtype_ { value: mlx_dtype_::MLX_BFLOAT16, has_value: true };
        Self::check(unsafe {
            mlx_dequantize(&mut out, w, scales, biases, gs, b, c"affine".as_ptr(), std::mem::zeroed(), dtype, self.stream)
        }, "dequantize")?;
        Ok(out)
    }

    pub fn dequantize_mxfp4(
        &self, w: mlx_array, scales: mlx_array,
        group_size: i32,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: 4, has_value: true };
        let dtype = mlx_optional_dtype_ { value: mlx_dtype_::MLX_FLOAT16, has_value: true };
        // mxfp4 doesn't use biases — pass a zeroed array
        let null_biases: mlx_array = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_dequantize(&mut out, w, scales, null_biases, gs, b, c"mxfp4".as_ptr(), std::mem::zeroed(), dtype, self.stream)
        }, "dequantize_mxfp4")?;
        Ok(out)
    }

    pub fn quantized_matmul(
        &self, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array,
        transpose: bool, group_size: i32, bits: i32,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: bits, has_value: true };
        Self::check(unsafe {
            mlx_quantized_matmul(&mut out, x, w, scales, biases, transpose, gs, b, c"affine".as_ptr(), self.stream)
        }, "quantized_matmul")?;
        Ok(out)
    }

    /// Quantized matmul with configurable mode (e.g., "affine", "mxfp4").
    /// For MXFP4: scales are U8, biases can be a null/empty array.
    pub fn quantized_matmul_mode(
        &self, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array,
        transpose: bool, group_size: i32, bits: i32, mode: &str,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let mode_c =
            CString::new(mode).map_err(|e| anyhow::anyhow!("mode contains NUL byte: {e}"))?;
        let gs = mlx_optional_int_ { value: group_size, has_value: true };
        let b = mlx_optional_int_ { value: bits, has_value: true };
        Self::check(unsafe {
            mlx_quantized_matmul(&mut out, x, w, scales, biases, transpose, gs, b, mode_c.as_ptr(), self.stream)
        }, "quantized_matmul")?;
        Ok(out)
    }

    pub fn evaluate(&self, arr: mlx_array) -> Result<()> {
        Self::check(unsafe { mlx_array_eval(arr) }, "eval")?;
        Ok(())
    }

    /// Evaluate and synchronize — materializes result on CPU for reading.
    pub fn eval_and_sync(&self, arr: mlx_array) -> Result<()> {
        Self::check(unsafe { mlx_array_eval(arr) }, "eval")?;
        unsafe { mlx_synchronize(self.stream); }
        Ok(())
    }

    /// Number of elements in an array.
    pub fn array_size(&self, arr: mlx_array) -> usize {
        unsafe { mlx_array_size(arr) as usize }
    }

    /// Raw f32 data pointer (must eval_and_sync first).
    ///
    /// # Safety
    /// The caller must ensure the array has been evaluated and synchronized
    /// before reading from this pointer.
    pub unsafe fn array_data_f32(&self, arr: mlx_array) -> *const f32 {
        mlx_array_data_float32(arr)
    }

    /// Force-evaluate a vector of arrays (no sync).
    pub fn eval_all(&self, arrays: &[mlx_array]) -> Result<()> {
        if arrays.is_empty() {
            return Ok(());
        }
        let vec = unsafe { mlx_vector_array_new_data(arrays.as_ptr(), arrays.len()) };
        Self::check(unsafe { mlx_eval(vec) }, "eval_all")?;
        unsafe { mlx_vector_array_free(vec) };
        Ok(())
    }

    /// Async-evaluate a vector of arrays: enqueues GPU work and returns
    /// immediately without synchronizing. The next blocking readback
    /// (`to_vec_*`) is the sync point, so GPU compute overlaps CPU work.
    /// Pipelined decode uses this before reading back the sampled token.
    pub fn async_eval(&self, arrays: &[mlx_array]) -> Result<()> {
        tracing::trace!(target: "engine_mlx::mlx::eval", "async_eval n={}", arrays.len());
        if arrays.is_empty() {
            return Ok(());
        }
        let vec = unsafe { mlx_vector_array_new_data(arrays.as_ptr(), arrays.len()) };
        Self::check(unsafe { mlx_async_eval(vec) }, "async_eval")?;
        unsafe { mlx_vector_array_free(vec) };
        Ok(())
    }
}
