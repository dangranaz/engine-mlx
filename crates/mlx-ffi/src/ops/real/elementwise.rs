use anyhow::Result;
use crate::*;

impl super::MlxCtx {
    pub fn subtract(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_subtract(&mut out, a, b, self.stream) }, "subtract")?;
        Ok(out)
    }

    /// Element-wise remainder (modulo).
    pub fn remainder(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::remainder");
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_remainder(&mut out, a, b, self.stream) }, "remainder")?;
        Ok(out)
    }

    /// Create an int32 array filled with a single value.
    pub fn full_i32(&self, shape: &[i32], value: i32) -> Result<mlx_array> {
        // Create a scalar float array, then cast to int32 via mlx_full
        let val_arr = self.new_array(&[value as f32])?;
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_full(&mut out, shape.as_ptr(), shape.len(),
                val_arr, mlx_dtype_::MLX_INT32, self.stream)
        }, "full_i32")?;
        unsafe { mlx_array_free(val_arr); }
        Ok(out)
    }

    /// Slice with dynamic start position (lazy, works in compiled graphs).
    /// `start`: position array, `axes`: which axes it applies to, `size`: slice size per axis.
    pub fn slice_dynamic(&self, a: mlx_array, start: mlx_array, axes: &[i32], size: &[i32]) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::slice_dynamic axes={:?} size={:?}", axes, size);
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_slice_dynamic(&mut out, a, start,
                axes.as_ptr(), axes.len(),
                size.as_ptr(), size.len(),
                self.stream)
        }, "slice_dynamic")?;
        Ok(out)
    }

    pub fn log1p(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_log1p(&mut out, x, self.stream) }, "log1p")?;
        Ok(out)
    }

    /// softplus(x) = log(1 + exp(x))
    pub fn softplus(&self, x: mlx_array) -> Result<mlx_array> {
        let ex = self.exp(x)?;
        self.log1p(ex)
    }

    /// Softmax along the last axis.
    pub fn softmax(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let ndim = unsafe { mlx_array_ndim(x) };
        let axis = if ndim > 0 { ndim - 1 } else { 0 };
        Self::check(unsafe { mlx_softmax_axis(&mut out, x, axis as i32, false, self.stream) }, "softmax")?;
        Ok(out)
    }

    /// Softmax along a specific axis.
    pub fn softmax_axis(&self, x: mlx_array, axis: i32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_softmax_axis(&mut out, x, axis, false, self.stream) }, "softmax_axis")?;
        Ok(out)
    }

    /// Sum along a single axis. If keep_dims is false, the axis is squeezed.
    pub fn sum_axis(&self, x: mlx_array, axis: i32, keep_dims: bool) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_sum_axis(&mut out, x, axis, keep_dims, self.stream)
        }, "sum_axis")?;
        Ok(out)
    }

    /// Absolute value elementwise.
    pub fn abs(&self, x: mlx_array) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::abs");
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_abs(&mut out, x, self.stream) }, "abs")?;
        Ok(out)
    }

    /// Call bridge_forward_step FFI function.
    /// One FFI call handles embedding lookup, transformer forward, KV cache update + LM head.
    pub fn bridge_forward(
        &self,
        token_id: i32,
        offset: i32,
        kv_keys: &mut [mlx_array],
        kv_values: &mut [mlx_array],
        rswa_window: i32,
    ) -> Result<mlx_array> {
        // Load weights from model (call from caller)
        // This is a stub — the actual implementation needs weight pointers
        let _ = (token_id, offset, kv_keys, kv_values, rswa_window);
        anyhow::bail!("bridge_forward needs weight pointers — not yet integrated")
    }

    /// Max reduction (all axes, squeeze dims).
    pub fn max(&self, x: mlx_array) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::max");
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_max(&mut out, x, false, self.stream) }, "max")?;
        Ok(out)
    }
}
