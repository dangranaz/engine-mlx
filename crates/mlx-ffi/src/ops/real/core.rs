use anyhow::Result;
use crate::*;

impl super::MlxCtx {
    pub fn embed(&self, weights: mlx_array, ids: mlx_array) -> Result<mlx_array> {
        self.take(weights, ids)
    }

    pub fn take(&self, a: mlx_array, indices: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_take_axis(&mut out, a, indices, 0, self.stream) }, "take")?;
        Ok(out)
    }

    /// Create an int32 array from a slice of i32 values.
    pub fn new_array_i32(&self, data: &[i32]) -> Result<mlx_array> {
        let shape = [data.len() as i32];
        let out = unsafe {
            mlx_array_new_data(data.as_ptr() as *const std::ffi::c_void, shape.as_ptr(), shape.len() as i32, mlx_dtype_::MLX_INT32)
        };
        Ok(out)
    }

    pub fn ndim(&self, arr: mlx_array) -> Result<i32> {
        Ok(unsafe { mlx_array_ndim(arr) } as i32)
    }

    pub fn expand_dims(&self, x: mlx_array, axis: i32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_expand_dims(&mut out, x, axis, self.stream) }, "expand_dims")?;
        Ok(out)
    }

    pub fn copy_to_stream(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_copy(&mut out, x, self.stream) }, "copy")?;
        Ok(out)
    }

    pub fn arange(&self, start: f32, stop: f32, step: f32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_arange(&mut out, start as f64, stop as f64, step as f64, mlx_dtype::MLX_FLOAT32, self.stream) }, "arange")?;
        Ok(out)
    }

    pub fn equal_scalar(&self, a: mlx_array, val: f32) -> Result<mlx_array> {
        let b = unsafe { mlx_array_new_float(val) };
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_equal(&mut out, a, b, self.stream) }, "equal_scalar")?;
        Ok(out)
    }

}
