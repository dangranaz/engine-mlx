use anyhow::Result;
use crate::*;

impl super::MlxCtx {
    pub fn rms_norm(&self, x: mlx_array, weight: mlx_array, eps: f32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_fast_rms_norm(&mut out, x, weight, eps, self.stream) }, "rms_norm")?;
        Ok(out)
    }

    pub fn matmul(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_matmul(&mut out, a, b, self.stream) }, "matmul")?;
        Ok(out)
    }

    pub fn add(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_add(&mut out, a, b, self.stream) }, "add")?;
        Ok(out)
    }

    pub fn multiply(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_multiply(&mut out, a, b, self.stream) }, "multiply")?;
        Ok(out)
    }

    /// SiLU (Sigmoid Linear Unit) = x * sigmoid(x).
    /// mlx-c 0.6.0 exposes no fused `mlx_silu` in its C API, so this is composed
    /// from `mlx_sigmoid` + `mlx_multiply` (2 dispatches). Under `mlx_compile`
    /// these fuse into a single Metal kernel at graph-build time anyway.
    pub fn silu(&self, x: mlx_array) -> Result<mlx_array> {
        let mut sig = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_sigmoid(&mut sig, x, self.stream) }, "silu.sigmoid")?;
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_multiply(&mut out, x, sig, self.stream) }, "silu.multiply")?;
        unsafe { mlx_array_free(sig) };
        Ok(out)
    }

    /// Depthwise causal conv1d: groups=channels for per-channel convolution.
    pub fn conv1d(&self, input: mlx_array, weight: mlx_array, stride: i32, padding: i32, dilation: i32, groups: i32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_conv1d(&mut out, input, weight, stride, padding, dilation, groups, self.stream) }, "conv1d")?;
        Ok(out)
    }

    pub fn reshape(&self, x: mlx_array, shape: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_reshape(&mut out, x, shape.as_ptr(), shape.len(), self.stream) }, "reshape")?;
        Ok(out)
    }

    pub fn transpose(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_transpose(&mut out, x, self.stream) }, "transpose")?;
        Ok(out)
    }

    pub fn transpose_axes(&self, x: mlx_array, axes: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_transpose_axes(&mut out, x, axes.as_ptr(), axes.len(), self.stream) }, "transpose_axes")?;
        Ok(out)
    }

    pub fn rope(&self, x: mlx_array, dims: i32, theta: f32, offset: i32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let base = mlx_optional_float_ { value: theta, has_value: true };
        Self::check(unsafe { mlx_fast_rope(&mut out, x, dims, false, base, 1.0, offset, std::mem::zeroed(), self.stream) }, "rope")?;
        Ok(out)
    }

    /// RoPE with array offset (compatible with mlx_compile).
    pub fn rope_dynamic(&self, x: mlx_array, dims: i32, theta: f32, offset: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let base = mlx_optional_float_ { value: theta, has_value: true };
        Self::check(unsafe { mlx_fast_rope_dynamic(&mut out, x, dims, false, base, 1.0, offset, std::mem::zeroed(), self.stream) }, "rope_dynamic")?;
        Ok(out)
    }

    pub fn sdpa(
        &self, q: mlx_array, k: mlx_array, v: mlx_array,
        scale: f32, causal: bool,
    ) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let mode_ptr = if causal { c"causal".as_ptr() } else { c"".as_ptr() };
        Self::check(unsafe {
            mlx_fast_scaled_dot_product_attention(
                &mut out, q, k, v, scale,
                mode_ptr, std::mem::zeroed(), std::mem::zeroed(),
                self.stream,
            )
        }, "sdpa")?;
        Ok(out)
    }
}
