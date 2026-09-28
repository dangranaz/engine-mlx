use anyhow::Result;
use crate::*;

impl super::MlxCtx {
    /// Argmax: returns index of maximum value along last axis.
    pub fn argmax(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_argmax(&mut out, x, false, self.stream) }, "argmax")?;
        Ok(out)
    }

    /// Elementwise division: a / b.
    pub fn divide(&self, a: mlx_array, b: mlx_array) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::divide");
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_divide(&mut out, a, b, self.stream) }, "divide")?;
        Ok(out)
    }

    /// Elementwise round to nearest integer (decimals=0).
    pub fn round(&self, x: mlx_array) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::round");
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_round(&mut out, x, 0, self.stream) }, "round")?;
        Ok(out)
    }

    /// Create an array from a slice of f32 values.
    pub fn new_array(&self, data: &[f32]) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::new_array len={}", data.len());
        let shape = [data.len() as i32];
        let out = unsafe {
            mlx_array_new_data(data.as_ptr() as *const std::ffi::c_void, shape.as_ptr(), shape.len() as i32, mlx_dtype_::MLX_FLOAT32)
        };
        Ok(out)
    }

    // ── End elementwise math ops ────────────────────────────────────────

    /// Create a zeros array with given shape and dtype.
    pub fn zeros(&self, shape: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_zeros(&mut out, shape.as_ptr(), shape.len(), mlx_dtype_::MLX_FLOAT16, self.stream)
        }, "zeros")?;
        Ok(out)
    }

    /// Create zeros in f32 (for GDN recurrent state that needs f32 precision).
    pub fn zeros_f32(&self, shape: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_zeros(&mut out, shape.as_ptr(), shape.len(), mlx_dtype_::MLX_FLOAT32, self.stream)
        }, "zeros_f32")?;
        Ok(out)
    }

    /// Create zeros matching the dtype of an existing array.
    pub fn zeros_typed(&self, shape: &[i32], like: mlx_array) -> Result<mlx_array> {
        let dtype = unsafe { mlx_array_dtype(like) };
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_zeros(&mut out, shape.as_ptr(), shape.len(), dtype, self.stream)
        }, "zeros_typed")?;
        Ok(out)
    }

    /// Create a full array with given value and shape.
    pub fn full_f32(&self, shape: &[i32], value: f32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let val = unsafe { mlx_array_new_float(value) };
        Self::check(unsafe {
            mlx_full(&mut out, shape.as_ptr(), shape.len(), val, mlx_dtype_::MLX_FLOAT32, self.stream)
        }, "full_f32")?;
        Ok(out)
    }

    /// Create zeros with UINT32 dtype (for int4 packed KV data).
    pub fn zeros_u32(&self, shape: &[i32]) -> Result<mlx_array> {
        tracing::trace!(target: "engine_mlx::mlx::array", "zeros_u32 shape={:?}", shape);
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_zeros(&mut out, shape.as_ptr(), shape.len(), mlx_dtype_::MLX_UINT32, self.stream)
        }, "zeros_u32")?;
        Ok(out)
    }

    /// Create zeros with BF16 dtype (for scales/biases).
    pub fn zeros_bf16(&self, shape: &[i32]) -> Result<mlx_array> {
        tracing::trace!(target: "engine_mlx::mlx::array", "zeros_bf16 shape={:?}", shape);
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_zeros(&mut out, shape.as_ptr(), shape.len(), mlx_dtype_::MLX_BFLOAT16, self.stream)
        }, "zeros_bf16")?;
        Ok(out)
    }

    pub fn to_vec_u32(&self, arr: mlx_array) -> Result<Vec<u32>> {
        unsafe {
            let result = mlx_array_eval(arr);
            if result != 0 {
                anyhow::bail!("mlx_array_eval failed with code {}", result);
            }
            mlx_synchronize(self.stream);
            let size = mlx_array_size(arr) as usize;
            let ptr = mlx_array_data_uint32(arr);
            if ptr.is_null() {
                anyhow::bail!("Failed to get u32 data pointer (size={})", size);
            }
            let slice = std::slice::from_raw_parts(ptr, size);
            Ok(slice.to_vec())
        }
    }

    pub fn to_vec_f16_as_f32(&self, arr: mlx_array) -> Result<Vec<f32>> {
        unsafe {
            let result = mlx_array_eval(arr);
            if result != 0 {
                anyhow::bail!("mlx_array_eval failed with code {}", result);
            }
            mlx_synchronize(self.stream);
            let size = mlx_array_size(arr) as usize;
            let ptr = mlx_array_data_uint16(arr);
            if ptr.is_null() {
                anyhow::bail!("Failed to get f16 data pointer (size={})", size);
            }
            let slice = std::slice::from_raw_parts(ptr, size);
            let f32s: Vec<f32> = slice.iter().map(|&h| {
                let sign = ((h >> 15) as u32) << 31;
                let exp = (h >> 10) & 0x1F;
                let mant = h & 0x3FF;
                let f32_bits = if exp == 0 {
                    if mant == 0 { sign } else { sign | (((mant as u32) << 13) + (112 << 23)) }
                } else if exp == 31 {
                    sign | 0x7F800000 | ((mant as u32) << 13)
                } else {
                    sign | (((exp as u32) + 112) << 12 << 11) | ((mant as u32) << 13)
                };
                f32::from_bits(f32_bits)
            }).collect();
            Ok(f32s)
        }
    }

    pub fn to_vec_bf16_as_f32(&self, arr: mlx_array) -> Result<Vec<f32>> {
        unsafe {
            let result = mlx_array_eval(arr);
            if result != 0 {
                anyhow::bail!("mlx_array_eval failed with code {}", result);
            }
            mlx_synchronize(self.stream);
            let size = mlx_array_size(arr) as usize;
            let ptr = mlx_array_data_uint16(arr);
            if ptr.is_null() {
                anyhow::bail!("Failed to get bf16 data pointer (size={})", size);
            }
            let slice = std::slice::from_raw_parts(ptr, size);
            let f32s: Vec<f32> = slice.iter().map(|&h| {
                // bf16: 1 sign + 8 exponent (bias 127) + 7 mantissa -> F32 1-8-23
                let sign = ((h >> 15) as u32) << 31;
                let exp = (h >> 7) & 0xFF;
                let mant = h & 0x7F;
                let f32_bits = if exp == 0 {
                    if mant == 0 { sign } else { sign | ((mant as u32) << 16) }
                } else if exp == 0xFF {
                    sign | 0x7F800000 | ((mant as u32) << 16)
                } else {
                    sign | ((exp as u32) << 23) | ((mant as u32) << 16)
                };
                f32::from_bits(f32_bits)
            }).collect();
            Ok(f32s)
        }
    }

    pub fn to_vec_f32(&self, arr: mlx_array) -> Result<Vec<f32>> {
        unsafe {
            let result = mlx_array_eval(arr);
            if result != 0 {
                anyhow::bail!("mlx_array_eval failed with code {}", result);
            }
            mlx_synchronize(self.stream);
            let size = mlx_array_size(arr) as usize;
            let ptr = mlx_array_data_float32(arr);
            if ptr.is_null() {
                anyhow::bail!("Failed to get f32 data pointer (size={})", size);
            }
            let slice = std::slice::from_raw_parts(ptr, size);
            Ok(slice.to_vec())
        }
    }
}
