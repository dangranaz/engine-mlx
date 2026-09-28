use anyhow::Result;
use crate::*;
pub struct MlxCtx { pub stream: mlx_stream, }
impl MlxCtx {
    pub fn new(stream: mlx_stream) -> Self { Self { stream } }
    pub fn cpu() -> Self { let stream = unsafe { mlx_default_cpu_stream_new() }; Self { stream } }
    pub fn gpu() -> Self { let stream = unsafe { mlx_default_gpu_stream_new() }; Self { stream } }
    pub fn check(val: i32, op: &str) -> Result<()> { if val != 0 { anyhow::bail!("MLX op '{}' failed with code {}", op, val); } Ok(()) }
    pub fn embed(&self, _w: mlx_array, _i: mlx_array) -> Result<mlx_array> { anyhow::bail!("embed not implemented - needs mlx-c FFI") }
    pub fn take(&self, _a: mlx_array, _i: mlx_array) -> Result<mlx_array> { anyhow::bail!("take not implemented - needs mlx-c FFI") }
    pub fn new_array_i32(&self, _d: &[i32]) -> Result<mlx_array> { anyhow::bail!("new_array_i32 not implemented - needs mlx-c FFI") }
    pub fn ndim(&self, _a: mlx_array) -> Result<i32> { anyhow::bail!("ndim not implemented - needs mlx-c FFI") }
    pub fn expand_dims(&self, _x: mlx_array, _a: i32) -> Result<mlx_array> { anyhow::bail!("expand_dims not implemented - needs mlx-c FFI") }
    pub fn copy_to_stream(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("copy_to_stream not implemented - needs mlx-c FFI") }
    pub fn arange(&self, _s: f32, _t: f32, _st: f32) -> Result<mlx_array> { anyhow::bail!("arange not implemented - needs mlx-c FFI") }
    pub fn equal_scalar(&self, _a: mlx_array, _v: f32) -> Result<mlx_array> { anyhow::bail!("equal_scalar not implemented - needs mlx-c FFI") }
    pub fn rms_norm(&self, _x: mlx_array, _w: mlx_array, _e: f32) -> Result<mlx_array> { anyhow::bail!("rms_norm not implemented - needs mlx-c FFI") }
    pub fn matmul(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("matmul not implemented - needs mlx-c FFI") }
    pub fn add(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("add not implemented - needs mlx-c FFI") }
    pub fn multiply(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("multiply not implemented - needs mlx-c FFI") }
    pub fn silu(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("silu not implemented - needs mlx-c FFI") }
    pub fn conv1d(&self, _i: mlx_array, _w: mlx_array, _s: i32, _p: i32, _d: i32, _g: i32) -> Result<mlx_array> { anyhow::bail!("conv1d not implemented - needs mlx-c FFI") }
    pub fn reshape(&self, _x: mlx_array, _s: &[i32]) -> Result<mlx_array> { anyhow::bail!("reshape not implemented - needs mlx-c FFI") }
    pub fn transpose(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("transpose not implemented - needs mlx-c FFI") }
    pub fn transpose_axes(&self, _x: mlx_array, _a: &[i32]) -> Result<mlx_array> { anyhow::bail!("transpose_axes not implemented - needs mlx-c FFI") }
    pub fn rope(&self, _x: mlx_array, _d: i32, _t: f32, _o: i32) -> Result<mlx_array> { anyhow::bail!("rope not implemented - needs mlx-c FFI") }
    pub fn rope_dynamic(&self, _x: mlx_array, _d: i32, _t: f32, _o: mlx_array) -> Result<mlx_array> { anyhow::bail!("rope_dynamic not implemented - needs mlx-c FFI") }
    pub fn sdpa(&self, _q: mlx_array, _k: mlx_array, _v: mlx_array, _s: f32, _c: bool) -> Result<mlx_array> { anyhow::bail!("sdpa not implemented - needs mlx-c FFI") }
    pub fn sdpa_masked(&self, _q: mlx_array, _k: mlx_array, _v: mlx_array, _s: f32, _m: mlx_array) -> Result<mlx_array> { anyhow::bail!("sdpa_masked not implemented - needs mlx-c FFI") }
    pub fn tile(&self, _x: mlx_array, _r: &[i32]) -> Result<mlx_array> { anyhow::bail!("tile not implemented - needs mlx-c FFI") }
    pub fn astype(&self, _x: mlx_array, _d: u32) -> Result<mlx_array> { anyhow::bail!("astype not implemented - needs mlx-c FFI") }
    pub fn dequantize_weight(&self, _w: mlx_array, _s: mlx_array, _b: mlx_array, _g: i32, _bi: i32) -> Result<mlx_array> { anyhow::bail!("dequantize_weight not implemented - needs mlx-c FFI") }
    pub fn dequantize_mxfp4(&self, _w: mlx_array, _s: mlx_array, _g: i32) -> Result<mlx_array> { anyhow::bail!("dequantize_mxfp4 not implemented - needs mlx-c FFI") }
    pub fn quantized_matmul(&self, _x: mlx_array, _w: mlx_array, _s: mlx_array, _b: mlx_array, _t: bool, _g: i32, _bi: i32) -> Result<mlx_array> { anyhow::bail!("quantized_matmul not implemented - needs mlx-c FFI") }
    pub fn quantized_matmul_mode(&self, _x: mlx_array, _w: mlx_array, _s: mlx_array, _b: mlx_array, _t: bool, _g: i32, _bi: i32, _m: &str) -> Result<mlx_array> { anyhow::bail!("quantized_matmul_mode not implemented - needs mlx-c FFI") }
    pub fn evaluate(&self, _a: mlx_array) -> Result<()> { anyhow::bail!("evaluate not implemented - needs mlx-c FFI") }
    pub fn eval_and_sync(&self, _a: mlx_array) -> Result<()> { anyhow::bail!("eval_and_sync not implemented - needs mlx-c FFI") }
    pub fn array_size(&self, _a: mlx_array) -> usize { 0 }
    pub unsafe fn array_data_f32(&self, _a: mlx_array) -> *const f32 { std::ptr::null() }
    pub fn eval_all(&self, _a: &[mlx_array]) -> Result<()> { anyhow::bail!("eval_all not implemented - needs mlx-c FFI") }
    pub fn async_eval(&self, _a: &[mlx_array]) -> Result<()> { anyhow::bail!("async_eval not implemented - needs mlx-c FFI") }
    pub fn eval_all_and_sync(&self, _a: &[mlx_array]) -> Result<()> { anyhow::bail!("eval_all_and_sync not implemented - needs mlx-c FFI") }
    pub fn materialize(&self, _a: mlx_array) -> Result<mlx_array> { anyhow::bail!("materialize not implemented - needs mlx-c FFI") }
    pub fn concatenate(&self, _a: mlx_array, _b: mlx_array, _ax: i32) -> Result<mlx_array> { anyhow::bail!("concatenate not implemented - needs mlx-c FFI") }
    pub fn split_sections(&self, _a: mlx_array, _s: &[i32], _ax: i32) -> Result<Vec<mlx_array>> { anyhow::bail!("split_sections not implemented - needs mlx-c FFI") }
    pub fn stack(&self, _a: &[mlx_array], _ax: i32) -> Result<mlx_array> { anyhow::bail!("stack not implemented - needs mlx-c FFI") }
    pub fn slice(&self, _a: mlx_array, _s: &[i32], _t: &[i32], _st: &[i32]) -> Result<mlx_array> { anyhow::bail!("slice not implemented - needs mlx-c FFI") }
    pub fn slice_update(&self, _s: mlx_array, _u: mlx_array, _st: &[i32], _t: &[i32], _st2: &[i32]) -> Result<mlx_array> { anyhow::bail!("slice_update not implemented - needs mlx-c FFI") }
    pub fn slice_update_dynamic(&self, _s: mlx_array, _u: mlx_array, _st: mlx_array, _a: &[i32]) -> Result<mlx_array> { anyhow::bail!("slice_update_dynamic not implemented - needs mlx-c FFI") }
    pub fn scatter_single(&self, _src: mlx_array, _i: mlx_array, _u: mlx_array, _ax: i32) -> Result<mlx_array> { anyhow::bail!("scatter_single not implemented - needs mlx-c FFI") }
    pub fn shape(&self, _x: mlx_array) -> Result<Vec<i32>> { anyhow::bail!("shape not implemented - needs mlx-c FFI") }
    pub fn sigmoid(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("sigmoid not implemented - needs mlx-c FFI") }
    pub fn exp(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("exp not implemented - needs mlx-c FFI") }
    pub fn negative(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("negative not implemented - needs mlx-c FFI") }
    pub fn subtract(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("subtract not implemented - needs mlx-c FFI") }
    pub fn remainder(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("remainder not implemented - needs mlx-c FFI") }
    pub fn full_i32(&self, _s: &[i32], _v: i32) -> Result<mlx_array> { anyhow::bail!("full_i32 not implemented - needs mlx-c FFI") }
    pub fn slice_dynamic(&self, _a: mlx_array, _s: mlx_array, _ax: &[i32], _sz: &[i32]) -> Result<mlx_array> { anyhow::bail!("slice_dynamic not implemented - needs mlx-c FFI") }
    pub fn log1p(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("log1p not implemented - needs mlx-c FFI") }
    pub fn softplus(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("softplus not implemented - needs mlx-c FFI") }
    pub fn softmax(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("softmax not implemented - needs mlx-c FFI") }
    pub fn softmax_axis(&self, _x: mlx_array, _a: i32) -> Result<mlx_array> { anyhow::bail!("softmax_axis not implemented - needs mlx-c FFI") }
    pub fn sum_axis(&self, _x: mlx_array, _a: i32, _k: bool) -> Result<mlx_array> { anyhow::bail!("sum_axis not implemented - needs mlx-c FFI") }
    pub fn abs(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("abs not implemented - needs mlx-c FFI") }
    pub fn bridge_forward(&self, _t: i32, _o: i32, _k: &mut [mlx_array], _v: &mut [mlx_array], _r: i32) -> Result<mlx_array> { anyhow::bail!("bridge_forward not implemented - needs mlx-c FFI") }
    pub fn max(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("max not implemented - needs mlx-c FFI") }
    pub fn argmax(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("argmax not implemented - needs mlx-c FFI") }
    pub fn divide(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> { anyhow::bail!("divide not implemented - needs mlx-c FFI") }
    pub fn round(&self, _x: mlx_array) -> Result<mlx_array> { anyhow::bail!("round not implemented - needs mlx-c FFI") }
    pub fn new_array(&self, _d: &[f32]) -> Result<mlx_array> { anyhow::bail!("new_array not implemented - needs mlx-c FFI") }
    pub fn zeros(&self, _s: &[i32]) -> Result<mlx_array> { anyhow::bail!("zeros not implemented - needs mlx-c FFI") }
    pub fn zeros_f32(&self, _s: &[i32]) -> Result<mlx_array> { anyhow::bail!("zeros_f32 not implemented - needs mlx-c FFI") }
    pub fn zeros_typed(&self, _s: &[i32], _l: mlx_array) -> Result<mlx_array> { anyhow::bail!("zeros_typed not implemented - needs mlx-c FFI") }
    pub fn full_f32(&self, _s: &[i32], _v: f32) -> Result<mlx_array> { anyhow::bail!("full_f32 not implemented - needs mlx-c FFI") }
    pub fn to_vec_u32(&self, _a: mlx_array) -> Result<Vec<u32>> { anyhow::bail!("to_vec_u32 not implemented - needs mlx-c FFI") }
    pub fn to_vec_f16_as_f32(&self, _a: mlx_array) -> Result<Vec<f32>> { anyhow::bail!("to_vec_f16_as_f32 not implemented - needs mlx-c FFI") }
    pub fn to_vec_bf16_as_f32(&self, _a: mlx_array) -> Result<Vec<f32>> { anyhow::bail!("to_vec_bf16_as_f32 not implemented - needs mlx-c FFI") }
    pub fn to_vec_f32(&self, _a: mlx_array) -> Result<Vec<f32>> { anyhow::bail!("to_vec_f32 not implemented - needs mlx-c FFI") }
    pub fn quantize_kv_native(&self, _x: mlx_array, _g: i32, _b: i32) -> Result<(mlx_array, mlx_array, mlx_array)> { anyhow::bail!("quantize_kv_native not implemented - needs mlx-c FFI") }
    pub fn dequantize_kv_native(&self, _d: mlx_array, _s: mlx_array, _b: mlx_array, _g: i32, _bi: i32) -> Result<mlx_array> { anyhow::bail!("dequantize_kv_native not implemented - needs mlx-c FFI") }
}
