#![allow(missing_docs)]
//! MLX-C FFI bindings for engine-mlx inference engine.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires:
//! - mlx-c headers installed (`brew install mlx-c` or clone to `vendors/mlx-c`)
//! - Complete bindgen wrapper for MLX-C C API (matmul, rope, quantized_matmul, etc.)

// Always available dtype constants
pub const MLX_FLOAT32: u32 = 0;
pub const MLX_FLOAT16: u32 = 1;
pub const MLX_BFLOAT16: u32 = 2;
pub const MLX_INT32: u32 = 3;
pub const MLX_INT16: u32 = 4;
pub const MLX_INT8: u32 = 5;
pub const MLX_UINT32: u32 = 6;
pub const MLX_UINT16: u32 = 7;
pub const MLX_UINT8: u32 = 8;
pub const MLX_BOOL: u32 = 9;
pub const MLX_UINT4: u32 = 10;
pub const MLX_INT4: u32 = 11;
pub const MLX_COMPLEX64: u32 = 12;
pub const MLX_COMPLEX32: u32 = 13;

#[cfg(feature = "mlx")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(not(feature = "mlx"))]
mod stub {
    #[repr(C)] #[derive(Copy, Clone, Debug)] pub struct mlx_array(pub *mut std::ffi::c_void);
    #[repr(u32)] #[derive(Copy, Clone, Debug, PartialEq, Eq)] pub enum mlx_dtype {
        MLX_BOOL = 0,
        MLX_UINT8 = 1,
        MLX_UINT16 = 2,
        MLX_UINT32 = 3,
        MLX_UINT64 = 4,
        MLX_INT8 = 5,
        MLX_INT16 = 6,
        MLX_INT32 = 7,
        MLX_INT64 = 8,
        MLX_FLOAT16 = 9,
        MLX_FLOAT32 = 10,
        MLX_FLOAT64 = 11,
        MLX_BFLOAT16 = 12,
        MLX_COMPLEX64 = 13,
    }
    #[repr(C)] #[derive(Copy, Clone, Debug, PartialEq, Eq)] pub enum mlx_device_type { CPU=0, GPU=1 }
    #[repr(C)] #[derive(Copy, Clone)] pub struct mlx_stream(pub *mut std::ffi::c_void);
    #[repr(C)] #[derive(Copy, Clone, Debug)] pub struct mlx_optional_int_ { pub value: i32, pub has_value: bool }
    #[repr(C)] #[derive(Copy, Clone, Debug)] pub struct mlx_optional_float_ { pub value: f32, pub has_value: bool }
    #[repr(C)] #[derive(Copy, Clone, Debug)] pub struct mlx_optional_dtype_ { pub value: mlx_dtype, pub has_value: bool }
    macro_rules! stub_fn { ($name:ident ($($arg:ident: $ty:ty),*) -> $ret:ty) => { pub fn $name($($arg: $ty),*) -> $ret { panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c."); } }; ($name:ident ($($arg:ident: $ty:ty),*) ) => { pub fn $name($($arg: $ty),*) { panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c."); } }; }
    stub_fn!(mlx_default_gpu_stream_new() -> mlx_stream);
    stub_fn!(mlx_default_cpu_stream_new() -> mlx_stream);
    stub_fn!(mlx_synchronize(stream: mlx_stream) -> i32);
    stub_fn!(mlx_synchronize_stream(stream: mlx_stream) -> i32);
    stub_fn!(mlx_array_new_data(data: *const std::ffi::c_void, shape: *const i32, ndim: i32, dtype: mlx_dtype) -> mlx_array);
    stub_fn!(mlx_array_free(arr: mlx_array) -> i32);
    stub_fn!(mlx_array_eval(arr: mlx_array) -> i32);
    stub_fn!(mlx_eval(arr: mlx_array) -> i32);
    stub_fn!(mlx_vector_array_new_data(arrays: *const mlx_array, len: usize) -> *mut std::ffi::c_void);
    stub_fn!(mlx_vector_array_free(vec: *mut std::ffi::c_void) -> i32);
    stub_fn!(mlx_vector_array_size(vec: *mut std::ffi::c_void) -> usize);
    stub_fn!(mlx_vector_array_get(out: *mut mlx_array, vec: *mut std::ffi::c_void, index: usize) -> i32);
    stub_fn!(mlx_array_ndim(arr: mlx_array) -> usize);
    stub_fn!(mlx_array_shape(arr: mlx_array) -> *const i32);
    stub_fn!(mlx_array_data_float32(arr: mlx_array) -> *const f32);
    stub_fn!(mlx_reshape(out: *mut mlx_array, arr: mlx_array, shape: *const i32, ndim: i32) -> i32);
    stub_fn!(mlx_transpose(out: *mut mlx_array, arr: mlx_array, axes: *const i32, ndim: i32) -> i32);
    stub_fn!(mlx_concatenate(out: *mut mlx_array, arrays: *const mlx_array, count: i32, axis: i32) -> i32);
    stub_fn!(mlx_quantized_matmul(out: *mut mlx_array, a: mlx_array, b: mlx_array, scales: mlx_array, biases: mlx_array, transpose: bool, group_size: i32, bits: i32, mode: *const i8, stream: mlx_stream) -> i32);
    stub_fn!(mlx_fast_rope(out: *mut mlx_array, x: mlx_array, dim: i32, traditional: bool, base: f32, scale: f32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_fast_rope_dynamic(out: *mut mlx_array, x: mlx_array, dim: i32, traditional: bool, base: f32, scale: f32, freq_scale: f32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_rms_norm(out: *mut mlx_array, x: mlx_array, weight: mlx_array, eps: f32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_layer_norm(out: *mut mlx_array, x: mlx_array, weight: mlx_array, bias: mlx_array, eps: f32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_silu(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_softmax(out: *mut mlx_array, x: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_multiply(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_add(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_matmul(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_take(out: *mut mlx_array, x: mlx_array, indices: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_astype(out: *mut mlx_array, arr: mlx_array, dtype: mlx_dtype, stream: mlx_stream) -> i32);
    stub_fn!(mlx_max_axis(out: *mut mlx_array, x: mlx_array, axis: i32, keepdims: bool, stream: mlx_stream) -> i32);
    stub_fn!(mlx_max(out: *mut mlx_array, x: mlx_array, keepdims: bool, stream: mlx_stream) -> i32);
    stub_fn!(mlx_maximum(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_minimum(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_round(out: *mut mlx_array, x: mlx_array, decimals: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_clip(out: *mut mlx_array, x: mlx_array, min: mlx_array, max: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_divide(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_abs(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_dequantize(out: *mut mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: i32, bits: i32, mode: *const i8, global_scale: mlx_array, dtype: mlx_dtype, stream: mlx_stream) -> i32);
    stub_fn!(mlx_quantize(out: *mut *mut mlx_array, x: mlx_array, group_size: i32, bits: i32, mode: *const i8, global_scale: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_take_axis(out: *mut mlx_array, a: mlx_array, indices: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_transpose_axes(out: *mut mlx_array, arr: mlx_array, axes: *const i32, ndim: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_concatenate_axis(out: *mut mlx_array, vec: *mut std::ffi::c_void, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_split_sections(out: *mut *mut mlx_array, a: mlx_array, sections: *const i32, len: usize, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_zeros(out: *mut mlx_array, shape: *const i32, len: i32, dtype: mlx_dtype, stream: mlx_stream) -> i32);
    stub_fn!(mlx_full(out: *mut mlx_array, shape: *const i32, len: i32, val: mlx_array, dtype: mlx_dtype, stream: mlx_stream) -> i32);
    stub_fn!(mlx_arange(out: *mut mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, stream: mlx_stream) -> i32);
    stub_fn!(mlx_exp(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_negative(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_subtract(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_sigmoid(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_softmax_axis(out: *mut mlx_array, x: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_sum_axis(out: *mut mlx_array, x: mlx_array, axis: i32, keep_dims: bool, stream: mlx_stream) -> i32);
    stub_fn!(mlx_argmax(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_fast_rms_norm(out: *mut mlx_array, x: mlx_array, weight: mlx_array, eps: f32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_conv1d(out: *mut mlx_array, input: mlx_array, weight: mlx_array, stride: i32, padding: i32, dilation: i32, groups: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_tile(out: *mut mlx_array, x: mlx_array, reps: *const i32, len: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_slice_update_dynamic(out: *mut mlx_array, src: mlx_array, update: mlx_array, start: mlx_array, axes: *const i32, len: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_scatter_single(out: *mut mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_slice_dynamic(out: *mut mlx_array, a: mlx_array, start: mlx_array, axes: *const i32, len: i32, size: *const i32, len2: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_copy(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_expand_dims(out: *mut mlx_array, x: mlx_array, axis: i32, stream: mlx_stream) -> i32);
    stub_fn!(mlx_fast_scaled_dot_product_attention(out: *mut mlx_array, q: mlx_array, k: mlx_array, v: mlx_array, scale: f32, mode: *const i8, mask: mlx_array, optional: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_log1p(out: *mut mlx_array, x: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_remainder(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    // Metal kernel stubs (for gdn_metal)
    #[repr(C)] #[derive(Copy, Clone)] pub struct mlx_vector_string(pub *mut std::ffi::c_void);
    #[repr(C)] #[derive(Copy, Clone)] pub struct mlx_fast_metal_kernel(pub *mut std::ffi::c_void);
    #[repr(C)] #[derive(Copy, Clone)] pub struct mlx_fast_metal_kernel_config(pub *mut std::ffi::c_void);
    stub_fn!(mlx_vector_string_new() -> mlx_vector_string);
    stub_fn!(mlx_vector_string_free(vec: mlx_vector_string) -> i32);
    stub_fn!(mlx_vector_string_append_value(vec: mlx_vector_string, value: *const i8) -> i32);
    stub_fn!(mlx_fast_metal_kernel_new(name: *const i8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: *const i8, header: *const i8, ensure_row_contiguous: bool, atomic_outputs: bool) -> mlx_fast_metal_kernel);
    stub_fn!(mlx_fast_metal_kernel_free(kernel: mlx_fast_metal_kernel) -> i32);
    stub_fn!(mlx_fast_metal_kernel_config_new() -> mlx_fast_metal_kernel_config);
    stub_fn!(mlx_fast_metal_kernel_config_free(config: mlx_fast_metal_kernel_config) -> i32);
    stub_fn!(mlx_fast_metal_kernel_config_add_output_arg(config: mlx_fast_metal_kernel_config, shape: *const i32, ndim: usize, dtype: mlx_dtype) -> i32);
    stub_fn!(mlx_fast_metal_kernel_config_set_grid(config: mlx_fast_metal_kernel_config, x: i32, y: i32, z: i32) -> i32);
    stub_fn!(mlx_fast_metal_kernel_config_set_thread_group(config: mlx_fast_metal_kernel_config, x: i32, y: i32, z: i32) -> i32);
    stub_fn!(mlx_fast_metal_kernel_config_add_template_arg_int(config: mlx_fast_metal_kernel_config, name: *const i8, value: i32) -> i32);
    stub_fn!(mlx_fast_metal_kernel_apply(outputs: *mut mlx_vector_array, kernel: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, stream: mlx_stream) -> i32);
    // Additional MLX API stubs used by gdn.rs
    stub_fn!(mlx_array_size(arr: mlx_array) -> usize);
    stub_fn!(mlx_array_dtype(arr: mlx_array) -> mlx_dtype);
    stub_fn!(mlx_array_data_uint32(arr: mlx_array) -> *const u32);
    stub_fn!(mlx_array_data_uint16(arr: mlx_array) -> *const u16);
    stub_fn!(mlx_array_dim(arr: mlx_array, axis: i32) -> i32);
    stub_fn!(mlx_array_new_float(value: f32) -> mlx_array);
    stub_fn!(mlx_equal(out: *mut mlx_array, a: mlx_array, b: mlx_array, stream: mlx_stream) -> i32);
    stub_fn!(mlx_slice(out: *mut mlx_array, a: mlx_array, start: *const i32, start_len: usize, stop: *const i32, stop_len: usize, strides: *const i32, strides_len: usize, stream: mlx_stream) -> i32);
    stub_fn!(mlx_slice_update(out: *mut mlx_array, src: mlx_array, update: mlx_array, start: *const i32, start_len: usize, stop: *const i32, stop_len: usize, strides: *const i32, strides_len: usize, stream: mlx_stream) -> i32);
    pub type mlx_vector_array = *mut std::ffi::c_void;
    // Closure stubs
    #[repr(C)] #[derive(Copy, Clone)] pub struct mlx_closure(pub *mut std::ffi::c_void);
    stub_fn!(mlx_closure_new() -> mlx_closure);
    stub_fn!(mlx_closure_free(cls: mlx_closure) -> i32);
    stub_fn!(mlx_closure_new_func(fun: Option<unsafe extern "C" fn(*mut mlx_vector_array, mlx_vector_array) -> i32>) -> mlx_closure);
    stub_fn!(mlx_closure_new_func_payload(fun: Option<unsafe extern "C" fn(*mut mlx_vector_array, mlx_vector_array, *mut std::ffi::c_void) -> i32>, payload: *mut std::ffi::c_void, dtor: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>) -> mlx_closure);
    stub_fn!(mlx_closure_set(dst: *mut mlx_closure, src: mlx_closure) -> i32);
    stub_fn!(mlx_closure_apply(out: *mut mlx_vector_array, cls: mlx_closure, input: mlx_vector_array) -> i32);
    stub_fn!(mlx_compile(out: *mut mlx_closure, fun: mlx_closure, shapeless: bool) -> i32);
    pub mod bindings { pub use super::*; }
}

#[cfg(not(feature = "mlx"))]
pub use stub::*;

pub mod bindings {
    #[cfg(feature = "mlx")]
    pub use crate::{
        mlx_array, mlx_stream, mlx_dtype, mlx_dtype_, mlx_device_type,
        mlx_optional_int, mlx_optional_float, mlx_optional_dtype,
        mlx_string, mlx_vector_array,
    };
    #[cfg(not(feature = "mlx"))]
    pub use crate::stub::*;
}

pub mod context;
pub mod ops;
pub mod compile;

pub use context::MlxContext;
pub use ops::MlxCtx;
pub use compile::{Closure, compile, compile_shapeless};

#[cfg(not(feature = "mlx"))]
unsafe impl Send for crate::stub::mlx_stream {}
#[cfg(not(feature = "mlx"))]
unsafe impl Sync for crate::stub::mlx_stream {}
#[cfg(not(feature = "mlx"))]
unsafe impl Send for crate::stub::mlx_array {}
#[cfg(not(feature = "mlx"))]
unsafe impl Sync for crate::stub::mlx_array {}

#[cfg(feature = "mlx")]
unsafe impl Send for mlx_array {}
#[cfg(feature = "mlx")]
unsafe impl Sync for mlx_array {}
#[cfg(feature = "mlx")]
unsafe impl Send for mlx_stream {}
#[cfg(feature = "mlx")]
unsafe impl Sync for mlx_stream {}
#[cfg(feature = "mlx")]
unsafe impl Send for mlx_vector_array {}
#[cfg(feature = "mlx")]
unsafe impl Sync for mlx_vector_array {}
