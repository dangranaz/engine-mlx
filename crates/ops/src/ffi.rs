//! MLX-C FFI re-exports.
//!
//! **NOTE**: Stub for workspace structure.

pub use engine_mlx_ffi::*;

// Re-export common types — aliases for engine_mlx_ffi FFI types.
/// Opaque handle to an MLX array on the GPU.
pub type mlx_array = engine_mlx_ffi::mlx_array;
/// Opaque handle to an MLX command stream.
pub type mlx_stream = engine_mlx_ffi::mlx_stream;
/// MLX data type enum (F16, BF16, F32, INT32, etc.).
pub type mlx_dtype = engine_mlx_ffi::mlx_dtype;
/// Optional int wrapper for FFI calls with group_size/bits.
pub type mlx_optional_int_ = engine_mlx_ffi::mlx_optional_int_;
/// Optional float wrapper for FFI calls.
pub type mlx_optional_float_ = engine_mlx_ffi::mlx_optional_float_;
/// Optional dtype wrapper for FFI calls (e.g., dequantize target).
pub type mlx_optional_dtype_ = engine_mlx_ffi::mlx_optional_dtype_;

// Re-export common functions (wildcard to avoid binding mismatches across mlx versions)
