//! Typed error types for the engine-mlx-ops library crate.
//!
//! Per RULES: library-level errors use `thiserror` with structured variants.

use thiserror::Error;

/// Errors that can originate from GDN operations.
#[derive(Debug, Error)]
pub enum GdnError {
    /// A NUL byte was found in a kernel input or output name passed to the
    /// Metal kernel builder. This indicates a programming error — kernel
    /// names are compile-time constants and should never contain NUL.
    #[error("NUL byte in kernel name '{name}': {source}")]
    NulInName {
        name: &'static str,
        #[source]
        source: std::ffi::NulError,
    },

    /// The Metal kernel object could not be created (returned null).
    /// Check that the Metal device is available and the kernel source is valid.
    #[error("failed to create GDN Metal kernel (null handle)")]
    MetalKernelCreate,

    /// The Metal kernel executed but returned a non-zero status code.
    /// The status value corresponds to the Metal kernel's return code.
    #[error("GDN Metal kernel failed with status {status}")]
    MetalKernelFailed { status: i32 },

    /// Failed to allocate or create a primitive MLX array (e.g., ones weight
    /// for unweighted RMSNorm, scalar constants).
    #[error("{context}")]
    ArrayCreation { context: String },
}

/// Errors that can originate from quant operations.
#[derive(Debug, Error)]
pub enum QuantError {
    /// Weight quantization parameters are inconsistent.
    #[error("quantization config mismatch: {detail}")]
    ConfigMismatch { detail: String },

    /// Group size or bits value is out of the supported range.
    #[error("unsupported quantization params: group_size={group_size} bits={bits}")]
    UnsupportedParams { group_size: i32, bits: i32 },
}

/// Top-level error type for the `engine-mlx-ops` crate.
#[derive(Debug, Error)]
pub enum OpsError {
    /// GDN-related error.
    #[error(transparent)]
    Gdn(#[from] GdnError),

    /// Quant-related error.
    #[error(transparent)]
    Quant(#[from] QuantError),

    /// Kernel or FFI call failed.
    #[error("kernel execution failed: {detail}")]
    KernelFailed { detail: String },
}
