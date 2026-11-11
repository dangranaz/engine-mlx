//! Low-level MLX-C FFI helpers for the forward pass.

#[cfg(feature = "mlx")]
pub mod real;

#[cfg(feature = "mlx")]
pub use real::MlxCtx;

#[cfg(not(feature = "mlx"))]
mod stub;

#[cfg(not(feature = "mlx"))]
pub use stub::MlxCtx;
