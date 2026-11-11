//! # engine-mlx-ops
//!
//! Atomic MLX-C FFI operations for engine-mlx crates.

#![allow(clippy::too_many_arguments)]

pub mod attention;
pub mod ctx;
pub mod embed;
pub mod ffi;
pub mod fp8;
pub mod gated_attn;
pub mod gdn;
pub mod gdn_forward;
pub mod gdn_metal;
pub mod kv_traits;
pub mod linear;
pub mod lm_head;
pub mod mlp;
pub mod moe;
pub mod quant;
pub mod rope;

pub mod error;
pub use error::{GdnError, OpsError, QuantError};

pub use engine_mlx_ffi::*;
pub use kv_traits::{KvCacheBackend, TensorRef};
pub use quant::{qmatmul, QuantWeights};
