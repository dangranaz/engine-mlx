//! # engine-mlx-serve
//!
//! MLX-C inference server — OpenAI-compatible HTTP API.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires complete MLX-C FFI implementation.

pub mod bench;
pub mod compiled;
pub mod config;
pub mod decode_compiled;
pub mod decode_eager;
pub mod engine;
pub mod engine_ext;
pub mod forward;
pub mod forward_qwen3;
pub mod generate;
pub mod handle;
pub mod loader;
pub mod sampler;
pub mod server;
pub mod tokenizer;

pub use engine::Qwen3Engine;