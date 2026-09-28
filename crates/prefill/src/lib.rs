//! # engine-mlx-prefill
//!
//! Prefill pipeline with two stages:
//! 1. Prefix cache — skip tokens already processed (radix tree + SHA1)
//! 2. Chunked batch prefill — process in chunks (not 1 token at a time)
//!
//! Each stage activates only when needed. Short prompts go straight to chunk.
//! Repeated prompts hit the prefix cache.
//!
//! ## Standalone
//!
//! The model forward function is passed as a closure, making the pipeline
//! backend-agnostic — it does not depend on any specific MLX bindings.

pub mod config;
pub mod interface;
pub mod prefix_cache;
pub mod chunked_prefill;
pub mod pipeline;

pub use config::PrefillConfig;
pub use interface::PrefillPipelineTrait;
pub use pipeline::{PrefillPipeline, PrefillResult};
