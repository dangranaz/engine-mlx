//! # engine-mlx-attention
//!
//! Attention layer implementations for MLX-C inference.
//!
//! Composes `engine_mlx_ops` (atomic ops: qmatmul, rope, transpose) with
//! `engine_mlx_kvcache` (cache backends) into complete attention layers.
//!
//! ## Attention Types
//!
//! | Type | Module | Used by |
//! |------|--------|---------|
//! | GQA (Grouped-Query Attention) | [`gqa`] | Qwen3, Llama, Gemma4, SmolLM3, MiniCPM |
//! | Sliding Window Attention | [`sliding_window`] | Qwen3 SWA layers, Mistral |
//! | Gated Output Attention | [`gated`] | Qwen3.5 full-attention layers |
//! | Gated DeltaNet (GDN) | [`gdn`] | Qwen3.5 (75% layers), Ornith |
//!
//! ## Combinability
//!
//! Each attention type is generic over `KvCache`:
//!
//! ```rust,ignore
//! use engine_mlx_kvcache::{ConcatCache, Fp8Cache, RotatingCache};
//! use engine_mlx_attention::gqa::gqa_attention;
//!
//! // Same attention logic, different cache strategies:
//! gqa_attention(ctx, x, weights, &mut ConcatCache::new(), config)?;
//! gqa_attention(ctx, x, weights, &mut Fp8Cache::new(128, 64), config)?;
//! ```
//!
//! ## Architecture
//!
//! ```text
//! engine_mlx_ops         →  atomic ops (qmatmul, rope, transpose, sdpa)
//!       ↓
//! engine_mlx_kvcache     →  cache storage (concat, fp8, rotating)
//!       ↓
//! engine_mlx_attention   →  THIS: composes ops + cache into attention layers
//!       ↓
//! model crate            →  composes attention + mlp + norm into transformer blocks
//! ```

pub mod gqa;
pub mod sliding_window;
pub mod gated;
pub mod gdn;

// Re-export commonly used types from ops for convenience
pub use engine_mlx_ops::quant::{QuantWeights, qmatmul};
pub use engine_mlx_ops::gdn::{GdnConfig, GdnState};
