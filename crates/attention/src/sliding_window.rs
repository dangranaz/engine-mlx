//! Sliding Window Attention — attention limitata a una finestra di token recenti.
//!
//! Identica a GQA ma la cache mantiene solo gli ultimi `window_size` token.
//! Usata dai layer SWA di Qwen3 e Mistral.
//!
//! La finestra è gestita dalla `RotatingCache` — questo modulo si limita
//! a comporre l'attention usando quella cache. Il vantaggio è che SDPA
//! opera su un KV ridotto → meno compute, memoria costante.
//!
//! ## Differenza con GQA
//!
//! - Stessa logica QKV + RoPE + SDPA
//! - La cache è `RotatingCache` che taglia automaticamente
//! - Per il modello è trasparente: chiama `sliding_window_attention()`
//!   invece di `gqa_attention()`, il resto è identico

use anyhow::Result;
use engine_mlx_ffi::mlx_array;
use engine_mlx_ffi::MlxCtx;
use engine_mlx_kvcache::KvCache;

use crate::gqa::{GqaConfig, GqaWeights};

/// Sliding window attention configuration.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Base GQA config.
    pub gqa: GqaConfig,
    /// Window size (number of recent tokens to attend over).
    pub window_size: usize,
}

/// Sliding window attention forward pass.
///
/// Identical to `gqa_attention` but designed to be used with a `RotatingCache`.
/// The cache handles the windowing; this function just composes the ops.
///
/// Any `KvCache` implementation can be passed, but `RotatingCache` gives
/// the intended sliding window behavior with O(1) memory.
pub fn sliding_window_attention<C: KvCache>(
    ctx: &MlxCtx,
    x: mlx_array,
    weights: &GqaWeights,
    cache: &mut C,
    config: &SlidingWindowConfig,
    offset: i32,
) -> Result<mlx_array> {
    // Delegate to GQA — the cache backend enforces the window
    crate::gqa::gqa_attention(ctx, x, weights, cache, &config.gqa, offset)
}
