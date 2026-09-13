//! Qwen3 weight loader — safetensors loading via Rust safetensors crate + mlx-c arrays.
//!
//! Weights are stored in flat structs for zero-overhead access in the forward pass.

use std::path::Path;

use anyhow::{Context, Result};
use nxm_modelplan::ModelManifest;

use engine_mlx_ffi::mlx_array;


/// Pre-resolved weight pointers for a single transformer layer.
/// Zero HashMap lookups in the hot path.
#[derive(Clone)]
pub struct Qwen3LayerWeights {
    pub input_layernorm: mlx_array,
    pub post_attention_layernorm: mlx_array,
    pub qkv_proj_w: mlx_array,
    pub qkv_proj_s: mlx_array,
    pub qkv_proj_b: mlx_array,
    pub o_proj_w: mlx_array,
    pub o_proj_s: mlx_array,
    pub o_proj_b: mlx_array,
    pub q_norm: Option<mlx_array>,
    pub k_norm: Option<mlx_array>,
    pub gate_up_proj_w: mlx_array,
    pub gate_up_proj_s: mlx_array,
    pub gate_up_proj_b: mlx_array,
    pub down_proj_w: mlx_array,
    pub down_proj_s: mlx_array,
    pub down_proj_b: mlx_array,
}

#[derive(Clone)]
pub struct Qwen3Weights {
    pub embed_tokens: mlx_array,
    pub embed_scales: mlx_array,
    pub embed_biases: mlx_array,
    pub final_norm: mlx_array,
    pub layers: Vec<Qwen3LayerWeights>,
    // LM head — None means tied (reuse embed_tokens).
    pub lm_head_w: Option<mlx_array>,
    pub lm_head_s: Option<mlx_array>,
    pub lm_head_b: Option<mlx_array>,
}

/// Engine initialization config containing manifest, config, and weights.
#[derive(Clone)]
pub struct EngineInit {
    pub manifest: ModelManifest,
    pub config: crate::config::Qwen3Config,
    pub weights: Qwen3Weights,
}

pub mod tensor;
pub mod weights;
use weights::load_weights_from_manifest;

/// Load engine configuration (manifest + config + weights) from model directory.
pub fn load_engine(model_dir: &Path) -> Result<EngineInit> {
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_engine START model_dir={}",
        model_dir.display()
    );
    let start = std::time::Instant::now();

    // 1. Load manifest (generates/caches model.plan.toml + sidecar)
    let manifest =
        nxm_modelplan::ensure_plan(model_dir).with_context(|| "failed to ensure_plan")?;
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "manifest loaded family={} layers={}",
        manifest.model.family,
        manifest.layers.total
    );

    // 2. Build config from manifest
    let config = crate::config::Qwen3Config::from_manifest(&manifest)?;

    // 3. Load weights
    tracing::info!(target: "engine_mlx::qwen3::loader", "load_weights START");
    let weights = load_weights_from_manifest(model_dir, &manifest)?;
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_weights END layers={}",
        config.num_hidden_layers
    );

    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "Loaded {} layers, hidden={}, heads={}/kv={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    );

    let elapsed = start.elapsed();
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_engine END elapsed_ms={}",
        elapsed.as_millis()
    );

    Ok(EngineInit {
        manifest,
        config,
        weights,
    })
}

/// Load weights from model directory (convenience wrapper).
pub fn load_weights(model_dir: &Path) -> Result<Qwen3Weights> {
    let manifest = nxm_modelplan::ensure_plan(model_dir).with_context(|| "failed to ensure_plan")?;
    load_weights_from_manifest(model_dir, &manifest)
}
