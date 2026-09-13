//! Qwen3 model config — parsed from config.json or ModelManifest.

use anyhow::{Context, Result};
use nxm_modelplan::ModelManifest;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
/// Model architecture configuration for Qwen3, parsed from config.json or ModelManifest.
pub struct Qwen3Config {
    /// Hidden size (e.g., 2048 for Qwen3-0.6B, 3584 for Qwen3-1.7B).
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
/// Quantization config — bits and group_size for weight dequantization.
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

impl Qwen3Config {
    /// Build config from ModelManifest (preferred — single source of truth).
    pub fn from_manifest(manifest: &ModelManifest) -> Result<Self> {
        tracing::info!(target: "engine_mlx::qwen3::config", "Qwen3Config::from_manifest START");
        let result = (|| {
            let m = &manifest.model;
            let attn = manifest
                .attention
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("attention config missing in manifest"))?;
            let quant = manifest.quant.as_ref().map(|q| QuantizationConfig {
                group_size: q.group_size as i32,
                bits: q
                    .format
                    .strip_prefix('q')
                    .unwrap_or("4")
                    .parse()
                    .unwrap_or(4),
            });
            Ok(Self {
                hidden_size: m.hidden_size as i32,
                num_hidden_layers: manifest.layers.total as i32,
                intermediate_size: manifest.mlp.intermediate_size as i32,
                num_attention_heads: attn.num_heads as i32,
                num_key_value_heads: attn.num_kv_heads as i32,
                vocab_size: m.vocab_size as i32,
                rms_norm_eps: manifest.norm.eps as f32,
                rope_theta: attn.rope_theta as f32,
                head_dim: Some(attn.head_dim as i32),
                quantization: quant,
            })
        })();
        tracing::info!(target: "engine_mlx::qwen3::config", "Qwen3Config::from_manifest END");
        result
    }

    /// Legacy: parse config.json directly. Deprecated — prefer from_manifest.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        tracing::info!(
            target: "engine_mlx::qwen3::config",
            "Qwen3Config::from_dir START (deprecated) model_dir={}",
            model_dir.display()
        );
        let text = std::fs::read_to_string(model_dir.join("config.json"))
            .with_context(|| format!("cannot read config.json in {}", model_dir.display()))?;
        let result: Self = serde_json::from_str(&text)
            .with_context(|| "invalid config.json")?;
        tracing::info!(target: "engine_mlx::qwen3::config", "Qwen3Config::from_dir END");
        Ok(result)
    }

    /// Head dimension — falls back to hidden_size / num_attention_heads if not specified.
    pub fn head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    /// Quantization bits (e.g., 4 for Q4, 0 if unquantized).
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(0, |q| q.bits)
    }

    /// Quantization group size (e.g., 64, 0 if unquantized).
    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(0, |q| q.group_size)
    }
}

/// EOS token IDs for Qwen3.
pub const EOS_IDS: &[u32] = &[151643, 151645];

/// Think tokens to skip in non-thinking mode.
pub const THINK_TOKEN: u32 = 151667; // <think>
/// Think-end token — emitted after thinking content, skipped in non-thinking mode.
pub const THINK_END_TOKEN: u32 = 151668; // </think>

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}
