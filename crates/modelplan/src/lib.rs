//! nxm-modelplan – model introspection crate.
//!
//! This crate analyses a model directory (config.json, tokenizer config,
//! safetensors headers) and produces a `ModelManifest` describing the
//! required operations, layer configuration and engine hints.
//!
//! The manifest can be cached as `model.plan.toml` in the model directory
//! together with a side‑car `.sha256` file used for invalidation.

pub mod manifest;
pub mod parser;
pub mod ops_detector;
pub mod plan;
pub mod cache;

pub use manifest::{
    AttentionConfig, EmbeddingConfig, EngineHints, GdnConfig, LayersConfig, MlpConfig,
    ModelInfo, ModelManifest, MoEConfig, NormConfig, OpsRequired, QuantConfig, SafetensorsInfo,
};

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

macro_rules! log_event {
    ($level:expr, $event:expr $(, $key:ident = $val:expr)*) => {
        eprintln!("[{}] {}", $level, $event);
    };
}

use crate::parser::RawModelConfig;
use crate::ops_detector::detect_ops;

/// Main entry point: introspect a model directory and produce a manifest.
pub fn introspect(model_path: &Path) -> Result<ModelManifest> {
    log_event!("info", "introspect START", model_path = model_path.display().to_string());
    let result = (move || {
        let config = parser::parse_model_dir(model_path)
            .with_context(|| format!("Failed to parse model directory: {}", model_path.display()))?;

        let (ops_required, engine_hints, layers) = detect_ops(&config);

        let head_dim = config.head_dim.unwrap_or_else(|| {
            config.hidden_size.checked_div(config.num_attention_heads).unwrap_or(128)
        });

        // Build attention config if model uses attention
        let attention = if ops_required.flash_attn_decode || ops_required.rope {
            let attn_type = if config.linear_attention.is_some() {
                "hybrid".to_string()
            } else if config.sliding_window.is_some() {
                // Stratified sliding window → hybrid
                if config.sliding_window.unwrap_or(0) < config.num_hidden_layers {
                    "hybrid"
                } else {
                    "sliding"
                }.to_string()
            } else {
                "full".to_string()
            };

            Some(AttentionConfig {
                attn_type,
                head_dim,
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                sliding_window: config.sliding_window,
                rope_theta: config.rope_theta,
            })
        } else {
            None
        };

        // Build GDN config if needed
        let gdn = config.linear_attention.as_ref().map(|la| {
            GdnConfig {
                dk: la.dk.unwrap_or(head_dim),
                dv: la.dv.unwrap_or(head_dim),
                num_heads: la.num_heads.unwrap_or(config.num_attention_heads),
                num_kv_heads: la.num_kv_heads.unwrap_or(config.num_key_value_heads),
                conv_kernel_size: la.conv_kernel_size.unwrap_or(4),
            }
        });

        // Build MoE config if present
        let moe = config.num_experts.and_then(|ne| {
            if ne > 0 {
                Some(MoEConfig {
                    num_experts: ne,
                    num_experts_per_tok: config.num_experts_per_tok.unwrap_or(2),
                    num_shared_experts: config.num_shared_experts.unwrap_or(0),
                    intermediate_size: config.intermediate_size,
                })
            } else {
                None
            }
        });

        // MLP config
        let mlp_type = if ops_required.swiglu {
            "swiglu".to_string()
        } else if ops_required.silu {
            "silu".to_string()
        } else {
            "gelu".to_string()
        };

        let mlp = MlpConfig {
            mlp_type,
            intermediate_size: config.intermediate_size,
        };

        // Norm config
        let norm = NormConfig {
            norm_type: "rms_norm".to_string(),
            eps: config.rms_norm_eps,
        };

        // Embedding config
        let embedding = EmbeddingConfig {
            emb_type: "standard".to_string(),
            tied_lm_head: config.tie_word_embeddings,
        };

        // Quant config
        let quant = config.quantization_config.as_ref().map(|qc| {
            let format = if qc.quant_method.contains("mxfp4") {
                "mxfp4".to_string()
            } else if qc.quant_method.contains("mxfp8") {
                "mxfp8".to_string()
            } else if qc.bits == Some(4) {
                "q4".to_string()
            } else if qc.bits == Some(8) {
                "q8".to_string()
            } else {
                qc.quant_method.clone()
            };

            QuantConfig {
                format,
                scales_dtype: "f16".to_string(),
                embedding_dtype: "f32".to_string(),
                lm_head_dtype: "f32".to_string(),
                group_size: qc.group_size.unwrap_or(32),
            }
        });

        let params_b = estimate_params_b(&config);

        let model_info = ModelInfo {
            family: config.model_type.clone(),
            architecture: config.architectures.first().cloned().unwrap_or_default(),
            params_b,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
        };

        Ok(ModelManifest {
            model: model_info,
            layers,
            attention,
            gdn,
            moe,
            mlp,
            norm,
            embedding,
            quant,
            safetensors: config.safetensors_info,
            ops_required,
            engine_hints,
        })
    })();
    log_event!("info", "introspect END", result_type = stringify!(Result<ModelManifest>));
    result
}

/// Persist the manifest as `model.plan.toml` + side‑car `.sha256`.
pub fn save_plan(manifest: &ModelManifest, model_dir: &Path) -> Result<()> {
    log_event!("info", "save_plan START", model_dir = model_dir.display().to_string());
    let result = (move || {
        let toml_str = toml::to_string_pretty(manifest)
            .with_context(|| "Failed to serialize manifest to TOML")?;
        let plan_path = model_dir.join("model.plan.toml");
        fs::write(&plan_path, toml_str)
            .with_context(|| format!("Failed to write {}", plan_path.display()))?;

        // Compute hash of source files and write side‑car
        let hash = cache::hash_sources(model_dir)?;
        cache::write_sidecar(model_dir, &hash)?;
        Ok(())
    })();
    log_event!("info", "save_plan END", result_type = "Result<()>");
    result
}

/// Load the manifest from `model.plan.toml` if the side‑car hash matches.
/// Returns `Ok(None)` if file missing or stale.
pub fn load_plan(model_dir: &Path) -> Result<Option<ModelManifest>> {
    log_event!("info", "load_plan START", model_dir = model_dir.display().to_string());
    let result = (move || {
        let plan_path = model_dir.join("model.plan.toml");
        if !plan_path.exists() {
            return Ok(None);
        }
        // Validate cache via side‑car
        if !cache::validate_hash(model_dir)? {
            return Ok(None); // stale
        }
        let content = fs::read_to_string(&plan_path)
            .with_context(|| format!("Failed to read {}", plan_path.display()))?;
        let manifest: ModelManifest = toml::from_str(&content)
            .with_context(|| "Failed to parse TOML manifest")?;
        Ok(Some(manifest))
    })();
    log_event!("info", "load_plan END", result_type = "Option<ModelManifest>");
    result
}

/// Convenience: load valid plan, or introspect + save if missing/stale.
pub fn ensure_plan(model_dir: &Path) -> Result<ModelManifest> {
    log_event!("info", "ensure_plan START", model_dir = model_dir.display().to_string());
    let result = (move || {
        if let Some(m) = load_plan(model_dir)? {
            return Ok(m);
        }
        let m = introspect(model_dir)?;
        save_plan(&m, model_dir)?;
        Ok(m)
    })();
    log_event!("info", "ensure_plan END", result_type = "ModelManifest");
    result
}

/// Rough parameter estimate in billions.
fn estimate_params_b(config: &RawModelConfig) -> f64 {
    let h = config.hidden_size as f64;
    let l = config.num_hidden_layers as f64;
    let inter = config.intermediate_size as f64;
    let vocab = config.vocab_size as f64;

    let attn_params = 4.0 * h * h * l;
    let mlp_params = 3.0 * h * inter * l;
    let emb_params = vocab * h;

    let total = attn_params + mlp_params + emb_params;
    (total / 1e9 * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_test_model_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = base.join(format!("nxm_modelplan_introspect_{pid}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        // Minimal config.json for Qwen3‑0.6B (full attention)
        let config = r#"{
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true
        }"#;
        let mut f = fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(config.as_bytes()).unwrap();
        // Minimal tokenizer_config.json
        fs::write(dir.join("tokenizer_config.json"), r#"{"vocab_size":151936}"#).unwrap();
        dir
    }

    #[test]
    fn introspect_works_on_minimal_config() {
        let dir = make_test_model_dir();
        let manifest = introspect(&dir).unwrap();
        assert_eq!(manifest.model.family, "qwen3");
        assert_eq!(manifest.layers.total, 28);
        assert_eq!(manifest.attention.as_ref().unwrap().attn_type, "full");
        assert!(manifest.ops_required.flash_attn_decode);
        assert!(manifest.ops_required.swiglu);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = make_test_model_dir();
        let m1 = introspect(&dir).unwrap();
        save_plan(&m1, &dir).unwrap();
        let m2_opt = load_plan(&dir).unwrap();
        assert!(m2_opt.is_some());
        let m2 = m2_opt.unwrap();
        assert_eq!(m1.model.family, m2.model.family);
        assert_eq!(m1.layers.total, m2.layers.total);
    }

    #[test]
    fn ensure_plan_is_idempotent() {
        let dir = make_test_model_dir();
        let m1 = ensure_plan(&dir).unwrap();
        let m2 = ensure_plan(&dir).unwrap();
        assert_eq!(m1.model.family, m2.model.family);
    }
}