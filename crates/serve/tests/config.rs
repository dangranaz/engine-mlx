//! Unit tests for `engine_mlx_serve::config` module.

use engine_mlx_serve::config::Qwen3Config;
use nxm_modelplan::{AttentionConfig, LayersConfig, MlpConfig, ModelInfo, ModelManifest, NormConfig};

fn minimal_manifest() -> ModelManifest {
    ModelManifest {
        model: ModelInfo {
            family: "qwen3".into(),
            architecture: "Qwen3ForCausalLM".into(),
            params_b: 0.6,
            hidden_size: 1024,
            vocab_size: 151936,
            max_position_embeddings: 4096,
        },
        layers: LayersConfig {
            total: 28,
            attention_layers: (0..28).collect(),
            gdn_layers: vec![],
            moe_layers: vec![],
        },
        attention: Some(AttentionConfig {
            attn_type: "full".into(),
            head_dim: 128,
            num_heads: 16,
            num_kv_heads: 8,
            sliding_window: Some(128),
            rope_theta: 1_000_000.0,
        }),
        mlp: MlpConfig {
            mlp_type: "swiglu".into(),
            intermediate_size: 3072,
        },
        norm: NormConfig {
            norm_type: "rms_norm".into(),
            eps: 1e-6,
        },
        ..Default::default()
    }
}

#[test]
fn qwen3_config_from_manifest_ok() {
    let m = minimal_manifest();
    let cfg = Qwen3Config::from_manifest(&m).expect("from_manifest should succeed");
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 28);
}

#[test]
fn qwen3_config_from_dir_missing_bails() {
    let dir = std::env::temp_dir().join("nxm_cfg_missing_test");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::remove_file(dir.join("config.json"));
    let err = Qwen3Config::from_dir(&dir).unwrap_err();
    assert!(err.to_string().contains("config.json") || err.to_string().contains("cannot read"));
}

#[test]
fn qwen3_config_from_manifest_missing_attention_bails() {
    let mut m = minimal_manifest();
    m.attention = None;
    let err = Qwen3Config::from_manifest(&m).unwrap_err();
    assert!(err.to_string().contains("attention"));
}
