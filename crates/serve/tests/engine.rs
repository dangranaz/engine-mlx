//! Unit tests for `engine_mlx_serve::engine` (Qwen3Engine tracer).

use engine_mlx_serve::config::Qwen3Config;
use engine_mlx_serve::engine::Qwen3Engine;
use engine_mlx_serve::loader::{EngineInit, Qwen3LayerWeights, Qwen3Weights};
use engine_mlx_ffi::mlx_array;
use nxm_modelplan::{AttentionConfig, LayersConfig, MlpConfig, ModelInfo, ModelManifest, NormConfig};
use std::path::Path;

fn dummy_mlx_array() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

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
            total: 1,
            attention_layers: vec![0],
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

fn dummy_weights() -> Qwen3Weights {
    let layer = Qwen3LayerWeights {
        input_layernorm: dummy_mlx_array(),
        post_attention_layernorm: dummy_mlx_array(),
        qkv_proj_w: dummy_mlx_array(),
        qkv_proj_s: dummy_mlx_array(),
        qkv_proj_b: dummy_mlx_array(),
        o_proj_w: dummy_mlx_array(),
        o_proj_s: dummy_mlx_array(),
        o_proj_b: dummy_mlx_array(),
        q_norm: None,
        k_norm: None,
        gate_up_proj_w: dummy_mlx_array(),
        gate_up_proj_s: dummy_mlx_array(),
        gate_up_proj_b: dummy_mlx_array(),
        down_proj_w: dummy_mlx_array(),
        down_proj_s: dummy_mlx_array(),
        down_proj_b: dummy_mlx_array(),
    };
    Qwen3Weights {
        embed_tokens: dummy_mlx_array(),
        embed_scales: dummy_mlx_array(),
        embed_biases: dummy_mlx_array(),
        final_norm: dummy_mlx_array(),
        layers: vec![layer],
        lm_head_w: None,
        lm_head_s: None,
        lm_head_b: None,
    }
}

fn write_minimal_tokenizer(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 3, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 4, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "WhitespaceSplit"},
  "post_processor": null,
  "decoder": null,
  "model": {"type": "BPE", "vocab": {"[UNK]":0, "hello":1, "world":2, "<|im_end|>":3, "<|im_start|>":4, "test":5, "hello world":6}, "merges": [], "unk_token": "[UNK]"}
}"#;
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();
}

#[test]
fn engine_load_missing_dir_bails_not_panics() {
    let dir = Path::new("/tmp/nxm_engine_missing_dir_12345");
    let result = std::panic::catch_unwind(|| Qwen3Engine::load(dir));
    assert!(result.is_ok(), "Qwen3Engine::load panicked");
    assert!(result.unwrap().is_err(), "expected Err for missing model_dir");
}

#[test]
fn engine_from_dummy_init_without_mlx_does_not_panic() {
    let dir = std::env::temp_dir().join("nxm_engine_dummy_init_test");
    write_minimal_tokenizer(&dir);
    let manifest = minimal_manifest();
    let config = Qwen3Config::from_manifest(&manifest).unwrap();
    let weights = dummy_weights();
    let init = EngineInit {
        manifest,
        config,
        weights,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Qwen3Engine::from_init(init, &dir)
    }));
    assert!(result.is_ok(), "from_init panicked without mlx");
    let engine = result.unwrap().expect("from_init should succeed without mlx when tokenizer valid");
    assert_eq!(engine.offset, 0);
    assert!(engine.embed_table.is_none(), "embed_table should be None without mlx");
}

#[test]
fn engine_generate_with_dummy_init_produces_output() {
    let dir = std::env::temp_dir().join("nxm_engine_generate_test");
    write_minimal_tokenizer(&dir);
    let manifest = minimal_manifest();
    let config = Qwen3Config::from_manifest(&manifest).unwrap();
    let weights = dummy_weights();
    let init = EngineInit {
        manifest,
        config,
        weights,
    };
    let mut engine = Qwen3Engine::from_init(init, &dir).expect("from_init failed");
    let out = engine.generate("hello world", 3).expect("generate failed");
    // With fallback echo, decode should be non-empty (contains hello/world or unk)
    assert!(!out.is_empty(), "generate returned empty string");
    assert!(engine.offset > 0, "offset should have advanced");
    engine.reset();
    assert_eq!(engine.offset, 0);
}

#[test]
fn engine_embed_without_table_bails() {
    let dir = std::env::temp_dir().join("nxm_engine_embed_test");
    write_minimal_tokenizer(&dir);
    let manifest = minimal_manifest();
    let config = Qwen3Config::from_manifest(&manifest).unwrap();
    let init = EngineInit {
        manifest,
        config,
        weights: dummy_weights(),
    };
    let engine = Qwen3Engine::from_init(init, &dir).unwrap();
    let err = engine.embed(&[1, 2]).unwrap_err();
    assert!(err.to_string().contains("embed_table not available"));
}
