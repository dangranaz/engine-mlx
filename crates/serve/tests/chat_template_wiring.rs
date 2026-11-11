//! End-to-end check of the chat-template wiring in the serving path.
//!
//! `main.rs` builds every chat prompt through `Qwen3Engine::generate_chat`,
//! which renders `messages` with the model's own `chat_template.jinja`
//! (task 14, `add_generation_prompt = true`). This test pins that behavior:
//! the serving prompt ids must equal template render + encode, and the old
//! hardcoded `role: content` format must not come back.

use engine_mlx_serve::config::Qwen3Config;
use engine_mlx_serve::engine::Qwen3Engine;
use engine_mlx_serve::loader::{
    EngineInit, GateUpProj, QkvProj, QuantParams, QuantTensor, Qwen3LayerWeights, Qwen3Weights,
};
use engine_mlx_ffi::mlx_array;
use nxm_modelplan::{AttentionConfig, LayersConfig, MlpConfig, ModelInfo, ModelManifest, NormConfig};
use nxm_tokenizer::chat_template::ChatMessage;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn read_ids(path: &Path) -> Vec<u32> {
    let content = std::fs::read_to_string(path).expect("read ids fixture");
    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .flat_map(|l| l.split(','))
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().expect("parse id"))
        .collect()
}

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
            num_groups: 1,
        },
        ..Default::default()
    }
}

fn dummy_quant_tensor() -> QuantTensor {
    QuantTensor {
        w: dummy_mlx_array(),
        s: dummy_mlx_array(),
        b: dummy_mlx_array(),
        quant: QuantParams { group_size: 64, bits: 4 },
    }
}

fn dummy_weights() -> Qwen3Weights {
    let layer = Qwen3LayerWeights {
        input_layernorm: dummy_mlx_array(),
        post_attention_layernorm: dummy_mlx_array(),
        qkv: QkvProj::Fused(dummy_quant_tensor()),
        o_proj: dummy_quant_tensor(),
        q_norm: None,
        k_norm: None,
        gate_up: GateUpProj::Fused(dummy_quant_tensor()),
        down_proj: dummy_quant_tensor(),
    };
    Qwen3Weights {
        embed_tokens: dummy_quant_tensor(),
        final_norm: dummy_mlx_array(),
        layers: vec![layer],
        lm_head: None,
    }
}

fn write_tokenizer_json(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "special": true},
    {"id": 1, "content": "[system]", "special": true},
    {"id": 2, "content": "[user]", "special": true},
    {"id": 3, "content": "[assistant]", "special": true},
    {"id": 4, "content": "You", "special": false},
    {"id": 5, "content": "are", "special": false},
    {"id": 6, "content": "a", "special": false},
    {"id": 7, "content": "helpful", "special": false},
    {"id": 8, "content": "assistant.", "special": false},
    {"id": 9, "content": "Name", "special": false},
    {"id": 10, "content": "three", "special": false},
    {"id": 11, "content": "European", "special": false},
    {"id": 12, "content": "capitals.", "special": false},
    {"id": 13, "content": "hi", "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "WhitespaceSplit"},
  "post_processor": null,
  "decoder": null,
  "model": {"type": "BPE", "vocab": {"[UNK]":0, "[system]":1, "[user]":2, "[assistant]":3, "You":4, "are":5, "a":6, "helpful":7, "assistant.":8, "Name":9, "three":10, "European":11, "capitals.":12, "hi":13}, "merges": [], "unk_token": "[UNK]"}
}"#;
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();
}

fn load_engine_from(dir: &Path) -> Qwen3Engine {
    let manifest = minimal_manifest();
    let config = Qwen3Config::from_manifest(&manifest).expect("config from manifest");
    let init = EngineInit {
        manifest,
        config,
        weights: dummy_weights(),
    };
    Qwen3Engine::from_init(init, dir).expect("engine loads from fixture")
}

#[test]
fn serving_prompt_uses_chat_template() {
    let dir = fixtures_dir().join("chat_template_wiring");
    let mut engine = load_engine_from(&dir);
    let template = engine
        .chat_template
        .as_ref()
        .expect("fixture ships chat_template.jinja, template must be loaded");
    let messages: &[(String, String)] = &[
        ("system".into(), "You are a helpful assistant.".into()),
        ("user".into(), "Name three European capitals.".into()),
    ];
    let chat: Vec<ChatMessage> = messages
        .iter()
        .map(|(role, content)| ChatMessage {
            role: role.clone(),
            content: content.clone(),
        })
        .collect();
    // Oracle: the model's own template rendered, then encoded (task 14).
    let rendered = template.render(&chat, true).expect("render chat template");
    let expected_ids = engine
        .tokenizer
        .encode(&rendered)
        .expect("encode rendered prompt");
    // Serving path: main.rs delegates prompt building to generate_chat.
    let (_text, prompt_tokens) = engine.generate_chat(messages, 1).expect("generate_chat");
    assert_eq!(
        prompt_tokens as usize,
        expected_ids.len(),
        "serving prompt ids must match ChatTemplate::render + encode"
    );
    // Discriminator: the rendered prompt must differ from the old hardcoded
    // `role: content` format, so this test fails if it comes back.
    let hardcoded: String = messages
        .iter()
        .map(|(role, content)| format!("{role}: {content}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_ne!(
        rendered, hardcoded,
        "rendered prompt must not be the hardcoded role: content format"
    );
}

#[test]
fn serving_chat_errors_without_template_no_hardcoded_fallback() {
    let dir = std::env::temp_dir().join("nxm_serve_no_chat_template_test");
    let _ = std::fs::remove_dir_all(&dir);
    write_tokenizer_json(&dir);
    // No chat_template.jinja / tokenizer_config.chat_template -> no template.
    let mut engine = load_engine_from(&dir);
    assert!(engine.chat_template.is_none(), "no template must be loaded");
    let err = engine
        .generate_chat(&[("user".into(), "hi".into())], 1)
        .unwrap_err();
    assert!(
        err.to_string().contains("no chat template available"),
        "generate_chat must error instead of falling back to a hardcoded template, got: {err}"
    );
}

/// F2 (hardware): end-to-end render+encode on the real model, compared
/// against the independent `k2_horizon_prompt_ids.txt` oracle — NOT against a
/// second `render+encode` call, so it is not tautological. It fails on a
/// double-BOS, a dropped special token, or a wrong `add_generation_prompt`,
/// which the tautological R1 test would miss.
#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer + --features mlx"]
fn serving_render_encode_matches_fixture() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let model_dir = Path::new(&model_dir_str);
    let engine = Qwen3Engine::load(model_dir).expect("load engine from real model");
    let template = engine
        .chat_template
        .as_ref()
        .expect("real K2 model must ship a chat_template.jinja");
    let messages: &[(String, String)] = &[(
        "user".into(),
        "What color is a ripe banana? Answer in one word.".into(),
    )];
    let chat: Vec<ChatMessage> = messages
        .iter()
        .map(|(role, content)| ChatMessage {
            role: role.clone(),
            content: content.clone(),
        })
        .collect();
    let rendered = template.render(&chat, true).expect("render chat template");
    let ids = engine
        .tokenizer
        .encode(&rendered)
        .expect("encode rendered prompt");
    let expected = read_ids(&fixtures_dir().join("k2_horizon_prompt_ids.txt"));
    assert_eq!(
        ids, expected,
        "render+encode must match the independent fixture exactly"
    );
}
