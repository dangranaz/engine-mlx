use nxm_modelplan::*;
use std::fs;
use std::io::Write;

fn make_test_model_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // Per-call counter guarantees a unique directory even when two tests in the
    // same process (same pid) observe the same `nanos`. Without it, parallel
    // tests could share a directory: `test_cache_invalidation` rewrites
    // config.json, which would invalidate the sidecar hash and make
    // `save_load_roundtrip`'s `load_plan` return None intermittently.
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = base.join(format!("nxm_modelplan_introspect_{pid}_{nanos}_{seq}"));
    fs::create_dir_all(&dir).unwrap();
    // Uniform quantization (Qwen3 style): only top-level scalar keys, no overrides.
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
        "tie_word_embeddings": true,
        "quantization_config": {
            "quant_method": "awq",
            "group_size": 32,
            "bits": 4,
            "mode": "affine"
        }
    }"#;
    let mut f = fs::File::create(dir.join("config.json")).unwrap();
    f.write_all(config.as_bytes()).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"vocab_size":151936}"#,
    )
    .unwrap();
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

#[test]
fn test_cache_invalidation() {
    let dir = make_test_model_dir();
    let m = introspect(&dir).unwrap();
    save_plan(&m, &dir).unwrap();
    assert!(load_plan(&dir).unwrap().is_some());

    let config_path = dir.join("config.json");
    let mut content = fs::read_to_string(&config_path).unwrap();
    content = content.replace("\"qwen3\"", "\"qwen3-modified\"");
    fs::write(&config_path, content).unwrap();

    assert!(load_plan(&dir).unwrap().is_none());
}

#[test]
fn test_introspect_qwen3_06b_4bit() {
    let candidates = [
        std::path::Path::new("engines/serve-mlx-c/models/Qwen3-0.6B-4bit"),
        std::path::Path::new("../../engines/serve-mlx-c/models/Qwen3-0.6B-4bit"),
        std::path::Path::new("../serve-mlx-c/models/Qwen3-0.6B-4bit"),
    ];

    let model_dir = match candidates.iter().find(|p| p.exists()) {
        Some(d) => d,
        None => {
            eprintln!("Skipping test_introspect_qwen3: model directory not found");
            return;
        }
    };

    let manifest = introspect(model_dir).unwrap();
    assert_eq!(manifest.model.family, "qwen3");
    assert_eq!(manifest.layers.total, 28);
    assert_eq!(manifest.model.vocab_size, 151936);
    let attn = manifest.attention.as_ref().expect("attention config");
    assert_eq!(attn.attn_type, "full");
    assert_eq!(attn.num_heads, 16);
    assert_eq!(attn.num_kv_heads, 8);
    assert!(manifest.ops_required.flash_attn_decode);
    assert!(manifest.ops_required.swiglu);

    save_plan(&manifest, model_dir).unwrap();
    let loaded = load_plan(model_dir).unwrap();
    assert!(loaded.is_some());
    let roundtrip = loaded.unwrap();
    assert_eq!(roundtrip.model.family, manifest.model.family);
    assert_eq!(roundtrip.layers.total, manifest.layers.total);
    assert_eq!(roundtrip.attention.as_ref().unwrap().attn_type, "full");
}

/// Build a temp model dir with a K2-Horizon-shaped config.json.
fn make_k2_horizon_model_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = base.join(format!("nxm_modelplan_k2_{pid}_{nanos}_{seq}"));
    fs::create_dir_all(&dir).unwrap();
    // Shape mirrors the real K2-Horizon-3.7B config.json: grouped RMSNorm
    // (layernorm_num_groups=2), untied lm_head, vocab 250624, 36 layers,
    // hidden 2560. Mixed quantization: default {group_size:64, bits:4}
    // with per-tensor overrides (embed_tokens, q_proj, gate_proj, lm_head).
    let config = r#"{
        "model_type": "k2_horizon",
        "architectures": ["K2HorizonForCausalLM"],
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "intermediate_size": 9728,
        "vocab_size": 250624,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "layernorm_num_groups": 2,
        "tie_word_embeddings": false,
        "quantization_config": {
            "quant_method": "awq",
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.embed_tokens": {"group_size": 64, "bits": 8, "mode": "affine"},
            "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 8, "mode": "affine"},
            "model.layers.0.mlp.gate_proj": {"group_size": 32, "bits": 4, "mode": "affine"},
            "lm_head": {"group_size": 64, "bits": 8, "mode": "affine"}
        }
    }"#;
    let mut f = fs::File::create(dir.join("config.json")).unwrap();
    f.write_all(config.as_bytes()).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"vocab_size":250624}"#,
    )
    .unwrap();
    dir
}

#[test]
fn introspect_recognizes_k2_horizon_config() {
    let dir = make_k2_horizon_model_dir();
    let manifest = introspect(&dir).unwrap();

    assert_eq!(manifest.model.family, "k2_horizon");
    assert_eq!(manifest.layers.total, 36);
    assert_eq!(manifest.model.hidden_size, 2560);
    assert_eq!(manifest.model.vocab_size, 250624);

    // Grouped RMSNorm: 2 groups.
    assert_eq!(manifest.norm.num_groups, 2);

    // lm_head is NOT tied (separate output projection).
    assert!(!manifest.embedding.tied_lm_head);

    // Standard dense transformer: all layers are attention, no GDN/MoE.
    let attn = manifest.attention.as_ref().expect("attention config");
    assert_eq!(attn.attn_type, "full");
    assert_eq!(attn.num_heads, 20);
    assert_eq!(attn.num_kv_heads, 4);
    assert!(manifest.gdn.is_none());
    assert!(manifest.moe.is_none());

    // k2_horizon ops: rope + flash attn + swiglu, no MoE routing.
    assert!(manifest.ops_required.rope);
    assert!(manifest.ops_required.flash_attn_decode);
    assert!(manifest.ops_required.swiglu);
    assert!(manifest.ops_required.rmsnorm);
    assert!(!manifest.ops_required.moe_routing);
    assert!(!manifest.ops_required.fused_gdn_step);
}

#[test]
fn layernorm_num_groups_defaults_to_one_when_absent() {
    // A config without layernorm_num_groups (standard qwen3) must default to 1.
    let dir = make_test_model_dir();
    let manifest = introspect(&dir).unwrap();
    assert_eq!(manifest.norm.num_groups, 1);
}

/// Test that uniform quantization (Qwen3 style) has no overrides.
#[test]
fn quant_uniform_has_no_overrides() {
    let dir = make_test_model_dir();
    let manifest = introspect(&dir).unwrap();
    let quant = manifest.quant.expect("quant config should exist");
    // Qwen3 has uniform quantization - no per-tensor overrides
    assert!(quant.overrides.is_empty(), "uniform quant should have no overrides");
    // Default values from the test config
    assert_eq!(quant.group_size, 32); // default fallback
    assert_eq!(quant.format, "q4"); // bits=4 -> q4
}

/// Test that mixed quantization (K2-Horizon style) parses per-tensor overrides.
#[test]
fn quant_mixed_parses_per_tensor_overrides() {
    let dir = make_k2_horizon_model_dir();
    let manifest = introspect(&dir).unwrap();
    let quant = manifest.quant.expect("quant config should exist");

    // K2-Horizon has default {group:64, bits:4} + per-tensor overrides
    assert_eq!(quant.group_size, 64);
    assert_eq!(quant.format, "q4"); // bits=4 -> q4

    // Check specific overrides exist
    assert!(quant.overrides.contains_key("model.embed_tokens"), "embed_tokens override");
    assert!(quant.overrides.contains_key("model.layers.0.self_attn.q_proj"), "q_proj override");
    assert!(quant.overrides.contains_key("model.layers.0.mlp.gate_proj"), "gate_proj override");
    assert!(quant.overrides.contains_key("lm_head"), "lm_head override");

    // Check override values
    let embed_override = quant.overrides.get("model.embed_tokens").unwrap();
    assert_eq!(embed_override.bits, Some(8));
    assert_eq!(embed_override.group_size, Some(64));

    let q_proj_override = quant.overrides.get("model.layers.0.self_attn.q_proj").unwrap();
    assert_eq!(q_proj_override.bits, Some(8));
    assert_eq!(q_proj_override.group_size, Some(64));

    let gate_proj_override = quant.overrides.get("model.layers.0.mlp.gate_proj").unwrap();
    assert_eq!(gate_proj_override.bits, Some(4));
    assert_eq!(gate_proj_override.group_size, Some(32));

    let lm_head_override = quant.overrides.get("lm_head").unwrap();
    assert_eq!(lm_head_override.bits, Some(8));
    assert_eq!(lm_head_override.group_size, Some(64));
}

/// Test that the quant_for lookup falls back to default for names without overrides.
#[test]
fn quant_lookup_falls_back_to_default() {
    let dir = make_k2_horizon_model_dir();
    let manifest = introspect(&dir).unwrap();
    let quant = manifest.quant.expect("quant config should exist");

    // A tensor name that has no explicit override should fall back to default
    let (gs, bits) = quant.quant_for("model.layers.99.down_proj");
    assert_eq!(gs, 64, "should fall back to default group_size");
    assert_eq!(bits, 4, "should fall back to default bits");

    // An override should return the override values
    let (gs, bits) = quant.quant_for("model.layers.0.self_attn.q_proj");
    assert_eq!(gs, 64, "override group_size");
    assert_eq!(bits, 8, "override bits");

    // Another override
    let (gs, bits) = quant.quant_for("model.layers.0.mlp.gate_proj");
    assert_eq!(gs, 32, "override group_size for gate_proj");
    assert_eq!(bits, 4, "override bits for gate_proj");
}

/// Build a temp model dir with a given config.json body.
fn make_model_dir_with_config(body: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = base.join(format!("nxm_modelplan_rope_{pid}_{nanos}_{seq}"));
    fs::create_dir_all(&dir).unwrap();
    let mut f = fs::File::create(dir.join("config.json")).unwrap();
    f.write_all(body.as_bytes()).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"vocab_size":250624}"#).unwrap();
    dir
}

/// K2-Horizon keeps `rope_theta` nested inside `rope_parameters`, not at the top
/// level. The parser must read it from there; otherwise it falls back to the
/// 10000 default, which shifts every RoPE position and breaks token parity.
#[test]
fn rope_theta_read_from_nested_rope_parameters() {
    let dir = make_model_dir_with_config(
        r#"{
            "model_type": "k2_horizon",
            "architectures": ["K2HorizonForCausalLM"],
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 10240,
            "vocab_size": 250624,
            "max_position_embeddings": 524288,
            "rms_norm_eps": 1e-6,
            "layernorm_num_groups": 2,
            "tie_word_embeddings": false,
            "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}
        }"#,
    );
    let manifest = introspect(&dir).unwrap();
    let attn = manifest.attention.as_ref().expect("attention config");
    assert_eq!(
        attn.rope_theta, 10_000_000.0,
        "rope_theta must come from rope_parameters.rope_theta, not the 10000 default"
    );
}

/// When `rope_theta` is only at the top level (older configs, Qwen3 style) it is
/// still honored.
#[test]
fn rope_theta_read_from_top_level_when_no_rope_parameters() {
    let dir = make_model_dir_with_config(
        r#"{
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
            "rms_norm_eps": 1e-6
        }"#,
    );
    let manifest = introspect(&dir).unwrap();
    let attn = manifest.attention.as_ref().expect("attention config");
    assert_eq!(attn.rope_theta, 1_000_000.0);
}

/// Nested `rope_parameters.rope_theta` takes precedence over a stale top-level value.
#[test]
fn rope_theta_nested_takes_precedence_over_top_level() {
    let dir = make_model_dir_with_config(
        r#"{
            "model_type": "k2_horizon",
            "architectures": ["K2HorizonForCausalLM"],
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 10240,
            "vocab_size": 250624,
            "max_position_embeddings": 524288,
            "rms_norm_eps": 1e-6,
            "layernorm_num_groups": 2,
            "tie_word_embeddings": false,
            "rope_theta": 10000.0,
            "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}
        }"#,
    );
    let manifest = introspect(&dir).unwrap();
    let attn = manifest.attention.as_ref().expect("attention config");
    assert_eq!(
        attn.rope_theta, 10_000_000.0,
        "nested rope_parameters.rope_theta must win over the top-level value"
    );
}
