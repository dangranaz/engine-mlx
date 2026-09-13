use nxm_modelplan::*;
use std::fs;
use std::io::Write;

fn make_test_model_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = base.join(format!("nxm_modelplan_introspect_{pid}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
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
