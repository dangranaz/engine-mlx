//! Unit tests for `engine_mlx_serve::tokenizer` module.

use engine_mlx_serve::tokenizer::Tokenizer;

#[test]
fn tokenizer_load_missing_returns_error() {
    let dir = std::env::temp_dir().join("nxm_tokenizer_missing_test2");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::remove_file(dir.join("tokenizer.json"));
    let err = Tokenizer::load(&dir).unwrap_err();
    assert!(err.to_string().contains("tokenizer.json not found"));
}

#[test]
fn tokenizer_load_invalid_json_bails() {
    let dir = std::env::temp_dir().join("nxm_tokenizer_invalid_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("tokenizer.json");
    let _ = std::fs::write(&path, "{ invalid json");
    let err = Tokenizer::load(&dir).unwrap_err();
    // Should be a parse error wrapped with context
    assert!(!err.to_string().is_empty());
}

/// R3 (task 18): EOS ids are resolved from the loaded model's configs, not
/// from a per-model hardcode. Priority: generation_config.json > config.json
/// > tokenizer.json added_tokens > EOS_IDS constant (documented last resort).
#[test]
fn eos_ids_resolved_from_model() {
    let cases: &[(&str, &str, &str, Vec<u32>)] = &[
        // (dir name, config file name, content, expected eos ids)
        (
            "nxm_eos_k2_style_test",
            "generation_config.json",
            r#"{"eos_token_id": [1, 250019]}"#,
            vec![1, 250019],
        ),
        (
            "nxm_eos_qwen_style_test",
            "config.json",
            r#"{"eos_token_id": 151645}"#,
            vec![151645],
        ),
    ];
    for (dir_name, file_name, content, expected) in cases {
        let dir = std::env::temp_dir().join(dir_name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file_name), content).unwrap();
        assert_eq!(
            Tokenizer::resolve_eos_ids_from_dir(&dir),
            *expected,
            "{file_name} eos_token_id must resolve to {expected:?}",
        );
    }

    // Fallback: no EOS source anywhere -> the documented EOS_IDS constant.
    let empty_dir = std::env::temp_dir().join("nxm_eos_empty_test");
    let _ = std::fs::remove_dir_all(&empty_dir);
    std::fs::create_dir_all(&empty_dir).unwrap();
    assert_eq!(
        Tokenizer::resolve_eos_ids_from_dir(&empty_dir),
        engine_mlx_serve::config::EOS_IDS.to_vec(),
        "no EOS source must fall back to the documented EOS_IDS constant",
    );
}

