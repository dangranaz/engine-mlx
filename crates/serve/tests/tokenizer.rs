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
