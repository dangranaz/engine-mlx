use nxm_modelplan::cache::{hash_sources, read_sidecar, validate_hash, write_sidecar};
use std::fs;
use std::io::Write;
use std::path::Path;

fn make_temp_dir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = base.join(format!("nxm_modelplan_test_{pid}_{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let p = dir.join(name);
    let mut f = fs::File::create(&p).unwrap();
    f.write_all(content.as_bytes()).unwrap();
}

#[test]
fn hash_is_deterministic() {
    let dir = make_temp_dir();
    write_file(&dir, "config.json", "{\"a\":1}");
    let h1 = hash_sources(&dir).unwrap();
    let h2 = hash_sources(&dir).unwrap();
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64);
}

#[test]
fn hash_changes_when_source_changes() {
    let dir = make_temp_dir();
    write_file(&dir, "config.json", "{\"a\":1}");
    let h1 = hash_sources(&dir).unwrap();
    write_file(&dir, "config.json", "{\"a\":2}");
    let h2 = hash_sources(&dir).unwrap();
    assert_ne!(h1, h2);
}

#[test]
fn sidecar_roundtrip() {
    let dir = make_temp_dir();
    write_file(&dir, "config.json", "{\"a\":1}");
    let h = hash_sources(&dir).unwrap();
    write_sidecar(&dir, &h).unwrap();
    let read = read_sidecar(&dir).unwrap();
    assert_eq!(read, Some(h.clone()));
    assert!(validate_hash(&dir).unwrap());
}

#[test]
fn validate_hash_detects_mismatch() {
    let dir = make_temp_dir();
    write_file(&dir, "config.json", "{\"a\":1}");
    write_sidecar(&dir, "deadbeef").unwrap();
    assert!(!validate_hash(&dir).unwrap());
}

#[test]
fn validate_hash_missing_sidecar() {
    let dir = make_temp_dir();
    write_file(&dir, "config.json", "{\"a\":1}");
    assert!(!validate_hash(&dir).unwrap());
}
