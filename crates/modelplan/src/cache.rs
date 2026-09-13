//! Side‑car SHA‑256 cache for the model manifest.
//!
//! When a `model.plan.toml` is generated the crate computes a SHA‑256
//! fingerprint over the source files (config.json, tokenizer config files
//! and safetensors index files) and stores it as
//! `<model_dir>/model.plan.toml.sha256`.
//!
//! Subsequent calls to `load_plan` recompute the hash and compare it to
//! the side‑car; a mismatch indicates the manifest is stale and must be
//! regenerated.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub const SIDECAR_NAME: &str = "model.plan.toml.sha256";

/// Compute SHA‑256 of the source files that influence the manifest.
///
/// Files included (in deterministic order):
/// 1. `config.json` – required.
/// 2. `tokenizer.json`, `tokenizer_config.json` – if present.
/// 3. `*.safetensors.index.json` – if present (sharded models).
pub fn hash_sources(model_dir: &Path) -> Result<String> {
    let config_path = model_dir.join("config.json");
    let config_bytes = fs::read(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"config.json\n");
    hasher.update(&config_bytes);

    // Tokenizer files – optional, sorted.
    let mut tokenizer_files: Vec<String> = Vec::new();
    if model_dir.join("tokenizer.json").exists() {
        tokenizer_files.push("tokenizer.json".to_string());
    }
    if model_dir.join("tokenizer_config.json").exists() {
        tokenizer_files.push("tokenizer_config.json".to_string());
    }
    tokenizer_files.sort();
    for name in &tokenizer_files {
        let p = model_dir.join(name);
        let bytes = fs::read(&p)
            .with_context(|| format!("Failed to read {}", p.display()))?;
        hasher.update(name.as_bytes());
        hasher.update(b"\n");
        hasher.update(&bytes);
    }

    // Safetensors index files – optional, sorted.
    let mut index_files: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
                if fname.ends_with(".safetensors.index.json") {
                    index_files.push(fname.to_string());
                }
            }
        }
    }
    index_files.sort();
    for name in &index_files {
        let p = model_dir.join(name);
        let bytes = fs::read(&p)
            .with_context(|| format!("Failed to read {}", p.display()))?;
        hasher.update(name.as_bytes());
        hasher.update(b"\n");
        hasher.update(&bytes);
    }

    let digest = hasher.finalize();
    Ok(hex::encode_lower(digest))
}

/// Write the side‑car file containing `hash`.
pub fn write_sidecar(model_dir: &Path, hash: &str) -> Result<()> {
    let path = model_dir.join(SIDECAR_NAME);
    fs::write(&path, format!("{hash}\n"))
        .with_context(|| format!("Failed to write sidecar {}", path.display()))?;
    Ok(())
}

/// Read the side‑car file if present.
pub fn read_sidecar(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join(SIDECAR_NAME);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read sidecar {}", path.display()))?;
    Ok(Some(content.trim().to_string()))
}

/// Returns `true` if the side‑car hash matches the recomputed hash,
/// `false` if missing or stale.
pub fn validate_hash(model_dir: &Path) -> Result<bool> {
    let stored = match read_sidecar(model_dir)? {
        Some(h) => h,
        None => return Ok(false),
    };
    let current = hash_sources(model_dir)?;
    Ok(stored == current)
}

// `hex` is not yet a declared dependency. Add a tiny local helper to avoid
// pulling the `hex` crate; hex‑encode a SHA‑256 digest (32 bytes → 64 chars).
mod hex {
    pub fn encode_lower(bytes: impl AsRef<[u8]>) -> String {
        const HEX: &[u8] = b"0123456789abcdef";
        let bytes = bytes.as_ref();
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_temp_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let base = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        // Per-call counter guarantees uniqueness even across parallel test
        // threads in the same process (same pid, possibly same nanos).
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!("nxm_modelplan_test_{pid}_{nanos}_{seq}"));
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
        assert_eq!(h1.len(), 64); // SHA‑256 hex = 64 chars
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
}
