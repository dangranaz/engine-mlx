//! Tokenizer — thin wrapper for chat encode/decode.
//!
//! Uses `nxm_tokenizer::PureTokenizer` (pure Rust, 1.5× faster than HfTokenizer)
//! with interior mutability for the LRU cache. Falls back to HfTokenizer only
//! if Pure load fails (should not happen for Qwen3 BPE).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use nxm_tokenizer::PureTokenizer;

/// Opaque tokenizer handle used by the engine.
///
/// Wraps `PureTokenizer` (BPE/WordPiece/Unigram, LRU 10k cache) via `Mutex`
/// to satisfy `Tokenize: Send+Sync` (`&self` encode). HfTokenizer kept as
/// dev fallback for parity tests only.
pub struct Tokenizer {
    inner: Mutex<PureTokenizer>,
    vocab_size: usize,
    eos_ids: Vec<u32>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl Tokenizer {
    /// Load tokenizer from `model_dir/tokenizer.json` via PureTokenizer.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        if !path.exists() {
            anyhow::bail!("tokenizer.json not found in {}", model_dir.display());
        }
        let inner = PureTokenizer::load(&path)
            .with_context(|| format!("failed to load tokenizer.json from {}", path.display()))?;
        // Vocab size from file (pure's vocab is private)
        let vocab_size = Self::vocab_size_from_file(&path).unwrap_or(151936);
        // EOS ids resolved from the model (generation_config.json > config.json >
        // tokenizer.json added_tokens > EOS_IDS constant fallback), never a
        // model-specific hardcode in the decode path.
        let eos_ids = Self::resolve_eos_ids_from_dir(model_dir);
        tracing::info!(
            target: "engine_mlx::qwen3::tokenizer",
            "tokenizer loaded (Pure) vocab_size={} eos={:?} path={}",
            vocab_size,
            eos_ids,
            path.display()
        );
        Ok(Self {
            inner: Mutex::new(inner),
            vocab_size,
            eos_ids,
        })
    }

    fn vocab_size_from_file(path: &Path) -> Result<usize> {
        let content = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(dict) = v.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
            Ok(dict.len())
        } else if let Some(list) = v.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_array()) {
            Ok(list.len())
        } else {
            anyhow::bail!("vocab not found")
        }
    }

    fn eos_ids_from_file(path: &Path) -> Vec<u32> {
        // Qwen3 fixed, fallback to file scan
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap_or(serde_json::Value::Null);
        let mut ids = Vec::new();
        if let Some(arr) = v.get("added_tokens").and_then(|v| v.as_array()) {
            for tok in arr {
                if let (Some(content), Some(id)) = (tok.get("content").and_then(|c| c.as_str()), tok.get("id").and_then(|i| i.as_u64())) {
                    if matches!(content, "<|im_end|>" | "<|endoftext|>" | "</s>") {
                        ids.push(id as u32);
                    }
                }
            }
        }
        ids
    }

    /// Read a numeric `eos_token_id` (scalar or list) from a JSON config file.
    /// Returns an empty Vec when the file/key is missing or malformed.
    fn eos_ids_from_config(path: &Path) -> Vec<u32> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let v: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|id| vec![id as u32]).unwrap_or_default()
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|x| x.as_u64().map(|id| id as u32))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Resolve the model's EOS ids at load time, in priority order:
    /// `generation_config.json` -> `config.json` -> tokenizer.json added_tokens
    /// -> the Qwen3 `EOS_IDS` constant as last resort (documented fallback for
    /// model dirs that ship no EOS source at all). The decode path uses these,
    /// so EOS provenance is the loaded model, never a per-model hardcode.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use engine_mlx_serve::tokenizer::Tokenizer;
    /// let dir = std::path::Path::new("/tmp/nxm-model");
    /// let eos = Tokenizer::resolve_eos_ids_from_dir(dir);
    /// // K2-style generation_config.json wins: eos_token_id = [1, 250019]
    /// assert_eq!(eos, vec![1, 250019]);
    /// ```
    pub fn resolve_eos_ids_from_dir(model_dir: &Path) -> Vec<u32> {
        let generation_config = model_dir.join("generation_config.json");
        let config_json = model_dir.join("config.json");
        let tokenizer_json = model_dir.join("tokenizer.json");
        let mut ids = Self::eos_ids_from_config(&generation_config);
        if ids.is_empty() {
            ids = Self::eos_ids_from_config(&config_json);
        }
        if ids.is_empty() {
            ids = Self::eos_ids_from_file(&tokenizer_json);
        }
        if ids.is_empty() {
            tracing::warn!(
                target: "engine_mlx::qwen3::tokenizer",
                "no eos_token_id found in {} (falling back to Qwen3 EOS_IDS constant)",
                model_dir.display()
            );
            ids = crate::config::EOS_IDS.to_vec();
        }
        ids
    }

    /// Encode raw text to token IDs (with special tokens).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| anyhow::anyhow!("tokenizer mutex poisoned: {e}"))?;
        // PureTokenizer::encode is infallible & &mut
        Ok(guard.encode(text))
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| anyhow::anyhow!("tokenizer mutex poisoned: {e}"))?;
        Ok(guard.decode(ids))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
}

/// Decode helper mirroring historic `tokenizer::decode`.
pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer.decode(ids)
}

/// Convenience: load tokenizer directly from a `tokenizer.json` path.
pub fn load_from_file(path: &Path) -> Result<Tokenizer> {
    let inner = PureTokenizer::load(path)
        .with_context(|| format!("failed to load tokenizer from {}", path.display()))?;
    let vocab_size = Tokenizer::vocab_size_from_file(path).unwrap_or(151936);
    let mut eos_ids = Tokenizer::eos_ids_from_file(path);
    if eos_ids.is_empty() {
        // No model dir to resolve from: documented EOS_IDS constant fallback.
        eos_ids = crate::config::EOS_IDS.to_vec();
    }
    Ok(Tokenizer {
        inner: Mutex::new(inner),
        vocab_size,
        eos_ids,
    })
}


