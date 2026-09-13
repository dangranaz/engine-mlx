//! Tokenizer — thin wrapper for chat encode/decode.
//!
//! Uses `nxm_tokenizer::PureTokenizer` (pure Rust, 1.5× faster than HfTokenizer)
//! with interior mutability for the LRU cache. Falls back to HfTokenizer only
//! if Pure load fails (should not happen for Qwen3 BPE).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use nxm_tokenizer::{PureTokenizer, Tokenize as _};

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
        let eos_ids = Self::eos_ids_from_file(&path);
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
        if ids.is_empty() {
            ids = vec![151643, 151645];
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

/// Build a ChatML prompt and encode it.
///
/// Mirrors historic `tokenizer::encode_chat` which injects `/no_think` system
/// prompt to disable thinking mode for faster generation.
pub fn encode_chat(tokenizer: &Tokenizer, messages: &[(String, String)]) -> Result<Vec<u32>> {
    let mut prompt = String::new();
    // System message disables thinking mode for faster generation
    prompt.push_str("<|im_start|>system\n/no_think<|im_end|>\n");
    for (role, content) in messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    tokenizer
        .encode(&prompt)
        .with_context(|| "encode_chat failed")
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
    let eos_ids = Tokenizer::eos_ids_from_file(path);
    Ok(Tokenizer {
        inner: Mutex::new(inner),
        vocab_size,
        eos_ids,
    })
}


