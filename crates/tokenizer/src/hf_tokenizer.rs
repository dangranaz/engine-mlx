//! HuggingFace tokenizers wrapper.
//!
//! Wraps the `tokenizers` crate (v0.23) to implement the `Tokenize` trait.
//! Loads from tokenizer.json files (BPE, Unigram, WordPiece, etc.).

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{Result, TokenizerError};
use crate::traits::Tokenize;

/// HuggingFace tokenizer wrapper.
pub struct HfTokenizer {
    inner: Tokenizer,
    eos_ids: Vec<u32>,
    vocab_size: usize,
}

impl HfTokenizer {
    /// Load from a tokenizer.json file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path.as_ref()).map_err(|e| {
            TokenizerError::Load(format!("{}: {}", path.as_ref().display(), e))
        })?;

        let vocab_size = inner.get_vocab_size(true);

        // Try to find EOS token(s)
        let mut eos_ids = Vec::new();
        for name in &["<|endoftext|>", "<|im_end|>", "</s>", "<eos>", "<|end|>"] {
            if let Some(id) = inner.token_to_id(name) {
                eos_ids.push(id);
            }
        }
        // Fallback: check added tokens
        if eos_ids.is_empty() {
            if let Some(id) = inner.token_to_id("<|endoftext|>") {
                eos_ids.push(id);
            }
        }

        Ok(Self {
            inner,
            eos_ids,
            vocab_size,
        })
    }

    /// Load from a model directory (looks for tokenizer.json).
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        if !path.exists() {
            return Err(TokenizerError::Load(format!(
                "tokenizer.json not found in {}",
                dir.as_ref().display()
            )));
        }
        Self::from_file(path)
    }

    /// Set EOS token IDs manually (override auto-detection).
    pub fn with_eos_ids(mut self, eos_ids: Vec<u32>) -> Self {
        self.eos_ids = eos_ids;
        self
    }
}

impl Tokenize for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Result<Vec<u32>> {
        // Use default chatml format
        let mut text = String::new();
        for (role, content) in messages {
            text.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        text.push_str("<|im_start|>assistant\n");
        self.encode(&text)
    }
}
