//! Tokenizer trait — the core contract for text ↔ token conversion.
//!
//! Implementations:
//! - `HfTokenizer` (this crate) — wraps HuggingFace `tokenizers` crate
//! - Future: Perplexity Unigram (5× faster, trie-based, zero-copy, SIMD)

use crate::error::Result;

/// Tokenizer contract — encode text to tokens and back.
///
/// All engines need tokenization. This trait ensures the server
/// and all backends can tokenize without knowing the specific implementation.
pub trait Tokenize: Send + Sync {
    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text.
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// End-of-sequence token ID(s).
    fn eos_ids(&self) -> &[u32];

    /// Beginning-of-sequence token ID (if applicable).
    fn bos_id(&self) -> Option<u32> {
        None
    }

    /// Encode without special tokens.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text)
    }

    /// Get the token string for an ID (for debugging).
    fn id_to_token(&self, id: u32) -> Option<String>;
}
