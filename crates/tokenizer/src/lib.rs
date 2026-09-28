//! # nxm-tokenizer
//!
//! Tokenizer trait and HuggingFace wrapper for engine-mlx inference.
//!
//! Provides:
//! - `Tokenize` trait — the contract for text ↔ token conversion
//! - `HfTokenizer` — production implementation wrapping `tokenizers` v0.23
//!
//! Loads tokenizer.json (BPE, Unigram, WordPiece) from model directories.

pub mod error;
pub mod hf_tokenizer;
pub mod pure;
pub mod traits;

pub use error::{Result, TokenizerError};
pub use hf_tokenizer::HfTokenizer;
pub use pure::Tokenizer as PureTokenizer;
pub use traits::Tokenize;
