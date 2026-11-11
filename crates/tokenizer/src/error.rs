//! Error types for tokenizer operations.

use thiserror::Error;

/// Errors from tokenizer operations.
#[derive(Error, Debug)]
pub enum TokenizerError {
    /// Failed to load tokenizer file.
    #[error("tokenizer load error: {0}")]
    Load(String),

    /// Failed to encode text.
    #[error("encode error: {0}")]
    Encode(String),

    /// Failed to decode tokens.
    #[error("decode error: {0}")]
    Decode(String),
}

/// Result type for tokenizer operations.
pub type Result<T> = std::result::Result<T, TokenizerError>;
