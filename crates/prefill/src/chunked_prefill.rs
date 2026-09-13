//! Chunked batch prefill — processes tokens in fixed-size chunks.
//!
//! Instead of processing 1 token per forward pass (O(n) calls),
//! processes chunk_size tokens per call (O(n/chunk) calls).
//! Each chunk benefits from MLX kernel fusion internally.

/// Result of chunked prefill.
pub struct ChunkedPrefillResult {
    /// Final logits from last chunk (for first token sampling).
    pub logits: Vec<f32>,
    /// Total tokens processed.
    pub tokens_processed: usize,
    /// Number of chunks used.
    pub num_chunks: usize,
}

/// Split tokens into chunks for batch prefill.
pub fn make_chunks(tokens: &[u32], chunk_size: usize) -> Vec<&[u32]> {
    tokens.chunks(chunk_size).collect()
}

/// Compute optimal chunk size based on model and hardware.
/// Larger chunks = fewer forward passes but more memory per pass.
pub fn optimal_chunk_size(total_tokens: usize, head_dim: usize, num_layers: usize) -> usize {
    // Heuristic: KV cache memory per chunk = chunk_size * num_layers * 2 * head_dim * sizeof(f16)
    // On M1 16GB, keep chunk KV under ~500MB
    let kv_per_token = num_layers * 2 * head_dim * 2; // bytes per token in KV
    let max_kv_bytes = 512 * 1024 * 1024; // 512 MB
    let max_chunk = max_kv_bytes / kv_per_token;

    // Clamp between 64 and 1024, round to power of 2
    let clamped = max_chunk.clamp(64, 1024);
    let rounded = clamped.next_power_of_two() / 2; // conservative

    // Don't make chunks bigger than the total
    rounded.min(total_tokens)
}
