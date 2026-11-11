//! Prefill pipeline orchestrator — runs all stages in sequence.

use anyhow::Result;
use tracing::info;

use crate::config::PrefillConfig;
use crate::prefix_cache::PrefixCache;
use crate::chunked_prefill::make_chunks;

/// Result of the full prefill pipeline.
#[derive(Debug)]
pub struct PrefillResult {
    /// Logits from the final prefill step.
    pub logits: Vec<f32>,
    /// Prompt tokens (user message only).
    pub prompt_tokens: usize,
    /// Tokens skipped via prefix cache.
    pub cached_tokens: usize,
    /// Tokens actually processed (forward passes).
    pub processed_tokens: usize,
    /// Number of chunks used.
    pub num_chunks: usize,
    /// Prefill time in ms.
    pub prefill_ms: u128,
}

impl Default for PrefillResult {
    fn default() -> Self {
        Self {
            logits: Vec::new(),
            prompt_tokens: 0,
            cached_tokens: 0,
            processed_tokens: 0,
            num_chunks: 0,
            prefill_ms: 0,
        }
    }
}

/// The prefill pipeline. Holds caches and config.
pub struct PrefillPipeline {
    pub config: PrefillConfig,
    pub prefix_cache: PrefixCache,
    state_counter: u64,
}

impl PrefillPipeline {
    pub fn new(config: PrefillConfig) -> Self {
        Self {
            prefix_cache: PrefixCache::new(config.prefix_cache_max_entries),
            config,
            state_counter: 0,
        }
    }

    /// Run the prefill pipeline.
    ///
    /// `full_tokens` = system + history + user (already assembled).
    /// `forward_fn` = model forward function for a chunk of tokens → logits.
    pub fn run<F>(
        &mut self,
        full_tokens: &[u32],
        forward_fn: &mut F,
    ) -> Result<PrefillResult>
    where
        F: FnMut(&[u32]) -> Result<Vec<f32>>,
    {
        let t0 = std::time::Instant::now();
        let total = full_tokens.len();

        // Stage 1: Prefix cache lookup — skip tokens already processed.
        let (cached_tokens, remaining_start) = if self.config.prefix_cache_enabled {
            match self.prefix_cache.lookup(full_tokens) {
                Some(hit) => (hit.token_count, hit.token_count),
                None => (0, 0),
            }
        } else {
            (0, 0)
        };

        let remaining = &full_tokens[remaining_start..];

        // Stage 2: Chunked batch prefill.
        let chunks = make_chunks(remaining, self.config.chunk_size);
        let num_chunks = chunks.len();
        let mut logits = Vec::new();

        for chunk in &chunks {
            logits = forward_fn(chunk)?;
        }

        // Stage 3: Store in prefix cache for the next request.
        self.state_counter += 1;
        self.prefix_cache.store(full_tokens, self.state_counter);

        let prefill_ms = t0.elapsed().as_millis();
        let processed = remaining.len();

        info!(
            target: "prefill",
            prompt = total,
            cached = cached_tokens,
            processed = processed,
            chunks = num_chunks,
            prefill_ms = prefill_ms,
            "prefill complete"
        );

        Ok(PrefillResult {
            logits,
            prompt_tokens: total,
            cached_tokens,
            processed_tokens: processed,
            num_chunks,
            prefill_ms,
        })
    }

    /// Run the prefill pipeline (simplified alias).
    pub fn run_simple<F>(
        &mut self,
        full_tokens: &[u32],
        forward_fn: &mut F,
    ) -> Result<PrefillResult>
    where
        F: FnMut(&[u32]) -> Result<Vec<f32>>,
    {
        self.run(full_tokens, forward_fn)
    }

    /// Reset the pipeline (clear caches).
    pub fn reset(&mut self) {
        self.prefix_cache.clear();
        self.state_counter = 0;
    }
}
