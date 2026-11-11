//! Interface trait for prefill pipeline operations.

use anyhow::Result;

use crate::pipeline::PrefillResult;

/// Trait defining the prefill pipeline contract.
pub trait PrefillPipelineTrait {
    /// Run the prefill pipeline.
    ///
    /// `full_tokens` = system + history + user tokens.
    /// `forward_fn` = model forward function.
    fn run<F>(
        &mut self,
        full_tokens: &[u32],
        forward_fn: &mut F,
    ) -> Result<PrefillResult>
    where
        F: FnMut(&[u32]) -> Result<Vec<f32>>;

    /// Reset the pipeline (clear caches).
    fn reset(&mut self);
}
