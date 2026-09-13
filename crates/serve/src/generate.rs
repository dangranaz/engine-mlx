//! MLX generate loop — prefill, first token, compiled decode.
//!
//! Split from `forward_qwen3.rs` per RULES (max 300 lines/file).

use anyhow::Result;
use anyhow::Context as _;

use crate::engine::Qwen3Engine;
use crate::sampler::sample_from_logits;

impl Qwen3Engine {
    #[cfg(feature = "mlx")]
    pub(crate) fn generate_via_mlx(&mut self, tokens: &[u32], max_tokens: usize) -> Result<String> {
        if tokens.is_empty() { anyhow::bail!("prompt produced no tokens"); }

        // Reset all per-sequence state and free the previous request's MLX
        // handles + buffer cache BEFORE building this request's state. This
        // mirrors the proven serve-mlx-c design: every request starts from a
        // clean allocator. The previous "reuse across requests" fast path kept
        // pre-computed masks / offset arrays / the compiled closure alive
        // between requests; combined with async_eval pipelining that starved
        // Metal's resource pool after a few requests (`[metal::malloc] Resource
        // limit exceeded`), making arrays come back empty and output degenerate
        // to garbage from ~the 3rd request. Rebuilding fresh each request costs
        // some peak throughput but is stable under sustained/varied load.
        self.offset = 0;
        #[cfg(feature = "mlx")]
        self.reset();

        // Static KV cache with pre-allocated buffers + masked SDPA.
        // This is the default — no flag needed — because it eliminates the
        // O(KV) degradation that concatenation KV cache causes at 300+ tokens
        // (78 t/s → 16.7 t/s on Qwen3-0.6B-4bit). Static KV uses
        // slice_update_dynamic for in-place writes (zero realloc) and an
        // additive attention mask instead of growing the buffer.
        let capacity = std::cmp::max(tokens.len() + max_tokens + 16, 256);

        let t0 = std::time::Instant::now();

        // Always rebuild fresh (no cross-request reuse — see note above).
        {
            self.alloc_kv_static(capacity)?;

            // Compile the static-KV closure. If compile fails, fall back to
            // concat KV + concat-compiled (decode_step handles the dispatch).
            // OPT-4: SplitStarts eval (tiny GPU, ~0.1ms) runs synchronously here;
            // the heavy tracing+compile (CPU, ~200ms) runs in a background thread
            // and overlaps with prefill.
            let cap = self.kv_static_cap;
            let compile_stream = self.stream;
            let weights_clone = self.weights.clone();
            let config_clone = self.config.clone();
            let starts = crate::compiled::CompiledStep::create_starts(compile_stream, &self.config)?;
            let compile_handle = std::thread::spawn(move || {
                crate::compiled::CompiledStep::from_starts(
                    weights_clone, config_clone, compile_stream, starts, cap,
                )
            });

            // ── Prefill (GPU) — overlaps with background compilation thread ──
            let x = self.embed(tokens)?;
            let logits = self.prefill(x, tokens.len())?;
            let prefill_ms = t0.elapsed().as_millis();
            match compile_handle.join() {
                Ok(Ok(closure)) => { self.compiled_static = Some(closure); }
                Ok(Err(e)) => {
                    tracing::warn!(
                        target: "engine_mlx::qwen3::forward",
                        "static compile failed, falling back to concat KV: {e:?}"
                    );
                    self.kv_static_cap = 0;
                    let x2 = self.embed(tokens)?;
                    let logits2 = self.prefill(x2, tokens.len())?;
                    return self.finish_generate(logits2, tokens, max_tokens, prefill_ms);
                }
                Err(_) => anyhow::bail!("compilation thread panicked"),
            }
            return self.finish_generate(logits, tokens, max_tokens, prefill_ms);
        }
    }

    /// Common tail of the generate loop: first token, decode loop, decode text.
    #[cfg(feature = "mlx")]
    fn finish_generate(&mut self, logits: engine_mlx_ffi::mlx_array, tokens: &[u32], max_tokens: usize, prefill_ms: u128) -> Result<String> {
        // Compute first token (greedy argmax on GPU, or CPU sampling if temp > 0)
        let mut current = if self.sampling.temperature < 1e-6 {
            let first_arr = self.ctx.argmax(logits)?;
            let first_vals = self.ctx.to_vec_u32(first_arr)?;
            // Free the argmax result; the token value is already read out.
            unsafe { engine_mlx_ffi::mlx_array_free(first_arr); }
            first_vals[0]
        } else {
            sample_from_logits(&self.ctx, logits, &self.sampling, &mut self.rng)?
        };
        // The prefill logits are consumed here; release the handle.
        unsafe { engine_mlx_ffi::mlx_array_free(logits); }
        self.offset = tokens.len();

        tracing::info!(
            target: "engine_mlx::qwen3::forward",
            "prefill done seq_len={} prefill_ms={}", tokens.len(), prefill_ms
        );

        let mut generated = vec![current];
        if crate::config::EOS_IDS.contains(&current) {
            return Ok(self.tokenizer.decode(&generated).context("tokenizer decode failed")?);
        }

        // Compile the concat-KV closure (fallback path if static failed).
        // The compiled decode step drives: launch → async_eval → single readback
        // → commit KV. This gives the pipelining needed for high throughput.
        if !self.is_compiled() && !self.is_compiled_static() {
            if let Err(e) = self.ensure_compiled() {
                tracing::info!(
                    target: "engine_mlx::qwen3::forward",
                    "compile failed (using eager decode) err={e:?}"
                );
            }
        }
        let decode_start = std::time::Instant::now();
        for i in 1..max_tokens {
            let next = self.decode_step(current)?;
            if crate::config::EOS_IDS.contains(&next) { break; }
            generated.push(next);
            current = next;
            if generated.len() >= max_tokens { break; }
        }
        let decode_ms = decode_start.elapsed().as_millis();
        let n_decode = generated.len().saturating_sub(1);
        let tps = if decode_ms > 0 { (n_decode as f64) / (decode_ms as f64 / 1000.0) } else { 0.0 };
        tracing::info!(
            target: "engine_mlx::qwen3::forward",
            "decode done n_tokens={} decode_ms={} tps={:.1}", n_decode, decode_ms, tps
        );
        Ok(self.tokenizer.decode(&generated).context("tokenizer decode failed")?)
    }
}
