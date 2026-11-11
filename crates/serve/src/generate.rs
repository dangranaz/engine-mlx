//! MLX generate loop — prefill, first token, compiled decode.
//!
//! Split from `forward_qwen3.rs` per RULES (max 300 lines/file).

use anyhow::Result;
use anyhow::Context as _;

use crate::engine::Qwen3Engine;
use crate::forward::exercise_mlx;
use crate::sampler::sample_from_logits;

impl Qwen3Engine {
    #[cfg(feature = "mlx")]
    pub(crate) fn generate_via_mlx(&mut self, tokens: &[u32], max_tokens: usize) -> Result<String> {
        if tokens.is_empty() { anyhow::bail!("prompt produced no tokens"); }

        // Per-request state reset is owned by `generate` (called before every
        // path, including this one).

        // Fixed-capacity static KV (E5 fix): the closure is compiled ONCE at
        // MAX_CAP during engine init. We only need to zero the KV data and
        // reset offset for each request. Requests exceeding MAX_CAP are
        // rejected with a clear error (they would require recompilation,
        // defeating the leak fix).
        let required = tokens.len() + max_tokens + 16;
        if required > self.kv_static_cap {
            anyhow::bail!(
                "request requires {} tokens (prompt {} + max_tokens {} + 16) > MAX_CAP {}; reduce max_tokens or use a model with larger context",
                required, tokens.len(), max_tokens, self.kv_static_cap
            );
        }

        // KV buffers and offset already reset by reset() (called by generate()).
        let t0 = std::time::Instant::now();

        // ── Prefill (GPU) ──
        let x = self.embed(tokens)?;
        let logits = self.prefill(x, tokens.len())?;
        let prefill_ms = t0.elapsed().as_millis();

        // After prefill, seed the static mask: positions 0..=prompt_len-1 are valid (0.0),
        // positions >= prompt_len are masked (-1e9). This ensures the prompt is "seen"
        // by attention (fixes the bug where prompt positions were masked).
        self.update_static_mask_cpu(tokens.len().saturating_sub(1))?;

        self.finish_generate(logits, tokens, max_tokens, prefill_ms)
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
        if self.is_eos(current) {
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
            if self.is_eos(next) { break; }
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

impl Qwen3Engine {

    /// True when `id` is one of the loaded model's EOS ids (provenance:
    /// `Tokenizer::eos_ids()`, resolved from the model at load time — never a
    /// per-model hardcode in the decode path).
    fn is_eos(&self, id: u32) -> bool {
        self.tokenizer.eos_ids().contains(&id)
    }

    /// Generate from a raw prompt string (no chat templating).
    ///
    /// Used by the `main.rs` self-test and by any caller that already has a
    /// fully-built prompt. Does **not** touch the chat template.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Per-request state reset: every generation starts from a clean logical
        // position and cycle state, regardless of the previous request's
        // outcome (static KV allocation is rebuilt per request in the mlx path;
        // reset clears it consistently with the path in use). This runs before
        // any early return so every request — including invalid ones — starts
        // from a clean state.
        self.reset();
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::generate START prompt_len={} max_tokens={}",
            prompt.len(),
            max_tokens
        );
        if max_tokens == 0 {
            anyhow::bail!("max_tokens must be > 0");
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .with_context(|| "tokenizer encode failed")?;
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt produced no tokens");
        }

        // Try real eager forward when mlx is available.
        #[cfg(feature = "mlx")]
        {
            if self.embed_table.is_some() {
                let toks = prompt_tokens.clone();
                match self.generate_via_mlx(&toks, max_tokens) {
                    Ok(text) => {
                        tracing::info!(
                            target: "engine_mlx::qwen3::engine",
                            "Qwen3Engine::generate END (mlx) offset={}",
                            self.offset
                        );
                        return Ok(text);
                    }
                    Err(e) => {
                        tracing::info!(
                            target: "engine_mlx::qwen3::engine",
                            "mlx forward failed, falling back to echo err={e:?}"
                        );
                        self.kv_bufs.clear();
                        self.offset = 0;
                    }
                }
            }
        }

        // Tracer fallback: echo last prompt token.
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        let fallback = *prompt_tokens.last().context("prompt_tokens empty after check")?;
        for _ in 0..max_tokens {
            let _ = exercise_mlx(
                &self.ctx,
                &self.config,
                &self.weights,
                self.embed_table,
                fallback,
            );
            let next = fallback;
            if self.is_eos(next) {
                break;
            }
            generated.push(next);
            self.offset += 1;
            if generated.len() >= max_tokens {
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(&generated)
            .with_context(|| "tokenizer decode failed")?;
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::generate END (fallback) generated={} offset={}",
            generated.len(),
            self.offset
        );
        Ok(text)
    }
}
