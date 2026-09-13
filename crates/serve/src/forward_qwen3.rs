//! Qwen3 forward — eager prefill + decode (not compiled).

use anyhow::Result;
use engine_mlx_ffi::mlx_array;

use crate::engine::Qwen3Engine;
use crate::sampler::sample_from_logits;

impl Qwen3Engine {
    #[cfg(feature = "mlx")]
    pub fn prefill(&mut self, input: mlx_array, seq_len: usize) -> Result<mlx_array> {
        tracing::info!(
            target: "engine_mlx::qwen3::forward",
            "prefill START seq_len={} offset={} kv_bufs={}",
            seq_len,
            self.offset,
            self.kv_bufs.len()
        );
        let t = seq_len as i32;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        let offset_arr = self.ctx.new_array_i32(&[self.offset as i32])?;
        let scale = 1.0 / (hd as f32).sqrt();
        let (group_size, bits) = self
            .config
            .quantization
            .as_ref()
            .map(|q| (q.group_size as i32, q.bits as i32))
            .unwrap_or((64, 4));
        let mut x = input;
        let nl = self.weights.layers.len();
        // Arena of owned intermediate handles created during this prefill.
        // mlx_array is a refcounted C handle; every op returns a NEW handle and
        // nothing frees them, so a single prefill leaked ~20 arrays/layer plus
        // the replaced KV buffers — hundreds of live MLX resources per request
        // that accumulate until `[metal::malloc] Resource limit (499000)` fires.
        // We collect every intermediate we own here and free them once the graph
        // for `logits` is built; MLX output arrays retain their inputs through
        // graph edges (shared_ptr), so releasing our copies is safe. Weights and
        // stored `self.kv_bufs` entries are NEVER pushed here.
        let mut scratch: Vec<mlx_array> = Vec::with_capacity(nl * 24 + 8);
        // KV buffers we replaced this prefill. They stay referenced by the new
        // (lazy) kv_bufs until those are evaluated, so we free them only after
        // the tail eval below.
        let mut stale_kv: Vec<mlx_array> = Vec::with_capacity(nl * 2);
        scratch.push(offset_arr);
        scratch.push(input); // the per-request embedding is ours to free
        macro_rules! keep { ($e:expr) => {{ let a = $e; scratch.push(a); a }} }
        // Production path always uses static or concat KV — int4 is debug-only
        // (see tests/int4_debug.rs). No int4 branch in prefill.
        let is_static = self.kv_static_cap > 0 && (self.kv_bufs.len() == nl * 2 || self.kv_bufs.len() == nl * 6);
        if !is_static {
            self.kv_bufs.clear();
            self.kv_bufs.reserve(nl * 2);
        }
        for li in 0..nl {
            let lw = &self.weights.layers[li];
            let normed = keep!(self.ctx.rms_norm(x, lw.input_layernorm, self.config.rms_norm_eps)?);
            let qkv = keep!(self.ctx.quantized_matmul(normed, lw.qkv_proj_w, lw.qkv_proj_s, lw.qkv_proj_b, true, group_size, bits)?);
            let q = keep!(self.ctx.slice(qkv, &[0, 0, 0], &[1, t, nh * hd], &[1, 1, 1])?);
            let k = keep!(self.ctx.slice(qkv, &[0, 0, nh * hd], &[1, t, nh * hd + nkv * hd], &[1, 1, 1])?);
            let v = keep!(self.ctx.slice(qkv, &[0, 0, nh * hd + nkv * hd], &[1, t, nh * hd + 2 * nkv * hd], &[1, 1, 1])?);
            let q = keep!(self.ctx.reshape(q, &[1, t, nh, hd])?);
            let mut k = keep!(self.ctx.reshape(k, &[1, t, nkv, hd])?);
            let mut v = keep!(self.ctx.reshape(v, &[1, t, nkv, hd])?);
            let q = if let Some(qn) = lw.q_norm { keep!(self.ctx.rms_norm(q, qn, self.config.rms_norm_eps)?) } else { q };
            k = if let Some(kn) = lw.k_norm { keep!(self.ctx.rms_norm(k, kn, self.config.rms_norm_eps)?) } else { k };
            let q = keep!(self.ctx.transpose_axes(q, &[0, 2, 1, 3])?);
            k = keep!(self.ctx.transpose_axes(k, &[0, 2, 1, 3])?);
            v = keep!(self.ctx.transpose_axes(v, &[0, 2, 1, 3])?);
            let q = keep!(self.ctx.rope_dynamic(q, hd, self.config.rope_theta, offset_arr)?);
            k = keep!(self.ctx.rope_dynamic(k, hd, self.config.rope_theta, offset_arr)?);
            if is_static && self.kv_bufs.len() == nl * 6 {
                // int4 KV path (debug-only, see tests/int4_debug.rs): 6 buffers/layer.
                // Store into int4 buffers: quantize each position and slice_update
                let base = li * 6;
                let mut k_data = self.kv_bufs[base];
                let mut k_scales = self.kv_bufs[base + 1];
                let mut k_biases = self.kv_bufs[base + 2];
                let mut v_data = self.kv_bufs[base + 3];
                let mut v_scales = self.kv_bufs[base + 4];
                let mut v_biases = self.kv_bufs[base + 5];
                for pos in 0..t {
                    let p = self.offset as i32 + pos;
                    let p_arr = keep!(self.ctx.new_array_i32(&[p])?);
                    // slice [1,nkv,1,hd] at position p
                    let k_p = keep!(self.ctx.slice(k, &[0, 0, pos, 0], &[1, nkv, pos + 1, hd], &[1, 1, 1, 1])?);
                    let v_p = keep!(self.ctx.slice(v, &[0, 0, pos, 0], &[1, nkv, pos + 1, hd], &[1, 1, 1, 1])?);
                    // reshape to [1,nkv,1,hd] for quantize (already that shape after slice)
                    let k_p = keep!(self.ctx.reshape(k_p, &[1, nkv, 1, hd])?);
                    let v_p = keep!(self.ctx.reshape(v_p, &[1, nkv, 1, hd])?);
                    let (k_qd, k_qs, k_qb) = self.ctx.quantize_kv_native(k_p, 64, 4)?;
                    let (v_qd, v_qs, v_qb) = self.ctx.quantize_kv_native(v_p, 64, 4)?;
                    scratch.push(k_qd); scratch.push(k_qs); scratch.push(k_qb);
                    scratch.push(v_qd); scratch.push(v_qs); scratch.push(v_qb);
                    // slice_update_dynamic replaces the running buffer; the previous
                    // intermediate (not the original cache) is owned — free via scratch.
                    let nk_data = self.ctx.slice_update_dynamic(k_data, k_qd, p_arr, &[2])?;
                    if pos > 0 { scratch.push(k_data); } k_data = nk_data;
                    let nk_scales = self.ctx.slice_update_dynamic(k_scales, k_qs, p_arr, &[2])?;
                    if pos > 0 { scratch.push(k_scales); } k_scales = nk_scales;
                    let nk_biases = self.ctx.slice_update_dynamic(k_biases, k_qb, p_arr, &[2])?;
                    if pos > 0 { scratch.push(k_biases); } k_biases = nk_biases;
                    let nv_data = self.ctx.slice_update_dynamic(v_data, v_qd, p_arr, &[2])?;
                    if pos > 0 { scratch.push(v_data); } v_data = nv_data;
                    let nv_scales = self.ctx.slice_update_dynamic(v_scales, v_qs, p_arr, &[2])?;
                    if pos > 0 { scratch.push(v_scales); } v_scales = nv_scales;
                    let nv_biases = self.ctx.slice_update_dynamic(v_biases, v_qb, p_arr, &[2])?;
                    if pos > 0 { scratch.push(v_biases); } v_biases = nv_biases;
                }
                // Free the ORIGINAL int4 cache buffers we are replacing.
                // Free the ORIGINAL int4 cache buffers we are replacing — but
                // only AFTER the new buffers are evaluated (deferred below), since
                // slice_update_dynamic output depends on the original buffer until
                // the graph is materialized. Record them as stale for now.
                stale_kv.push(self.kv_bufs[base]);
                stale_kv.push(self.kv_bufs[base + 1]);
                stale_kv.push(self.kv_bufs[base + 2]);
                stale_kv.push(self.kv_bufs[base + 3]);
                stale_kv.push(self.kv_bufs[base + 4]);
                stale_kv.push(self.kv_bufs[base + 5]);
                self.kv_bufs[base] = k_data;
                self.kv_bufs[base + 1] = k_scales;
                self.kv_bufs[base + 2] = k_biases;
                self.kv_bufs[base + 3] = v_data;
                self.kv_bufs[base + 4] = v_scales;
                self.kv_bufs[base + 5] = v_biases;
            } else if is_static {
                // Static KV: single bulk slice_update per layer (no per-position loop).
                // `k`/`v` already have shape [1,nkv,t,hd] after transpose+rope;
                // write them into cache at [0,0,base,0]→[1,nkv,base+t,hd] in one op.
                let base = self.offset as i32;
                let old_k = self.kv_bufs[li * 2];
                let old_v = self.kv_bufs[li * 2 + 1];
                self.kv_bufs[li * 2] = self.ctx.slice_update(
                    old_k, k, &[0, 0, base, 0], &[1, nkv, base + t, hd], &[1, 1, 1, 1],
                )?;
                self.kv_bufs[li * 2 + 1] = self.ctx.slice_update(
                    old_v, v, &[0, 0, base, 0], &[1, nkv, base + t, hd], &[1, 1, 1, 1],
                )?;
                // The new kv_bufs entries are lazy and still reference `old_k`/
                // `old_v` until evaluated. Defer freeing until after the tail
                // eval (below); freeing here corrupts the graph (closure apply
                // fails status 1). Record as stale.
                stale_kv.push(old_k);
                stale_kv.push(old_v);
            } else {
                // Concat KV: k/v (the rope outputs) become the stored cache, so
                // remove them from the scratch arena to avoid freeing live cache.
                scratch.retain(|&a| a.ctx != k.ctx && a.ctx != v.ctx);
                self.kv_bufs.push(k);
                self.kv_bufs.push(v);
            }
            let attn = keep!(self.ctx.sdpa(q, k, v, scale, true)?);
            let attn = keep!(self.ctx.transpose_axes(attn, &[0, 2, 1, 3])?);
            let attn = keep!(self.ctx.reshape(attn, &[1, t, nh * hd])?);
            let o = keep!(self.ctx.quantized_matmul(attn, lw.o_proj_w, lw.o_proj_s, lw.o_proj_b, true, group_size, bits)?);
            let x_res = keep!(self.ctx.add(x, o)?);
            let normed2 = keep!(self.ctx.rms_norm(x_res, lw.post_attention_layernorm, self.config.rms_norm_eps)?);
            let gup = keep!(self.ctx.quantized_matmul(normed2, lw.gate_up_proj_w, lw.gate_up_proj_s, lw.gate_up_proj_b, true, group_size, bits)?);
            let ff = self.config.intermediate_size;
            let gate = keep!(self.ctx.slice(gup, &[0, 0, 0], &[1, t, ff], &[1, 1, 1])?);
            let up = keep!(self.ctx.slice(gup, &[0, 0, ff], &[1, t, 2 * ff], &[1, 1, 1])?);
            let silu = keep!(self.ctx.silu(gate)?);
            let hidden = keep!(self.ctx.multiply(silu, up)?);
            let down = keep!(self.ctx.quantized_matmul(hidden, lw.down_proj_w, lw.down_proj_s, lw.down_proj_b, true, group_size, bits)?);
            x = keep!(self.ctx.add(x_res, down)?);
        }
        x = keep!(self.ctx.rms_norm(x, self.weights.final_norm, self.config.rms_norm_eps)?);
        let last = keep!(self.ctx.slice(x, &[0, t - 1, 0], &[1, t, self.config.hidden_size], &[1, 1, 1])?);
        let last = keep!(self.ctx.reshape(last, &[1, self.config.hidden_size])?);
        let (lm_w, lm_s, lm_b) = match (&self.weights.lm_head_w, &self.weights.lm_head_s, &self.weights.lm_head_b) {
            (Some(hw), Some(hs), Some(hb)) => (*hw, *hs, *hb),
            _ => (self.weights.embed_tokens, self.weights.embed_scales, self.weights.embed_biases),
        };
        let logits = self.ctx.quantized_matmul(last, lm_w, lm_s, lm_b, true, group_size, bits)?;
        // Evaluate the logits graph AND the updated KV buffers, THEN free every
        // intermediate we own. Evaluating kv_bufs materializes the slice_update
        // outputs so they no longer reference the previous (stale) KV buffers,
        // making it safe to release those. `logits` retains what it needs. This
        // releases hundreds of per-request buffers that previously leaked.
        let mut to_eval: Vec<mlx_array> = Vec::with_capacity(self.kv_bufs.len() + 1);
        to_eval.push(logits);
        to_eval.extend_from_slice(&self.kv_bufs);
        self.ctx.eval_all(&to_eval)?;
        unsafe {
            for &a in &scratch { engine_mlx_ffi::mlx_array_free(a); }
            for &a in &stale_kv { engine_mlx_ffi::mlx_array_free(a); }
        }
        tracing::info!(
            target: "engine_mlx::qwen3::forward",
            "prefill END seq_len={}",
            seq_len
        );
        Ok(logits)
    }
    #[cfg(not(feature = "mlx"))]
    pub fn prefill(&mut self, _input: mlx_array, _seq_len: usize) -> Result<mlx_array> {
        anyhow::bail!("mlx feature not enabled — prefill unavailable")
    }
    #[cfg(feature = "mlx")]
    pub fn decode_step(&mut self, token_id: u32) -> Result<u32> {
        tracing::info!(
            target: "engine_mlx::qwen3::forward",
            "decode_step START token_id={} offset={}",
            token_id,
            self.offset
        );
        // Fast path: static compiled closure (mask input, no concat).
        if self.compiled_static.is_some() {
            match self.decode_step_static(token_id) {
                Ok(tok) => return Ok(tok),
                Err(e) => {
                    tracing::warn!(
                        target: "engine_mlx::qwen3::forward",
                        "static decode failed, falling back: {e:?}"
                    );
                    self.compiled_static = None; // disable static path
                }
            }
        }
        // Fast path: compiled closure (argmax in-graph, ~82 t/s).
        // If compiled step fails, fall back to eager.
        if self.compiled_full.is_some() {
            match self.decode_step_compiled(token_id) {
                Ok(tok) => return Ok(tok),
                Err(e) => {
                    tracing::warn!(
                        target: "engine_mlx::qwen3::forward",
                        "compiled decode failed, falling back to eager: {e:?}"
                    );
                    self.compiled_full = None; // disable compiled path
                }
            }
        }
        // Eager fallback.
        self.decode_step_eager(token_id)
    }

    #[cfg(not(feature = "mlx"))]
    pub fn decode_step(&mut self, _token_id: u32) -> Result<u32> {
        anyhow::bail!("mlx feature not enabled — decode_step unavailable")
    }
    pub fn verify_draft(&mut self, draft_tokens: &[u32]) -> Result<(usize, Vec<u32>)> {
        if draft_tokens.is_empty() { return Ok((0, Vec::new())); }
        anyhow::bail!("verify_draft not implemented — requires batched slice_update support")
    }
}
