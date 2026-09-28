//! Eager decode step — fallback when compiled closure fails.
//!
//! Split from `forward_qwen3.rs` per RULES (max 300 lines/file).

use anyhow::Result;
use engine_mlx_ffi::mlx_array;

use crate::engine::Qwen3Engine;
use crate::sampler::sample_from_logits;

impl Qwen3Engine {
    #[cfg(feature = "mlx")]
    pub fn decode_step_eager(&mut self, token_id: u32) -> Result<u32> {
        let x = self.embed(&[token_id])?;
        let is_int4 = self.kv_bufs.len() == self.weights.layers.len() * 6;
        let is_static = !is_int4 && self.kv_static_cap > 0;
        // Use pre-allocated offset array in static path (avoids per-step H2D).
        // Falls back to new_array_i32 for concat/int4 paths where offset is unbounded.
        let offset_arr = if is_static {
            self.static_offsets.get(self.offset)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("static_offsets not initialized at offset={}", self.offset))?
        } else {
            self.ctx.new_array_i32(&[self.offset as i32])?
        };
        // Track whether offset_arr is borrowed from the pre-allocated static
        // pool (must NOT free) or freshly allocated here (ours to free).
        let is_static_offset = is_static;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        let (group_size, bits) = self
            .config
            .quantization
            .as_ref()
            .map(|q| (q.group_size as i32, q.bits as i32))
            .unwrap_or((64, 4));
        let mut x_cur = x;
        let nl = self.weights.layers.len();
        // Arena of owned intermediate handles created during this eager step.
        // Every ctx op returns a NEW refcounted mlx_array; without freeing them
        // a single decode step leaks ~20 arrays/layer (×28 layers) that stay
        // live, so num_resources_ climbs by thousands per request until
        // `[metal::malloc] Resource limit (499000)` fires and decode collapses.
        // Freed at the end after logits+kv_bufs are evaluated (see tail). Weights
        // and stored kv_bufs entries are NEVER pushed here; the pre-allocated
        // static offset/mask are also not ours to free.
        let mut scratch: Vec<mlx_array> = Vec::with_capacity(nl * 24 + 8);
        scratch.push(x);
        if !is_static_offset { scratch.push(offset_arr); }
        // KV buffers replaced this step (freed after tail eval, once the new
        // kv_bufs — which reference them lazily — are materialized).
        let mut stale_kv: Vec<mlx_array> = Vec::with_capacity(nl * 2);
        macro_rules! keep { ($e:expr) => {{ let a = $e; scratch.push(a); a }} }
        // int4 path is debug-only (see tests/int4_debug.rs); production always
        // uses static KV (bf16, slice_update + masked SDPA) or concat KV.
        let is_static = self.kv_static_cap > 0;
        // Pre-compute mask once (same for all layers in this step).
        let static_mask = if is_static { Some(self.static_mask(self.offset)?) } else { None };
        for li in 0..nl {
            let lw = &self.weights.layers[li];
            let normed = keep!(self.ctx.rms_norm(x_cur, lw.input_layernorm, self.config.rms_norm_eps)?);
            let qkv = keep!(self.ctx.quantized_matmul(normed, lw.qkv_proj_w, lw.qkv_proj_s, lw.qkv_proj_b, true, group_size, bits)?);
            let q = keep!(self.ctx.slice(qkv, &[0, 0, 0], &[1, 1, nh * hd], &[1, 1, 1])?);
            let k = keep!(self.ctx.slice(qkv, &[0, 0, nh * hd], &[1, 1, nh * hd + nkv * hd], &[1, 1, 1])?);
            let v = keep!(self.ctx.slice(qkv, &[0, 0, nh * hd + nkv * hd], &[1, 1, nh * hd + 2 * nkv * hd], &[1, 1, 1])?);
            let q = keep!(self.ctx.reshape(q, &[1, 1, nh, hd])?);
            let mut k = keep!(self.ctx.reshape(k, &[1, 1, nkv, hd])?);
            let mut v = keep!(self.ctx.reshape(v, &[1, 1, nkv, hd])?);
            let q = if let Some(qn) = lw.q_norm { keep!(self.ctx.rms_norm(q, qn, self.config.rms_norm_eps)?) } else { q };
            k = if let Some(kn) = lw.k_norm { keep!(self.ctx.rms_norm(k, kn, self.config.rms_norm_eps)?) } else { k };
            let q = keep!(self.ctx.transpose_axes(q, &[0, 2, 1, 3])?);
            k = keep!(self.ctx.transpose_axes(k, &[0, 2, 1, 3])?);
            v = keep!(self.ctx.transpose_axes(v, &[0, 2, 1, 3])?);
            let q = keep!(self.ctx.rope_dynamic(q, hd, self.config.rope_theta, offset_arr)?);
            k = keep!(self.ctx.rope_dynamic(k, hd, self.config.rope_theta, offset_arr)?);
            // int4 path: only when kv_bufs has 6 buffers/layer (debug-only, see int4_debug.rs)
            let (k_full, v_full) = if self.kv_bufs.len() == nl * 6 {
                let base = li * 6;
                let (k_data, k_scales, k_biases, v_data, v_scales, v_biases, k_deq, v_deq) = self.int4_update_and_dequant(
                    self.kv_bufs[base],
                    self.kv_bufs[base + 1],
                    self.kv_bufs[base + 2],
                    self.kv_bufs[base + 3],
                    self.kv_bufs[base + 4],
                    self.kv_bufs[base + 5],
                    k,
                    v,
                    offset_arr,
                )?;
                // Old int4 cache buffers are replaced; defer their free.
                for j in 0..6 { stale_kv.push(self.kv_bufs[base + j]); }
                self.kv_bufs[base] = k_data;
                self.kv_bufs[base + 1] = k_scales;
                self.kv_bufs[base + 2] = k_biases;
                self.kv_bufs[base + 3] = v_data;
                self.kv_bufs[base + 4] = v_scales;
                self.kv_bufs[base + 5] = v_biases;
                // Slice dequantized KV to actual seq_len (=offset+1) to avoid attending to future zeros
                let seq_len = self.offset as i32 + 1;
                let k_deq2 = keep!(self.ctx.slice(k_deq, &[0, 0, 0, 0], &[1, nkv, seq_len, hd], &[1, 1, 1, 1])?);
                let v_deq2 = keep!(self.ctx.slice(v_deq, &[0, 0, 0, 0], &[1, nkv, seq_len, hd], &[1, 1, 1, 1])?);
                (k_deq2, v_deq2)
            } else if is_static {
                // Static KV: scatter_single at offset (avoids full-buffer copy overhead).
                let off_arr = self.static_offsets.get(self.offset)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("static_offsets not initialized at offset={}", self.offset))?;
                let old_k = self.kv_bufs[li * 2];
                let old_v = self.kv_bufs[li * 2 + 1];
                // scatter_single updates.ndim == indices.ndim(1) + a.ndim(4);
                // reshape [1,nkv,1,hd] -> [1,1,nkv,1,hd] (leading index dim).
                let k_s = keep!(self.ctx.reshape(k, &[1, 1, nkv, 1, hd])?);
                let v_s = keep!(self.ctx.reshape(v, &[1, 1, nkv, 1, hd])?);
                let k_upd = self.ctx.scatter_single(old_k, off_arr, k_s, 2)?;
                let v_upd = self.ctx.scatter_single(old_v, off_arr, v_s, 2)?;
                self.kv_bufs[li * 2] = k_upd;
                self.kv_bufs[li * 2 + 1] = v_upd;
                // scatter_single returns a NEW array that references old_k/old_v
                // until evaluated; defer their free until the tail eval.
                stale_kv.push(old_k);
                stale_kv.push(old_v);
                (k_upd, v_upd)
            } else {
                let k_cache = self.kv_bufs[li * 2];
                let v_cache = self.kv_bufs[li * 2 + 1];
                let k_full = self.ctx.concatenate(k_cache, k, 2)?;
                let v_full = self.ctx.concatenate(v_cache, v, 2)?;
                self.kv_bufs[li * 2] = k_full;
                self.kv_bufs[li * 2 + 1] = v_full;
                // Old concat cache replaced; defer free (new cache references it).
                stale_kv.push(k_cache);
                stale_kv.push(v_cache);
                (k_full, v_full)
            };
            let attn = match &static_mask {
                Some(mask) => keep!(self.ctx.sdpa_masked(q, k_full, v_full, scale, *mask)?),
                None => keep!(self.ctx.sdpa(q, k_full, v_full, scale, true)?),
            };
            let attn = keep!(self.ctx.transpose_axes(attn, &[0, 2, 1, 3])?);
            let attn = keep!(self.ctx.reshape(attn, &[1, 1, nh * hd])?);
            let o = keep!(self.ctx.quantized_matmul(attn, lw.o_proj_w, lw.o_proj_s, lw.o_proj_b, true, group_size, bits)?);
            x_cur = keep!(self.ctx.add(x_cur, o)?);
            let normed2 = keep!(self.ctx.rms_norm(x_cur, lw.post_attention_layernorm, self.config.rms_norm_eps)?);
            let gup = keep!(self.ctx.quantized_matmul(normed2, lw.gate_up_proj_w, lw.gate_up_proj_s, lw.gate_up_proj_b, true, group_size, bits)?);
            let ff = self.config.intermediate_size;
            let gate = keep!(self.ctx.slice(gup, &[0, 0, 0], &[1, 1, ff], &[1, 1, 1])?);
            let up = keep!(self.ctx.slice(gup, &[0, 0, ff], &[1, 1, 2 * ff], &[1, 1, 1])?);
            let silu = keep!(self.ctx.silu(gate)?);
            let hidden = keep!(self.ctx.multiply(silu, up)?);
            let down = keep!(self.ctx.quantized_matmul(hidden, lw.down_proj_w, lw.down_proj_s, lw.down_proj_b, true, group_size, bits)?);
            x_cur = keep!(self.ctx.add(x_cur, down)?);
        }
        let x_norm = keep!(self.ctx.rms_norm(x_cur, self.weights.final_norm, self.config.rms_norm_eps)?);
        let x2d = keep!(self.ctx.reshape(x_norm, &[1, self.config.hidden_size])?);
        let (lm_w, lm_s, lm_b) = match (&self.weights.lm_head_w, &self.weights.lm_head_s, &self.weights.lm_head_b) {
            (Some(hw), Some(hs), Some(hb)) => (*hw, *hs, *hb),
            _ => (self.weights.embed_tokens, self.weights.embed_scales, self.weights.embed_biases),
        };
        let logits = self.ctx.quantized_matmul(x2d, lm_w, lm_s, lm_b, true, group_size, bits)?;
        // Evaluate logits + updated kv_bufs so the graph materializes and the
        // stale KV / intermediates are no longer referenced, THEN free them.
        let mut to_eval: Vec<mlx_array> = Vec::with_capacity(self.kv_bufs.len() + 1);
        to_eval.push(logits);
        to_eval.extend_from_slice(&self.kv_bufs);
        self.ctx.eval_all(&to_eval)?;
        // Sampling: temperature 0 => greedy, otherwise top-p/top-k
        let tok = if self.sampling.temperature < 1e-6 {
            let tok_arr = self.ctx.argmax(logits)?;
            let vals = self.ctx.to_vec_u32(tok_arr)?;
            unsafe { engine_mlx_ffi::mlx_array_free(tok_arr); }
            vals[0]
        } else {
            sample_from_logits(&self.ctx, logits, &self.sampling, &mut self.rng)?
        };
        unsafe {
            for &a in &scratch { engine_mlx_ffi::mlx_array_free(a); }
            for &a in &stale_kv { engine_mlx_ffi::mlx_array_free(a); }
            engine_mlx_ffi::mlx_array_free(logits);
        }
        self.offset += 1;
        Ok(tok)
    }
    #[cfg(not(feature = "mlx"))]
    pub fn decode_step_eager(&mut self, _token_id: u32) -> Result<u32> {
        anyhow::bail!("mlx feature not enabled — decode_step_eager unavailable")
    }
}
