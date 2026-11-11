//! Qwen3Engine helpers — embed, KV, reset (extracted to keep engine.rs <300).

use anyhow::{Context, Result};
use engine_mlx_ffi::mlx_array;

use crate::engine::Qwen3Engine;

impl Qwen3Engine {
    /// Reset sequence state.
    pub fn reset(&mut self) {
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::reset offset={}",
            self.offset
        );
        self.offset = 0;
        #[cfg(feature = "mlx")]
        {
            // E5 fix: the static KV buffers, single static mask, offset arrays,
            // and compiled closure are persistent across requests (allocated once
            // at MAX_CAP during init). reset() only resets the logical position;
            // it does NOT free or replace the buffers, mask, offset arrays, or
            // compiled closure. Stale KV positions are excluded by the additive
            // mask and overwritten as prefill/decode advance.
            // NOTE: do NOT call mlx_clear_cache() here — it would clear the
            // compiled closure's cached code, forcing recompilation and
            // defeating the leak fix. The closure is compiled once at init and
            // reused.
            self.zero_kv_static().ok(); // best effort; ignore error in reset
            self.offset = 0;
        }
        #[cfg(not(feature = "mlx"))]
        {
            self.kv_bufs.clear();
        }
    }

    /// Borrow tokenizer.
    pub fn get_tokenizer(&self) -> &crate::tokenizer::Tokenizer {
        &self.tokenizer
    }

    /// Embed a slice of token ids using the (optional) dequantized table.
    /// Returns `[1, seq_len, hidden]`.
    pub fn embed(&self, ids: &[u32]) -> Result<mlx_array> {
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::embed ids_len={}",
            ids.len()
        );
        let table = self
            .embed_table
            .ok_or_else(|| anyhow::anyhow!("embed_table not available (mlx disabled or dequant failed)"))?;
        let ids_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let idx_arr = self
            .ctx
            .new_array_i32(&ids_i32)
            .context("new_array_i32 failed")?;
        let out = self.ctx.embed(table, idx_arr).context("embed take failed")?;
        // The id-index array is an owned intermediate: the returned activation
        // retains it as a graph input until evaluated, so releasing our handle
        // here (after the take is built) frees it exactly once.
        unsafe { engine_mlx_ffi::mlx_array_free(idx_arr); }
        // Add batch dim: [seq, hidden] -> [1, seq, hidden]
        let out = self.ctx.expand_dims(out, 0).context("expand_dims failed")?;
        Ok(out)
    }

    /// Pre-allocate zero-filled KV buffers for all layers.
    /// Each layer gets 2 arrays: [k_cache, v_cache], shape `[1, 0, nkv, hd]`.
    /// BF16 to match K/V dtype (avoids promotion on concat).
    #[cfg(feature = "mlx")]
    pub fn alloc_kv(&mut self) -> Result<()> {
        let nl = self.weights.layers.len();
        let nkv = self.config.num_key_value_heads as usize;
        let hd = self.config.head_dim() as usize;
        self.kv_bufs.clear();
        self.kv_bufs.reserve(nl * 2);
        for _ in 0..nl {
            let k = self.ctx.zeros_bf16(&[1, 0, nkv as i32, hd as i32])?;
            let v = self.ctx.zeros_bf16(&[1, 0, nkv as i32, hd as i32])?;
            self.kv_bufs.push(k);
            self.kv_bufs.push(v);
        }
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "alloc_kv done n_layers={} nkv={} head_dim={}", nl, nkv, hd
        );
        Ok(())
    }

    /// Pre-allocate int4 KV buffers (6 per layer) with fixed capacity.
    /// Capacity = max(seq_len + max_tokens + 16, 256) as in serve-mlx-c.
    /// Layout per layer: [k_data, k_scales, k_biases, v_data, v_scales, v_biases]
    ///   data:   [1, nkv, capacity, hd/8]  UINT32 (8×4bit packed)
    ///   scales: [1, nkv, capacity, hd/64] BF16
    ///   biases: [1, nkv, capacity, hd/64] BF16
    #[cfg(feature = "mlx")]
    pub fn alloc_kv_int4(&mut self, capacity: usize) -> Result<()> {
        let nl = self.weights.layers.len();
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        let capacity = capacity as i32;
        let packed_dim = hd / 8;
        let groups_per_head = hd / 64;
        let shape_data = [1, nkv, capacity, packed_dim];
        let shape_meta = [1, nkv, capacity, groups_per_head];
        self.kv_bufs.clear();
        self.kv_bufs.reserve(nl * 6);
        for _ in 0..nl {
            let k_data = self.ctx.zeros_u32(&shape_data)?;
            let k_scales = self.ctx.zeros_bf16(&shape_meta)?;
            let k_biases = self.ctx.zeros_bf16(&shape_meta)?;
            let v_data = self.ctx.zeros_u32(&shape_data)?;
            let v_scales = self.ctx.zeros_bf16(&shape_meta)?;
            let v_biases = self.ctx.zeros_bf16(&shape_meta)?;
            self.kv_bufs.push(k_data);
            self.kv_bufs.push(k_scales);
            self.kv_bufs.push(k_biases);
            self.kv_bufs.push(v_data);
            self.kv_bufs.push(v_scales);
            self.kv_bufs.push(v_biases);
        }
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "alloc_kv_int4 done n_layers={} nkv={} hd={} capacity={}", nl, nkv, hd, capacity
        );
        Ok(())
    }

    /// Int4 KV update + dequant helper (mirrors nxm-mlx-ops kv_cache.rs).
    /// Quantizes k_new/v_new (shape [1,nkv,1,hd]), slice-updates into cache at offset, dequantizes whole cache.
    #[cfg(feature = "mlx")]
    pub fn int4_update_and_dequant(
        &self,
        cache_k_data: mlx_array,
        cache_k_scales: mlx_array,
        cache_k_biases: mlx_array,
        cache_v_data: mlx_array,
        cache_v_scales: mlx_array,
        cache_v_biases: mlx_array,
        k_new: mlx_array,
        v_new: mlx_array,
        offset_arr: mlx_array,
    ) -> Result<(mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array)> {
        let (k_qdata, k_qscales, k_qbiases) = self.ctx.quantize_kv_native(k_new, 64, 4)?;
        let (v_qdata, v_qscales, v_qbiases) = self.ctx.quantize_kv_native(v_new, 64, 4)?;
        let k_data = self.ctx.slice_update_dynamic(cache_k_data, k_qdata, offset_arr, &[2])?;
        let k_scales = self.ctx.slice_update_dynamic(cache_k_scales, k_qscales, offset_arr, &[2])?;
        let k_biases = self.ctx.slice_update_dynamic(cache_k_biases, k_qbiases, offset_arr, &[2])?;
        let v_data = self.ctx.slice_update_dynamic(cache_v_data, v_qdata, offset_arr, &[2])?;
        let v_scales = self.ctx.slice_update_dynamic(cache_v_scales, v_qscales, offset_arr, &[2])?;
        let v_biases = self.ctx.slice_update_dynamic(cache_v_biases, v_qbiases, offset_arr, &[2])?;
        let k_dequant = self.ctx.dequantize_kv_native(k_data, k_scales, k_biases, 64, 4)?;
        let v_dequant = self.ctx.dequantize_kv_native(v_data, v_scales, v_biases, 64, 4)?;
        Ok((k_data, k_scales, k_biases, v_data, v_scales, v_biases, k_dequant, v_dequant))
    }

    /// Convert existing BF16 KV (2 per layer) to int4 (6 per layer) for compiled path.
    /// `t` is current seq_len (prompt length), `capacity` is max(seq+max_tokens+16,256).
    #[cfg(feature = "mlx")]
    pub fn convert_kv_to_int4(&mut self, t: usize, capacity: usize) -> Result<()> {
        let nl = self.weights.layers.len();
        if self.kv_bufs.len() != nl * 2 {
            anyhow::bail!("convert_kv_to_int4: expected {} BF16 bufs, got {}", nl * 2, self.kv_bufs.len());
        }
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        // Save old BF16 caches
        let old_bufs = self.kv_bufs.clone();
        // Allocate new int4 buffers
        self.alloc_kv_int4(capacity)?;
        // Copy each position p in 0..t
        for li in 0..nl {
            let k_bf16 = old_bufs[li * 2];
            let v_bf16 = old_bufs[li * 2 + 1];
            let base = li * 6;
            let mut k_data = self.kv_bufs[base];
            let mut k_scales = self.kv_bufs[base + 1];
            let mut k_biases = self.kv_bufs[base + 2];
            let mut v_data = self.kv_bufs[base + 3];
            let mut v_scales = self.kv_bufs[base + 4];
            let mut v_biases = self.kv_bufs[base + 5];
            for pos in 0..t as i32 {
                let p_arr = self.ctx.new_array_i32(&[pos])?;
                // k_bf16 shape [1,nkv,t,hd] -> slice [1,nkv,1,hd] at pos
                let k_p = self.ctx.slice(k_bf16, &[0, 0, pos, 0], &[1, nkv, pos + 1, hd], &[1, 1, 1, 1])?;
                let v_p = self.ctx.slice(v_bf16, &[0, 0, pos, 0], &[1, nkv, pos + 1, hd], &[1, 1, 1, 1])?;
                // quantize expects [1,nkv,1,hd] – our slice is already that, but ensure reshape
                let k_p = self.ctx.reshape(k_p, &[1, nkv, 1, hd])?;
                let v_p = self.ctx.reshape(v_p, &[1, nkv, 1, hd])?;
                let (k_qd, k_qs, k_qb) = self.ctx.quantize_kv_native(k_p, 64, 4)?;
                let (v_qd, v_qs, v_qb) = self.ctx.quantize_kv_native(v_p, 64, 4)?;
                k_data = self.ctx.slice_update_dynamic(k_data, k_qd, p_arr, &[2])?;
                k_scales = self.ctx.slice_update_dynamic(k_scales, k_qs, p_arr, &[2])?;
                k_biases = self.ctx.slice_update_dynamic(k_biases, k_qb, p_arr, &[2])?;
                v_data = self.ctx.slice_update_dynamic(v_data, v_qd, p_arr, &[2])?;
                v_scales = self.ctx.slice_update_dynamic(v_scales, v_qs, p_arr, &[2])?;
                v_biases = self.ctx.slice_update_dynamic(v_biases, v_qb, p_arr, &[2])?;
            }
            self.kv_bufs[base] = k_data;
            self.kv_bufs[base + 1] = k_scales;
            self.kv_bufs[base + 2] = k_biases;
            self.kv_bufs[base + 3] = v_data;
            self.kv_bufs[base + 4] = v_scales;
            self.kv_bufs[base + 5] = v_biases;
        }
        tracing::info!(target: "engine_mlx::qwen3::engine", "convert_kv_to_int4 done t={} capacity={}", t, capacity);
        Ok(())
    }

    /// Whether the compiled decode closure is active.
    pub fn is_compiled(&self) -> bool {
        tracing::debug!(target: "engine_mlx::qwen3::engine", "is_compiled={}", self.compiled_full.is_some());
        self.compiled_full.is_some()
    }

    /// Prepare the static KV state for a new sequence without clearing storage.
    ///
    /// Despite the historical name, this does not zero the KV buffers: prefill
    /// and decode overwrite every valid position, while the additive mask
    /// excludes stale positions. Reusing the buffers avoids per-request,
    /// per-layer allocations that leaked live Metal resources.
    #[cfg(feature = "mlx")]
    pub fn zero_kv_static(&mut self) -> Result<()> {
        self.offset = 0;
        Ok(())
    }

    /// Update the single static mask for the current offset.
    /// Sets positions 0..=offset to 0.0 and positions >offset to -1e9.
    /// This is called after prefill (to seed 0..=prompt_len-1 as valid) and
    /// after each decode step (to advance the valid window by 1).
    /// Uses separate bindings (no shadowing) so the new_array and reshape
    /// intermediates are freed after materialization — only the final astype
    /// array is retained.
    #[cfg(feature = "mlx")]
    pub fn update_static_mask_cpu(&mut self, offset: usize) -> Result<()> {
        let cap = self.kv_static_cap as i32;
        // Create new mask values: 0.0 for 0..=offset, -1e9 for >offset
        let mut v = Vec::with_capacity(cap as usize);
        for p in 0..cap {
            v.push(if (p as usize) <= offset { 0.0 } else { -1e9 });
        }
        let raw = self.ctx.new_array(&v)?;
        let reshaped = self.ctx.reshape(raw, &[1, 1, 1, cap])?;
        let final_arr = self.ctx.astype(reshaped, engine_mlx_ffi::mlx_dtype_::MLX_BFLOAT16)?;
        // Materialize final (and transitive inputs reshaped/raw) before freeing intermediates.
        self.ctx.eval_all(&[final_arr])?;
        unsafe { engine_mlx_ffi::mlx_synchronize(self.stream); }
        // Intermediates are no longer needed — final is fully materialized.
        unsafe { engine_mlx_ffi::mlx_array_free(raw); }
        unsafe { engine_mlx_ffi::mlx_array_free(reshaped); }
        // Free old mask and replace — no handle accumulation.
        if let Some(old) = self.static_mask {
            unsafe { engine_mlx_ffi::mlx_array_free(old); }
        }
        self.static_mask = Some(final_arr);
        Ok(())
    }

    /// Pre-allocate static BF16 KV buffers (2 per layer) with fixed capacity.
    /// Also creates a SINGLE additive attention mask [1,1,1,cap] that will be
    /// updated in-place per step (token-exact vs pre-materialized O(cap) masks
    /// or per-step reconstruction).
    #[cfg(feature = "mlx")]
    pub fn alloc_kv_static(&mut self, capacity: usize) -> Result<()> {
        let nl = self.weights.layers.len();
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        self.kv_bufs.clear();
        self.kv_bufs.reserve(nl * 2);
        for _ in 0..nl {
            let k = self.ctx.zeros_bf16(&[1, nkv, capacity as i32, hd])?;
            let v = self.ctx.zeros_bf16(&[1, nkv, capacity as i32, hd])?;
            self.kv_bufs.push(k);
            self.kv_bufs.push(v);
        }
        // Create a SINGLE additive mask [1,1,1,cap] initialized to all -1e9.
        // Will be updated in-place via update_static_mask() after prefill and
        // each decode step. This avoids O(cap) pre-materialized masks and
        // per-step handle allocation (both leak sources).
        // Separate bindings (no shadowing) so intermediates can be freed.
        let cap = capacity as i32;
        let mut v = vec![-1e9; cap as usize];
        let raw_mask = self.ctx.new_array(&v)?;
        let reshaped_mask = self.ctx.reshape(raw_mask, &[1, 1, 1, cap])?;
        let final_mask = self.ctx.astype(reshaped_mask, engine_mlx_ffi::mlx_dtype_::MLX_BFLOAT16)?;
        self.ctx.eval_all(&[final_mask])?;
        unsafe { engine_mlx_ffi::mlx_array_free(raw_mask); }
        unsafe { engine_mlx_ffi::mlx_array_free(reshaped_mask); }
        self.static_mask = Some(final_mask);
        // Pre-allocate offset arrays [0], [1], ..., [cap-1] so the decode hot loop
        // avoids a new_array_i32 FFI call + 4-byte H2D transfer per step.
        self.static_offsets.clear();
        self.static_offsets.reserve(capacity);
        for off in 0..capacity {
            self.static_offsets.push(self.ctx.new_array_i32(&[off as i32])?);
        }
        self.kv_static_cap = capacity;
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "alloc_kv_static done n_layers={} nkv={} hd={} capacity={}", nl, nkv, hd, capacity
        );
        Ok(())
    }

    /// Lazily compile the decode step closure.
    #[cfg(feature = "mlx")]
    pub fn ensure_compiled(&mut self) -> Result<()> {
        let needs_recompile = self.compiled_full.is_none();
        if needs_recompile {
            tracing::info!(target: "engine_mlx::qwen3::engine", "compiling decode step...");
            self.compiled_full = Some(crate::compiled::CompiledStep::new(
                self.weights.clone(),
                self.config.clone(),
                self.stream,
            )?);
            tracing::info!(target: "engine_mlx::qwen3::engine", "decode step compiled");
        }
        Ok(())
    }

    /// Lazily compile the static-KV decode step closure (mask input, no concat).
    #[cfg(feature = "mlx")]
    pub fn ensure_compiled_static(&mut self) -> Result<()> {
        if self.compiled_static.is_none() {
            tracing::info!(target: "engine_mlx::qwen3::engine", "compiling static decode step...");
            self.compiled_static = Some(crate::compiled::CompiledStep::new_static(
                self.weights.clone(),
                self.config.clone(),
                self.stream,
                self.kv_static_cap,
            )?);
            tracing::info!(target: "engine_mlx::qwen3::engine", "static decode step compiled");
        }
        Ok(())
    }

    /// Whether the static compiled closure is active.
    pub fn is_compiled_static(&self) -> bool {
        tracing::debug!(target: "engine_mlx::qwen3::engine", "is_compiled_static={}", self.compiled_static.is_some());
        self.compiled_static.is_some()
    }
}