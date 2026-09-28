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
            // Free the MLX handles before clearing the Vecs. `mlx_array` handles
            // are refcounted C objects: Vec::clear() drops the Rust copies but
            // does NOT release the underlying Metal buffers, so they linger as
            // live resources (num_resources_). Accumulating them across requests
            // exhausts the Metal resource limit (`[metal::malloc] Resource limit
            // (499000)` — a live-buffer COUNT, not bytes), after which arrays
            // come back empty and decode degenerates to garbage. Explicit frees
            // + clear_cache keep the resource count flat between requests.
            unsafe {
                for &a in &self.kv_bufs { engine_mlx_ffi::mlx_array_free(a); }
                for &a in &self.static_masks { engine_mlx_ffi::mlx_array_free(a); }
                for &a in &self.static_offsets { engine_mlx_ffi::mlx_array_free(a); }
            }
            self.kv_bufs.clear();
            self.kv_static_cap = 0;
            self.static_masks.clear();
            self.static_offsets.clear();
            self.compiled_static = None;
            self.compiled_full = None;
            unsafe {
                engine_mlx_ffi::mlx_clear_cache();
            }
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

    /// Re-zero the static KV data buffers for a new sequence while keeping the
    /// (capacity-invariant) masks, offset arrays and compiled closure intact.
    /// Used by the server's per-request reuse path to avoid recompiling.
    ///
    /// Currently unused: the cross-request reuse path was removed because it
    /// starved Metal's resource pool under sustained load (see
    /// `generate_via_mlx`). Kept for a future, leak-safe reuse implementation.
    #[cfg(feature = "mlx")]
    #[allow(dead_code)]
    pub fn zero_kv_static(&mut self) -> Result<()> {
        let nl = self.weights.layers.len();
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim();
        let cap = self.kv_static_cap as i32;
        self.kv_bufs.clear();
        self.kv_bufs.reserve(nl * 2);
        for _ in 0..nl {
            let k = self.ctx.zeros_bf16(&[1, nkv, cap, hd])?;
            let v = self.ctx.zeros_bf16(&[1, nkv, cap, hd])?;
            self.kv_bufs.push(k);
            self.kv_bufs.push(v);
        }
        Ok(())
    }

    /// Pre-allocate static BF16 KV buffers (2 per layer) with fixed capacity.
    /// Also pre-computes attention masks for every offset (0..cap) so the
    /// hot loop avoids CPU mask construction + H2D transfer per step.
    /// Token-exact vs concat; decode writes in place via `slice_update_dynamic`.
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
        // Pre-compute all positional masks: [1,1,1,cap] with 0.0 at p<=offset, -1e9 beyond.
        // Done ONCE here so the decode hot loop only does an O(1) array lookup.
        self.static_masks.clear();
        self.static_masks.reserve(capacity);
        for offset in 0..capacity {
            let mut v = Vec::with_capacity(capacity);
            for p in 0..capacity {
                v.push(if p <= offset { 0.0 } else { -1e9 });
            }
            let m = self.ctx.new_array(&v)?;
            let m = self.ctx.reshape(m, &[1, 1, 1, capacity as i32])?;
            // MLX fast SDPA requires the mask dtype to promote to the output
            // dtype (bf16). An f32 mask is wider than bf16 and is rejected
            // ("Mask type must promote to output type bfloat16"), so cast it.
            let m = self.ctx.astype(m, engine_mlx_ffi::mlx_dtype_::MLX_BFLOAT16)?;
            self.static_masks.push(m);
        }
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
            "alloc_kv_static done n_layers={} nkv={} hd={} capacity={} masks={}", nl, nkv, hd, capacity, self.static_masks.len()
        );
        Ok(())
    }

    /// Retrieve pre-computed positional mask for static KV decode.
    /// O(1) lookup — no CPU construction or H2D transfer per step.
    #[cfg(feature = "mlx")]
    pub fn static_mask(&self, offset: usize) -> Result<mlx_array> {
        tracing::debug!(target: "engine_mlx::qwen3::engine", "static_mask offset={}", offset);
        self.static_masks.get(offset).copied()
            .ok_or_else(|| anyhow::anyhow!("static_mask offset {} out of range (cap={}, masks={})", offset, self.kv_static_cap, self.static_masks.len()))
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
