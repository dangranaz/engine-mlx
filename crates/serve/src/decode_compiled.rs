//! Compiled/static decode steps — launch/commit + pipelined readback.
//!
//! Split from `forward_qwen3.rs` per RULES (max 300 lines/file). Eager math
//! stays in `forward_qwen3.rs`; this module only drives the closures.

use anyhow::Result;
use engine_mlx_ffi::mlx_array;

use crate::engine::Qwen3Engine;

impl Qwen3Engine {
    /// Static decode step: mask input + slice_update_dynamic KV, pipelined.
    #[cfg(feature = "mlx")]
    pub(crate) fn decode_step_static(&mut self, token_id: u32) -> Result<u32> {
        let x = self.embed(&[token_id])?;
        let offset_arr = self.static_offsets.get(self.offset).copied()
            .ok_or_else(|| anyhow::anyhow!("static_offsets not initialized at offset={}", self.offset))?;
        let mask = self.static_mask(self.offset)?;
        if self.offset >= self.kv_static_cap {
            anyhow::bail!("static KV overflow offset={} cap={}", self.offset, self.kv_static_cap);
        }
        let nl = self.weights.layers.len();
        let mut inputs: Vec<mlx_array> = Vec::with_capacity(3 + nl * 2);
        inputs.push(x);
        inputs.push(offset_arr);
        inputs.push(mask);
        for buf in &self.kv_bufs {
            inputs.push(*buf);
        }
        let outputs = self
            .compiled_static
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("static closure not initialized"))?
            .apply(&inputs)?;
        self.ctx.async_eval(&outputs)?;
        let vals = self.ctx.to_vec_u32(outputs[0])?;
        self.offset += 1;
        // The compiled closure returns FRESH KV arrays each step; the previous
        // buffer handles are now unreferenced. mlx_array handles are refcounted C
        // objects and dropping the Rust copy does not release them, so overwriting
        // without freeing leaks two full-capacity KV buffers per layer per step
        // (≈3.6 GB per 128-token request → the server OOMs/pages and decode
        // throughput collapses from ~75 t/s to ~30 t/s). Free the old buffers,
        // the token output, and the per-step embedding input.
        for li in 0..nl {
            let old_k = self.kv_bufs[li * 2];
            let old_v = self.kv_bufs[li * 2 + 1];
            self.kv_bufs[li * 2] = outputs[1 + li * 2];
            self.kv_bufs[li * 2 + 1] = outputs[1 + li * 2 + 1];
            unsafe {
                engine_mlx_ffi::mlx_array_free(old_k);
                engine_mlx_ffi::mlx_array_free(old_v);
            }
        }
        unsafe {
            engine_mlx_ffi::mlx_array_free(outputs[0]);
            engine_mlx_ffi::mlx_array_free(x);
        }
        Ok(vals[0])
    }

    /// Launch one compiled decode step without evaluating — returns lazy outputs
    /// plus the per-step transient inputs (`x`, `offset_arr`) so the caller can
    /// free them AFTER the graph is evaluated.
    #[cfg(feature = "mlx")]
    pub(crate) fn step_launch_compiled(&mut self, token_id: u32) -> Result<(Vec<mlx_array>, mlx_array, mlx_array)> {
        let x = self.embed(&[token_id])?;
        let offset_arr = self.ctx.new_array_i32(&[self.offset as i32])?;
        let nl = self.weights.layers.len();
        let per_layer = if self.kv_bufs.len() == nl * 6 { 6 } else { 2 };
        let mut inputs: Vec<mlx_array> = Vec::with_capacity(2 + nl * per_layer);
        inputs.push(x);
        inputs.push(offset_arr);
        for buf in &self.kv_bufs {
            inputs.push(*buf);
        }

        let outputs = self.compiled_full
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("compiled closure not initialized"))?
            .apply(&inputs)?;
        Ok((outputs, x, offset_arr))
    }

    /// Commit compiled outputs: bump offset + swap KV buffers, freeing the KV
    /// buffers being replaced. mlx_array is a refcounted C handle; overwriting
    /// the Vec entry without freeing leaks the old cache buffer every step,
    /// accumulating live MLX resources until `[metal::malloc] Resource limit
    /// (499000)` is hit and decode collapses. The new (concat) outputs have been
    /// evaluated by the caller before commit, so releasing the old buffers is
    /// safe.
    #[cfg(feature = "mlx")]
    pub(crate) fn step_commit_compiled(&mut self, outputs: &[mlx_array]) {
        let nl = self.weights.layers.len();
        let per_layer = if self.kv_bufs.len() == nl * 6 { 6 } else { 2 };
        self.offset += 1;
        for li in 0..nl {
            for j in 0..per_layer {
                let idx = li * per_layer + j;
                let old = self.kv_bufs[idx];
                self.kv_bufs[idx] = outputs[1 + idx];
                unsafe { engine_mlx_ffi::mlx_array_free(old); }
            }
        }
    }

    #[cfg(feature = "mlx")]
    pub(crate) fn decode_step_compiled(&mut self, token_id: u32) -> Result<u32> {
        // Pipelined: launch (lazy) -> async_eval (non-blocking GPU kick) ->
        // readback (single sync point) -> commit KV.
        let (outputs, x, offset_arr) = self.step_launch_compiled(token_id)?;
        self.ctx.async_eval(&outputs)?;
        let vals = self.ctx.to_vec_u32(outputs[0])?;
        self.step_commit_compiled(&outputs);
        // Free the transient inputs and the token output. KV outputs are kept by
        // step_commit_compiled; the old KV buffers were freed there.
        unsafe {
            engine_mlx_ffi::mlx_array_free(outputs[0]);
            engine_mlx_ffi::mlx_array_free(x);
            engine_mlx_ffi::mlx_array_free(offset_arr);
        }
        Ok(vals[0])
    }
}
