//! Compiled decode step — argmax in-graph for 82 t/s.
//!
//! Ports the `CompiledStep` / `StepGlobals` pattern from the historic
//! `serve-mlx-c` engine. A single `Closure` is traced once, compiled
//! with `compile_shapeless`, and replayed for every decode step.

#![allow(unsafe_code)]

use anyhow::Result;
use engine_mlx_ffi::{mlx_array, MlxCtx};

use crate::config::Qwen3Config;
use crate::loader::Qwen3Weights;

// ── Payload globals (owned by closure) ──────────────────────────────

/// Split-start constants for fused projections inside the traced graph.
///
/// Static `mlx_slice`/`split` fail at compiled-graph eval with mlx-c 0.6.0,
/// but `slice_dynamic` (start passed as an array) works (see
/// `tests/compile_probe.rs`: `probe_slice_dynamic` OK). These fresh constant
/// arrays let `build_step` keep fused `qkv_proj`/`gate_up_proj` matmuls
/// (4 dispatches/layer like eager/mlx_lm) with dynamic splits.
pub(crate) struct SplitStarts {
    pub zero: mlx_array,
    pub k: mlx_array,
    pub v: mlx_array,
    pub up: mlx_array,
}

pub(crate) struct StepGlobals {
    pub weights: Qwen3Weights,
    pub starts: SplitStarts,
    pub config: Qwen3Config,
    pub stream: engine_mlx_ffi::mlx_stream,
    /// Static KV capacity (>0 = mask input at index 2, slice_update_dynamic writes).
    /// 0 = concat (growing) KV mode.
    pub kv_static_cap: usize,
}

unsafe impl Send for StepGlobals {}

unsafe extern "C" fn drop_globals(payload: *mut std::ffi::c_void) {
    drop(unsafe { Box::from_raw(payload as *mut StepGlobals) });
}

// ── C callback: one decode step (argmax in-graph) ──────────────────
//
// Inputs (via `mlx_vector_array`):
//   0  x             [1, 1, hidden]   — token embedding
//   1  offset        [1]              — position (INT32)
//   2..2+nl*2-1      — k_buf_i, v_buf_i for each layer (concat KV)
//
// Outputs (via `*outputs`):
//   0  token         [1]              — argmax result (INT32)
//   1..1+nl*2-1      — updated k_buf_i, v_buf_i

unsafe extern "C" fn step_trace_payload(
    outputs: *mut engine_mlx_ffi::mlx_vector_array,
    inputs: engine_mlx_ffi::mlx_vector_array,
    payload: *mut std::ffi::c_void,
) -> i32 {
    let g = &*(payload as *const StepGlobals);
    let result = match build_step(g, inputs) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                target: "engine_mlx::qwen3::compiled",
                "step_trace build_step FAILED: {e:?}"
            );
            return -2;
        }
    };
    *outputs = engine_mlx_ffi::mlx_vector_array_new_data(result.as_ptr(), result.len());
    0
}

fn build_step(g: &StepGlobals, inp: engine_mlx_ffi::mlx_vector_array) -> Result<Vec<mlx_array>> {
    let ctx = MlxCtx::new(g.stream);
    let cfg = &g.config;
    let w = &g.weights;
    let nl = w.layers.len();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    let scale = 1.0 / (hd as f32).sqrt();
    let (group_size, bits) = cfg
        .quantization
        .as_ref()
        .map(|q| (q.group_size as i32, q.bits as i32))
        .unwrap_or((64, 4));

    let n_inp = unsafe { engine_mlx_ffi::mlx_vector_array_size(inp) } as usize;
    // Static KV: [x, offset, mask, k/v...] (mask avoids in-graph slicing).
    // Concat KV: [x, offset, k/v...]. int4 compiled is not supported:
    // `mlx_slice`/`split` fail at compiled-graph eval with mlx-c 0.6.0
    // (see tests/compile_probe.rs); int4 stays eager-only.
    let is_static = g.kv_static_cap > 0;
    let expected = if is_static { 3 + nl * 2 } else { 2 + nl * 2 };
    if n_inp != expected {
        anyhow::bail!("step_trace: expected {} inputs (static={}), got {}", expected, is_static, n_inp);
    }
    if w.layers.len() != nl {
        anyhow::bail!("step_trace: layer count mismatch");
    }

    let arr = |idx: usize| -> mlx_array {
        let mut a: mlx_array = unsafe { std::mem::zeroed() };
        let _ = unsafe { engine_mlx_ffi::mlx_vector_array_get(&mut a, inp, idx) };
        a
    };

    let mut x = arr(0);
    let offset_arr = arr(1);

    // reshape with shape context for trace debugging (#23)
    let rsh = |ctx: &MlxCtx, x: mlx_array, shape: &[i32], what: &str| -> anyhow::Result<mlx_array> {
        let in_shape = ctx.shape(x).unwrap_or(vec![-999]);
        ctx.reshape(x, shape).map_err(|e| anyhow::anyhow!("reshape {} {:?} from {:?}: {:?}", what, shape, in_shape, e))
    };

    let mut outputs: Vec<mlx_array> = Vec::with_capacity(1 + nl * 2);

    // Dynamic split helper: fused activation [1,1,F] -> [1,1,len] at `start`.
    // Uses slice_dynamic (start as array) — the only split form that survives
    // mlx-c compiled eval (static slice/split fail at apply, status 1).
    let dsplit = |ctx: &MlxCtx, a: mlx_array, start: mlx_array, len: i32, what: &str| -> anyhow::Result<mlx_array> {
        ctx.slice_dynamic(a, start, &[2], &[1, 1, len])
            .map_err(|e| anyhow::anyhow!("dsplit {} len={}: {:?}", what, len, e))
    };

    for i in 0..nl {
        let lw = &w.layers[i];
        // Fused projections + dynamic splits (4 dispatches/layer, like eager).
        let normed = ctx.rms_norm(x, lw.input_layernorm, cfg.rms_norm_eps)?;
        let qkv = ctx.quantized_matmul(normed, lw.qkv_proj_w, lw.qkv_proj_s, lw.qkv_proj_b, true, group_size, bits)?;
        let q = dsplit(&ctx, qkv, g.starts.zero, nh * hd, "q")?;
        let k = dsplit(&ctx, qkv, g.starts.k, nkv * hd, "k")?;
        let v = dsplit(&ctx, qkv, g.starts.v, nkv * hd, "v")?;

        let q = rsh(&ctx, q, &[1, 1, nh, hd], "q")?;
        let mut k = rsh(&ctx, k, &[1, 1, nkv, hd], "k")?;
        let mut v = rsh(&ctx, v, &[1, 1, nkv, hd], "v")?;
        let q = if let Some(qn) = lw.q_norm { ctx.rms_norm(q, qn, cfg.rms_norm_eps)? } else { q };
        k = if let Some(kn) = lw.k_norm { ctx.rms_norm(k, kn, cfg.rms_norm_eps)? } else { k };
        let q = ctx.transpose_axes(q, &[0, 2, 1, 3])?;
        k = ctx.transpose_axes(k, &[0, 2, 1, 3])?;
        v = ctx.transpose_axes(v, &[0, 2, 1, 3])?;
        let q = ctx.rope_dynamic(q, hd, cfg.rope_theta, offset_arr)?;
        k = ctx.rope_dynamic(k, hd, cfg.rope_theta, offset_arr)?;

        // Attention input: static (in-place write + masked SDPA over capacity)
        // or concat (growing cache + causal SDPA). Shared tail below.
        let attn = if is_static {
            // No concat realloc, no in-graph slicing (both break mlx-c compile).
            let mask = arr(2);
            let k_cache = arr(3 + i * 2);
            let v_cache = arr(3 + i * 2 + 1);
            // scatter_single: updates.ndim == indices.ndim(1) + a.ndim(4).
            let k_s = ctx.reshape(k, &[1, 1, nkv, 1, hd])?;
            let v_s = ctx.reshape(v, &[1, 1, nkv, 1, hd])?;
            let k_upd = ctx.scatter_single(k_cache, offset_arr, k_s, 2)?;
            let v_upd = ctx.scatter_single(v_cache, offset_arr, v_s, 2)?;
            // NOTE: sdpa_masked takes (q,k,v,scale,mask); mask [1,1,1,cap].
            let a = ctx.sdpa_masked(q, k_upd, v_upd, scale, mask)?;
            outputs.push(k_upd);
            outputs.push(v_upd);
            a
        } else {
            let k_cache = arr(2 + i * 2);
            let v_cache = arr(2 + i * 2 + 1);
            let k_full = ctx.concatenate(k_cache, k, 2)?;
            let v_full = ctx.concatenate(v_cache, v, 2)?;
            outputs.push(k_full);
            outputs.push(v_full);
            ctx.sdpa(q, k_full, v_full, scale, true)?
        };
        let attn = ctx.transpose_axes(attn, &[0, 2, 1, 3])?;
        let attn = rsh(&ctx, attn, &[1, 1, nh * hd], "attn")?;
        let o = ctx.quantized_matmul(attn, lw.o_proj_w, lw.o_proj_s, lw.o_proj_b, true, group_size, bits)?;
        x = ctx.add(x, o)?;
        let normed2 = ctx.rms_norm(x, lw.post_attention_layernorm, cfg.rms_norm_eps)?;
        let gup = ctx.quantized_matmul(normed2, lw.gate_up_proj_w, lw.gate_up_proj_s, lw.gate_up_proj_b, true, group_size, bits)?;
        let gate = dsplit(&ctx, gup, g.starts.zero, cfg.intermediate_size, "gate")?;
        let up = dsplit(&ctx, gup, g.starts.up, cfg.intermediate_size, "up")?;
        let hidden = ctx.multiply(ctx.silu(gate)?, up)?;
        let down = ctx.quantized_matmul(hidden, lw.down_proj_w, lw.down_proj_s, lw.down_proj_b, true, group_size, bits)?;
        x = ctx.add(x, down)?;
    }

    x = ctx.rms_norm(x, w.final_norm, cfg.rms_norm_eps)?;
    x = rsh(&ctx, x, &[1, cfg.hidden_size], "lm")?;
    let (lm_w, lm_s, lm_b) = match (&w.lm_head_w, &w.lm_head_s, &w.lm_head_b) {
        (Some(hw), Some(hs), Some(hb)) => (*hw, *hs, *hb),
        _ => (w.embed_tokens, w.embed_scales, w.embed_biases),
    };
    let logits = ctx.quantized_matmul(x, lm_w, lm_s, lm_b, true, group_size, bits)?;
    let token = ctx.argmax(logits)?;
    outputs.insert(0, token);
    Ok(outputs)
}

// ── CompiledStep ────────────────────────────────────────────────────

pub(crate) struct CompiledStep {
    closure: engine_mlx_ffi::compile::Closure,
    n_layers: usize,
    pub(crate) kv_static: bool,
}

impl CompiledStep {
    pub fn new(weights: Qwen3Weights, config: Qwen3Config, stream: engine_mlx_ffi::mlx_stream) -> Result<Self> {
        Self::new_inner(weights, config, stream, false, 0)
    }

    pub fn new_static(weights: Qwen3Weights, config: Qwen3Config, stream: engine_mlx_ffi::mlx_stream, kv_static_cap: usize) -> Result<Self> {
        if kv_static_cap == 0 {
            anyhow::bail!("new_static requires capacity > 0");
        }
        let starts = Self::create_starts(stream, &config)?;
        Self::from_starts(weights, config, stream, starts, kv_static_cap)
    }

    /// Create SplitStarts constants + evaluate on GPU (tiny, ~0.1ms).
    /// Called separately so compilation can overlap with prefill in a background thread.
    pub fn create_starts(stream: engine_mlx_ffi::mlx_stream, config: &Qwen3Config) -> Result<SplitStarts> {
        let sctx = MlxCtx::new(stream);
        let nhd = config.num_attention_heads * config.head_dim();
        let nkvd = config.num_key_value_heads * config.head_dim();
        let ff = config.intermediate_size;
        let starts = SplitStarts {
            zero: sctx.new_array_i32(&[0])?,
            k: sctx.new_array_i32(&[nhd])?,
            v: sctx.new_array_i32(&[nhd + nkvd])?,
            up: sctx.new_array_i32(&[ff])?,
        };
        let all = [starts.zero, starts.k, starts.v, starts.up];
        sctx.eval_all(&all).map_err(|e| anyhow::anyhow!("starts eval: {:?}", e))?;
        unsafe { engine_mlx_ffi::mlx_synchronize(stream) };
        Ok(starts)
    }

    /// Compile from pre-computed SplitStarts (CPU-only, no GPU work).
    /// Can be called from a background thread to overlap with GPU prefill.
    pub fn from_starts(
        weights: Qwen3Weights,
        config: Qwen3Config,
        stream: engine_mlx_ffi::mlx_stream,
        starts: SplitStarts,
        kv_static_cap: usize,
    ) -> Result<Self> {
        let n_layers = weights.layers.len();
        let globals = Box::into_raw(Box::new(StepGlobals { weights, starts, config, stream, kv_static_cap }));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(step_trace_payload),
                globals as *mut std::ffi::c_void,
                Some(drop_globals),
            )
        };
        let closure = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let closure = engine_mlx_ffi::compile::compile_shapeless(closure)?;
        tracing::info!(
            target: "engine_mlx::qwen3::compiled",
            "step compiled (argmax in-graph, unfused) n_layers={}", n_layers
        );
        Ok(Self { closure, n_layers, kv_static: kv_static_cap > 0 })
    }

    fn new_inner(weights: Qwen3Weights, config: Qwen3Config, stream: engine_mlx_ffi::mlx_stream, _is_int4: bool, kv_static_cap: usize) -> Result<Self> {
        let starts = Self::create_starts(stream, &config)?;
        Self::from_starts(weights, config, stream, starts, kv_static_cap)
    }

    pub fn apply(&self, inputs: &[mlx_array]) -> Result<Vec<mlx_array>> {
        self.closure.apply(inputs)
    }

    pub fn n_inputs(&self) -> usize {
        if self.kv_static {
            3 + self.n_layers * 2
        } else {
            2 + self.n_layers * 2
        }
    }
}
