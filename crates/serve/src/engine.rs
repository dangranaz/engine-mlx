//! Minimal Qwen3 engine tracer bullet.
//!
//! Loads weights via `loader::load_engine`, sets up MLX stream/context,
//! configures memory limits, dequantizes the embed table and exposes a
//! minimal `generate` that exercises `MlxCtx` ops without requiring full
//! attention / KV cache / prefill pipelines.

use std::path::Path;

use anyhow::{Context, Result};
use engine_mlx_ffi::{mlx_array, mlx_stream, MlxCtx};
use nxm_tokenizer::chat_template::ChatTemplate;

use crate::compiled::CompiledStep;
use crate::config::Qwen3Config;
use crate::forward::sysinfo_total_ram;
use crate::loader::{EngineInit, Qwen3Weights};
use crate::sampler::{SamplingParams, SimpleRng};
use crate::tokenizer::Tokenizer;

pub mod chat;

/// Minimal engine for the tracer bullet.
pub struct Qwen3Engine {
    pub config: Qwen3Config,
    pub weights: Qwen3Weights,
    pub ctx: MlxCtx,
    pub tokenizer: Tokenizer,
    pub embed_table: Option<mlx_array>,
    pub stream: mlx_stream,
    pub offset: usize,
    pub kv_bufs: Vec<mlx_array>,
    /// Static KV capacity (>0 = preallocated [1,nkv,cap,hd] buffers + masked SDPA).
    /// 0 = concat (growing) KV mode.
    pub kv_static_cap: usize,
    /// Single additive attention mask [1,1,1,MAX_CAP] for static KV decode.
    /// Updated in-place per step (0.0 at 0..=offset, -1e9 beyond). Token-exact
    /// vs pre-materialized O(cap) masks or per-step reconstruction.
    pub static_mask: Option<mlx_array>,
    /// Pre-allocated offset arrays `[0], [1], ..., [cap-1]` for static KV decode.
    /// Eliminates per-step `new_array_i32` FFI call + 4-byte H2D transfer.
    pub static_offsets: Vec<mlx_array>,
    pub(crate) compiled_full: Option<CompiledStep>,
    pub(crate) compiled_static: Option<CompiledStep>,
    pub sampling: SamplingParams,
    pub(crate) rng: SimpleRng,
    /// The model's own chat template, loaded once at load time from `model_dir`.
    /// `None` when the model ships no `chat_template.jinja` / `chat_template`
    /// key — in that case `generate_chat` returns an error rather than falling
    /// back to a hardcoded template.
    pub chat_template: Option<ChatTemplate>,
}

unsafe impl Send for Qwen3Engine {}
unsafe impl Sync for Qwen3Engine {}

/// Backwards compat for stub tests – old `Engine` type.
pub type Engine = Qwen3Engine;

impl Qwen3Engine {
    /// Build from a pre-loaded `EngineInit`.
    pub fn from_init(init: EngineInit, model_dir: &Path) -> Result<Self> {
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::from_init START model_dir={}",
            model_dir.display()
        );

        let stream = {
            #[cfg(feature = "mlx")]
            {
                #[allow(unused_unsafe)]
                unsafe { engine_mlx_ffi::mlx_default_gpu_stream_new() }
            }
            #[cfg(not(feature = "mlx"))]
            {
                unsafe { std::mem::zeroed() }
            }
        };
        let ctx = MlxCtx::new(stream);

        // ── Memory limits (70% RAM, 128 MB cache, conditional wired) ──
        #[cfg(feature = "mlx")]
        unsafe {
            let sys_ram = sysinfo_total_ram();
            let mut prev: usize = 0;
            let mem_limit = (sys_ram as f64 * 0.70) as usize;
            engine_mlx_ffi::mlx_set_memory_limit(&mut prev, mem_limit);
            engine_mlx_ffi::mlx_set_cache_limit(&mut prev, 128 * 1024 * 1024);
            let params_b = init.manifest.model.params_b as f64;
            let model_bytes = (params_b * 0.55 * 1024.0 * 1024.0 * 1024.0) as usize;
            let headroom = sys_ram.saturating_sub(model_bytes + 4 * 1024 * 1024 * 1024);
            if headroom > 2 * 1024 * 1024 * 1024 {
                let wired = model_bytes + 3 * 1024 * 1024 * 1024;
                engine_mlx_ffi::mlx_set_wired_limit(&mut prev, wired);
                tracing::info!(
                    target: "engine_mlx::qwen3::engine",
                    "memory configured wired mem_limit_mb={} wired_limit_mb={}",
                    mem_limit / (1024 * 1024),
                    wired / (1024 * 1024)
                );
            } else {
                tracing::info!(
                    target: "engine_mlx::qwen3::engine",
                    "memory configured no-wired mem_limit_mb={} headroom_mb={}",
                    mem_limit / (1024 * 1024),
                    headroom / (1024 * 1024)
                );
            }
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = sysinfo_total_ram();
            tracing::info!(
                target: "engine_mlx::qwen3::engine",
                "memory limits skipped (mlx feature disabled)"
            );
        }

        // ── Embed table: dequant if mlx available, otherwise None ──
        let embed_table: Option<mlx_array> = {
            #[cfg(feature = "mlx")]
            {
                let (group_size, bits) = init
                    .weights
                    .embed_tokens
                    .quant
                    .to_ffi("model.embed_tokens")?;
                match ctx.dequantize_weight(
                    init.weights.embed_tokens.w,
                    init.weights.embed_tokens.s,
                    init.weights.embed_tokens.b,
                    group_size,
                    bits,
                ) {
                    Ok(tbl) => {
                        if let Err(e) = ctx.evaluate(tbl) {
                            tracing::info!(
                                target: "engine_mlx::qwen3::engine",
                                "embed evaluate failed (lazy) err={e:?}"
                            );
                        }
                        tracing::info!(target: "engine_mlx::qwen3::engine", "embed_table ready");
                        Some(tbl)
                    }
                    Err(e) => {
                        tracing::info!(
                            target: "engine_mlx::qwen3::engine",
                            "embed dequant skipped err={e:?}"
                        );
                        None
                    }
                }
            }
            #[cfg(not(feature = "mlx"))]
            {
                tracing::info!(
                    target: "engine_mlx::qwen3::engine",
                    "embed_table None (mlx feature disabled)"
                );
                None
            }
        };

        let tokenizer =
            Tokenizer::load(model_dir).with_context(|| "failed to load tokenizer")?;

        // ── Chat template: load the model's own template once at load time. ─
        let chat_template = match ChatTemplate::from_model_dir(model_dir) {
            Ok(t) => {
                tracing::info!(
                    target: "engine_mlx::serve::engine",
                    "chat template loaded from {}",
                    model_dir.display()
                );
                Some(t)
            }
            Err(e) => {
                tracing::warn!(
                    target: "engine_mlx::serve::engine",
                    "no chat template found in {}: {}",
                    model_dir.display(),
                    e
                );
                None
            }
        };

        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::from_init END hidden={} layers={} offset=0",
            init.config.hidden_size,
            init.config.num_hidden_layers
        );

        // ── Fixed-capacity static KV + compiled closure (E5 fix) ────────────
        // Compile once at MAX_CAP to avoid per-request recompilation that
        // accumulates in MLX's global compile cache and triggers Resource limit.
        // MAX_CAP must cover the largest request we expect (tests use up to
        // 2048 max_tokens + prompt). 4096 gives headroom while keeping KV
        // memory reasonable (~0.28 GB at 4096 for K2-Horizon 3.7B).
        const MAX_CAP: usize = 4096;
        let mut eng = Self {
            config: init.config,
            weights: init.weights,
            ctx,
            tokenizer,
            embed_table,
            stream,
            offset: 0,
            kv_bufs: Vec::new(),
            kv_static_cap: 0,
            static_mask: None,
            static_offsets: Vec::new(),
            compiled_full: None,
            compiled_static: None,
            sampling: SamplingParams::default(),
            rng: SimpleRng::new(42),
            chat_template,
        };
        #[cfg(feature = "mlx")]
        {
            eng.alloc_kv_static(MAX_CAP)?;
            eng.ensure_compiled_static()?;
        }
        Ok(eng)
    }

    /// Set sampling params (temperature/top-p/top-k/seed).
    pub fn set_sampling(&mut self, params: SamplingParams) {
        self.sampling = params.clone();
        self.rng = SimpleRng::new(params.seed);
    }

    /// Compatibility loader from `model_dir` (calls `loader::load_engine`).
    pub fn load(model_dir: &Path) -> Result<Self> {
        tracing::info!(
            target: "engine_mlx::qwen3::engine",
            "Qwen3Engine::load START model_dir={}",
            model_dir.display()
        );
        let init = crate::loader::load_engine(model_dir).context("load_engine failed")?;
        let eng = Self::from_init(init, model_dir)?;
        tracing::info!(target: "engine_mlx::qwen3::engine", "Qwen3Engine::load END");
        Ok(eng)
    }
}
