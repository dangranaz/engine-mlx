//! Qwen3 weight loader — safetensors loading via Rust safetensors crate + mlx-c arrays.
//!
//! Weights are stored in flat structs for zero-overhead access in the forward pass.

use std::path::Path;

use anyhow::{Context, Result};
use nxm_modelplan::ModelManifest;

use engine_mlx_ffi::mlx_array;

/// Quantization parameters of a single weight tensor.
/// `group_size: 0, bits: 0` means "not quantized" (same 0 = none convention as `Qwen3Config`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantParams {
    pub group_size: usize,
    pub bits: usize,
}

impl QuantParams {
    /// Convert to the `(group_size, bits)` i32 pair expected by the quantized FFI
    /// ops (`quantized_matmul`, `dequantize_weight`).
    ///
    /// Bails when `group_size == 0`: that is the "not quantized" sentinel
    /// (`manifest.quant == None`), and passing 0 to a quantized op would be a
    /// silent bug — so we stop with a contextual error naming the projection.
    ///
    /// # Example
    /// ```
    /// # use engine_mlx_serve::loader::QuantParams;
    /// let qp = QuantParams { group_size: 64, bits: 8 };
    /// assert_eq!(qp.to_ffi("qkv_proj").unwrap(), (64, 8));
    /// assert!(QuantParams { group_size: 0, bits: 0 }.to_ffi("qkv_proj").is_err());
    /// ```
    pub fn to_ffi(&self, what: &str) -> Result<(i32, i32)> {
        if self.group_size == 0 {
            anyhow::bail!(
                "{what}: quant params are (0,0) — model not quantized (manifest.quant is None); \
                 refusing to call a quantized op with group_size=0"
            );
        }
        Ok((self.group_size as i32, self.bits as i32))
    }
}

/// A quantized weight tensor (weight + scales + biases) with its quant params.
#[derive(Clone)]
pub struct QuantTensor {
    pub w: mlx_array,
    pub s: mlx_array,
    pub b: mlx_array,
    pub quant: QuantParams,
}

/// Q/K/V projection: fused into one tensor (the case the loader builds today)
/// or kept as separate q/k/v tensors.
///
/// TODO(task 09): the loader builds `Separate` when the per-tensor quant params
/// of q/k/v diverge. TODO(task C): the forward pass consumes `Separate`.
#[derive(Clone)]
pub enum QkvProj {
    Fused(QuantTensor),
    Separate {
        q: QuantTensor,
        k: QuantTensor,
        v: QuantTensor,
    },
}

impl QkvProj {
    /// Fused view of the projection: `Some((w, s, b, quant))` for `Fused`, `None` for `Separate`.
    ///
    /// # Example
    /// ```
    /// # use engine_mlx_serve::loader::{QkvProj, QuantParams, QuantTensor};
    /// # use engine_mlx_ffi::mlx_array;
    /// # fn dummy() -> mlx_array { unsafe { std::mem::zeroed() } }
    /// # let t = QuantTensor { w: dummy(), s: dummy(), b: dummy(), quant: QuantParams { group_size: 64, bits: 8 } };
    /// let proj = QkvProj::Fused(t);
    /// let (w, s, b, quant) = proj.fused().unwrap();
    /// assert_eq!(quant, QuantParams { group_size: 64, bits: 8 });
    /// ```
    pub fn fused(&self) -> Option<(&mlx_array, &mlx_array, &mlx_array, QuantParams)> {
        match self {
            QkvProj::Fused(t) => Some((&t.w, &t.s, &t.b, t.quant)),
            QkvProj::Separate { .. } => None,
        }
    }
}

/// Gate/Up projection: fused into one tensor (the case the loader builds today)
/// or kept as separate gate/up tensors.
///
/// TODO(task 09): the loader builds `Separate` when the per-tensor quant params
/// of gate/up diverge. TODO(task C): the forward pass consumes `Separate`.
#[derive(Clone)]
pub enum GateUpProj {
    Fused(QuantTensor),
    Separate { gate: QuantTensor, up: QuantTensor },
}

impl GateUpProj {
    /// Fused view of the projection: `Some((w, s, b, quant))` for `Fused`, `None` for `Separate`.
    ///
    /// # Example
    /// ```
    /// # use engine_mlx_serve::loader::{GateUpProj, QuantParams, QuantTensor};
    /// # use engine_mlx_ffi::mlx_array;
    /// # fn dummy() -> mlx_array { unsafe { std::mem::zeroed() } }
    /// # let t = QuantTensor { w: dummy(), s: dummy(), b: dummy(), quant: QuantParams { group_size: 32, bits: 4 } };
    /// let proj = GateUpProj::Fused(t);
    /// assert!(proj.fused().is_some());
    /// ```
    pub fn fused(&self) -> Option<(&mlx_array, &mlx_array, &mlx_array, QuantParams)> {
        match self {
            GateUpProj::Fused(t) => Some((&t.w, &t.s, &t.b, t.quant)),
            GateUpProj::Separate { .. } => None,
        }
    }
}

/// Pre-resolved weight pointers for a single transformer layer.
/// Zero HashMap lookups in the hot path.
#[derive(Clone)]
pub struct Qwen3LayerWeights {
    pub input_layernorm: mlx_array,
    pub post_attention_layernorm: mlx_array,
    pub qkv: QkvProj,
    pub o_proj: QuantTensor,
    pub q_norm: Option<mlx_array>,
    pub k_norm: Option<mlx_array>,
    pub gate_up: GateUpProj,
    pub down_proj: QuantTensor,
}

#[derive(Clone)]
pub struct Qwen3Weights {
    pub embed_tokens: QuantTensor,
    pub final_norm: mlx_array,
    pub layers: Vec<Qwen3LayerWeights>,
    // LM head — None means tied (reuse embed_tokens).
    pub lm_head: Option<QuantTensor>,
}

/// Engine initialization config containing manifest, config, and weights.
#[derive(Clone)]
pub struct EngineInit {
    pub manifest: ModelManifest,
    pub config: crate::config::Qwen3Config,
    pub weights: Qwen3Weights,
}

pub mod fusion;
pub mod quant_params;
pub mod tensor;
pub mod weights;
use weights::load_weights_from_manifest;

/// Load engine configuration (manifest + config + weights) from model directory.
pub fn load_engine(model_dir: &Path) -> Result<EngineInit> {
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_engine START model_dir={}",
        model_dir.display()
    );
    let start = std::time::Instant::now();

    // 1. Load manifest (generates/caches model.plan.toml + sidecar)
    let manifest =
        nxm_modelplan::ensure_plan(model_dir).with_context(|| "failed to ensure_plan")?;
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "manifest loaded family={} layers={}",
        manifest.model.family,
        manifest.layers.total
    );

    // 2. Build config from manifest
    let config = crate::config::Qwen3Config::from_manifest(&manifest)?;

    // 3. Load weights
    tracing::info!(target: "engine_mlx::qwen3::loader", "load_weights START");
    let weights = load_weights_from_manifest(model_dir, &manifest)?;
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_weights END layers={}",
        config.num_hidden_layers
    );

    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "Loaded {} layers, hidden={}, heads={}/kv={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    );

    let elapsed = start.elapsed();
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_engine END elapsed_ms={}",
        elapsed.as_millis()
    );

    Ok(EngineInit {
        manifest,
        config,
        weights,
    })
}

/// Load weights from model directory (convenience wrapper).
pub fn load_weights(model_dir: &Path) -> Result<Qwen3Weights> {
    let manifest =
        nxm_modelplan::ensure_plan(model_dir).with_context(|| "failed to ensure_plan")?;
    load_weights_from_manifest(model_dir, &manifest)
}
