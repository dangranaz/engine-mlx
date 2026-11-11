//! Per-projection quantization parameter resolution from the model manifest.
//!
//! Maps module names (as used in config.json) to `(group_size, bits)` via
//! `QuantConfig::quant_for` so the loader can store them in the weight structs
//! (`QuantTensor` / `QkvProj` / `GateUpProj`). Task 09 uses them to gate fusion.
//!
//! Logging: all five public functions funnel through [`quant_params_for`], so a
//! single `trace!` there covers them without duplicate per-call log lines in
//! the per-layer loader loop.

use nxm_modelplan::ModelManifest;

use super::QuantParams;

/// Resolve the quant params for one module name from the manifest.
///
/// Returns `QuantParams { group_size: 0, bits: 0 }` when the model is not
/// quantized (`manifest.quant` is `None`), matching the `0 = none` convention
/// of `Qwen3Config::quant_bits`/`quant_group_size`.
pub fn quant_params_for(manifest: &ModelManifest, module_name: &str) -> QuantParams {
    let params = match &manifest.quant {
        Some(q) => {
            let (group_size, bits) = q.quant_for(module_name);
            QuantParams { group_size, bits }
        }
        None => QuantParams {
            group_size: 0,
            bits: 0,
        },
    };
    tracing::trace!(
        target: "engine_mlx::qwen3::quant",
        module = module_name,
        group_size = params.group_size,
        bits = params.bits,
        "quant_params_for"
    );
    params
}

/// Quant params of all projections of a single layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerQuantParams {
    /// Params of the fused `qkv_proj`, taken from `q_proj` (task 08 always builds
    /// `Fused`; task 09 gates on q/k/v params equality).
    pub qkv: QuantParams,
    pub o: QuantParams,
    /// Params of the fused `gate_up_proj`, taken from `gate_proj`.
    pub gate_up: QuantParams,
    pub down: QuantParams,
}

/// Resolve the per-projection quant params for layer `i` of the manifest.
pub fn layer_quant_params(manifest: &ModelManifest, i: usize) -> LayerQuantParams {
    let p = format!("model.layers.{i}");
    LayerQuantParams {
        qkv: quant_params_for(manifest, &format!("{p}.self_attn.q_proj")),
        o: quant_params_for(manifest, &format!("{p}.self_attn.o_proj")),
        gate_up: quant_params_for(manifest, &format!("{p}.mlp.gate_proj")),
        down: quant_params_for(manifest, &format!("{p}.mlp.down_proj")),
    }
}

/// Quant params of `model.embed_tokens`.
pub fn embed_quant_params(manifest: &ModelManifest) -> QuantParams {
    quant_params_for(manifest, "model.embed_tokens")
}

/// Quant params of `lm_head`.
pub fn lm_head_quant_params(manifest: &ModelManifest) -> QuantParams {
    quant_params_for(manifest, "lm_head")
}
