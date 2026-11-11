//! Load-time fusion gating for q/k/v and gate/up (task 09 / B02).
//!
//! A group of projections is fused at load time ONLY when every tensor in the
//! group shares the same `(group_size, bits)`; otherwise the tensors stay
//! separate, so the loader never produces a fused tensor with inconsistent
//! quant params. The decision is pure (manifest -> params -> decision) and
//! unit-tested in `tests/loader_fusion.rs`.

use nxm_modelplan::ModelManifest;

use super::quant_params::quant_params_for;
use super::QuantParams;

/// Whether a group of projections is fused at load time or kept separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fusion {
    Fused,
    Separate,
}

/// Fuse only when every tensor in the group has identical quant params.
///
/// An empty group never fuses (nothing to concatenate).
pub fn fusion_decision(params: &[QuantParams]) -> Fusion {
    match params.split_first() {
        Some((first, rest)) if rest.iter().all(|p| p == first) => Fusion::Fused,
        Some(_) => Fusion::Separate,
        None => Fusion::Separate,
    }
}

/// Quant params of q/k/v for layer `i`, in concat order (q, k, v).
pub fn qkv_params(manifest: &ModelManifest, i: usize) -> [QuantParams; 3] {
    let p = format!("model.layers.{i}.self_attn");
    [
        quant_params_for(manifest, &format!("{p}.q_proj")),
        quant_params_for(manifest, &format!("{p}.k_proj")),
        quant_params_for(manifest, &format!("{p}.v_proj")),
    ]
}

/// Quant params of gate/up for layer `i`, in concat order (gate, up).
pub fn gate_up_params(manifest: &ModelManifest, i: usize) -> [QuantParams; 2] {
    let p = format!("model.layers.{i}.mlp");
    [
        quant_params_for(manifest, &format!("{p}.gate_proj")),
        quant_params_for(manifest, &format!("{p}.up_proj")),
    ]
}

/// Fusion decision for the q/k/v group of layer `i`.
pub fn qkv_fusion(manifest: &ModelManifest, i: usize) -> Fusion {
    fusion_decision(&qkv_params(manifest, i))
}

/// Fusion decision for the gate/up group of layer `i`.
pub fn gate_up_fusion(manifest: &ModelManifest, i: usize) -> Fusion {
    fusion_decision(&gate_up_params(manifest, i))
}
