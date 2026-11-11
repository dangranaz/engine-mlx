//! Tests for per-projection quant params in the loader weight structs (task 08 / B01).
//!
//! The loader itself needs mlx-c (FFI), so these tests exercise the pure
//! resolution (`loader::quant_params`) plus the fused/separate representation
//! accessors with dummy arrays.

use std::collections::HashMap;

use engine_mlx_ffi::mlx_array;
use engine_mlx_serve::loader::quant_params::{
    embed_quant_params, layer_quant_params, lm_head_quant_params, quant_params_for,
};
use engine_mlx_serve::loader::{GateUpProj, QkvProj, QuantParams, QuantTensor};
use nxm_modelplan::{ModelManifest, PerTensorQuantConfig, QuantConfig};

fn dummy() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn dummy_quant_tensor(quant: QuantParams) -> QuantTensor {
    QuantTensor {
        w: dummy(),
        s: dummy(),
        b: dummy(),
        quant,
    }
}

fn manifest_with_quant(
    format: &str,
    group_size: usize,
    overrides: &[(&str, usize, usize)],
) -> ModelManifest {
    let overrides = overrides
        .iter()
        .map(|(k, gs, bits)| {
            (
                k.to_string(),
                PerTensorQuantConfig {
                    bits: Some(*bits),
                    group_size: Some(*gs),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    ModelManifest {
        quant: Some(QuantConfig {
            format: format.into(),
            group_size,
            overrides,
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn loader_layer_quant_params() {
    // K2-Horizon-like mixed config: q/k/v 8-bit/group 64, gate/up 4-bit/group 32.
    let m = manifest_with_quant(
        "q4",
        64,
        &[
            ("model.layers.0.self_attn.q_proj", 64, 8),
            ("model.layers.0.mlp.gate_proj", 32, 4),
        ],
    );
    let p = layer_quant_params(&m, 0);
    // Fused qkv_proj records the params of q_proj (task 08 always builds Fused).
    assert_eq!(
        p.qkv,
        QuantParams {
            group_size: 64,
            bits: 8
        }
    );
    // Fused gate_up_proj records the params of gate_proj.
    assert_eq!(
        p.gate_up,
        QuantParams {
            group_size: 32,
            bits: 4
        }
    );
    // o_proj / down_proj have no override -> global default (64, 4).
    assert_eq!(
        p.o,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        p.down,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );

    // The loader builds the Fused variant carrying those params.
    let qkv = QkvProj::Fused(dummy_quant_tensor(p.qkv));
    let (_, _, _, qp) = qkv.fused().expect("Fused variant must expose a fused view");
    assert_eq!(
        qp,
        QuantParams {
            group_size: 64,
            bits: 8
        }
    );
    let gup = GateUpProj::Fused(dummy_quant_tensor(p.gate_up));
    assert_eq!(
        gup.fused().map(|(_, _, _, q)| q),
        Some(QuantParams {
            group_size: 32,
            bits: 4
        })
    );
}

#[test]
fn loader_toplevel_quant_params() {
    let m = manifest_with_quant(
        "q4",
        64,
        &[("model.embed_tokens", 32, 4), ("lm_head", 64, 8)],
    );
    assert_eq!(
        embed_quant_params(&m),
        QuantParams {
            group_size: 32,
            bits: 4
        }
    );
    assert_eq!(
        lm_head_quant_params(&m),
        QuantParams {
            group_size: 64,
            bits: 8
        }
    );
}

#[test]
fn loader_quant_params_default_fallback() {
    // Uniform q4 / group 64, no overrides: every projection reports (64, 4).
    let m = manifest_with_quant("q4", 64, &[]);
    let p = layer_quant_params(&m, 3);
    assert_eq!(
        p.qkv,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        p.o,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        p.gate_up,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        p.down,
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        embed_quant_params(&m),
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
    assert_eq!(
        lm_head_quant_params(&m),
        QuantParams {
            group_size: 64,
            bits: 4
        }
    );
}

#[test]
fn separate_variant_has_no_fused_view() {
    // TODO(task 09): the loader builds Separate when quant params diverge;
    // until then the accessor must return None for it (forward bails via context).
    let qt = || {
        dummy_quant_tensor(QuantParams {
            group_size: 64,
            bits: 8,
        })
    };
    assert!(QkvProj::Separate {
        q: qt(),
        k: qt(),
        v: qt()
    }
    .fused()
    .is_none());
    assert!(GateUpProj::Separate {
        gate: qt(),
        up: qt()
    }
    .fused()
    .is_none());
}

#[test]
fn unquantized_manifest_yields_zero_params() {
    // manifest.quant == None -> (0, 0), matching the 0 = none convention of Qwen3Config.
    let m = ModelManifest::default();
    assert_eq!(
        quant_params_for(&m, "model.layers.0.self_attn.q_proj"),
        QuantParams {
            group_size: 0,
            bits: 0
        }
    );
}
