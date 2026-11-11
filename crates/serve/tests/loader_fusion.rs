//! Tests for load-time fusion gating (task 09 / B02): q/k/v and gate/up are
//! fused at load time ONLY when their per-tensor quant params `(group,bits)`
//! coincide; otherwise they stay separate (no fused tensor with inconsistent
//! quant params). The loader itself needs mlx-c, so these tests exercise the
//! pure decision in `loader::fusion` plus the Fused/Separate representation.

use std::collections::HashMap;

use engine_mlx_ffi::mlx_array;
use engine_mlx_serve::loader::fusion::{fusion_decision, gate_up_fusion, qkv_fusion, Fusion};
use engine_mlx_serve::loader::{GateUpProj, QkvProj, QuantParams, QuantTensor};
use nxm_modelplan::{ModelManifest, PerTensorQuantConfig, QuantConfig};

fn dummy() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn qt(group_size: usize, bits: usize) -> QuantTensor {
    QuantTensor {
        w: dummy(),
        s: dummy(),
        b: dummy(),
        quant: QuantParams { group_size, bits },
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
fn fusion_enabled_when_quant_params_match() {
    // K2-Horizon real case: q/k/v all (64,8), gate/up all (32,4) -> both Fused.
    let m = manifest_with_quant(
        "q4",
        64,
        &[
            ("model.layers.0.self_attn.q_proj", 64, 8),
            ("model.layers.0.self_attn.k_proj", 64, 8),
            ("model.layers.0.self_attn.v_proj", 64, 8),
            ("model.layers.0.mlp.gate_proj", 32, 4),
            ("model.layers.0.mlp.up_proj", 32, 4),
        ],
    );
    assert_eq!(qkv_fusion(&m, 0), Fusion::Fused);
    assert_eq!(gate_up_fusion(&m, 0), Fusion::Fused);
    assert_eq!(
        fusion_decision(&[
            QuantParams {
                group_size: 64,
                bits: 8
            },
            QuantParams {
                group_size: 64,
                bits: 8
            },
            QuantParams {
                group_size: 64,
                bits: 8
            },
        ]),
        Fusion::Fused
    );
    // The built layer is in Fused form, exactly like the previous behavior.
    let qkv = QkvProj::Fused(qt(64, 8));
    let (_, _, _, qp) = qkv.fused().expect("Fused variant must expose a fused view");
    assert_eq!(
        qp,
        QuantParams {
            group_size: 64,
            bits: 8
        }
    );
}

#[test]
fn fusion_disabled_when_quant_params_diverge() {
    // q_proj=(64,8) but k_proj=(32,4): no fused qkv tensor may be produced.
    let m = manifest_with_quant(
        "q4",
        64,
        &[
            ("model.layers.0.self_attn.q_proj", 64, 8),
            ("model.layers.0.self_attn.k_proj", 32, 4),
        ],
    );
    assert_eq!(qkv_fusion(&m, 0), Fusion::Separate);
    assert_eq!(
        fusion_decision(&[
            QuantParams {
                group_size: 64,
                bits: 8
            },
            QuantParams {
                group_size: 32,
                bits: 4
            },
            QuantParams {
                group_size: 64,
                bits: 4
            }, // v_proj: default fallback
        ]),
        Fusion::Separate
    );
    // Separate keeps each piece with its own quant params; no fused view.
    let qkv = QkvProj::Separate {
        q: qt(64, 8),
        k: qt(32, 4),
        v: qt(64, 4),
    };
    assert!(
        qkv.fused().is_none(),
        "no fused qkv tensor when params diverge"
    );
    match qkv {
        QkvProj::Separate { q, k, v } => {
            assert_eq!(
                q.quant,
                QuantParams {
                    group_size: 64,
                    bits: 8
                }
            );
            assert_eq!(
                k.quant,
                QuantParams {
                    group_size: 32,
                    bits: 4
                }
            );
            assert_eq!(
                v.quant,
                QuantParams {
                    group_size: 64,
                    bits: 4
                }
            );
        }
        QkvProj::Fused(_) => panic!("divergent params must not produce a Fused qkv"),
    }
}

#[test]
fn fusion_decision_is_per_group() {
    // q/k/v coincide but gate != up: qkv Fused and gate_up Separate, same layer.
    let m = manifest_with_quant(
        "q4",
        64,
        &[
            ("model.layers.0.self_attn.q_proj", 64, 8),
            ("model.layers.0.self_attn.k_proj", 64, 8),
            ("model.layers.0.self_attn.v_proj", 64, 8),
            ("model.layers.0.mlp.gate_proj", 32, 4),
            ("model.layers.0.mlp.up_proj", 64, 8),
        ],
    );
    assert_eq!(qkv_fusion(&m, 0), Fusion::Fused);
    assert_eq!(gate_up_fusion(&m, 0), Fusion::Separate);
    let gup = GateUpProj::Separate {
        gate: qt(32, 4),
        up: qt(64, 8),
    };
    assert!(gup.fused().is_none());
    match gup {
        GateUpProj::Separate { gate, up } => {
            assert_eq!(
                gate.quant,
                QuantParams {
                    group_size: 32,
                    bits: 4
                }
            );
            assert_eq!(
                up.quant,
                QuantParams {
                    group_size: 64,
                    bits: 8
                }
            );
        }
        GateUpProj::Fused(_) => panic!("divergent params must not produce a Fused gate_up"),
    }
}

#[test]
fn fusion_decision_edge_cases() {
    // Single tensor always fuses; empty group never fuses (no unwrap/panic).
    assert_eq!(
        fusion_decision(&[QuantParams {
            group_size: 64,
            bits: 4
        }]),
        Fusion::Fused
    );
    assert_eq!(fusion_decision(&[]), Fusion::Separate);
}
