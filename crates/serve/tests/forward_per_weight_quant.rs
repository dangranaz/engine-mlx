//! Tests for per-weight quant params consumption in the forward (task 11 / C1).
//!
//! R1 (every quantized op receives its own projection's `(group_size, bits)`)
//! is not observable without the mlx FFI graph, so it is pinned here at the
//! param-resolution level (`QuantParams::to_ffi` per projection) and end-to-end
//! by the hardware parity test `k2_horizon_token_parity` (R3, `#[ignore]`).
//! R2 (`group_size == 0` must bail, never reach a quantized op) is pure.

use std::collections::HashMap;

use engine_mlx_serve::loader::fusion::{gate_up_params, qkv_params};
use engine_mlx_serve::loader::quant_params::layer_quant_params;
use engine_mlx_serve::loader::QuantParams;
use nxm_modelplan::{ModelManifest, PerTensorQuantConfig, QuantConfig};

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

/// R2: `QuantParams { group_size: 0, bits: 0 }` is the "not quantized" sentinel
/// (`manifest.quant == None`). Passing it to a quantized op would be a silent
/// bug, so `to_ffi` must return a contextual Err naming the projection.
#[test]
fn quant_params_zero_group_size_bails() {
    let err = QuantParams {
        group_size: 0,
        bits: 0,
    }
    .to_ffi("model.layers.0.self_attn.qkv_proj")
    .expect_err("group_size == 0 must bail, never reach a quantized op");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("qkv_proj"),
        "error must name the projection: {msg}"
    );
    assert!(
        msg.contains("group_size"),
        "error must mention group_size: {msg}"
    );

    // Valid params convert to the i32 pair the FFI ops expect.
    assert_eq!(
        QuantParams {
            group_size: 64,
            bits: 8
        }
        .to_ffi("qkv_proj")
        .unwrap(),
        (64, 8)
    );
}

/// R1 (param level): each forward site gets the params of ITS projection via
/// `QuantParams::to_ffi`, and they diverge across projections in a mixed-quant
/// config — the old single global pair (fallback (64,4)) would be wrong for
/// qkv (64,8) and gate_up (32,4). The wiring into `quantized_matmul` itself is
/// covered end-to-end by `k2_horizon_token_parity` (needs GPU+model).
#[test]
fn forward_uses_per_weight_quant_params() {
    // K2-Horizon-like manifest: q/k/v (64,8), gate/up (32,4), o/down default (64,4).
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
    let qkv = qkv_params(&m, 0);
    assert_eq!(qkv[0].to_ffi("self_attn.q_proj").unwrap(), (64, 8));
    assert_eq!(qkv[1].to_ffi("self_attn.k_proj").unwrap(), (64, 8));
    assert_eq!(qkv[2].to_ffi("self_attn.v_proj").unwrap(), (64, 8));
    let gup = gate_up_params(&m, 0);
    assert_eq!(gup[0].to_ffi("mlp.gate_proj").unwrap(), (32, 4));
    assert_eq!(gup[1].to_ffi("mlp.up_proj").unwrap(), (32, 4));
    let lp = layer_quant_params(&m, 0);
    assert_eq!(lp.o.to_ffi("self_attn.o_proj").unwrap(), (64, 4));
    assert_eq!(lp.down.to_ffi("mlp.down_proj").unwrap(), (64, 4));

    // The per-weight params really diverge from the old global fallback.
    assert_ne!(qkv[0].to_ffi("qkv_proj").unwrap(), (64, 4));
    assert_ne!(gup[0].to_ffi("gate_up_proj").unwrap(), (64, 4));
}

/// R3 (hardware gate): token-exact parity of the forward against the reference
/// mlx_lm (greedy/argmax) on K2-Horizon-3.7B-4bit.
///
/// This is an instruct model: the raw prompt degenerates into repetition, so
/// the reference and this test both use the model's CHAT TEMPLATE. The test
/// embeds the pre-templated prompt ids from `k2_horizon_prompt_ids.txt`
/// DIRECTLY (it does not re-tokenize — the engine's tokenizer does not apply
/// the model's Jinja chat template), so engine and reference share the exact
/// same input and the gate isolates the forward pass.
///
/// The gate prompt is a NON-degenerate factual QA ("What color is a ripe
/// banana? Answer in one word.") whose greedy continuation terminates
/// with EOS within 34 tokens and contains no repeated ~6-token subsequence, so the
/// assertion is an EXACT match over all N_TOKENS — any divergence is a real
/// bug, not float noise on a degenerate region (the old list-style prompt
/// degenerated at ~134 tokens and made exact match impossible).
///
/// Run on Apple Silicon with mlx-c and the model:
/// ```bash
/// ENGINE_MLX_MODEL=$HOME/nexum/models/mlx-community/K2-Horizon-3.7B-4bit \
///   cargo test -p engine-mlx-serve --features mlx k2_horizon_token_parity -- --ignored --nocapture
/// ```
///
/// Regenerate both fixtures with `gen_reference_fixture.py` (reference =
/// mlx_lm, the external oracle — never the engine) when the model, prompt, or
/// template changes. Keep N_TOKENS in sync with the script.
#[cfg(feature = "mlx")]
#[test]
#[ignore = "needs GPU + K2-Horizon-3.7B-4bit via ENGINE_MLX_MODEL + reference fixtures"]
fn k2_horizon_token_parity() {
    use std::path::Path;

    /// Parse a fixture: skip `#` comment lines, read one comma-separated u32 line.
    fn read_ids(path: &Path) -> Vec<u32> {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
        let line = text
            .lines()
            .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .unwrap_or_else(|| panic!("no data line in {}", path.display()));
        line.trim()
            .split(',')
            .map(|s| s.trim().parse().expect("token id"))
            .collect()
    }

    let model_dir = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon-3.7B-4bit");
    const N_TOKENS: usize = 34;

    let prompt_ids = read_ids(Path::new("tests/fixtures/k2_horizon_prompt_ids.txt"));
    let expected = read_ids(Path::new("tests/fixtures/k2_horizon_token_parity.txt"));
    assert_eq!(
        expected.len(),
        N_TOKENS,
        "parity fixture must hold {N_TOKENS} token ids"
    );

    let mut eng = engine_mlx_serve::Qwen3Engine::load(Path::new(&model_dir)).expect("load failed");
    // Embed the templated prompt ids directly (no re-tokenization).
    let x = eng.embed(&prompt_ids).expect("embed failed");
    let logits = eng.prefill(x, prompt_ids.len()).expect("prefill failed");
    let first = eng.ctx.argmax(logits).expect("argmax failed");
    let first_vals = eng.ctx.to_vec_u32(first).expect("to_vec_u32 failed");
    let mut generated = vec![first_vals[0]];
    eng.offset = prompt_ids.len();
    let mut current = first_vals[0];
    for i in 1..N_TOKENS {
        let next = eng.decode_step(current).expect("decode_step failed");
        generated.push(next);
        current = next;
    }
    // Exact match vs the mlx_lm reference: the gate prompt is non-degenerate
    // (EOS-terminated factual QA), so any divergence is a real bug.
    assert_eq!(
        generated, expected,
        "token-exact parity vs mlx_lm reference failed: first mismatch at index {}\ngenerated={:?}\nexpected ={:?}",
        generated
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(N_TOKENS),
        generated,
        expected
    );
    println!("PARITY: all {N_TOKENS} tokens match the mlx_lm reference exactly");
}
