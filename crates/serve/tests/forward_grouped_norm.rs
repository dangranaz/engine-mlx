//! Forward grouped RMSNorm parity tests.
//!
//! These tests exercise the grouped_rms_norm wiring as used in the Qwen3
//! forward path, with non-constant inputs that discriminate between correct
//! group boundaries and incorrect group-swap/RMS-global bugs.

#[cfg(feature = "mlx")]
use engine_mlx_ffi::{MlxCtx, mlx_dtype};

/// Reference: GroupedRMSNorm with groups=2 on a 1-D input of length `dims`
/// (dims divisible by 2). Normalises each half separately in f32, no weight
/// during normalisation, then multiplies the full-width weight afterwards.
fn reference_groups2(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dims = x.len();
    let half = dims / 2;
    let mut out = vec![0.0f32; dims];
    for g in 0..2 {
        let start = g * half;
        let mut mean_sq = 0.0f32;
        for i in start..start + half {
            mean_sq += x[i] * x[i];
        }
        mean_sq /= half as f32;
        let rms = (mean_sq + eps).sqrt();
        for i in start..start + half {
            out[i] = weight[i] * (x[i] / rms);
        }
    }
    out
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

#[cfg(feature = "mlx")]
#[test]
fn forward_grouped_norm_groups2_matches_reference() {
    // Non-constant input with distinct values per group to catch group-boundary
    // errors (group-swap, global RMS, wrong weight application).
    // Group 0: [1.0, 2.0], Group 1: [5.0, 8.0]
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[1.0, 2.0, 5.0, 8.0]).unwrap();
    // Weight in f32 first, then cast to bf16 (lossy) to test the real path
    let w_f32 = ctx.new_array(&[1.1, 2.7, 0.5, 3.3]).unwrap();
    let w_bf16 = ctx.astype(w_f32, mlx_dtype::MLX_BFLOAT16).unwrap();
    let eps = 1e-5f32;

    let out = ctx.grouped_rms_norm(x, w_bf16, eps, 2).unwrap();
    let got = ctx.to_vec_f32(out).unwrap();

    let expected = reference_groups2(&[1.0, 2.0, 5.0, 8.0], &[1.1, 2.7, 0.5, 3.3], eps);
    // bf16 weight has ~8 bits mantissa, tolerance ~2e-2 relative
    assert!(
        approx_eq(&got, &expected, 2e-2),
        "groups=2 mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
}

#[cfg(feature = "mlx")]
#[test]
fn forward_groups1_unchanged() {
    // Non-regression: groups=1 must delegate to fused rms_norm exactly.
    let ctx = MlxCtx::cpu();
    // Non-constant input to also test the groups=1 path thoroughly
    let x = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let w = ctx.new_array(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    let eps = 1e-5f32;

    let grouped = ctx.grouped_rms_norm(x, w.clone(), eps, 1).unwrap();
    let plain = ctx.rms_norm(x, w, eps).unwrap();

    let g = ctx.to_vec_f32(grouped).unwrap();
    let p = ctx.to_vec_f32(plain).unwrap();
    assert!(approx_eq(&g, &p, 1e-5), "grouped(1) != plain rms_norm: {:?} vs {:?}", g, p);
}