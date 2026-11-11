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

/// Reference: GroupedRMSNorm with arbitrary `groups` on a 1-D input.
fn reference_groups(x: &[f32], weight: &[f32], eps: f32, groups: i32) -> Vec<f32> {
    let dims = x.len();
    let dims_per_group = dims / groups as usize;
    let mut out = vec![0.0f32; dims];
    for g in 0..groups as usize {
        let start = g * dims_per_group;
        let mut mean_sq = 0.0f32;
        for i in start..start + dims_per_group {
            mean_sq += x[i] * x[i];
        }
        mean_sq /= dims_per_group as f32;
        let rms = (mean_sq + eps).sqrt();
        for i in start..start + dims_per_group {
            out[i] = weight[i] * (x[i] / rms);
        }
    }
    out
}

/// Reference: GroupedRMSNorm on multi-dimensional input [b, s, dims].
fn reference_multidim(x: &[f32], weight: &[f32], eps: f32, groups: i32, shape: &[usize]) -> Vec<f32> {
    let dims = shape[shape.len() - 1];
    let dims_per_group = dims / groups as usize;
    let mut out = vec![0.0f32; x.len()];
    // For each batch/seq element, apply grouped norm on the last dimension
    let outer_size: usize = shape.iter().take(shape.len() - 1).product();
    for outer_idx in 0..outer_size {
        let base = outer_idx * dims;
        for g in 0..groups as usize {
            let start = base + g * dims_per_group;
            let mut mean_sq = 0.0f32;
            for i in start..start + dims_per_group {
                mean_sq += x[i] * x[i];
            }
            mean_sq /= dims_per_group as f32;
            let rms = (mean_sq + eps).sqrt();
            for i in start..start + dims_per_group {
                out[i] = weight[i - base] * (x[i] / rms);
            }
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
fn grouped_rms_norm_groups1_matches_plain() {
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let w = ctx.new_array(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    let eps = 1e-5f32;

    let grouped = ctx.grouped_rms_norm(x, w.clone(), eps, 1).unwrap();
    let plain = ctx.rms_norm(x, w, eps).unwrap();

    let g = ctx.to_vec_f32(grouped).unwrap();
    let p = ctx.to_vec_f32(plain).unwrap();
    assert!(approx_eq(&g, &p, 1e-5), "grouped(1) != plain rms_norm: {:?} vs {:?}", g, p);
}

#[cfg(feature = "mlx")]
#[test]
fn grouped_rms_norm_groups2_matches_reference() {
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[4.0, 4.0, 4.0, 4.0]).unwrap();
    let w = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let eps = 1e-5f32;

    let out = ctx.grouped_rms_norm(x, w, eps, 2).unwrap();
    let got = ctx.to_vec_f32(out).unwrap();

    let expected = reference_groups2(&[4.0, 4.0, 4.0, 4.0], &[1.0, 2.0, 3.0, 4.0], eps);
    assert!(
        approx_eq(&got, &expected, 1e-5),
        "groups=2 mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
}

#[cfg(feature = "mlx")]
#[test]
fn grouped_rms_norm_rejects_indivisible() {
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap(); // dims=4
    let w = ctx.new_array(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    let eps = 1e-5f32;

    let result = ctx.grouped_rms_norm(x, w, eps, 3); // 4 % 3 != 0
    assert!(result.is_err(), "expected Err for dims=4, groups=3");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("divisible") || err.to_string().contains("group"),
        "unexpected error message: {err}"
    );
}

#[cfg(feature = "mlx")]
#[test]
fn grouped_rms_norm_groups4() {
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]).unwrap(); // dims=8
    let w = ctx.new_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let eps = 1e-5f32;

    let out = ctx.grouped_rms_norm(x, w, eps, 4).unwrap();
    let got = ctx.to_vec_f32(out).unwrap();

    let expected = reference_groups(&[4.0; 8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], eps, 4);
    assert!(
        approx_eq(&got, &expected, 1e-5),
        "groups=4 mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
}

#[cfg(feature = "mlx")]
#[test]
fn grouped_rms_norm_multidim() {
    let ctx = MlxCtx::cpu();
    // shape [2, 3, 4] - batch=2, seq=3, dims=4
    let x_data = vec![4.0f32; 24]; // 2*3*4 = 24
    let w_data = vec![1.0, 2.0, 3.0, 4.0];
    let x = ctx.new_array(&x_data).unwrap();
    let w = ctx.new_array(&w_data).unwrap();
    let eps = 1e-5f32;

    // Reshape x to [2, 3, 4]
    let x_3d = ctx.reshape(x, &[2, 3, 4]).unwrap();

    let out = ctx.grouped_rms_norm(x_3d, w, eps, 2).unwrap();
    let got = ctx.to_vec_f32(out).unwrap();

    // Reference: for each [2, 3] element, apply groups=2 on last dim=4
    let expected = reference_multidim(&x_data, &w_data, eps, 2, &[2, 3, 4]);
    assert!(
        approx_eq(&got, &expected, 1e-5),
        "multidim mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
}

#[cfg(feature = "mlx")]
#[test]
fn grouped_rms_norm_bf16_weight_matches_reference() {
    let ctx = MlxCtx::cpu();
    let x = ctx.new_array(&[4.0, 4.0, 4.0, 4.0]).unwrap();
    // Create weight in f32 first, then cast to bf16
    let w_f32 = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let w_bf16 = ctx.astype(w_f32, mlx_dtype::MLX_BFLOAT16).unwrap();
    let eps = 1e-5f32;

    let out = ctx.grouped_rms_norm(x, w_bf16, eps, 2).unwrap();
    let got = ctx.to_vec_f32(out).unwrap();

    let expected = reference_groups2(&[4.0, 4.0, 4.0, 4.0], &[1.0, 2.0, 3.0, 4.0], eps);
    // bf16 has ~8 bits mantissa, so tolerance needs to be larger (~2e-2 relative)
    assert!(
        approx_eq(&got, &expected, 2e-2),
        "bf16 weight mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
}