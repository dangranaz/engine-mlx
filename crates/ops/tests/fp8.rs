//! Tests for `engine_mlx_ops::fp8` — extracted from inline `#[cfg(test)]`.

use engine_mlx_ops::fp8::{e4m3fn_dequant, fp8_round_trip_slice};

#[test]
fn e4m3fn_roundtrip_zero() {
    let out = e4m3fn_dequant(0.0);
    assert_eq!(out, 0.0);
}

#[test]
fn e4m3fn_roundtrip_one() {
    let out = e4m3fn_dequant(1.0);
    assert!((out - 1.0).abs() < 0.01);
}

#[test]
fn e4m3fn_roundtrip_negative() {
    let out = e4m3fn_dequant(-2.5);
    assert!((out - (-2.5)).abs() < 0.1);
}

#[test]
fn e4m3fn_roundtrip_small() {
    let out = e4m3fn_dequant(0.001);
    assert!(out >= 0.0);
    assert!(out <= 0.01);
}

#[test]
fn fp8_round_trip_slice_basic() {
    let mut data = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 0.25];
    fp8_round_trip_slice(&mut data, 8, 0);
    for (i, &expected) in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 0.25].iter().enumerate() {
        assert!(
            (data[i] - expected).abs() < 0.1,
            "data[{}]={} != {}",
            i, data[i], expected
        );
    }
}

#[test]
fn fp8_round_trip_slice_with_rope() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 999.0, 888.0];
    fp8_round_trip_slice(&mut data, 8, 2);
    assert_eq!(data[6], 999.0);
    assert_eq!(data[7], 888.0);
}

#[test]
fn fp8_round_trip_multiple_rows() {
    let mut data = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
    fp8_round_trip_slice(&mut data, 4, 0);
    assert!((data[0] - 1.0).abs() < 0.1);
    assert!((data[1] - (-1.0)).abs() < 0.1);
    assert!((data[4] - 0.5).abs() < 0.1);
}

#[test]
fn fp8_roundtrip_preserves_sign() {
    let vals = [-100.0, -10.0, -1.0, -0.1, 0.1, 1.0, 10.0, 100.0];
    for &v in &vals {
        let out = e4m3fn_dequant(v);
        assert_eq!(out < 0.0, v < 0.0, "sign mismatch for {}", v);
    }
}
