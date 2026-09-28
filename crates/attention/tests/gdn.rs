#![cfg(not(feature = "mlx"))] // stub-mode tests: assert the "needs mlx-c FFI" marker; skip under --features mlx
//! Unit tests for `engine_mlx_attention::gdn` module (Gated Delta Network).

use engine_mlx_attention::{gdn, GdnConfig, GdnState};
use engine_mlx_ops::ffi::mlx_array;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() })
}

fn config() -> GdnConfig {
    GdnConfig {
        num_key_heads: 4,
        num_value_heads: 8,
        key_head_dim: 16,
        value_head_dim: 16,
        conv_kernel_size: 4,
        rms_norm_eps: 1e-6,
        use_metal_kernel: false,
    }
}

#[test]
fn gdn_config_fields_accessible() {
    let cfg = config();
    assert_eq!(cfg.num_key_heads, 4);
    assert_eq!(cfg.num_value_heads, 8);
}

#[test]
fn gdn_compute_gate_bails_with_mlx_marker() {
    // compute_gate hits ffi::mlx_astype directly which panics when mlx feature disabled
    let result = std::panic::catch_unwind(|| {
        gdn::compute_gate(
            &null_ctx(),
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        )
    });
    match result {
        Ok(Ok(_)) => panic!("expected error or panic"),
        Ok(Err(e)) => assert!(e.to_string().contains("needs mlx-c FFI")),
        Err(panic) => {
            let msg = panic.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(msg.contains("MLX feature not enabled") || msg.contains("needs mlx-c FFI"), "unexpected panic: {}", msg);
        }
    }
}

#[test]
fn gdn_step_bails_with_mlx_marker() {
    let err = gdn::gdn_step(
        &null_ctx(),
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        4,
        8,
        16,
        16,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn gdn_forward_bails_with_mlx_marker() {
    let mut state = GdnState { h: None, conv_buf: None };
    let dummy_w = engine_mlx_ops::quant::QuantWeights {
        weight: unsafe { std::mem::zeroed() },
        scales: unsafe { std::mem::zeroed() },
        biases: unsafe { std::mem::zeroed() },
        bits: 4,
        group_size: 64,
        mode: "affine",
    };
    let err = gdn::gdn_forward(
        &null_ctx(),
        unsafe { std::mem::zeroed() },
        &config(),
        &mut state,
        &dummy_w,
        &dummy_w,
        &dummy_w,
        &dummy_w,
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        unsafe { std::mem::zeroed() },
        &dummy_w,
        1,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
