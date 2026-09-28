#![cfg(not(feature = "mlx"))] // stub-mode tests: assert the "needs mlx-c FFI" marker; skip under --features mlx
//! Unit tests for `engine_mlx_attention::gqa` module.

use engine_mlx_attention::gqa::{GqaConfig, GqaWeights, gqa_attention};
use engine_mlx_kvcache::KvCache;
use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;

fn null_ctx() -> MlxCtx {
    engine_mlx_ops::MlxCtx::new(unsafe { std::mem::zeroed() })
}

fn dummy_array() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn dummy_weights() -> engine_mlx_ops::quant::QuantWeights {
    engine_mlx_ops::quant::QuantWeights {
        weight: dummy_array(),
        scales: dummy_array(),
        biases: dummy_array(),
        group_size: 64,
        bits: 4,
        mode: "affine",
    }
}

struct DummyCache;
impl KvCache for DummyCache {
    fn append(&mut self, _ctx: &MlxCtx, _k: mlx_array, _v: mlx_array) -> anyhow::Result<()> {
        Ok(())
    }
    fn get(&self, _ctx: &MlxCtx) -> anyhow::Result<(mlx_array, mlx_array)> {
        Ok((dummy_array(), dummy_array()))
    }
    fn len(&self) -> usize { 0 }
    fn reset(&mut self) {}
    fn memory_bytes(&self) -> usize { 0 }
}

#[test]
fn gqa_config_is_constructible() {
    let cfg = GqaConfig {
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
        use_rope: true,
    };
    assert_eq!(cfg.n_heads, 32);
    assert_eq!(cfg.n_kv_heads, 8);
}

#[test]
fn gqa_attention_bails_with_mlx_marker() {
    let cfg = GqaConfig {
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
        use_rope: true,
    };
    let weights = GqaWeights {
        q_proj: dummy_weights(),
        k_proj: dummy_weights(),
        v_proj: dummy_weights(),
        o_proj: dummy_weights(),
        q_norm: None,
        k_norm: None,
    };
    let mut cache = DummyCache;
    let err = gqa_attention(&null_ctx(), dummy_array(), &weights, &mut cache, &cfg, 0)
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
