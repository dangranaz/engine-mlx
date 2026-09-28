//! Unit tests for `engine_mlx_attention` root types and re-exports.

use engine_mlx_attention::{GdnConfig, QuantWeights};
use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::kv_traits::{KvCacheBackend, TensorRef};

#[test]
fn quant_weights_is_constructible() {
    let w = QuantWeights {
        weight: unsafe { std::mem::zeroed() },
        scales: unsafe { std::mem::zeroed() },
        biases: unsafe { std::mem::zeroed() },
        group_size: 128,
        bits: 2,
        mode: "affine",
    };
    assert_eq!(w.group_size, 128);
    assert_eq!(w.bits, 2);
    assert_eq!(w.mode, "affine");
}

#[test]
fn gdn_config_fields_accessible() {
    let cfg = GdnConfig {
        num_key_heads: 4,
        num_value_heads: 8,
        key_head_dim: 64,
        value_head_dim: 64,
        conv_kernel_size: 4,
        rms_norm_eps: 1e-6,
        use_metal_kernel: false,
    };
    assert_eq!(cfg.num_key_heads, 4);
    assert_eq!(cfg.num_value_heads, 8);
    assert_eq!(cfg.key_head_dim, 64);
}

#[test]
fn kv_cache_backend_trait_object_works() {
    struct Dummy;
    impl KvCacheBackend for Dummy {
        fn append(&mut self, _k: TensorRef, _v: TensorRef) -> anyhow::Result<()> {
            Ok(())
        }
        fn decode_to_f32(&self) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            Ok((vec![], vec![]))
        }
        fn len(&self) -> usize {
            0
        }
    }

    let mut d = Dummy;
    d.append(TensorRef::F32(vec![1.0]), TensorRef::F32(vec![1.0]))
        .unwrap();
    assert_eq!(d.len(), 0);
    assert!(d.is_empty());
}
