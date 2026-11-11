//! Unit tests for `engine_mlx_ops::kv_traits` module.

use engine_mlx_ops::{KvCacheBackend, TensorRef};

struct DummyBackend {
    data: Vec<f32>,
    capacity: usize,
}

impl KvCacheBackend for DummyBackend {
    fn append(&mut self, _k: TensorRef, _v: TensorRef) -> anyhow::Result<()> {
        self.data.push(1.0);
        Ok(())
    }

    fn decode_to_f32(&self) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        Ok((self.data.clone(), self.data.clone()))
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[test]
fn kv_cache_backend_trait_is_satisfiable() {
    let mut backend = DummyBackend {
        data: Vec::new(),
        capacity: 1024,
    };
    assert_eq!(backend.capacity, 1024);
    assert_eq!(backend.len(), 0);
    backend.append(TensorRef::F32(vec![1.0, 2.0]), TensorRef::F32(vec![3.0, 4.0])).unwrap();
    assert_eq!(backend.len(), 1);
    let (k, v) = backend.decode_to_f32().unwrap();
    assert_eq!(k.len(), 1);
    assert_eq!(v.len(), 1);
}

#[test]
fn tensor_ref_enum_is_satisfiable() {
    let t = TensorRef::F32(vec![1.0, 2.0, 3.0]);
    assert!(matches!(t, TensorRef::F32(_)));
    if let TensorRef::F32(v) = t {
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }
}
