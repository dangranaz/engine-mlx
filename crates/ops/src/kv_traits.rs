//! KV Cache traits — quantized backend abstraction.

use anyhow::Result;

/// Reference to host data for KV cache append.
#[derive(Debug)]
pub enum TensorRef {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

impl TensorRef {
    /// View as an F32 slice, if the variant holds F32 data.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorRef::F32(v) => Some(v),
            _ => None,
        }
    }
}

/// Backend for quantized KV cache.
///
/// Used by `attention::kv_cache_quant`.
pub trait KvCacheBackend: Send {
    fn append(&mut self, k: TensorRef, v: TensorRef) -> Result<()>;
    fn decode_to_f32(&self) -> Result<(Vec<f32>, Vec<f32>)>;
    fn len(&self) -> usize { 0 }
    fn is_empty(&self) -> bool { self.len() == 0 }
}
