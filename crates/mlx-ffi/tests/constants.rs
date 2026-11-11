//! Unit tests for the MLX-C dtype constants exposed by `engine_mlx_ffi`.
//!
//! These constants are always available (independent of the `mlx` feature)
//! and define the canonical dtype indexing used across the engine.

use engine_mlx_ffi::{
    MLX_INT4, MLX_INT8, MLX_BOOL, MLX_BFLOAT16, MLX_COMPLEX32, MLX_COMPLEX64, MLX_FLOAT16,
    MLX_FLOAT32, MLX_INT16, MLX_INT32, MLX_UINT4, MLX_UINT8, MLX_UINT16, MLX_UINT32,
};

#[test]
fn dtype_constants_match_canonical_index() {
    assert_eq!(MLX_FLOAT32, 0);
    assert_eq!(MLX_FLOAT16, 1);
    assert_eq!(MLX_BFLOAT16, 2);
    assert_eq!(MLX_INT32, 3);
    assert_eq!(MLX_INT16, 4);
    assert_eq!(MLX_INT8, 5);
    assert_eq!(MLX_UINT32, 6);
    assert_eq!(true, MLX_UINT16 == 7);
    assert_eq!(MLX_UINT8, 8);
    assert_eq!(MLX_BOOL, 9);
    assert_eq!(MLX_UINT4, 10);
    assert_eq!(MLX_INT4, 11);
    assert_eq!(MLX_COMPLEX64, 12);
    assert_eq!(MLX_COMPLEX32, 13);
}
