//! Unit tests for `engine_mlx_ffi::compile` (graph capture).
//!
//! Without `mlx` feature, compile/compile_shapeless bail with "not bound yet".

use engine_mlx_ffi::compile::{compile, compile_shapeless, Closure};

#[test]
#[cfg(not(feature = "mlx"))]
fn compile_bails_without_mlx() {
    let c = Closure::new(unsafe { std::mem::zeroed() });
    match compile(c) {
        Ok(_) => panic!("expected error"),
        Err(e) => assert!(e.to_string().contains("not bound")),
    }
}

#[test]
#[cfg(not(feature = "mlx"))]
fn compile_shapeless_bails_without_mlx() {
    let c = Closure::new(unsafe { std::mem::zeroed() });
    match compile_shapeless(c) {
        Ok(_) => panic!("expected error"),
        Err(e) => assert!(e.to_string().contains("not bound")),
    }
}
