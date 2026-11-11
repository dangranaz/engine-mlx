//! Diagnostic probe for the sustained-load Metal resource leak (E5).
//!
//! NOT a gate — a measurement harness. Prints active + peak MLX memory per
//! request so we can see whether the leak is bytes or buffer-COUNT, and whether
//! it accumulates across requests (cross-request) or within the decode loop.
//!
//! Run:
//!   ENGINE_MLX_MODEL=$HOME/nexum/models/mlx-community/K2-Horizon-3.7B-4bit \
//!     cargo test -p engine-mlx-serve --release --features mlx \
//!       --test leak_probe -- --ignored --nocapture

use engine_mlx_serve::engine::Qwen3Engine;

fn ocean() -> Vec<(String, String)> {
    vec![("user".into(), "Write a short paragraph about the ocean.".into())]
}

#[cfg(feature = "mlx")]
fn active() -> usize {
    let mut v: usize = 0;
    unsafe { engine_mlx_ffi::mlx_get_active_memory(&mut v); }
    v
}
#[cfg(feature = "mlx")]
fn peak() -> usize {
    let mut v: usize = 0;
    unsafe { engine_mlx_ffi::mlx_get_peak_memory(&mut v); }
    v
}
#[cfg(not(feature = "mlx"))]
fn active() -> usize { 0 }
#[cfg(not(feature = "mlx"))]
fn peak() -> usize { 0 }

#[test]
#[ignore = "requires ENGINE_MLX_MODEL + --features mlx"]
fn probe_sustained_leak() {
    let dir = std::env::var("ENGINE_MLX_MODEL").expect("ENGINE_MLX_MODEL");
    let mut engine = Qwen3Engine::load(std::path::Path::new(&dir)).expect("load");

    // Pure 256-tok sustained — same capacity every request, so compile-cache
    // and mask-count should be CONSTANT across requests if reset() frees them.
    // If active/peak climb monotonically, something survives reset().
    for i in 1..=30 {
        let a0 = active();
        let (ans, _) = engine.generate_chat(&ocean(), 256).expect("gen");
        let a1 = active();
        let coherent = ans.trim().chars().any(|c| c.is_alphabetic());
        println!(
            "[PROBE] req {i:2} active {:>5}MB -> {:>5}MB (d={:+5}MB) peak {:>5}MB coherent={} chars={}",
            a0 / 1048576,
            a1 / 1048576,
            (a1 as i64 - a0 as i64) / 1048576,
            peak() / 1048576,
            coherent,
            ans.chars().count(),
        );
    }
}
