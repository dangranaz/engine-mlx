//! Task 23 (E4): red-capable loop for the sustained-load degradation.
//!
//! On a fresh engine, >=6 sequential `generate_chat` requests of 256 tokens on
//! the SAME engine must ALL stay coherent. Today requests 1-5 are coherent and
//! request 6+ degenerates to whitespace-only output, irreversibly, inside the
//! same process. This test is the loop: RED today, GREEN once the root cause
//! (per-request compile-cache accumulation) is fixed.
//!
//! Run:
//!   ENGINE_MLX_MODEL=$HOME/nexum/models/mlx-community/K2-Horizon-3.7B-4bit \
//!     cargo test -p engine-mlx-serve --release --features mlx \
//!       --test load_degradation_loop -- --ignored --nocapture

use engine_mlx_serve::engine::Qwen3Engine;

fn ocean_prompt() -> Vec<(String, String)> {
    vec![(
        "user".into(),
        "Write a short paragraph about the ocean.".into(),
    )]
}

/// Coherent = real text, not whitespace/newline-only garbage.
fn assert_coherent(answer: &str, which: &str) {
    assert!(
        answer.trim().chars().any(|c| c.is_alphabetic()),
        "{which} request degenerated to whitespace-only output: {answer:?}"
    );
    assert!(
        answer.trim().len() > 100,
        "{which} request too short to be a coherent paragraph ({} chars): {answer:?}",
        answer.trim().len()
    );
}

#[cfg(feature = "mlx")]
fn active_memory() -> usize {
    let mut v: usize = 0;
    unsafe {
        engine_mlx_ffi::mlx_get_active_memory(&mut v);
    }
    v
}

#[cfg(not(feature = "mlx"))]
fn active_memory() -> usize {
    0
}

#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer + --features mlx"]
fn six_long_requests_stay_coherent() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let mut engine = Qwen3Engine::load(std::path::Path::new(&model_dir_str))
        .expect("load engine from real model");

    const N: usize = 6;
    let mut last = String::new();
    for i in 1..=N {
        let mem_before = active_memory();
        let (answer, _) = engine
            .generate_chat(&ocean_prompt(), 256)
            .expect("generate_chat");
        let mem_after = active_memory();
        println!(
            "[LOOP] req {i}/{N} chars={} active_mem {mem_before} -> {mem_after} answer={:?}",
            answer.chars().count(),
            &answer[..answer.len().min(80)]
        );
        assert_coherent(&answer, &format!("req {i}"));
        last = answer;
    }
    assert_coherent(&last, "last");
}
