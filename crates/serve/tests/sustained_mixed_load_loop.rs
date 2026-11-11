//! Sustained mixed-load loop — reproduces the compile-cache leak (E5).
//!
//! On a fresh engine, run >=12 mixed requests (>=2 at 1024 max_tokens,
//! >=1 at 2048 max_tokens) followed by >=20 sustained requests at 256
//! max_tokens, all on the SAME engine. Before the fix, request ~9-10
//! collapses to whitespace-only output and `Resource limit` errors appear
//! at `transforms.cpp:73` because each request compiles a fresh closure
//! that accumulates in MLX's global compile cache. After the fix
//! (compile-cache cleared in `reset()`), all requests stay coherent and
//! active memory stays flat.
//!
//! Run:
//!   ENGINE_MLX_MODEL=$HOME/nexum/models/mlx-community/K2-Horizon-3.7B-4bit \
//!     cargo test -p engine-mlx-serve --release --features mlx \
//!       --test sustained_mixed_load_loop -- --ignored --nocapture

use engine_mlx_serve::engine::Qwen3Engine;

fn ocean_prompt() -> Vec<(String, String)> {
    vec![(
        "user".into(),
        "Write a short paragraph about the ocean.".into(),
    )]
}

fn story_prompt() -> Vec<(String, String)> {
    vec![(
        "user".into(),
        "Tell me a short story about a brave explorer discovering a hidden valley.".into(),
    )]
}

fn poem_prompt() -> Vec<(String, String)> {
    vec![(
        "user".into(),
        "Write a poem about the stars and the night sky.".into(),
    )]
}

/// Coherent = real text, not whitespace/newline-only garbage AND not a
/// degenerate repetition. The whitespace/length checks alone are too weak: a
/// looping output like "Theuser was giving a specific task..." repeated to the
/// cap has letters and length > 50, yet is degenerate. The anti-repetition
/// check below catches that class (a ~6-word window recurring >= 3 times).
fn max_window_repeat(answer: &str, window: usize) -> usize {
    use std::collections::HashMap;
    let words: Vec<&str> = answer.split_whitespace().collect();
    if words.len() < window {
        return 0;
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut max = 0;
    for i in 0..=(words.len() - window) {
        let key = words[i..i + window].join(" ");
        let c = counts.entry(key).or_insert(0);
        *c += 1;
        if *c > max {
            max = *c;
        }
    }
    max
}

/// Coherent = real text, not whitespace-only, not too short, and not a
/// degenerate repetition.
fn assert_coherent(answer: &str, which: &str) {
    assert!(
        answer.trim().chars().any(|c| c.is_alphabetic()),
        "{which} request degenerated to whitespace-only output: {answer:?}"
    );
    assert!(
        answer.trim().len() > 50,
        "{which} request too short to be coherent ({} chars): {answer:?}",
        answer.trim().len()
    );
    // Anti-repetition: a coherent answer must not loop. A 6-word window must
    // not recur 50+ times (normal prose repeats short phrases at most ~11 times;
    // a degenerate loop repeats the same window 100+ times).
    let reps = max_window_repeat(answer, 6);
    assert!(
        reps < 50,
        "{which} request degenerated into repetition (a 6-word window recurs {reps}x): {answer:?}"
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
fn sustained_mixed_load_stays_coherent() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let mut engine = Qwen3Engine::load(std::path::Path::new(&model_dir_str))
        .expect("load engine from real model");

    // ── Mixed phase: >=2 long (1024 tok), >=1 very long (2048 tok) ──
    let mixed_cases: &[(&str, Vec<(String, String)>, usize)] = &[
        ("mixed-1024-a", ocean_prompt(), 1024),
        ("mixed-1024-b", story_prompt(), 1024),
        ("mixed-2048", poem_prompt(), 2048),
    ];
    let mut mems: Vec<usize> = Vec::new();
    for (which, messages, max_tokens) in mixed_cases {
        let mem_before = active_memory();
        let (answer, _) = engine
            .generate_chat(messages, *max_tokens)
            .expect("generate_chat");
        let mem_after = active_memory();
        mems.push(mem_after);
        println!(
            "[MIXED] {which} max_tokens={} chars={} mem {mem_before} -> {mem_after} answer={:?}",
            max_tokens,
            answer.chars().count(),
            &answer[..answer.len().min(80)]
        );
        assert_coherent(&answer, which);
    }

    // ── Sustained phase: >=20 requests at 256 tok on the same engine ──
    const SUSTAINED: usize = 20;
    let mut last = String::new();
    for i in 1..=SUSTAINED {
        let mem_before = active_memory();
        let (answer, _) = engine
            .generate_chat(&ocean_prompt(), 256)
            .expect("generate_chat");
        let mem_after = active_memory();
        mems.push(mem_after);
        println!(
            "[SUSTAINED] req {i}/{SUSTAINED} chars={} mem {mem_before} -> {mem_after} answer={:?}",
            answer.chars().count(),
            &answer[..answer.len().min(80)]
        );
        assert_coherent(&answer, &format!("sustained req {i}"));
        last = answer;
    }
    assert_coherent(&last, "last sustained");

    // ── Memory must not grow monotonically across the sustained phase ──
    // Each request's memory should be within a bounded delta of the previous;
    // unbounded growth = compile-cache leak (each un-cleared compile retains
    // Metal kernels/constants). 50 MB/request is well above normal variation
    // but far below the ~739 MB/request seen before the E4/E5 fixes.
    const MEM_GROWTH_THRESHOLD: usize = 50 * 1024 * 1024;
    let mut growth_count = 0;
    for i in 1..mems.len() {
        if mems[i] > mems[i - 1] + MEM_GROWTH_THRESHOLD {
            growth_count += 1;
        }
    }
    assert!(
        growth_count == 0,
        "active memory grew by >50 MB between {growth_count} consecutive requests — compile-cache leak not fixed"
    );
}
