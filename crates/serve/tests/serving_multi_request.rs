//! R1 (task 18): multi-request coherence gate on the real model.
//!
//! Two sequential `generate_chat` requests on the SAME engine must both be
//! coherent; before the per-request state reset, the 2nd degenerated to
//! newlines only (stale `offset`/`kv_bufs` corrupted the prefill).

use engine_mlx_serve::engine::Qwen3Engine;

/// The math prompt from the K2-Horizon benchmark: the coherent answer
/// contains "18" ("...The answer is 18").
fn math_prompt() -> Vec<(String, String)> {
    vec![(
        "user".into(),
        "A store has 3 crates. Each crate holds 4 boxes. Each box holds 6 apples. \
         The store sells 54 apples. How many apples are left?"
            .into(),
    )]
}

fn assert_coherent(answer: &str, which: &str) {
    assert!(
        !answer.trim().is_empty(),
        "{which} request produced whitespace/newline-only output (state leaked)"
    );
    assert!(
        answer.contains("18"),
        "{which} request did not produce the coherent answer, got: {answer:?}"
    );
}

#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer + --features mlx"]
fn serving_second_request_is_coherent() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let mut engine = Qwen3Engine::load(std::path::Path::new(&model_dir_str))
        .expect("load engine from real model");

    let (first, _) = engine
        .generate_chat(&math_prompt(), 128)
        .expect("first generate_chat");
    assert_coherent(&first, "first");

    let (second, _) = engine
        .generate_chat(&math_prompt(), 128)
        .expect("second generate_chat");
    assert_coherent(&second, "second");
}

/// The short prompt: expects the answer to echo "ping".
fn short_prompt() -> Vec<(String, String)> {
    vec![("user".into(), "Say exactly: ping".into())]
}

/// F4 (task 20): 3+ sequential requests with mixed prompt lengths on the SAME
/// engine. The resource-exhaustion / KV-capacity-recompute failure mode noted
/// in `generate_via_mlx` manifested "from ~the 3rd request"; this gate covers
/// long -> short -> long so a capacity or offset rebuild error past the 2nd
/// request surfaces here.
#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer + --features mlx"]
fn serving_many_requests_stay_coherent() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let mut engine = Qwen3Engine::load(std::path::Path::new(&model_dir_str))
        .expect("load engine from real model");

    let cases: &[(&str, Vec<(String, String)>, &str)] = &[
        ("first long", math_prompt(), "18"),
        ("short", short_prompt(), "ping"),
        ("second long", math_prompt(), "18"),
    ];
    for (which, messages, expected) in cases {
        let (answer, _) = engine
            .generate_chat(messages, 128)
            .expect("generate_chat");
        assert!(
            !answer.trim().is_empty(),
            "{which} produced whitespace/newline-only output (state leaked)"
        );
        assert!(
            answer.contains(expected),
            "{which} did not contain {expected:?}, got: {answer:?}"
        );
    }
}
