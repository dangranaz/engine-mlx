//! Unit tests for the pure discriminator functions used by the 5 standard
//! bench tests. These encode the "tolerant but meaningful" requirement (R2):
//! discriminators must search the answer anywhere (preamble-tolerant for
//! thinking models) but must still FAIL on degenerate output (empty /
//! whitespace / wrong answer) and PASS on short direct answers (R4).

use engine_bench::discriminators::{coherence_match, factual_match, instruction_match};

#[test]
fn factual_match_rejects_degenerate() {
    assert!(!factual_match(""));
    assert!(!factual_match("   \n\n   "));
    assert!(!factual_match("The capital of France is London."));
    assert!(!factual_match("I don't know."));
}

#[test]
fn factual_match_accepts_direct_and_preamble() {
    assert!(factual_match("Paris"));
    assert!(factual_match("The capital of France is Paris."));
    assert!(factual_match("paris"));
    // Preamble-tolerant: reasoning before the answer still matches.
    assert!(factual_match("\n\nLet me think...\nThe capital of France is Paris."));
}

#[test]
fn instruction_match_rejects_degenerate() {
    assert!(!instruction_match(""));
    assert!(!instruction_match("   \n\n   "));
    assert!(!instruction_match("The task is complete."));
    assert!(!instruction_match("OK"));
}

#[test]
fn instruction_match_accepts_direct_and_preamble() {
    assert!(instruction_match("DONE"));
    assert!(instruction_match("Done"));
    assert!(instruction_match("done"));
    // Preamble-tolerant: thinking before DONE still matches.
    assert!(instruction_match("\n\n<ifm|think>reasoning...</ifm|think>DONE"));
    // No length constraint: a long-ish answer with DONE still passes.
    assert!(instruction_match("Some reasoning text here. DONE."));
}

#[test]
fn coherence_match_rejects_degenerate() {
    assert!(!coherence_match(""));
    assert!(!coherence_match("   \n\n   "));
    assert!(!coherence_match("The answer is 42."));
    assert!(!coherence_match("I don't know."));
    // A bare number without working is not enough on its own unless it's via
    // the explicit ANSWER line; "42" alone fails both arms.
    assert!(!coherence_match("42"));
}

#[test]
fn coherence_match_accepts_direct_and_preamble() {
    assert!(coherence_match("ANSWER: 18"));
    assert!(coherence_match("answer = 18"));
    // Reasoning preamble + answer line: passes.
    assert!(coherence_match(
        "\n\n3 crates × 4 boxes = 12 boxes. 12 × 6 apples = 72 apples. \
         72 − 54 = 18.\nANSWER: 18"
    ));
    // Loose path: standalone 18 + shows_work (72 or 12).
    assert!(coherence_match("12 boxes, 72 apples, 18 left"));
    assert!(coherence_match("72 apples minus 54 equals 18"));
}

// ── Task 39: R4 — coherence_match loose arm must not false-positive ──
#[test]
fn coherence_match_rejects_false_positive_loose() {
    // Mentions 12 (shows_work) and 18 (loose) but has no real ANSWER: line
    // and no coherent chain — the "18" is just a side remark ("is prime").
    // This must NOT pass: the loose arm alone is not sufficient.
    assert!(!coherence_match("…leaves 12… no wait, 18 is prime…"));
    // A standalone 18 with no working at all must fail.
    assert!(!coherence_match("The answer is 18."));
    // But a real ANSWER: line still passes.
    assert!(coherence_match("ANSWER: 18"));
    // Reasoning + ANSWER: line still passes (preamble-tolerant).
    assert!(coherence_match(
        "\n\n3 crates × 4 boxes = 12 boxes. 12 × 6 apples = 72 apples. \
         72 − 54 = 18.\nANSWER: 18"
    ));
}
