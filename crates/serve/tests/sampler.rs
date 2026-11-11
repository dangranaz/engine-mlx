//! Unit tests for CPU sampler logic — greedy, temperature, top-k.
//!
//! These test the pure-CPU sampling algorithm (no GPU required).

use engine_mlx_serve::sampler::{sample_token, SamplingParams, SimpleRng};

#[test]
fn greedy_picks_max() {
    let logits = vec![1.0, 5.0, 2.0, 3.0];
    let mut rng = SimpleRng::new(42);
    let tok = sample_token(&logits, &SamplingParams::greedy(), &mut rng).unwrap();
    assert_eq!(tok, 1);
}

#[test]
fn temperature_diversifies() {
    let logits = vec![1.0, 5.0, 2.0, 3.0, 4.0];
    let mut rng = SimpleRng::new(1);
    let params = SamplingParams::default()
        .with_temperature(1.0)
        .with_top_p(1.0)
        .with_top_k(0);
    let tok = sample_token(&logits, &params, &mut rng).unwrap();
    assert!(tok < 5);
}

#[test]
fn top_k_filters() {
    let logits = vec![10.0, 9.0, 1.0, 1.0, 1.0];
    let mut rng = SimpleRng::new(0);
    let params = SamplingParams::default()
        .with_temperature(1.0)
        .with_top_k(2);
    // With top_k=2, only first two should ever be sampled (high prob)
    for _ in 0..20 {
        let tok = sample_token(&logits, &params, &mut rng).unwrap();
        assert!(tok == 0 || tok == 1, "top_k=2 should only sample 0 or 1, got {tok}");
    }
}
