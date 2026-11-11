//! Unit tests for the profile-threshold wiring (task 38) and hardening (task 39).
//!
//! Verifies that the `min_tps` from the *profile* (e.g. 5 for k2_horizon) is
//! the one actually applied to `passed`, overriding the binary default (20).
//! No double source of truth: the profile value is the single source.
//!
//! Task 39 hardening: empty steady-state sets must FAIL, never pass vacuously.

use engine_bench::config::{RunArgs, Thresholds};
use engine_bench::pass_check::{ramp_pass, sustained_pass};

#[test]
fn sustained_pass_uses_profile_floor_not_default() {
    // Real K2 measurements (reviser-verified): min 7.2 t/s, degr 12.7%.
    // Against the honest profile floor min_tps=5 -> PASS.
    assert!(sustained_pass(12.7, 7.2, 15.0, 5.0));
    // Against the binary default min_tps=20 -> FAIL (7.2 < 20).
    assert!(!sustained_pass(12.7, 7.2, 15.0, 20.0));
}

#[test]
fn ramp_pass_uses_profile_floor_not_default() {
    // Real K2 measurements: ramp drop 0.5%, min 8.0 t/s.
    // Against the honest profile floor min_tps=5 -> PASS.
    assert!(ramp_pass(0.5, 8.0, 60.0, 5.0));
    // Against the binary default min_tps=20 -> FAIL (8.0 < 20).
    assert!(!ramp_pass(0.5, 8.0, 60.0, 20.0));
}

#[test]
fn default_floor_20_still_applies_without_profile_override() {
    // R4 (task 38): a profile without override inherits the default 20, so a
    // healthy K2 measurement (7.2 t/s) correctly fails against it — proving
    // the profile value (5) is what flips the verdict, not a relaxed guard.
    assert!(!sustained_pass(12.7, 7.2, 15.0, 20.0));
    assert!(!ramp_pass(0.5, 8.0, 60.0, 20.0));
}

// ── Task 39: R1 — empty steady-state set must FAIL, never pass vacuously ──
#[test]
fn sustained_pass_rejects_empty_steady_state() {
    // No steady-state samples (sustained_reps <= 1): min_tps = INFINITY,
    // which would vacuously satisfy `min_tps >= floor`. Must be FAIL.
    assert!(!sustained_pass(0.0, f64::INFINITY, 15.0, 5.0));
}

#[test]
fn ramp_pass_rejects_empty_steady_state() {
    // No steady-state ramp points (ramp with 1 entry): min_tps = INFINITY.
    assert!(!ramp_pass(0.0, f64::INFINITY, 60.0, 5.0));
}

#[test]
fn sustained_pass_accepts_non_empty_steady_state() {
    // With at least one steady-state sample, normal evaluation applies.
    assert!(sustained_pass(10.0, 7.0, 15.0, 5.0));
}

#[test]
fn ramp_pass_accepts_non_empty_steady_state() {
    assert!(ramp_pass(5.0, 7.0, 60.0, 5.0));
}

// ── Task 39: R2 — profile flag is the single source of truth for min_tps ──
#[test]
fn profile_k2_horizon_overrides_default_min_tps() {
    // --profile k2_horizon must load min_tps=5 from the profile JSON,
    // overriding the binary default of 20.
    let cli = run_args_with_profile(Some("k2_horizon".into()));
    let t = Thresholds::resolve(&cli).expect("profile must load");
    assert_eq!(t.min_tps, 5.0);
}

#[test]
fn explicit_cli_min_tps_wins_over_profile() {
    // --min-tps 99 must win over the profile's 5.
    let cli = run_args_with_profile(Some("k2_horizon".into()));
    let cli = RunArgs { min_tps: Some(99.0), ..cli };
    let t = Thresholds::resolve(&cli).expect("profile must load");
    assert_eq!(t.min_tps, 99.0);
}

#[test]
fn without_profile_default_min_tps_is_20() {
    // No --profile → binary default min_tps=20 (R4 task 38, no regression).
    let cli = run_args_with_profile(None);
    let t = Thresholds::resolve(&cli).expect("defaults must load");
    assert_eq!(t.min_tps, 20.0);
}

// ── Task 39: R6 --profile must fail loudly on missing/malformed file ──
#[test]
fn profile_missing_file_errors_loudly() {
    let cli = run_args_with_profile(Some("does-not-exist".into()));
    let err = Thresholds::resolve(&cli).unwrap_err();
    assert!(err.to_string().contains("profile"), "error must mention profile: {err}");
}

#[test]
fn profile_malformed_json_errors_loudly() {
    let cli = RunArgs {
        profile: Some(std::path::PathBuf::from("/tmp/nxm-bench-malformed-profile.json")),
        ..run_args_with_profile(None)
    };
    // Write a malformed JSON file.
    std::fs::write("/tmp/nxm-bench-malformed-profile.json", "{ not valid json }").unwrap();
    let err = Thresholds::resolve(&cli).unwrap_err();
    assert!(
        err.to_string().contains("profile") || err.to_string().contains("parse"),
        "error must mention profile or parse: {err}"
    );
    let _ = std::fs::remove_file("/tmp/nxm-bench-malformed-profile.json");
}

// Helper: build a minimal RunArgs for threshold resolution tests.
fn run_args_with_profile(profile: Option<&str>) -> engine_bench::config::RunArgs {
    engine_bench::config::RunArgs {
        url: "http://127.0.0.1:11435".into(),
        engine: "engine".into(),
        model: None,
        sustained_reps: 20,
        sustained_tokens: 256,
        ramp: "256,512,1024".into(),
        out_dir: std::path::PathBuf::from("results"),
        assert: true,
        thresholds: None,
        min_tps: None,
        max_degradation_pct: None,
        engine_repo: None,
        profile: profile.map(|p| p.into()),
    }
}
