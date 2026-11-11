//! Pure pass/fail evaluation for the degradation tests.
//!
//! Extracted from `src/tests.rs` so the threshold wiring can be unit-tested
//! in isolation (task 38). No dependencies on the HTTP client or the CLI.

/// Sustained throughput stability: pass when the first→last drop is within
/// the guard *and* the minimum observed tps is at or above the floor.
///
/// An empty steady-state set (no samples after warmup skip) yields
/// `min_tps = f64::INFINITY`, which would vacuously satisfy the floor —
/// this is treated as **FAIL** (insufficient samples), never a vacuous pass.
pub fn sustained_pass(
    first_last_drop_pct: f64,
    min_tps: f64,
    max_degradation_pct: f64,
    min_tps_floor: f64,
) -> bool {
    if min_tps == f64::INFINITY {
        return false;
    }
    first_last_drop_pct <= max_degradation_pct && min_tps >= min_tps_floor
}

/// Length ramp: pass when the ramp drop is within the guard *and* the
/// minimum observed tps is at or above the floor.
///
/// An empty steady-state set (no ramp points after warmup skip) yields
/// `min_tps = f64::INFINITY`, treated as **FAIL** (insufficient samples).
pub fn ramp_pass(
    ramp_drop_pct: f64,
    min_tps: f64,
    max_ramp_drop_pct: f64,
    min_tps_floor: f64,
) -> bool {
    if min_tps == f64::INFINITY {
        return false;
    }
    ramp_drop_pct <= max_ramp_drop_pct && min_tps >= min_tps_floor
}
