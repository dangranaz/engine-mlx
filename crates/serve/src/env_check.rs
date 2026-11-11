//! Runtime MLX liveness probe.
//!
//! Invoked via `nxm-engine-mlx --check-env`. Queries the linked MLX library
//! through `mlx_version()` and reports whether the library is reachable and
//! returns a readable version string. The number printed is the **core MLX
//! version**, not the mlx-c package version: mlx-c is pinned at build-time
//! (task 02, `MLXCConfigVersion.cmake` gate) and is not judged at runtime.
//! No server is started in this mode: it prints the outcome and exits.

use anyhow::{Context, Result};
use tracing::info;

const TARGET: &str = "engine_mlx::env";

/// Decide whether the MLX library is reachable from a version string.
///
/// Returns `true` when a non-empty version string was read (library reachable),
/// `false` when the string is empty or unreadable. No version range is judged
/// here — that lives only in the build-time gate (task 02).
///
/// # Example
///
/// ```
/// use engine_mlx_serve::env_check::probe_decision;
/// assert!(probe_decision("0.32.4"));
/// assert!(!probe_decision(""));
/// ```

pub fn probe_decision(version_str: &str) -> bool {
    !version_str.trim().is_empty()
}

#[cfg(feature = "mlx")]
fn run_check_mlx() -> Result<i32> {
    use engine_mlx_ffi::runtime::mlx_version_string;

    let version_str = mlx_version_string()
        .context("MLX library not reachable — is mlx-c installed and linked?")?;
    if probe_decision(&version_str) {
        println!("MLX runtime library reachable; reported core version: {version_str}");
        info!(target: TARGET, version = %version_str, "MLX runtime liveness probe OK");
        Ok(0)
    } else {
        println!("MLX library not reachable — empty version string returned");
        info!(target: TARGET, version = %version_str, "MLX runtime liveness probe FAILED");
        Ok(1)
    }
}

#[cfg(not(feature = "mlx"))]
fn run_check_mlx() -> Result<i32> {
    info!(target: TARGET, "runtime MLX check unavailable: stub build (feature `mlx` not enabled)");
    println!("MLX runtime check: not available (stub build, feature `mlx` not enabled)");
    Ok(0)
}

/// Run the runtime MLX liveness probe.
///
/// Returns `Ok(0)` when the library is reachable, `Ok(1)` when it is not
/// reachable, and `Err` when the version string could not be read at all.
pub fn run_check() -> Result<i32> {
    run_check_mlx()
}
