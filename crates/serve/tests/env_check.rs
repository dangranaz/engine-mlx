use engine_mlx_ffi::mlx_c_version::parse_runtime_version;
use engine_mlx_serve::env_check::probe_decision;

#[test]
fn parse_runtime_version_handles_suffixes() {
    assert_eq!(parse_runtime_version("0.7.0"), Some((0, 7, 0)));
    assert_eq!(parse_runtime_version("0.7.0-dev"), Some((0, 7, 0)));
    assert_eq!(parse_runtime_version("0.7.0 (abc123)"), Some((0, 7, 0)));
    assert_eq!(parse_runtime_version("mlx-c version 0.7.0"), Some((0, 7, 0)));
    assert_eq!(parse_runtime_version("no version here"), None);
    assert_eq!(parse_runtime_version(""), None);
}

#[test]
fn probe_fails_only_when_unreadable() {
    // Any non-empty version string means the library is reachable; no range is
    // judged here — that lives only in the build-time gate (task 02).
    assert!(probe_decision("0.32.4"));
    assert!(probe_decision("0.7.0"));
    assert!(!probe_decision(""));
    assert!(!probe_decision("   "));
}
