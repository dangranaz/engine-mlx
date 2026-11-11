// mlx-c version gate.
// Parses the mlx-c version exported by its CMake config file and compares it
// against the supported range. The supported range is declared here and
// nowhere else.

/// Minimum supported mlx-c version (inclusive), as `(major, minor, patch)`.
pub const MLX_C_MIN: (u32, u32, u32) = (0, 7, 0);

/// Maximum supported mlx-c version (exclusive), as `(major, minor, patch)`.
pub const MLX_C_MAX_EXCL: (u32, u32, u32) = (0, 9, 0);

/// Parse `set(PACKAGE_VERSION "X.Y.Z")` from the content of a
/// `MLXCConfigVersion.cmake` file.
///
/// Returns `None` when the file does not declare a parsable `PACKAGE_VERSION`.
///
/// # Example
///
/// ```
/// # use engine_mlx_ffi::mlx_c_version::parse_cmake_version;
/// let cmake = r#"
/// set(PACKAGE_NAME "MLXC")
/// set(PACKAGE_VERSION "0.7.0")
/// "#;
/// assert_eq!(parse_cmake_version(cmake), Some((0, 7, 0)));
/// ```

pub fn parse_cmake_version(content: &str) -> Option<(u32, u32, u32)> {
    let marker = "set(PACKAGE_VERSION \"";
    for line in content.lines() {
        let Some(rest) = line.trim().strip_prefix(marker) else { continue };
        let version = rest.split('"').next()?;
        let mut parts = version.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        return Some((major, minor, patch));
    }
    None
}

/// Return `true` when `version` falls inside the supported range
/// `[MLX_C_MIN, MLX_C_MAX_EXCL)`.
///
/// # Example
///
/// ```
/// # use engine_mlx_ffi::mlx_c_version::{is_supported, MLX_C_MIN, MLX_C_MAX_EXCL};
/// assert!(is_supported(MLX_C_MIN));
/// assert!(!is_supported(MLX_C_MAX_EXCL));
/// ```

pub fn is_supported(version: (u32, u32, u32)) -> bool {
    version >= MLX_C_MIN && version < MLX_C_MAX_EXCL
}

/// Parse a runtime version string like `"0.7.0"`, `"0.7.0-dev"` or
/// `"0.7.0 (abc123)"` into `(major, minor, patch)`, ignoring any trailing
/// suffix. Returns `None` when no `major.minor.patch` triplet can be found.
///
/// # Example
///
/// ```
/// # use engine_mlx_ffi::mlx_c_version::parse_runtime_version;
/// assert_eq!(parse_runtime_version("0.7.0-dev"), Some((0, 7, 0)));
/// assert_eq!(parse_runtime_version("0.7.0 (abc123)"), Some((0, 7, 0)));
/// assert_eq!(parse_runtime_version("not a version"), None);
/// ```

pub fn parse_runtime_version(s: &str) -> Option<(u32, u32, u32)> {
    let normalized: String = s
        .chars()
        .map(|c| if c.is_ascii_digit() || c == '.' { c } else { ' ' })
        .collect();
    let token = normalized
        .split_whitespace()
        .find(|t| t.split('.').count() == 3)?;
    let mut parts = token.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}
