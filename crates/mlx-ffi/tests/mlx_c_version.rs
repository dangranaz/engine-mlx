use engine_mlx_ffi::mlx_c_version::*;

#[test]
fn supported_range_is_0_7_through_0_8_x() {
    assert!(is_supported((0, 7, 0)));
    assert!(is_supported((0, 7, 9)));
    assert!(is_supported((0, 8, 0)));
    assert!(!is_supported((0, 6, 9)));
    assert!(!is_supported((0, 9, 0)));
}

#[test]
fn parse_cmake_version_extracts_triplet() {
    let real = r#"
set(PACKAGE_NAME "MLXC")
set(PACKAGE_VERSION "0.7.0")
set(PACKAGE_VERSION_STRING "0.7.0")
"#;
    assert_eq!(parse_cmake_version(real), Some((0, 7, 0)));

    let no_version = r#"
set(PACKAGE_NAME "MLXC")
set(PACKAGE_VERSION_STRING "unknown")
"#;
    assert_eq!(parse_cmake_version(no_version), None);
}

#[test]
fn is_supported_boundaries() {
    assert_eq!(is_supported((0, 7, 0)), true);
    assert_eq!(is_supported((0, 7, 5)), true);
    assert_eq!(is_supported((0, 8, 0)), true);
    assert_eq!(is_supported((0, 6, 9)), false);
    assert_eq!(is_supported((0, 9, 0)), false);
}
