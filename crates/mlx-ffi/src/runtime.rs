//! Runtime MLX version query.
//!
//! Only compiled when the `mlx` feature is enabled, i.e. when the real
//! bindgen-generated MLX-C FFI bindings are available.

use std::ffi::CStr;

use crate::{mlx_string_data, mlx_string_free, mlx_string_new, mlx_version};

/// Query the linked MLX library for its version string via `mlx_version()`.
///
/// Returns `None` when the call fails (non-zero return code) or when the
/// returned string cannot be read as UTF-8.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "mlx")] {
/// # use engine_mlx_ffi::runtime::mlx_version_string;
/// match mlx_version_string() {
///     Some(version) => println!("linked MLX core version: {version}"),
///     None => eprintln!("could not read MLX version"),
/// }
/// # }
/// ```

pub fn mlx_version_string() -> Option<String> {
    unsafe {
        let mut s = mlx_string_new();
        let rc = mlx_version(&mut s);
        if rc != 0 {
            // `mlx_string_new` already allocated `s`; free it on the error path
            // too, otherwise every failed call leaks one `mlx_string`.
            let _ = mlx_string_free(s);
            return None;
        }
        // `mlx_string_data` does NOT consume the string: it returns a pointer to
        // the contents, valid for the life of the string. Copy into an owned
        // String, then release with `mlx_string_free` to avoid leaking one
        // `mlx_string` per call.
        let ptr = mlx_string_data(s);
        let version = CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_owned());
        let _ = mlx_string_free(s);
        version
    }
}
