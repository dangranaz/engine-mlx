//! Build script — generates MLX-C FFI bindings via bindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_C_PATH");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=wrapper.h");

    // Resolve the mlx-c / mlx install prefixes robustly. An explicit env var
    // wins ONLY if it actually contains the library; otherwise we fall back to
    // the standard Homebrew prefix so an end-user never has to guess.
    let mlx_c_prefix = resolve_prefix(
        &["MLX_C_PATH", "MLX_C_PREFIX"],
        "lib/libmlxc.dylib",
        &["/opt/homebrew/opt/mlx-c", "/usr/local/opt/mlx-c"],
    );
    let mlx_prefix = resolve_prefix(
        &["MLX_PREFIX"],
        "lib/libmlx.dylib",
        &["/opt/homebrew/opt/mlx", "/usr/local/opt/mlx"],
    );

    let is_mlx_feature = env::var("CARGO_FEATURE_MLX").is_ok();

    if is_mlx_feature {
        // When mlx feature is on, link homebrew-installed MLX dynamic libraries.
        let mlx_c_prefix_path = PathBuf::from(&mlx_c_prefix);
        let mlx_prefix_path = PathBuf::from(&mlx_prefix);

        let mlxc_dylib = mlx_c_prefix_path.join("lib/libmlxc.dylib");
        let mlx_dylib = mlx_prefix_path.join("lib/libmlx.dylib");

        if mlxc_dylib.exists() {
            println!("cargo:rustc-link-search=native={}", mlx_c_prefix_path.join("lib").display());
            println!("cargo:rustc-link-lib=dylib=mlxc");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlx_c_prefix_path.join("lib").display());
        }
        if mlx_dylib.exists() {
            println!("cargo:rustc-link-search=native={}", mlx_prefix_path.join("lib").display());
            println!("cargo:rustc-link-lib=dylib=mlx");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlx_prefix_path.join("lib").display());
        }
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else {
        // Stub path — no real linking needed, functions panic.
        // Still link for bindgen header resolution.
        let mlxc_lib = format!("{mlx_c_prefix}/lib/libmlxc.dylib");
        let mlx_lib = format!("{mlx_prefix}/lib/libmlx.dylib");
        if PathBuf::from(&mlxc_lib).exists() {
            println!("cargo:rustc-link-search=native={mlx_c_prefix}/lib");
            println!("cargo:rustc-link-lib=dylib=mlxc");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_c_prefix}/lib");
        }
        if PathBuf::from(&mlx_lib).exists() {
            println!("cargo:rustc-link-search=native={mlx_prefix}/lib");
            println!("cargo:rustc-link-lib=dylib=mlx");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_prefix}/lib");
        }
    }

    // Bindgen - only when mlx feature is enabled or headers are available.
    // For stub (ubuntu CI without mlx-c), generate empty bindings to allow cargo check.
    if !is_mlx_feature {
        let has_header = PathBuf::from(format!("{mlx_c_prefix}/include/mlx/c/mlx.h")).exists()
            || PathBuf::from("/opt/homebrew/include/mlx/c/mlx.h").exists()
            || PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
                .join("../../../vendors/mlx-c/include/mlx/c/mlx.h")
                .exists();
        if !has_header {
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            let bindings_path = out_path.join("bindings.rs");
            std::fs::write(&bindings_path, "// stub bindings - mlx feature disabled, no headers\n").unwrap();
            return;
        }
    }

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("mlx_.*")
        .allowlist_function("_mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .rustified_enum("mlx_dtype_")
        .rustified_enum("mlx_device_type_")
        .derive_default(true)
        .layout_tests(false);

    builder = builder.clang_arg("-I/opt/homebrew/include");

    let mlxc_lib = format!("{mlx_c_prefix}/lib/libmlxc.dylib");
    if PathBuf::from(&mlxc_lib).exists() {
        builder = builder.clang_arg(format!("-I{mlx_c_prefix}/include"));
    }
    let mlx_lib = format!("{mlx_prefix}/lib/libmlx.dylib");
    if PathBuf::from(&mlx_lib).exists() {
        builder = builder.clang_arg(format!("-I{mlx_prefix}/include"));
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendors_mlx_c = manifest_dir.join("../../../vendors/mlx-c/include");
    let vendors_mlx = manifest_dir.join("../../../vendors/mlx/include");
    if vendors_mlx_c.exists() {
        builder = builder.clang_arg(format!("-I{}", vendors_mlx_c.display()));
    }
    if vendors_mlx.exists() {
        builder = builder.clang_arg(format!("-I{}", vendors_mlx.display()));
    }

    let bindings = builder
        .generate()
        .expect("bindgen failed. Check mlx-c headers: brew install mlx-c or set MLX_C_PATH");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("could not write bindings.rs");

    let raw = std::fs::read_to_string(&bindings_path).unwrap();
    let cleaned: String = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("#!["))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&bindings_path, cleaned).unwrap();
}

/// Resolve an install prefix for an MLX library.
///
/// Priority:
/// 1. Each env var in `env_vars`, but only if the prefix actually contains
///    `lib_rel` (a bad/stale env var is ignored rather than silently breaking
///    the link).
/// 2. The first `fallback` prefix that contains `lib_rel` (Homebrew layout).
/// 3. The first env var value as-is (last resort, so bindgen include dirs still
///    resolve on unusual setups).
///
/// A leading `~` is expanded to `$HOME`.
fn resolve_prefix(env_vars: &[&str], lib_rel: &str, fallbacks: &[&str]) -> String {
    let expand = |p: String| -> String {
        if let Some(rest) = p.strip_prefix("~") {
            if let Ok(home) = env::var("HOME") {
                return format!("{home}{rest}");
            }
        }
        p
    };
    let has_lib = |prefix: &str| PathBuf::from(prefix).join(lib_rel).exists();

    // 1. Env var that actually points at a real library.
    for var in env_vars {
        if let Ok(val) = env::var(var) {
            let val = expand(val);
            if has_lib(&val) {
                return val;
            }
        }
    }
    // 2. First fallback prefix that contains the library.
    for fb in fallbacks {
        if has_lib(fb) {
            return (*fb).to_owned();
        }
    }
    // 3. Last resort: first env var value, else first fallback.
    for var in env_vars {
        if let Ok(val) = env::var(var) {
            return expand(val);
        }
    }
    fallbacks.first().map(|s| (*s).to_owned()).unwrap_or_default()
}
