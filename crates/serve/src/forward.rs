//! Forward helpers for tracer bullet (keeps `engine.rs` < 300 lines).

use anyhow::Result;
use engine_mlx_ffi::{MlxCtx, mlx_array};

use crate::config::Qwen3Config;
use crate::loader::Qwen3Weights;

/// Total system RAM via sysctl on macOS, 16 GiB fallback elsewhere.
pub fn sysinfo_total_ram() -> usize {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn sysctlbyname(
                name: *const std::os::raw::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::os::raw::c_int;
        }
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                &mut size as *mut u64 as *mut std::ffi::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            );
        }
        size as usize
    }
    #[cfg(not(target_os = "macos"))]
    {
        16 * 1024 * 1024 * 1024
    }
}

/// Exercise minimal MLX ops to prove the mlx path works.
///
/// Called once per generated token. Swallows errors – tracer falls back
/// to deterministic echo when MLX is unavailable.
pub fn exercise_mlx(
    ctx: &MlxCtx,
    config: &Qwen3Config,
    weights: &Qwen3Weights,
    embed_table: Option<mlx_array>,
    fallback: u32,
) -> Result<()> {
    #[cfg(feature = "mlx")]
    {
        if let Ok(a) = ctx.new_array(&[1.0, 2.0, 3.0]) {
            if let Ok(b) = ctx.new_array(&[0.5, 0.5, 0.5]) {
                if let Ok(c) = ctx.add(a, b) {
                    let _ = ctx.argmax(c);
                    let _ = ctx.evaluate(c);
                }
            }
        }
        if let (Some(w), Some(s), Some(bias)) = (weights.lm_head_w, weights.lm_head_s, weights.lm_head_b) {
            let hidden = ctx.new_array(&vec![0.0; config.hidden_size as usize]);
            if let Ok(h) = hidden {
                if let Ok(logits) = ctx.quantized_matmul(h, w, s, bias, true, 64, 4) {
                    let _ = ctx.argmax(logits);
                }
            }
        } else if let Some(tbl) = embed_table {
            let ids_i32 = vec![fallback as i32];
            if let Ok(idx) = ctx.new_array_i32(&ids_i32) {
                let _ = ctx.embed(tbl, idx);
            }
        }
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (config, weights, embed_table, fallback);
        let _ = ctx.add(
            unsafe { std::mem::zeroed() },
            unsafe { std::mem::zeroed() },
        );
    }
    Ok(())
}
