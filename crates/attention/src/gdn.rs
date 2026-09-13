//! Gated DeltaNet (GDN) — linear recurrent attention replacement.
//!
//! Used by Qwen3.5 (75% of layers), Ornith, and other hybrid models.
//! GDN has O(1) memory per step — no KV cache growth.
//!
//! Unlike standard attention which stores K/V and grows linearly with context,
//! GDN maintains a fixed-size recurrent state `[B, Hv, Dv, Dk]`.
//!
//! ## Recurrent step
//!
//! ```text
//! g = exp(-exp(A_log) * softplus(a + dt_bias))   // decay
//! state = state * g                               // forget
//! kv_mem = (state * k).sum(axis=-1)               // recall
//! delta = (v - kv_mem) * sigmoid(b)               // write signal
//! state = state + k.unsqueeze(-2) * delta.unsqueeze(-1)  // update
//! y = (state * q).sum(axis=-1)                    // read
//! ```
//!
//! ## Usage
//!
//! GDN does NOT use a `KvCache` — it maintains its own recurrent state.
//! It's an attention *replacement*, not an attention *variant*.
//!
//! ```rust,ignore
//! let mut gdn_state = GdnState::new();
//! let output = gdn_forward(ctx, x, &gdn_cfg, &mut gdn_state, &weights, seq_len)?;
//! ```
//!
//! ## Reference
//!
//! - `vendor/mlx-lm/mlx_lm/models/qwen3_5.py` — Python GDN implementation
//! - `engine_mlx_ops::gdn` — atomic GDN ops (compute_gate, gdn_step)
//! - `engine_mlx_ops::gdn_metal` — fused Metal kernel via mlx_fast_metal_kernel

// Re-export the GDN types and forward from engine_mlx_ops
// gdn_forward is in its own module (split from gdn.rs per RULES 300-line limit)
pub use engine_mlx_ops::gdn::{GdnConfig, GdnState, compute_gate, gdn_step};
pub use engine_mlx_ops::gdn_forward::gdn_forward;
pub use engine_mlx_ops::gdn_metal::gdn_step_metal;
