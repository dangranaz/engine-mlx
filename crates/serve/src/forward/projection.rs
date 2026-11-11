//! Projection execution for Fused|Separate weight groups (task 12 / C2).
//!
//! A `Separate` projection runs one `quantized_matmul` per piece — each with its
//! own `QuantParams` — and concatenates the activations on the LAST axis in
//! concat order (q,k,v / gate,up). That reproduces the layout of the fused path
//! (the loader concatenates the weights on axis 0, and `quantized_matmul` with
//! `transpose=true` maps weight rows to output features), so the downstream
//! slices stay valid and the result matches the fused branch on identical weights.
//!
//! Scratch contract: every helper that creates an intermediate owns it
//! exactly once — it pushes it into `scratch` before returning (eager paths
//! free each entry once at the end of the pass) and returns the final
//! activation, which the caller owns (e.g. via `keep!`). A caller may pass
//! pieces it already pushed; helpers must never re-push those or the array
//! would be freed twice at cleanup.

use anyhow::{Context, Result};
use engine_mlx_ffi::{mlx_array, MlxCtx};

use crate::loader::{GateUpProj, QkvProj, QuantTensor};

/// Pure planning: the `(group_size, bits)` i32 pair each separate piece will use,
/// in concat order. Bails (via [`QuantParams::to_ffi`]) if any piece has
/// `group_size == 0` — the not-quantized sentinel must never reach an op.
///
/// # Example
/// ```
/// # use engine_mlx_serve::forward::projection::separate_piece_params;
/// # use engine_mlx_serve::loader::{QuantParams, QuantTensor};
/// # use engine_mlx_ffi::mlx_array;
/// # fn dummy() -> mlx_array { unsafe { std::mem::zeroed() } }
/// # let t = QuantTensor { w: dummy(), s: dummy(), b: dummy(), quant: QuantParams { group_size: 64, bits: 8 } };
/// let params = separate_piece_params(&[&t, &t], &["q_proj", "k_proj"]).unwrap();
/// assert_eq!(params, vec![(64, 8), (64, 8)]);
/// ```
pub fn separate_piece_params(pieces: &[&QuantTensor], names: &[&str]) -> Result<Vec<(i32, i32)>> {
    pieces
        .iter()
        .zip(names.iter())
        .map(|(t, name)| t.quant.to_ffi(name))
        .collect()
}

/// One quantized matmul for a single piece, using the piece's own params.
fn quant_matmul_piece(
    ctx: &MlxCtx,
    x: mlx_array,
    t: &QuantTensor,
    what: &str,
    scratch: &mut Vec<mlx_array>,
) -> Result<mlx_array> {
    let (gs, bits) = t.quant.to_ffi(what)?;
    let out = ctx
        .quantized_matmul(x, t.w, t.s, t.b, true, gs, bits)
        .with_context(|| format!("quantized_matmul failed for {what}"))?;
    scratch.push(out);
    Ok(out)
}

/// Concatenate activations on the last (feature) axis, in order.
///
/// Ownership: the caller is responsible for every part passed in `parts`
/// (e.g. `quant_matmul_piece` pushes each piece result into `scratch`
/// before it is wrapped here). This helper never re-pushes a part; it owns
/// and registers only the intermediate concat results it creates below.
/// The returned activation is owned by the caller.
fn concat_last_axis(
    ctx: &MlxCtx,
    parts: Vec<mlx_array>,
    what: &str,
    scratch: &mut Vec<mlx_array>,
) -> Result<mlx_array> {
    let mut it = parts.into_iter();
    let first = it
        .next()
        .ok_or_else(|| anyhow::anyhow!("{what}: no pieces to concatenate"))
        .with_context(|| format!("{what}: concat_last_axis requires pieces"))?;
    let ndim = ctx
        .shape(first)
        .with_context(|| format!("{what}: shape failed"))?
        .len() as i32;
    let axis = ndim - 1;
    // The first part is already tracked by the caller (`quant_matmul_piece`
    // pushed it); re-pushing it here would put the same handle in `scratch`
    // twice and free it twice at cleanup. So `acc` first round = not ours.
    let mut acc_is_ours = false;
    let mut acc = first;
    for p in it {
        let next = ctx
            .concatenate(acc, p, axis)
            .with_context(|| format!("{what}: concatenate failed on axis {axis}"))?;
        // Free each earlier accumulator exactly once: parts owned by the
        // caller are skipped (`acc_is_ours == false` on the first round),
        // intermediates created by this loop are pushed here.
        if acc_is_ours {
            scratch.push(acc);
        } else {
            acc_is_ours = true;
        }
        acc = next;
    }
    Ok(acc)
}

/// Execute a `QkvProj` (fused or separate) and return the qkv activation with the
/// same feature layout as the fused path (q|k|v on the last axis).
pub fn run_qkv(
    ctx: &MlxCtx,
    x: mlx_array,
    proj: &QkvProj,
    layer: usize,
    scratch: &mut Vec<mlx_array>,
) -> Result<mlx_array> {
    match proj {
        QkvProj::Fused(t) => {
            let (gs, bits) = t.quant.to_ffi("self_attn.qkv_proj")?;
            ctx.quantized_matmul(x, t.w, t.s, t.b, true, gs, bits)
                .with_context(|| "quantized_matmul failed for self_attn.qkv_proj".to_string())
        }
        QkvProj::Separate { q, k, v } => {
            let n = |p: &str| format!("layer {layer} self_attn.{p}");
            let qo = quant_matmul_piece(ctx, x, q, &n("q_proj"), scratch)?;
            let ko = quant_matmul_piece(ctx, x, k, &n("k_proj"), scratch)?;
            let vo = quant_matmul_piece(ctx, x, v, &n("v_proj"), scratch)?;
            concat_last_axis(ctx, vec![qo, ko, vo], &n("qkv_proj"), scratch)
        }
    }
}

/// Execute a `GateUpProj` (fused or separate); same contract as [`run_qkv`],
/// concat order gate|up on the last axis.
pub fn run_gate_up(
    ctx: &MlxCtx,
    x: mlx_array,
    proj: &GateUpProj,
    layer: usize,
    scratch: &mut Vec<mlx_array>,
) -> Result<mlx_array> {
    match proj {
        GateUpProj::Fused(t) => {
            let (gs, bits) = t.quant.to_ffi("mlp.gate_up_proj")?;
            ctx.quantized_matmul(x, t.w, t.s, t.b, true, gs, bits)
                .with_context(|| "quantized_matmul failed for mlp.gate_up_proj".to_string())
        }
        GateUpProj::Separate { gate, up } => {
            let n = |p: &str| format!("layer {layer} mlp.{p}");
            let go = quant_matmul_piece(ctx, x, gate, &n("gate_proj"), scratch)?;
            let uo = quant_matmul_piece(ctx, x, up, &n("up_proj"), scratch)?;
            concat_last_axis(ctx, vec![go, uo], &n("gate_up_proj"), scratch)
        }
    }
}
