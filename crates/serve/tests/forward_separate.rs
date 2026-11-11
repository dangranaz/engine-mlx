//! Tests for the Separate branch of projections in the forward (task 12 / C2).
//!
//! R3 (each piece uses its own params; `group_size == 0` bails) is pure and runs
//! in the default suite. R1/R2 (Separate output == Fused output on identical
//! weights) need the mlx graph, so they are `#[cfg(feature = "mlx")]` and run with:
//! `cargo test -p engine-mlx-serve --features mlx forward_separate`

use engine_mlx_ffi::mlx_array;
use engine_mlx_serve::forward::projection::separate_piece_params;
use engine_mlx_serve::loader::{QuantParams, QuantTensor};

fn dummy() -> mlx_array {
    unsafe { std::mem::zeroed() }
}

fn qt(group_size: usize, bits: usize) -> QuantTensor {
    QuantTensor {
        w: dummy(),
        s: dummy(),
        b: dummy(),
        quant: QuantParams { group_size, bits },
    }
}

/// R3: each separate piece is planned with its OWN `(group_size, bits)`, in
/// concat order — never the first piece's params for all. A piece with
/// `group_size == 0` bails with a contextual error naming it (task 11 sentinel).
#[test]
fn separate_pieces_use_own_params() {
    let q = qt(64, 8);
    let k = qt(32, 4);
    let v = qt(64, 4);
    let params = separate_piece_params(
        &[&q, &k, &v],
        &[
            "layer 0 self_attn.q_proj",
            "layer 0 self_attn.k_proj",
            "layer 0 self_attn.v_proj",
        ],
    )
    .unwrap();
    assert_eq!(params, vec![(64, 8), (32, 4), (64, 4)]);

    let zero = qt(0, 0);
    let err = separate_piece_params(&[&q, &zero], &["q_proj", "k_proj"])
        .expect_err("group_size == 0 piece must bail");
    let msg = format!("{err:#}");
    assert!(msg.contains("k_proj"), "error must name the piece: {msg}");
}

#[cfg(feature = "mlx")]
mod mlx_equiv {
    use super::*;
    use engine_mlx_ffi::MlxCtx;
    use engine_mlx_serve::forward::projection::{run_gate_up, run_qkv};
    use engine_mlx_serve::loader::{GateUpProj, QkvProj};

    fn pattern(n: usize, offset: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.013 + offset) % 1.0 - 0.5)
            .collect()
    }

    fn quantized_piece(
        ctx: &MlxCtx,
        vals: &[f32],
        rows: i32,
        cols: i32,
        quant: QuantParams,
    ) -> QuantTensor {
        let w = ctx
            .reshape(ctx.new_array(vals).unwrap(), &[rows, cols])
            .unwrap();
        let (d, s, b) = ctx
            .quantize_kv_native(w, quant.group_size as i32, quant.bits as i32)
            .unwrap();
        QuantTensor { w: d, s, b, quant }
    }

    fn assert_close(ctx: &MlxCtx, a: mlx_array, b: mlx_array, what: &str) {
        let a = ctx.to_vec_f32(a).unwrap();
        let b = ctx.to_vec_f32(b).unwrap();
        assert_eq!(a.len(), b.len(), "{what}: shape mismatch");
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "{what}: max abs diff {max_diff}");
    }

    /// R1: with identical weights and matching params, the Separate branch
    /// (3 matmuls + concat on the feature axis) equals the Fused branch
    /// (1 matmul on the concatenated weight) — same layout, same numbers.
    #[test]
    fn qkv_separate_matches_fused() {
        let ctx = MlxCtx::cpu();
        let qp = QuantParams {
            group_size: 64,
            bits: 4,
        };
        let q = quantized_piece(&ctx, &pattern(16 * 64, 0.00), 16, 64, qp);
        let k = quantized_piece(&ctx, &pattern(16 * 64, 0.17), 16, 64, qp);
        let v = quantized_piece(&ctx, &pattern(16 * 64, 0.31), 16, 64, qp);

        // Fused weight = concat of the pieces on axis 0 (what the loader does).
        let fused = QkvProj::Fused(QuantTensor {
            w: ctx
                .concatenate(ctx.concatenate(q.w, k.w, 0).unwrap(), v.w, 0)
                .unwrap(),
            s: ctx
                .concatenate(ctx.concatenate(q.s, k.s, 0).unwrap(), v.s, 0)
                .unwrap(),
            b: ctx
                .concatenate(ctx.concatenate(q.b, k.b, 0).unwrap(), v.b, 0)
                .unwrap(),
            quant: qp,
        });
        let separate = QkvProj::Separate { q, k, v };

        let xvals = pattern(2 * 64, 0.07);
        let mk_x = |ctx: &MlxCtx| {
            ctx.reshape(ctx.new_array(&xvals).unwrap(), &[1, 2, 64])
                .unwrap()
        };
        let mut scratch = Vec::new();
        let out_fused = run_qkv(&ctx, mk_x(&ctx), &fused, 0, &mut scratch).unwrap();
        let out_sep = run_qkv(&ctx, mk_x(&ctx), &separate, 0, &mut scratch).unwrap();
        assert_eq!(ctx.shape(out_fused).unwrap(), vec![1, 2, 48]);
        assert_close(&ctx, out_fused, out_sep, "qkv separate vs fused");
    }

    /// R2: same equivalence for gate/up (2 matmuls + concat).
    #[test]
    fn gate_up_separate_matches_fused() {
        let ctx = MlxCtx::cpu();
        let qp = QuantParams {
            group_size: 32,
            bits: 4,
        };
        let gate = quantized_piece(&ctx, &pattern(8 * 64, 0.11), 8, 64, qp);
        let up = quantized_piece(&ctx, &pattern(8 * 64, 0.43), 8, 64, qp);

        let fused = GateUpProj::Fused(QuantTensor {
            w: ctx.concatenate(gate.w, up.w, 0).unwrap(),
            s: ctx.concatenate(gate.s, up.s, 0).unwrap(),
            b: ctx.concatenate(gate.b, up.b, 0).unwrap(),
            quant: qp,
        });
        let separate = GateUpProj::Separate { gate, up };

        let xvals = pattern(2 * 64, 0.23);
        let mk_x = |ctx: &MlxCtx| {
            ctx.reshape(ctx.new_array(&xvals).unwrap(), &[1, 2, 64])
                .unwrap()
        };
        let mut scratch = Vec::new();
        let out_fused = run_gate_up(&ctx, mk_x(&ctx), &fused, 0, &mut scratch).unwrap();
        let out_sep = run_gate_up(&ctx, mk_x(&ctx), &separate, 0, &mut scratch).unwrap();
        assert_eq!(ctx.shape(out_fused).unwrap(), vec![1, 2, 16]);
        assert_close(&ctx, out_fused, out_sep, "gate_up separate vs fused");
    }
}
