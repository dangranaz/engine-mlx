//! Decode op-class profiler — attributes per-token GPU time across dispatch
//! classes on the real loaded model, to localize the recoverable gap vs mlx_lm.
//!
//! Run:
//!   ENGINE_MLX_MODEL=$HOME/models/lmstudio-community/Qwen3-1.7B-MLX-4bit \
//!   MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx \
//!   cargo test --release --features mlx -p engine-mlx-serve --test profile_decode \
//!     -- --ignored --nocapture --test-threads=1
//!
//! Method: for each op class, replay it over all layers' real weights at decode
//! shapes (batch=1, seq=1), ITERS times, eval+sync, and measure wall time.
//! Sum across layers = one "token" of that class; report ms/token and % share.

#[cfg(feature = "mlx")]
mod prof {
    use std::path::Path;
    use std::time::Instant;
    use engine_mlx_serve::Qwen3Engine;

    fn model_dir() -> String {
        std::env::var("ENGINE_MLX_MODEL").unwrap_or_else(|_| {
            let h = std::env::var("HOME").unwrap_or_default();
            format!("{h}/models/lmstudio-community/Qwen3-1.7B-MLX-4bit")
        })
    }

    #[test]
    #[ignore = "hardware profiler"]
    fn profile_decode_op_classes() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() { eprintln!("skip: model not found {md}"); return; }
        let eng = Qwen3Engine::load(dir).expect("load");
        let ctx = &eng.ctx;
        let cfg = &eng.config;
        let w = &eng.weights;
        let nl = w.layers.len();
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let hidden = cfg.hidden_size as i32;
        let ff = cfg.intermediate_size;
        let (gs, bits) = cfg.quantization.as_ref().map(|q| (q.group_size as i32, q.bits as i32)).unwrap_or((64, 4));
        let scale = 1.0 / (hd as f32).sqrt();
        let cap = 256i32;
        let eps = cfg.rms_norm_eps;
        println!("model={md}");
        println!("hidden={hidden} layers={nl} nh={nh} nkv={nkv} hd={hd} ff={ff} bits={bits} gs={gs}");

        // Decode-shaped inputs.
        let x = ctx.zeros_bf16(&[1, 1, hidden]).unwrap();          // token hidden
        let q4 = ctx.zeros_bf16(&[1, nh, 1, hd]).unwrap();          // q [1,nh,1,hd]
        let kfull = ctx.zeros_bf16(&[1, nkv, cap, hd]).unwrap();    // k cache full
        let vfull = ctx.zeros_bf16(&[1, nkv, cap, hd]).unwrap();
        let kv1 = ctx.zeros_bf16(&[1, nkv, 1, hd]).unwrap();        // one-pos k/v to write
        let mask = ctx.zeros_bf16(&[1, 1, 1, cap]).unwrap();
        let off = ctx.new_array_i32(&[7]).unwrap();
        let start0 = ctx.new_array_i32(&[0]).unwrap();
        let attn_flat = ctx.zeros_bf16(&[1, 1, nh * hd]).unwrap();  // post-attn for O proj
        let gate = ctx.zeros_bf16(&[1, 1, ff]).unwrap();           // for silu/mul
        let hidden_ff = ctx.zeros_bf16(&[1, 1, ff]).unwrap();      // for down proj

        const ITERS: usize = 200;

        // Helper: time a closure that produces arrays for all layers, eval+sync.
        macro_rules! time_class {
            ($name:expr, $body:expr) => {{
                // warmup
                { let outs = $body; ctx.eval_all_and_sync(&outs).unwrap(); }
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    let outs = $body;
                    ctx.eval_all_and_sync(&outs).unwrap();
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                println!("{:<28} {:>8.3} ms/token", $name, ms);
                ms
            }};
        }

        println!("\n=== per-op-class time (sum over {nl} layers, {ITERS} iters) ===");

        // QKV proj (fused quantized_matmul), all layers.
        let t_qkv = time_class!("qkv_proj (qmatmul)", {
            let mut v = Vec::with_capacity(nl);
            for l in &w.layers { v.push(ctx.quantized_matmul(x, l.qkv_proj_w, l.qkv_proj_s, l.qkv_proj_b, true, gs, bits).unwrap()); }
            v
        });
        // O proj.
        let t_o = time_class!("o_proj (qmatmul)", {
            let mut v = Vec::with_capacity(nl);
            for l in &w.layers { v.push(ctx.quantized_matmul(attn_flat, l.o_proj_w, l.o_proj_s, l.o_proj_b, true, gs, bits).unwrap()); }
            v
        });
        // gate_up proj (fused).
        let t_gup = time_class!("gate_up_proj (qmatmul)", {
            let mut v = Vec::with_capacity(nl);
            for l in &w.layers { v.push(ctx.quantized_matmul(x, l.gate_up_proj_w, l.gate_up_proj_s, l.gate_up_proj_b, true, gs, bits).unwrap()); }
            v
        });
        // down proj.
        let t_down = time_class!("down_proj (qmatmul)", {
            let mut v = Vec::with_capacity(nl);
            for l in &w.layers { v.push(ctx.quantized_matmul(hidden_ff, l.down_proj_w, l.down_proj_s, l.down_proj_b, true, gs, bits).unwrap()); }
            v
        });
        // rms_norm (input + post-attn = 2 per layer) + q/k norm (2 per layer).
        let t_norm = time_class!("rms_norm x4/layer", {
            let mut v = Vec::with_capacity(nl * 4);
            for l in &w.layers {
                v.push(ctx.rms_norm(x, l.input_layernorm, eps).unwrap());
                v.push(ctx.rms_norm(x, l.post_attention_layernorm, eps).unwrap());
                if let Some(qn) = l.q_norm { v.push(ctx.rms_norm(q4, qn, eps).unwrap()); }
                if let Some(kn) = l.k_norm { v.push(ctx.rms_norm(kv1, kn, eps).unwrap()); }
            }
            v
        });
        // rope_dynamic (q + k = 2 per layer).
        let t_rope = time_class!("rope_dynamic x2/layer", {
            let mut v = Vec::with_capacity(nl * 2);
            for _ in 0..nl {
                v.push(ctx.rope_dynamic(q4, hd, cfg.rope_theta, off).unwrap());
                v.push(ctx.rope_dynamic(kv1, hd, cfg.rope_theta, off).unwrap());
            }
            v
        });
        // sdpa_masked (1 per layer).
        let t_sdpa = time_class!("sdpa_masked x1/layer", {
            let mut v = Vec::with_capacity(nl);
            for _ in 0..nl { v.push(ctx.sdpa_masked(q4, kfull, vfull, scale, mask).unwrap()); }
            v
        });
        // slice_update_dynamic (KV write, 2 per layer).
        let t_kvw = time_class!("slice_update_dyn x2/layer", {
            let mut v = Vec::with_capacity(nl * 2);
            for _ in 0..nl {
                v.push(ctx.slice_update_dynamic(kfull, kv1, off, &[2]).unwrap());
                v.push(ctx.slice_update_dynamic(vfull, kv1, off, &[2]).unwrap());
            }
            v
        });
        // scatter_single (KV write, 2 per layer) — test if lower overhead than slice_update_dynamic.
        // updates.ndim must == indices.ndim + a.ndim (5 here); reshape kv1 [1,8,1,128] -> [1,1,8,1,128].
        let indices_arr = ctx.new_array_i32(&[7]).unwrap(); // same offset as `off`
        let kv1_flat = ctx.reshape(kv1, &[1, 1, nkv, 1, hd]).unwrap();
        let t_kvw_scatter = time_class!("scatter_single x2/layer", {
            let mut v = Vec::with_capacity(nl * 2);
            for _ in 0..nl {
                v.push(ctx.scatter_single(kfull, indices_arr, kv1_flat, 2).unwrap());
                v.push(ctx.scatter_single(vfull, indices_arr, kv1_flat, 2).unwrap());
            }
            v
        });
        // slice_dynamic (q/k/v + gate/up = 5 per layer).
        let qkv_flat = ctx.zeros_bf16(&[1, 1, (nh + 2 * nkv) * hd]).unwrap();
        let gup_flat = ctx.zeros_bf16(&[1, 1, 2 * ff]).unwrap();
        let t_slice = time_class!("slice_dynamic x5/layer", {
            let mut v = Vec::with_capacity(nl * 5);
            for _ in 0..nl {
                v.push(ctx.slice_dynamic(qkv_flat, start0, &[2], &[1, 1, nh * hd]).unwrap());
                v.push(ctx.slice_dynamic(qkv_flat, start0, &[2], &[1, 1, nkv * hd]).unwrap());
                v.push(ctx.slice_dynamic(qkv_flat, start0, &[2], &[1, 1, nkv * hd]).unwrap());
                v.push(ctx.slice_dynamic(gup_flat, start0, &[2], &[1, 1, ff]).unwrap());
                v.push(ctx.slice_dynamic(gup_flat, start0, &[2], &[1, 1, ff]).unwrap());
            }
            v
        });
        // transpose_axes (q/k/v + attn = 4 per layer).
        let t_tr = time_class!("transpose_axes x4/layer", {
            let mut v = Vec::with_capacity(nl * 4);
            for _ in 0..nl {
                v.push(ctx.transpose_axes(ctx.zeros_bf16(&[1,1,nh,hd]).unwrap(), &[0,2,1,3]).unwrap());
                v.push(ctx.transpose_axes(ctx.zeros_bf16(&[1,1,nkv,hd]).unwrap(), &[0,2,1,3]).unwrap());
                v.push(ctx.transpose_axes(ctx.zeros_bf16(&[1,1,nkv,hd]).unwrap(), &[0,2,1,3]).unwrap());
                v.push(ctx.transpose_axes(q4, &[0,2,1,3]).unwrap());
            }
            v
        });
        // silu + swiglu multiply (silu=2 ops + 1 mul = 3 per layer).
        let t_silu = time_class!("silu+mul x1/layer", {
            let mut v = Vec::with_capacity(nl);
            for _ in 0..nl { v.push(ctx.multiply(ctx.silu(gate).unwrap(), hidden_ff).unwrap()); }
            v
        });
        // residual adds (2 per layer).
        let t_add = time_class!("add x2/layer", {
            let mut v = Vec::with_capacity(nl * 2);
            for _ in 0..nl { v.push(ctx.add(x, x).unwrap()); v.push(ctx.add(x, x).unwrap()); }
            v
        });
        // lm_head (1 per token, big matmul over vocab).
        let (lm_w, lm_s, lm_b) = match (&w.lm_head_w, &w.lm_head_s, &w.lm_head_b) {
            (Some(a),Some(b),Some(c)) => (*a,*b,*c),
            _ => (w.embed_tokens, w.embed_scales, w.embed_biases),
        };
        let x2 = ctx.zeros_bf16(&[1, hidden]).unwrap();
        let t_lm = time_class!("lm_head+argmax x1/token", {
            let logits = ctx.quantized_matmul(x2, lm_w, lm_s, lm_b, true, gs, bits).unwrap();
            vec![ctx.argmax(logits).unwrap()]
        });

        let total = t_qkv + t_o + t_gup + t_down + t_norm + t_rope + t_sdpa + t_kvw + t_slice + t_tr + t_silu + t_add + t_lm;
        let matmul = t_qkv + t_o + t_gup + t_down + t_lm;
        let overhead = total - matmul;
        println!("\n=== summary ===");
        println!("TOTAL modeled     {:>8.3} ms/token  (~{:.1} t/s ceiling)", total, 1000.0/total);
        println!("matmul (qmatmul)  {:>8.3} ms/token  {:>5.1}%", matmul, 100.0*matmul/total);
        println!("non-matmul overhead {:>6.3} ms/token  {:>5.1}%", overhead, 100.0*overhead/total);
        println!("  norm            {:>8.3} ms  {:>5.1}%", t_norm, 100.0*t_norm/total);
        println!("  rope            {:>8.3} ms  {:>5.1}%", t_rope, 100.0*t_rope/total);
        println!("  sdpa            {:>8.3} ms  {:>5.1}%", t_sdpa, 100.0*t_sdpa/total);
        println!("  kv_write        {:>8.3} ms  {:>5.1}%", t_kvw, 100.0*t_kvw/total);
        println!("  kv_write_scatter {:>8.3} ms  {:>5.1}%", t_kvw_scatter, 100.0*t_kvw_scatter/total);
        println!("  slice           {:>8.3} ms  {:>5.1}%", t_slice, 100.0*t_slice/total);
        println!("  transpose       {:>8.3} ms  {:>5.1}%", t_tr, 100.0*t_tr/total);
        println!("  silu+mul        {:>8.3} ms  {:>5.1}%", t_silu, 100.0*t_silu/total);
        println!("  add             {:>8.3} ms  {:>5.1}%", t_add, 100.0*t_add/total);
    }

    /// Equivalence: slice_update (start/stop) vs scatter_single (indices) must
    /// write identical values into the cache at the same offset.
    #[test]
    #[ignore = "hardware equivalence check"]
    fn scatter_equals_slice_update() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() { eprintln!("skip: model not found {md}"); return; }
        let eng = Qwen3Engine::load(dir).expect("load");
        let ctx = &eng.ctx;
        let (nkv, hd) = (8i32, 128i32);
        let cap = 32i32;
        let off = 7i32;
        // Build deterministic non-zero cache [1,nkv,cap,hd] with known pattern.
        let cache_len = (1 * nkv * cap * hd) as usize;
        let mut cache_data: Vec<f32> = (0..cache_len).map(|i| (i as f32) * 0.5 + 1.0).collect();
        // Make update values clearly distinct (e.g. all 999.0) to detect misplacement.
        let upd_data: Vec<f32> = vec![999.0; (1 * nkv * 1 * hd) as usize];
        let cache_src = ctx.new_array(&cache_data).unwrap();
        let cache = ctx.reshape(cache_src, &[1, nkv, cap, hd]).unwrap();
        let upd_src = ctx.new_array(&upd_data).unwrap();
        let upd = ctx.reshape(upd_src, &[1, nkv, 1, hd]).unwrap();

        // slice_update path (original behavior).
        let via_slice = ctx.slice_update(cache, upd, &[0, 0, off, 0], &[1, nkv, off + 1, hd], &[1, 1, 1, 1]).unwrap();
        // scatter_single path: updates ndim = indices(1) + a(4) = 5.
        let idx = ctx.new_array_i32(&[off]).unwrap();
        let upd5 = ctx.reshape(upd, &[1, 1, nkv, 1, hd]).unwrap();
        let via_scatter = ctx.scatter_single(cache, idx, upd5, 2).unwrap();

        let a = ctx.to_vec_bf16_as_f32(via_slice).unwrap();
        let b = ctx.to_vec_bf16_as_f32(via_scatter).unwrap();
        assert_eq!(a.len(), b.len(), "length mismatch");
        let mut mismatches = 0usize;
        for i in 0..a.len() {
            if a[i] != b[i] { mismatches += 1; if mismatches <= 8 { println!("idx {i}: slice={} scatter={}", a[i], b[i]); } }
        }
        assert_eq!(mismatches, 0, "scatter_single != slice_update at {mismatches} elements of {}", a.len());
        let _ = cache_data; // keep borrow-checker happy if unused
        println!("scatter_single == slice_update bit-identical ({ } elements)", a.len());
    }
}
