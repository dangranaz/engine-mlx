//! Debug int4 vs bf16 token divergence — ignored by default, run with --ignored --nocapture
#[cfg(feature = "mlx")]
mod int4_debug {
    use std::path::Path;

    fn model_dir_06b() -> String {
        std::env::var("HOME").map(|h| format!("{h}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit")).unwrap()
    }

    #[test]
    #[ignore = "int4 debug requires mlx"]
    fn bf16_vs_int4_first_decode() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { eprintln!("skip no model"); return; }
        // BF16 path (default)
        let mut eng_bf16 = engine_mlx_serve::Qwen3Engine::load(dir).expect("load bf16");
        let prompt = "Hello, how are you?";
        let tokens = eng_bf16.get_tokenizer().encode(prompt).unwrap();
        println!("prompt tokens {:?} len {}", tokens, tokens.len());
        let x = eng_bf16.embed(&tokens).unwrap();
        let logits = eng_bf16.prefill(x, tokens.len()).unwrap();
        let first = eng_bf16.ctx.argmax(logits).unwrap();
        let first_vals = eng_bf16.ctx.to_vec_u32(first).unwrap();
        println!("bf16 prefill first token {} -> {:?}", first_vals[0], eng_bf16.get_tokenizer().decode(&first_vals).unwrap());
        eng_bf16.offset = tokens.len();
        let next_bf16 = eng_bf16.decode_step(first_vals[0]).expect("bf16 decode");
        println!("bf16 next token {} -> {:?}", next_bf16, eng_bf16.get_tokenizer().decode(&[next_bf16]).unwrap());

        // Int4 path (convert KV buffers directly — env var no longer used)
        let mut eng_int4 = engine_mlx_serve::Qwen3Engine::load(dir).expect("load int4");
        let x2 = eng_int4.embed(&tokens).unwrap();
        let logits2 = eng_int4.prefill(x2, tokens.len()).unwrap();
        let first2 = eng_int4.ctx.argmax(logits2).unwrap();
        let first2_vals = eng_int4.ctx.to_vec_u32(first2).unwrap();
        println!("int4 prefill first token {} -> {:?}", first2_vals[0], eng_int4.get_tokenizer().decode(&first2_vals).unwrap());
        // Convert to int4
        let cap = std::cmp::max(tokens.len() + 10 + 16, 256);
        eng_int4.convert_kv_to_int4(tokens.len(), cap).expect("convert");
        eng_int4.offset = tokens.len();
        // Ensure compiled is for int4 (will recompile)
        let _ = eng_int4.ensure_compiled();
        println!("bf16 compiled={} int4 compiled={}", eng_bf16.is_compiled(), eng_int4.is_compiled());
        let next_int4 = eng_int4.decode_step(first2_vals[0]).expect("int4 decode");
        println!("after decode bf16 compiled={} int4 compiled={}", eng_bf16.is_compiled(), eng_int4.is_compiled());
        println!("int4 next token {} -> {:?}", next_int4, eng_int4.get_tokenizer().decode(&[next_int4]).unwrap());
        println!("bf16 next {} vs int4 next {} -> {}", next_bf16, next_int4, if next_bf16==next_int4 {"MATCH"} else {"MISMATCH"});
        // Also test second decode
        let next2_bf16 = eng_bf16.decode_step(next_bf16).unwrap();
        let next2_int4 = eng_int4.decode_step(next_int4).unwrap();
        println!("bf16 2nd {} vs int4 2nd {} -> {}", next2_bf16, next2_int4, if next2_bf16==next2_int4 {"MATCH"} else {"MISMATCH"});
    }

    #[test]
    #[ignore = "dtype trace"]
    fn dtype_trace() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        let dt = |a: u32| match a {
            10 => "F32",
            12 => "BF16",
            9 => "F16",
            _ => "?",
        };
        let dq = |arr: engine_mlx_ffi::mlx_array| unsafe {
            engine_mlx_ffi::mlx_array_dtype(arr) as u32
        };
        println!("embed_table {:?}", eng.embed_table.map(dq).map(dt));
        println!("qkv_w {:?} rms_w {:?}", dq(eng.weights.layers[0].qkv_proj_w), dq(eng.weights.layers[0].input_layernorm));
        let tokens = eng.get_tokenizer().encode("Hello, how are you?").unwrap();
        let x = eng.embed(&tokens).unwrap();
        println!("embed out {:?}", dt(dq(x)));
        let n = eng.ctx.rms_norm(x, eng.weights.layers[0].input_layernorm, 1e-6).unwrap();
        println!("rms {:?}", dt(dq(n)));
        let lw = &eng.weights.layers[0];
        let qkv = eng.ctx.quantized_matmul(n, lw.qkv_proj_w, lw.qkv_proj_s, lw.qkv_proj_b, true, 64, 4).unwrap();
        println!("qkv {:?}", dt(dq(qkv)));
        let k = eng.ctx.slice(qkv, &[0, 0, 1024], &[1, 6, 2048], &[1, 1, 1]).unwrap();
        println!("k slice {:?}", dt(dq(k)));
        let k = eng.ctx.reshape(k, &[1, 6, 8, 128]).unwrap();
        println!("k reshape {:?}", dt(dq(k)));
        let k = eng.ctx.transpose_axes(k, &[0, 2, 1, 3]).unwrap();
        println!("k transpose {:?}", dt(dq(k)));
        let off = eng.ctx.new_array_i32(&[0]).unwrap();
        let k = eng.ctx.rope_dynamic(k, 128, 1000000.0, off).unwrap();
        println!("k rope {:?}", dt(dq(k)));
        println!("kv_bufs empty (prefill not run)");
        let _ = eng.prefill(eng.embed(&tokens).unwrap(), tokens.len()).unwrap();
        println!("kv0 {:?}", dt(dq(eng.kv_bufs[0])));
    }

    #[test]
    #[ignore = "decode-only tps"]
    fn decode_only_tps() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        let prompt = "Hello, how are you?";
        let tokens = eng.get_tokenizer().encode(prompt).unwrap();
        let x = eng.embed(&tokens).unwrap();
        let logits = eng.prefill(x, tokens.len()).unwrap();
        let first = eng.ctx.to_vec_u32(eng.ctx.argmax(logits).unwrap()).unwrap()[0];
        eng.offset = tokens.len();
        eng.ensure_compiled().unwrap();
        // warmup
        let mut cur = first;
        for _ in 0..3 {
            cur = eng.decode_step(cur).unwrap();
        }
        let n = 60;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            cur = eng.decode_step(cur).unwrap();
        }
        let ms = t0.elapsed().as_millis();
        println!(
            "decode-only compiled={} {}tok {}ms tps={:.1}",
            eng.is_compiled(),
            n,
            ms,
            n as f64 / (ms as f64 / 1000.0)
        );
    }

    #[test]
    #[ignore = "eager vs compiled ab"]
    fn eager_vs_compiled_ab() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let prompt = "Hello, how are you?";
        // Eager: fresh engine, never compile
        let mut eng_e = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        let t0 = std::time::Instant::now();
        let text_e = eng_e.generate(prompt, 30).unwrap();
        let ms_e = t0.elapsed().as_millis();
        println!("eager: compiled={} {}ms text={:?}", eng_e.is_compiled(), ms_e, &text_e.chars().take(40).collect::<String>());
        // Compiled: pre-compile (untimed) then generate
        let mut eng_c = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        let tokens = eng_c.get_tokenizer().encode(prompt).unwrap();
        let x = eng_c.embed(&tokens).unwrap();
        let _ = eng_c.prefill(x, tokens.len()).unwrap();
        eng_c.offset = tokens.len();
        eng_c.ensure_compiled().unwrap();
        eng_c.reset();
        let t1 = std::time::Instant::now();
        let text_c = eng_c.generate(prompt, 30).unwrap();
        let ms_c = t1.elapsed().as_millis();
        println!("compiled: compiled={} {}ms text={:?}", eng_c.is_compiled(), ms_c, &text_c.chars().take(40).collect::<String>());
        println!("match={}", text_e == text_c);
    }

    #[test]
    #[ignore = "bf16 compiled bench"]
    fn bf16_compiled_bench() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        for n in [10usize, 128usize] {
            let t0 = std::time::Instant::now();
            let text = eng.generate("Hello, how are you?", n).unwrap();
            let ms = t0.elapsed().as_millis();
            let ntok = eng.get_tokenizer().encode(&text).unwrap().len();
            println!(
                "n={} compiled={} {}ms tps={:.1} text={:?}",
                n,
                eng.is_compiled(),
                ms,
                ntok as f64 / (ms as f64 / 1000.0),
                &text.chars().take(50).collect::<String>()
            );
        }
    }

    #[test]
    #[ignore = "bf16 compiled status"]
    fn bf16_compiled_status() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        println!("before generate compiled={}", eng.is_compiled());
        // Manual prefill + ensure_compiled to surface error
        let tokens = eng.get_tokenizer().encode("Hello, how are you?").unwrap();
        let x = eng.embed(&tokens).unwrap();
        let _logits = eng.prefill(x, tokens.len()).unwrap();
        eng.offset = tokens.len();
        match eng.ensure_compiled() {
            Ok(_) => println!("ensure_compiled OK compiled={}", eng.is_compiled()),
            Err(e) => println!("ensure_compiled FAILED: {:?}", e),
        }
        let t0 = std::time::Instant::now();
        let text = eng.generate("Hello, how are you?", 10).unwrap();
        let ms = t0.elapsed().as_millis();
        println!("generate => {:?} in {}ms compiled={}", text, ms, eng.is_compiled());
    }

    #[test]
    #[ignore = "int4 quant roundtrip"]
    fn quant_roundtrip_error() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        // Create a dummy k of shape [1,8,1,128] with random values via arange
        let k = eng.ctx.zeros(&[1, 8, 1, 128]).unwrap();
        // Use arange to get non-zero
        let k = eng.ctx.full_f32(&[1,8,1,128], 0.5).unwrap();
        let (qd, qs, qb) = eng.ctx.quantize_kv_native(k, 64, 4).unwrap();
        let k_deq = eng.ctx.dequantize_kv_native(qd, qs, qb, 64, 4).unwrap();
        let orig = eng.ctx.to_vec_f32(k).unwrap();
        // dequant is BF16, read as bf16
        let deq = eng.ctx.to_vec_bf16_as_f32(k_deq).unwrap_or_else(|_| eng.ctx.to_vec_f32(k_deq).unwrap());
        let max_err = orig.iter().zip(deq.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
        println!("quant roundtrip max_err {} orig[0]={} deq[0]={}", max_err, orig[0], deq[0]);
        // Don't assert, just print
        println!("orig first 5 {:?}", &orig[..5]);
        println!("deq first 5 {:?}", &deq[..5]);
    }

    #[test]
    #[ignore = "int4 actual k error"]
    fn actual_k_quant_error() {
        let md = model_dir_06b();
        let dir = Path::new(&md);
        if !dir.exists() { return; }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).unwrap();
        let prompt = "Hello, how are you?";
        let tokens = eng.get_tokenizer().encode(prompt).unwrap();
        let x = eng.embed(&tokens).unwrap();
        let _logits = eng.prefill(x, tokens.len()).unwrap();
        // Get k for first layer at pos 0
        let k_bf16 = eng.kv_bufs[0]; // [1,8,6,128] for 0.6B
        // Slice first position
        let k_p = eng.ctx.slice(k_bf16, &[0,0,0,0], &[1,8,1,128], &[1,1,1,1]).unwrap();
        let k_p = eng.ctx.reshape(k_p, &[1,8,1,128]).unwrap();
        let orig = eng.ctx.to_vec_f32(k_p).unwrap();
        let (mut mn, mut mx, mut sum, mut cnt_nan) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0);
        for &v in &orig {
            if !v.is_finite() { cnt_nan += 1; continue; }
            mn = mn.min(v); mx = mx.max(v); sum += v as f64;
        }
        println!("k_p stats min {} max {} mean {} nonfinite {} len {}", mn, mx, sum/orig.len() as f64, cnt_nan, orig.len());
        // Also try bf16 reading
        let orig_bf16 = eng.ctx.to_vec_bf16_as_f32(k_p).unwrap_or_else(|_| orig.clone());
        println!("k_p orig f32 first 5 {:?}", &orig[..5]);
        println!("k_p orig bf16 first 5 {:?}", &orig_bf16[..5]);
        let k_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(k_p) } as u32;
        println!("k_p dtype {} shape {:?}", k_dtype, eng.ctx.shape(k_p).unwrap());
        let (qd, qs, qb) = eng.ctx.quantize_kv_native(k_p, 64, 4).unwrap();
        let qd_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(qd) } as u32;
        let qs_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(qs) } as u32;
        let qb_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(qb) } as u32;
        println!("qd dtype {} shape {:?} qs dtype {} qb dtype {}", qd_dtype, eng.ctx.shape(qd).unwrap(), qs_dtype, qb_dtype);
        // qs dtype 10=F32, read as f32 not bf16
        let qs_vals = if qs_dtype == 10 { eng.ctx.to_vec_f32(qs).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(qs).unwrap() };
        let qb_vals = if qb_dtype == 10 { eng.ctx.to_vec_f32(qb).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(qb).unwrap() };
        println!("qs first 5 {:?}", &qs_vals[..5.min(qs_vals.len())]);
        println!("qb first 5 {:?}", &qb_vals[..5.min(qb_vals.len())]);
        let k_deq = eng.ctx.dequantize_kv_native(qd, qs, qb, 64, 4).unwrap();
        let deq_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(k_deq) } as u32;
        println!("k_deq dtype {}", deq_dtype);
        let deq = if deq_dtype == 10 { eng.ctx.to_vec_f32(k_deq).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(k_deq).unwrap() };
        let max_err = orig.iter().zip(deq.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
        println!("actual k quant max_err {} orig0 {} deq0 {}", max_err, orig[0], deq[0]);
        println!("deq first 5 {:?}", &deq[..5.min(deq.len())]);
        // Isolate group 0 of head 0: hd[0..64]
        let g0 = eng.ctx.slice(k_p, &[0,0,0,0], &[1,1,1,64], &[1,1,1,1]).unwrap();
        let g0 = eng.ctx.reshape(g0, &[64]).unwrap();
        let g0v = eng.ctx.to_vec_f32(g0).unwrap();
        let (gmn, gmx) = g0v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a,b),&v| (a.min(v), b.max(v)));
        println!("g0 min {} max {} vals {:?}", gmn, gmx, &g0v[..10]);
        let (gqd, gqs, gqb) = eng.ctx.quantize_kv_native(g0, 64, 4).unwrap();
        println!("g0 qd shape {:?} qs shape {:?}", eng.ctx.shape(gqd).unwrap(), eng.ctx.shape(gqs).unwrap());
        let gqs_v = eng.ctx.to_vec_f32(gqs).unwrap();
        let gqb_v = eng.ctx.to_vec_f32(gqb).unwrap();
        println!("g0 qs {:?} qb {:?}", gqs_v, gqb_v);
        let gdeq = eng.ctx.dequantize_kv_native(gqd, gqs, gqb, 64, 4).unwrap();
        let gdeq_v = eng.ctx.to_vec_bf16_as_f32(gdeq).unwrap_or_else(|_| eng.ctx.to_vec_f32(gdeq).unwrap());
        let gerr = g0v.iter().zip(gdeq_v.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
        println!("g0 roundtrip max_err {} deq {:?}", gerr, &gdeq_v[..10]);
        // Try BF16 input path
        let k_bf16_cast = eng.ctx.astype(k_p, engine_mlx_ffi::mlx_dtype_::MLX_BFLOAT16).unwrap();
        let k_bf16_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(k_bf16_cast) } as u32;
        println!("k_bf16 dtype {}", k_bf16_dtype);
        let (qd2, qs2, qb2) = eng.ctx.quantize_kv_native(k_bf16_cast, 64, 4).unwrap();
        let qs2_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(qs2) } as u32;
        println!("qs2 dtype {} shape {:?}", qs2_dtype, eng.ctx.shape(qs2).unwrap());
        let qs2_vals = if qs2_dtype == 10 { eng.ctx.to_vec_f32(qs2).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(qs2).unwrap() };
        let qb2_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(qb2) } as u32;
        let qb2_vals = if qb2_dtype == 10 { eng.ctx.to_vec_f32(qb2).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(qb2).unwrap() };
        println!("qs2 first 5 {:?}", &qs2_vals[..5.min(qs2_vals.len())]);
        println!("qb2 first 5 {:?}", &qb2_vals[..5.min(qb2_vals.len())]);
        let k_deq2 = eng.ctx.dequantize_kv_native(qd2, qs2, qb2, 64, 4).unwrap();
        let deq2_dtype = unsafe { engine_mlx_ffi::mlx_array_dtype(k_deq2) } as u32;
        let deq2 = if deq2_dtype == 10 { eng.ctx.to_vec_f32(k_deq2).unwrap() } else { eng.ctx.to_vec_bf16_as_f32(k_deq2).unwrap() };
        let max_err2 = orig.iter().zip(deq2.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
        println!("bf16-input quant max_err {} deq2 first 5 {:?}", max_err2, &deq2[..5.min(deq2.len())]);
        // Check if error is large
        if max_err > 0.5 {
            println!("large error >0.5, int4 may be lossy");
        }
    }
}
