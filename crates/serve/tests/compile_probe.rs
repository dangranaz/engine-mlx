//! Probe which ops survive mlx compile_shapeless — ignored, run with --ignored --nocapture
#[cfg(feature = "mlx")]
mod probe {
    use engine_mlx_ffi::{mlx_array, MlxCtx};

    fn ctx() -> (MlxCtx, engine_mlx_ffi::mlx_stream) {
        let stream = unsafe { engine_mlx_ffi::mlx_default_gpu_stream_new() };
        (MlxCtx::new(stream), stream)
    }

    unsafe extern "C" fn trace_rms(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        // tiny graph: rms_norm with ones weight + argmax
        let w = match ctx.full_f32(&[8], 1.0) {
            Ok(w) => w,
            Err(_) => return -2,
        };
        let y = match ctx.rms_norm(x, w, 1e-6) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("rms trace FAILED: {:?}", e);
                return -2;
            }
        };
        let t = match ctx.argmax(y) {
            Ok(t) => t,
            Err(_) => return -3,
        };
        let v = vec![t];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_concat(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut a: mlx_array = std::mem::zeroed();
        let mut b: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut a, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut b, inputs, 1);
        let c = match ctx.concatenate(a, b, 2) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("concat trace FAILED: {:?}", e);
                return -2;
            }
        };
        let v = vec![c];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_rms_argmax() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_rms),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.full_f32(&[1, 8], 0.5).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => {
                let v = ctx.to_vec_u32(out[0]).unwrap();
                println!("rms+argmax compiled OK -> {:?}", v);
            }
            Err(e) => println!("rms+argmax compiled APPLY FAILED: {:?}", e),
        }
    }

    struct QW {
        w: mlx_array,
        s: mlx_array,
        b: mlx_array,
    }
    unsafe impl Send for QW {}
    unsafe impl Sync for QW {}

    unsafe extern "C" fn trace_qmatmul(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let qw = &*(payload as *const QW);
        let stream = engine_mlx_ffi::mlx_default_gpu_stream_new();
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let y = match ctx.quantized_matmul(x, qw.w, qw.s, qw.b, true, 64, 4) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("qmatmul trace FAILED: {:?}", e);
                return -2;
            }
        };
        let v = vec![y];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_rope(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        let mut off: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut off, inputs, 1);
        let y = match ctx.rope_dynamic(x, 128, 1000000.0, off) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("rope trace FAILED: {:?}", e);
                return -2;
            }
        };
        let v = vec![y];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_qmatmul() {
        let (ctx, stream) = ctx();
        let _ = stream;
        let s2 = unsafe { engine_mlx_ffi::mlx_default_gpu_stream_new() };
        let c2 = MlxCtx::new(s2);
        // synthetic qkv weights: out 4096, in 1024, group 64
        let w = c2.zeros_u32(&[4096, 128]).unwrap();
        let s = c2.zeros_bf16(&[4096, 16]).unwrap();
        let b = c2.zeros_bf16(&[4096, 16]).unwrap();
        let payload = Box::into_raw(Box::new(QW { w, s, b }));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_qmatmul),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 1024]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("qmatmul compiled OK shape {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("qmatmul compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_rope() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_rope),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 16, 1, 128]).unwrap();
        let off = ctx.new_array_i32(&[6]).unwrap();
        match cl.apply(&[x, off]) {
            Ok(out) => println!("rope compiled OK shape {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("rope compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_split(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let parts = match ctx.split_sections(x, &[2048, 3072], 2) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("split trace FAILED: {:?}", e);
                return -2;
            }
        };
        if parts.len() != 3 {
            eprintln!("split trace wrong count {}", parts.len());
            return -3;
        }
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(parts.as_ptr(), parts.len());
        0
    }

    unsafe extern "C" fn trace_sdpa(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut q: mlx_array = std::mem::zeroed();
        let mut k: mlx_array = std::mem::zeroed();
        let mut v: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut q, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut k, inputs, 1);
        engine_mlx_ffi::mlx_vector_array_get(&mut v, inputs, 2);
        let y = match ctx.sdpa(q, k, v, 0.088, true) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("sdpa trace FAILED: {:?}", e);
                return -2;
            }
        };
        let v = vec![y];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_dup(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
    ) -> i32 {
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let v = vec![x, x];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_dup3(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
    ) -> i32 {
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let v = vec![x, x, x];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_split_add(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let parts = match ctx.split_sections(x, &[2048, 3072], 2) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("split_add trace FAILED: {:?}", e);
                return -2;
            }
        };
        // consume splits: add each part to itself (same shape) then add results needs same shape — instead return first part only
        let t = match ctx.add(parts[0], parts[0]) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("split_add add FAILED: {:?}", e);
                return -3;
            }
        };
        let v = vec![t];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_dup() {
        let (ctx, _stream) = ctx();
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func(Some(trace_dup))
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 8]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("dup compiled OK n={}", out.len()),
            Err(e) => println!("dup compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_slices(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let q = match ctx.slice(x, &[0, 0, 0], &[1, 1, 2048], &[1, 1, 1]) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("slices q FAILED: {:?}", e);
                return -2;
            }
        };
        let k = match ctx.slice(x, &[0, 0, 2048], &[1, 1, 3072], &[1, 1, 1]) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("slices k FAILED: {:?}", e);
                return -3;
            }
        };
        let v = match ctx.slice(x, &[0, 0, 3072], &[1, 1, 4096], &[1, 1, 1]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("slices v FAILED: {:?}", e);
                return -4;
            }
        };
        let vout = vec![q, k, v];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_slices() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_slices),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!(
                "slices compiled OK {:?} {:?} {:?}",
                ctx.shape(out[0]).unwrap(),
                ctx.shape(out[1]).unwrap(),
                ctx.shape(out[2]).unwrap()
            ),
            Err(e) => println!("slices compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_slice_copy(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let q = match ctx.slice(x, &[0, 0, 0], &[1, 1, 2048], &[1, 1, 1]) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("slice_copy slice FAILED: {:?}", e);
                return -2;
            }
        };
        let q = match ctx.materialize(q) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("slice_copy materialize FAILED: {:?}", e);
                return -3;
            }
        };
        let vout = vec![q];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_slice_copy() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_slice_copy),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("slice_copy compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("slice_copy compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_add_slice(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let y = match ctx.add(x, x) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("add_slice add FAILED: {:?}", e);
                return -2;
            }
        };
        let q = match ctx.slice(y, &[0, 0, 0], &[1, 1, 2048], &[1, 1, 1]) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("add_slice slice FAILED: {:?}", e);
                return -3;
            }
        };
        let vout = vec![q];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_add_slice() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_add_slice),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("add_slice compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("add_slice compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_reshape_transpose(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        // fresh compute then reshape (no slice involved)
        let y = match ctx.add(x, x) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("rt add FAILED: {:?}", e);
                return -2;
            }
        };
        let r = match ctx.reshape(y, &[1, 1, 16, 128]) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rt reshape FAILED: {:?}", e);
                return -3;
            }
        };
        let t = match ctx.transpose_axes(r, &[0, 2, 1, 3]) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("rt transpose FAILED: {:?}", e);
                return -4;
            }
        };
        let vout = vec![t];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_reshape_transpose() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_reshape_transpose),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 2048]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("reshape_transpose compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("reshape_transpose compiled APPLY FAILED: {:?}", e),
        }
    }

    struct HalfW {
        qw: mlx_array,
        qs: mlx_array,
        qb: mlx_array,
        norm: mlx_array,
    }
    unsafe impl Send for HalfW {}
    unsafe impl Sync for HalfW {}

    // Mirror of build_step first half: rms -> qmatmul -> reshape -> transpose -> rope -> concat -> sdpa
    unsafe extern "C" fn trace_half(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let hw = &*(payload as *const HalfW);
        let stream = engine_mlx_ffi::mlx_default_gpu_stream_new();
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        let mut kc: mlx_array = std::mem::zeroed();
        let mut vc: mlx_array = std::mem::zeroed();
        let mut off: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut kc, inputs, 1);
        engine_mlx_ffi::mlx_vector_array_get(&mut vc, inputs, 2);
        engine_mlx_ffi::mlx_vector_array_get(&mut off, inputs, 3);
        macro_rules! ck {
            ($e:expr, $n:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("half {} FAILED: {:?}", $n, e);
                        return -2;
                    }
                }
            };
        }
        let normed = ck!(ctx.rms_norm(x, hw.norm, 1e-6), "rms");
        let q = ck!(ctx.quantized_matmul(normed, hw.qw, hw.qs, hw.qb, true, 64, 4), "qmatmul");
        let q = ck!(ctx.reshape(q, &[1, 1, 8, 128]), "reshape");
        let q = ck!(ctx.transpose_axes(q, &[0, 2, 1, 3]), "transpose");
        let q = ck!(ctx.rope_dynamic(q, 128, 1000000.0, off), "rope");
        // k/v via reshape+transpose+rope of second/third matmuls (unfused, no split)
        let k = ck!(ctx.quantized_matmul(normed, hw.qw, hw.qs, hw.qb, true, 64, 4), "kmatmul");
        let k = ck!(ctx.reshape(k, &[1, 1, 8, 128]), "kreshape");
        let k = ck!(ctx.transpose_axes(k, &[0, 2, 1, 3]), "ktranspose");
        let k = ck!(ctx.rope_dynamic(k, 128, 1000000.0, off), "krope");
        let v = ck!(ctx.quantized_matmul(normed, hw.qw, hw.qs, hw.qb, true, 64, 4), "vmatmul");
        let v = ck!(ctx.reshape(v, &[1, 1, 8, 128]), "vreshape");
        let v = ck!(ctx.transpose_axes(v, &[0, 2, 1, 3]), "vtranspose");
        let kf = ck!(ctx.concatenate(kc, k, 2), "kconcat");
        let vf = ck!(ctx.concatenate(vc, v, 2), "vconcat");
        let a = ck!(ctx.sdpa(q, kf, vf, 0.088, true), "sdpa");
        let a = ck!(ctx.transpose_axes(a, &[0, 2, 1, 3]), "atranspose");
        let a = ck!(ctx.reshape(a, &[1, 1, 1024]), "areshape");
        let o = ck!(ctx.quantized_matmul(a, hw.qw, hw.qs, hw.qb, true, 64, 4), "omatmul");
        let xr = ck!(ctx.add(x, o), "add");
        let n2 = ck!(ctx.rms_norm(xr, hw.norm, 1e-6), "rms2");
        let g = ck!(ctx.quantized_matmul(n2, hw.qw, hw.qs, hw.qb, true, 64, 4), "gmatmul");
        let u = ck!(ctx.quantized_matmul(n2, hw.qw, hw.qs, hw.qb, true, 64, 4), "umatmul");
        let h = ck!(ctx.multiply(ck!(ctx.silu(g), "silu"), u), "mul");
        let d = ck!(ctx.quantized_matmul(h, hw.qw, hw.qs, hw.qb, true, 64, 4), "dmatmul");
        let xr2 = ck!(ctx.add(xr, d), "add2");
        let vout = vec![xr2, kf, vf];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_half() {
        let (ctx, _s) = ctx();
        let s2 = unsafe { engine_mlx_ffi::mlx_default_gpu_stream_new() };
        let c2 = MlxCtx::new(s2);
        let qw = c2.zeros_u32(&[1024, 128]).unwrap();
        let qs = c2.zeros_bf16(&[1024, 16]).unwrap();
        let qb = c2.zeros_bf16(&[1024, 16]).unwrap();
        let norm = c2.full_f32(&[1024], 1.0).unwrap();
        let payload = Box::into_raw(Box::new(HalfW { qw, qs, qb, norm }));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_half),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 1024]).unwrap();
        let kc = ctx.zeros(&[1, 8, 6, 128]).unwrap();
        let vc = ctx.zeros(&[1, 8, 6, 128]).unwrap();
        let off = ctx.new_array_i32(&[6]).unwrap();
        match cl.apply(&[x, kc, vc, off]) {
            Ok(out) => println!("half compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("half compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_dup59(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
    ) -> i32 {
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let v = vec![x; 59];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    unsafe extern "C" fn trace_bigargmax(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        let t = match ctx.argmax(x) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("bigargmax FAILED: {:?}", e);
                return -2;
            }
        };
        let v = vec![t];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(v.as_ptr(), v.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_dup59() {
        let (ctx, _stream) = ctx();
        let raw = unsafe { engine_mlx_ffi::mlx_closure_new_func(Some(trace_dup59)) };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 8]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("dup59 compiled OK n={}", out.len()),
            Err(e) => println!("dup59 compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_bigargmax() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_bigargmax),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 151936]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => {
                let v = ctx.to_vec_u32(out[0]).unwrap();
                println!("bigargmax compiled OK {:?}", v);
            }
            Err(e) => println!("bigargmax compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_sud(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut cache: mlx_array = std::mem::zeroed();
        let mut upd: mlx_array = std::mem::zeroed();
        let mut start: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut cache, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut upd, inputs, 1);
        engine_mlx_ffi::mlx_vector_array_get(&mut start, inputs, 2);
        let y = match ctx.slice_update_dynamic(cache, upd, start, &[2]) {
            Ok(y) => y,
            Err(e) => {
                eprintln!("sud trace FAILED: {:?}", e);
                return -2;
            }
        };
        let vout = vec![y];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_slice_update_dynamic() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_sud),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let cache = ctx.zeros(&[1, 2, 256, 4]).unwrap();
        let upd = ctx.full_f32(&[1, 2, 1, 4], 0.5).unwrap();
        let start = ctx.new_array_i32(&[6]).unwrap();
        match cl.apply(&[cache, upd, start]) {
            Ok(out) => println!("sud compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("sud compiled APPLY FAILED: {:?}", e),
        }
        // second apply at different offset (static KV steady state)
        let cache2 = ctx.zeros(&[1, 2, 256, 4]).unwrap();
        let start2 = ctx.new_array_i32(&[7]).unwrap();
        match cl.apply(&[cache2, upd, start2]) {
            Ok(out) => println!("sud 2nd compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("sud 2nd compiled APPLY FAILED: {:?}", e),
        }
    }

    unsafe extern "C" fn trace_sdyn(
        outputs: *mut engine_mlx_ffi::mlx_vector_array,
        inputs: engine_mlx_ffi::mlx_vector_array,
        payload: *mut std::ffi::c_void,
    ) -> i32 {
        let stream = *(payload as *const engine_mlx_ffi::mlx_stream);
        let ctx = MlxCtx::new(stream);
        let mut x: mlx_array = std::mem::zeroed();
        let mut st: mlx_array = std::mem::zeroed();
        engine_mlx_ffi::mlx_vector_array_get(&mut x, inputs, 0);
        engine_mlx_ffi::mlx_vector_array_get(&mut st, inputs, 1);
        let q = match ctx.slice_dynamic(x, st, &[2], &[1, 1, 2048]) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("sdyn FAILED: {:?}", e);
                return -2;
            }
        };
        let vout = vec![q];
        *outputs = engine_mlx_ffi::mlx_vector_array_new_data(vout.as_ptr(), vout.len());
        0
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_slice_dynamic() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_sdyn),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        let st = ctx.new_array_i32(&[0]).unwrap();
        match cl.apply(&[x, st]) {
            Ok(out) => println!("sdyn compiled OK {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("sdyn compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_dup3() {
        let (ctx, _stream) = ctx();
        let raw = unsafe { engine_mlx_ffi::mlx_closure_new_func(Some(trace_dup3)) };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 8]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("dup3 compiled OK n={}", out.len()),
            Err(e) => println!("dup3 compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_split_add() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_split_add),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!("split_add compiled OK shape {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("split_add compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_split() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_split),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let x = ctx.zeros(&[1, 1, 4096]).unwrap();
        match cl.apply(&[x]) {
            Ok(out) => println!(
                "split compiled OK {:?} {:?} {:?}",
                ctx.shape(out[0]).unwrap(),
                ctx.shape(out[1]).unwrap(),
                ctx.shape(out[2]).unwrap()
            ),
            Err(e) => println!("split compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_sdpa() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_sdpa),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let q = ctx.zeros(&[1, 16, 1, 128]).unwrap();
        let k = ctx.zeros(&[1, 8, 6, 128]).unwrap();
        let v = ctx.zeros(&[1, 8, 6, 128]).unwrap();
        match cl.apply(&[q, k, v]) {
            Ok(out) => println!("sdpa compiled OK shape {:?}", ctx.shape(out[0]).unwrap()),
            Err(e) => println!("sdpa compiled APPLY FAILED: {:?}", e),
        }
    }

    #[test]
    #[ignore = "compile probe"]
    fn probe_concat() {
        let (ctx, stream) = ctx();
        let payload = Box::into_raw(Box::new(stream));
        let raw = unsafe {
            engine_mlx_ffi::mlx_closure_new_func_payload(
                Some(trace_concat),
                payload as *mut std::ffi::c_void,
                None,
            )
        };
        let cl = engine_mlx_ffi::compile::Closure::from_raw(raw);
        let cl = engine_mlx_ffi::compile::compile_shapeless(cl).expect("wrap");
        let a = ctx.full_f32(&[1, 2, 6, 4], 0.5).unwrap();
        let b = ctx.full_f32(&[1, 2, 1, 4], 0.5).unwrap();
        match cl.apply(&[a, b]) {
            Ok(out) => {
                println!("concat compiled OK shape {:?}", ctx.shape(out[0]).unwrap());
            }
            Err(e) => println!("concat compiled APPLY FAILED: {:?}", e),
        }
        // second apply with grown cache (simulates next decode step)
        let a2 = ctx.full_f32(&[1, 2, 7, 4], 0.5).unwrap();
        let b2 = ctx.full_f32(&[1, 2, 1, 4], 0.5).unwrap();
        match cl.apply(&[a2, b2]) {
            Ok(out) => {
                println!("concat 2nd compiled OK shape {:?}", ctx.shape(out[0]).unwrap());
            }
            Err(e) => println!("concat 2nd compiled APPLY FAILED: {:?}", e),
        }
    }
}
