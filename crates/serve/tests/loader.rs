//! Unit tests for `engine_mlx_serve::loader` module.

use std::path::Path;

use engine_mlx_serve::loader::load_engine;

#[test]
fn load_engine_missing_dir_bails_not_panics() {
    let dir = Path::new("/tmp/nxm_loader_missing_test_no_such_dir_12345");
    let res = load_engine(dir);
    assert!(res.is_err(), "expected Err for missing dir");
    let msg = res.err().unwrap().to_string();
    assert!(
        msg.contains("failed to ensure_plan")
            || msg.contains("model.safetensors")
            || msg.contains("not found")
            || msg.contains("No such file"),
        "unexpected error: {msg}"
    );
}

#[test]
fn load_weights_missing_bails_not_panics() {
    let dir = std::env::temp_dir().join("nxm_loader_missing_weights_test");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("model_fused.safetensors"));
    let result = std::panic::catch_unwind(|| load_engine(&dir));
    assert!(result.is_ok(), "load_engine panicked instead of returning Err");
    assert!(result.unwrap().is_err(), "expected Err for missing weights");
}

#[cfg(feature = "mlx")]
mod mlx_tests {
    use super::*;
    use std::path::Path;

    fn model_dir() -> String {
        std::env::var("HOME")
            .map(|h| format!("{h}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit"))
            .unwrap_or_default()
    }

    #[test]
    fn load_qwen3_06b_from_lmstudio() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping: model dir not found");
            return;
        }
        let init = load_engine(dir).expect("load_engine failed for Qwen3-0.6B-MLX-4bit");
        let cfg = &init.config;
        assert_eq!(cfg.hidden_size, 1024, "hidden_size");
        assert_eq!(cfg.num_hidden_layers, 28, "num_hidden_layers");
        assert_eq!(cfg.num_attention_heads, 16, "num_attention_heads");
        assert_eq!(cfg.num_key_value_heads, 8, "num_key_value_heads");
        assert_eq!(cfg.head_dim(), 128, "head_dim");
        assert_eq!(cfg.intermediate_size, 3072, "intermediate_size");
        assert_eq!(cfg.vocab_size, 151936, "vocab_size");
        assert!(init.weights.lm_head_w.is_none(), "lm_head should be tied");
        assert_eq!(init.weights.layers.len(), 28, "layers count");
        let l0 = &init.weights.layers[0];
        assert!(!l0.input_layernorm.ctx.is_null(), "layer0 input_layernorm");
        assert!(!l0.qkv_proj_w.ctx.is_null(), "layer0 qkv_proj_w");
        assert!(!l0.o_proj_w.ctx.is_null(), "layer0 o_proj_w");
        assert!(!l0.gate_up_proj_w.ctx.is_null(), "layer0 gate_up_proj_w");
        assert!(!l0.down_proj_w.ctx.is_null(), "layer0 down_proj_w");
        assert!(l0.q_norm.is_some(), "layer0 q_norm present");
        assert!(l0.k_norm.is_some(), "layer0 k_norm present");
        println!("Qwen3-0.6B loaded OK: {cfg:?}");
    }

    #[test]
    fn forward_pass_produces_output() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping: model dir not found");
            return;
        }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir)
            .expect("Qwen3Engine::load failed");

        let prompt = "Hello, how are you?";
        let t0 = std::time::Instant::now();
        let result = eng.generate(prompt, 10).expect("generate failed");
        let elapsed_ms = t0.elapsed().as_millis();
        println!("generate({prompt:?}, 10) => {result:?} in {elapsed_ms}ms");
        assert!(!result.is_empty(), "generate produced empty output");
    }

    #[test]
    fn forward_pass_debug_tokens() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping: model dir not found");
            return;
        }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir)
            .expect("Qwen3Engine::load failed");

        let prompt = "Hello, how are you?";
        let prompt_tokens = eng.get_tokenizer().encode(prompt).expect("encode failed");
        println!("prompt tokens: {prompt_tokens:?}");

        // Embed
        let x = eng.embed(&prompt_tokens).expect("embed failed");
        println!("embed shape: {:?}", eng.ctx.shape(x).unwrap());

        // Prefill
        let logits = eng.prefill(x, prompt_tokens.len()).expect("prefill failed");
        println!("logits shape: {:?}", eng.ctx.shape(logits).unwrap());

        // Argmax first token
        let first_arr = eng.ctx.argmax(logits).expect("argmax failed");
        let first_vals = eng.ctx.to_vec_u32(first_arr).expect("to_vec_u32 failed");
        println!("first token: {:?}", first_vals);

        // Decode first token
        let first_text = eng.get_tokenizer().decode(&first_vals).expect("decode failed");
        println!("first token text: {:?}", first_text);

        // Generate a few steps
        eng.offset = prompt_tokens.len();
        let mut current = first_vals[0];
        for i in 0..5 {
            let next = eng.decode_step(current).expect("decode_step failed");
            println!("step {i}: token {next}");
            current = next;
        }
    }

    #[test]
    fn forward_via_mlx_debug() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping: model dir not found");
            return;
        }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir)
            .expect("Qwen3Engine::load failed");

        let prompt = "Hello, how are you?";
        let prompt_tokens = eng.get_tokenizer().encode(prompt).expect("encode failed");
        println!("prompt tokens: {prompt_tokens:?}");

        // Call generate_via_mlx directly to see what tokens it produces
        let tokens = prompt_tokens.clone();
        let max_tokens = 10;
        
        // Note: kv_bufs is private, but generate_via_mlx clears it internally
        // We just need to reset offset
        eng.offset = 0;
        let x = eng.embed(&tokens).expect("embed failed");
        let logits = eng.prefill(x, tokens.len()).expect("prefill failed");
        
        let first_arr = eng.ctx.argmax(logits).expect("argmax failed");
        let first_vals = eng.ctx.to_vec_u32(first_arr).expect("to_vec_u32 failed");
        let mut current = first_vals[0];
        eng.offset = tokens.len();
        let mut generated = vec![current];
        println!("first token: {current} -> {:?}", eng.get_tokenizer().decode(&[current]).unwrap());

        // Try compile
        let _ = eng.ensure_compiled();
        
        for i in 1..max_tokens {
            let next = eng.decode_step(current).expect("decode_step failed");
            println!("step {i}: token {next} -> {:?}", eng.get_tokenizer().decode(&[next]).unwrap());
            if engine_mlx_serve::config::EOS_IDS.contains(&next) { break; }
            generated.push(next);
            current = next;
            if generated.len() >= max_tokens { break; }
        }
        
        let text = eng.get_tokenizer().decode(&generated).expect("decode failed");
        println!("final text: {text:?}");
    }
}
