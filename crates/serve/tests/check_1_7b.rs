//! Validation for Qwen3-1.7B (issue #24) — token-exact vs mlx_lm, bench, loader.
//! All tests are `#[ignore]` (require model + mlx feature + Apple GPU).

#[cfg(feature = "mlx")]
fn resolve_model_dir() -> Option<String> {
    if let Ok(m) = std::env::var("ENGINE_MLX_MODEL") {
        if std::path::Path::new(&m).exists() {
            return Some(m);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    for suffix in [
        "models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        "models/lmstudio-community/Qwen3-1.7B-MLX-4bit",
        "models/mlx-community/Qwen3-1.7B-4bit",
        "models/mlx-community/Qwen3-1.7B-8bit",
    ] {
        let p = format!("{home}/{suffix}");
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(feature = "mlx")]
fn mlx_lm_generate(model_dir: &str, prompt: &str, max_tokens: usize) -> Option<String> {
    let out = std::process::Command::new("python3")
        .args([
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            model_dir,
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens.to_string(),
            "--temp",
            "0",
            "--ignore-chat-template",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("mlx_lm stderr: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    // output is "==========\n<text>\n==========\nPrompt: ...\nGeneration: ..."
    let parts: Vec<&str> = stdout.split("==========").collect();
    if parts.len() >= 2 {
        Some(parts[1].trim().to_string())
    } else {
        // fallback: first non-empty line before "Prompt:"
        for line in stdout.lines() {
            if line.contains("Prompt:") {
                break;
            }
            if !line.trim().is_empty() && line.trim() != "==========" {
                return Some(line.trim().to_string());
            }
        }
        None
    }
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "1.7B validation — load"]
fn check_1_7b_load() {
    let md = match resolve_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("skip: no 1.7B model found (checked ENGINE_MLX_MODEL and ~/models)");
            return;
        }
    };
    let dir = std::path::Path::new(&md);
    println!("checking model dir: {md}");
    let init = engine_mlx_serve::loader::load_engine(dir).expect("load 1.7B");
    println!(
        "1.7B config: hidden={} layers={} heads={}/kv={} vocab={} bits={:?} hd={} intermediate={}",
        init.config.hidden_size,
        init.config.num_hidden_layers,
        init.config.num_attention_heads,
        init.config.num_key_value_heads,
        init.config.vocab_size,
        init.config.quantization,
        init.config.head_dim(),
        init.config.intermediate_size
    );
    // Generic asserts per issue #24 — introspected from manifest/config.json
    assert_eq!(init.config.hidden_size, 2048, "hidden_size should be 2048 for 1.7B");
    assert_eq!(init.config.num_hidden_layers, 28, "layers should be 28");
    assert_eq!(init.config.num_attention_heads, 16);
    assert_eq!(init.config.num_key_value_heads, 8);
    assert_eq!(init.config.head_dim(), 128);
    assert_eq!(init.config.intermediate_size, 6144);
    assert_eq!(init.config.vocab_size, 151936);
    assert_eq!(init.weights.layers.len(), 28);
    // quantization generic (4 or 8 bits, group 64)
    if let Some(q) = &init.config.quantization {
        assert_eq!(q.group_size, 64, "group_size should be 64");
        assert!(q.bits == 4 || q.bits == 8, "bits should be 4 or 8, got {}", q.bits);
    }
    println!("1.7B load OK — wired limit headroom checked via engine.rs:70");
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "1.7B bench 10/128"]
fn bench_1_7b_matrix() {
    let md = match resolve_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("skip: no 1.7B model found");
            return;
        }
    };
    let dir = std::path::Path::new(&md);
    let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).expect("load");
    let prompt = "Hello, how are you?";
    let points = [10usize, 128usize];
    let results = engine_mlx_serve::bench::bench_matrix(&mut eng, prompt, &points).expect("bench_matrix");
    for r in &results {
        assert!(r.tps > 0.0, "tps should be >0 for {}", r.max_tokens);
        assert!(!r.generated_text.is_empty(), "empty generation for {}", r.max_tokens);
        println!(
            "1.7B bench n={} tps={:.1} elapsed={}ms text={:?}",
            r.max_tokens,
            r.tps,
            r.elapsed_ms,
            &r.generated_text.chars().take(80).collect::<String>()
        );
    }
    // Ensure not OOM — if we got here, wired limit handled it
    println!("1.7B bench OK — atteso ~30-40 t/s eager, 60-80 compiled (vs 0.6B 64 t/s)");
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "1.7B bench 10 (legacy alias)"]
fn bench_1_7b_10() {
    let md = match resolve_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("skip no model");
            return;
        }
    };
    let dir = std::path::Path::new(&md);
    let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).expect("load");
    let prompt = "Hello, how are you?";
    let t0 = std::time::Instant::now();
    let text = eng.generate(prompt, 10).expect("generate");
    let ms = t0.elapsed().as_millis();
    println!("1.7B generate 10 tok => {text:?} in {ms}ms");
    assert!(!text.is_empty());
}

#[cfg(feature = "mlx")]
#[test]
#[ignore = "1.7B token-exact vs mlx_lm (greedy 10 tok)"]
fn token_exact_10_vs_mlx_lm() {
    let md = match resolve_model_dir() {
        Some(p) => p,
        None => {
            eprintln!("skip: no 1.7B model found");
            return;
        }
    };
    let dir = std::path::Path::new(&md);
    println!("token-exact check model_dir={md}");

    // Quick check mlx_lm available
    let mlx_check = std::process::Command::new("python3")
        .args(["-m", "mlx_lm", "--help"])
        .output();
    if mlx_check.is_err() || !mlx_check.unwrap().status.success() {
        eprintln!("skip: mlx_lm not installed");
        return;
    }

    let prompt = "Hello, how are you?";
    let max_tokens = 10;

    let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).expect("load engine for token-exact");
    // Ensure greedy
    eng.set_sampling(engine_mlx_serve::sampler::SamplingParams::greedy());
    let t0 = std::time::Instant::now();
    let eng_text = eng.generate(prompt, max_tokens).expect("engine generate");
    let eng_ms = t0.elapsed().as_millis();
    let eng_ids = eng
        .get_tokenizer()
        .encode(&eng_text)
        .expect("encode engine text");
    println!("engine ({eng_ms}ms) text={eng_text:?} ids={eng_ids:?}");

    let mlx_text = match mlx_lm_generate(&md, prompt, max_tokens) {
        Some(t) => t,
        None => {
            eprintln!("skip: mlx_lm generate failed or parse error");
            return;
        }
    };
    // Encode mlx text with same tokenizer for token-exact comparison
    let mlx_ids = eng
        .get_tokenizer()
        .encode(&mlx_text)
        .expect("encode mlx text");
    println!("mlx_lm text={mlx_text:?} ids={mlx_ids:?}");

    // Token-exact greedy — must match (allow leading-space tokenizer artefact on first token).
    // Engine decodes with leading space (token 358 " I") while mlx_lm stdout strips it (token 40 "I"),
    // but trimmed text and remaining ids are the real validation.
    let eng_trim = eng_text.trim();
    let mlx_trim = mlx_text.trim();
    if eng_trim != mlx_trim {
        eprintln!("MISMATCH trimmed text:");
        eprintln!(" eng: {eng_trim:?} ids={eng_ids:?}");
        eprintln!(" mlx: {mlx_trim:?} ids={mlx_ids:?}");
    }
    assert_eq!(
        eng_trim, mlx_trim,
        "token-exact failed: trimmed text differs — eng={eng_text:?} mlx={mlx_text:?}"
    );
    // Also verify suffix ids equal (skip first token which carries the leading-space difference)
    if eng_ids.len() == mlx_ids.len() && eng_ids.len() > 1 {
        assert_eq!(
            &eng_ids[1..],
            &mlx_ids[1..],
            "suffix token mismatch after first token — eng_ids={eng_ids:?} mlx_ids={mlx_ids:?}"
        );
    }
    println!("token-exact PASSED for 10 tok greedy — text={eng_trim:?}");
}
