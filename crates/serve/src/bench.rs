//! Bench harness — prefill/decode t/s at 0/128/512/2048 vs mlx_lm.
//!
//! Run: `cargo test -p engine-mlx-serve --features mlx -- bench --ignored --nocapture`
//! Compares eager 23 t/s (now) vs mlx_lm market standard when metal unavailable.

use anyhow::{Context, Result};

use crate::engine::Qwen3Engine;

/// Result for a single bench point.
#[derive(Debug)]
pub struct BenchResult {
    pub prompt: String,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub generated_text: String,
    pub generated_tokens: usize,
    pub elapsed_ms: u128,
    pub tps: f64,
}

/// Run a single bench point: generate `max_tokens` from `prompt` and measure.
pub fn bench_once(engine: &mut Qwen3Engine, prompt: &str, max_tokens: usize) -> Result<BenchResult> {
    tracing::info!(
        target: "engine_mlx::bench",
        "bench_once START prompt_len={} max_tokens={}",
        prompt.len(),
        max_tokens
    );
    engine.reset();
    let prompt_tokens = engine
        .get_tokenizer()
        .encode(prompt)
        .context("encode prompt for bench")?
        .len();
    let t0 = std::time::Instant::now();
    let text = engine.generate(prompt, max_tokens).context("generate failed")?;
    let elapsed_ms = t0.elapsed().as_millis();
    let generated_tokens = engine
        .get_tokenizer()
        .encode(&text)
        .map(|v| v.len())
        .unwrap_or(max_tokens);
    let tps = if elapsed_ms > 0 {
        generated_tokens as f64 / (elapsed_ms as f64 / 1000.0)
    } else {
        0.0
    };
    tracing::info!(
        target: "engine_mlx::bench",
        "bench_once END max_tokens={} elapsed_ms={} tps={:.1}",
        max_tokens,
        elapsed_ms,
        tps
    );
    Ok(BenchResult {
        prompt: prompt.to_string(),
        prompt_tokens,
        max_tokens,
        generated_text: text,
        generated_tokens,
        elapsed_ms,
        tps,
    })
}

/// Run matrix 10/128/512 (2048 optional, slow eager) and print table.
/// Returns results for caller to assert or compare vs mlx_lm.
pub fn bench_matrix(engine: &mut Qwen3Engine, prompt: &str, points: &[usize]) -> Result<Vec<BenchResult>> {
    let mut out = Vec::with_capacity(points.len());
    for &n in points {
        let r = bench_once(engine, prompt, n)?;
        println!(
            "bench n={:4} prompt_tok={:3} gen_tok={:3} elapsed={:5}ms tps={:5.1} preview={:?}",
            r.max_tokens,
            r.prompt_tokens,
            r.generated_tokens,
            r.elapsed_ms,
            r.tps,
            &r.generated_text.chars().take(60).collect::<String>()
        );
        out.push(r);
    }
    let mode = if engine.is_compiled_static() {
        "static"
    } else if engine.is_compiled() {
        "compiled"
    } else {
        "eager"
    };
    println!("=== Bench matrix (engine-mlx {mode}) ===");
    for r in &out {
        println!("{:4} tok => {:5.1} t/s ({}ms)", r.max_tokens, r.tps, r.elapsed_ms);
    }
    // Compare vs mlx_lm if available (best-effort, no fail if missing)
    if let Ok(mlx_tps) = bench_mlx_lm(prompt, points[0]) {
        println!("mlx_lm reference (first point): {mlx_tps:.1} t/s — market standard");
        if let Some(first) = out.first() {
            let delta = (first.tps - mlx_tps) / mlx_tps * 100.0;
            println!("engine-mlx vs mlx_lm delta: {delta:+.1}%");
        }
    } else {
        println!("mlx_lm not available (python -m mlx_lm not found), skipping compare");
    }
    Ok(out)
}

/// Resolve model dir respecting ENGINE_MLX_MODEL env, then local 0.6B fallback.
/// Supports both Qwen3-0.6B and Qwen3-1.7B via `ENGINE_MLX_MODEL` (issue #24).
pub fn resolve_model_dir() -> String {
    if let Ok(m) = std::env::var("ENGINE_MLX_MODEL") {
        if std::path::Path::new(&m).exists() {
            return m;
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    // Default to 0.6B (dev model) for backward compat; 1.7B via ENGINE_MLX_MODEL.
    let default = format!("{home}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit");
    if std::path::Path::new(&default).exists() {
        return default;
    }
    for suffix in [
        "models/lmstudio-community/Qwen3-1.7B-MLX-8bit",
        "models/lmstudio-community/Qwen3-1.7B-MLX-4bit",
        "models/mlx-community/Qwen3-0.6B-4bit",
        "models/mlx-community/Qwen3-1.7B-4bit",
    ] {
        let p = format!("{home}/{suffix}");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    default
}

/// Best-effort check for mlx_lm throughput via python -m mlx_lm.generate.
/// Returns None if python or mlx_lm not installed. Synchronous, times one call.
fn bench_mlx_lm(prompt: &str, max_tokens: usize) -> Result<f64> {
    let model_arg = resolve_model_dir();
    // Fallback to HF id if local dir was a placeholder not present
    let model_arg = if std::path::Path::new(&model_arg).exists() {
        model_arg
    } else {
        "mlx-community/Qwen3-0.6B-4bit".to_string()
    };
    let out = std::process::Command::new("python3")
        .args([
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            &model_arg,
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens.to_string(),
            "--temp",
            "0",
            "--ignore-chat-template",
        ])
        .output()
        .context("spawn python mlx_lm")?;
    if !out.status.success() {
        anyhow::bail!("mlx_lm failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // mlx_lm prints: "Generation: 10 tokens, 130.651 tokens-per-sec"
    for line in stdout.lines() {
        if line.contains("Generation:") && line.contains("tokens-per-sec") {
            for (i, tok) in line.split_whitespace().enumerate() {
                if tok == "tokens-per-sec" {
                    if i > 0 {
                        if let Ok(v) = line.split_whitespace().nth(i - 1).unwrap_or("").parse::<f64>() {
                            return Ok(v);
                        }
                    }
                }
            }
        }
    }
    anyhow::bail!("could not parse mlx_lm tps from: {stdout}")
}
