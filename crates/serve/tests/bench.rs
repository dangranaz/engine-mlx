//! Bench vs mlx_lm — ignored by default, run with --ignored --nocapture.

#[cfg(feature = "mlx")]
mod mlx_bench {
    use std::path::Path;

    fn model_dir() -> String {
        if let Ok(m) = std::env::var("ENGINE_MLX_MODEL") {
            if Path::new(&m).exists() {
                return m;
            }
        }
        std::env::var("HOME")
            .map(|h| format!("{h}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit"))
            .unwrap_or_default()
    }

    #[test]
    #[ignore = "bench requires model + mlx, run with --ignored --nocapture"]
    fn bench_matrix_10_128() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping bench: model dir not found {md}");
            return;
        }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).expect("load");
        let prompt = "Hello, how are you?";
        let points = [10, 128];
        let results = engine_mlx_serve::bench::bench_matrix(&mut eng, prompt, &points).expect("bench_matrix");
        for r in &results {
            assert!(r.tps > 0.0, "tps should be >0");
            assert!(!r.generated_text.is_empty(), "empty generation");
        }
        println!("bench done: {:?}", results.iter().map(|r| r.tps).collect::<Vec<_>>());
    }
}
