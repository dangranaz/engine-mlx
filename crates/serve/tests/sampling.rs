//! Sampling test — verify temperature diversifies from greedy.

#[cfg(feature = "mlx")]
mod mlx_sampling {
    use std::path::Path;

    fn model_dir() -> String {
        std::env::var("HOME")
            .map(|h| format!("{h}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit"))
            .unwrap_or_default()
    }

    #[test]
    #[ignore = "sampling requires model + mlx"]
    fn sampling_diversifies_from_greedy() {
        let md = model_dir();
        let dir = Path::new(&md);
        if !dir.exists() {
            eprintln!("skipping: model not found {md}");
            return;
        }
        let mut eng = engine_mlx_serve::Qwen3Engine::load(dir).expect("load");
        let prompt = "Hello, how are you?";

        // Greedy
        let out_greedy = eng.generate(prompt, 20).expect("greedy generate");
        println!("greedy: {:?}", out_greedy);

        // Sampling
        eng.reset();
        eng.set_sampling(
            engine_mlx_serve::sampler::SamplingParams::default()
                .with_temperature(0.8)
                .with_top_p(0.9)
                .with_top_k(40)
                .with_seed(123),
        );
        let out_sample = eng.generate(prompt, 20).expect("sampled generate");
        println!("sampled (0.8/0.9/40): {:?}", out_sample);

        // Should not be identical repetitive
        assert_ne!(out_greedy, out_sample, "sampling should diversify");
        // Greedy is repetitive "I'm sorry..."
        // Sampled should not be all repeats (basic check)
        println!("sampling diversifies OK");
    }
}
