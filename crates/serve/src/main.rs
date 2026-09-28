//! engine-mlx-serve — OpenAI-compatible HTTP server.

use std::path::PathBuf;
use std::sync::mpsc;

use anyhow::Result;
use tracing::info;

use engine_mlx_serve::{handle::{InferHandle, InferRequest}, server};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let host = std::env::var("ENGINE_MLX_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    // Default bind port 11435 — the standard local server port (11434 is
    // Ollama, which we run for the TUI; 11435 keeps our engine distinct).
    let port: u16 = std::env::var("ENGINE_MLX_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(11435);
    let model_path = std::env::var("ENGINE_MLX_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{home}/models/lmstudio-community/Qwen3-0.6B-MLX-4bit")
    });

    info!(model = %model_path, host = %host, port, "starting engine-mlx");

    let (tx, rx) = mpsc::channel::<InferRequest>();
    let handle = InferHandle::new(tx);

    // MLX thread — all GPU ops here (single thread)
    let model_dir = PathBuf::from(&model_path);
    std::thread::spawn(move || {
        let mut engine = match engine_mlx_serve::Qwen3Engine::load(&model_dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!(target: "engine_mlx::main", "engine load failed: {e}");
                while let Ok(req) = rx.recv() {
                    if let InferRequest::Generate { reply, .. } = req {
                        let _ = reply.send(Err(anyhow::anyhow!("load failed: {e}")));
                    }
                }
                return;
            }
        };

        info!(target: "engine_mlx::main", "MLX ready — self test");
        let test_prompt = "Hello";
        match engine.generate(test_prompt, 2) {
            Ok(text) => info!(target: "engine_mlx::main", "self-test PASSED text={:?}", text),
            Err(e) => tracing::error!(target: "engine_mlx::main", "self-test FAILED: {e}"),
        }

        while let Ok(req) = rx.recv() {
            match req {
                InferRequest::Generate { messages, max_tokens, reply } => {
                    // Concatenate messages into prompt (simple chat template)
                    let prompt = messages
                        .iter()
                        .map(|(role, content)| format!("{role}: {content}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let start = std::time::Instant::now();
                    let prompt_tokens = engine
                        .get_tokenizer()
                        .encode(&prompt)
                        .map(|v| v.len() as u32)
                        .unwrap_or(0);
                    let result = engine.generate(&prompt, max_tokens as usize);
                    match result {
                        Ok(text) => {
                            let completion_tokens = engine
                                .get_tokenizer()
                                .encode(&text)
                                .map(|v| v.len() as u32)
                                .unwrap_or(0);
                            let elapsed = start.elapsed().as_secs_f64();
                            let tps = if elapsed > 0.0 { completion_tokens as f64 / elapsed } else { 0.0 };
                            let _ = reply.send(Ok((text, prompt_tokens, completion_tokens, tps)));
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e));
                        }
                    }
                }
            }
        }
    });

    server::run(handle, &host, port).await
}
