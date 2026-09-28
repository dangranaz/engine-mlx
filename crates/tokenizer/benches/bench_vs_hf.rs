//! Benchmark: pure-Rust `PureTokenizer` vs HuggingFace `tokenizers`.
//!
//! Encodes a fixed corpus with both tokenizers and compares throughput.
//! Both load the SAME `tokenizer.json`, so this is an apples-to-apples
//! encode-speed comparison (correctness parity is covered by unit tests).
//!
//! Usage:
//!   HF_TOKENIZER_PATH=/path/to/tokenizer.json cargo bench -p nxm-tokenizer
//!
//! If HF_TOKENIZER_PATH is unset, falls back to a default local model path.
//! The benchmark is skipped (with a printed notice) if the file is absent,
//! so `cargo bench` never fails just because the model isn't downloaded.

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use nxm_tokenizer::PureTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

/// Resolve the tokenizer.json path from env or a default local model.
fn tokenizer_path() -> String {
    std::env::var("HF_TOKENIZER_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        format!("{home}/models/mlx-community/Qwen3-0.6B-4bit/tokenizer.json")
    })
}

/// Representative corpus: mixed lengths, languages, code, chat markup.
fn corpus() -> Vec<&'static str> {
    vec![
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "I'm doing great, thanks! Don't you think it's a lovely day?",
        "<|im_start|>user\nExplain quantum computing in simple terms<|im_end|>\n<|im_start|>assistant\n",
        "La capitale della Francia è Parigi, una città ricca di storia e cultura.",
        "fn main() { let x: Vec<u32> = (0..100).map(|i| i * 2).collect(); println!(\"{:?}\", x); }",
        "2 + 2 = 4, and 17 * 23 = 391. Contact: test@email.com or visit https://example.com/path?q=1",
        "UPPER lower MixedCase 日本語 émojis and àccénts, plus\ttabs\nand newlines.",
    ]
}

fn bench_encode(c: &mut Criterion) {
    let path = tokenizer_path();
    if !Path::new(&path).exists() {
        eprintln!(
            "\n[bench skipped] tokenizer.json not found at {path}\n\
             Set HF_TOKENIZER_PATH=/path/to/tokenizer.json to run this benchmark.\n"
        );
        return;
    }

    let texts = corpus();
    let total_bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();

    let mut pure = PureTokenizer::load(Path::new(&path)).expect("load PureTokenizer");
    let hf = HfTokenizer::from_file(&path).expect("load HF tokenizer");

    let mut group = c.benchmark_group("encode_corpus");
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("pure_tokenizer", |b| {
        b.iter(|| {
            for t in &texts {
                black_box(pure.encode(black_box(t)));
            }
        })
    });

    group.bench_function("huggingface", |b| {
        b.iter(|| {
            for t in &texts {
                black_box(hf.encode(black_box(*t), false).expect("HF encode"));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
