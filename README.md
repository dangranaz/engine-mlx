# engine-mlx

A small, honest LLM inference engine for Apple Silicon, built on Apple's
[MLX](https://github.com/ml-explore/mlx) framework via its C API (`mlx-c`),
written in Rust.

It loads a quantized model, runs prefill and decode, serves an
OpenAI-compatible HTTP API, and generates text that matches `mlx_lm`
token-for-token at temperature 0.

## What it is

A **reference-quality baseline**: the goal was correctness and clarity, not
squeezing out every last token/second. It runs Qwen3-class models (0.6B, 1.7B)
end to end and is easy to read and audit.

| Crate        | Role                                                        |
|--------------|-------------------------------------------------------------|
| `mlx-ffi`    | MLX-C bindings (bindgen) + `MlxCtx` real ops / Metal link   |
| `ops`        | Atomic ops (quantized matmul, RoPE, SDPA, RMSNorm, …)       |
| `attention`  | Attention layers (GQA, sliding window, gated)               |
| `kvcache`    | KV cache backends (concat, fp8, rotating)                   |
| `prefill`    | Prefill pipeline (prefix cache + chunked batch prefill)     |
| `serve`      | Qwen3 engine + model loader + OpenAI HTTP server + bench    |
| `modelplan`  | Model introspection → `ModelManifest` (vendored)            |
| `tokenizer`  | Pure-Rust multi-format tokenizer, HF-parity (vendored)      |

## Status

- End-to-end forward pass works: load → embed → prefill → decode.
- **Token-exact vs `mlx_lm`** (`--temp 0 --ignore-chat-template`) on Qwen3-0.6B
  and Qwen3-1.7B.
- Static KV cache by default (pre-allocated buffers + masked SDPA), so decode
  throughput doesn't collapse as the context grows.
- BF16 pipeline end to end; pipelined `async_eval`.

Measured on Apple Silicon (indicative, reproduce with the bench harness):
Qwen3-0.6B ~57 t/s eager, ~55 t/s compiled decode.

## Honest limitations

- This is a baseline, not a speed record. `mlx_lm` is still faster in absolute
  throughput — the gap is kernel efficiency, not graph overhead.
- No long-context offload: very large prompts / contexts beyond RAM are not
  handled (there is no disk spill). Standard prompts are fine.
- The MLX-C C API doesn't expose the graph fusions available in Python
  `mx.compile`, which caps some optimizations.

## Build

```bash
# macOS with Apple Silicon
brew install mlx-c

cargo build --release --features mlx
```

Without the `mlx` feature the workspace builds against stubs (useful for CI on
non-Apple machines and for compiling the non-MLX crates).

## Run

```bash
cargo run --release --features mlx -p engine-mlx-serve -- \
  --model /path/to/Qwen3-0.6B-MLX-4bit
# then POST to the OpenAI-compatible /v1/chat/completions endpoint
```

## How this was built

This engine was built by **orchestrating AI coding agents** against objective,
verifiable acceptance tests — token-exact parity with `mlx_lm`, reproducible
benchmarks — rather than hand-writing every line. The engineering that mattered
was choosing the right targets, verifying relentlessly, and reporting results
(including limitations) honestly.

## License

MIT — see [LICENSE](./LICENSE).
