# STATUS — engine-mlx

## State: End-to-end forward pass working (eager + compiled)

## What works

- Cargo workspace with 8 crates (ops, mlx-ffi, attention, prefill, kvcache,
  serve, modelplan, tokenizer). All compile (default stubs + `mlx` feature).
- `mlx-ffi`: real bindgen bindings over `mlx-c`/`mlx`, direct link to
  `libmlx`/`libmlxc` + Metal/Foundation/QuartzCore/Accelerate. Non-aborting
  custom error handler.
- `MlxCtx`: real ops (`mlx_quantized_matmul`, `mlx_fast_rope`, `mlx_fast_sdpa`,
  `mlx_dequantize`, `mlx_rms_norm`, …) plus cfg-gated stubs.
- `serve`: `Qwen3Engine::load` → `embed` → `prefill` → `decode_step` on
  Qwen3-0.6B / Qwen3-1.7B / K2-Horizon-3.7B. Token-exact vs `mlx_lm`
  (`k2_horizon.py` for K2) at temperature 0.
- **Mixed per-weight quantization** (per-tensor `group_size`/`bits`, e.g. 8-bit
  attention + 4-bit MLP), **grouped RMSNorm**, and **thinking chat templates** —
  exercised by K2-Horizon-3.7B-4bit (token-exact 34/34, stable under sustained load).
- Static KV cache by default (pre-allocated + masked SDPA): no O(KV) decode
  degradation as context grows.
- BF16 pipeline end to end; pipelined `async_eval`.
- CPU sampling (`temperature`/`top-p`/`top-k`); OpenAI-compatible HTTP + SSE.
- Benchmark harness vs `mlx_lm`.
- Pure-Rust tokenizer with HuggingFace parity tests + reproducible benchmark.
- Stable under sustained/varied HTTP load: each request resets and rebuilds
  fresh KV state (no cross-request buffer accumulation), so long or repeated
  requests no longer degrade or produce garbage.

Indicative throughput (Apple Silicon): Qwen3-0.6B ~57 t/s eager, ~55 t/s
compiled decode. Qwen3-1.7B-MLX-4bit ~32-40 t/s (sustained degradation ~8%,
length-ramp ~18%). K2-Horizon-3.7B-4bit ~7-8 t/s (thinking model, decode not
yet optimized — honest and stable, not a ceiling; sustained ~12%, ramp ~11%).

Supported mlx-c range: `>=0.7.0, <0.9.0` (0.7.x and 0.8.x).

## Known limitations

- Absolute throughput trails `mlx_lm` — the gap is kernel efficiency.
- K2-Horizon decode is not yet optimized (per-token sync + full-capacity
  attention in the static path); correctness and stability are unaffected.
- No long-context disk offload (out of scope for this baseline).
- The MLX-C C API lacks the graph fusions of Python `mx.compile`.

## Next steps

1. CI (cargo check + clippy on macOS; stub check on Linux).
2. K2-Horizon decode optimization (pipeline the static path, `offset+1`
   attention) toward the ~15-18 t/s the model size predicts.
3. Sampling refinements and richer output controls.
