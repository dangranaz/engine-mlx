# engine-mlx

MLX-C inference engine — a reference baseline using Apple's MLX framework via
the `mlx-c` C API.

## Objective

A working, auditable MLX-based inference engine that:
1. Serves as a correctness-first reference baseline.
2. Provides a simpler development path (MLX handles kernel dispatch).
3. Runs Qwen3-class models end to end with an OpenAI-compatible HTTP API.

## Crates (workspace)

| Crate       | Role                                                       |
|-------------|------------------------------------------------------------|
| `mlx-ffi`   | MLX-C bindgen + `MlxCtx` (real ops, Metal link)            |
| `ops`       | Atomic ops (matmul, rope, quantized, sdpa) via `MlxCtx`    |
| `attention` | GQA + sliding window + gated attention                     |
| `kvcache`   | KV cache backends: concat / fp8 / rotating                 |
| `prefill`   | Prefix cache + chunked batch prefill                       |
| `serve`     | Qwen3 engine + loader + OpenAI HTTP + bench                |
| `modelplan` | Model introspection → `ModelManifest` (vendored)           |
| `tokenizer` | Pure-Rust multi-format tokenizer, HF-parity (vendored)     |

## Build prerequisites

```bash
brew install mlx-c
cargo build --release --features mlx
```

Without the `mlx` feature the crates build against stubs.

## Current status

- End-to-end forward pass: `Qwen3Engine::load` → `embed` → `prefill` →
  `decode_step`, eager and compiled.
- Generic loader via `ModelManifest`, supports `group_size`/`bits` from config
  (4-bit and 8-bit quantization).
- Token-exact greedy vs `mlx_lm` (`--ignore-chat-template --temp 0`) on
  Qwen3-0.6B and Qwen3-1.7B.
- Static KV cache is the default: pre-allocated buffers + `slice_update_dynamic`
  + masked SDPA, so throughput doesn't degrade with context length.
- BF16 pipeline end to end; pipelined `async_eval` (launch → async → readback).
- CPU sampling (`temperature`/`top-p`/`top-k`) in `serve`.

## Known limitations

- Absolute throughput trails `mlx_lm`; the gap is kernel efficiency, not graph
  overhead.
- No long-context offload (no disk spill); very large contexts beyond RAM are
  out of scope for this baseline.
- The MLX-C C API does not expose the graph fusions of Python `mx.compile`.

## Tooling

`scripts/` contains model-agnostic bash tooling to operate the server and
produce standard HTTP benchmarks. See `scripts/README.md`.
