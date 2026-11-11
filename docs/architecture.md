# Architecture

engine-mlx is a Rust workspace built on Apple's MLX via its C API (`mlx-c`),
linking Metal directly. Without the `mlx` feature the non-MLX crates build
against stubs, so the workspace compiles and tests on non-Apple machines for CI.

## Request flow

A chat request flows through the engine in four stages:

1. **Load** — the model loader reads `config.json`, resolves per-tensor
   quantization (`group_size`/`bits`), and maps the weights into `MlxCtx`. The
   model's chat template is parsed once.
2. **Prefill** — the prompt is rendered through the chat template, tokenized by
   the pure-Rust tokenizer, embedded, and run through the transformer to
   populate the KV cache.
3. **Decode** — one token per step: embed → per-layer attention (masked SDPA
   over a static, pre-allocated KV cache) + MLP → in-graph greedy argmax, with
   pipelined `async_eval`. A static KV cache means decode throughput does not
   collapse as context grows.
4. **Serve** — results are returned over an OpenAI-compatible HTTP API
   (`/v1/chat/completions`, `/v1/models`, `/health`), with SSE streaming.

State is rebuilt fresh per request, so long or repeated requests do not
accumulate buffers across calls.

## Crate layout

| Crate        | Role                                                         |
|--------------|--------------------------------------------------------------|
| `mlx-ffi`    | MLX-C bindings + `MlxCtx` real ops / Metal link (or stubs).  |
| `ops`        | Atomic ops (quantized matmul, RoPE, SDPA, RMSNorm, …).       |
| `attention`  | Attention layers (GQA, sliding window, gated).              |
| `kvcache`    | KV cache backends (concat, fp8, rotating).                  |
| `prefill`    | Prefill pipeline (prefix cache + chunked batch prefill).    |
| `serve`      | Engine + model loader + OpenAI HTTP server.                 |
| `modelplan`  | Model introspection → manifest.                             |
| `tokenizer`  | Pure-Rust multi-format tokenizer, HF-parity tested.         |

## Numerics

The engine runs a BF16 pipeline and is **token-exact** with the reference
(`mlx_lm`, or the model's own reference implementation) at temperature 0 —
verified by tests, not by inspection. Mixed-quantization models run attention
and MLP projections at different bit widths within the same forward pass;
grouped RMSNorm handles multi-group normalization alongside the fused
single-group path.

## MLX-C constraints

The MLX-C C API does not expose all the graph fusions of Python `mx.compile`,
which caps some kernel-level optimizations. The engine is a correctness-first
baseline: absolute throughput still trails `mlx_lm`, and the gap is kernel
efficiency rather than graph overhead.
