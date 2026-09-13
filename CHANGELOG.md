# CHANGELOG — engine-mlx

All notable changes are documented here.

## [Unreleased]

### Fixed
- Output no longer degenerates under sustained/varied HTTP load. Previously,
  from ~the 3rd request the server could emit garbage ("...") because the
  cross-request KV-cache reuse path kept masks/offset arrays/the compiled
  closure alive between requests; combined with `async_eval` pipelining this
  exhausted Metal's live-buffer count (`[metal::malloc] Resource limit`) — a
  buffer COUNT, not bytes. Each request now resets and rebuilds fresh KV state
  and explicitly frees refcounted `mlx_array` handles before clearing, keeping
  the live-buffer count flat. Verified stable across mixed 128/512/1024-token
  requests and long identical-request bursts on Qwen3-1.7B-MLX-4bit.

### Added
- End-to-end forward pass on Qwen3-class models: load → embed → prefill →
  decode, with eager and compiled decode paths.
- Generic model loader via `ModelManifest`, supporting 4-bit and 8-bit
  quantization (`group_size`/`bits` read from model config).
- Static KV cache by default: pre-allocated buffers + masked SDPA, avoiding
  decode-throughput degradation as context length grows.
- BF16 pipeline end to end; pipelined `async_eval`.
- CPU sampling (`temperature` / `top-p` / `top-k`).
- OpenAI-compatible HTTP API with SSE streaming.
- Benchmark harness comparing against `mlx_lm`.
- Pure-Rust multi-format tokenizer (BPE / WordPiece / Unigram) with
  HuggingFace parity tests and a reproducible benchmark.
- Model-agnostic server/benchmark tooling under `scripts/`.

### Notes
- Token-exact greedy output vs `mlx_lm` (`--temp 0 --ignore-chat-template`)
  on Qwen3-0.6B and Qwen3-1.7B.
- This is a correctness-first baseline; absolute throughput trails `mlx_lm`
  and the gap is kernel efficiency. Long-context disk offload is out of scope.
