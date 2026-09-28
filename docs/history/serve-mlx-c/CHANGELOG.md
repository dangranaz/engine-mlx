# Changelog — nxm-serve-mlx-c

All notable changes to this project will be documented in this format.

## [0.2.0] - 2026-08-04

### Added
- feat(loader): auto-fuse separate Q/K/V and gate/up projections at load time — supports standard HuggingFace model format without pre-processing
- feat(loader): lm_head support for non-tied models (Qwen3-8B) — loads separate lm_head.weight/scales/biases
- feat(engine): async eval pipeline decode loop — overlaps GPU compute with CPU readback (+106% throughput)
- feat(ops): `async_eval` / `async_eval_one` wrappers for `mlx_async_eval`
- feat(ops): `quantize_kv_native` / `dequantize_kv_native` — native MLX int4 affine quantization for KV cache
- feat(engine): adaptive memory limits — `mlx_set_memory_limit`, `mlx_set_cache_limit`, `mlx_set_wired_limit` based on model size
- feat(engine): `mlx_clear_cache` on reset for stable multi-request operation

### Changed
- perf(forward): GQA native SDPA in prefill — removed `repeat_axis` for K/V expansion (+13% throughput)
- perf(engine): KV cache upgraded from manual int8 symmetric to native MLX int4 affine (group_size=64) — 28% of bf16 memory footprint
- perf(alloc): tight KV capacity based on prompt_tokens + max_tokens instead of fixed sliding_window

### Fixed
- fix(engine): prompt cache disabled — was causing KV corruption when decode tokens contaminated cached buffers
- fix(ops/attention): `&qkv` reference error in refactored `project_qkv_fused`

### Performance
- Qwen3-4B-4bit: 8.7 → 16.2 t/s (+86%), coherence 3/3
- Qwen3-8B-4bit: broken → 10.7 t/s, coherence 3/3, stable 8/8 requests
- rapid-mlx comparison: at parity on 4B, 8B not achievable by rapid-mlx on 16GB

## [0.1.0] - 2026-07-31

### Added
- feat: Qwen3 engine (MLX-C) with fused weights, GPU argmax decode, slice_update_dynamic KV writes, sdpa_with_mask
- feat: tokenizer via `nxm-tokenizer` (BPE/WordPiece/Unigram dispatch) — `tokenizer.rs` wrapper
- feat: ChatML encode_chat / decode helpers
- feat: HTTP server (axum) — `/v1/chat/completions`
- test: end-to-end integration suite — model load, debug logits, generation coherence, 100-token benchmark
- test: Unigram integration — t5-small loaded through serve-mlx-c, encode parity vs HF + decode roundtrip

### Fixed
- fix: argmax byte-reinterpretation bug — `mlx_array_data_float32` on int32 array reinterprets bytes; switched to `to_vec_i32`, decoded tokens were all 0 before this fix

### Changed
- chore: dev-dependency `tokenizers = "0.20"` for HF parity tests
