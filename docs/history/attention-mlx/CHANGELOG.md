# Changelog — nxm-attention-mlx

All notable changes to this project will be documented in this format.

## [0.1.0] - 2026-08-04

### Added
- **SDPA backend** (feature `sdpa`, default):
  - `project_qkv`, `project_qkv_fused` — QKV projections (unfused/fused)
  - `optional_qk_norm` — per-head RMS norm for Q/K (Qwen3)
  - `transpose_for_attn` — layout conversion for SDPA
  - `apply_rope` — dynamic RoPE with offset array
  - `repeat_kv` — GQA head repetition
  - `attend_and_project` — transpose back + O projection
  - `sdpa_decode_pipeline` — composed decode pipeline
  - `gqa_ratio` helper
- **FlashAttention bridge** (feature `flash`):
  - `sdpa_flash` — auto-dispatch (single-kernel for kv_len ≤ 128, Split-K for longer)
  - `sdpa_flash_single_kernel` — explicit single-kernel path
  - `sdpa_flash_splitk` — explicit Split-K path
  - Zero-copy Metal buffer wrapping via unified memory
  - `SPLIT_K_THRESHOLD = 128` constant
- **Gated attention** (feature `gated`):
  - `project_q_gated` — split Q projection into queries + output gate
  - `apply_output_gate` — sigmoid(gate) * attn_output + O projection
  - `gated_attention_pipeline` — full gated attention pipeline
- **GDN placeholder** (feature `gdn`):
  - `GdnConfig` struct
  - `gdn_step` (unimplemented)
  - `gdn_init_state` (unimplemented)

### Dependencies
- `nxm-mlx-ops` — MLX-C FFI ops (sdpa, gated, gdn features)
- `nxm-metal-ops` — Metal kernels (flash feature)
- `nexum-inferentia-ffi` — MLX context with array data access (flash feature)
- `objc2-metal` — Metal FFI (flash feature)
- `nxm-attention-coord` — coordinator types (always)

### Tests
- Unit tests for each module
- Doc tests for public API

### Architecture
- Feature-gated: minimal default (`sdpa`), optional `flash`/`gated`/`gdn`
- Separation of concerns: low-level FFI in `nxm-mlx-ops`, Metal in `nxm-metal-ops`, coordination in `nxm-attention-coord`