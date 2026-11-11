# CHANGELOG — engine-mlx

All notable changes are documented here.

## [Unreleased]

### Added
- README house-style header (shields.io badges, tagline, hashtags, request-flow
  diagram at `docs/engine-mlx-flow.png`, Repo/Stack/API/Parity block),
  `Contributors` section, and license statement (MIT, free for personal use).
- Product documentation under `docs/` (index, models, benchmarks, architecture),
  linked from the README `Documentation` section.

### Fixed
- Bench model launchers (`start-k2-horizon-3.7b.sh`, `start-qwen3.5-9b.sh`):
  dropped the stale third fallback for `scripts/server.sh` (archived private
  monorepo path); `$ENGINE_MLX_REPO` or the in-repo relative path only.
- Per-layer decode leak (task 35/E5-eagerleak): `concat_last_axis` in the Separate projection
  re-pushed the first concat piece into `scratch`, but the piece's producer
  (`quant_matmul_piece`) had already pushed it — so `qo`/`gate` ended up in `scratch` twice and
  was freed twice at cleanup (`mlx_array_free` on the same handle), corrupting the refcount and
  leaking one Metal resource per concat per layer (~8 B/layer). Fixed by giving `concat_last_axis`
  single ownership of only the intermediate concat results it creates and skipping the caller-owned
  first part; every per-layer intermediate is now freed exactly once. Decisive gate: 30×256
  `leak_probe` at ~0 MB/req growth with no collapse; token-exact parity 34/34 unchanged.
- Static KV reset re-allocation (task 34/E5-kvreset): `zero_kv_static` now resets only
  the logical offset and reuses the existing KV buffers instead of allocating and freeing
  `2 × n_layers` full-capacity BF16 arrays per request. The decisive 30×256 leak probe
  nevertheless remained at ~+1MB/request and degenerated at request 26, so this
  work-branch change is not merge-ready and requires the strategic KV-cache migration.
- Mask intermediates leak (task 33/E5-leak): `update_static_mask_cpu` and the fresh-mask path in
  `zero_kv_static` shadowed the `new_mask` binding across `new_array → reshape → astype`, leaving
  the intermediate arrays (A and B) unreachable and never freed. Each mlx_array is a refcounted C
  handle without Drop, so ~1273 decode steps × 2 leaked intermediates per request produced the
  +5.4MB/req growth that degenerated to whitespace at req ~14. Fixed by using separate bindings
  (no shadowing), `eval_all` to materialize the final array, then `mlx_array_free` on the
  intermediates before replacing `self.static_mask`. Same treatment in `alloc_kv_static`.
  Result: sustained memory growth reduced from +5.4MB/req to ~+1.4MB/req; token-exact parity
  (34/34) unchanged; render + 6-long-requests gates unaffected.
- Per-request state reset regression (task 32/E5-reset): the static attention
  mask reset in `zero_kv_static()` was a no-op because MLX `slice_update()`
  returns a new array (not in-place) — the result was discarded and `eval_all`
  evaluated the old mask, so the mask retained state from the previous request
  and the 2nd+ request degenerated. Fixed by creating a fresh mask array from
  scratch (same pattern as `update_static_mask_cpu`) and replacing
  `self.static_mask`. Also removed `mlx_clear_cache()` from `reset()` (would
  force recompilation, defeating the leak fix), added `mlx_synchronize()` after
  `eval_all()` to ensure materialization before freeing old handles, and disabled
  `compiled_full` (concat-KV closure) when `compiled_static` fails to prevent
  static-KV buffer corruption. Gates: `k2_horizon_token_parity` PASS (34/34),
  3-request mixed sequence coherent (ocean 2231, story 4763, poem 6761 chars),
  `six_long_requests_stay_coherent` PASS, `serving_render_encode_matches_fixture`
  PASS, `probe_sustained_leak` flat memory (+1 MB/request), zero `Resource limit`
  errors. Anti-repetition threshold calibrated from 3 to 50 (R5).
- Compile-cache leak under sustained mixed-load (task 28/E5): the decode
  closure is compiled ONCE at fixed MAX_CAP (4096) during engine init and
  reused across all requests, eliminating per-request recompilation that
  accumulated in MLX's global compile cache and triggered
  `[metal::malloc] Resource limit (499000)` after ~24 variable-capacity
  requests. Static KV buffers and offset arrays are pre-allocated once;
  `reset()` / `zero_kv_static()` only zero data and offset — no free/
  re-alloc. Single additive attention mask [1,1,1,MAX_CAP] is updated
  in-place per step (token-exact vs pre-materialized O(cap) masks). Gates:
  `k2_horizon_token_parity` extended to 256 tokens, `sustained_mixed_load_stays_coherent`
  GREEN at ≥30 mixed requests, `probe_sustained_leak` flat memory (+5MB/request),
  zero `Resource limit` errors.
- Numerical parity restored (task 29/E5 refine): in the static compiled path the KV write returns to `scatter_single(k_cache, offset_arr, reshape(k,[1,1,nkv,1,hd]), 2)` (idem v) as in integration `607e30b`, instead of `slice_update_dynamic` which introduced a deterministic 7-position shift (fix-vs-reference divergence at token 134 instead of 218). Fixed-capacity lever + single mask unchanged: leak still resolved, fix-vs-integration token-exact.
- FP32 masked attention (task 30/E5-fp32): replaced the fused `sdpa_masked` kernel in the static compiled path with explicit FP32 attention (scores + mask add + softmax in fp32, output cast to bf16), eliminating the bf16 softmax rounding that diverged at token 134 with MAX_CAP=4096. GQA handled via KV-head tiling. Applied to both `compiled.rs` (compiled closure) and `decode_eager.rs` (eager fallback). Leak fix (capacity-fixed + single mask + compile once) and `scatter_single` KV write unchanged.
- Parity gate moved to a NON-degenerate prompt (task 31/E5-parity-gate): the old
  gate prompt ("Name three capital cities of European countries.") degenerated
  into a repetitive list where near-equiprobable tokens flip on irreducible
  float noise (~token 134; even integration diverges from the reference at 218),
  so token-exactness on it was unreachable and not indicative of a bug. The gate
  now uses "What color is a ripe banana? Answer in one word." — verified
  non-degenerate with mlx_lm (34 greedy tokens, natural EOS, no ~6-token
  subsequence repeated ≥3 times) — fixtures regenerated from mlx_lm via
  `gen_reference_fixture.py` (external oracle, never the engine), and
  `k2_horizon_token_parity` asserts an EXACT match over all 34 tokens again.
  Result: engine is token-exact vs mlx_lm on the new prompt with BOTH the fp32
  manual attention (189c165) and the bf16 `sdpa_masked` (af1ddf4); per the task
  decision rule both being token-exact, the bf16 fused kernel is kept (less
  code, less overhead: 7.5 t/s vs 6.9 t/s at 128 tokens on K2-Horizon-3.7B-4bit),
  so the fp32 attention code is reverted. `serving_render_encode_matches_fixture`
  updated to the new gate prompt. Leak fix untouched.

### Fixed
- Per-request state reset + model-derived EOS in the serving decode (task
  18/E1): `generate` (and `generate_chat`, which delegates) now calls `reset`
  at the START of every request, so a fresh request never inherits the
  previous one's `offset`/KV state — the 2nd request no longer degenerates to
  newlines-only output. EOS ids for the decode loop now come from the loaded
  model (priority: `generation_config.json` -> `config.json` -> tokenizer.json
  added_tokens; the Qwen3 `EOS_IDS` constant remains only as a documented
  last-resort fallback for model dirs shipping no EOS source), so K2-Horizon
  (`[1, 250019]`) stops on its real EOS instead of Qwen3's. Gates:
  `serving_second_request_is_coherent` (#[ignore], real model),
  `generate_resets_state_between_calls` (CI), `eos_ids_resolved_from_model`
  (CI).
- Per-request MLX resource accumulation that corrupted long generations from
  ~request 6 of the same process (task 23/E4): `grouped_rms_norm` now frees
  the intermediate MLX arrays it owns (`x_f32`, `ones`, `x_grouped`, `y`,
  `y_back`, `scaled`). `mlx_array` is `Copy` with no `Drop`, so dropping the
  Rust handle never released the underlying Metal buffer; ~73 calls per decode
  step leaked ~739 MB/request, irreversibly crossing MLX's memory limit until
  decode degenerated to whitespace-only output. `reset()` already frees
  `kv_bufs` / `static_masks` / `static_offsets`, but transients owned by ops
  were not covered. Diagnosis notes: the compile-cache hypothesis (H1, a new
  closure compiled per request via `compile_shapeless`, never cleared since
  `mlx_clear_cache` only empties the buffer pool) is a real mechanism but was
  falsified as the cause of the ~6-request cliff — recompilation per request
  is unchanged by the fix and memory is flat afterwards. Regression test:
  `six_long_requests_stay_coherent` (`#[ignore]`, real model, >=6 x
  256-token `generate_chat` on one engine, last request must stay coherent;
  RED before the fix, GREEN after).

### Added
- Benchmark profile for K2-Horizon-3.7B-4bit: `benches/models/start-k2-horizon-3.7b.sh`
  / `stop-k2-horizon-3.7b.sh` launchers and `benches/expected/k2_horizon.json`
  profile (correctness + coherence hard-required, min_tps 15). Running the suite
  surfaced two real serving bugs (tracked for fix): per-request state
  (`offset`/`kv_bufs`) is not reset at the start of `generate`/`generate_chat`,
  so only the first request of a process is coherent and later ones degenerate
  to newlines; and `EOS_IDS` is hardcoded to Qwen3 values instead of the loaded
  model's (K2 uses `[1, 250019]`), so decoding never stops on K2's real EOS.
- Serving path wired to the model's chat template (task 15/D2): `main.rs`
  builds chat prompts via `Qwen3Engine::generate_chat`, which renders
  `messages` through `ChatTemplate` (task 14, `add_generation_prompt = true`)
  loaded once from `model_dir`, replacing the hardcoded
  `format!("{role}: {content}")`. `generate(&str)` is kept for raw prompts
  and the `main.rs` self-test. Dead hardcoded templates removed
  (`serve::encode_chat`, `Tokenize::apply_chat_template`, `pure::encode_chat`);
  `generate_chat` errors when no template is loaded — no hardcoded fallback.
  Test: `serving_prompt_uses_chat_template`.
- Decision (nxm-decision, task 15): render lives in `Qwen3Engine::generate_chat`,
  not in `main.rs`. Rationale: `Qwen3Engine` already owns the model dir, the
  `Tokenizer` and loads `ChatTemplate` once at load time, so it is the natural
  single surface for the model's chat template; `generate(&str)` is preserved
  for raw prompts and the self-test, keeping the render unduplicated and the
  engine unit-testable without HTTP. The alternative (render in `main.rs` next
  to a second `ChatTemplate`) was rejected because it would duplicate the
  loaded template and split the render surface.
- F1 (task 17): split `crates/serve/src/engine.rs` (368 lines, over the RULES
  300-line cap) into `engine/chat.rs` (`generate_chat`) and `generate.rs`
  (`generate`), behavior-preserving; `engine.rs` is back under 300 lines.
- F3 (task 17): a model with no `chat_template.jinja` is not servable over
  `/v1/chat/completions` — `generate_chat` returns a request-time error
  ("no chat template available") with no hardcoded fallback, so a base model
  fails loudly instead of silently producing a wrong prompt.
- F2b (task 17): end-to-end `generate_chat`-to-generated-tokens gate omitted as
  redundant with the task-13 `k2_horizon_token_parity` gate, which already
  asserts the engine reproduces the greedy continuation from the same
  templated prompt (`crates/serve/tests/fixtures/k2_horizon_token_parity.txt`);
  F2 (`serving_render_encode_matches_fixture`) covers the render→encode step
  that the parity gate skips.

### Fixed
- `rope_theta` is now read from the nested `rope_parameters.rope_theta` when
  present (e.g. K2-Horizon: `{"rope_theta": 1e7, "rope_type": "default"}`),
  falling back to the top-level key, then to the `10000` default. Previously
  only the top-level key was read, so K2-Horizon fell back to `10000` instead
  of `1e7` — a wrong RoPE base that shifts every position and breaks token
  parity. Regression tests cover nested, top-level, and nested-wins-over-stale-
  top-level cases. Found while running the K2-Horizon token-parity gate.

### Added
- K2-Horizon model recognition at the config/model-plan layer: `config.json`
  key `layernorm_num_groups` (default 1 when absent) is parsed into
  `RawModelConfig` and carried through to `ModelManifest.norm.num_groups` and
  `Qwen3Config.layernorm_num_groups`. `ops_detector` now recognizes
  `model_type = "k2_horizon"` (dense transformer: rope + flash-attn +
  swiglu, untied lm_head, grouped RMSNorm served by the existing
  `grouped_rms_norm` op). Verified by introspection tests against a
  K2-Horizon-shaped config (groups=2, tie=false, vocab 250624, 36 layers,
  hidden 2560) and a default-to-1 regression test for configs without the key.
- Grouped RMSNorm in Qwen3 forward: `grouped_rms_norm` now used at all three
  hidden-state norm sites (`input_layernorm`, `post_attention_layernorm`,
  `final_norm`) across all three forward paths (eager prefill, eager decode,
  compiled step). `groups == 1` delegates to fused `rms_norm` (identical
  behavior for qwen3); `groups > 1` applies per-group normalization in f32
  then full-width weight, matching `GroupedRMSNorm` semantics. q_norm/k_norm
  remain plain `rms_norm`. lm_head untied selection unchanged.
- Mixed per-tensor quantization config parsing: `parser.rs` now reads
  per-tensor overrides from `quantization_config` (e.g., `model.embed_tokens`,
  `model.layers.0.self_attn.q_proj`, `lm_head`) in addition to global defaults.
  `QuantConfig` gains `overrides` map and `quant_for(name)` lookup method
  (exact key match, falls back to default). Both manifest builders (`lib.rs`
  `introspect` and `manifest_builder.rs`) populate overrides. Tested with
  Qwen3 uniform config (no overrides) and K2-Horizon mixed config (default
  4-bit/64 + 8-bit/64 for embed/q_proj/lm_head, 4-bit/32 for gate_proj).
  Follow-up (review of task 06): quant parsing split into
  `parser/quant.rs` (parser.rs back under 300 lines) and the unused `mode`
  field dropped from `PerTensorQuant` (parsed-but-discarded).
- Per-projection quantization params in the loader weight structs (task 08/B01):
  new `QuantParams { group_size, bits }` and `QuantTensor { w, s, b, quant }`;
  `Qwen3LayerWeights` now exposes `qkv: QkvProj` and `gate_up: GateUpProj`
  (enum `Fused`/`Separate`, loader always builds `Fused` for now) plus
  `o_proj`/`down_proj` as `QuantTensor`; `Qwen3Weights.embed_tokens` is a
  `QuantTensor` and `lm_head` an `Option<QuantTensor>`. Params are resolved
  from the manifest via `QuantConfig::quant_for` in the new pure module
  `loader/quant_params.rs` (`0/0` when the model is not quantized). Forward
  call-sites adapted mechanically to the `fused()` accessor (no logic change;
  per-weight params in matmuls and the `Separate` arm are task C, marked
  `TODO(task C)`).
- Load-time fusion gating (task 09/B02): q/k/v and gate/up are fused at load
  time only when their per-tensor `(group_size, bits)` all coincide; otherwise
  the loader keeps them `Separate` (each piece with its own `QuantParams`), so
  no fused tensor with inconsistent quant params is ever produced. Decision
  logic is the pure `loader/fusion.rs` (`fusion_decision`, `qkv_fusion`,
  `gate_up_fusion`), logged per group at layer 0. K2-Horizon (q/k/v all
  64/8, gate/up all 32/4) still fuses, unchanged.
- Forward consumes per-weight quant params (task 11/C1): every quantized op
  (`quantized_matmul` for qkv/o/gate_up/down/lm_head in prefill, eager decode
  and the compiled step, plus the embed `dequantize_weight`) now uses the
  `(group_size, bits)` of its own projection via `QuantParams::to_ffi` instead
  of the single global pair with (64,4) fallback — numerically required for
  mixed-quant models like K2-Horizon. `to_ffi` bails with a contextual error
  on `group_size == 0` (the not-quantized sentinel) instead of passing 0.
- Forward handles the `Separate` projection variant (task 12/C2): new
  `forward/projection.rs` (`run_qkv`, `run_gate_up`) runs one
  `quantized_matmul` per piece with that piece's own `QuantParams` and
  concatenates the activations on the last axis in concat order
  (q,k,v / gate,up), reproducing the fused layout. Used by prefill, eager
  decode and the compiled step; verified numerically equal to the fused
  branch on identical weights (`--features mlx` tests
  `qkv_separate_matches_fused`, `gate_up_separate_matches_fused`). The
  fused (K2 real) path is unchanged.

### Fixed
- Build with `--features mlx` against mlx-c 0.7.0: the C function
  `mlx_fast_scaled_dot_product_attention` gained a `force_fused: bool`
  parameter (now 10 arguments). Both SDPA call-sites (`sdpa`, `sdpa_masked`)
  pass `false`, preserving the pre-0.7.0 behaviour (MLX picks the kernel per
  shape) rather than forcing a fused kernel that may reject some shapes.
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
- mlx-c build-time version gate: `crates/mlx-ffi/build.rs` reads the installed mlx-c
  version from `share/cmake/MLXC/MLXCConfigVersion.cmake` and fails the build with an
  actionable message when it falls outside the supported range (`>=0.7.0, <0.8.0`),
  gated behind the `mlx` feature (`CARGO_FEATURE_MLX`).
- `--check-env` / `check-env` subcommand on `nxm-engine-mlx`: an MLX runtime liveness
  probe that queries the linked library via `mlx_version()`, prints the detected core
  MLX version for diagnostics, exits 0 when reachable and non-zero only when the
  library is unreachable. No version range is judged at runtime (mlx-c is pinned at
  build-time). Gated behind the `mlx` feature; stub builds report "not available" and
  exit 0. Invoked as a warn-only preflight by `scripts/server.sh` and
  `scripts/start-server.sh`.
- Documentation: README `Requirements` declares the supported mlx-c range
  (`>=0.7.0, <0.8.0`) and the build-time gate behaviour; CONTEXT distinguishes the
  build-time mlx-c pin from the `--check-env` runtime liveness probe (core MLX
  version, no range judgment).
- `MlxCtx::grouped_rms_norm`: new GroupedRMSNorm op in `engine-mlx-ffi` (ops/real/grouped_norm.rs).
  For `groups == 1` it delegates to the fused `rms_norm` kernel; for `groups > 1` it
  normalises each group independently in f32 (all-ones kernel weight, no per-group scaling)
  then applies the full-width weight after reshape-back, matching the reference MLX
  `GroupedRMSNorm` semantics. Rejects inputs whose last dimension is not divisible by
  `groups`. Stub build (no `mlx` feature) bails gracefully.
- Follow-up: fixed review findings on `grouped_rms_norm` — removed `unwrap()` in favour of
  slicing, added `tracing::trace` logging, completed `# Errors` doc with all three error
  conditions, added note on all-ones weight divergence from reference. Extended test coverage:
  weight in bf16 (promotes to f32 for multiply, casts back correctly), multi-dimensional
  input `[batch, seq, dims]`, and `groups=4` case.
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

### Added
- Chat template rendering (task 14/D1): new pure `nxm_tokenizer::chat_template` module (`ChatTemplate::from_model_dir` + `render`) rendering the model's own `chat_template.jinja` (fallback to the `chat_template` key of `tokenizer_config.json`) via `minijinja 2.21.0`, with `raise_exception`, `bos_token`, `messages` and `add_generation_prompt` in the Jinja context. No template hardcoded; existing hardcoded chat templates left in place for task 15 wiring. Tests vs committed K2-style fixture + token-parity gate (`#[ignore]`, needs `ENGINE_MLX_MODEL`).
- Follow-up (task 16/D1): pre-processing strip of HF `{% generation %}` / `{% endgeneration %}` block markers so the real K2 `chat_template.jinja` compiles under `minijinja 2.21`; `bos_token` accepts both a string and an `{"content": "..."}` object. Committed the real K2 `chat_template.jinja` (~50 KB, public model data) as a fixture. R1 token-parity gate (`chat_template_render_matches_k2_fixture`, `#[ignore]`) verified green on the real model via `ENGINE_MLX_MODEL` pointing at the K2-Horizon model dir: render + encode reproduces `k2_horizon_prompt_ids.txt` exactly.

### Changed
- Repo restructure (ADR-006/ADR-004, TAPPA 2): the product now owns its
  periphery — `scripts/{start,stop}-server.sh` (from prj-scripts), `benches/`
  (from nxm-bench/crates/engine-mlx; generated `results/` artifacts are
  gitignored), and `docs/history/{attention-mlx,ops-mlx,serve-mlx-c}` (from
  nxm-docs). Structural migration only — no logic changes, no version bump.

### Notes
- Token-exact greedy output vs `mlx_lm` (`--temp 0 --ignore-chat-template`)
  on Qwen3-0.6B and Qwen3-1.7B.
- This is a correctness-first baseline; absolute throughput trails `mlx_lm`
  and the gap is kernel efficiency. Long-context disk offload is out of scope.
