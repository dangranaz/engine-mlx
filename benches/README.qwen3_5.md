# Qwen3.5-9B benchmark (engine-mlx)

Dedicated benchmark setup for the **Qwen3.5-9B-MLX-4bit** hybrid model on
engine-mlx. Qwen3.5 is architecturally different from the dense Qwen3 line: 3 of
every 4 layers are **Gated DeltaNet** (linear/recurrent attention), the rest are
**gated full attention with partial RoPE**. engine-mlx runs it through a
dedicated eager hybrid forward that is **token-exact vs `mlx_lm`** at
temperature 0 (verified by the engine's own test suite).

## Files

- `models/start-qwen3.5-9b.sh` / `models/stop-qwen3.5-9b.sh` — per-model
  launchers (baked-in model name, resolves the engine `server.sh` across repo
  layouts, points `ENGINE_MLX_MODELS_DIR` at `~/models` by default).
- `expected/qwen3_5.json` — a **dedicated pass/fail profile** for the hybrid
  model: correctness + coherence are hard requirements; the throughput floor
  (`min_tps = 3`) reflects the current eager (uncompiled) forward.

## Quick start

```bash
# 1. build the bench
cargo build --release

# 2. start engine-mlx serving Qwen3.5-9B (release build, ~50-75s to load 9B)
./models/start-qwen3.5-9b.sh

# 3. run the standard 5 tests, assert against the dedicated profile
./target/release/nxm-bench run \
  --url http://127.0.0.1:11435 --engine engine-mlx \
  --model qwen3.5-9b-mlx-4bit

# 4. judge against the qwen3_5 profile
./check_expected.sh "$(ls -t results/engine-mlx/*.json | head -1)" qwen3_5

# 5. stop
./models/stop-qwen3.5-9b.sh
```

## Measured behavior (MacBook Air M1, 16 GB)

- **Correctness / coherence**: token-exact with `mlx_lm` (temperature 0) — proven
  by the engine's `qwen3_5_*` tests (prefill first-token + top-5, multi-step
  decode, and 5 diverse prompts × 16 tokens).
- **Throughput**: steady-state **~4 tok/s** for the eager hybrid forward. There
  is a one-time **MLX JIT warmup** (~15 s on the first request of a fresh
  process); subsequent requests start fast (JIT kernel cache). Do NOT compare
  these t/s to the dense Qwen3 numbers — different regime (Qwen3 uses a compiled
  decode step; the hybrid path is not yet compiled).

## Sustained load & long prompts — RESOLVED

The intra-request memory peak in the Gated DeltaNet prefill is fixed. The
per-timestep recurrent scan in `gdn_forward` (which allocated ~20 lazy MLX
arrays per token per GDN layer and used to OOM a 9B-4bit model on a 16 GB Mac)
is replaced by a **single fused Metal kernel that loops over the whole sequence
internally** (`gdn_prefill_metal`), state kept in registers, one dispatch per
layer. It mirrors `mlx_lm`'s sequential recurrence kernel bit-for-bit, so the
engine stays **token-exact** (all four `qwen3_5_*` engine tests still pass).

Alongside the byte-peak fix, the per-request live-buffer *count* is now flat: the
GDN layer, the gated full-attention layer, and the decode loop all free their
transient MLX handles, so long generations no longer exhaust Metal's resource
limit (499000) and degenerate to garbage.

Verified (MacBook Air M1, 16 GB, release):

- Intra-request peak: ~42 GB → ~11–12 GB for a ~70-token prompt; a single
  **1024-token** generation completes cleanly, and **5 consecutive 1024-token
  requests produce coherent, identical (greedy) output with zero resource-limit
  errors**.
- The full 5-test `nxm-bench` suite (sustained 20×128, length ramp 128/512/1024,
  coherence) **completes with no EOF/OOM**. Judged with `--assert` against the
  `qwen3_5` profile: **instruction PASS, coherence PASS**, and the **sustained
  throughput floor passes** (`min_tps ≈ 5.4 ≥ 3`).

Remaining (orthogonal to memory): the eager forward does not stop at the chat
turn boundary, so it bleeds the template (e.g. emits `Parisuser: …`). This makes
the `factual` `\bparis\b` regex miss even though "Paris" is the answer, and it is
a stop-token/template issue — not the GDN memory peak. The generic `--assert`
`min_tps=20` also over-fails `sustained`/`length_ramp` for the eager regime; the
authoritative `qwen3_5` profile floor is 3 and passes. Tracked in the engine repo
issue tracker.
