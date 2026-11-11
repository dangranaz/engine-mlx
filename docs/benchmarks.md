# Benchmarks

engine-mlx ships a reproducible benchmark harness: the exact tests, thresholds,
and reference profiles live in the repo, so these numbers can be re-run on your
own hardware rather than taken on faith.

Measured on a **MacBook Air M1, 16 GB**, greedy decode (temperature 0).

## Generation speed

| Model                 | Generation speed | Sustained (20×)   | Length ramp (256→1024) |
|-----------------------|------------------|-------------------|------------------------|
| Qwen3-0.6B-MLX-4bit   | ~55–57 t/s       | —                 | —                      |
| Qwen3-1.7B-MLX-4bit   | ~34–41 t/s       | ~6% degradation   | ~16% drop              |
| K2-Horizon-3.7B-4bit  | ~7–8 t/s¹        | ~12% degradation  | ~11% drop              |

- **Generation speed** — decode tokens/second at steady state (the one-time MLX
  JIT warmup on the first request per process is excluded).
- **Sustained** — 20 identical requests back to back; throughput must not drift
  from first to last (this is the memory-leak guard, threshold 15%).
- **Length ramp** — throughput across 256 / 512 / 1024 output tokens, measuring
  how decode scales with KV-cache growth.

> ¹ **K2-Horizon decode is not yet optimized.** ~7–8 t/s is honest and stable,
> not a ceiling. For a 3.7B 4-bit model the size alone predicts ~15–18 t/s; the
> gap is the engine's static-decode path (per-token CPU↔GPU synchronization and
> attention computed over the full KV capacity instead of the used window), not
> the model. Correctness (token-exact) and stability (no leak) are unaffected.
> A decode-path optimization is planned to close the gap.

## What the benchmark checks

Every run evaluates five tests across three categories:

| Test            | Category     | What it asserts                                              |
|-----------------|--------------|-------------------------------------------------------------|
| Factual recall  | Correctness  | A deterministic factual answer appears in the output.       |
| Instruction     | Correctness  | A strict-format instruction is followed.                    |
| Sustained       | Degradation  | Throughput stays flat across 20 identical requests.         |
| Length ramp     | Degradation  | Throughput scales acceptably as output length grows.        |
| Reasoning       | Coherence    | A multi-step word problem reaches the correct final answer. |

The correctness and coherence discriminators are **preamble-tolerant**: they
look for the answer anywhere in the output, so a thinking model that reasons
before answering is judged on its final answer — without parsing or stripping
the reasoning. They remain meaningful: empty, whitespace-only, or wrong-answer
output still fails.

## Thresholds and profiles

Pass/fail thresholds are model-aware. A per-model **profile** declares the
minimum acceptable throughput floor and which tests are hard requirements; the
degradation guard (15% drift across the sustained run) is the same for all
models and is never relaxed to make a model pass. For a thinking model whose
honest throughput is lower, the profile sets a floor below the measured speed
with margin — documented as such, not as a way to hide a regression.

## Reproducing

Start the server on a model, then run the harness against it. The harness
records a timestamped JSON + Markdown report and a HEALTHY/UNHEALTHY verdict per
profile. Because results depend on your chip and thermal state, re-run on a
cool machine for comparable numbers — sustained throughput on Apple Silicon can
dip once the chip has been under load for several minutes (thermal throttling),
which is a property of the hardware, not the engine.
