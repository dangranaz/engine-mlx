# Supported models

engine-mlx targets dense transformers in MLX format. The table lists the
configurations that are actually exercised and verified; the loader reads
`group_size` / `bits` per tensor from the model config, so other MLX
checkpoints of the same shapes should load, but only these are tested.

| Model             | Quantization                  | Verification                               |
|-------------------|-------------------------------|--------------------------------------------|
| Qwen3-0.6B        | 4-bit (MLX)                   | token-exact vs `mlx_lm`                    |
| Qwen3-1.7B        | 4-bit (MLX)                   | token-exact vs `mlx_lm`                    |
| Qwen3-1.7B        | 8-bit (MLX)                   | token-exact vs `mlx_lm` (4/8-bit per config) |
| K2-Horizon-3.7B   | mixed: 8-bit attn / 4-bit MLP | token-exact vs its reference (34/34 tokens) |

## Qwen3 family (dense)

The baseline target. Loaded from the standard MLX layout (`config.json` +
`model.safetensors` + `tokenizer.json`), served over the OpenAI-compatible API.
Generation is token-for-token identical to `mlx_lm` at temperature 0, verified
by tests rather than inspection.

## K2-Horizon-3.7B-4bit

K2-Horizon exercises the capabilities added for mixed-architecture models:

- **Mixed per-weight quantization** — attention projections at 8-bit
  (group size 64), MLP projections at 4-bit (group size 32), in the same model.
  The loader resolves `group_size`/`bits` per tensor; projections with different
  quantization are run as separate matmuls and recombined.
- **Grouped RMSNorm** — normalization over multiple groups (2 for this model),
  alongside the standard single-group fused path.
- **Nested `rope_theta`** — the model's rotary base is read from config and
  applied per layer.
- **Thinking chat template** — the model emits a reasoning preamble before its
  final answer. engine-mlx renders the model's own chat template (including the
  reasoning channel) and serves the full output; it does not strip or
  special-case the reasoning.

Correctness is verified token-exact against the model's reference implementation
(34/34 tokens at temperature 0), and the engine stays coherent and leak-free
under sustained load (30+ consecutive requests with flat memory and no output
degradation).

> **Throughput note.** K2-Horizon decode is correct and stable but **not yet
> optimized** — see [benchmarks](./benchmarks.md). The current speed reflects
> the engine's static-decode path, not the model; a speedup is planned.

## Getting a model

engine-mlx does not ship weights. Download an MLX-format model into the
HuggingFace-style layout and point the server at it:

```
<models-dir>/<org>/<model-name>/
    ├── config.json          # required
    ├── model.safetensors
    └── tokenizer.json
```

By default the engine searches under `~/models`; override with the
`ENGINE_MLX_MODELS_DIR` environment variable, or pass an absolute `--model` path.
