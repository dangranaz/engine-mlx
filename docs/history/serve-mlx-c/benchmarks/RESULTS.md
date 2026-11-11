# mlx-c Engine Benchmarks

Hardware: Apple M1 16GB | OS: macOS | Branch: develop

---

## 2026-08-02

### Commits

- `06355aa` — argmax in-graph (elimina dispatch separata)
- `a460742` — KV cache int8 quantization (riduce bandwidth SDPA)

### Qwen3-0.6B-4bit (28L, hidden=1024, heads=16/kv=8)

| Scenario | Baseline | After argmax-in-graph | Δ |
|----------|----------|----------------------|---|
| 100 tok decode | ~80 t/s | **91.7 t/s** (peak) | +14% |
| Step time | ~12ms | ~11ms | |

### Qwen3-1.7B-4bit (28L, hidden=2048, heads=16/kv=8)

| Scenario | bf16 KV | int8 KV | Δ |
|----------|---------|---------|---|
| 100 tok decode | 41.6 t/s | 39.9 t/s | -4% (quant overhead) |
| 500 tok decode | 34.8 t/s | **37.5 t/s** | **+8%** |
| Step time @100 | 24.5ms | 25.3ms | |
| Step time @500 | 28.0ms | 26.5ms | |

### Observations

- **0.6B**: dispatch-latency bound. Argmax in-graph eliminates 1 evaluate+sync per step.
- **1.7B @short seq**: memory-bandwidth bound (weight reads dominate). KV quant overhead ≈ gain.
- **1.7B @long seq**: KV cache SDPA read becomes significant. Int8 halves KV bandwidth → +8%.
- `compile_shapeless`: negligible impact on both models (MLX graph too linear for fusion benefit).
- Scaling prediction: 4B/8B will benefit more from KV int8 (larger KV heads, longer default context).

### Pending

- [ ] Qwen3-4B-4bit benchmark (download in progress)
- [ ] Qwen3-8B-4bit benchmark
- [ ] K8V4 (TurboQuant V) for further KV compression
- [ ] Sliding window mode with quantized KV (build_step_sliding not yet updated)
