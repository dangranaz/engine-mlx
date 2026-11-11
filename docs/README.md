# engine-mlx documentation

Product documentation for **engine-mlx**, a local LLM inference engine for
Apple Silicon. For the project overview and quick start, see the
[top-level README](../README.md).

## Contents

- **[Supported models](./models.md)** — the model families and quantization
  formats that are exercised and verified, and what makes each one notable.
- **[Benchmarks](./benchmarks.md)** — generation speed, sustained-load
  stability, and length-ramp behavior, with the methodology to reproduce them.
- **[Architecture](./architecture.md)** — how a request flows through the
  engine: load → prefill → decode → serve, and the crate layout.

All numbers in these documents are measured on an Apple Silicon Mac at
temperature 0 (greedy) and are meant to be reproduced, not trusted — the
benchmark harness is public and runs on your own hardware.
