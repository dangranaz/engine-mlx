# Server & benchmark scripts

Bash tooling to start/stop the `engine-mlx` OpenAI-compatible server and run
standard HTTP benchmarks against it — with optional head-to-head comparison
against `mlx_lm.server`.

All scripts are **model-agnostic**: the model is chosen via `ENGINE_MLX_MODEL`
(engine-mlx) or the `--model` request field (bench client). Nothing assumes a
specific model family. Qwen3 is only a convenience default path.

## Layout

| Script | Purpose |
|--------|---------|
| `lib/common.sh` | Shared library: logging, config, HTTP/readiness helpers, PID mgmt, percentiles. Sourced by the others. |
| `server.sh` | start/stop/status/restart/logs for the `engine-mlx` server (via cargo). |
| `mlx_lm_server.sh` | start/stop/status for `mlx_lm.server` as a comparison baseline. |
| `bench.sh` | HTTP benchmark client for any OpenAI-compatible `/v1/chat/completions`. |
| `bench_compare.sh` | Orchestrator: run the same bench on engine-mlx and mlx_lm, print a comparison. |

Runtime artifacts (pid/log/JSON) live in `scripts/.run/` (gitignored).

## Requirements

- `cargo` (to build/run engine-mlx), plus the `mlx` feature prerequisites
  (mlx-c headers, Apple GPU) — see repo `CONTEXT.md`.
- `curl` and `jq` for the bench client.
- `python3` with `mlx-lm` installed for the baseline (`pip install mlx-lm`).

## Quick start

```bash
# 1. see what models are available under $ENGINE_MLX_MODELS_DIR (default ~/models)
scripts/server.sh models list

# 2. start the engine-mlx server for a model (by name, org/name, or path)
scripts/server.sh start Qwen3-0.6B-MLX-4bit   # add --release for an optimized build

# 3. benchmark it over HTTP
scripts/bench.sh --matrix "32 128 512" --repeat 5

# 4. stop it
scripts/server.sh stop
```

## server.sh

```
scripts/server.sh start <model> [--release]   # resolve, build+run, wait /health
scripts/server.sh stop                        # graceful TERM, then KILL
scripts/server.sh restart <model> [--release]
scripts/server.sh status                      # running state + /health probe
scripts/server.sh logs                        # tail the server log
scripts/server.sh models list                 # list discovered models
scripts/server.sh models scan                 # (re)generate registry.json
```

### Model selection

`start` takes a `<model>` argument (or falls back to `ENGINE_MLX_MODEL`). It is
resolved to an absolute model dir with a **hybrid** strategy:

1. **Absolute path** to a valid model dir (must contain `config.json`) → used as-is.
2. **Registry alias** (or `"default"`) from `$ENGINE_MLX_MODELS_DIR/registry.json`, if present.
3. **Dynamic scan** under `$ENGINE_MLX_MODELS_DIR` (HuggingFace layout `<org>/<name>/`):
   - exact `<org>/<name>` match, or
   - unique bare `<name>` match.

A model dir is considered valid only if it contains `config.json`. If the name
is **not found**, `start` aborts with an error and points you at
`models list`. If a bare name is **ambiguous** (matches multiple orgs), it
aborts and lists the candidates — qualify it with `<org>/<name>`.

```bash
scripts/server.sh start Qwen3-0.6B-MLX-4bit           # bare name
scripts/server.sh start lmstudio-community/Qwen3-1.7B-MLX-8bit  # org/name
scripts/server.sh start /abs/path/to/model            # absolute path
scripts/server.sh start default                       # registry default
```

### Registry (optional)

`models scan` writes `$ENGINE_MLX_MODELS_DIR/registry.json`:

```json
{
  "default": null,
  "models": {
    "lmstudio-community/Qwen3-0.6B-MLX-4bit": "/abs/.../Qwen3-0.6B-MLX-4bit",
    "Qwen3-0.6B-MLX-4bit": "/abs/.../Qwen3-0.6B-MLX-4bit"
  }
}
```

Both `<org>/<name>` and unambiguous bare names are indexed. Set `.default` to a
key to make `start default` work. The registry is optional — resolution falls
back to the dynamic scan when it's absent — but takes priority when present.

Config via env:

| Var | Default | Meaning |
|-----|---------|---------|
| `ENGINE_MLX_HOST` | `127.0.0.1` | bind host |
| `ENGINE_MLX_PORT` | `11435` | bind port |
| `ENGINE_MLX_MODEL` | Qwen3-0.6B path | fallback model spec when `start` gets no argument |
| `ENGINE_MLX_MODELS_DIR` | `$HOME/models` | search root for model resolution |
| `READY_TIMEOUT` | `300` | seconds to wait for `/health` (model load is slow) |

## mlx_lm_server.sh

Same command surface as `server.sh`. Defaults to port **8081** so it can run
alongside engine-mlx (11435). `mlx_lm` has no `/health` route, so readiness is
probed on `/v1/models`. It accepts the same `<model>` argument, resolved the
same way — but since `mlx_lm` also accepts a bare HuggingFace id, a resolution
miss is a soft fallback (the spec is passed to `--model` as-is).

```
scripts/mlx_lm_server.sh start <model>   # or omit to use ENGINE_MLX_MODEL
scripts/mlx_lm_server.sh stop|status|restart|logs
```

Env: `MLXLM_HOST` (127.0.0.1), `MLXLM_PORT` (8081), `ENGINE_MLX_MODEL`,
`ENGINE_MLX_MODELS_DIR`, `PYTHON` (python3).

## bench.sh

HTTP client that measures:

- **throughput (t/s)** — `completion_tokens / total_latency`, taken from the
  OpenAI `usage` field (falls back to a word count if absent).
- **latency p50 / p95 / mean (ms)** — per-request wall time.
- **TTFT (ms)** — time to first SSE chunk, `--stream` only, best-effort.
- **RPS + aggregate t/s** — under `--concurrency > 1`.

```
scripts/bench.sh [options]

  -u, --url URL          Base URL (default http://127.0.0.1:11435)
  -m, --model NAME       model name in the request body (default "default")
  -p, --prompt TEXT      prompt text
  -n, --max-tokens N     max_tokens per request (default 128)
  -r, --repeat N         reps per matrix point (default 5)
  -c, --concurrency N    concurrent requests for the RPS test (default 1)
  -t, --temperature F    sampling temperature (default 0)
  --matrix "A B C"       space-separated max_tokens points (runs a matrix)
  --stream               use SSE streaming to measure TTFT
  --json FILE            write machine-readable results to FILE
  --label NAME           label for the run
```

Example:

```bash
scripts/bench.sh \
  --url http://127.0.0.1:11435 \
  --model qwen3 \
  --matrix "32 128 512" \
  --repeat 5 \
  --concurrency 8 \
  --json scripts/.run/engine.json
```

### TTFT caveat

engine-mlx currently generates the **full** completion and then fake-streams it
word by word over SSE. Its measured "TTFT" therefore reflects near-total
generation time, not a true first-token latency. `mlx_lm` streams for real.
**TTFT is only comparable between engines that both stream token-by-token.**
The script prints a warning in `--stream` mode to remind you.

## bench_compare.sh

Starts both servers (on distinct ports), runs the identical bench on each, and
prints a side-by-side table. Servers it started are stopped on exit, including
on Ctrl-C.

```
scripts/bench_compare.sh [options]

  --matrix "A B C"   max_tokens points (default "128")
  --prompt TEXT      prompt
  --repeat N         reps per point (default 5)
  --model SPEC       model both engines serve (path|alias|org/name|name)
  --release          build engine-mlx in release mode
  --skip-engine      don't start engine-mlx (assume :11435 already up)
  --skip-mlxlm       don't start mlx_lm (assume :8081 already up)
```

Example:

```bash
scripts/bench_compare.sh --model Qwen3-1.7B-MLX-8bit --matrix "128 512" --repeat 5 --release
```

Output includes a table like:

```
max_tokens     engine t/s     mlx_lm t/s      delta %     eng p95 ms
128                 57.30         135.10        -57.6          2234.0
512                 62.10         138.40        -55.1          8241.0
```

Positive `delta %` means engine-mlx is faster than mlx_lm. Per-engine JSON is
saved to `scripts/.run/compare_engine-mlx.json` and `compare_mlx_lm.json`.

## Notes

- For a fair comparison, serve the **same** `ENGINE_MLX_MODEL` on both engines.
- The in-process Rust bench harness (`crates/serve/src/bench.rs`) measures the
  engine directly, without HTTP. These scripts measure the full end-to-end HTTP
  path, which is what a real OpenAI client sees.
