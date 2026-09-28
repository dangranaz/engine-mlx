# engine-bench

Cross-engine benchmark for OpenAI-compatible inference servers. It speaks only
HTTP (`/v1/chat/completions`, `/v1/models`, `/health`), so the **same** tool and
the **same** 5 tests run against any engine — you just point it at a different
URL/port. Results are written to timestamped JSON + Markdown files so you keep a
history of how each engine behaves over time.

> **AI agents:** for a machine-actionable procedure + result-interpretation guide
> (triage table, engine-vs-model rule, per-size expectations), read
> [`AGENT_TESTING.md`](AGENT_TESTING.md).

## The 5 standard tests

| # | test | category | what it checks |
|---|------|----------|----------------|
| 1 | Factual recall | correctness | Deterministic answer contains `Paris` |
| 2 | Instruction following | correctness | Obeys "reply just DONE" format |
| 3 | Sustained throughput | degradation | N identical requests don't lose t/s (leak guard) |
| 4 | Length ramp | degradation | t/s across 128/512/1024 tokens (KV scaling) |
| 5 | Multi-step reasoning | coherence | Word problem → correct answer `18` with working |

Correctness/coherence tests use `temperature=0` (greedy) and **tolerant regex**,
so they travel across engines and model families. Full model outputs are always
captured in the report for inspection.

## Requirements

- Rust (`cargo`) to build.
- A running OpenAI-compatible server (use `start.sh`, or start your own).
- `curl` (used by `start.sh` for readiness).

## Quick start

nexum's standard local server port is **11435** (Ollama uses 11434, which we run
for the TUI; 11435 keeps our engine distinct).

```bash
# 1. build
cargo build --release

# 2. start engine-mlx for a model on port 11435.
#    Per-model launchers bake the model name in (one command per model):
MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx \
  ./models/start-qwen3-0.6b.sh
#    ...or the generic launcher with an explicit MODEL:
#    MODEL=Qwen3-0.6B-MLX-4bit RELEASE=1 ./start.sh

# 3. full run: Hurl declarative gates + nxm-bench measurements (one command)
./run.sh

# 4. or run only nxm-bench directly
./target/release/nxm-bench run --url http://127.0.0.1:11435 --engine engine-mlx --assert

# 5. stop the engine
./models/stop-qwen3-0.6b.sh        # or ./stop.sh

# 6. compare two runs (or two engines) side by side
./target/release/nxm-bench compare --latest engine-mlx,engine-metal
```

## Two test layers

engine-bench combines two complementary layers, orchestrated by `run.sh`:

1. **Hurl** (`hurl/`) — declarative HTTP tests (from
   [Orange-OpenSource/hurl](https://github.com/Orange-OpenSource/hurl)). Fast
   pass/fail gates for API shape, error handling, semantic **coherence**, and a
   latency gate. The coherence gate is **blocking** (corrupted output → fail,
   per nexum RULES). Requires `brew install hurl`; if absent, `run.sh` skips this
   layer with a warning.
2. **nxm-bench** (Rust) — what Hurl can't do well: aggregate throughput (t/s),
   **sustained-degradation** (memory-leak guard), KV-cache length ramp, real
   SSE TTFT, and timestamped JSON+MD reports.

## Per-model scripts

Each model has its own launcher under `models/` that embeds the model name, so
starting a specific model is a single command:

```
models/start-qwen3-0.6b.sh   models/stop-qwen3-0.6b.sh
models/start-qwen3-1.7b.sh   models/stop-qwen3-1.7b.sh
```

They wrap the generic `start.sh`/`stop.sh` (which accept `MODEL=`/`ENGINE=`
env). Add a new model by copying one and changing the `MODEL=` line.

## Subcommands

- `nxm-bench run [OPTIONS]` — benchmark a live server (the 5 standard tests).
- `nxm-bench compare <FILES...> | --latest <engine,engine>` — diff historical
  result JSONs.

## `run` — CLI

```
nxm-bench run [OPTIONS]

  -u, --url <URL>              server base URL          [default: http://127.0.0.1:11435]
  -e, --engine <NAME>          engine label (report dir) [default: engine]
  -m, --model <ID>             model id in requests     [default: first /v1/models id]
      --sustained-reps <N>     sustained test requests  [default: 20]
      --sustained-tokens <N>   sustained max_tokens     [default: 128]
      --ramp <A,B,C>           length-ramp points       [default: 128,512,1024]
  -o, --out-dir <DIR>          report root              [default: results]
      --assert                 enable pass/fail + exit code
      --thresholds <FILE>      thresholds.toml (with --assert)
      --min-tps <F>            override min tokens/sec
      --max-degradation-pct <F> override sustained degradation limit
      --engine-repo <DIR>      git repo to record engine commit
```

## `compare` — diff historical reports

Line up two or more result JSONs side by side. The **first** report is the
baseline; deltas are computed relative to it. Works both engine-vs-engine
(different engines, same tests) and run-vs-run (same engine, over time).

```
nxm-bench compare [FILES...] [OPTIONS]

  [FILES...]                 result JSON files (first = baseline)
      --latest <e1,e2,...>   auto-pick the newest results/<engine>/*.json per engine
  -o, --out-dir <DIR>        report root for --latest        [default: results]
      --md-out <FILE>        where to write the comparison MD [default: results/compare/<ts>.md]
```

Examples:

```bash
# two explicit runs of the same engine
nxm-bench compare results/engine-mlx/A.json results/engine-mlx/B.json

# newest run of each engine, side by side
nxm-bench compare --latest engine-mlx,engine-metal,vllm
```

Output: a `Sources` table (engine, model, timestamp, commit, verdict) and a
`Metrics` matrix with per-metric Δ vs baseline. Arrows: **▲** better than
baseline, **▼** worse, **=** unchanged (higher t/s is better; lower drop% is
better). A copy is written to `results/compare/<timestamp>.md`.

## Modes: measure-only vs assert

- **Measure-only** (no `--assert`): runs all tests, writes the report, prints
  metrics, always exits `0`. No verdict — good for exploration.
- **Assert** (`--assert`): same report **plus** a per-test and overall pass/fail
  verdict driven by thresholds. Exit code: `0` all passed, `1` a test failed,
  `2` an error occurred. Good for CI and regression gates.

Thresholds come from `thresholds.toml` (or `--thresholds FILE`) and can be
overridden per-run with `--min-tps` / `--max-degradation-pct`.

## Reports

Written to `results/<engine>/<YYYY-MM-DD_HH-MM-SS>.{json,md}`:

- **JSON** — machine-readable: an `environment` block (engine, url, model,
  timestamp, hostname, engine commit, bench version), the thresholds used,
  per-test metrics + full per-request samples + captured outputs, and the
  overall verdict. Ideal for historical comparison and CI.
- **Markdown** — human-readable summary table + per-test detail.

The `results/` directory is **committed** on purpose: it is the history of what
happened, per engine, over time.

## Benchmarking another engine

Point it at any OpenAI-compatible server:

```bash
# assume a server is already up on :8081
./target/release/nxm-bench run --url http://127.0.0.1:8081 --engine vllm --assert
```

To have `start.sh` launch a non-default engine, pass a full command:

```bash
BENCH_START_CMD="python -m vllm.entrypoints.openai.api_server --port 11435 --model <id>" \
  ENGINE=vllm ./start.sh
```

## TTFT note

Time-to-first-token is only meaningful when the server streams token-by-token.
The client can measure real TTFT over SSE (`chat_stream`), but engines that
generate the whole completion and then fake-stream it will report a TTFT close
to total latency. TTFT is therefore recorded only where streaming is real; the
current standard suite reports throughput and latency.
