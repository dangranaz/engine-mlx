# The story of engine-mlx


## What I built

[engine-mlx](https://github.com/dangranaz/engine-mlx) is a small, honest LLM
inference engine for Apple Silicon, written in Rust on top of Apple's MLX
framework (via its C API). It loads a quantized model, runs prefill and decode,
serves an OpenAI-compatible HTTP API, and generates text that matches `mlx_lm`
token-for-token at temperature 0.

Alongside it I published:
- [prj-bench](https://github.com/dangranaz/prj-bench) — a reproducible
  benchmark harness *and* the real reference numbers, so anyone can re-run them.
- [prj-scripts](https://github.com/dangranaz/prj-scripts) — simple start/stop
  scripts to run the server.

All three have green CI.

## Why I'm building this

The goal is **real local AI on consumer computers — for everyone.** Inference
that runs on the machine people already own, without shipping their data to the
cloud and without depending on frontier models behind paid APIs.

Starting on limited hardware wasn't an accident, and it shaped everything: if it
runs well on a 16 GB MacBook Air, it runs on the laptops most people and small
teams actually have. I also wanted to **learn frontier technology independently
— without depending on paid APIs or expensive hardware** to do it.

And it's meant to be a **foundation, not a demo.** A small, correct, auditable
engine like this is a base you can build vertical products on — for small and
medium businesses — and it scales naturally: swap in **open-weight models from
labs around the world** (Qwen, and others) instead of renting frontier models
from the cloud. That's the bet: local, private, affordable AI as a real
alternative, not a compromise.

## The honest part

I built engine-mlx primarily by **orchestrating AI coding agents** — guiding
them, and re-engineering work I had done before — rather than typing every line
by hand. The skill on display here isn't "I can write MLX kernels in Rust from
memory." It is **directing AI agents to build and verify complex systems
software, and knowing when a result is real and when it isn't.**

That distinction matters, because the second skill is the one that actually
shipped a working engine.

## Why Rust

I chose **Rust** deliberately, for robustness: strong typing, no garbage
collector, explicit memory ownership, and errors you handle rather than discover
at runtime — exactly what you want in a systems component like an inference
engine. The trade-off is honest: compared to Python, the Rust ecosystem for MLX
is still young — bindings are thinner, examples fewer — so more has to be built
and verified from the primitives. That friction is part of why the verification
discipline below mattered so much.

## The setup (and the constraint)

Everything here — the engine, the fixes, the benchmarks — was done on a
**MacBook Air M1 with 16 GB of unified memory**. That's not a footnote: the
hard constraint shaped the work. The nastiest bug was about keeping a live
Metal-buffer count flat on limited memory; "it works" had to mean "it works on
this machine," not on a workstation.

I work mainly with **OpenCode** and **Pi**, using the **free** AI models in
OpenCode and the **free models offered by NVIDIA** — no paid model
subscriptions. The leverage isn't expensive tooling; it's steering these agents
well and holding them to a hard, verifiable bar for "done."

## What "verify" looked like

The interesting engineering wasn't typing code — it was choosing objective,
unforgeable targets and holding the work to them:

- **Token-exact parity with `mlx_lm`** at temperature 0. Not "looks similar" —
  the *same token IDs*. If one differs, it's a bug.
- **A reproducible benchmark**, published with the numbers, so the performance
  claims can be checked, not trusted.
- **Reporting limitations honestly**: engine-mlx is a correctness-first
  baseline; its absolute throughput still trails `mlx_lm`, and the gap is
  kernel efficiency, not graph overhead. That's in the README.

## The bug that proves the point

Before publishing, the benchmarks caught something ugly: under sustained HTTP
load the engine's output degenerated into garbage ("...") from about the third
request. The tempting move is to patch symptoms. Instead I diagnosed it:

- It **pre-existed** and was masked because tests reset state between runs while
  the HTTP path didn't — proven by reproducing it on the older code.
- The error was `[metal::malloc] Resource limit (499000)` while memory was only
  ~4 GB of a ~15 GB limit — so it wasn't bytes, it was a **live-buffer count**.
- The fix came from reading my own earlier working design and aligning to it:
  reset and rebuild fresh state per request, and explicitly free the refcounted
  MLX buffers instead of just dropping the Rust handles.

After the fix: stable across mixed 128/512/1024-token requests and long bursts,
with sane numbers (~32–40 t/s on Qwen3-1.7B-4bit, ~8% sustained degradation).

## Why I'm sharing this

The value isn't a flashy benchmark. It's a small, auditable engine that proves
its correctness, tells you exactly where it stands, and was built by steering AI
agents with a rigorous bar for "done." That's the way I work — and it's the
skill I want to be known for.
