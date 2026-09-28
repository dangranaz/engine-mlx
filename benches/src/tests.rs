//! The 5 standard tests.
//!
//! Correctness (2): does it answer sensibly and follow instructions?
//! Degradation (2): sustained throughput stability + length-ramp scaling.
//! Coherence  (1): multi-step reasoning with a verifiable answer.
//!
//! Correctness/coherence use temperature=0 (greedy) and tolerant regex, so the
//! same test travels across engines and model families. Full outputs are always
//! captured in the report for inspection.

use anyhow::Result;
use regex::Regex;

use crate::client::Client;
use crate::config::Thresholds;
use crate::report::{Category, Sample, TestResult};

/// Truncate a string for storage in the report.
fn truncate(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}

/// assert flag decides whether a verdict is produced.
pub struct Ctx<'a> {
    pub client: &'a Client,
    pub thresholds: &'a Thresholds,
    pub do_assert: bool,
    pub sustained_reps: u32,
    pub sustained_tokens: u32,
    pub ramp: Vec<u32>,
}

/// Run all 5 tests in order and return their results.
pub fn run_all(ctx: &Ctx) -> Result<Vec<TestResult>> {
    Ok(vec![
        test_factual(ctx)?,
        test_instruction(ctx)?,
        test_sustained(ctx)?,
        test_ramp(ctx)?,
        test_coherence(ctx)?,
    ])
}

// ── 1. Correctness: factual/deterministic ──────────────────────────────────
fn test_factual(ctx: &Ctx) -> Result<TestResult> {
    let prompt = "What is the capital of France? Answer with just the city name.";
    let r = ctx.client.chat_blocking(prompt, 32, 0.0)?;
    let re = Regex::new(r"(?i)\bparis\b").unwrap();
    let hit = re.is_match(&r.content);
    let mut notes = vec![format!("looked for /paris/i in response")];
    notes.push(if hit { "matched".into() } else { "NOT matched".into() });
    Ok(TestResult {
        id: "factual".into(),
        name: "Factual recall (capital of France)".into(),
        category: Category::Correctness,
        description: "Deterministic factual question; response must contain 'Paris'.".into(),
        prompt: prompt.into(),
        passed: ctx.do_assert.then_some(hit),
        notes,
        metrics: vec![("tps".into(), r.tps), ("latency_ms".into(), r.latency_ms)],
        samples: vec![Sample {
            index: 0,
            max_tokens: 32,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            latency_ms: r.latency_ms,
            tps: r.tps,
            ttft_ms: None,
        }],
        outputs: vec![truncate(&r.content, 400)],
    })
}

// ── 2. Correctness: instruction following ───────────────────────────────────
fn test_instruction(ctx: &Ctx) -> Result<TestResult> {
    let prompt = "Reply with exactly one word: DONE. Do not add anything else.";
    let r = ctx.client.chat_blocking(prompt, 16, 0.0)?;
    // Tolerant: the word DONE appears (case-insensitive), ignoring punctuation.
    let re = Regex::new(r"(?i)\bdone\b").unwrap();
    let hit = re.is_match(&r.content);
    // Extra signal: response is short (followed the "one word" constraint).
    let word_count = r.content.split_whitespace().count();
    let concise = word_count <= 5;
    let pass = hit && concise;
    let notes = vec![
        format!("looked for /done/i: {}", if hit { "matched" } else { "NOT matched" }),
        format!("word_count={word_count} (concise<=5: {concise})"),
    ];
    Ok(TestResult {
        id: "instruction".into(),
        name: "Instruction following (reply DONE)".into(),
        category: Category::Correctness,
        description: "Must follow a strict format instruction: reply just 'DONE'.".into(),
        prompt: prompt.into(),
        passed: ctx.do_assert.then_some(pass),
        notes,
        metrics: vec![
            ("tps".into(), r.tps),
            ("latency_ms".into(), r.latency_ms),
            ("word_count".into(), word_count as f64),
        ],
        samples: vec![Sample {
            index: 0,
            max_tokens: 16,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            latency_ms: r.latency_ms,
            tps: r.tps,
            ttft_ms: None,
        }],
        outputs: vec![truncate(&r.content, 200)],
    })
}

// ── 3. Degradation: sustained throughput stability ──────────────────────────
fn test_sustained(ctx: &Ctx) -> Result<TestResult> {
    let prompt = "Write a short paragraph about the ocean.";
    let mut samples = Vec::with_capacity(ctx.sustained_reps as usize);
    let mut outputs = Vec::new();
    for i in 0..ctx.sustained_reps {
        let r = ctx.client.chat_blocking(prompt, ctx.sustained_tokens, 0.0)?;
        if i == 0 || i + 1 == ctx.sustained_reps {
            outputs.push(truncate(&r.content, 200));
        }
        samples.push(Sample {
            index: i,
            max_tokens: ctx.sustained_tokens,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            latency_ms: r.latency_ms,
            tps: r.tps,
            ttft_ms: None,
        });
    }
    let tpss: Vec<f64> = samples.iter().map(|s| s.tps).collect();
    let max_tps = tpss.iter().cloned().fold(0.0_f64, f64::max);
    let min_tps = tpss.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_tps = if tpss.is_empty() { 0.0 } else { tpss.iter().sum::<f64>() / tpss.len() as f64 };
    // Degradation: drop from the fastest observed to the slowest observed.
    let degradation_pct = if max_tps > 0.0 {
        (max_tps - min_tps) / max_tps * 100.0
    } else {
        0.0
    };
    // First-vs-last drop is the clearest leak signal (monotonic slowdown).
    let first = tpss.first().copied().unwrap_or(0.0);
    let last = tpss.last().copied().unwrap_or(0.0);
    let first_last_drop_pct = if first > 0.0 { (first - last) / first * 100.0 } else { 0.0 };

    let pass = first_last_drop_pct <= ctx.thresholds.max_degradation_pct
        && min_tps >= ctx.thresholds.min_tps;
    let notes = vec![
        format!("reps={} tokens={}", ctx.sustained_reps, ctx.sustained_tokens),
        format!("tps min={min_tps:.1} mean={mean_tps:.1} max={max_tps:.1}"),
        format!("degradation(max→min)={degradation_pct:.1}% first→last={first_last_drop_pct:.1}% (leak guard uses first→last)"),
        format!(
            "thresholds: first→last drop<={:.0}% min_tps>={:.0}",
            ctx.thresholds.max_degradation_pct, ctx.thresholds.min_tps
        ),
    ];
    Ok(TestResult {
        id: "sustained".into(),
        name: "Sustained throughput stability".into(),
        category: Category::Degradation,
        description: "Repeated identical requests must not degrade in throughput (leak guard).".into(),
        prompt: prompt.into(),
        passed: ctx.do_assert.then_some(pass),
        notes,
        metrics: vec![
            ("min_tps".into(), min_tps),
            ("mean_tps".into(), mean_tps),
            ("max_tps".into(), max_tps),
            ("degradation_pct".into(), degradation_pct),
            ("first_last_drop_pct".into(), first_last_drop_pct),
        ],
        samples,
        outputs,
    })
}

// ── 4. Degradation: length ramp ─────────────────────────────────────────────
fn test_ramp(ctx: &Ctx) -> Result<TestResult> {
    let prompt = "Tell me a detailed story about a journey across the sea.";
    let mut samples = Vec::with_capacity(ctx.ramp.len());
    let mut outputs = Vec::new();
    for (i, &n) in ctx.ramp.iter().enumerate() {
        let r = ctx.client.chat_blocking(prompt, n, 0.0)?;
        if i == 0 || i + 1 == ctx.ramp.len() {
            outputs.push(truncate(&r.content, 160));
        }
        samples.push(Sample {
            index: i as u32,
            max_tokens: n,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            latency_ms: r.latency_ms,
            tps: r.tps,
            ttft_ms: None,
        });
    }
    let first_tps = samples.first().map(|s| s.tps).unwrap_or(0.0);
    let last_tps = samples.last().map(|s| s.tps).unwrap_or(0.0);
    let ramp_drop_pct = if first_tps > 0.0 {
        (first_tps - last_tps) / first_tps * 100.0
    } else {
        0.0
    };
    let min_tps = samples.iter().map(|s| s.tps).fold(f64::INFINITY, f64::min);
    let pass = ramp_drop_pct <= ctx.thresholds.max_ramp_drop_pct
        && min_tps >= ctx.thresholds.min_tps;
    let mut notes = vec![
        format!("points={:?}", ctx.ramp),
        format!("tps first={first_tps:.1} last={last_tps:.1} drop={ramp_drop_pct:.1}%"),
        format!(
            "thresholds: ramp_drop<={:.0}% min_tps>={:.0}",
            ctx.thresholds.max_ramp_drop_pct, ctx.thresholds.min_tps
        ),
    ];
    for s in &samples {
        notes.push(format!("  {:>5} tok => {:.1} t/s ({:.0} ms)", s.max_tokens, s.tps, s.latency_ms));
    }
    let metrics = vec![
        ("first_tps".into(), first_tps),
        ("last_tps".into(), last_tps),
        ("ramp_drop_pct".into(), ramp_drop_pct),
        ("min_tps".into(), min_tps),
    ];
    Ok(TestResult {
        id: "length_ramp".into(),
        name: "Length ramp (KV scaling)".into(),
        category: Category::Degradation,
        description: "Throughput across growing output lengths; measures KV-cache scaling.".into(),
        prompt: prompt.into(),
        passed: ctx.do_assert.then_some(pass),
        notes,
        metrics,
        samples,
        outputs,
    })
}

// ── 5. Coherence: multi-step reasoning with a verifiable answer ─────────────
fn test_coherence(ctx: &Ctx) -> Result<TestResult> {
    // A small word-problem with a single correct numeric answer (18). The model
    // must chain: 3 crates × 4 boxes = 12 boxes; 12 × 6 apples = 72; minus 54 = 18.
    let prompt = "\
A store has 3 crates. Each crate holds 4 boxes. Each box holds 6 apples. \
The store sells 54 apples. How many apples are left? \
Think step by step, then end with a line exactly like: ANSWER: <number>.";
    let r = ctx.client.chat_blocking(prompt, 400, 0.0)?;

    // Correct final answer is 18. Accept it via the ANSWER: line, or as a
    // standalone number anywhere as a fallback.
    let answer_re = Regex::new(r"(?i)answer\s*[:=]\s*\$?\s*0*18\b").unwrap();
    let loose_re = Regex::new(r"\b18\b").unwrap();
    let answer_line = answer_re.is_match(&r.content);
    let loose = loose_re.is_match(&r.content);
    // Coherence signal: shows working (mentions an intermediate like 72 or 12)
    // and produced a properly-formatted ANSWER line.
    let shows_work = Regex::new(r"\b(72|12)\b").unwrap().is_match(&r.content);
    let has_answer_line = Regex::new(r"(?i)answer\s*[:=]").unwrap().is_match(&r.content);

    let pass = answer_line || (loose && shows_work);
    let notes = vec![
        format!("expected final answer 18 (3×4×6 − 54)"),
        format!("answer_line(18)={answer_line} loose(18)={loose} shows_work(72|12)={shows_work} formatted_answer_line={has_answer_line}"),
    ];
    Ok(TestResult {
        id: "coherence".into(),
        name: "Multi-step reasoning coherence".into(),
        category: Category::Coherence,
        description: "Multi-step word problem; final answer must be 18 with coherent working.".into(),
        prompt: prompt.into(),
        passed: ctx.do_assert.then_some(pass),
        notes,
        metrics: vec![("tps".into(), r.tps), ("latency_ms".into(), r.latency_ms)],
        samples: vec![Sample {
            index: 0,
            max_tokens: 400,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            latency_ms: r.latency_ms,
            tps: r.tps,
            ttft_ms: None,
        }],
        outputs: vec![truncate(&r.content, 800)],
    })
}
