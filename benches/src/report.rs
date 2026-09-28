//! Report data model — serialized to JSON and rendered to Markdown.

use serde::{Deserialize, Serialize};

/// Environment block: makes every historical result self-describing.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Environment {
    /// Human label for the engine under test (e.g. "engine-mlx").
    pub engine: String,
    /// Base URL benchmarked.
    pub url: String,
    /// Model id as advertised by GET /v1/models (or the requested model).
    pub model: String,
    /// ISO-8601 local timestamp of the run.
    pub timestamp: String,
    /// Hostname of the machine running the bench.
    pub hostname: String,
    /// engine git commit, if discoverable (optional).
    pub engine_commit: Option<String>,
    /// engine-bench version.
    pub bench_version: String,
}

/// A single per-request sample inside a test.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Sample {
    pub index: u32,
    pub max_tokens: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub latency_ms: f64,
    pub tps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
}

/// The category a test belongs to (for grouping in the report).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Correctness,
    Degradation,
    Coherence,
}

/// Result of one of the 5 standard tests.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TestResult {
    pub id: String,
    pub name: String,
    pub category: Category,
    /// One-line description of what the test checks.
    pub description: String,
    /// The exact question/prompt sent to the model. Recorded so the report
    /// explains *what was asked* alongside the answer, making each benchmark
    /// self-documenting. Empty for tests that issue many identical prompts is
    /// avoided — we always store the representative prompt used.
    #[serde(default)]
    pub prompt: String,
    /// Whether the test passed. `None` when asserts were not requested
    /// (measure-only mode) — the test still records data.
    pub passed: Option<bool>,
    /// Human-readable notes (matched regex, degradation %, failure reason...).
    pub notes: Vec<String>,
    /// Key scalar metrics for quick scanning (name -> value).
    pub metrics: Vec<(String, f64)>,
    /// Full per-request samples.
    pub samples: Vec<Sample>,
    /// Representative model output(s) captured for inspection (truncated).
    pub outputs: Vec<String>,
}

impl TestResult {
    pub fn metric(&self, key: &str) -> Option<f64> {
        self.metrics.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

/// The full report.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Report {
    pub environment: Environment,
    /// The thresholds used (null when asserts disabled).
    pub thresholds: Option<crate::config::Thresholds>,
    /// Overall pass/fail (None in measure-only mode).
    pub passed: Option<bool>,
    pub tests: Vec<TestResult>,
}

impl Report {
    /// Overall verdict accessor (None in measure-only mode).
    pub fn report_passed(&self) -> Option<bool> {
        self.passed
    }

    /// Overall pass = all tests with a verdict passed.
    pub fn compute_overall(&mut self) {
        let verdicts: Vec<bool> = self.tests.iter().filter_map(|t| t.passed).collect();
        self.passed = if verdicts.is_empty() {
            None
        } else {
            Some(verdicts.iter().all(|&b| b))
        };
    }
}
