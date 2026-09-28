//! CLI arguments and pass/fail thresholds.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};

/// Cross-engine OpenAI-compatible HTTP benchmark.
///
/// `run` benchmarks a live server (5 standard tests, timestamped JSON+MD
/// report, optional pass/fail asserts). `compare` diffs two or more historical
/// result JSONs side by side with per-test deltas.
#[derive(Debug, Parser)]
#[command(name = "nxm-bench", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Run the 5 standard tests against a live server.
    Run(RunArgs),
    /// Compare two or more historical result JSON files.
    Compare(CompareArgs),
}

/// Arguments for `compare`.
#[derive(Debug, Parser)]
pub struct CompareArgs {
    /// Result JSON files to compare (2+). The first is the baseline; deltas are
    /// computed relative to it. If omitted, --latest is used.
    #[arg(value_name = "FILE")]
    pub files: Vec<PathBuf>,

    /// Auto-pick the latest result JSON for each of these engine labels under
    /// --out-dir (results/<engine>/*.json). Mutually convenient with `files`.
    #[arg(long, value_delimiter = ',')]
    pub latest: Vec<String>,

    /// Report root used with --latest.
    #[arg(short, long, default_value = "results")]
    pub out_dir: PathBuf,

    /// Optional path to write the comparison Markdown. Defaults to
    /// results/compare/<timestamp>.md.
    #[arg(long)]
    pub md_out: Option<PathBuf>,
}

/// Arguments for `run` (the original benchmark flags).
#[derive(Debug, Parser)]
pub struct RunArgs {
    /// Base URL of the OpenAI-compatible server.
    #[arg(short, long, default_value = "http://127.0.0.1:11435")]
    pub url: String,

    /// Human label for the engine under test (used in the report path).
    #[arg(short, long, default_value = "engine")]
    pub engine: String,

    /// Model id to send in requests. Defaults to the first id from /v1/models,
    /// or "default" if that lookup fails.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Sustained-degradation test: number of identical requests.
    #[arg(long, default_value_t = 20)]
    pub sustained_reps: u32,

    /// Sustained-degradation test: max_tokens per request.
    #[arg(long, default_value_t = 128)]
    pub sustained_tokens: u32,

    /// Length-ramp test points (max_tokens), comma-separated.
    #[arg(long, default_value = "128,512,1024")]
    pub ramp: String,

    /// Output directory root for reports (results/<engine>/<timestamp>.{json,md}).
    #[arg(short, long, default_value = "results")]
    pub out_dir: PathBuf,

    /// Enable pass/fail assertions + exit code. Without it, measure-only.
    #[arg(long)]
    pub assert: bool,

    /// Optional thresholds.toml overriding defaults (only used with --assert).
    #[arg(long)]
    pub thresholds: Option<PathBuf>,

    /// Override: minimum acceptable tokens/sec (with --assert).
    #[arg(long)]
    pub min_tps: Option<f64>,

    /// Override: maximum acceptable degradation percent in the sustained test.
    #[arg(long)]
    pub max_degradation_pct: Option<f64>,

    /// Git repo path for the engine, to record its commit in the report.
    #[arg(long)]
    pub engine_repo: Option<PathBuf>,
}

/// Pass/fail thresholds. Loaded from thresholds.toml then overridden by CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    /// Minimum tokens/sec any request must sustain (degradation tests).
    pub min_tps: f64,
    /// Max percent throughput may drop from the fastest to the slowest request
    /// in the sustained test. Guards against per-request leaks/degradation.
    pub max_degradation_pct: f64,
    /// Max acceptable throughput drop (percent) across the length ramp,
    /// from the shortest to the longest sequence.
    pub max_ramp_drop_pct: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_tps: 20.0,
            max_degradation_pct: 15.0,
            max_ramp_drop_pct: 60.0,
        }
    }
}

impl Thresholds {
    /// Resolve final thresholds: defaults <- thresholds.toml <- CLI overrides.
    pub fn resolve(cli: &RunArgs) -> Result<Self> {
        let mut t = if let Some(path) = &cli.thresholds {
            let s = std::fs::read_to_string(path)
                .with_context(|| format!("read thresholds file {}", path.display()))?;
            toml::from_str(&s).with_context(|| "parse thresholds.toml")?
        } else {
            Thresholds::default()
        };
        if let Some(v) = cli.min_tps {
            t.min_tps = v;
        }
        if let Some(v) = cli.max_degradation_pct {
            t.max_degradation_pct = v;
        }
        Ok(t)
    }
}

/// Parse a comma-separated list of u32 (for --ramp).
pub fn parse_u32_list(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<u32>().with_context(|| format!("invalid number '{p}'")))
        .collect()
}
