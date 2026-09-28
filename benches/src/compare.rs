//! `compare` — diff two or more historical result JSON reports.
//!
//! Works for engine-vs-engine (different engines, same tests) and run-vs-run
//! (same engine, different times). The first report is the baseline; deltas are
//! computed relative to it. Prints a side-by-side table and writes a Markdown
//! comparison file.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::report::Report;

/// A loaded report plus its source path.
struct Loaded {
    report: Report,
    #[allow(dead_code)]
    path: PathBuf,
    /// Short column label: "<engine>@<timestamp>".
    label: String,
}

/// The metrics we line up across reports, per test id.
/// (test_id, metric_key, human label, higher_is_better)
const METRICS: &[(&str, &str, &str, bool)] = &[
    ("factual", "tps", "factual t/s", true),
    ("instruction", "tps", "instruction t/s", true),
    ("sustained", "min_tps", "sustained min t/s", true),
    ("sustained", "mean_tps", "sustained mean t/s", true),
    ("sustained", "first_last_drop_pct", "sustained drop %", false),
    ("length_ramp", "first_tps", "ramp first t/s", true),
    ("length_ramp", "last_tps", "ramp last t/s", true),
    ("length_ramp", "ramp_drop_pct", "ramp drop %", false),
    ("coherence", "tps", "coherence t/s", true),
];

fn short_label(r: &Report) -> String {
    let ts = r
        .environment
        .timestamp
        .split('.')
        .next()
        .unwrap_or(&r.environment.timestamp)
        .replace('T', " ");
    format!("{}@{}", r.environment.engine, ts)
}

fn load(path: &Path) -> Result<Loaded> {
    let s = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let report: Report = serde_json::from_str(&s)
        .with_context(|| format!("parse report json {}", path.display()))?;
    let label = short_label(&report);
    Ok(Loaded {
        report,
        path: path.to_path_buf(),
        label,
    })
}

/// Find the newest `*.json` under `<out_dir>/<engine>/`.
fn latest_for_engine(out_dir: &Path, engine: &str) -> Result<PathBuf> {
    let dir = out_dir.join(engine);
    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("read dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    if entries.is_empty() {
        bail!("no result JSON found under {}", dir.display());
    }
    // Timestamped filenames sort lexicographically = chronologically.
    entries.sort();
    Ok(entries.pop().unwrap())
}

/// Metric value for a test id + key in a report, if present.
fn metric_of(r: &Report, test_id: &str, key: &str) -> Option<f64> {
    r.tests
        .iter()
        .find(|t| t.id == test_id)
        .and_then(|t| t.metric(key))
}

/// Verdict string for a report (overall).
fn verdict(r: &Report) -> &'static str {
    match r.report_passed() {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "—",
    }
}

pub fn run(args: &crate::config::CompareArgs) -> Result<()> {
    // Gather sources: explicit files first, then --latest engines.
    let mut paths: Vec<PathBuf> = args.files.clone();
    for engine in &args.latest {
        paths.push(latest_for_engine(&args.out_dir, engine)?);
    }
    if paths.len() < 2 {
        bail!(
            "compare needs at least 2 reports (got {}). Pass files or --latest <engine,engine>.",
            paths.len()
        );
    }

    let loaded: Vec<Loaded> = paths.iter().map(|p| load(p)).collect::<Result<_>>()?;
    let baseline = &loaded[0];

    // ── Build the comparison text (also written to MD) ──
    let mut out = String::new();
    out.push_str("# engine-bench comparison\n\n");
    out.push_str(&format!("Baseline: **{}**\n\n", baseline.label));

    // Sources table
    out.push_str("## Sources\n\n");
    out.push_str("| col | engine | model | timestamp | commit | verdict |\n");
    out.push_str("|-----|--------|-------|-----------|--------|---------|\n");
    for (i, l) in loaded.iter().enumerate() {
        let e = &l.report.environment;
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            if i == 0 { "base".into() } else { format!("#{i}") },
            e.engine,
            e.model,
            e.timestamp,
            e.engine_commit.as_deref().unwrap_or("—"),
            verdict(&l.report),
        ));
    }
    out.push('\n');

    // Metric matrix: one row per metric, one column per report, delta vs baseline.
    out.push_str("## Metrics (Δ = vs baseline)\n\n");
    let mut header = String::from("| metric | base ");
    for l in loaded.iter().skip(1) {
        header.push_str(&format!("| {} | Δ ", l.label));
    }
    header.push_str("|\n");
    out.push_str(&header);

    let mut sep = String::from("|---|---:");
    for _ in loaded.iter().skip(1) {
        sep.push_str("|---:|---:");
    }
    sep.push_str("|\n");
    out.push_str(&sep);

    for (test_id, key, label, higher_better) in METRICS {
        let base_val = metric_of(&baseline.report, test_id, key);
        let mut row = format!(
            "| {} | {} ",
            label,
            base_val.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into())
        );
        for l in loaded.iter().skip(1) {
            let v = metric_of(&l.report, test_id, key);
            let cell = v.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
            let delta = match (base_val, v) {
                (Some(b), Some(v)) if b.abs() > 1e-9 => {
                    let pct = (v - b) / b.abs() * 100.0;
                    let good = if *higher_better { pct >= 0.0 } else { pct <= 0.0 };
                    let arrow = if pct.abs() < 0.05 {
                        "="
                    } else if good {
                        "▲"
                    } else {
                        "▼"
                    };
                    format!("{arrow}{pct:+.1}%")
                }
                _ => "—".into(),
            };
            row.push_str(&format!("| {cell} | {delta} "));
        }
        row.push_str("|\n");
        out.push_str(&row);
    }
    out.push('\n');
    out.push_str("_Δ arrows: ▲ better than baseline, ▼ worse, = unchanged (higher t/s is better; lower drop% is better)._\n");

    // Print to console.
    print!("{out}");

    // Write MD.
    let md_path = match &args.md_out {
        Some(p) => p.clone(),
        None => {
            let dir = args.out_dir.join("compare");
            fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
            let stamp = chrono::Local::now()
                .to_rfc3339()
                .replace(':', "-")
                .replace('T', "_");
            let stamp = stamp.split('.').next().unwrap_or(&stamp).to_string();
            dir.join(format!("{stamp}.md"))
        }
    };
    if let Some(parent) = md_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&md_path, &out).with_context(|| format!("write {}", md_path.display()))?;
    println!("\ncomparison MD: {}", md_path.display());

    if loaded.is_empty() {
        return Err(anyhow!("nothing compared"));
    }
    Ok(())
}
