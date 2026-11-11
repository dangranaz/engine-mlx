//! Write the report to timestamped JSON + Markdown files.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::report::{Category, Report};

/// Paths written for a run.
pub struct WrittenPaths {
    pub json: PathBuf,
    pub md: PathBuf,
}

/// Write `results/<engine>/<timestamp>.{json,md}` (timestamp already in env).
pub fn write(report: &Report, out_dir: &Path) -> Result<WrittenPaths> {
    let dir = out_dir.join(&report.environment.engine);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    // Filesystem-safe timestamp: 2026-09-09T17-14-43 -> 2026-09-09_17-14-43
    let stamp = report
        .environment
        .timestamp
        .replace(':', "-")
        .replace('T', "_");
    let stamp = stamp.split('.').next().unwrap_or(&stamp).to_string();

    let json_path = dir.join(format!("{stamp}.json"));
    let md_path = dir.join(format!("{stamp}.md"));

    let json = serde_json::to_string_pretty(report).context("serialize report json")?;
    fs::write(&json_path, json).with_context(|| format!("write {}", json_path.display()))?;

    let md = render_markdown(report);
    fs::write(&md_path, md).with_context(|| format!("write {}", md_path.display()))?;

    Ok(WrittenPaths {
        json: json_path,
        md: md_path,
    })
}

fn verdict_icon(passed: Option<bool>) -> &'static str {
    match passed {
        Some(true) => "✅ PASS",
        Some(false) => "❌ FAIL",
        None => "— (measure-only)",
    }
}

fn cat_label(c: Category) -> &'static str {
    match c {
        Category::Correctness => "Correctness",
        Category::Degradation => "Degradation",
        Category::Coherence => "Coherence",
    }
}

fn render_markdown(r: &Report) -> String {
    let e = &r.environment;
    let mut s = String::new();
    s.push_str(&format!("# engine-bench report — {}\n\n", e.engine));
    s.push_str(&format!(
        "**Overall:** {}\n\n",
        verdict_icon(r.passed)
    ));

    s.push_str("## Environment\n\n");
    s.push_str("| field | value |\n|---|---|\n");
    s.push_str(&format!("| engine | {} |\n", e.engine));
    s.push_str(&format!("| url | {} |\n", e.url));
    s.push_str(&format!("| model | {} |\n", e.model));
    s.push_str(&format!("| timestamp | {} |\n", e.timestamp));
    s.push_str(&format!("| hostname | {} |\n", e.hostname));
    s.push_str(&format!(
        "| engine_commit | {} |\n",
        e.engine_commit.as_deref().unwrap_or("—")
    ));
    s.push_str(&format!("| bench_version | {} |\n", e.bench_version));
    if let Some(t) = &r.thresholds {
        s.push_str(&format!(
            "| thresholds | min_tps={:.0}, max_degradation={:.0}%, max_ramp_drop={:.0}% |\n",
            t.min_tps, t.max_degradation_pct, t.max_ramp_drop_pct
        ));
    }
    s.push('\n');

    // Summary table
    s.push_str("## Summary\n\n");
    s.push_str("| # | test | category | verdict | key metric |\n");
    s.push_str("|---|------|----------|---------|------------|\n");
    for (i, t) in r.tests.iter().enumerate() {
        let key = summary_metric(t);
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            i + 1,
            t.name,
            cat_label(t.category),
            verdict_icon(t.passed),
            key
        ));
    }
    s.push('\n');

    // Per-test detail
    s.push_str("## Details\n\n");
    for (i, t) in r.tests.iter().enumerate() {
        s.push_str(&format!("### {}. {} — {}\n\n", i + 1, t.name, verdict_icon(t.passed)));
        s.push_str(&format!("*{}*\n\n", t.description));
        if !t.metrics.is_empty() {
            s.push_str("Metrics: ");
            let m: Vec<String> = t
                .metrics
                .iter()
                .map(|(k, v)| format!("`{k}={v:.2}`"))
                .collect();
            s.push_str(&m.join(", "));
            s.push_str("\n\n");
        }
        if !t.notes.is_empty() {
            for n in &t.notes {
                s.push_str(&format!("- {n}\n"));
            }
            s.push('\n');
        }
        if !t.prompt.is_empty() {
            s.push_str("Question (prompt sent to the model):\n\n");
            s.push_str("```\n");
            s.push_str(&t.prompt);
            s.push_str("\n```\n\n");
        }
        if !t.outputs.is_empty() {
            s.push_str("Answer (model output sample(s)):\n\n");
            for o in &t.outputs {
                s.push_str("```\n");
                s.push_str(o);
                s.push_str("\n```\n\n");
            }
        }
    }
    s
}

/// A compact key metric string for the summary table.
fn summary_metric(t: &crate::report::TestResult) -> String {
    match t.id.as_str() {
        "sustained" => format!(
            "min {:.0} / max {:.0} t/s, degr {:.0}%",
            t.metric("min_tps").unwrap_or(0.0),
            t.metric("max_tps").unwrap_or(0.0),
            t.metric("degradation_pct").unwrap_or(0.0)
        ),
        "length_ramp" => format!(
            "first {:.0} → last {:.0} t/s, drop {:.0}%",
            t.metric("first_tps").unwrap_or(0.0),
            t.metric("last_tps").unwrap_or(0.0),
            t.metric("ramp_drop_pct").unwrap_or(0.0)
        ),
        _ => format!("{:.0} t/s", t.metric("tps").unwrap_or(0.0)),
    }
}
