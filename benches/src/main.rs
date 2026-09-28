//! nxm-bench — cross-engine OpenAI-compatible HTTP benchmark.

mod client;
mod compare;
mod config;
mod report;
mod tests;
mod writer;

use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;

use client::Client;
use config::{parse_u32_list, Cli, Command, RunArgs, Thresholds};
use report::{Environment, Report};

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn git_commit(repo: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["-C", &repo.to_string_lossy(), "rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn run_bench(cli: &RunArgs) -> Result<bool> {
    let ramp = parse_u32_list(&cli.ramp)?;

    let client_probe = Client::new(&cli.url, "probe");
    // Resolve model id: explicit > /v1/models > "default".
    let model = cli
        .model
        .clone()
        .or_else(|| client_probe.first_model_id())
        .unwrap_or_else(|| "default".into());

    println!("engine-bench: engine={} url={} model={}", cli.engine, cli.url, model);
    if !client_probe.health() {
        eprintln!(
            "warning: {}/health did not return 200 — is the server up? proceeding anyway.",
            cli.url
        );
    }

    let client = Client::new(&cli.url, &model);
    let thresholds = Thresholds::resolve(&cli)?;

    let ctx = tests::Ctx {
        client: &client,
        thresholds: &thresholds,
        do_assert: cli.assert,
        sustained_reps: cli.sustained_reps,
        sustained_tokens: cli.sustained_tokens,
        ramp,
    };

    let test_results = tests::run_all(&ctx)?;

    let env = Environment {
        engine: cli.engine.clone(),
        url: cli.url.clone(),
        model,
        timestamp: chrono::Local::now().to_rfc3339(),
        hostname: hostname(),
        engine_commit: cli.engine_repo.as_deref().and_then(git_commit),
        bench_version: env!("CARGO_PKG_VERSION").into(),
    };

    let mut rpt = Report {
        environment: env,
        thresholds: cli.assert.then(|| thresholds.clone()),
        passed: None,
        tests: test_results,
    };
    rpt.compute_overall();

    let paths = writer::write(&rpt, &cli.out_dir)?;

    // Console summary.
    println!("\n── results ──");
    for (i, t) in rpt.tests.iter().enumerate() {
        let v = match t.passed {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "----",
        };
        println!("  {}. [{}] {} {}", i + 1, v, t.name, notes_tail(t));
    }
    println!("\nreport JSON: {}", paths.json.display());
    println!("report  MD: {}", paths.md.display());

    let overall_pass = match rpt.passed {
        Some(p) => {
            println!("overall: {}", if p { "PASS ✅" } else { "FAIL ❌" });
            p
        }
        None => {
            println!("overall: measure-only (no --assert)");
            true
        }
    };
    Ok(overall_pass)
}

/// A short trailing metric for the console line.
fn notes_tail(t: &report::TestResult) -> String {
    match t.id.as_str() {
        "sustained" => format!(
            "(min {:.0}/max {:.0} t/s, degr {:.0}%)",
            t.metric("min_tps").unwrap_or(0.0),
            t.metric("max_tps").unwrap_or(0.0),
            t.metric("degradation_pct").unwrap_or(0.0)
        ),
        "length_ramp" => format!(
            "(drop {:.0}%)",
            t.metric("ramp_drop_pct").unwrap_or(0.0)
        ),
        _ => format!("({:.0} t/s)", t.metric("tps").unwrap_or(0.0)),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Run(args) => run_bench(args),
        Command::Compare(args) => compare::run(args).map(|_| true),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
