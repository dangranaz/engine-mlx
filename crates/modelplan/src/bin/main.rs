use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::Path;

use nxm_modelplan::{introspect, load_plan, ensure_plan, plan::from_manifest, plan::EnginePlan};

#[derive(Parser)]
#[command(name = "nxm-modelplan")]
#[command(about = "Model introspection – generate ModelManifest from model directory")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a model directory and print the manifest
    Inspect { model_dir: String },
    /// Introspect and save the manifest (model.plan.toml + sidecar)
    Save { model_dir: String },
    /// Check cache validity (valid / stale / missing)
    Check { model_dir: String },
    /// Show engine plan summary derived from manifest
    Plan { model_dir: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { model_dir } => {
            let manifest = introspect(Path::new(&model_dir))?;
            println!("{}", manifest);
        }
        Commands::Save { model_dir } => {
            let manifest = ensure_plan(Path::new(&model_dir))?;
            println!("✓ Manifest saved to {}/model.plan.toml", model_dir);
            println!("  family: {}", manifest.model.family);
            println!("  layers: {}", manifest.layers.total);
            println!("  attention: {}", manifest.attention.as_ref().map(|a| &a.attn_type).unwrap_or(&"none".to_string()));
        }
        Commands::Check { model_dir } => {
            let dir = Path::new(&model_dir);
            let plan_path = dir.join("model.plan.toml");
            let sidecar_path = dir.join("model.plan.toml.sha256");

            if !plan_path.exists() {
                println!("missing");
                return Ok(());
            }
            if !sidecar_path.exists() {
                println!("stale (no sidecar)");
                return Ok(());
            }
            match load_plan(dir)? {
                Some(_) => println!("valid"),
                None => println!("stale"),
            }
        }
        Commands::Plan { model_dir } => {
            let manifest = introspect(Path::new(&model_dir))?;
            let engine_plan: EnginePlan = from_manifest(&manifest);
            println!("EnginePlan for {}", manifest.model.family);
            println!("  layers: {}", engine_plan.layers.len());
            println!("  required kernels: {}", engine_plan.required_kernels.len());
            for k in &engine_plan.required_kernels {
                println!("    - {:?}", k);
            }
            println!("  scratch: attention={} mlp={} total={}",
                engine_plan.scratch_sizes.attention_scratch,
                engine_plan.scratch_sizes.mlp_scratch,
                engine_plan.scratch_sizes.total);
        }
    }
    Ok(())
}