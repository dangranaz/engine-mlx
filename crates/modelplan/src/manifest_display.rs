//! Human-readable rendering of a `ModelManifest`.

use std::fmt;

use crate::manifest::ModelManifest;

impl fmt::Display for ModelManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "╭─ Model: {} ─────────────────────────────────╮",
            self.model.family
        )?;
        writeln!(f, "│ Family: {:<40}│", self.model.family)?;
        writeln!(f, "│ Arch:   {:<40}│", self.model.architecture)?;
        writeln!(
            f,
            "│ Params: {:<40}│",
            format!("{:.1}B", self.model.params_b)
        )?;
        writeln!(f, "│ Vocab:  {:<40}│", self.model.vocab_size)?;
        writeln!(f, "│ Context:{:<40}│", self.model.max_position_embeddings)?;
        writeln!(f, "│ Layers: {:<40}│", self.layers.total)?;
        if let Some(ref quant) = self.quant {
            writeln!(
                f,
                "│ Quant:  {:<40}│",
                format!("{} (group={})", quant.format, quant.group_size)
            )?;
        }
        writeln!(f, "├─ Ops Required ─────────────────────────────────────┤")?;
        let ops = self.ops_required.as_list();
        for chunk in ops.chunks(4) {
            let line = chunk.join(", ");
            writeln!(f, "│ ✓ {:<49}│", line)?;
        }
        writeln!(f, "├─ Engine Hints ─────────────────────────────────────┤")?;
        writeln!(
            f,
            "│ Metal pipelines: {:<35}│",
            self.engine_hints.metal_pipelines_needed
        )?;
        writeln!(
            f,
            "│ Scratch bytes:   {:<35}│",
            format_bytes(self.engine_hints.max_scratch_bytes)
        )?;
        writeln!(
            f,
            "│ Cold‑start safe: {:<35}│",
            self.engine_hints.cold_start_safe
        )?;
        writeln!(f, "╰────────────────────────────────────────────────────╯")?;
        Ok(())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
