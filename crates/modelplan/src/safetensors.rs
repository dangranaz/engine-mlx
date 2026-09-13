//! Safetensors metadata parsing.
//!
//! Handles both sharded models (via `*.safetensors.index.json`) and
//! single-file / degraded models (counts shards via `ls *.safetensors`).

use anyhow::{Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::manifest::SafetensorsInfo;

pub fn parse_safetensors(path: &Path) -> Result<(SafetensorsInfo, Vec<String>)> {
    if let Ok(index) = parse_safetensors_index(path) {
        return Ok(index);
    }
    parse_safetensors_scan(path)
}

fn parse_safetensors_scan(path: &Path) -> Result<(SafetensorsInfo, Vec<String>)> {
    let mut total_size_bytes: u64 = 0;
    let mut tensor_names: Vec<String> = Vec::new();

    let entries: Vec<_> = fs::read_dir(path)
        .with_context(|| format!("Failed to read directory {}", path.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    let num_shards = entries.len();

    for entry in &entries {
        let file_path = entry.path();
        total_size_bytes += fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);

        if let Ok(names) = parse_safetensors_header(&file_path) {
            tensor_names.extend(names);
        }
    }

    Ok((
        SafetensorsInfo {
            num_shards,
            total_size_bytes,
            tensor_count: tensor_names.len(),
        },
        tensor_names,
    ))
}

fn parse_safetensors_index(path: &Path) -> Result<(SafetensorsInfo, Vec<String>)> {
    let index_files: Vec<_> = fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.ends_with(".safetensors.index.json"))
                .unwrap_or(false)
        })
        .collect();

    if index_files.is_empty() {
        anyhow::bail!("no safetensors index")
    }

    let index_path = index_files[0].path();
    let index_str = fs::read_to_string(&index_path)?;
    let index: Value =
        serde_json::from_str(&index_str).with_context(|| "Failed to parse safetensors index")?;

    let tensor_names: Vec<String> = index
        .get("weight_map")
        .and_then(|w| w.as_object())
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();

    let shard_files: HashSet<String> = index
        .get("weight_map")
        .and_then(|w| w.as_object())
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let num_shards = shard_files.len();
    let mut total_size_bytes: u64 = 0;
    for shard in &shard_files {
        total_size_bytes += fs::metadata(path.join(shard)).map(|m| m.len()).unwrap_or(0);
    }

    Ok((
        SafetensorsInfo {
            num_shards,
            total_size_bytes,
            tensor_count: tensor_names.len(),
        },
        tensor_names,
    ))
}

fn parse_safetensors_header(path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 8 {
        anyhow::bail!("File too small to be safetensors");
    }

    // First 8 bytes: header size as u64 LE
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;

    if mmap.len() < 8 + header_size {
        anyhow::bail!("File too small for declared header size");
    }

    let header_json = &mmap[8..8 + header_size];
    let header: Value = serde_json::from_slice(header_json)
        .with_context(|| "Failed to parse safetensors header JSON")?;

    Ok(header
        .as_object()
        .map(|obj| {
            obj.keys()
                .filter(|k| *k != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default())
}
