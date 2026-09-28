//! Parser for model directories.
//!
//! Reads config.json, tokenizer_config.json, and safetensors headers
//! to extract raw model configuration without loading weights.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

use crate::manifest::SafetensorsInfo;

/// Raw model configuration extracted from the model directory.
#[derive(Debug, Clone)]
pub struct RawModelConfig {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub num_shared_experts: Option<usize>,
    pub linear_attention: Option<LinearAttentionRaw>,
    pub quantization_config: Option<QuantizationRaw>,
    pub safetensors_info: SafetensorsInfo,
    pub tensor_names: Vec<String>,
    pub tie_word_embeddings: bool,
}

/// Raw linear attention config from config.json.
#[derive(Debug, Clone)]
pub struct LinearAttentionRaw {
    pub attn_type: String,
    pub dk: Option<usize>,
    pub dv: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub conv_kernel_size: Option<usize>,
}

/// Raw quantization config from config.json.
#[derive(Debug, Clone)]
pub struct QuantizationRaw {
    pub quant_method: String,
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
}

/// Parse a model directory, extracting all metadata.
pub fn parse_model_dir(path: &Path) -> Result<RawModelConfig> {
    let config_path = path.join("config.json");
    let config_str = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let config: Value = serde_json::from_str(&config_str)
        .with_context(|| "Failed to parse config.json")?;

    let model_type = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let architectures = config.get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let hidden_size = get_usize(&config, "hidden_size").unwrap_or(0);
    let num_hidden_layers = get_usize(&config, "num_hidden_layers").unwrap_or(0);
    let num_attention_heads = get_usize(&config, "num_attention_heads").unwrap_or(0);
    let num_key_value_heads = get_usize(&config, "num_key_value_heads")
        .unwrap_or(num_attention_heads);
    let intermediate_size = get_usize(&config, "intermediate_size").unwrap_or(0);
    let vocab_size = get_usize(&config, "vocab_size").unwrap_or(0);
    let max_position_embeddings = get_usize(&config, "max_position_embeddings").unwrap_or(2048);
    let rope_theta = config.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);
    let rms_norm_eps = config.get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6);
    let sliding_window = get_usize(&config, "sliding_window");
    let head_dim = get_usize(&config, "head_dim");
    let num_experts = get_usize(&config, "num_experts")
        .or_else(|| get_usize(&config, "num_local_experts"));
    let num_experts_per_tok = get_usize(&config, "num_experts_per_tok")
        .or_else(|| get_usize(&config, "num_experts_per_token"));
    let num_shared_experts = get_usize(&config, "num_shared_experts");
    let tie_word_embeddings = config.get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Parse linear_attention nested config
    let linear_attention = parse_linear_attention(&config);

    // Parse quantization_config nested config
    let quantization_config = parse_quantization_config(&config);

    // Parse safetensors metadata
    let (safetensors_info, tensor_names) = parse_safetensors(path)?;

    // Fallback vocab_size from tokenizer_config.json
    let vocab_size = if vocab_size == 0 {
        parse_tokenizer_vocab_size(path).unwrap_or(0)
    } else {
        vocab_size
    };

    Ok(RawModelConfig {
        model_type,
        architectures,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
        sliding_window,
        head_dim,
        num_experts,
        num_experts_per_tok,
        num_shared_experts,
        linear_attention,
        quantization_config,
        safetensors_info,
        tensor_names,
        tie_word_embeddings,
    })
}

fn parse_linear_attention(config: &Value) -> Option<LinearAttentionRaw> {
    // Check multiple possible locations for linear attention config
    let la_config = config.get("linear_attention_config")
        .or_else(|| config.get("linear_attention"))
        .or_else(|| config.get("gated_delta_net_config"))?;

    let attn_type = la_config.get("type")
        .or_else(|| la_config.get("attn_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("gated_delta_net")
        .to_string();

    Some(LinearAttentionRaw {
        attn_type,
        dk: get_usize(la_config, "dk").or_else(|| get_usize(la_config, "head_dim")),
        dv: get_usize(la_config, "dv").or_else(|| get_usize(la_config, "head_dim")),
        num_heads: get_usize(la_config, "num_heads"),
        num_kv_heads: get_usize(la_config, "num_kv_heads"),
        conv_kernel_size: get_usize(la_config, "conv_kernel_size")
            .or_else(|| get_usize(la_config, "conv_size")),
    })
}

fn parse_quantization_config(config: &Value) -> Option<QuantizationRaw> {
    let qc = config.get("quantization_config")?;

    let quant_method = qc.get("quant_method")
        .or_else(|| qc.get("quant_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    Some(QuantizationRaw {
        quant_method,
        bits: get_usize(qc, "bits"),
        group_size: get_usize(qc, "group_size"),
    })
}

/// Parse safetensors headers without loading weights.
/// Format: first 8 bytes are u64 LE header size, then JSON header.
fn parse_safetensors(path: &Path) -> Result<(SafetensorsInfo, Vec<String>)> {
    let mut total_size_bytes: u64 = 0;
    let mut tensor_count: usize = 0;
    let mut tensor_names: Vec<String> = Vec::new();

    // Find all .safetensors files
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
        let file_size = fs::metadata(&file_path)
            .map(|m| m.len())
            .unwrap_or(0);
        total_size_bytes += file_size;

        // Read only the header
        match parse_safetensors_header(&file_path) {
            Ok(names) => {
                tensor_count += names.len();
                tensor_names.extend(names);
            }
            Err(_) => {}
        }
    }

    Ok((
        SafetensorsInfo {
            num_shards,
            total_size_bytes,
            tensor_count,
        },
        tensor_names,
    ))
}

/// Parse a single safetensors file header.
fn parse_safetensors_header(path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

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

    let names: Vec<String> = header
        .as_object()
        .map(|obj| {
            obj.keys()
                .filter(|k| *k != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    Ok(names)
}

/// Parse tokenizer_config.json for vocab_size fallback.
fn parse_tokenizer_vocab_size(path: &Path) -> Option<usize> {
    let tokenizer_path = path.join("tokenizer_config.json");
    let content = fs::read_to_string(&tokenizer_path).ok()?;
    let config: Value = serde_json::from_str(&content).ok()?;
    get_usize(&config, "vocab_size")
}

/// Helper to extract a usize from a JSON value.
fn get_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|v| v.as_u64().map(|n| n as usize))
}
