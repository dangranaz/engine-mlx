//! Parsing of nested config sections from `config.json`.

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct LinearAttentionRaw {
    pub attn_type: String,
    pub dk: Option<usize>,
    pub dv: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub conv_kernel_size: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct QuantizationRaw {
    pub quant_method: String,
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
}

pub fn parse_linear_attention(config: &Value) -> Option<LinearAttentionRaw> {
    let la_config = config
        .get("linear_attention_config")
        .or_else(|| config.get("linear_attention"))
        .or_else(|| config.get("gated_delta_net_config"))?;

    let attn_type = la_config
        .get("type")
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

pub fn parse_quantization_config(config: &Value) -> Option<QuantizationRaw> {
    let qc = config.get("quantization_config")?;

    let quant_method = qc
        .get("quant_method")
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

pub fn get_usize(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(|v| v.as_u64().map(|n| n as usize))
}
