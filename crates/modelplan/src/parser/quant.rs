//! Quantization config parsing extracted from `parser.rs` (RULES: max 300 lines/file).

use serde_json::Value;

use super::get_usize;

/// Raw quantization config from config.json.
/// Supports both uniform (global) and mixed (per-tensor override) quantization.
#[derive(Debug, Clone)]
pub struct QuantizationRaw {
    pub quant_method: String,
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
    /// Per-tensor quantization overrides, keyed by module/tensor prefix.
    /// Example keys: "model.embed_tokens", "model.layers.0.self_attn.q_proj", "lm_head".
    /// Each value is an object with optional `bits`, `group_size`.
    pub overrides: std::collections::HashMap<String, PerTensorQuant>,
}

/// Per-tensor quantization override.
#[derive(Debug, Clone, Default)]
pub struct PerTensorQuant {
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
}

/// Parse the `quantization_config` nested object from config.json.
pub(super) fn parse_quantization_config(config: &Value) -> Option<QuantizationRaw> {
    let qc = config.get("quantization_config")?;

    let quant_method = qc.get("quant_method")
        .or_else(|| qc.get("quant_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let bits = get_usize(qc, "bits");
    let group_size = get_usize(qc, "group_size");

    // Parse per-tensor overrides: any key that's an object (not a scalar)
    // and not one of the known top-level scalar keys.
    let scalar_keys = ["quant_method", "quant_type", "bits", "group_size"];
    let mut overrides = std::collections::HashMap::new();

    if let Some(obj) = qc.as_object() {
        for (key, value) in obj {
            if scalar_keys.contains(&key.as_str()) {
                continue;
            }
            // This is a per-tensor override if it's an object
            if let Some(override_obj) = value.as_object() {
                // Convert Map to Value for get_usize
                let override_val = Value::Object(override_obj.clone());
                let bits = get_usize(&override_val, "bits");
                let group_size = get_usize(&override_val, "group_size");
                overrides.insert(
                    key.clone(),
                    PerTensorQuant {
                        bits,
                        group_size,
                    },
                );
            }
        }
    }

    Some(QuantizationRaw {
        quant_method,
        bits,
        group_size,
        overrides,
    })
}
