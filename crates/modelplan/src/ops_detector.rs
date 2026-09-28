//! Ops detector: maps model architecture to required operations.
//!
//! Given a `RawModelConfig`, deduces the set of kernel operations,
//! engine hints, and layer classification.

use crate::manifest::{EngineHints, LayersConfig, OpsRequired};
use crate::parser::RawModelConfig;

/// Detect required operations, engine hints, and layer config from raw model config.
pub fn detect_ops(config: &RawModelConfig) -> (OpsRequired, EngineHints, LayersConfig) {
    let mut ops = base_ops_for_model_type(&config.model_type);

    // MoE detection
    if config.num_experts.unwrap_or(0) > 0 {
        ops.moe_routing = true;
    }

    // GDN / linear attention detection
    if config.linear_attention.is_some() || has_gdn_tensors(&config.tensor_names) {
        ops.fused_gdn_step = true;
        ops.conv1d = true;
    }

    // Quantization format detection
    if let Some(ref qc) = config.quantization_config {
        match qc.quant_method.as_str() {
            "mlx" | "gptq" | "awq" => {
                if qc.bits == Some(4) {
                    ops.quant_matmul_q4 = true;
                }
            }
            _ => {}
        }
        // Check for mxfp4 in method name
        if qc.quant_method.contains("mxfp4") {
            ops.quant_matmul_mxfp4 = true;
        }
        if qc.quant_method.contains("mxfp8") {
            ops.quant_matmul_mxfp8 = true;
        }
    }

    // Also check tensor names for quant format hints
    if config.tensor_names.iter().any(|n| n.contains("scales") && n.contains("mxfp4")) {
        ops.quant_matmul_mxfp4 = true;
    }

    // Engine hints
    let hints = compute_engine_hints(config, &ops);

    // Layer classification
    let layers = classify_layers(config);

    (ops, hints, layers)
}

/// Base ops for a given model_type string.
fn base_ops_for_model_type(model_type: &str) -> OpsRequired {
    let mut ops = OpsRequired::default();

    // Common ops for all transformer models
    ops.rmsnorm = true;
    ops.residual_add = true;
    ops.embedding_lookup = true;
    ops.lm_head_matmul = true;

    match model_type {
        "qwen2" | "qwen3" | "llama" | "phi" | "phi3" | "minicpm" | "smollm" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
        }
        "qwen3_moe" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
            ops.moe_routing = true;
        }
        "gemma" | "gemma2" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
        }
        "gemma4" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
            ops.moe_routing = true;
        }
        "deepseek_v2" | "deepseek_v3" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
            ops.moe_routing = true;
        }
        "ornith" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.fused_gdn_step = true;
            ops.conv1d = true;
            ops.silu = true;
            ops.swiglu = true;
            ops.moe_routing = true;
        }
        "lfm" => {
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.conv1d = true;
            ops.silu = true;
            ops.swiglu = true;
            ops.moe_routing = true;
        }
        "mamba" | "mamba2" => {
            ops.mamba2_scan = true;
            ops.conv1d = true;
            ops.silu = true;
            // No attention‑based ops
            ops.rope = false;
            ops.flash_attn_decode = false;
        }
        _ => {
            // Unknown model type — conservative defaults (standard transformer)
            ops.rope = true;
            ops.flash_attn_decode = true;
            ops.silu = true;
            ops.swiglu = true;
        }
    }
    ops
}

/// Check if tensor names contain GDN/delta indicators.
fn has_gdn_tensors(tensor_names: &[String]) -> bool {
    tensor_names.iter().any(|name| {
        name.contains("gdn") || name.contains("delta_net") || name.contains("gated_delta")
    })
}

/// Compute engine hints based on model config and ops.
fn compute_engine_hints(config: &RawModelConfig, ops: &OpsRequired) -> EngineHints {
    let num_experts = config.num_experts.unwrap_or(0);

    // embed_cpu_resident: if embedding is f32 and too large for GPU
    // Heuristic: vocab_size * hidden_size * 4 bytes > 500 MiB
    let embed_bytes = config.vocab_size as u64 * config.hidden_size as u64 * 4;
    let embed_cpu_resident = embed_bytes > 500 * 1024 * 1024;

    // expert_streaming: if more than 16 experts
    let expert_streaming = num_experts > 16;

    // max_scratch_bytes: estimate based on model dimensions
    let max_scratch_bytes = estimate_scratch_bytes(config);

    // metal_pipelines_needed: count unique ops
    let metal_pipelines_needed = ops.count();

    // cold_start_safe: true if model fits in reasonable memory
    let cold_start_safe = config.safetensors_info.total_size_bytes < 16 * 1024 * 1024 * 1024;

    EngineHints {
        embed_cpu_resident,
        expert_streaming,
        max_scratch_bytes,
        metal_pipelines_needed,
        cold_start_safe,
    }
}

/// Estimate scratch buffer size needed.
fn estimate_scratch_bytes(config: &RawModelConfig) -> u64 {
    let head_dim = config.head_dim.unwrap_or(
        if config.num_attention_heads > 0 {
            config.hidden_size / config.num_attention_heads
        } else {
            128
        }
    );
    // Scratch ≈ batch * seq_len * hidden + attention scratch
    // Conservative: 4 * hidden_size * max(intermediate_size, hidden_size) * sizeof(f16)
    let scratch = (config.hidden_size as u64)
        .max(config.intermediate_size as u64)
        * (head_dim as u64)
        * 4 // multiple buffers
        * 2; // f16

    // Clamp to reasonable range: min 16 MiB, max 256 MiB
    scratch.clamp(16 * 1024 * 1024, 256 * 1024 * 1024)
}

/// Classify layers into attention, GDN, and MoE categories.
fn classify_layers(config: &RawModelConfig) -> LayersConfig {
    let total = config.num_hidden_layers;
    let num_experts = config.num_experts.unwrap_or(0);

    let (attention_layers, gdn_layers) = match config.model_type.as_str() {
        "ornith" => {
            // Ornith: alternating attention and GDN layers
            // Pattern: [attn, attn, gdn, gdn, attn, attn, gdn, gdn, ...]
            let mut attn = Vec::new();
            let mut gdn = Vec::new();
            for i in 0..total {
                if (i % 4) < 2 {
                    attn.push(i);
                } else {
                    gdn.push(i);
                }
            }
            (attn, gdn)
        }
        "lfm" => {
            // LFM: first layers conv, middle layers attention, pattern varies
            let mut attn = Vec::new();
            let gdn = Vec::new();
            // All layers have attention in LFM (with conv)
            for i in 0..total {
                attn.push(i);
            }
            (attn, gdn)
        }
        "mamba" | "mamba2" => {
            // Pure SSM — no attention layers
            (Vec::new(), Vec::new())
        }
        _ => {
            // Standard transformer: all layers are attention
            let attn: Vec<usize> = (0..total).collect();
            (attn, Vec::new())
        }
    };

    // MoE layers: if num_experts > 0, all layers (or specific pattern)
    let moe_layers = if num_experts > 0 {
        match config.model_type.as_str() {
            "deepseek_v2" | "deepseek_v3" => {
                // DeepSeek: first layer is dense, rest are MoE
                (1..total).collect()
            }
            _ => (0..total).collect(),
        }
    } else {
        Vec::new()
    };

    LayersConfig {
        total,
        attention_layers,
        gdn_layers,
        moe_layers,
    }
}
