use crate::manifest::{
    AttentionConfig, EmbeddingConfig, GdnConfig, MlpConfig, MoEConfig, ModelInfo, ModelManifest,
    NormConfig, QuantConfig,
};
use crate::ops_detector::detect_ops;
use crate::parser::RawModelConfig;

pub fn build_manifest(config: &RawModelConfig) -> ModelManifest {
    let (ops_required, engine_hints, layers) = detect_ops(config);

    let head_dim = config.head_dim.unwrap_or_else(|| {
        config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .unwrap_or(128)
    });

    let attention = if ops_required.flash_attn_decode || ops_required.rope {
        let attn_type = if config.linear_attention.is_some() {
            "hybrid".to_string()
        } else if config.sliding_window.is_some() {
            if config.max_window_layers.unwrap_or(0) < config.num_hidden_layers {
                "hybrid"
            } else {
                "sliding"
            }
            .to_string()
        } else {
            "full".to_string()
        };

        Some(AttentionConfig {
            attn_type,
            head_dim,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            sliding_window: config.sliding_window,
            rope_theta: config.rope_theta,
        })
    } else {
        None
    };

    let gdn = config.linear_attention.as_ref().map(|la| GdnConfig {
        dk: la.dk.unwrap_or(head_dim),
        dv: la.dv.unwrap_or(head_dim),
        num_heads: la.num_heads.unwrap_or(config.num_attention_heads),
        num_kv_heads: la.num_kv_heads.unwrap_or(config.num_key_value_heads),
        conv_kernel_size: la.conv_kernel_size.unwrap_or(4),
    });

    let moe = config.num_experts.and_then(|ne| {
        if ne > 0 {
            Some(MoEConfig {
                num_experts: ne,
                num_experts_per_tok: config.num_experts_per_tok.unwrap_or(2),
                num_shared_experts: config.num_shared_experts.unwrap_or(0),
                intermediate_size: config.intermediate_size,
            })
        } else {
            None
        }
    });

    let mlp_type = if ops_required.swiglu {
        "swiglu".to_string()
    } else if ops_required.silu {
        "silu".to_string()
    } else {
        "gelu".to_string()
    };

    let mlp = MlpConfig {
        mlp_type,
        intermediate_size: config.intermediate_size,
    };

    let norm = NormConfig {
        norm_type: "rms_norm".to_string(),
        eps: config.rms_norm_eps,
    };

    let embedding = EmbeddingConfig {
        emb_type: "standard".to_string(),
        tied_lm_head: config.tie_word_embeddings,
    };

    let quant = config.quantization_config.as_ref().map(|qc| {
        let format = if qc.quant_method.contains("mxfp4") {
            "mxfp4".to_string()
        } else if qc.quant_method.contains("mxfp8") {
            "mxfp8".to_string()
        } else if qc.bits == Some(4) {
            "q4".to_string()
        } else if qc.bits == Some(8) {
            "q8".to_string()
        } else {
            qc.quant_method.clone()
        };

        QuantConfig {
            format,
            scales_dtype: "f16".to_string(),
            embedding_dtype: "f32".to_string(),
            lm_head_dtype: "f32".to_string(),
            group_size: qc.group_size.unwrap_or(32),
        }
    });

    let params_b = estimate_params_b(config);

    let model_info = ModelInfo {
        family: config.model_type.clone(),
        architecture: config.architectures.first().cloned().unwrap_or_default(),
        params_b,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
        max_position_embeddings: config.max_position_embeddings,
    };

    ModelManifest {
        model: model_info,
        layers,
        attention,
        gdn,
        moe,
        mlp,
        norm,
        embedding,
        quant,
        safetensors: config.safetensors_info.clone(),
        ops_required,
        engine_hints,
    }
}

fn estimate_params_b(config: &RawModelConfig) -> f64 {
    let h = config.hidden_size as f64;
    let l = config.num_hidden_layers as f64;
    let inter = config.intermediate_size as f64;
    let vocab = config.vocab_size as f64;

    let attn_params = 4.0 * h * h * l;
    let mlp_params = 3.0 * h * inter * l;
    let emb_params = vocab * h;

    let total = attn_params + mlp_params + emb_params;
    (total / 1e9 * 10.0).round() / 10.0
}
