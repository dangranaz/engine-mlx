use crate::manifest::LayersConfig;
use crate::parser::RawModelConfig;

pub fn classify_layers(config: &RawModelConfig) -> LayersConfig {
    let total = config.num_hidden_layers;
    let num_experts = config.num_experts.unwrap_or(0);

    let (attention_layers, gdn_layers) = match config.model_type.as_str() {
        "ornith" => {
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
            let attn: Vec<usize> = (0..total).collect();
            (attn, Vec::new())
        }
        "mamba" | "mamba2" => (Vec::new(), Vec::new()),
        _ => {
            let attn: Vec<usize> = (0..total).collect();
            (attn, Vec::new())
        }
    };

    let moe_layers = if num_experts > 0 {
        match config.model_type.as_str() {
            "deepseek_v2" | "deepseek_v3" => (1..total).collect(),
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
