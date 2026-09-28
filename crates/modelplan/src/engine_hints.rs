use crate::manifest::{EngineHints, OpsRequired};
use crate::parser::RawModelConfig;

pub fn compute_engine_hints(config: &RawModelConfig, ops: &OpsRequired) -> EngineHints {
    let num_experts = config.num_experts.unwrap_or(0);

    // embed_cpu_resident: embedding too large for GPU (vocab*hidden*4B > 500 MiB)
    let embed_bytes = config.vocab_size as u64 * config.hidden_size as u64 * 4;
    let embed_cpu_resident = embed_bytes > 500 * 1024 * 1024;

    // expert_streaming: more than 16 experts
    let expert_streaming = num_experts > 16;

    let max_scratch_bytes = estimate_scratch_bytes(config);
    let metal_pipelines_needed = ops.count();

    // cold_start_safe: model fits in reasonable memory
    let cold_start_safe = config.safetensors_info.total_size_bytes < 16 * 1024 * 1024 * 1024;

    EngineHints {
        embed_cpu_resident,
        expert_streaming,
        max_scratch_bytes,
        metal_pipelines_needed,
        cold_start_safe,
    }
}

fn estimate_scratch_bytes(config: &RawModelConfig) -> u64 {
    let head_dim = config.head_dim.unwrap_or_else(|| {
        config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .unwrap_or(128)
    });
    // Conservative: 4 * hidden_size * max(intermediate_size, hidden_size) * sizeof(f16)
    let scratch = (config.hidden_size as u64).max(config.intermediate_size as u64)
        * (head_dim as u64)
        * 4
        * 2;

    scratch.clamp(16 * 1024 * 1024, 256 * 1024 * 1024)
}
