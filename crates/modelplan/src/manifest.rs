//! Model manifest types for nxm‑modelplan.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Top‑level manifest produced by introspection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelManifest {
    pub model: ModelInfo,
    pub layers: LayersConfig,
    pub attention: Option<AttentionConfig>,
    pub gdn: Option<GdnConfig>,
    pub moe: Option<MoEConfig>,
    pub mlp: MlpConfig,
    pub norm: NormConfig,
    pub embedding: EmbeddingConfig,
    pub quant: Option<QuantConfig>,
    pub safetensors: SafetensorsInfo,
    pub ops_required: OpsRequired,
    pub engine_hints: EngineHints,
}

/// Basic model identification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    pub family: String,
    pub architecture: String,
    pub params_b: f64,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

/// Layer count and classification by type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayersConfig {
    pub total: usize,
    pub attention_layers: Vec<usize>,
    pub gdn_layers: Vec<usize>,
    pub moe_layers: Vec<usize>,
}

/// Attention configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub attn_type: String, // "sliding", "full", "hybrid"
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub sliding_window: Option<usize>,
    pub rope_theta: f64,
}

/// Gated Delta Net configuration (linear attention variant).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GdnConfig {
    pub dk: usize,
    pub dv: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub conv_kernel_size: usize,
}

/// Mixture of Experts configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoEConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize,
    pub intermediate_size: usize,
}

/// MLP configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MlpConfig {
    pub mlp_type: String, // "swiglu", "gelu", "relu"
    pub intermediate_size: usize,
}

/// Normalization configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NormConfig {
    pub norm_type: String, // "rms_norm", "layer_norm"
    pub eps: f64,
}

/// Embedding configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub emb_type: String, // "standard", "rotary"
    pub tied_lm_head: bool,
}

/// Quantization configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantConfig {
    pub format: String, // "q4", "mxfp4", "mxfp8", "awq", "none"
    pub scales_dtype: String,
    pub embedding_dtype: String,
    pub lm_head_dtype: String,
    pub group_size: usize,
}

/// Safetensors file metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SafetensorsInfo {
    pub num_shards: usize,
    pub total_size_bytes: u64,
    pub tensor_count: usize,
}

/// Boolean flags for each operation the model requires.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpsRequired {
    pub rmsnorm: bool,
    pub rope: bool,
    pub flash_attn_decode: bool,
    pub fused_gdn_step: bool,
    pub quant_matmul_q4: bool,
    pub quant_matmul_mxfp4: bool,
    pub quant_matmul_mxfp8: bool,
    pub moe_routing: bool,
    pub silu: bool,
    pub swiglu: bool,
    pub residual_add: bool,
    pub embedding_lookup: bool,
    pub lm_head_matmul: bool,
    pub conv1d: bool,
    pub mamba2_scan: bool,
}

/// Hints for the engine to optimise boot and runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineHints {
    pub embed_cpu_resident: bool,
    pub expert_streaming: bool,
    pub max_scratch_bytes: u64,
    pub metal_pipelines_needed: usize,
    pub cold_start_safe: bool,
}

impl Default for EngineHints {
    fn default() -> Self {
        Self {
            embed_cpu_resident: false,
            expert_streaming: false,
            max_scratch_bytes: 64 * 1024 * 1024, // 64 MiB default
            metal_pipelines_needed: 6,
            cold_start_safe: true,
        }
    }
}

impl fmt::Display for ModelManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╭─ Model: {} ─────────────────────────────────╮", self.model.family)?;
        writeln!(f, "│ Family: {:<40}│", self.model.family)?;
        writeln!(f, "│ Arch:   {:<40}│", self.model.architecture)?;
        writeln!(f, "│ Params: {:<40}│", format!("{:.1}B", self.model.params_b))?;
        writeln!(f, "│ Vocab:  {:<40}│", self.model.vocab_size)?;
        writeln!(f, "│ Context:{:<40}│", self.model.max_position_embeddings)?;
        writeln!(f, "│ Layers: {:<40}│", self.layers.total)?;
        if let Some(ref quant) = self.quant {
            writeln!(f, "│ Quant:  {:<40}│", format!("{} (group={})", quant.format, quant.group_size))?;
        }
        writeln!(f, "├─ Ops Required ─────────────────────────────────────┤")?;
        let ops = self.ops_required.as_list();
        for chunk in ops.chunks(4) {
            let line = chunk.join(", ");
            writeln!(f, "│ ✓ {:<49}│", line)?;
        }
        writeln!(f, "├─ Engine Hints ─────────────────────────────────────┤")?;
        writeln!(f, "│ Metal pipelines: {:<35}│", self.engine_hints.metal_pipelines_needed)?;
        writeln!(f, "│ Scratch bytes:   {:<35}│", format_bytes(self.engine_hints.max_scratch_bytes))?;
        writeln!(f, "│ Cold‑start safe: {:<35}│", self.engine_hints.cold_start_safe)?;
        writeln!(f, "╰────────────────────────────────────────────────────╯")?;
        Ok(())
    }
}

impl OpsRequired {
    /// Return a list of enabled op names.
    pub fn as_list(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();
        if self.rmsnorm { ops.push("rmsnorm"); }
        if self.rope { ops.push("rope"); }
        if self.flash_attn_decode { ops.push("flash_attn_decode"); }
        if self.fused_gdn_step { ops.push("fused_gdn_step"); }
        if self.quant_matmul_q4 { ops.push("quant_matmul_q4"); }
        if self.quant_matmul_mxfp4 { ops.push("quant_matmul_mxfp4"); }
        if self.quant_matmul_mxfp8 { ops.push("quant_matmul_mxfp8"); }
        if self.moe_routing { ops.push("moe_routing"); }
        if self.silu { ops.push("silu"); }
        if self.swiglu { ops.push("swiglu"); }
        if self.residual_add { ops.push("residual_add"); }
        if self.embedding_lookup { ops.push("embedding_lookup"); }
        if self.lm_head_matmul { ops.push("lm_head_matmul"); }
        if self.conv1d { ops.push("conv1d"); }
        if self.mamba2_scan { ops.push("mamba2_scan"); }
        ops
    }
    pub fn count(&self) -> usize { self.as_list().len() }
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
