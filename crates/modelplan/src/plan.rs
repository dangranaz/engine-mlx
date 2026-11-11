//! Engine plan generation from manifest.
//!
//! Converts a `ModelManifest` into an `EnginePlan` that the engine consumes
//! to specialise its forward loop without runtime branching.

use serde::{Deserialize, Serialize};

use crate::manifest::{
    AttentionConfig, EngineHints, GdnConfig, MlpConfig, MoEConfig, ModelManifest, NormConfig,
};

/// Runtime plan consumed by the engine for specialised execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePlan {
    pub layers: Vec<LayerPlan>,
    pub scratch_sizes: ScratchSizes,
    pub required_kernels: Vec<KernelId>,
    pub hints: EngineHints,
}

/// Scratch buffer sizing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchSizes {
    pub attention_scratch: u64,
    pub mlp_scratch: u64,
    pub total: u64,
}

/// Plan for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPlan {
    pub index: usize,
    pub pre_norm: NormConfig,
    pub core: CoreOp,
    pub mlp: MlpOp,
}

/// Core operation for a layer (attention variant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoreOp {
    Attention(AttentionConfig),
    GatedDeltaNet(GdnConfig),
    LinearAttention,
    Mamba2,
}

/// MLP operation for a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MlpOp {
    Dense(MlpConfig),
    MoE(MoEConfig),
    None,
}

/// Identifies a specific Metal kernel pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelId {
    RmsNorm,
    Rope,
    FlashAttnDecode,
    FusedGdnStep,
    QuantMatmulQ4,
    QuantMatmulMxfp4,
    QuantMatmulMxfp8,
    MoeRouting,
    Silu,
    SwiGlu,
    ResidualAdd,
    EmbeddingLookup,
    LmHeadMatmul,
    Conv1d,
    Mamba2Scan,
}

/// Build an `EnginePlan` from a `ModelManifest`.
pub fn from_manifest(manifest: &ModelManifest) -> EnginePlan {
    let layers = build_layer_plans(manifest);
    let required_kernels = required_metal_pipelines_from_manifest(manifest);
    let scratch_sizes = compute_scratch_sizes(manifest);

    EnginePlan {
        layers,
        scratch_sizes,
        required_kernels,
        hints: manifest.engine_hints.clone(),
    }
}

/// Determine which Metal pipelines are needed.
pub fn required_metal_pipelines(plan: &EnginePlan) -> Vec<KernelId> {
    plan.required_kernels.clone()
}

fn build_layer_plans(manifest: &ModelManifest) -> Vec<LayerPlan> {
    let mut plans = Vec::with_capacity(manifest.layers.total);

    for i in 0..manifest.layers.total {
        let core = if manifest.layers.gdn_layers.contains(&i) {
            if let Some(ref gdn) = manifest.gdn {
                CoreOp::GatedDeltaNet(gdn.clone())
            } else {
                CoreOp::LinearAttention
            }
        } else if manifest.layers.attention_layers.contains(&i) {
            if let Some(ref attn) = manifest.attention {
                CoreOp::Attention(attn.clone())
            } else {
                CoreOp::LinearAttention
            }
        } else {
            // Mamba2 or other pure SSM
            CoreOp::Mamba2
        };

        let mlp = if manifest.layers.moe_layers.contains(&i) {
            if let Some(ref moe) = manifest.moe {
                MlpOp::MoE(moe.clone())
            } else {
                MlpOp::Dense(manifest.mlp.clone())
            }
        } else {
            MlpOp::Dense(manifest.mlp.clone())
        };

        plans.push(LayerPlan {
            index: i,
            pre_norm: manifest.norm.clone(),
            core,
            mlp,
        });
    }

    plans
}

fn required_metal_pipelines_from_manifest(manifest: &ModelManifest) -> Vec<KernelId> {
    let ops = &manifest.ops_required;
    let mut kernels = Vec::new();

    if ops.rmsnorm { kernels.push(KernelId::RmsNorm); }
    if ops.rope { kernels.push(KernelId::Rope); }
    if ops.flash_attn_decode { kernels.push(KernelId::FlashAttnDecode); }
    if ops.fused_gdn_step { kernels.push(KernelId::FusedGdnStep); }
    if ops.quant_matmul_q4 { kernels.push(KernelId::QuantMatmulQ4); }
    if ops.quant_matmul_mxfp4 { kernels.push(KernelId::QuantMatmulMxfp4); }
    if ops.quant_matmul_mxfp8 { kernels.push(KernelId::QuantMatmulMxfp8); }
    if ops.moe_routing { kernels.push(KernelId::MoeRouting); }
    if ops.silu { kernels.push(KernelId::Silu); }
    if ops.swiglu { kernels.push(KernelId::SwiGlu); }
    if ops.residual_add { kernels.push(KernelId::ResidualAdd); }
    if ops.embedding_lookup { kernels.push(KernelId::EmbeddingLookup); }
    if ops.lm_head_matmul { kernels.push(KernelId::LmHeadMatmul); }
    if ops.conv1d { kernels.push(KernelId::Conv1d); }
    if ops.mamba2_scan { kernels.push(KernelId::Mamba2Scan); }

    kernels
}

fn compute_scratch_sizes(manifest: &ModelManifest) -> ScratchSizes {
    let attention_scratch = manifest.engine_hints.max_scratch_bytes / 2;
    let mlp_scratch = manifest.engine_hints.max_scratch_bytes / 2;
    let total = manifest.engine_hints.max_scratch_bytes;

    ScratchSizes {
        attention_scratch,
        mlp_scratch,
        total,
    }
}
