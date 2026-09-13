//! Engine hints and required-ops flags for the manifest.

use serde::{Deserialize, Serialize};

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

impl OpsRequired {
    /// Return a list of enabled op names.
    pub fn as_list(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();
        if self.rmsnorm {
            ops.push("rmsnorm");
        }
        if self.rope {
            ops.push("rope");
        }
        if self.flash_attn_decode {
            ops.push("flash_attn_decode");
        }
        if self.fused_gdn_step {
            ops.push("fused_gdn_step");
        }
        if self.quant_matmul_q4 {
            ops.push("quant_matmul_q4");
        }
        if self.quant_matmul_mxfp4 {
            ops.push("quant_matmul_mxfp4");
        }
        if self.quant_matmul_mxfp8 {
            ops.push("quant_matmul_mxfp8");
        }
        if self.moe_routing {
            ops.push("moe_routing");
        }
        if self.silu {
            ops.push("silu");
        }
        if self.swiglu {
            ops.push("swiglu");
        }
        if self.residual_add {
            ops.push("residual_add");
        }
        if self.embedding_lookup {
            ops.push("embedding_lookup");
        }
        if self.lm_head_matmul {
            ops.push("lm_head_matmul");
        }
        if self.conv1d {
            ops.push("conv1d");
        }
        if self.mamba2_scan {
            ops.push("mamba2_scan");
        }
        ops
    }
    pub fn count(&self) -> usize {
        self.as_list().len()
    }
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
            max_scratch_bytes: 64 * 1024 * 1024,
            metal_pipelines_needed: 6,
            cold_start_safe: true,
        }
    }
}
