use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use nxm_modelplan::ModelManifest;
use safetensors::SafeTensors;
use engine_mlx_ffi::{MlxCtx, mlx_array, mlx_default_gpu_stream_new};

use crate::loader::tensor::{load_tensor_as_mlx, concatenate_many};
use crate::loader::{Qwen3LayerWeights, Qwen3Weights};

/// Load weights using manifest for tensor name mapping.
pub fn load_weights_from_manifest(model_dir: &Path, manifest: &ModelManifest) -> Result<Qwen3Weights> {
    tracing::info!(target: "engine_mlx::qwen3::loader", "load_weights START");
    let start = std::time::Instant::now();

    // Prefer fused model if available
    let fused_path = model_dir.join("model_fused.safetensors");
    let safetensors_path = if fused_path.exists() {
        tracing::info!(
            target: "engine_mlx::qwen3::loader",
            "using fused weights path={}",
            fused_path.display()
        );
        fused_path
    } else {
        let p = model_dir.join("model.safetensors");
        if !p.exists() {
            anyhow::bail!("model.safetensors not found at {:?}", p);
        }
        tracing::info!(
            target: "engine_mlx::qwen3::loader",
            "using base weights path={}",
            p.display()
        );
        p
    };

    let file = File::open(&safetensors_path)
        .with_context(|| format!("failed to open {}", safetensors_path.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| "mmap safetensors failed")?;
    let safetensors =
        SafeTensors::deserialize(&mmap).with_context(|| "SafeTensors::deserialize failed")?;

    let _ = unsafe { mlx_default_gpu_stream_new() };

    let mut all_tensors: HashMap<String, mlx_array> = HashMap::new();
    for name in safetensors.names() {
        let view = safetensors
            .tensor(name)
            .with_context(|| format!("missing tensor {}", name))?;
        let arr = load_tensor_as_mlx(name, view)?;
        all_tensors.insert(name.to_string(), arr);
    }

    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "tensors loaded from safetensors count={}",
        all_tensors.len()
    );

    let embed_tokens = all_tensors
        .get("model.embed_tokens.weight")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing embed_tokens.weight"))?;
    let embed_scales = all_tensors
        .get("model.embed_tokens.scales")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing embed_tokens.scales"))?;
    let embed_biases = all_tensors
        .get("model.embed_tokens.biases")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing embed_tokens.biases"))?;
    let final_norm = all_tensors
        .get("model.norm.weight")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing norm.weight"))?;

    let config = crate::config::Qwen3Config::from_manifest(manifest)?;

    // We need MlxCtx for concatenation when fusing separate projections.
    let fuse_ctx = MlxCtx::gpu();

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{}", i);
        let get = |name: &str| -> Result<mlx_array> {
            let key = format!("{}.{}", p, name);
            all_tensors
                .get(&key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing {}", key))
        };
        let get_opt = |name: &str| -> Option<mlx_array> {
            let key = format!("{}.{}", p, name);
            all_tensors.get(&key).cloned()
        };

        // QKV projection: prefer fused qkv_proj, else concatenate q/k/v_proj along dim 0.
        let (qkv_w, qkv_s, qkv_b) = if let Some(w) = get_opt("self_attn.qkv_proj.weight") {
            let s = get("self_attn.qkv_proj.scales")?;
            let b = get("self_attn.qkv_proj.biases")?;
            (w, s, b)
        } else {
            if i == 0 {
                tracing::info!(
                    target: "engine_mlx::qwen3::loader",
                    "fusing separate q/k/v_proj into qkv_proj at load time"
                );
            }
            let q_w = get("self_attn.q_proj.weight")?;
            let k_w = get("self_attn.k_proj.weight")?;
            let v_w = get("self_attn.v_proj.weight")?;
            let q_s = get("self_attn.q_proj.scales")?;
            let k_s = get("self_attn.k_proj.scales")?;
            let v_s = get("self_attn.v_proj.scales")?;
            let q_b = get("self_attn.q_proj.biases")?;
            let k_b = get("self_attn.k_proj.biases")?;
            let v_b = get("self_attn.v_proj.biases")?;
            (
                concatenate_many(&fuse_ctx, &[q_w, k_w, v_w], 0)?,
                concatenate_many(&fuse_ctx, &[q_s, k_s, v_s], 0)?,
                concatenate_many(&fuse_ctx, &[q_b, k_b, v_b], 0)?,
            )
        };

        // Gate+Up projection: prefer fused gate_up_proj, else concatenate gate/up_proj along dim 0.
        let (gup_w, gup_s, gup_b) = if let Some(w) = get_opt("mlp.gate_up_proj.weight") {
            let s = get("mlp.gate_up_proj.scales")?;
            let b = get("mlp.gate_up_proj.biases")?;
            (w, s, b)
        } else {
            if i == 0 {
                tracing::info!(
                    target: "engine_mlx::qwen3::loader",
                    "fusing separate gate/up_proj into gate_up_proj at load time"
                );
            }
            let g_w = get("mlp.gate_proj.weight")?;
            let u_w = get("mlp.up_proj.weight")?;
            let g_s = get("mlp.gate_proj.scales")?;
            let u_s = get("mlp.up_proj.scales")?;
            let g_b = get("mlp.gate_proj.biases")?;
            let u_b = get("mlp.up_proj.biases")?;
            (
                concatenate_many(&fuse_ctx, &[g_w, u_w], 0)?,
                concatenate_many(&fuse_ctx, &[g_s, u_s], 0)?,
                concatenate_many(&fuse_ctx, &[g_b, u_b], 0)?,
            )
        };

        layers.push(Qwen3LayerWeights {
            input_layernorm: get("input_layernorm.weight")?,
            post_attention_layernorm: get("post_attention_layernorm.weight")?,
            qkv_proj_w: qkv_w,
            qkv_proj_s: qkv_s,
            qkv_proj_b: qkv_b,
            o_proj_w: get("self_attn.o_proj.weight")?,
            o_proj_s: get("self_attn.o_proj.scales")?,
            o_proj_b: get("self_attn.o_proj.biases")?,
            q_norm: get_opt("self_attn.q_norm.weight"),
            k_norm: get_opt("self_attn.k_norm.weight"),
            gate_up_proj_w: gup_w,
            gate_up_proj_s: gup_s,
            gate_up_proj_b: gup_b,
            down_proj_w: get("mlp.down_proj.weight")?,
            down_proj_s: get("mlp.down_proj.scales")?,
            down_proj_b: get("mlp.down_proj.biases")?,
        });
    }

    let weights = Qwen3Weights {
        embed_tokens,
        embed_scales,
        embed_biases,
        final_norm,
        layers,
        // Load separate lm_head if present (non-tied models like 8B).
        lm_head_w: all_tensors.get("lm_head.weight").cloned(),
        lm_head_s: all_tensors.get("lm_head.scales").cloned(),
        lm_head_b: all_tensors.get("lm_head.biases").cloned(),
    };

    if weights.lm_head_w.is_some() {
        tracing::info!(
            target: "engine_mlx::qwen3::loader",
            "lm_head loaded (separate, non-tied)"
        );
    }

    let elapsed = start.elapsed();
    tracing::info!(
        target: "engine_mlx::qwen3::loader",
        "load_weights END layers={} elapsed_ms={}",
        config.num_hidden_layers,
        elapsed.as_millis()
    );

    Ok(weights)
}
