use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use memmap2::MmapOptions;
use nxm_modelplan::ModelManifest;
use safetensors::SafeTensors;
use engine_mlx_ffi::{MlxCtx, mlx_array, mlx_default_gpu_stream_new};

use crate::loader::fusion;
use crate::loader::quant_params;
use crate::loader::tensor::{load_tensor_as_mlx, concatenate_many};
use crate::loader::{GateUpProj, QkvProj, QuantTensor, Qwen3LayerWeights, Qwen3Weights};

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

        // QKV projection: prefer a shipped fused qkv_proj; otherwise fuse q/k/v at
        // load time ONLY when their quant params match (task 09), else keep separate.
        let qkv_p = fusion::qkv_params(manifest, i as usize);
        let qkv = if let Some(w) = get_opt("self_attn.qkv_proj.weight") {
            let s = get("self_attn.qkv_proj.scales")?;
            let b = get("self_attn.qkv_proj.biases")?;
            QkvProj::Fused(QuantTensor {
                w,
                s,
                b,
                quant: qkv_p[0],
            })
        } else {
            let q = QuantTensor {
                w: get("self_attn.q_proj.weight")?,
                s: get("self_attn.q_proj.scales")?,
                b: get("self_attn.q_proj.biases")?,
                quant: qkv_p[0],
            };
            let k = QuantTensor {
                w: get("self_attn.k_proj.weight")?,
                s: get("self_attn.k_proj.scales")?,
                b: get("self_attn.k_proj.biases")?,
                quant: qkv_p[1],
            };
            let v = QuantTensor {
                w: get("self_attn.v_proj.weight")?,
                s: get("self_attn.v_proj.scales")?,
                b: get("self_attn.v_proj.biases")?,
                quant: qkv_p[2],
            };
            match fusion::fusion_decision(&qkv_p) {
                fusion::Fusion::Fused => {
                    if i == 0 {
                        tracing::info!(
                            target: "engine_mlx::qwen3::loader",
                            "fusing separate q/k/v_proj into qkv_proj at load time (quant params match {:?})",
                            qkv_p[0]
                        );
                    }
                    QkvProj::Fused(QuantTensor {
                        w: concatenate_many(&fuse_ctx, &[q.w, k.w, v.w], 0)?,
                        s: concatenate_many(&fuse_ctx, &[q.s, k.s, v.s], 0)?,
                        b: concatenate_many(&fuse_ctx, &[q.b, k.b, v.b], 0)?,
                        quant: qkv_p[0],
                    })
                }
                fusion::Fusion::Separate => {
                    if i == 0 {
                        tracing::info!(
                            target: "engine_mlx::qwen3::loader",
                            "keeping q/k/v_proj separate at load time (quant params diverge: {:?})",
                            qkv_p
                        );
                    }
                    QkvProj::Separate { q, k, v }
                }
            }
        };

        // Gate+Up projection: prefer a shipped fused gate_up_proj; otherwise fuse
        // gate/up at load time ONLY when their quant params match (task 09).
        let gup_p = fusion::gate_up_params(manifest, i as usize);
        let gate_up = if let Some(w) = get_opt("mlp.gate_up_proj.weight") {
            let s = get("mlp.gate_up_proj.scales")?;
            let b = get("mlp.gate_up_proj.biases")?;
            GateUpProj::Fused(QuantTensor {
                w,
                s,
                b,
                quant: gup_p[0],
            })
        } else {
            let gate = QuantTensor {
                w: get("mlp.gate_proj.weight")?,
                s: get("mlp.gate_proj.scales")?,
                b: get("mlp.gate_proj.biases")?,
                quant: gup_p[0],
            };
            let up = QuantTensor {
                w: get("mlp.up_proj.weight")?,
                s: get("mlp.up_proj.scales")?,
                b: get("mlp.up_proj.biases")?,
                quant: gup_p[1],
            };
            match fusion::fusion_decision(&gup_p) {
                fusion::Fusion::Fused => {
                    if i == 0 {
                        tracing::info!(
                            target: "engine_mlx::qwen3::loader",
                            "fusing separate gate/up_proj into gate_up_proj at load time (quant params match {:?})",
                            gup_p[0]
                        );
                    }
                    GateUpProj::Fused(QuantTensor {
                        w: concatenate_many(&fuse_ctx, &[gate.w, up.w], 0)?,
                        s: concatenate_many(&fuse_ctx, &[gate.s, up.s], 0)?,
                        b: concatenate_many(&fuse_ctx, &[gate.b, up.b], 0)?,
                        quant: gup_p[0],
                    })
                }
                fusion::Fusion::Separate => {
                    if i == 0 {
                        tracing::info!(
                            target: "engine_mlx::qwen3::loader",
                            "keeping gate/up_proj separate at load time (quant params diverge: {:?})",
                            gup_p
                        );
                    }
                    GateUpProj::Separate { gate, up }
                }
            }
        };

        // Per-projection quant params for the non-fusible projections (o_proj, down_proj).
        let qp = quant_params::layer_quant_params(manifest, i as usize);

        layers.push(Qwen3LayerWeights {
            input_layernorm: get("input_layernorm.weight")?,
            post_attention_layernorm: get("post_attention_layernorm.weight")?,
            qkv,
            o_proj: QuantTensor {
                w: get("self_attn.o_proj.weight")?,
                s: get("self_attn.o_proj.scales")?,
                b: get("self_attn.o_proj.biases")?,
                quant: qp.o,
            },
            q_norm: get_opt("self_attn.q_norm.weight"),
            k_norm: get_opt("self_attn.k_norm.weight"),
            gate_up,
            down_proj: QuantTensor {
                w: get("mlp.down_proj.weight")?,
                s: get("mlp.down_proj.scales")?,
                b: get("mlp.down_proj.biases")?,
                quant: qp.down,
            },
        });
    }

    let weights = Qwen3Weights {
        embed_tokens: QuantTensor {
            w: embed_tokens,
            s: embed_scales,
            b: embed_biases,
            quant: quant_params::embed_quant_params(manifest),
        },
        final_norm,
        layers,
        // Load separate lm_head if present (non-tied models like 8B).
        lm_head: match (
            all_tensors.get("lm_head.weight"),
            all_tensors.get("lm_head.scales"),
            all_tensors.get("lm_head.biases"),
        ) {
            (Some(w), Some(s), Some(b)) => Some(QuantTensor {
                w: w.clone(),
                s: s.clone(),
                b: b.clone(),
                quant: quant_params::lm_head_quant_params(manifest),
            }),
            _ => None,
        },
    };

    if weights.lm_head.is_some() {
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
