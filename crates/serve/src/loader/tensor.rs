use anyhow::{Context, Result};
use engine_mlx_ffi::{MlxCtx, mlx_array, mlx_array_new_data, mlx_dtype};
use safetensors::tensor::TensorView;

fn safetensors_dtype_to_mlx(dtype: safetensors::Dtype) -> Result<mlx_dtype> {
    match dtype {
        safetensors::Dtype::F32 => Ok(mlx_dtype::MLX_FLOAT32),
        safetensors::Dtype::F16 => Ok(mlx_dtype::MLX_FLOAT16),
        safetensors::Dtype::BF16 => Ok(mlx_dtype::MLX_BFLOAT16),
        safetensors::Dtype::U32 => Ok(mlx_dtype::MLX_UINT32),
        safetensors::Dtype::I32 => Ok(mlx_dtype::MLX_INT32),
        safetensors::Dtype::U8 => Ok(mlx_dtype::MLX_UINT8),
        safetensors::Dtype::BOOL => Ok(mlx_dtype::MLX_BOOL),
        other => anyhow::bail!("Unsupported safetensors dtype: {:?}", other),
    }
}

fn shape_to_i32(shape: &[usize]) -> Vec<i32> {
    shape.iter().map(|&s| s as i32).collect()
}

pub(crate) fn load_tensor_as_mlx(name: &str, tensor_view: TensorView<'_>) -> Result<mlx_array> {
    let dtype = safetensors_dtype_to_mlx(tensor_view.dtype())?;
    let shape = shape_to_i32(tensor_view.shape());
    let data = tensor_view.data();

    let out = match dtype {
        mlx_dtype::MLX_UINT32 => {
            let u32_data: Vec<u32> = data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            unsafe {
                mlx_array_new_data(
                    u32_data.as_ptr() as *const std::ffi::c_void,
                    shape.as_ptr(),
                    shape.len() as i32,
                    dtype,
                )
            }
        }
        mlx_dtype::MLX_FLOAT32 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            unsafe {
                mlx_array_new_data(
                    f32_data.as_ptr() as *const std::ffi::c_void,
                    shape.as_ptr(),
                    shape.len() as i32,
                    dtype,
                )
            }
        }
        mlx_dtype::MLX_FLOAT16 | mlx_dtype::MLX_BFLOAT16 => unsafe {
            mlx_array_new_data(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
            )
        },
        mlx_dtype::MLX_INT32 => {
            let i32_data: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            unsafe {
                mlx_array_new_data(
                    i32_data.as_ptr() as *const std::ffi::c_void,
                    shape.as_ptr(),
                    shape.len() as i32,
                    dtype,
                )
            }
        }
        _ => anyhow::bail!("Unhandled dtype {:?} for tensor {}", dtype, name),
    };

    Ok(out)
}

/// Helper to concatenate 2-3 arrays along axis 0 using MlxCtx.
///
/// Historic code used `fuse_ctx.concatenate(&[a,b,c], 0)` with a slice API.
/// The current `engine_mlx_ffi::MlxCtx::concatenate` takes exactly two arrays,
/// so we chain calls for 3-way concats.
pub(crate) fn concatenate_many(ctx: &MlxCtx, arrays: &[mlx_array], axis: i32) -> Result<mlx_array> {
    match arrays.len() {
        0 => anyhow::bail!("concatenate_many: empty input"),
        1 => Ok(arrays[0]),
        2 => ctx
            .concatenate(arrays[0], arrays[1], axis)
            .with_context(|| "concatenate 2-way failed"),
        3 => {
            let ab = ctx
                .concatenate(arrays[0], arrays[1], axis)
                .with_context(|| "concatenate first 2 failed")?;
            ctx.concatenate(ab, arrays[2], axis)
                .with_context(|| "concatenate third failed")
        }
        n => anyhow::bail!("concatenate_many: unsupported arity {}", n),
    }
}
