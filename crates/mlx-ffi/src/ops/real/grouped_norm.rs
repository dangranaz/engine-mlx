//! Grouped RMS Normalisation op.
//!
//! Replicates the reference `GroupedRMSNorm` semantics:
//!   - groups == 1: plain `rms_norm(x, weight, eps)` (weight applied by the fused kernel).
//!   - groups > 1: reshape to `[..., groups, dims/groups]`, normalise each group in f32
//!     **without** weight, reshape back, then multiply by the full-width `weight` and cast
//!     back to the input dtype. The weight is therefore applied full-width *after*
//!     normalisation, never per-group inside the kernel.
//!
//! # Divergence from reference
//!
//! The MLX Python reference calls normalisation with `weight=None` (no weight applied
//! inside the normalisation). Our kernel `mlx_fast_rms_norm` requires a weight array, so
//! we pass an all-ones weight of length `dims/groups`. This achieves the same effect:
//! no per-group scaling inside the kernel, with the real full-width weight applied
//! afterwards via element-wise multiplication.

use anyhow::Result;
use crate::*;

#[cfg(feature = "mlx")]
impl super::MlxCtx {
    /// Grouped RMS Normalisation.
    ///
    /// When `groups == 1` this delegates to the fused [`MlxCtx::rms_norm`] kernel, so the
    /// result is identical to `rms_norm(x, weight, eps)`.
    ///
    /// When `groups > 1` the input is reshaped to `[..., groups, dims/groups]`, each group
    /// is normalised independently in f32 (no weight applied inside the kernel — a weight of
    /// all-ones is passed so the kernel is a no-op scaling), then reshaped back to the
    /// original shape, multiplied by the full-width `weight`, and cast back to the input
    /// dtype. This matches the reference MLX `GroupedRMSNorm` semantics exactly.
    ///
    /// # Errors
    ///
    /// Returns `Err` in three cases:
    /// - `groups < 1` — the group count must be at least 1.
    /// - `ndim < 1` — the input must have at least one dimension.
    /// - `dims % groups != 0` — the last dimension must be divisible by the group count.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use engine_mlx_ffi::{MlxCtx, mlx_dtype_};
    /// # let ctx = MlxCtx::cpu();
    /// # let x = ctx.new_array(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    /// # let w = ctx.new_array(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    /// let out = ctx.grouped_rms_norm(x, w, 1e-5, 1).unwrap();
    /// ```
    pub fn grouped_rms_norm(
        &self,
        x: mlx_array,
        weight: mlx_array,
        eps: f32,
        groups: i32,
    ) -> Result<mlx_array> {
        tracing::trace!(target: "engine_mlx::mlx::blas", "grouped_rms_norm groups={} eps={}", groups, eps);

        if groups < 1 {
            anyhow::bail!("grouped_rms_norm: groups must be >= 1, got {groups}");
        }

        // Fast path: groups == 1 is exactly the fused rms_norm kernel.
        if groups == 1 {
            return self.rms_norm(x, weight, eps);
        }

        let ndim = unsafe { mlx_array_ndim(x) } as i32;
        if ndim < 1 {
            anyhow::bail!("grouped_rms_norm: input must have at least 1 dimension");
        }

        let dims = unsafe { mlx_array_dim(x, ndim - 1) };
        if dims % groups != 0 {
            anyhow::bail!(
                "grouped_rms_norm: last dimension {dims} is not divisible by groups {groups}"
            );
        }
        let dims_per_group = dims / groups;

        let input_dtype = unsafe { mlx_array_dtype(x) };

        // 1. Cast input to f32 for the normalisation.
        let x_f32 = self.astype(x, mlx_dtype_::MLX_FLOAT32)?;

        // 2. Build the grouped shape: [..., groups, dims/groups].
        // Original shape of x (needed for the reshape-back after normalisation).
        let mut orig_shape = Vec::with_capacity(ndim as usize);
        for i in 0..ndim {
            orig_shape.push(unsafe { mlx_array_dim(x, i) });
        }

        // Grouped shape: [..., groups, dims/groups] — take first (ndim-1) dims,
        // then append groups and dims_per_group.
        let mut grouped_shape = orig_shape.clone();
        grouped_shape.truncate((ndim - 1) as usize);
        grouped_shape.push(groups);
        grouped_shape.push(dims_per_group);

        // 3. Reshape to [..., groups, dims/groups].
        let x_grouped = self.reshape(x_f32, &grouped_shape)?;

        // 4. RMS-normalise each group with an all-ones weight (kernel applies weight==1,
        //    i.e. no per-group scaling — the real weight is applied full-width afterwards).
        let ones = self.full_f32(&[dims_per_group], 1.0)?;
        let y = self.rms_norm(x_grouped, ones, eps)?;

        // 5. Reshape back to the original shape (now in f32).
        let y_back = self.reshape(y, &orig_shape)?;

        // 6. Multiply by the full-width weight, then cast back to the input dtype.
        let scaled = self.multiply(y_back, weight)?;
        let out = self.astype(scaled, input_dtype)?;

        // Free the intermediates this helper owns. `mlx_array` handles are
        // refcounted C objects: dropping the Rust copy does NOT release the
        // underlying Metal buffer, so every intermediate left unfreed stays a
        // live resource. `out` retains its inputs through graph edges until
        // evaluated, so releasing our handles here is safe. Without this,
        // each call leaks ~35 KB (f32 intermediates + the ones weight); at
        // 73 calls per decode step that is ~2.6 MB/step, and after ~6 long
        // requests the Metal resource count crosses its limit and decode
        // degenerates to garbage (task 23).
        unsafe {
            mlx_array_free(x_f32);
            mlx_array_free(ones);
            mlx_array_free(x_grouped);
            mlx_array_free(y);
            mlx_array_free(y_back);
            mlx_array_free(scaled);
        }

        Ok(out)
    }
}