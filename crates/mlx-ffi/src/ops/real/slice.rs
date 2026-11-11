use anyhow::Result;
use crate::*;

impl super::MlxCtx {
    /// Evaluate and synchronize multiple arrays in one GPU pass.
    /// Needed before reading array data (e.g. via array_data_f32).
    pub fn eval_all_and_sync(&self, arrays: &[mlx_array]) -> Result<()> {
        self.eval_all(arrays)?;
        unsafe { mlx_synchronize(self.stream); }
        Ok(())
    }

    /// Copy an array to materialize lazy Load arrays.
    pub fn materialize(&self, arr: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_copy(&mut out, arr, self.stream) }, "materialize")?;
        Ok(out)
    }

    pub fn concatenate(&self, a: mlx_array, b: mlx_array, axis: i32) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        let arrays = [a, b];
        let vec = unsafe { mlx_vector_array_new_data(arrays.as_ptr(), 2) };
        Self::check(unsafe { mlx_concatenate_axis(&mut out, vec, axis, self.stream) }, "concatenate")?;
        unsafe { mlx_vector_array_free(vec) };
        Ok(out)
    }

    /// Split `a` at the given INDICES along `axis` (MLX semantics: indices, not lengths).
    /// E.g. `split_sections(qkv_4096, &[2048, 3072], 2)` -> `[2048, 1024, 1024]`.
    /// Returns the list of section arrays (output shapes are known from the
    /// indices, so this works under shapeless compile tracing).
    pub fn split_sections(&self, a: mlx_array, sections: &[i32], axis: i32) -> Result<Vec<mlx_array>> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(
            unsafe {
                mlx_split_sections(
                    &mut out,
                    a,
                    sections.as_ptr(),
                    sections.len(),
                    axis,
                    self.stream,
                )
            },
            "split_sections",
        )?;
        let n = unsafe { mlx_vector_array_size(out) } as usize;
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr: mlx_array = unsafe { std::mem::zeroed() };
            unsafe { mlx_vector_array_get(&mut arr, out, i) };
            res.push(arr);
        }
        unsafe { mlx_vector_array_free(out) };
        Ok(res)
    }

    /// Stack arrays along a new axis (placeholder — requires mlx_stack_axis binding).
    pub fn stack(&self, _arrays: &[mlx_array], _axis: i32) -> Result<mlx_array> {
        anyhow::bail!("stack not available - mlx_stack_axis binding missing")
    }

    /// Slice: extract sub-array along axes.
    pub fn slice(&self, a: mlx_array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_slice(&mut out, a,
                start.as_ptr(), start.len(),
                stop.as_ptr(), stop.len(),
                strides.as_ptr(), strides.len(),
                self.stream)
        }, "slice")?;
        Ok(out)
    }

    /// Slice update: write `update` into `src` at position [start:stop:stride].
    pub fn slice_update(&self, src: mlx_array, update: mlx_array, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_slice_update(&mut out, src, update,
                start.as_ptr(), start.len(),
                stop.as_ptr(), stop.len(),
                strides.as_ptr(), strides.len(),
                self.stream)
        }, "slice_update")?;
        Ok(out)
    }

    /// Dynamic slice update: write `update` into `src` at dynamic position.
    /// `start` is an mlx_array (for compiled graphs with runtime offsets).
    /// `axes` specifies which axes the start applies to.
    pub fn slice_update_dynamic(&self, src: mlx_array, update: mlx_array, start: mlx_array, axes: &[i32]) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::slice_update_dynamic axes={:?}", axes);
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_slice_update_dynamic(&mut out, src, update, start,
                axes.as_ptr(), axes.len(),
                self.stream)
        }, "slice_update_dynamic")?;
        Ok(out)
    }

    /// Scatter `updates` into `src` at `indices` along `axis`.
    /// Unlike slice_update, uses index-based addressing (may have lower dispatch overhead).
    pub fn scatter_single(&self, src: mlx_array, indices: mlx_array, updates: mlx_array, axis: i32) -> Result<mlx_array> {
        tracing::trace!("MlxCtx::scatter_single axis={}", axis);
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe {
            mlx_scatter_single(&mut out, src, indices, updates, axis, self.stream)
        }, "scatter_single")?;
        Ok(out)
    }

    // ── Elementwise math ops ─────────────────────────────────────────────

    pub fn shape(&self, x: mlx_array) -> Result<Vec<i32>> {
        unsafe {
            let ndim = mlx_array_ndim(x) as usize;
            let mut shape = vec![0i32; ndim];
            for i in 0..ndim {
                shape[i] = mlx_array_dim(x, i as i32);
            }
            Ok(shape)
        }
    }

    pub fn sigmoid(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_sigmoid(&mut out, x, self.stream) }, "sigmoid")?;
        Ok(out)
    }

    pub fn exp(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_exp(&mut out, x, self.stream) }, "exp")?;
        Ok(out)
    }

    pub fn negative(&self, x: mlx_array) -> Result<mlx_array> {
        let mut out = unsafe { std::mem::zeroed() };
        Self::check(unsafe { mlx_negative(&mut out, x, self.stream) }, "negative")?;
        Ok(out)
    }

}
