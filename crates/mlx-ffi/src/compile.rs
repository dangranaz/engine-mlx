//! MLX-C compile / graph capture utilities.

#![allow(unsafe_code)]

#[cfg(feature = "mlx")]
mod real {
    use anyhow::Result;
    use crate::*;

    /// RAII wrapper around mlx_closure.
    pub struct Closure {
        pub(crate) inner: mlx_closure,
    }

    unsafe impl Send for Closure {}

    impl Drop for Closure {
        fn drop(&mut self) {
            unsafe { mlx_closure_free(self.inner); }
        }
    }

    impl Closure {
        pub fn from_raw(inner: mlx_closure) -> Self {
            Self { inner }
        }

        pub fn from_fn(fun: extern "C" fn(*mut mlx_vector_array, mlx_vector_array) -> i32) -> Self {
            let inner = unsafe { mlx_closure_new_func(Some(fun)) };
            Self { inner }
        }

        pub fn from_fn_payload(
            fun: unsafe extern "C" fn(*mut mlx_vector_array, mlx_vector_array, *mut std::ffi::c_void) -> i32,
            payload: *mut std::ffi::c_void,
            free_payload: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
        ) -> Self {
            let inner = unsafe { mlx_closure_new_func_payload(Some(fun), payload, free_payload) };
            Self { inner }
        }

        pub fn apply(&self, inputs: &[mlx_array]) -> Result<Vec<mlx_array>> {
            let vec_in = unsafe { mlx_vector_array_new_data(inputs.as_ptr(), inputs.len()) };
            let mut vec_out = unsafe { mlx_vector_array_new() };
            let status = unsafe { mlx_closure_apply(&mut vec_out, self.inner, vec_in) };
            unsafe { mlx_vector_array_free(vec_in); }
            if status != 0 {
                unsafe { mlx_vector_array_free(vec_out); }
                anyhow::bail!("mlx_closure_apply failed (status {status})");
            }
            let n = unsafe { mlx_vector_array_size(vec_out) };
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut arr: mlx_array = unsafe { std::mem::zeroed() };
                unsafe { mlx_vector_array_get(&mut arr, vec_out, i); }
                out.push(arr);
            }
            unsafe { mlx_vector_array_free(vec_out); }
            Ok(out)
        }
    }

    pub fn compile(fun: Closure) -> Result<Closure> {
        let mut res: mlx_closure = unsafe { mlx_closure_new() };
        let status = unsafe { mlx_compile(&mut res, fun.inner, false) };
        if status != 0 {
            anyhow::bail!("mlx_compile failed (status {status})");
        }
        std::mem::forget(fun);
        Ok(Closure { inner: res })
    }

    pub fn compile_shapeless(fun: Closure) -> Result<Closure> {
        let mut res: mlx_closure = unsafe { mlx_closure_new() };
        let status = unsafe { mlx_compile(&mut res, fun.inner, true) };
        if status != 0 {
            anyhow::bail!("mlx_compile_shapeless failed (status {status})");
        }
        std::mem::forget(fun);
        Ok(Closure { inner: res })
    }
}

#[cfg(feature = "mlx")]
pub use real::{compile, compile_shapeless, Closure};

#[cfg(not(feature = "mlx"))]
mod stub {
    use anyhow::Result;
    use crate::mlx_closure;

    pub struct Closure {
        pub inner: mlx_closure,
    }

    impl Closure {
        pub fn new(inner: mlx_closure) -> Self { Self { inner } }
        pub fn from_raw(inner: mlx_closure) -> Self { Self { inner } }
        pub fn from_fn(_fun: extern "C" fn(*mut crate::mlx_vector_array, crate::mlx_vector_array) -> i32) -> Self {
            Self { inner: unsafe { std::mem::zeroed() } }
        }
        pub fn from_fn_payload(
            _fun: unsafe extern "C" fn(*mut crate::mlx_vector_array, crate::mlx_vector_array, *mut std::ffi::c_void) -> i32,
            _payload: *mut std::ffi::c_void,
            _free: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
        ) -> Self { Self { inner: unsafe { std::mem::zeroed() } } }
        pub fn apply(&self, _inputs: &[crate::mlx_array]) -> Result<Vec<crate::mlx_array>> {
            anyhow::bail!("mlx_compile not bound yet — enable --features mlx with mlx-c")
        }
    }

    pub fn compile(_fun: Closure) -> Result<Closure> {
        anyhow::bail!("mlx_compile not bound yet")
    }

    pub fn compile_shapeless(_fun: Closure) -> Result<Closure> {
        anyhow::bail!("mlx_compile_shapeless not bound yet")
    }
}

#[cfg(not(feature = "mlx"))]
pub use stub::{compile, compile_shapeless, Closure};
