//! `engine-bench` library surface, exposed so integration tests in
//! `tests/` can import the pure discriminator functions without pulling in
//! the binary's CLI machinery.
//!
//! The binary (`src/main.rs`) remains the entry point for `nxm-bench`; this
//! module only re-exports what integration tests need.

pub mod config;
pub mod discriminators;
pub mod pass_check;
