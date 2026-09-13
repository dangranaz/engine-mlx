//! Integration test entry point.
//!
//! Cargo only auto-discovers test targets from files directly under `tests/`.
//! The actual cross-crate integration tests live in `tests/integration/`
//! (per RULES.md); this file wires them in.

#[path = "integration/engine_pipeline.rs"]
mod engine_pipeline;
