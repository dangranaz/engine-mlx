//! Unit tests for `engine_mlx_serve::server` module.

#[test]
fn server_run_exists() {
    // server::run is an async fn, just verify it is linkable
    // The real server requires MLX + InferHandle; stub just checks compilation.
    let _ = engine_mlx_serve::server::run as fn(_, _, _) -> _;
}
