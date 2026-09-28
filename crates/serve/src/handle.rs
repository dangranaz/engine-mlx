//! Inference handle — channel to MLX thread (single-threaded GPU).

use std::sync::mpsc;

/// Request from HTTP to MLX thread.
pub enum InferRequest {
    Generate {
        messages: Vec<(String, String)>,
        max_tokens: u32,
        reply: mpsc::Sender<anyhow::Result<(String, u32, u32, f64)>>,
    },
}

/// Clone handle for HTTP handlers.
#[derive(Clone)]
pub struct InferHandle {
    tx: mpsc::Sender<InferRequest>,
}

impl InferHandle {
    /// Create a new handle bound to an MLX thread channel sender.
    pub fn new(tx: mpsc::Sender<InferRequest>) -> Self {
        Self { tx }
    }

    /// Send a generate request to the MLX thread and block until the reply arrives.
    pub fn generate(
        &self,
        messages: Vec<(String, String)>,
        max_tokens: u32,
    ) -> anyhow::Result<(String, u32, u32, f64)> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(InferRequest::Generate {
                messages,
                max_tokens,
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("MLX thread dead"))?;
        reply_rx.recv().map_err(|_| anyhow::anyhow!("MLX thread dead"))?
    }
}
