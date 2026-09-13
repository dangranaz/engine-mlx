//! HTTP server — axum routes, OpenAI-compatible /v1/chat/completions.

use crate::handle::InferHandle;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};
use tower_http::cors::{Any, CorsLayer};

/// App state.
pub struct AppState {
    pub handle: InferHandle,
}

/// Start server.
pub async fn run(handle: InferHandle, host: &str, port: u16) -> anyhow::Result<()> {
    let state = Arc::new(AppState { handle });
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(cors)
        .with_state(state);
    let addr = format!("{host}:{port}");
    tracing::info!(target: "engine_mlx::server", "listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Request/Response ────────────────────────────────────────────

/// OpenAI-compatible chat completion request body.
#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub model: Option<String>,
}

/// A single chat message with role and content.
#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: Message,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct SseDelta {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<SseChoice>,
}

#[derive(Serialize)]
struct SseChoice {
    index: u32,
    delta: Message,
    finish_reason: Option<&'static str>,
}

fn default_max_tokens() -> Option<u32> {
    Some(256)
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Handlers ────────────────────────────────────────────────────

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let messages: Vec<(String, String)> = req.messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();
    let model = req.model.clone().unwrap_or_else(|| "qwen3".to_string());
    let stream = req.stream;

    // Clone handle for blocking task
    let handle = state.handle.clone();

    if stream {
        // Streaming via SSE — generate then stream tokens word-by-word (engine is non-streaming)
        let result = tokio::task::spawn_blocking(move || handle.generate(messages, max_tokens))
            .await;
        match result {
            Ok(Ok((text, prompt_tokens, completion_tokens, _tps))) => {
                let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
                let created = now_unix();
                // Split text into words for streaming demo
                let words: Vec<String> = text.split_whitespace().map(|w| format!("{w} ")).collect();
                let sse_stream: Vec<Result<Event, Infallible>> = words
                    .into_iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let delta = SseDelta {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model.clone(),
                            choices: vec![SseChoice {
                                index: 0,
                                delta: Message { role: "assistant".to_string(), content: w },
                                finish_reason: None,
                            }],
                        };
                        Ok(Event::default().json_data(delta).unwrap_or_else(|e| {
                            tracing::error!(target: "engine_mlx::server", "SSE JSON serialization failed: {e}");
                            Event::default()
                        }))
                    })
                    .collect();
                let mut stream_vec = sse_stream;
                // Final stop event
                let done_delta = SseDelta {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![SseChoice {
                        index: 0,
                        delta: Message { role: "assistant".to_string(), content: String::new() },
                        finish_reason: Some("stop"),
                    }],
                };
                stream_vec.push(Ok(Event::default().json_data(done_delta).unwrap_or_else(|e| {
                    tracing::error!(target: "engine_mlx::server", "SSE JSON serialization failed: {e}");
                    Event::default()
                })));
                // Need to produce a Stream
                let s = stream::iter(stream_vec);
                Sse::new(s).into_response()
            }
            Ok(Err(e)) => {
                tracing::error!(target: "engine_mlx::server", "generation failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, format!("{{\"error\":\"{e}\"}}")).into_response()
            }
            Err(e) => {
                tracing::error!(target: "engine_mlx::server", "spawn failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".to_string()).into_response()
            }
        }
    } else {
        let result = tokio::task::spawn_blocking(move || handle.generate(messages, max_tokens)).await;
        match result {
            Ok(Ok((text, prompt_tokens, completion_tokens, _tps))) => {
                let resp = ChatResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion",
                    created: now_unix(),
                    model,
                    choices: vec![Choice {
                        index: 0,
                        message: Message { role: "assistant".to_string(), content: text },
                        finish_reason: "stop",
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };
                Json(resp).into_response()
            }
            Ok(Err(e)) => {
                tracing::error!(target: "engine_mlx::server", "generation failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, format!("{{\"error\":\"{e}\"}}")).into_response()
            }
            Err(e) => {
                tracing::error!(target: "engine_mlx::server", "spawn failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".to_string()).into_response()
            }
        }
    }
}

async fn list_models() -> impl IntoResponse {
    // Report the model actually loaded (from ENGINE_MLX_MODEL) instead of a hardcoded
    // id, so /v1/models is truthful and benchmark reports are labelled correctly.
    let id = std::env::var("ENGINE_MLX_MODEL")
        .ok()
        .and_then(|p| {
            std::path::Path::new(&p)
                .file_name()
                .map(|s| s.to_string_lossy().to_lowercase())
        })
        .unwrap_or_else(|| "qwen3-0.6b".to_string());
    Json(serde_json::json!({
        "object": "list",
        "data": [{"id": id, "object": "model", "owned_by": "engine-mlx"}]
    }))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}
