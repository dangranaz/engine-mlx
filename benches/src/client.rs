//! Minimal OpenAI-compatible HTTP client.
//!
//! Two request paths:
//! - `chat_blocking`: POST /v1/chat/completions (no stream), returns the full
//!   completion plus usage and total wall time.
//! - `chat_stream`: POST with `stream:true`, reads SSE chunks and records the
//!   time-to-first-token (TTFT). Only meaningful when the server streams for
//!   real (token-by-token). Engines that generate everything then fake-stream
//!   will report a TTFT close to total latency — the caller decides how to
//!   interpret it.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;

/// Outcome of a single blocking chat completion.
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Wall time for the whole request (ms).
    pub latency_ms: f64,
    /// completion_tokens / (latency_ms/1000). Falls back to word count when
    /// the server omits usage.
    pub tps: f64,
}

/// Outcome of a streaming chat completion (for real TTFT).
/// Reserved for engines that stream token-by-token; not yet used by the
/// standard suite (see README "TTFT note").
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StreamResult {
    pub content: String,
    pub completion_tokens: u32,
    pub latency_ms: f64,
    /// Time from request send to first non-empty content delta (ms).
    pub ttft_ms: f64,
    pub tps: f64,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct ChatChoiceMsg {
    #[serde(default)]
    content: String,
}
#[derive(Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatChoiceMsg>,
}
#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

/// A chat client bound to a base URL and model id.
pub struct Client {
    base_url: String,
    model: String,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(base_url: &str, model: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(600))
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            agent,
        }
    }

    /// GET /v1/models — returns the first model id advertised, if any.
    pub fn first_model_id(&self) -> Option<String> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self.agent.get(&url).call().ok()?;
        let v: serde_json::Value = resp.into_json().ok()?;
        v.get("data")?
            .as_array()?
            .first()?
            .get("id")?
            .as_str()
            .map(|s| s.to_string())
    }

    /// GET /health — returns true on HTTP 200.
    pub fn health(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        matches!(self.agent.get(&url).call(), Ok(r) if r.status() == 200)
    }

    /// Blocking chat completion. `temperature` and `max_tokens` are passed through.
    pub fn chat_blocking(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ChatResult> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": false,
        });
        let t0 = Instant::now();
        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_json(body)
            .with_context(|| format!("POST {url}"))?;
        let parsed: ChatResponse = resp.into_json().context("parse chat response")?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let content = parsed
            .choices
            .first()
            .and_then(|c| c.message.as_ref())
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let (prompt_tokens, completion_tokens) = match parsed.usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, content.split_whitespace().count() as u32),
        };
        let tps = if latency_ms > 0.0 {
            completion_tokens as f64 / (latency_ms / 1000.0)
        } else {
            0.0
        };
        Ok(ChatResult {
            content,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            tps,
        })
    }

    /// Streaming chat completion — measures real TTFT from SSE deltas.
    /// Not yet wired into the standard suite; kept for engines that stream for
    /// real (see README "TTFT note").
    #[allow(dead_code)]
    pub fn chat_stream(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<StreamResult> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": true,
        });
        let t0 = Instant::now();
        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_json(body)
            .with_context(|| format!("POST(stream) {url}"))?;

        let reader = BufReader::new(resp.into_reader());
        let mut content = String::new();
        let mut ttft_ms: Option<f64> = None;
        let mut chunks: u32 = 0;

        for line in reader.lines() {
            let line = line.context("read SSE line")?;
            let data = match line.strip_prefix("data:") {
                Some(d) => d.trim(),
                None => continue,
            };
            if data == "[DONE]" {
                break;
            }
            if data.is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let delta = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|s| s.as_str())
                .unwrap_or("");
            if !delta.is_empty() {
                if ttft_ms.is_none() {
                    ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
                content.push_str(delta);
                chunks += 1;
            }
        }
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = ttft_ms.ok_or_else(|| anyhow!("no content deltas received (not streaming?)"))?;
        // Approximate completion tokens by streamed chunks (server usually emits
        // one token per chunk; falls back to word count if larger).
        let completion_tokens = chunks.max(content.split_whitespace().count() as u32);
        let tps = if latency_ms > 0.0 {
            completion_tokens as f64 / (latency_ms / 1000.0)
        } else {
            0.0
        };
        Ok(StreamResult {
            content,
            completion_tokens,
            latency_ms,
            ttft_ms,
            tps,
        })
    }
}
