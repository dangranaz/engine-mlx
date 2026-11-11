//! Chat completion rendering via the model's own chat template.
//!
//! Split from `engine.rs` per RULES (max 300 lines/file). `generate_chat`
//! renders `messages` through `ChatTemplate` (task 14, `add_generation_prompt
//! = true`) and delegates the decode loop to `generate`.

use anyhow::{Context, Result};
use nxm_tokenizer::chat_template::{ChatMessage, ChatTemplate};

use super::Qwen3Engine;

impl Qwen3Engine {
    /// Generate a chat completion using the model's own chat template.
    ///
    /// Renders `messages` through the model's `chat_template.jinja` (with
    /// `add_generation_prompt=true`) and then runs [`generate`] on the
    /// resulting prompt string. Returns an error if no chat template was
    /// loaded at engine load time — there is **no** hardcoded fallback, so
    /// the model's own template is the single source of truth.
    ///
    /// Returns `(generated_text, prompt_token_count)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # let model_dir = std::path::Path::new("/tmp/nxm-model");
    /// # let mut engine = engine_mlx_serve::engine::Qwen3Engine::load(model_dir).expect("load");
    /// let (text, prompt_tokens) = engine.generate_chat(
    ///     &[("user".into(), "hello".into())],
    ///     256,
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate_chat(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
    ) -> Result<(String, u32)> {
        let chat_template = self
            .chat_template
            .as_ref()
            .context("no chat template available; cannot render chat messages")?;
        let chat_messages: Vec<ChatMessage> = messages
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();
        let prompt = chat_template
            .render(&chat_messages, true)
            .with_context(|| "failed to render chat template")?;
        tracing::debug!(
            target: "engine_mlx::serve::engine",
            "rendered chat prompt (len={})",
            prompt.len()
        );
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt)
            .with_context(|| "tokenizer encode failed")?
            .len() as u32;
        let text = self.generate(&prompt, max_tokens)?;
        Ok((text, prompt_tokens))
    }
}
