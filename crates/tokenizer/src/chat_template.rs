//! # Chat template rendering
//!
//! Pure rendering of a model's own chat template into a prompt string.
//!
//! This module reads the real `chat_template.jinja` shipped with a model
//! (falling back to the `chat_template` key of `tokenizer_config.json`)
//! and renders it with `minijinja` into a prompt string, given a list of
//! `(role, content)` messages and an `add_generation_prompt` flag.
//!
//! No template is hardcoded here: the model's own template is the single
//! source of truth. See [`ChatTemplate`] for the rendering entry point.

use anyhow::{Context, Result};
use fancy_regex::Regex;
use minijinja::Environment;
use serde::Serialize;
use std::path::Path;

/// A single chat message: `role` (e.g. `user`, `assistant`, `system`) and
/// its `content`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Renders a model's chat template into a prompt string.
///
/// The template is loaded once from `model_dir/chat_template.jinja` if
/// present, otherwise from the `chat_template` key of
/// `model_dir/tokenizer_config.json`. If neither source exists,
/// `from_model_dir` returns an error.
///
/// Before compilation the template text is normalized: the HF-specific
/// `{% generation %}` / `{% endgeneration %}` block markers (and their
/// whitespace-control variants) are removed while preserving the body they
/// enclose. These markers are semantically transparent to the rendered text
/// (they only mark the assistant-generated span for
/// `return_assistant_tokens_mask`) but `minijinja` 2.21 has no parser for
/// them and exposes no API to register a custom block statement, so they are
/// stripped as a preprocessing step. No other template keyword is touched —
/// in particular `add_generation_prompt` (a variable, not a block) is left
/// intact.
///
/// Rendering uses `minijinja` with the HF chat-template conventions:
/// * `raise_exception(msg)` — global function that aborts rendering with `msg`
/// * `bos_token` — beginning-of-sequence token (from `tokenizer_config.json`,
///   as a string or as `{"content": "..."}`, empty string when absent)
/// * `messages` — list of `{ role, content }` objects
/// * `add_generation_prompt` — boolean flag
///
/// # Example
///
/// ```no_run
/// use nxm_tokenizer::chat_template::ChatTemplate;
/// let tpl = ChatTemplate::from_model_dir(std::path::Path::new("model_dir"))?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ChatTemplate {
    env: Environment<'static>,
    source: TemplateSource,
    bos_token: String,
}

/// Where the template text came from, kept for error messages.
#[derive(Clone, Debug)]
enum TemplateSource {
    File { path: std::path::PathBuf },
    ConfigKey { path: std::path::PathBuf },
}

/// Compile the regex matching HF `{% generation %}` / `{% endgeneration %}`
/// block markers (including whitespace-control variants). Only the keywords
/// `generation` and `endgeneration` are matched — never `add_generation_prompt`
/// (a variable, not a block).
fn compile_generation_marker_re() -> Result<Regex> {
    Regex::new(r"{%\s*-?\s*(?:generation|endgeneration)\s*-?\s*%}")
        .context("compiling generation-marker strip regex")
}

/// Remove the `{% generation %}` / `{% endgeneration %}` markers from
/// `template`, preserving the body they enclose.
fn strip_generation_markers(template: &str, re: &Regex) -> String {
    re.replace_all(template, "").to_string()
}

impl ChatTemplate {
    /// Load the model's chat template from `model_dir`.
    ///
    /// Reads `chat_template.jinja` when present, otherwise the `chat_template`
    /// key of `tokenizer_config.json`. Returns a contextual error when neither
    /// source exists or the template cannot be compiled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nxm_tokenizer::chat_template::ChatTemplate;
    /// let tpl = ChatTemplate::from_model_dir(std::path::Path::new("model_dir"))?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let jinja_path = model_dir.join("chat_template.jinja");
        let config_path = model_dir.join("tokenizer_config.json");
        let re = compile_generation_marker_re()?;

        let (template, source, bos_token) = if jinja_path.exists() {
            let content = std::fs::read_to_string(&jinja_path)
                .with_context(|| format!("reading chat template from {}", jinja_path.display()))?;
            let bos_token = read_bos_token(&config_path);
            (
                strip_generation_markers(&content, &re),
                TemplateSource::File { path: jinja_path },
                bos_token,
            )
        } else if config_path.exists() {
            let content = std::fs::read_to_string(&config_path).with_context(|| {
                format!("reading tokenizer config from {}", config_path.display())
            })?;
            let config: serde_json::Value = serde_json::from_str(&content).with_context(|| {
                format!("parsing tokenizer config from {}", config_path.display())
            })?;
            let template = config
                .get("chat_template")
                .and_then(|v| v.as_str())
                .with_context(|| {
                    format!("`chat_template` key not found in {}", config_path.display())
                })?
                .to_string();
            let bos_token = read_bos_token(&config_path);
            (
                strip_generation_markers(&template, &re),
                TemplateSource::ConfigKey { path: config_path },
                bos_token,
            )
        } else {
            anyhow::bail!(
                "no chat template found in {}: expected `chat_template.jinja` or a `chat_template` key in `tokenizer_config.json`",
                model_dir.display()
            )
        };

        let mut env = Environment::new();
        env.add_template_owned("chat_template", template)
            .with_context(|| "failed to compile chat template")?;
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<(), minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("raise_exception: {msg}"),
                ))
            },
        );

        tracing::info!(
            target: "engine_mlx::tokenizer::chat_template",
            "chat template loaded from {} (bos_token={:?})",
            source_label(&source),
            bos_token
        );

        Ok(Self {
            env,
            source,
            bos_token,
        })
    }

    /// Render the chat template for `messages` into a prompt string.
    ///
    /// `add_generation_prompt` controls whether the assistant turn opener is
    /// appended, as dictated by the model's own template.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use nxm_tokenizer::chat_template::{ChatMessage, ChatTemplate};
    /// # let tpl = ChatTemplate::from_model_dir(std::path::Path::new("model_dir"))?;
    /// let prompt = tpl.render(&[ChatMessage { role: "user".into(), content: "hello".into() }], true)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the template references an undefined variable or
    /// calls `raise_exception`, or if rendering fails for any other reason.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        tracing::debug!(
            target: "engine_mlx::tokenizer::chat_template",
            "render chat template (n_messages={}, add_generation_prompt={})",
            messages.len(),
            add_generation_prompt
        );
        let tmpl = self
            .env
            .get_template("chat_template")
            .with_context(|| "chat template not compiled")?;
        let ctx = RenderContext {
            messages,
            add_generation_prompt,
            bos_token: &self.bos_token,
        };
        tmpl.render(&ctx).with_context(|| {
            format!(
                "failed to render chat template from {}",
                source_label(&self.source)
            )
        })
    }
}

fn source_label(source: &TemplateSource) -> String {
    match source {
        TemplateSource::File { path } => format!("file:{}", path.display()),
        TemplateSource::ConfigKey { path } => format!("config-key:{}", path.display()),
    }
}

/// Reads `bos_token` from `tokenizer_config.json`, or an empty string when
/// the file/key is absent. `bos_token` may be a plain string or an object
/// `{"content": "..."}` (common in HF configs); the object's `.content` is
/// extracted in that case. Documented choice: an absent `bos_token` renders
/// as empty so templates that start with `{{- bos_token }}` degrade
/// gracefully instead of failing. Read/parse errors are logged as warnings
/// (not surfaced as errors) so a broken `tokenizer_config.json` never masks
/// a valid `chat_template.jinja`.
fn read_bos_token(config_path: &Path) -> String {
    let content = match std::fs::read_to_string(config_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(target: "engine_mlx::tokenizer::chat_template",
                "cannot read tokenizer_config.json for bos_token: {e}");
            return String::new();
        }
    };
    let config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(target: "engine_mlx::tokenizer::chat_template",
                "cannot parse tokenizer_config.json for bos_token: {e}");
            return String::new();
        }
    };
    match config.get("bos_token") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(m)) => m
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

#[derive(Serialize)]
struct RenderContext<'a> {
    messages: &'a [ChatMessage],
    add_generation_prompt: bool,
    bos_token: &'a str,
}
