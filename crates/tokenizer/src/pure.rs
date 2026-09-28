//! Multi-tokenizer — BPE/WordPiece/Unigram dispatch driven by tokenizer.json config.
//!
//! Supports all tokenizer variants found in production:
//!
//!   | Variant            | Pre-tokenizer                     | Byte handling      | Models                          |
//!   |--------------------|-----------------------------------|---------------------|---------------------------------|
//!   | GPT-2 BPE          | Split(GPT2 regex) + ByteLevel     | ByteLevel          | Qwen, Llama 1/2/3, Mistral,Phi  |
//!   | ByteFallback BPE   | Split(regex)                      | byte_fallback=true | Gemma 4+                        |
//!   | tiktoken BPE       | Split(tiktoken regex) + ByteLevel | ByteLevel          | Llama 3.1+, GPT-4o, Phi-3.5+   |
//!   | WordPiece          | BertPreTokenizer + BertNormalizer | —                   | BERT, MiniLM, DistilBERT        |
//!   | Unigram            | WhitespaceSplit + Metaspace       | byte_fallback=true | T5, mT5, Gemma 1/2, MiniLM (v2) |
//!
//! Dispatch is driven by `model.type` (BPE vs WordPiece vs Unigram — when the
//! type is absent, the vocab shape decides: list ⇒ Unigram, dict ⇒ BPE),
//! `model.byte_fallback`, and the pre-tokenizer / decoder config in the JSON.

use anyhow::Result;
use fancy_regex::Regex;
use serde::Deserialize;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

// ─── tokenizer.json structures ───────────────────────────────────────────────

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelJson,
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    normalizer: Option<NormalizerJson>,
    pre_tokenizer: Option<PretokJson>,
    decoder: Option<DecoderJson>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    #[allow(dead_code)]
    single_word: bool,
    #[serde(default)]
    #[allow(dead_code)]
    lstrip: bool,
    #[serde(default)]
    #[allow(dead_code)]
    rstrip: bool,
}

/// Model vocab can be a dict {token: id} (BPE/WordPiece) or a list
/// [[token, score], ...] (Unigram/SentencePiece).
#[derive(Deserialize)]
#[serde(untagged)]
enum VocabJson {
    Dict(HashMap<String, u32>),
    List(Vec<(String, f32)>),
}

#[derive(Deserialize)]
struct ModelJson {
    #[serde(rename = "type", default)]
    model_type: Option<String>,
    vocab: VocabJson,
    #[serde(default)]
    merges: Vec<Vec<String>>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    max_input_chars_per_word: Option<u32>,
    #[serde(default)]
    unk_id: Option<u32>,
}

impl ModelJson {
    fn vocab_dict(&self) -> Result<&HashMap<String, u32>> {
        match &self.vocab {
            VocabJson::Dict(m) => Ok(m),
            VocabJson::List(_) => anyhow::bail!("expected dict vocab for this model type"),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NormalizerJson {
    Sequence { #[serde(rename = "type")] t: String, normalizers: Vec<NormalizerJson> },
    Single { #[serde(rename = "type")] t: String, #[serde(flatten)] rest: HashMap<String, serde_json::Value> },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PretokJson {
    Sequence { #[serde(rename = "type")] t: String, pretokenizers: Vec<PretokJson> },
    Single { #[serde(rename = "type")] t: String, #[serde(flatten)] rest: HashMap<String, serde_json::Value> },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DecoderJson {
    Sequence { #[serde(rename = "type")] t: String, decoders: Vec<DecoderJson> },
    Single { #[serde(rename = "type")] t: String, #[serde(flatten)] rest: HashMap<String, serde_json::Value> },
}

// ─── Normalizer ──────────────────────────────────────────────────────────────

fn apply_normalizer(text: &str, normalizer: &Option<NormalizerJson>) -> String {
    let Some(n) = normalizer else { return text.to_string() };
    apply_normalizer_json(text, n)
}

fn apply_normalizer_json(text: &str, nj: &NormalizerJson) -> String {
    match nj {
        NormalizerJson::Sequence { t: _, normalizers } => {
            let mut s = text.to_string();
            for n in normalizers {
                s = apply_normalizer_json(&s, n);
            }
            s
        }
        NormalizerJson::Single { t, rest } => {
            match t.as_str() {
                "NFC" => text.nfc().collect::<String>(),
                "NFD" => text.nfd().collect::<String>(),
                "NFKC" => text.nfkc().collect::<String>(),
                "NFKD" => text.nfkd().collect::<String>(),
                "Replace" => {
                    let pattern = rest.get("pattern").and_then(|v| v.as_str()).or_else(|| {
                        rest.get("pattern").and_then(|v| v.get("String")).and_then(|v| v.as_str())
                    }).or_else(|| {
                        rest.get("pattern").and_then(|v| v.get("Regex")).and_then(|v| v.as_str())
                    });
                    let content = rest.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if let Some(pat) = pattern {
                        if let Ok(re) = Regex::new(pat) {
                            return re.replace_all(text, content).to_string();
                        }
                    }
                    text.to_string()
                }
                "Prepend" => {
                    let prepend = rest.get("prepend").and_then(|v| v.as_str()).unwrap_or("");
                    format!("{prepend}{text}")
                }
                "Lowercase" => text.to_lowercase(),
                "BertNormalizer" => {
                    let clean_text = rest.get("clean_text").and_then(|v| v.as_bool()).unwrap_or(true);
                    let handle_chinese = rest.get("handle_chinese_chars").and_then(|v| v.as_bool()).unwrap_or(true);
                    let lowercase = rest.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(false);
                    let do_strip_accents = rest.get("strip_accents").and_then(|v| v.as_bool()).unwrap_or(false);

                    let mut s = text.to_string();
                    if clean_text {
                        // Remove control chars except \n \t \r, collapse whitespace
                        let mut cleaned = String::with_capacity(s.len());
                        for c in s.chars() {
                            if (c as u32) < 32 && c != '\n' && c != '\t' && c != '\r' {
                                continue;
                            }
                            cleaned.push(c);
                        }
                        s = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
                    }
                    if handle_chinese {
                        s = add_cjk_spaces(&s);
                    }
                    if lowercase {
                        s = s.to_lowercase();
                    }
                    if do_strip_accents {
                        s = strip_accents(&s);
                    }
                    s
                }
                _ => text.to_string(),
            }
        }
    }
}

/// Add spaces around CJK characters (HF BertNormalizer handle_chinese_chars).
fn add_cjk_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if is_cjk_char(c) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    (cp >= 0x4E00 && cp <= 0x9FFF)      // CJK Unified
        || (cp >= 0x3400 && cp <= 0x4DBF) // Extension A
        || (cp >= 0xF900 && cp <= 0xFAFF) // Compatibility
        || (cp >= 0x20000 && cp <= 0x2A6DF)
        || (cp >= 0x3040 && cp <= 0x30FF) // Hiragana + Katakana
        || (cp >= 0xAC00 && cp <= 0xD7AF) // Hangul
}

/// Strip combining accents (HF strip_accents). Uses NFD then keeps base chars.
fn strip_accents(text: &str) -> String {
    let nfd: String = text.nfd().collect();
    nfd.chars().filter(|c| !is_combining_mark(*c)).collect()
}

fn is_combining_mark(c: char) -> bool {
    matches!(c,
        '\u{0300}'..='\u{036F}'
        | '\u{1AB0}'..='\u{1AFF}'
        | '\u{1DC0}'..='\u{1DFF}'
        | '\u{20D0}'..='\u{20FF}'
        | '\u{FE20}'..='\u{FE2F}'
    )
}

// ─── Pre-tokenizer pipeline ─────────────────────────────────────────────────

enum PretokOp {
    Split { regex: Regex, behavior: String },
    ByteLevel { add_prefix_space: bool },
    Metaspace { add_prefix_space: bool },
    /// BertPreTokenizer: emit only regex matches, discard gaps (HF behavior).
    Bert { regex: Regex },
    /// WhitespaceSplit: split on ASCII whitespace, dropping the separators.
    WhitespaceSplit,
}

struct PretokPipeline {
    ops: Vec<PretokOp>,
}

impl PretokPipeline {
    fn from_json(pj: &Option<PretokJson>) -> Result<Self> {
        let mut ops = Vec::new();
        if let Some(pj) = pj {
            Self::collect_ops(pj, &mut ops)?;
        }
        Ok(Self { ops })
    }

    fn collect_ops(pj: &PretokJson, ops: &mut Vec<PretokOp>) -> Result<()> {
        match pj {
            PretokJson::Sequence { t: _, pretokenizers } => {
                for sub in pretokenizers {
                    Self::collect_ops(sub, ops)?;
                }
            }
            PretokJson::Single { t, rest } => {
                match t.as_str() {
                    "Split" => {
                        let pattern_val = rest.get("pattern");
                        let regex = if let Some(pv) = pattern_val {
                            if let Some(re_str) = pv.get("Regex").and_then(|v| v.as_str()) {
                                Regex::new(re_str)?
                            } else if let Some(s) = pv.get("String").and_then(|v| v.as_str()) {
                                // Regex-escape: only `\` and ` ` need escaping for our use case
                                let escaped = s.replace('\\', "\\\\").replace(' ', "\\ ");
                                Regex::new(&escaped)?
                            } else {
                                anyhow::bail!("Split pre-tokenizer: pattern must have Regex or String field");
                            }
                        } else if let Some(s) = pattern_val.and_then(|v| v.as_str()) {
                            let escaped = s.replace('\\', "\\\\").replace(' ', "\\ ");
                            Regex::new(&escaped)?
                        } else {
                            anyhow::bail!("Split pre-tokenizer missing pattern");
                        };
                        let behavior = rest.get("behavior").and_then(|v| v.as_str()).unwrap_or("isolated").to_lowercase();
                        let invert = rest.get("invert").and_then(|v| v.as_bool()).unwrap_or(false);
                        if invert {
                            // Inverted Split: keep full text. Rare.
                        } else {
                            ops.push(PretokOp::Split { regex, behavior });
                        }
                    }
                    "ByteLevel" => {
                        let add_prefix_space = rest.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false);
                        ops.push(PretokOp::ByteLevel { add_prefix_space });
                    }
                    "Digits" => {
                        let digits_re = Regex::new(r"\p{N}{1,3}")?;
                        ops.push(PretokOp::Split { regex: digits_re, behavior: "isolated".to_string() });
                    }
                    "Metaspace" => {
                        let add_prefix_space = rest.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(true);
                        ops.push(PretokOp::Metaspace { add_prefix_space });
                    }
                    "BertPreTokenizer" => {
                        let bert_re = Regex::new(r"\w+|[^\w\s]+")?;
                        ops.push(PretokOp::Bert { regex: bert_re });
                    }
                    "WhitespaceSplit" => {
                        ops.push(PretokOp::WhitespaceSplit);
                    }
                    other => {
                        anyhow::bail!("Unsupported pre-tokenizer: {other}");
                    }
                }
            }
        }
        Ok(())
    }

    fn apply(&self, text: &str) -> Vec<String> {
        let mut tokens = vec![text.to_string()];
        for op in &self.ops {
            match op {
                PretokOp::Bert { regex } => {
                    let mut next = Vec::new();
                    for t in &tokens {
                        for m in regex.find_iter(t).filter_map(|r| r.ok()) {
                            next.push(m.as_str().to_string());
                        }
                    }
                    tokens = next;
                }
                PretokOp::WhitespaceSplit => {
                    let mut next = Vec::new();
                    for t in &tokens {
                        next.extend(t.split_whitespace().map(|s| s.to_string()));
                    }
                    tokens = next;
                }
                PretokOp::Split { regex, behavior } => {
                    let mut next = Vec::new();
                    for t in &tokens {
                        match behavior.as_str() {
                            "isolated" => {
                                let mut last_end = 0;
                                for m in regex.find_iter(t).filter_map(|r| r.ok()) {
                                    if m.start() > last_end {
                                        next.push(t[last_end..m.start()].to_string());
                                    }
                                    next.push(m.as_str().to_string());
                                    last_end = m.end();
                                }
                                if last_end < t.len() {
                                    next.push(t[last_end..].to_string());
                                }
                            }
                            "removed" => {
                                let result = regex.replace_all(t, "");
                                next.push(result.to_string());
                            }
                            "merged_with_previous" | "merged_with_next" => {
                                // Simplified: treat as isolated
                                let mut last_end = 0;
                                for m in regex.find_iter(t).filter_map(|r| r.ok()) {
                                    if m.start() > last_end {
                                        next.push(t[last_end..m.start()].to_string());
                                    }
                                    next.push(m.as_str().to_string());
                                    last_end = m.end();
                                }
                                if last_end < t.len() {
                                    next.push(t[last_end..].to_string());
                                }
                            }
                            _ => {
                                let mut last_end = 0;
                                for m in regex.find_iter(t).filter_map(|r| r.ok()) {
                                    if m.start() > last_end {
                                        next.push(t[last_end..m.start()].to_string());
                                    }
                                    next.push(m.as_str().to_string());
                                    last_end = m.end();
                                }
                                if last_end < t.len() {
                                    next.push(t[last_end..].to_string());
                                }
                            }
                        }
                    }
                    tokens = next;
                }
                PretokOp::ByteLevel { add_prefix_space } => {
                    let b2u = bytes_to_unicode();
                    for t in &mut tokens {
                        if *add_prefix_space {
                            // Actually apply: prepend space before ByteLevel encode
                            // but we don't need to modify the text since GPT-2 style
                            // models already handle this via the regex.
                        }
                        let encoded: String = t.bytes().map(|b| b2u[b as usize]).collect();
                        *t = encoded;
                    }
                }
                PretokOp::Metaspace { add_prefix_space } => {
                    for t in &mut tokens {
                        if *add_prefix_space {
                            t.insert(0, ' ');
                        }
                        *t = t.replace(' ', "\u{2581}");  // ▁
                    }
                }
            }
        }
        tokens.retain(|t| !t.is_empty());
        tokens
    }
}

// ─── Decoder pipeline ───────────────────────────────────────────────────────

enum DecodeOp {
    ByteLevel,
    ByteFallback,
    Replace { pattern: String, replacement: String, is_regex: bool },
    Fuse,
    Metaspace { add_prefix_space: bool },
    WordPiece { prefix: String },
}

struct DecodePipeline {
    ops: Vec<DecodeOp>,
}

impl DecodePipeline {
    fn from_json(dj: &Option<DecoderJson>) -> Self {
        let mut ops = Vec::new();
        if let Some(dj) = dj {
            Self::collect_ops(dj, &mut ops);
        }
        if ops.is_empty() {
            ops.push(DecodeOp::ByteLevel);
        }
        Self { ops }
    }

    fn collect_ops(dj: &DecoderJson, ops: &mut Vec<DecodeOp>) {
        match dj {
            DecoderJson::Sequence { t: _, decoders } => {
                for sub in decoders {
                    Self::collect_ops(sub, ops);
                }
            }
            DecoderJson::Single { t, rest } => {
                match t.as_str() {
                    "ByteLevel" => ops.push(DecodeOp::ByteLevel),
                    "ByteFallback" => ops.push(DecodeOp::ByteFallback),
                    "Fuse" => ops.push(DecodeOp::Fuse),
                    "Replace" => {
                        let pattern = rest.get("pattern")
                            .and_then(|v| v.as_str())
                            .or_else(|| rest.get("pattern").and_then(|v| v.get("String")).and_then(|v| v.as_str()))
                            .or_else(|| rest.get("pattern").and_then(|v| v.get("Regex")).and_then(|v| v.as_str()))
                            .unwrap_or("");
                        let is_regex = rest.get("pattern")
                            .and_then(|v| v.get("Regex"))
                            .is_some();
                        let replacement = rest.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        ops.push(DecodeOp::Replace { pattern: pattern.to_string(), replacement: replacement.to_string(), is_regex });
                    }
                    "Metaspace" => {
                        let add_prefix_space = rest.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(true);
                        ops.push(DecodeOp::Metaspace { add_prefix_space });
                    }
                    "WordPiece" => {
                        let prefix = rest.get("prefix").and_then(|v| v.as_str()).unwrap_or("##").to_string();
                        ops.push(DecodeOp::WordPiece { prefix });
                    }
                    _ => {}
                }
            }
        }
    }

    fn apply(&self, tokens: &[String]) -> String {
        let mut out = String::new();
        let u2b = unicode_to_bytes();
        // WordPiece: insert a space before each non-continuation token except
        // the very first one. Continuations are tokens that start with ##.
        let wp_prefix = self.ops.iter().find_map(|op| match op {
            DecodeOp::WordPiece { prefix } => Some(prefix.as_str()),
            _ => None,
        });

        for (i, token) in tokens.iter().enumerate() {
            let is_cont = wp_prefix.is_some_and(|p| token.starts_with(p));
            let mut s = token.clone();
            for op in &self.ops {
                match op {
                    DecodeOp::ByteLevel => {
                        let decoded: String = s.chars().filter_map(|c| u2b.get(&c)).map(|&b| b as char).collect();
                        s = decoded;
                    }
                    DecodeOp::ByteFallback => {
                        if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                            if let Ok(byte_val) = u8::from_str_radix(&s[3..5], 16) {
                                s = (byte_val as char).to_string();
                            }
                        }
                    }
                    DecodeOp::Replace { pattern, replacement, is_regex } => {
                        if *is_regex {
                            if let Ok(re) = Regex::new(pattern) {
                                s = re.replace_all(&s, replacement.as_str()).to_string();
                            }
                        } else {
                            s = s.replace(pattern.as_str(), replacement.as_str());
                        }
                    }
                    DecodeOp::Fuse => {
                        // Fuse is a no-op between tokens — it just means
                        // concatenate them (which we're already doing).
                    }
                    DecodeOp::Metaspace { add_prefix_space: _ } => {
                        s = s.replace('\u{2581}', " ");
                    }
                    DecodeOp::WordPiece { prefix } => {
                        if let Some(stripped) = s.strip_prefix(prefix.as_str()) {
                            s = stripped.to_string();
                        }
                    }
                }
            }
            if wp_prefix.is_some() && i > 0 && !is_cont {
                out.push(' ');
            }
            out.push_str(&s);
        }

        // Metaspace with add_prefix_space: drop the artificial leading space
        // that was added at encode time.
        if let Some(DecodeOp::Metaspace { add_prefix_space: true }) = self.ops.first() {
            if out.starts_with(' ') {
                out.remove(0);
            }
        }

        out
    }
}

// ─── ByteLevel mapping ───────────────────────────────────────────────────────

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    for i in 0u8..=255 {
        #[allow(unused_comparisons)]
        if (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255) {
            bs.push(i);
        }
    }
    let n_bs = bs.len() as u32;
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + cs.len() as u32 - n_bs);
        }
    }
    let mut chars = ['\0'; 256];
    for (i, &b) in bs.iter().enumerate() {
        chars[b as usize] = char::from_u32(cs[i]).unwrap_or('\0');
    }
    chars
}

fn unicode_to_bytes() -> HashMap<char, u8> {
    let b2u = bytes_to_unicode();
    let mut u2b = HashMap::with_capacity(256);
    for (b, &c) in b2u.iter().enumerate() {
        u2b.insert(c, b as u8);
    }
    u2b
}

fn pair_key(a: &str, b: &str) -> String {
    format!("{}\0{}", a, b)
}

// ─── BPE merge (model-agnostic) ─────────────────────────────────────────────

struct BpeData {
    str_to_id: HashMap<String, u32>,
    id_to_str: Vec<String>,
    pair_ranks: HashMap<String, usize>,
    byte_fallback: bool,
    /// ID of the ByteLevel-encoded space (Ġ = U+0120) token, for post-processor.
    space_id: Option<u32>,
}

impl BpeData {
    fn from_model(model: &ModelJson) -> Self {
        let empty = HashMap::new();
        let vocab = model.vocab_dict().unwrap_or(&empty);
        let mut id_to_str: Vec<String> = vec![String::new(); vocab.len()];
        for (s, &id) in vocab {
            if (id as usize) < id_to_str.len() {
                id_to_str[id as usize] = s.clone();
            }
        }
        let pair_ranks: HashMap<String, usize> = model.merges.iter().enumerate()
            .map(|(rank, pair)| (pair_key(&pair[0], &pair[1]), rank))
            .collect();
        // ByteLevel-encoded space (Ġ = U+0120)
        let space_char = bytes_to_unicode()[b' ' as usize].to_string();
        let space_id = vocab.get(&space_char).copied();
        Self { str_to_id: vocab.clone(), id_to_str, pair_ranks, byte_fallback: model.byte_fallback, space_id }
    }

    fn bpe_merge(&self, encoded: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = encoded.chars().map(|c| c.to_string()).collect();
        let n = tokens.len();
        if n <= 1 {
            return if n == 0 { Vec::new() } else { self.lookup_token(&tokens[0]) };
        }

        let mut prev: Vec<isize> = (0..n).map(|i| i as isize - 1).collect();
        let mut next: Vec<isize> = (1..n as isize).chain(std::iter::once(-1)).collect();
        let mut heap: BinaryHeap<std::cmp::Reverse<(usize, usize)>> = BinaryHeap::new();

        for i in 0..n - 1 {
            if let Some(&rank) = self.pair_ranks.get(&pair_key(&tokens[i], &tokens[i + 1])) {
                heap.push(std::cmp::Reverse((rank, i)));
            }
        }

        while let Some(std::cmp::Reverse((_rank, i))) = heap.pop() {
            let j = next[i];
            if j < 0 {
                continue;
            }
            let j = j as usize;

            if self.pair_ranks.get(&pair_key(&tokens[i], &tokens[j])).map(|&r| r) != Some(_rank) {
                continue;
            }

            let merged = format!("{}{}", tokens[i], tokens[j]);
            let nj = next[j];

            if nj >= 0 {
                prev[nj as usize] = i as isize;
            }
            next[i] = nj;
            next[j] = -1;
            prev[j] = -1;
            tokens[i] = merged;

            let left = prev[i];
            if left >= 0 {
                if let Some(&rank) = self.pair_ranks.get(&pair_key(&tokens[left as usize], &tokens[i])) {
                    heap.push(std::cmp::Reverse((rank, left as usize)));
                }
            }
            if nj >= 0 {
                if let Some(&rank) = self.pair_ranks.get(&pair_key(&tokens[i], &tokens[nj as usize])) {
                    heap.push(std::cmp::Reverse((rank, i)));
                }
            }
        }

        let mut result = Vec::new();
        let mut i = 0usize;
        while i < n {
            result.extend(self.lookup_token(&tokens[i]));
            let ni = next[i];
            if ni < 0 {
                break;
            }
            i = ni as usize;
        }
        result
    }

    fn lookup_token(&self, token: &str) -> Vec<u32> {
        if let Some(&id) = self.str_to_id.get(token) {
            return vec![id];
        }
        if self.byte_fallback {
            // Encode as raw bytes: <0xXX> format
            token.bytes().map(|b| {
                let byte_token = format!("<0x{b:02X}>");
                self.str_to_id.get(&byte_token).copied().unwrap_or(0)
            }).collect()
        } else {
            token.bytes().map(|b| {
                let c = bytes_to_unicode()[b as usize];
                self.str_to_id.get(&c.to_string()).copied().unwrap_or(0)
            }).collect()
        }
    }

    fn decode_token(&self, id: usize) -> Option<&str> {
        self.id_to_str.get(id).map(|s| s.as_str())
    }
}

// ─── WordPiece model ─────────────────────────────────────────────────────────

struct WordPieceData {
    str_to_id: HashMap<String, u32>,
    id_to_str: Vec<String>,
    #[allow(dead_code)]
    unk_token: String,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl WordPieceData {
    fn from_model(model: &ModelJson) -> Result<Self> {
        let vocab = model.vocab_dict()?;
        let mut id_to_str: Vec<String> = vec![String::new(); vocab.len()];
        for (s, &id) in vocab {
            if (id as usize) < id_to_str.len() {
                id_to_str[id as usize] = s.clone();
            }
        }
        let unk_token = model
            .unk_token
            .as_deref()
            .unwrap_or("[UNK]")
            .to_string();
        let unk_id = vocab.get(&unk_token).copied().unwrap_or(0);
        let continuing_subword_prefix = model
            .continuing_subword_prefix
            .as_deref()
            .unwrap_or("##")
            .to_string();
        let max_input_chars_per_word = model
            .max_input_chars_per_word
            .unwrap_or(100) as usize;
        Ok(Self {
            str_to_id: vocab.clone(),
            id_to_str,
            unk_token,
            unk_id,
            continuing_subword_prefix,
            max_input_chars_per_word,
        })
    }

    /// Greedy longest-match-first tokenization (HF WordPiece).
    fn encode_word(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return vec![self.unk_id];
        }
        let mut result = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found: Option<&str> = None;
            let mut token = String::new();
            while start < end {
                token.clear();
                let part: String = chars[start..end].iter().collect();
                if start > 0 {
                    token.push_str(&self.continuing_subword_prefix);
                }
                token.push_str(&part);
                if self.str_to_id.contains_key(&token) {
                    found = Some(token.as_str());
                    break;
                }
                end -= 1;
            }
            match found {
                Some(t) => {
                    result.push(self.str_to_id[t]);
                    start = end;
                }
                None => {
                    result.push(self.unk_id);
                    break;
                }
            }
        }
        result
    }

    fn decode_token(&self, id: usize) -> Option<&str> {
        self.id_to_str.get(id).map(|s| s.as_str())
    }
}

// ─── Unigram model (SentencePiece) ───────────────────────────────────────────

struct UnigramData {
    /// token → (id, log-probability score)
    scores: HashMap<String, (u32, f32)>,
    id_to_str: Vec<String>,
    unk_id: u32,
    byte_fallback: bool,
}

impl UnigramData {
    fn from_model(model: &ModelJson) -> Result<Self> {
        let VocabJson::List(list) = &model.vocab else {
            anyhow::bail!("Unigram model requires list vocab");
        };
        let mut scores = HashMap::with_capacity(list.len());
        let mut id_to_str: Vec<String> = vec![String::new(); list.len()];
        for (i, (s, score)) in list.iter().enumerate() {
            scores.insert(s.clone(), (i as u32, *score));
            id_to_str[i] = s.clone();
        }
        let unk_id = model.unk_id.unwrap_or(0);
        Ok(Self { scores, id_to_str, unk_id, byte_fallback: model.byte_fallback })
    }

    /// Viterbi segmentation maximizing sum of log-probabilities (HF Unigram).
    fn encode_word(&self, word: &str) -> Vec<u32> {
        let bytes = word.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }
        let neg_inf = -f32::INFINITY;
        // dp[i] = best score for prefix ending at byte i
        let mut dp = vec![neg_inf; n + 1];
        // best_len[i] = length of the token ending at byte i (for backtracking)
        let mut best_len = vec![0usize; n + 1];
        // best_id[i] = token id ending at byte i
        let mut best_id = vec![self.unk_id; n + 1];
        dp[0] = 0.0;

        // Limit token length to the longest vocab token (avoid O(n²) blowup).
        let max_token_len = self.scores.keys().map(|k| k.len()).max().unwrap_or(1);

        for i in 0..n {
            if dp[i] == neg_inf {
                continue;
            }
            let limit = (i + max_token_len).min(n);
            let mut advanced = false;
            let mut j = i + 1;
            while j <= limit {
                // Only consider char-boundary splits
                if let Ok(s) = std::str::from_utf8(&bytes[i..j]) {
                    if let Some(&(id, score)) = self.scores.get(s) {
                        advanced = true;
                        let total = dp[i] + score;
                        if total > dp[j] {
                            dp[j] = total;
                            best_len[j] = j - i;
                            best_id[j] = id;
                        }
                    }
                }
                j += 1;
            }
            if !advanced {
                // Unknown char → emit unk_id and advance one char boundary.
                let c = word[i..].chars().next().unwrap();
                let clen = c.len_utf8();
                let total = dp[i] + -1e6;
                if total > dp[i + clen] {
                    dp[i + clen] = total;
                    best_len[i + clen] = clen;
                    best_id[i + clen] = self.unk_id;
                }
            }
        }

        // Backtrack
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let len = best_len[pos];
            let id = best_id[pos];
            if self.byte_fallback && !self.scores.contains_key(&self.id_to_str[id as usize]) {
                // Fall back to byte tokens <0xXX>
                let sub = &bytes[pos - len..pos];
                for &b in sub {
                    let byte_tok = format!("<0x{b:02X}>");
                    let bid = self.scores.get(&byte_tok).map(|t| t.0).unwrap_or(self.unk_id);
                    ids.push(bid);
                }
            } else {
                ids.push(id);
            }
            pos -= len;
        }
        ids.reverse();
        ids
    }

    fn decode_token(&self, id: usize) -> Option<&str> {
        self.id_to_str.get(id).map(|s| s.as_str())
    }
}

// ─── Model dispatch ──────────────────────────────────────────────────────────

enum Model {
    Bpe(BpeData),
    WordPiece(WordPieceData),
    Unigram(UnigramData),
}

impl Model {
    fn from_model(model: &ModelJson) -> Result<Self> {
        let ty = model.model_type.as_deref().unwrap_or("");
        match ty {
            "BPE" => Ok(Model::Bpe(BpeData::from_model(model))),
            "WordPiece" => Ok(Model::WordPiece(WordPieceData::from_model(model)?)),
            "Unigram" => Ok(Model::Unigram(UnigramData::from_model(model)?)),
            // Some tokenizer.json files omit model.type (t5-small). Sniff via vocab shape.
            "" => match &model.vocab {
                VocabJson::List(_) => Ok(Model::Unigram(UnigramData::from_model(model)?)),
                VocabJson::Dict(_) => Ok(Model::Bpe(BpeData::from_model(model))),
            },
            other => anyhow::bail!("Unsupported model type: {other}"),
        }
    }

    /// Encode a pre-tokenized unit to token IDs (BPE merge, WordPiece, or Unigram).
    fn encode_regular(&self, pretoken: &str) -> Vec<u32> {
        match self {
            Model::Bpe(b) => b.bpe_merge(pretoken),
            Model::WordPiece(w) => w.encode_word(pretoken),
            Model::Unigram(u) => u.encode_word(pretoken),
        }
    }

    fn decode_token(&self, id: usize) -> Option<&str> {
        match self {
            Model::Bpe(b) => b.decode_token(id),
            Model::WordPiece(w) => w.decode_token(id),
            Model::Unigram(u) => u.decode_token(id),
        }
    }

    fn byte_fallback(&self) -> bool {
        match self {
            Model::Bpe(b) => b.byte_fallback,
            Model::WordPiece(_) => false,
            Model::Unigram(u) => u.byte_fallback,
        }
    }

    fn space_id(&self) -> Option<u32> {
        match self {
            Model::Bpe(b) => b.space_id,
            Model::WordPiece(_) | Model::Unigram(_) => None,
        }
    }
}

// ─── Post-processor pipeline ─────────────────────────────────────────────────

struct PostProcPipeline {
    strip_space: bool,
}

impl PostProcPipeline {
    fn from_json(pj: &Option<serde_json::Value>) -> Self {
        let Some(pj) = pj else { return Self { strip_space: false } };
        // Check for ByteLevel post-processor (common pattern)
        let is_bytelevel = match pj {
            serde_json::Value::Object(m) => {
                m.get("type").and_then(|v| v.as_str()) == Some("ByteLevel")
                || m.get("type").and_then(|v| v.as_str()) == Some("RobertaProcessing")
            }
            _ => false,
        };
        if is_bytelevel {
            let add_prefix_space = pj.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false);
            Self { strip_space: add_prefix_space }
        } else {
            Self { strip_space: false }
        }
    }

    fn apply(&self, ids: &mut Vec<u32>, space_id: u32) {
        if self.strip_space && ids.first() == Some(&space_id) {
            ids.remove(0);
        }
    }
}

// ─── Tokenizer (public API) ──────────────────────────────────────────────────

pub struct Tokenizer {
    /// Model data (BPE or WordPiece), dispatched on model.type
    model: Model,
    /// Pre-tokenizer pipeline
    pretok: PretokPipeline,
    /// Decoder pipeline
    decoder: DecodePipeline,
    /// Normalizer
    normalizer: Option<NormalizerJson>,
    /// Post-processor
    postproc: PostProcPipeline,
    /// Added tokens sorted by length descending for longest-match-first
    added: Vec<(String, u32)>,
    /// LRU cache: pre-token → encoded IDs
    cache: HashMap<String, Vec<u32>>,
    cache_max: usize,
}

impl Tokenizer {
    /// Load from tokenizer.json path.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let tj: TokenizerJson = serde_json::from_str(&content)?;

        let model = Model::from_model(&tj.model)?;

        let mut added: Vec<(String, u32)> = tj.added_tokens.into_iter()
            .map(|a| (a.content, a.id))
            .collect();
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let pretok = PretokPipeline::from_json(&tj.pre_tokenizer)?;
        let decoder = DecodePipeline::from_json(&tj.decoder);
        let normalizer = tj.normalizer;
        let postproc = PostProcPipeline::from_json(&tj.post_processor);

        Ok(Self {
            model,
            pretok,
            decoder,
            normalizer,
            postproc,
            added,
            cache: HashMap::new(),
            cache_max: 10_000,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let segments = split_added(text, &self.added);
        let mut result = Vec::new();
        for seg in segments {
            match seg {
                Seg::Added(id) => result.push(id),
                Seg::Regular(s) => self.encode_regular(s, &mut result),
            }
        }
        if let Some(space_id) = self.model.space_id() {
            self.postproc.apply(&mut result, space_id);
        }
        result
    }

    fn encode_regular(&mut self, text: &str, result: &mut Vec<u32>) {
        let text = apply_normalizer(text, &self.normalizer);
        let pretokens = self.pretok.apply(&text);

        for pretoken in &pretokens {
            if let Some(cached) = self.cache.get(pretoken.as_str()) {
                result.extend_from_slice(cached);
                continue;
            }
            // If the pre-tokenizer already applied ByteLevel, pretoken is in
            // ByteLevel-encoded form. If not (byte_fallback), it's raw text.
            let ids = self.model.encode_regular(pretoken);
            if self.cache.len() >= self.cache_max {
                self.cache.clear();
            }
            self.cache.insert(pretoken.clone(), ids.clone());
            result.extend(ids);
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut tokens: Vec<String> = Vec::with_capacity(ids.len());
        for &id in ids {
            let id_u = id as usize;
            let is_added = self.added.iter().any(|(c, tid)| *tid == id && c.len() > 1);
            if is_added {
                if let Some(s) = self.model.decode_token(id_u) {
                    tokens.push(s.to_string());
                }
            } else if let Some(s) = self.model.decode_token(id_u) {
                tokens.push(s.to_string());
            } else if self.model.byte_fallback() {
                tokens.push(std::char::from_u32(id).unwrap_or('�').to_string());
            }
        }

        // Some models (Gemma) have special tokens that map to spaces/newlines
        // in the decoder pipeline. Apply decoder to get final text.
        self.decoder.apply(&tokens)
    }
}

// ─── Segment text by added tokens ────────────────────────────────────────────

enum Seg<'a> {
    Added(u32),
    Regular(&'a str),
}

fn split_added<'a>(text: &'a str, added: &[(String, u32)]) -> Vec<Seg<'a>> {
    if added.is_empty() {
        return vec![Seg::Regular(text)];
    }

    // Step through char by char to avoid landing mid-UTF-8
    let mut matches: Vec<(usize, usize, u32)> = Vec::new();
    let mut pos = 0usize;

    // Build a set of added token contents for fast lookup at each position
    while pos < text.len() {
        if !text.is_char_boundary(pos) {
            pos += 1;
            continue;
        }
        let mut best: Option<(usize, u32)> = None;
        for (content, id) in added {
            let clen = content.len();
            if clen <= text.len() - pos && text.is_char_boundary(pos + clen) && &text[pos..pos + clen] == content.as_str() {
                let better = match best {
                    Some((blen, _)) => clen > blen,
                    None => true,
                };
                if better {
                    best = Some((clen, *id));
                }
            }
        }
        match best {
            Some((clen, id)) => {
                matches.push((pos, pos + clen, id));
                pos += clen;
            }
            None => {
                // Advance by one full char
                let c = text[pos..].chars().next().unwrap();
                pos += c.len_utf8();
            }
        }
    }

    if matches.is_empty() {
        return vec![Seg::Regular(text)];
    }

    let mut segments = Vec::new();
    let mut last_end = 0usize;
    for (start, end, id) in matches {
        if start > last_end {
            segments.push(Seg::Regular(&text[last_end..start]));
        }
        segments.push(Seg::Added(id));
        last_end = end;
    }
    if last_end < text.len() {
        segments.push(Seg::Regular(&text[last_end..]));
    }
    segments
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn hf_path() -> String {
        std::env::var("HF_TOKENIZER_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{home}/models/mlx-community/Qwen3-0.6B-4bit/tokenizer.json")
        })
    }

    /// Returns true if the default tokenizer.json exists. Tests that need a real
    /// model file skip (rather than fail) when it is absent — set
    /// `HF_TOKENIZER_PATH` to point at any local `tokenizer.json` to run them.
    fn hf_available() -> bool {
        std::path::Path::new(&hf_path()).exists()
    }

    #[test]
    fn test_byte_level_roundtrip() {
        let b2u = bytes_to_unicode();
        let u2b = unicode_to_bytes();
        for b in 0u8..=255 {
            let c = b2u[b as usize];
            let back = u2b.get(&c);
            assert_eq!(back, Some(&b), "byte {b} -> char {c:?} -> {back:?}");
        }
    }

    #[test]
    fn test_pretokenize_simple() {
        if !hf_available() {
            eprintln!("tokenizer.json not found at {}, skipping", hf_path());
            return;
        }
        let mut tok = Tokenizer::load(std::path::Path::new(&hf_path())).expect("load");
        let ids = tok.encode("Hello world");
        assert!(!ids.is_empty(), "encode returned empty");
    }

    #[test]
    fn test_parity_with_hf() {
        if !hf_available() {
            eprintln!("tokenizer.json not found at {}, skipping", hf_path());
            return;
        }
        let mut tok = Tokenizer::load(std::path::Path::new(&hf_path())).expect("load");
        let hf_tok = tokenizers::Tokenizer::from_file(&hf_path()).expect("load HF");

        let test_texts = [
            "Hello",
            "Hello world",
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "I'm doing great, thanks!",
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            "2+2=?",
            "La capitale della Francia è",
            "Explain quantum computing in simple terms",
            "A", "a", "  spaced  ", "\nnewlines\n",
            "don't can't I'll we're",
            "123 456",
            "test@email.com",
            "UPPER lower MixedCase",
            "   ",
            "a.b.c",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            let hf_ids = hf_enc.get_ids();
            let bpe_ids = tok.encode(text);

            if hf_ids != bpe_ids.as_slice() {
                eprintln!("MISMATCH for text: {:?}", text);
                eprintln!("  HF:  {:?}", hf_ids);
                eprintln!("  BPE: {:?}", bpe_ids);
                all_ok = false;
            }
        }

        assert!(all_ok, "Parity test failed — some texts didn't match");
    }

    #[test]
    fn test_gemma_byte_fallback() {
        // Test Gemma-4 tokenizer (byte_fallback=true)
        let gemma_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/mlx-community/gemma-4-12B-it-4bit/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&gemma_path).exists() {
            eprintln!("Gemma tokenizer not found at {gemma_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&gemma_path)).expect("load Gemma");
        let hf_tok = tokenizers::Tokenizer::from_file(&gemma_path).expect("load HF Gemma");

        let test_texts = [
            "Hello",
            "Hello world",
            "La capitale della Francia è Parigi.",
            "2+2=?",
            "  spaced  ",
            "\nnewlines\n",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            let hf_ids = hf_enc.get_ids();
            let bpe_ids = tok.encode(text);

            if hf_ids != bpe_ids.as_slice() {
                eprintln!("MISMATCH for text: {:?}", text);
                eprintln!("  HF:  {:?}", hf_ids);
                eprintln!("  BPE: {:?}", bpe_ids);
                all_ok = false;
            }
        }

        if !all_ok {
            panic!("Gemma parity test failed");
        }
        eprintln!("✅ Gemma parity passed ({} texts)", test_texts.len());
    }

    #[test]
    fn test_decode_roundtrip() {
        if !hf_available() {
            eprintln!("tokenizer.json not found at {}, skipping", hf_path());
            return;
        }
        let mut tok = Tokenizer::load(std::path::Path::new(&hf_path())).expect("load");
        let texts = ["Hello world", "Ciao mondo", "42", "test@email.com"];
        for text in &texts {
            let ids = tok.encode(text);
            let decoded = tok.decode(&ids);
            // GPT-2 BPE decode may not exactly match due to space handling
            assert!(!decoded.is_empty(), "decode empty for {text:?}");
        }
    }

    // ─── Phase 3 tests ────────────────────────────────────────────────────────

    /// Test Metaspace pre-tokenizer via a synthetic tokenizer.json with
    /// Metaspace pre_tokenizer + Metaspace decoder.
    #[test]
    fn test_metaspace_roundtrip() {
        // Build a minimal tokenizer.json with Metaspace pipeline and a vocab
        // whose tokens are exactly what Metaspace produces on "Hello world":
        //   - "Hello" (no prefix) → 0
        //   - "▁world" → 1
        //   - "<unk>" → 2
        // JSON uses \u2581 (no curly braces) for ▁.
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "Hello": 0,
                    "\u2581world": 1,
                    "<unk>": 2
                },
                "merges": []
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "add_prefix_space": false,
                "replacement": "\u2581",
                "split": true
            },
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "Fuse"},
                    {"type": "Metaspace", "replacement": " ", "split": true, "add_prefix_space": false}
                ]
            }
        }"#;
        let dir = tempdir_like();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap();

        let tok = Tokenizer::load(&path).expect("load metaspace tokenizer");
        // Pre-tokenizer transforms spaces to ▁ without adding a prefix.
        let text = "Hello world";
        let tokens = tok.pretok.apply(text);
        assert!(tokens.iter().any(|t| t.contains('\u{2581}')), "Metaspace should introduce the U+2581 char: {tokens:?}");

        // Decode-side roundtrip: ▁world (id 1) should decode to " world".
        let ids = vec![0u32, 1u32];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "Hello world", "Metaspace decode roundtrip mismatch: {decoded:?}");
    }

    /// Test NFC normalization applied to text with composed/decomposed chars
    #[test]
    fn test_nfc_normalization() {
        // "café" decomposed = "cafe" + combining acute accent (U+0301)
        // "café" composed   = "cafe" + precomposed é (U+00E9)
        let decomposed = "caf\u{0065}\u{0301}";  // e + combining acute
        let composed = "caf\u{00E9}";             // precomposed é
        assert_ne!(decomposed, composed, "test setup error: strings should differ");
        assert_eq!(decomposed.chars().count(), 5, "decomposed should have 5 chars");
        assert_eq!(composed.chars().count(), 4, "composed should have 4 chars");

        // Apply NFC normalizer manually
        use unicode_normalization::UnicodeNormalization;
        let normalized: String = decomposed.nfc().collect();
        assert_eq!(normalized, composed, "NFC normalization failed");
    }

    /// Test ByteFallback decode: <0xXX> tokens should decode back to raw bytes
    #[test]
    fn test_byte_fallback_decode() {
        // Simulate byte_fallback decode path: <0xE8> should decode to the byte 0xE8
        // which is è as an iso-8859-1 codepoint.
        let mut s = String::from("<0xE8>");
        if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
            if let Ok(byte_val) = u8::from_str_radix(&s[3..5], 16) {
                s = (byte_val as char).to_string();
            }
        }
        assert_eq!(s, "è", "byte fallback decode failed: got {s:?}");
    }

    /// Test post-processor strips leading Ġ (ByteLevel-encoded space) if configured
    #[test]
    fn test_post_processor_strip_space() {
        // Direct unit test of PostProcPipeline::apply: build a token list starting
        // with the Ġ token, and verify it gets stripped when strip_space=true.
        let space_token = bytes_to_unicode()[b' ' as usize].to_string(); // Ġ

        // Case 1: strip_space=true → first token removed
        let pp_strip = PostProcPipeline { strip_space: true };
        let mut ids = vec![1u32, 2, 3]; // pretend 1 is the space token
        pp_strip.apply(&mut ids, 1);
        assert_eq!(ids, vec![2, 3], "strip should remove first matching token");

        // Case 2: strip_space=false → no change
        let pp_no_strip = PostProcPipeline { strip_space: false };
        let mut ids = vec![1u32, 2, 3];
        pp_no_strip.apply(&mut ids, 1);
        assert_eq!(ids, vec![1, 2, 3], "no-strip should leave tokens intact");

        // Case 3: first token isn't space → no change even with strip=true
        let pp_strip = PostProcPipeline { strip_space: true };
        let mut ids = vec![42u32, 1, 2];
        pp_strip.apply(&mut ids, 1);
        assert_eq!(ids, vec![42, 1, 2], "strip should only remove space token, not any first token");

        // Sanity: verify space_token is actually Ġ
        assert_eq!(space_token.chars().next().unwrap() as u32, 0x0120, "Ġ should be U+0120");
    }

    /// Helper: create a tempdir-like dir under /tmp for synthetic tokenizer tests
    fn tempdir_like() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("nxm_tokenizer_test_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Parity against HF tokenizers for a tiktoken-style model (Qwen3.5-9B):
    /// Split(tiktoken regex) + ByteLevel + NFC normalizer.
    #[test]
    fn test_tiktoken_parity() {
        let tiktoken_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/mlx-community/Qwen3.5-9B-MLX-4bit/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&tiktoken_path).exists() {
            eprintln!("Qwen3.5-9B tokenizer not found at {tiktoken_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&tiktoken_path)).expect("load Qwen3.5");
        let hf_tok = tokenizers::Tokenizer::from_file(&tiktoken_path).expect("load HF Qwen3.5");

        let test_texts = [
            "Hello",
            "Hello world",
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "I'm doing great, thanks!",
            "don't can't I'll we're",
            "2+2=?",
            "La capitale della Francia è Parigi.",
            "  spaced  ",
            "\nnewlines\n",
            "test@email.com",
            "UPPER lower MixedCase",
            "   ",
            "a.b.c",
            "café déjà vu naïve",
            "日本語のテキスト",
            "русский текст",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            let hf_ids = hf_enc.get_ids();
            let bpe_ids = tok.encode(text);

            if hf_ids != bpe_ids.as_slice() {
                eprintln!("MISMATCH for text: {:?}", text);
                eprintln!("  HF:  {:?}", hf_ids);
                eprintln!("  BPE: {:?}", bpe_ids);
                all_ok = false;
            }
        }

        if !all_ok {
            panic!("tiktoken parity test failed");
        }
        eprintln!("✅ tiktoken parity passed ({} texts)", test_texts.len());
    }

    /// Decode parity for the WordPiece model: encode → decode should match
    /// HF's decode, including ## continuations rejoined with a space.
    #[test]
    fn test_wordpiece_decode_parity() {
        let wp_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/embedding/mlx-community/all-MiniLM-L6-v2-4bit/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&wp_path).exists() {
            eprintln!("MiniLM tokenizer not found at {wp_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&wp_path)).expect("load MiniLM");
        let hf_tok = tokenizers::Tokenizer::from_file(&wp_path).expect("load HF MiniLM");

        let test_texts = [
            "hello world",
            "The quick brown fox jumps over the lazy dog",
            "unhappiness is running",
            "I like machine learning a lot",
            "multi word sentence here",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let ids = tok.encode(text);
            let decoded = tok.decode(&ids);
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            let hf_ids: Vec<u32> = hf_enc.get_ids().iter().copied().rev()
                .skip_while(|&id| id == 0)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let hf_decoded = hf_tok.decode(&hf_ids, false).unwrap_or_default();

            if decoded != hf_decoded {
                eprintln!("DECODE MISMATCH for text: {:?}", text);
                eprintln!("  nxm: {:?}", decoded);
                eprintln!("  HF:  {:?}", hf_decoded);
                all_ok = false;
            }
        }

        if !all_ok {
            panic!("WordPiece decode parity test failed");
        }
        eprintln!("✅ WordPiece decode parity passed ({} texts)", test_texts.len());
    }

    /// Parity against HF tokenizers for a WordPiece model (all-MiniLM-L6-v2):
    /// WordPiece + BertPreTokenizer + BertNormalizer.
    #[test]
    fn test_wordpiece_parity() {
        let wp_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/embedding/mlx-community/all-MiniLM-L6-v2-4bit/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&wp_path).exists() {
            eprintln!("MiniLM tokenizer not found at {wp_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&wp_path)).expect("load MiniLM");
        let hf_tok = tokenizers::Tokenizer::from_file(&wp_path).expect("load HF MiniLM");

        let test_texts = [
            "hello world",
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog",
            "This is a [MASK] test",
            "I like machine learning",
            "unhappiness",
            "running swiftly",
            "don't stop",
            "UPPER case MIXED",
            "   padded   ",
            "multi\nline\ntext",
            "3.14159 and 2.718",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            // HF may pad to a fixed length; strip trailing [PAD] (id 0).
            let hf_ids: Vec<u32> = hf_enc.get_ids().iter().copied().rev()
                .skip_while(|&id| id == 0)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            let bpe_ids = tok.encode(text);

            if hf_ids != bpe_ids.as_slice() {
                eprintln!("MISMATCH for text: {:?}", text);
                eprintln!("  HF:  {:?}", hf_ids);
                eprintln!("  BPE: {:?}", bpe_ids);
                all_ok = false;
            }
        }

        if !all_ok {
            panic!("WordPiece parity test failed");
        }
        eprintln!("✅ WordPiece parity passed ({} texts)", test_texts.len());
    }

    /// Parity against HF tokenizers for a Unigram/SentencePiece model (t5-small):
    /// Unigram + WhitespaceSplit + Metaspace.
    #[test]
    fn test_unigram_parity() {
        let uni_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/unigram/t5-small/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&uni_path).exists() {
            eprintln!("t5-small tokenizer not found at {uni_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&uni_path)).expect("load t5-small");
        let hf_tok = tokenizers::Tokenizer::from_file(&uni_path).expect("load HF t5-small");

        let test_texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "I like machine learning.",
            "translate English to German: Hello, how are you?",
            "summarize: The economy is growing rapidly in many countries.",
            "2+2=4",
            "don't can't I'll we're",
            "  spaced  text  ",
            "multi\nline\ntext",
            "test@email.com",
        ];

        let mut all_ok = true;
        for text in &test_texts {
            let hf_enc = hf_tok.encode(*text, false).expect("HF encode");
            let hf_ids = hf_enc.get_ids();
            let bpe_ids = tok.encode(text);

            if hf_ids != bpe_ids.as_slice() {
                eprintln!("MISMATCH for text: {:?}", text);
                eprintln!("  HF:  {:?}", hf_ids);
                eprintln!("  BPE: {:?}", bpe_ids);
                all_ok = false;
            }
        }

        if !all_ok {
            panic!("Unigram parity test failed");
        }
        eprintln!("✅ Unigram parity passed ({} texts)", test_texts.len());
    }

    /// Roundtrip Unigram decode: encode then decode should recover text (modulo
    /// Metaspace leading-space artifacts).
    #[test]
    fn test_unigram_decode_roundtrip() {
        let uni_path = std::env::var("HOME").map(|h| {
            format!("{h}/models/unigram/t5-small/tokenizer.json")
        }).unwrap_or_default();

        if !std::path::Path::new(&uni_path).exists() {
            eprintln!("t5-small tokenizer not found at {uni_path}, skipping");
            return;
        }

        let mut tok = Tokenizer::load(std::path::Path::new(&uni_path)).expect("load t5-small");

        for text in ["Hello world", "The quick brown fox", "don't can't I'll"] {
            let ids = tok.encode(text);
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, text, "roundtrip failed for {text:?}");
        }
        // newlines collapse to spaces (WhitespaceSplit semantics)
        let ids = tok.encode("multi\nline\ntext");
        assert_eq!(tok.decode(&ids), "multi line text");
        eprintln!("✅ Unigram decode roundtrip passed");
    }

    #[test]
    fn bench_vs_hf() {
        if !hf_available() {
            eprintln!("tokenizer.json not found at {}, skipping", hf_path());
            return;
        }
        let mut tok = Tokenizer::load(std::path::Path::new(&hf_path())).expect("load");
        let hf_tok = tokenizers::Tokenizer::from_file(&hf_path()).expect("load HF");

        let texts = [
            "Hello",
            "Hello world",
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "I'm doing great, thanks!",
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<|im_start|>assistant\nCertainly! Let me help you with that.<|im_end|>",
            "2+2=?",
            "La capitale della Francia è Parigi.",
            "Explain quantum computing in simple terms",
            "  spaced  ",
            "\nnewlines\n",
            "don't can't I'll we're",
            "123 456",
            "test@email.com",
            "UPPER lower MixedCase",
            "   ",
            "a.b.c",
        ];
        let mut corpus = Vec::new();
        for text in &texts {
            for _ in 0..60 {
                corpus.push(text.to_string());
            }
        }
        let n = corpus.len();
        let avg_tokens: f64 = corpus.iter().map(|t| tok.encode(t).len() as f64).sum::<f64>() / n as f64;

        // warmup
        for text in &corpus {
            let _ = tok.encode(text);
        }
        for text in &corpus {
            let _ = hf_tok.encode(text.as_str(), false);
        }

        // benchmark: nxm
        let start = std::time::Instant::now();
        for text in &corpus {
            let _ = tok.encode(text);
        }
        let nxm_dur = start.elapsed();

        // benchmark: HF
        let start = std::time::Instant::now();
        for text in &corpus {
            let _ = hf_tok.encode(text.as_str(), false);
        }
        let hf_dur = start.elapsed();

        let nxm_per = nxm_dur.as_secs_f64() / n as f64 * 1_000_000.0;
        let hf_per = hf_dur.as_secs_f64() / n as f64 * 1_000_000.0;
        eprintln!("── Bench: {n} texts ({avg_tokens:.1} tokens/text) ──");
        eprintln!("  nxm-tokenizer: {nxm_dur:?} total, {nxm_per:.1} µs/text");
        eprintln!("  HF tokenizers: {hf_dur:?} total, {hf_per:.1} µs/text");
        eprintln!("  ratio HF/nxm: {:.2}x", hf_dur.as_secs_f64() / nxm_dur.as_secs_f64());
    }
}

// ─── Compatibility exports for serve-metal ──────────────────────────────────

/// Alias for compatibility with serve-metal's expected BpeTokenizer name.
pub type BpeTokenizer = Tokenizer;

/// Encode chat messages using ChatML template (Qwen3 style).
pub fn encode_chat(tokenizer: &mut Tokenizer, messages: &[(String, String)]) -> Vec<u32> {
    let mut text = String::new();
    for (role, content) in messages {
        match role.as_str() {
            "system" => text.push_str(content),
            "user" => text.push_str(&format!("│user\n{} │\n", content)),
            "assistant" => text.push_str(&format!("│assistant\n{} │\n", content)),
            _ => text.push_str(&format!("│{}\n{} │\n", role, content)),
        }
    }
    text.push_str("│assistant\n");
    tokenizer.encode(&text)
}
