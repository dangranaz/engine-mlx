use nxm_tokenizer::chat_template::{ChatMessage, ChatTemplate};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn read_ids(path: &Path) -> Vec<u32> {
    let content = std::fs::read_to_string(path).expect("read ids fixture");
    content
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .flat_map(|l| l.split(','))
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().expect("parse id"))
        .collect()
}

fn fixture_message() -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: "Name three capital cities of European countries.".to_string(),
    }
}

// F2 / R1 (CI-runnable): il vero template di K2 committato (~50 KB, con {% generation %})
// compila e renderizza (non vuoto). L'oracolo encode è il test #[ignore] sotto.
#[test]
fn chat_template_render_real_k2_template_compiles() {
    let dir = fixtures_dir().join("k2_template");
    let tpl = ChatTemplate::from_model_dir(&dir).expect("load real K2 template");
    let prompt = tpl.render(&[fixture_message()], true).expect("render");
    assert!(!prompt.is_empty(), "real K2 template rendered empty");
}

// F1: lo strip dei marcatori {% generation %} / {% endgeneration %} preserva il corpo
// e compila; add_generation_prompt (variabile) non viene toccato. CI-runnable, senza modello.
#[test]
fn chat_template_strip_generation_markers() {
    let dir = fixtures_dir().join("generation_markers");
    let tpl = ChatTemplate::from_model_dir(&dir).expect("load template with generation markers");
    let with = tpl
        .render(&[fixture_message()], true)
        .expect("render template with generation markers");
    let without = tpl
        .render(&[fixture_message()], false)
        .expect("render template with generation markers");
    // Il corpo del blocco generation ("X") deve essere reso; i marcatori no.
    assert!(
        with.contains("X"),
        "generation marker body missing, got: {with:?}"
    );
    assert!(
        !with.contains("{% generation %}") && !with.contains("endgeneration"),
        "generation markers leaked into render, got: {with:?}"
    );
    // add_generation_prompt=true vs false differiscono (la variabile non è toccata dallo strip).
    assert_ne!(with, without);
    assert!(
        with.contains("assistant"),
        "add_generation_prompt=true should include assistant opener, got: {with:?}"
    );
}

// R1 (hardware): render del vero template K2 + encode riproduce il fixture di ids.
// Richiede ENGINE_MLX_MODEL che punti alla K2-Horizon-3.7B-4bit (4 GB, non in repo).
#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer"]
fn chat_template_render_matches_k2_fixture() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let model_dir = Path::new(&model_dir_str);
    let tpl = ChatTemplate::from_model_dir(model_dir).expect("load K2 template");
    let prompt = tpl.render(&[fixture_message()], true).expect("render");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let mut tok = nxm_tokenizer::pure::Tokenizer::load(&tokenizer_path).expect("load tokenizer");
    let ids = tok.encode(&prompt);
    let expected = read_ids(&fixtures_dir().join("k2_horizon_prompt_ids.txt"));
    assert_eq!(
        ids, expected,
        "render+encode did not reproduce k2_horizon_prompt_ids.txt"
    );
}

// F4 (hardware): add_generation_prompt governa la coda sul template reale K2.
#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer"]
fn chat_template_real_add_generation_prompt_toggles() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let model_dir = Path::new(&model_dir_str);
    let tpl = ChatTemplate::from_model_dir(model_dir).expect("load K2 template");
    let with = tpl.render(&[fixture_message()], true).expect("render true");
    let without = tpl
        .render(&[fixture_message()], false)
        .expect("render false");
    assert_ne!(with, without);
    assert!(
        with.contains("<|ifm|im_start|>assistant"),
        "with add_generation_prompt should open assistant turn, got: {with:?}"
    );
    assert!(
        !without.contains("<|ifm|im_start|>assistant"),
        "without add_generation_prompt should not open assistant turn, got: {without:?}"
    );
}

// F4 (hardware): il template reale non esplode sul percorso normale (no tool_* definiti).
#[test]
#[ignore = "requires ENGINE_MLX_MODEL pointing at the real K2-Horizon model dir + tokenizer"]
fn chat_template_real_normal_path_no_panic() {
    let model_dir_str = std::env::var("ENGINE_MLX_MODEL")
        .expect("ENGINE_MLX_MODEL must point at K2-Horizon model dir");
    let model_dir = Path::new(&model_dir_str);
    let tpl = ChatTemplate::from_model_dir(model_dir).expect("load K2 template");
    let prompt = tpl.render(&[fixture_message()], true).expect("render");
    assert!(
        !prompt.is_empty(),
        "real K2 template rendered empty on normal path"
    );
}

// R2 (CI-runnable): raise_exception sul percorso preso produce Err contenente il messaggio.
#[test]
fn chat_template_raise_exception_bubbles() {
    let dir = fixtures_dir().join("raise_exception");
    let tpl = ChatTemplate::from_model_dir(&dir).expect("load raise_exception template");
    let err = match tpl.render(&[fixture_message()], true) {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(
        err.chain().any(|e| e.to_string().contains("boom")),
        "expected raise_exception message in error chain"
    );
}

// R3 (CI-runnable, contro il vero template K2 committato): add_generation_prompt governa la coda.
#[test]
fn chat_template_add_generation_prompt_toggles() {
    let dir = fixtures_dir().join("k2_template");
    let tpl = ChatTemplate::from_model_dir(&dir).expect("load committed K2 template");
    let with = tpl.render(&[fixture_message()], true).expect("render true");
    let without = tpl
        .render(&[fixture_message()], false)
        .expect("render false");
    assert_ne!(with, without);
    assert!(
        with.contains("<|ifm|im_start|>assistant"),
        "with add_generation_prompt should open assistant turn, got: {with:?}"
    );
    assert!(
        !without.contains("<|ifm|im_start|>assistant"),
        "without add_generation_prompt should not open assistant turn, got: {without:?}"
    );
}

// R4: risoluzione della fonte del template.
#[test]
fn chat_template_source_resolution() {
    // Solo .jinja -> Ok
    let jinja_only = fixtures_dir().join("source_jinja_only");
    ChatTemplate::from_model_dir(&jinja_only).expect("jinja-only dir should load");
    // Solo chiave chat_template in tokenizer_config.json -> Ok
    let config_only = fixtures_dir().join("source_config_only");
    ChatTemplate::from_model_dir(&config_only).expect("config-key-only dir should load");
    // Nessuna delle due -> Err chiaro
    let none = fixtures_dir().join("source_none");
    let err = match ChatTemplate::from_model_dir(&none) {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(
        err.to_string().contains("chat template"),
        "expected chat-template error, got: {err}"
    );
}

// F3: bos_token accetta stringa e oggetto {content}; assente -> "".
#[test]
fn bos_token_accepts_string_and_object() {
    // Stringa
    let str_dir = fixtures_dir().join("bos_token_string");
    let tpl = ChatTemplate::from_model_dir(&str_dir).expect("load bos_token string config");
    let prompt = tpl
        .render(&[fixture_message()], false)
        .expect("render bos_token string");
    assert!(
        prompt.starts_with("<|ifm|begin_of_text|>"),
        "bos_token string not rendered, got: {prompt:?}"
    );
    // Oggetto {content}
    let obj_dir = fixtures_dir().join("bos_token_object");
    let tpl = ChatTemplate::from_model_dir(&obj_dir).expect("load bos_token object config");
    let prompt = tpl
        .render(&[fixture_message()], false)
        .expect("render bos_token object");
    assert!(
        prompt.starts_with("<|ifm|begin_of_text|>"),
        "bos_token object not rendered, got: {prompt:?}"
    );
    // Assente -> ""
    let absent_dir = fixtures_dir().join("bos_token_absent");
    let tpl =
        ChatTemplate::from_model_dir(&absent_dir).expect("load jinja-only dir without bos_token");
    let prompt = tpl
        .render(&[fixture_message()], false)
        .expect("render no bos_token");
    assert!(
        !prompt.starts_with("<|ifm|begin_of_text|>"),
        "absent bos_token should not inject a token, got: {prompt:?}"
    );
}
