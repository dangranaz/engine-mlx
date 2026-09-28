# AGENT_TESTING.md — engine-bench

Machine-actionable testing guide for **any AI agent** (Kiro CLI, Claude, etc.)
to benchmark an OpenAI-compatible LLM server and **interpret the results**.

> Bilingual: each section is **EN** then **IT**.
> Bilingue: ogni sezione è prima in **EN**, poi in **IT**.

---

## 0. TL;DR

**EN** — Four commands, run from `engine-bench/`:

```bash
# 1. build the harness
cargo build --release

# 2. start engine-mlx serving a model on nexum's standard port 11435
MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx \
  ./models/start-qwen3-0.6b.sh

# 3. run everything (Hurl gates + nxm-bench) and archive reports
./run.sh

# 4. stop
./models/stop-qwen3-0.6b.sh
```

Exit code of `run.sh`: **0** = all passed, **1** = at least one failure,
**2** = tooling error. Reports land in `results/<engine>/`.

> **MANDATORY:** always stop the server when the tests are done (step 4),
> **even if the run failed**. A left-over server holds the model in GPU memory
> and blocks port 11435 for the next run.

**IT** — Quattro comandi, da `engine-bench/`:

```bash
cargo build --release
MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx \
  ./models/start-qwen3-0.6b.sh
./run.sh
./models/stop-qwen3-0.6b.sh
```

Exit code di `run.sh`: **0** = tutto ok, **1** = almeno un fallimento,
**2** = errore di tooling. I report finiscono in `results/<engine>/`.

> **OBBLIGATORIO:** ferma sempre il server a test finiti (passo 4), **anche se
> il run è fallito**. Un server lasciato attivo tiene il modello in memoria GPU
> e occupa la porta 11435 per il run successivo.

---

## 0b. Invocation for agents / Invocazione per agenti

**EN** — Give an AI agent (Kiro CLI, Claude, etc.) one of these prompts. It tells
the agent to follow this document end-to-end for a specific model + size profile.

Pick the profile by size: **tiny** ≤1B, **small** 1–4B, **large** ≥7B.

Kiro CLI — Qwen3-1.7B (profile `small`):

```bash
kiro-cli "Read ~/Projects/nexum/engine-bench/AGENT_TESTING.md and follow it to \
test model Qwen3-1.7B. From engine-bench/: (1) ./models/start-qwen3-1.7b.sh \
(2) ./run.sh (3) ./check_expected.sh \"\$(ls -t results/engine-mlx/*.json | head -1)\" small \
(4) ./models/stop-qwen3-1.7b.sh. Then report the verdict and interpret results \
with the doc's engine-vs-model golden rule (section 6) and per-size profile \
(section 7b). Profile is 'small' because Qwen3-1.7B is ~1.7B. Always run step 4 \
(stop the server) at the end, even if the tests failed."
```

Kiro CLI — Qwen3-0.6B (profile `tiny`):

```bash
kiro-cli "Read ~/Projects/nexum/engine-bench/AGENT_TESTING.md and follow it to \
test model Qwen3-0.6B. From engine-bench/: (1) ./models/start-qwen3-0.6b.sh \
(2) ./run.sh (3) ./check_expected.sh \"\$(ls -t results/engine-mlx/*.json | head -1)\" tiny \
(4) ./models/stop-qwen3-0.6b.sh. Report the verdict using the doc's golden rule \
(section 6) and per-size profile (section 7b). Profile is 'tiny' because it is <=1B. \
Always run step 4 (stop the server) at the end, even if the tests failed."
```

Generic prompt (any agent) — substitute `<MODEL>` and `<PROFILE>`:

```
Read ~/Projects/nexum/engine-bench/AGENT_TESTING.md and follow it to test <MODEL>.
From engine-bench/:
  1. ./models/start-<MODEL>.sh          (or MODEL=<id> ./start.sh)
  2. ./run.sh
  3. ./check_expected.sh "$(ls -t results/engine-mlx/*.json | head -1)" <PROFILE>
  4. ./models/stop-<MODEL>.sh
Report the verdict. Interpret with the engine-vs-model golden rule (section 6):
degradation/API/latency FAIL = engine bug; correctness/coherence FAIL on a small
model = expected (advisory). <PROFILE> = tiny (<=1B) | small (1-4B) | large (>=7B).
Always run step 4 (stop the server) at the end, even if the tests failed.
```

**IT** — Passa a un agente AI (Kiro CLI, Claude, ecc.) uno di questi prompt. Dice
all'agente di seguire questo documento dall'inizio alla fine per un modello +
profilo di taglia specifici. Scegli il profilo per taglia: **tiny** ≤1B,
**small** 1–4B, **large** ≥7B. I prompt sopra funzionano in italiano o inglese
(il documento è bilingue). Regola del gate: **run.sh exit 0 E check_expected.sh
exit 0** per il profilo giusto = tutto ok.

---

## 1. Prerequisites / Prerequisiti

**EN**
- macOS on Apple Silicon (for engine-mlx). `cargo`, `curl`, `jq`.
- `hurl` (`brew install hurl`) — optional; if missing, `run.sh` skips the
  declarative layer and still runs nxm-bench.
- MLX build env (engine-mlx only): `MLX_C_PATH=/opt/homebrew/opt/mlx-c` and
  `MLX_PREFIX=/opt/homebrew/opt/mlx`. **Critical:** if `MLX_C_PATH` points at a
  source dir without `lib/`, linking fails (`-lmlxc` missing).
- Standard server port is **11435** (11434 is Ollama, reserved for the TUI).
- The model must exist under `$NEXUM_MODELS_DIR` (default `~/nexum/models`).

**IT**
- macOS su Apple Silicon (per engine-mlx). `cargo`, `curl`, `jq`.
- `hurl` (`brew install hurl`) — opzionale; se assente, `run.sh` salta il layer
  dichiarativo ed esegue comunque nxm-bench.
- Env di build MLX (solo engine-mlx): `MLX_C_PATH=/opt/homebrew/opt/mlx-c` e
  `MLX_PREFIX=/opt/homebrew/opt/mlx`. **Critico:** se `MLX_C_PATH` punta a una
  cartella sorgente senza `lib/`, il link fallisce (`-lmlxc` mancante).
- La porta standard è **11435** (11434 è Ollama, riservata alla TUI).
- Il modello deve esistere in `$NEXUM_MODELS_DIR` (default `~/nexum/models`).

---

## 2. Step-by-step procedure / Procedura passo-passo

**EN**

1. **Build:** `cargo build --release` (from `engine-bench/`). On failure → exit,
   report the compiler error; do not proceed.
2. **Start the server** with a per-model script (each bakes in the model name):
   - `./models/start-qwen3-0.6b.sh` → Qwen3-0.6B-MLX-4bit
   - `./models/start-qwen3-1.7b.sh` → Qwen3-1.7B-MLX-8bit
   Wait for the readiness line `engine '<engine>' up at http://127.0.0.1:11435`.
   If it does not appear within `READY_TIMEOUT` (default 300s), read the log
   printed by the script and stop.
3. **Confirm the model id:** `curl -s http://127.0.0.1:11435/v1/models | jq -r '.data[0].id'`.
4. **Run:** `./run.sh`. It resolves the model automatically, runs Hurl then
   nxm-bench, writes reports, and prints a Summary with per-stage PASS/FAIL.
5. **Read the exit code** and the reports (§4).
6. **Stop:** `./models/stop-qwen3-0.6b.sh` (always, even on failure).

**IT**

1. **Build:** `cargo build --release` (da `engine-bench/`). Se fallisce → esci,
   riporta l'errore del compilatore; non proseguire.
2. **Avvia il server** con uno script per-modello (ognuno ha il nome incorporato):
   - `./models/start-qwen3-0.6b.sh` → Qwen3-0.6B-MLX-4bit
   - `./models/start-qwen3-1.7b.sh` → Qwen3-1.7B-MLX-8bit
   Attendi la riga `engine '<engine>' up at http://127.0.0.1:11435`. Se non
   appare entro `READY_TIMEOUT` (default 300s), leggi il log stampato dallo
   script e fermati.
3. **Conferma il model id:** `curl -s http://127.0.0.1:11435/v1/models | jq -r '.data[0].id'`.
4. **Esegui:** `./run.sh`. Risolve il modello da solo, lancia Hurl poi nxm-bench,
   scrive i report e stampa un Summary con PASS/FAIL per stage.
5. **Leggi l'exit code** e i report (§4).
6. **Ferma:** `./models/stop-qwen3-0.6b.sh` (sempre, anche in caso di fallimento).

---

## 3. What gets tested / Cosa viene testato

**EN** — Two layers:

- **Hurl** (`hurl/`, declarative HTTP): `api/chat_completions` (OpenAI shape),
  `api/error_handling` (rejects malformed input: 422/400/415),
  `coherence/basic` (semantic sanity — **blocking**), `perf/decode_latency`
  (< 5 s gate).
- **nxm-bench** (Rust, 5 tests): 2 correctness (factual, instruction), 2
  degradation (sustained throughput, KV length-ramp), 1 coherence (multi-step
  reasoning). Writes timestamped JSON+MD.

**IT** — Due layer:

- **Hurl** (`hurl/`, HTTP dichiarativo): `api/chat_completions` (formato OpenAI),
  `api/error_handling` (rifiuta input malformati: 422/400/415),
  `coherence/basic` (sanità semantica — **bloccante**), `perf/decode_latency`
  (gate < 5 s).
- **nxm-bench** (Rust, 5 test): 2 correttezza (factual, instruction), 2
  degradazione (throughput sostenuto, ramp di lunghezza KV), 1 coerenza
  (ragionamento multi-step). Scrive JSON+MD con timestamp.

---

## 4. Where the results are / Dove sono i risultati

**EN**

```
results/<engine>/
  <YYYY-MM-DD_HH-MM-SS>.json   # nxm-bench full report (machine-readable)
  <YYYY-MM-DD_HH-MM-SS>.md     # nxm-bench human summary
  hurl_<YYYY-MM-DD_HH-MM-SS>/  # hurl HTML + JSON reports
```

Inspect the latest nxm-bench result programmatically:

```bash
LATEST=$(ls -t results/engine-mlx/*.json | head -1)
jq '{model: .environment.model, commit: .environment.engine_commit,
     overall: .passed,
     tests: [.tests[] | {id, name, passed, metrics}]}' "$LATEST"
```

Key JSON fields: `environment` (engine, url, model, timestamp, hostname,
engine_commit, bench_version), `thresholds` (when `--assert`), `passed`
(overall bool or null), `tests[]` (each with `id`, `category`, `passed`,
`metrics`, `samples`, `outputs`).

**IT**

```
results/<engine>/
  <YYYY-MM-DD_HH-MM-SS>.json   # report completo nxm-bench (machine-readable)
  <YYYY-MM-DD_HH-MM-SS>.md     # riassunto leggibile nxm-bench
  hurl_<YYYY-MM-DD_HH-MM-SS>/  # report hurl HTML + JSON
```

Ispeziona l'ultimo risultato nxm-bench in modo programmatico con lo stesso
comando `jq` sopra. Campi JSON chiave: `environment`, `thresholds`, `passed`,
`tests[]` (con `id`, `category`, `passed`, `metrics`, `samples`, `outputs`).

---

## 5. Result analysis — triage table / Analisi dei risultati — tabella di triage

**EN** — For each test: what it measures, the pass condition, what a FAIL means,
and the action an agent should take.

| Test (id) | Layer | Measures | FAIL means | Agent action |
|-----------|-------|----------|-----------|--------------|
| `api/chat_completions` | Hurl | OpenAI response shape (id, object, choices, usage) | **Engine/API bug** — server not OpenAI-compatible | Treat as a real defect. Inspect server response; do not blame the model. |
| `api/error_handling` | Hurl | Rejects malformed input (422/400/415) | **Engine bug** — bad input not rejected, or wrong code | Real defect. Check the HTTP handler. |
| `perf/decode_latency` | Hurl | Short reply < 5 s | **Engine slow/stuck** — or machine under load | Check load; if idle-slow, real perf regression. |
| `coherence/basic` (Hurl) | Hurl | France→Paris, 2+2→4, Italy→Rome | **Model capability** (small model) OR corrupted output | See §6 golden rule. If output is garbage/repetition → engine bug. If a plausible wrong answer → model too small. |
| `factual` (nxm-bench) | correctness | Answer contains "Paris" | **Model capability** | Not an engine bug unless output is garbage. |
| `instruction` (nxm-bench) | correctness | Replies concisely "DONE" | **Model capability** (instruction following) | Small models often fail. Not an engine bug. |
| `sustained` (nxm-bench) | **degradation** | 20 identical reqs, first→last drop ≤ 15% AND min t/s ≥ 20 | **ENGINE BUG** — throughput degrades over requests (e.g. a per-request memory leak) | **High priority.** Real regression. Check per-step allocation / KV buffer freeing. |
| `length_ramp` (nxm-bench) | **degradation** | t/s across 128/512/1024, drop ≤ 60%, min t/s ≥ 20 | **Engine KV-scaling issue** — attention/cache scales badly | Real perf concern; investigate KV cache strategy. |
| `coherence` (nxm-bench) | coherence | Word problem → answer 18 with working | **Model capability** (reasoning) | Small models fail. Not an engine bug. |

**IT** — Per ogni test: cosa misura, condizione di pass, cosa significa un FAIL,
e l'azione che l'agente dovrebbe intraprendere.

| Test (id) | Layer | Misura | Un FAIL significa | Azione dell'agente |
|-----------|-------|--------|-------------------|--------------------|
| `api/chat_completions` | Hurl | Formato risposta OpenAI (id, object, choices, usage) | **Bug engine/API** — server non OpenAI-compatibile | Difetto reale. Ispeziona la risposta; non incolpare il modello. |
| `api/error_handling` | Hurl | Rifiuta input malformati (422/400/415) | **Bug engine** — input errato non rifiutato o codice sbagliato | Difetto reale. Controlla l'handler HTTP. |
| `perf/decode_latency` | Hurl | Risposta breve < 5 s | **Engine lento/bloccato** — o macchina sotto carico | Verifica il carico; se lento a vuoto, regressione reale. |
| `coherence/basic` (Hurl) | Hurl | Francia→Parigi, 2+2→4, Italia→Roma | **Capacità del modello** (piccolo) OPPURE output corrotto | Vedi regola d'oro §6. Se output spazzatura/ripetizione → bug engine. Se risposta plausibile ma sbagliata → modello troppo piccolo. |
| `factual` (nxm-bench) | correttezza | La risposta contiene "Paris" | **Capacità del modello** | Non è un bug engine, salvo output spazzatura. |
| `instruction` (nxm-bench) | correttezza | Risponde in modo conciso "DONE" | **Capacità del modello** (seguire istruzioni) | I modelli piccoli spesso falliscono. Non è un bug engine. |
| `sustained` (nxm-bench) | **degradazione** | 20 req identiche, calo first→last ≤ 15% E min t/s ≥ 20 | **BUG ENGINE** — throughput degrada nel tempo (es. memory leak per-richiesta) | **Priorità alta.** Regressione reale. Controlla allocazioni per-step / free dei buffer KV. |
| `length_ramp` (nxm-bench) | **degradazione** | t/s su 128/512/1024, calo ≤ 60%, min t/s ≥ 20 | **Problema di scaling KV** — attention/cache scala male | Problema di performance reale; indaga la strategia KV cache. |
| `coherence` (nxm-bench) | coerenza | Problema a step → risposta 18 con ragionamento | **Capacità del modello** (ragionamento) | I modelli piccoli falliscono. Non è un bug engine. |

Thresholds live in `thresholds.toml` (override via `--min-tps` /
`--max-degradation-pct`). / Le soglie sono in `thresholds.toml`.

---

## 6. Golden rule: engine health vs model capability / Regola d'oro: salute engine vs capacità modello

**EN** — This is the most important interpretation rule for an agent:

- **Degradation + API + latency tests measure the ENGINE.** A FAIL here is a
  real bug/regression → investigate and fix the engine.
- **Correctness + coherence tests measure the MODEL.** A FAIL here on a small
  model is *expected*, not an engine bug — **unless the output is garbage**
  (empty, repeated tokens, `?????`, or unrelated language). Garbage output IS an
  engine bug (e.g. a broken attention mask or KV cache), even though it surfaces
  in a "correctness" test.
- **Decision procedure:** read the captured `outputs` in the JSON. Coherent but
  wrong → model. Incoherent/garbage/looping → engine.

**IT** — È la regola d'interpretazione più importante per un agente:

- **Degradazione + API + latenza misurano l'ENGINE.** Un FAIL qui è un
  bug/regressione reale → indaga e correggi l'engine.
- **Correttezza + coerenza misurano il MODELLO.** Un FAIL qui su un modello
  piccolo è *atteso*, non un bug engine — **a meno che l'output sia spazzatura**
  (vuoto, token ripetuti, `?????`, o lingua non pertinente). L'output spazzatura
  È un bug engine (es. mask di attention rotta o KV cache), anche se emerge in un
  test di "correttezza".
- **Procedura decisionale:** leggi gli `outputs` catturati nel JSON. Coerente ma
  sbagliato → modello. Incoerente/spazzatura/loop → engine.

---

## 7. Expected PASS profile by model size / Profilo di PASS atteso per taglia del modello

**EN** — Use this to avoid false alarms. What SHOULD pass depends on model size:

| Model size | degradation (sustained + ramp) | API + latency (Hurl) | correctness + coherence | Overall verdict expectation |
|------------|-------------------------------|----------------------|-------------------------|-----------------------------|
| **~0.6B** (e.g. Qwen3-0.6B) | **MUST pass** | **MUST pass** | *May fail* (too small to be reliable) | FAIL overall is NORMAL — check ONLY that degradation+API passed. |
| **~1.7B–3B** | **MUST pass** | **MUST pass** | factual/instruction often pass; multi-step reasoning may fail | Partial pass expected. |
| **~7B+** | **MUST pass** | **MUST pass** | **SHOULD pass all** | Overall PASS expected; any FAIL is worth investigating. |

**Agent rule:** For a small model, do **not** report the run as "broken" just
because correctness/coherence failed. Report **broken** only if a
**degradation, API, latency** test failed, or if outputs are garbage. The
0.6B baseline (`results/engine-mlx/`) is expected to show overall FAIL with
degradation PASS — that is the correct healthy-engine + weak-model signature.

**IT** — Usa questa tabella per evitare falsi allarmi. Cosa DEVE passare dipende
dalla taglia del modello:

| Taglia modello | degradazione (sustained + ramp) | API + latenza (Hurl) | correttezza + coerenza | Attesa verdetto complessivo |
|----------------|--------------------------------|----------------------|------------------------|-----------------------------|
| **~0.6B** (es. Qwen3-0.6B) | **DEVE passare** | **DEVE passare** | *Può fallire* (troppo piccolo per essere affidabile) | Un FAIL complessivo è NORMALE — verifica SOLO che degradazione+API siano passate. |
| **~1.7B–3B** | **DEVE passare** | **DEVE passare** | factual/instruction spesso passano; il reasoning multi-step può fallire | Pass parziale atteso. |
| **~7B+** | **DEVE passare** | **DEVE passare** | **DOVREBBE passare tutto** | PASS complessivo atteso; ogni FAIL va indagato. |

**Regola per l'agente:** per un modello piccolo, **non** segnalare il run come
"rotto" solo perché correttezza/coerenza falliscono. Segnala **rotto** solo se
fallisce un test di **degradazione, API, latenza**, o se gli output sono
spazzatura. Il baseline 0.6B (`results/engine-mlx/`) è atteso con FAIL
complessivo ma degradazione PASS — è la firma corretta di engine sano + modello
debole.

---

## 7b. Recommended thresholds + automated verdict / Soglie raccomandate + verdetto automatico

**EN** — Machine-checkable profiles live in `expected/<profile>.json`
(`tiny` ≤1B, `small` 1–4B, `large` ≥7B). Each declares `must_pass` (required
tests), `advisory` (allowed to fail), and a `min_tps` throughput floor:

| Profile | Model size | `min_tps` floor | must_pass |
|---------|-----------|-----------------|-----------|
| `tiny`  | ≤1B  | **40** | engine-health only (api, error, latency, sustained, length_ramp) |
| `small` | 1–4B | **25** | engine-health + `factual` |
| `large` | ≥7B  | **15** | everything (health + correctness + coherence) |

`min_tps` *decreases* with size because larger models are inherently slower per
token — the floor is a sanity check, not a target.

Run the automated verdict (deterministic, exit 0 healthy / 1 unhealthy):

```bash
# latest result vs a profile matching the model size
./check_expected.sh results/engine-mlx/<timestamp>.json tiny

# defaults: newest result, profile "tiny"
./check_expected.sh
```

It fails the verdict only when a `must_pass` nxm-bench test fails or throughput
is below the floor; advisory (model-capability) failures are reported but do not
fail. Hurl `must_pass` entries are flagged as "verify via `run.sh` exit code"
(they aren't in the nxm-bench JSON). **Full gate = `run.sh` exit 0 AND
`check_expected.sh` exit 0** for the appropriate profile.

**IT** — Profili verificabili a macchina in `expected/<profile>.json`
(`tiny` ≤1B, `small` 1–4B, `large` ≥7B). Ognuno dichiara `must_pass` (test
obbligatori), `advisory` (può fallire) e un `min_tps` (soglia di throughput):

| Profilo | Taglia | `min_tps` | must_pass |
|---------|--------|-----------|-----------|
| `tiny`  | ≤1B  | **40** | solo salute engine (api, error, latency, sustained, length_ramp) |
| `small` | 1–4B | **25** | salute engine + `factual` |
| `large` | ≥7B  | **15** | tutto (salute + correttezza + coerenza) |

`min_tps` *diminuisce* con la taglia perché i modelli grandi sono per natura più
lenti per token — la soglia è un controllo di sanità, non un obiettivo.

Esegui il verdetto automatico (deterministico, exit 0 sano / 1 non sano):

```bash
./check_expected.sh results/engine-mlx/<timestamp>.json tiny
./check_expected.sh          # default: ultimo risultato, profilo "tiny"
```

Fallisce il verdetto solo se un test `must_pass` di nxm-bench fallisce o il
throughput è sotto la soglia; i fallimenti advisory (capacità del modello) sono
riportati ma non fanno fallire. Le voci `must_pass` di Hurl sono marcate come
"verifica via exit code di `run.sh`" (non sono nel JSON di nxm-bench). **Gate
completo = `run.sh` exit 0 E `check_expected.sh` exit 0** per il profilo giusto.

---

## 8. Historical comparison / Confronto storico

**EN** — Detect regressions across runs (or engines):

```bash
# newest run of each engine, side by side
./target/release/nxm-bench compare --latest engine-mlx,engine-metal

# two explicit result files (baseline first)
./target/release/nxm-bench compare results/engine-mlx/A.json results/engine-mlx/B.json
```

Read the Δ arrows: ▲ better than baseline, ▼ worse, = unchanged (higher t/s is
better; lower drop% is better). A ▼ on `sustained min t/s` or a rising
`degradation %` across runs is a **regression** → investigate.

**IT** — Rileva regressioni tra run (o tra engine): stessi comandi sopra. Leggi
le frecce Δ: ▲ meglio del baseline, ▼ peggio, = invariato (t/s più alto è
meglio; drop% più basso è meglio). Un ▼ su `sustained min t/s` o un
`degradation %` crescente tra i run è una **regressione** → indaga.

---

## 9. Troubleshooting

**EN**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Build fails, `-lmlxc` / undefined `_mlx_*` | `MLX_C_PATH` points at source dir without `lib/` | `export MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx` |
| Server never becomes ready | model not found / wrong path | `scripts/server.sh models list` (engine-mlx); check `$NEXUM_MODELS_DIR` |
| Port already in use | old server or Ollama on 11434 | stop it, or set `PORT=`; our engine is 11435, Ollama is 11434 |
| Throughput fast then drops over requests | per-request leak in decode (KV buffers not freed) | real engine bug — free superseded `mlx_array` handles each step |
| Coherence FAIL but output looks garbage/looping | broken attention mask / KV cache | real engine bug — check SDPA mask dtype/mode and static KV |
| `hurl: command not found` | hurl not installed | `brew install hurl` (or accept nxm-bench-only run) |

**IT**

| Sintomo | Causa | Rimedio |
|---------|-------|---------|
| Build fallisce, `-lmlxc` / `_mlx_*` non definiti | `MLX_C_PATH` punta a cartella sorgente senza `lib/` | `export MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx` |
| Il server non diventa mai ready | modello non trovato / path errato | `scripts/server.sh models list` (engine-mlx); controlla `$NEXUM_MODELS_DIR` |
| Porta già in uso | vecchio server o Ollama su 11434 | fermalo, o imposta `PORT=`; il nostro engine è 11435, Ollama è 11434 |
| Throughput veloce poi cala tra le richieste | leak per-richiesta nel decode (buffer KV non liberati) | bug engine reale — libera gli `mlx_array` superati a ogni step |
| Coherence FAIL con output spazzatura/loop | mask di attention rotta / KV cache | bug engine reale — controlla dtype/mode della mask SDPA e la KV statica |
| `hurl: command not found` | hurl non installato | `brew install hurl` (oppure accetta il run solo-nxm-bench) |

---

## 10. Quick reference / Riferimento rapido

```bash
# per-model start/stop (port 11435 baked in)
./models/start-qwen3-0.6b.sh   ./models/stop-qwen3-0.6b.sh
./models/start-qwen3-1.7b.sh   ./models/stop-qwen3-1.7b.sh

# full run (Hurl + nxm-bench), archives reports, exit 0/1/2
./run.sh

# nxm-bench only
./target/release/nxm-bench run --url http://127.0.0.1:11435 --engine engine-mlx --assert

# compare runs / engines
./target/release/nxm-bench compare --latest engine-mlx,engine-metal

# automated verdict against a size profile (tiny|small|large)
./check_expected.sh results/engine-mlx/<timestamp>.json tiny
```
