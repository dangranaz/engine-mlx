# DEMO — engine-bench su Qwen3-1.7B-MLX-4bit

Documento **dedicato alla demo**: esegue la suite di benchmark usando **solo** il
modello `Qwen3-1.7B-MLX-4bit`. È un duplicato semplificato di
[`AGENT_TESTING.md`](AGENT_TESTING.md), tarato su un unico modello e profilo.

> Bilingue: ogni sezione è prima in **IT**, poi in **EN**.
> Bilingual: each section is **IT** first, then **EN**.

Il modello 1.7B rientra nel profilo **`small`** (1–4B).

---

## 0. TL;DR

**IT** — Quattro comandi, da `engine-bench/`:

```bash
# 1. build della harness
cargo build --release

# 2. avvia engine-mlx con Qwen3-1.7B-MLX-4bit sulla porta standard 11435
MLX_C_PATH=/opt/homebrew/opt/mlx-c MLX_PREFIX=/opt/homebrew/opt/mlx \
  ./models/start-qwen3-1.7b-4bit.sh

# 3. esegui tutto (gate Hurl + nxm-bench) e archivia i report
./run.sh

# 4. verdetto automatico per il profilo small
./check_expected.sh "$(ls -t results/engine-mlx/*.json | head -1)" small

# 5. ferma il server (SEMPRE, anche se il run è fallito)
./models/stop-qwen3-1.7b-4bit.sh
```

Exit code di `run.sh`: **0** = tutto ok, **1** = almeno un fallimento,
**2** = errore di tooling. I report finiscono in `results/engine-mlx/`.

> **OBBLIGATORIO:** ferma sempre il server a test finiti (passo 5). Un server
> lasciato attivo tiene il modello in memoria GPU e occupa la porta 11435.

**EN** — Four commands from `engine-bench/`: build, start the 4bit model, run,
verify against the `small` profile, stop. `run.sh` exit code: **0** all passed,
**1** a failure, **2** tooling error. Always stop the server at the end.

---

## 0b. Invocazione per agenti / Invocation for agents

**IT** — Passa a un agente AI (Kiro CLI, Claude, ecc.) questo prompt. Dice
all'agente di seguire questo documento dall'inizio alla fine, per il **solo**
modello Qwen3-1.7B-4bit.

```bash
kiro-cli "Leggi ~/Projects/nexum/engine-bench/DEMO_QWEN3_1.7B_4BIT.md e seguilo per \
testare SOLO il modello Qwen3-1.7B-MLX-4bit. Da engine-bench/: \
(1) ./models/start-qwen3-1.7b-4bit.sh (2) ./run.sh \
(3) ./check_expected.sh \"\$(ls -t results/engine-mlx/*.json | head -1)\" small \
(4) ./models/stop-qwen3-1.7b-4bit.sh. Poi riporta il verdetto e interpreta i \
risultati con la regola d'oro engine-vs-modello (AGENT_TESTING.md §6) e il \
profilo 'small' (1-4B). Esegui SEMPRE il passo 4 (stop del server) alla fine, \
anche se i test falliscono."
```

**EN** — Same prompt works in English (the referenced doc is bilingual). Gate
rule for a profiled small model: **`check_expected.sh … small` exit 0 AND the
Hurl engine-health tests pass** (see §5). A raw `run.sh` non-zero caused only by
an advisory miss (`instruction`/reasoning) is **expected**, not "broken".

---

## 1. Prerequisiti / Prerequisites

**IT**
- macOS su Apple Silicon (per engine-mlx). `cargo`, `curl`, `jq`.
- `hurl` (`brew install hurl`) — opzionale; se assente, `run.sh` salta il layer
  dichiarativo ed esegue comunque nxm-bench.
- Env di build MLX: `MLX_C_PATH=/opt/homebrew/opt/mlx-c` e
  `MLX_PREFIX=/opt/homebrew/opt/mlx`.
- Porta standard **11435** (11434 è Ollama, riservata alla TUI).
- Il modello `Qwen3-1.7B-MLX-4bit` deve esistere in `$NEXUM_MODELS_DIR`
  (default `~/nexum/models`).

**EN** — Apple Silicon macOS; `cargo`/`curl`/`jq`; optional `hurl`; MLX build
env; port 11435; the `Qwen3-1.7B-MLX-4bit` model present under
`$NEXUM_MODELS_DIR`.

---

## 2. Procedura passo-passo / Step-by-step

**IT**

1. **Build:** `cargo build --release` (da `engine-bench/`). Se fallisce → esci,
   riporta l'errore del compilatore; non proseguire.
2. **Avvia il server:** `./models/start-qwen3-1.7b-4bit.sh`. Attendi la riga
   `engine 'engine-mlx' up at http://127.0.0.1:11435`. Se non appare entro
   `READY_TIMEOUT` (default 300s), leggi il log e fermati.
3. **Conferma il model id:**
   `curl -s http://127.0.0.1:11435/v1/models | jq -r '.data[0].id'`
   (deve essere `Qwen3-1.7B-MLX-4bit`).
4. **Esegui:** `./run.sh`. Lancia Hurl poi nxm-bench, scrive i report e stampa
   un Summary con PASS/FAIL per stage.
5. **Verdetto automatico:**
   `./check_expected.sh "$(ls -t results/engine-mlx/*.json | head -1)" small`.
6. **Ferma:** `./models/stop-qwen3-1.7b-4bit.sh` (sempre, anche in caso di
   fallimento).

**EN** — Build → start the 4bit model → confirm the id is `Qwen3-1.7B-MLX-4bit`
→ `./run.sh` → `check_expected.sh ... small` → stop.

---

## 3. Cosa fanno i 5 benchmark (domanda → cosa verifica) / What the 5 benchmarks do

**IT** — Ogni test invia una **domanda** (prompt) al modello e controlla la
**risposta**. Da ora il report registra sia la domanda sia la risposta, così è
auto-esplicativo (vedi §4).

| # | Test | Domanda inviata al modello | Cosa verifica |
|---|------|----------------------------|---------------|
| 1 | `factual` (correttezza) | "What is the capital of France? Answer with just the city name." | La risposta contiene `Paris`. |
| 2 | `instruction` (correttezza) | "Reply with exactly one word: DONE. Do not add anything else." | Segue il formato: risponde solo `DONE`, in modo conciso. |
| 3 | `sustained` (degradazione) | "Write a short paragraph about the ocean." (×20 identiche) | Il throughput (t/s) non cala tra le richieste — leak guard. |
| 4 | `length_ramp` (degradazione) | "Tell me a detailed story about a journey across the sea." (128/512/1024 token) | Come scala il t/s al crescere dell'output — scaling KV-cache. |
| 5 | `coherence` (coerenza) | "A store has 3 crates. Each crate holds 4 boxes. Each box holds 6 apples. The store sells 54 apples. How many apples are left? …" | Ragionamento multi-step: risposta finale `18` con passaggi. |

**EN** — Each test sends a **question** (prompt) and checks the **answer**. The
report now records both the question and the answer, so it is self-documenting.
Same table as above: factual→Paris, instruction→DONE, sustained→stable t/s,
length_ramp→KV scaling, coherence→answer 18.

---

## 4. Dove sono i risultati / Where the results are

**IT**

```
results/engine-mlx/
  <YYYY-MM-DD_HH-MM-SS>.json   # report completo nxm-bench (machine-readable)
  <YYYY-MM-DD_HH-MM-SS>.md     # riassunto leggibile nxm-bench
  hurl_<YYYY-MM-DD_HH-MM-SS>/  # report hurl HTML + JSON
```

**Il report Markdown/JSON ora contiene, per ogni test, sia la _Question
(prompt)_ sia la _Answer (model output)_.** Nel Markdown vedrai per ogni test:

```
### 1. Factual recall (capital of France) — ✅ PASS
*Deterministic factual question; response must contain 'Paris'.*
...
Question (prompt sent to the model):

    What is the capital of France? Answer with just the city name.

Answer (model output sample(s)):

    Paris
```

Nel JSON il campo è `tests[].prompt` (la domanda) accanto a `tests[].outputs`
(la risposta). Estrazione rapida:

```bash
LATEST=$(ls -t results/engine-mlx/*.json | head -1)
jq '[.tests[] | {id, question: .prompt, answer: .outputs, passed}]' "$LATEST"
```

**EN** — Reports land in `results/engine-mlx/`. The report now includes, per
test, both **Question (prompt)** and **Answer (model output)**. JSON: the
question is `tests[].prompt`, the answer is `tests[].outputs`.

---

## 5. Interpretazione / Interpretation

**IT** — Vale la **regola d'oro** di `AGENT_TESTING.md` §6:

- **Degradazione + API + latenza misurano l'ENGINE.** Un FAIL qui è un bug reale.
- **Correttezza + coerenza misurano il MODELLO.** Per un 1.7B (profilo `small`):
  `factual` di solito passa (è `must_pass`); `instruction` spesso passa; il
  reasoning multi-step (`coherence`) **può** fallire senza che sia un bug engine.
- **Output spazzatura** (vuoto, token ripetuti, `?????`, lingua non pertinente)
  in qualunque test = **bug engine**. Leggi la _Answer_ nel report per decidere.

Profilo `small`: `min_tps` = **25**; `must_pass` = salute engine + `factual`;
`coherence`/`instruction` sono advisory.

**Autorità del verdetto per un modello con profilo:** è `check_expected.sh` per
il profilo giusto (+ layer Hurl engine-health via `run.sh`), **non** l'AND grezzo
di tutti e 5 i test asserted. `nxm-bench --assert` (e quindi l'exit code di
`run.sh`) è il gate *crudo* e agnostico alla taglia: un advisory FAIL (es.
`instruction` su un 1.7B) lo rende rosso, ed è **atteso**. Per un modello
piccolo il verdetto autorevole è:

> **HEALTHY se `check_expected.sh … small` esce 0 E il layer Hurl engine-health
> passa** (api_*, error_handling, perf_decode_latency, coherence). Gli advisory
> (`instruction`, `coherence` reasoning) possono fallire senza rendere il run
> "rotto". Nella tabella riassuntiva di `check_expected.sh` un advisory è marcato
> `FAIL (advisory)` in giallo, distinto da un `FAIL` must_pass in rosso.

Non usare l'AND grezzo `run.sh 0 AND check 0` come gate per un modello piccolo:
quella regola tratta il gate crudo come se conoscesse la taglia. La consapevolezza
della taglia vive nei profili di `check_expected.sh`.

**EN** — Verdict authority for a profiled model is `check_expected.sh` for the
right profile (+ the Hurl engine-health layer via `run.sh`), **not** the raw AND
of all 5 asserted tests. `nxm-bench --assert` / `run.sh` exit code is the raw,
size-agnostic gate: an advisory FAIL (e.g. `instruction` on a 1.7B) reddens it
and is **expected**. For a small model: **HEALTHY if `check_expected.sh … small`
exits 0 AND the Hurl engine-health tests pass**; advisory misses don't make the
run "broken". Advisory rows show as yellow `FAIL (advisory)` vs red must_pass
`FAIL`. Golden rule (§6 of `AGENT_TESTING.md`): engine-health/API/latency FAIL =
real bug; correctness/coherence FAIL on a 1.7B is expected, unless the answer is
garbage (then it's an engine bug).

---

## 6. Riferimento rapido / Quick reference

```bash
# start/stop per il solo modello 4bit (porta 11435 incorporata)
./models/start-qwen3-1.7b-4bit.sh   ./models/stop-qwen3-1.7b-4bit.sh

# run completo (Hurl + nxm-bench), archivia i report, exit 0/1/2
./run.sh

# solo nxm-bench
./target/release/nxm-bench run --url http://127.0.0.1:11435 --engine engine-mlx --assert

# verdetto automatico contro il profilo small (1-4B)
./check_expected.sh "$(ls -t results/engine-mlx/*.json | head -1)" small
```
