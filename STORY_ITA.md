# La storia di engine-mlx


## Cosa ho costruito

[engine-mlx](https://github.com/dangranaz/engine-mlx) è un piccolo motore di
inferenza LLM per Apple Silicon, onesto, scritto in Rust sopra il framework MLX
di Apple (via la sua C API). Carica un modello quantizzato, esegue prefill e
decode, espone un'API HTTP OpenAI-compatibile e genera testo **identico a
`mlx_lm` token per token** a temperatura 0.

Insieme ho pubblicato:
- [prj-bench](https://github.com/dangranaz/prj-bench) — uno strumento di
  benchmark riproducibile *e* i numeri reali di riferimento, così chiunque può
  rieseguirli.
- [prj-scripts](https://github.com/dangranaz/prj-scripts) — semplici script
  start/stop per avviare il server.

Tutti e tre con CI verde.

## La parte onesta

Ho costruito engine-mlx principalmente **orchestrando agenti AI di coding** —
guidandoli e re-ingegnerizzando lavoro che avevo già fatto — più che scrivendo
ogni riga a mano. La competenza qui in mostra non è "so scrivere kernel MLX in
Rust a memoria". È **guidare agenti AI a costruire e verificare software di
sistema complesso, e saper riconoscere quando un risultato è reale e quando no.**

È una distinzione che conta, perché è la seconda competenza ad aver prodotto un
motore funzionante.

## Perché Rust

Ho scelto **Rust** di proposito, per robustezza: tipizzazione forte, niente
garbage collector, ownership della memoria esplicita, ed errori che gestisci
invece di scoprire a runtime — esattamente ciò che serve in un componente di
sistema come un motore di inferenza. Il compromesso è onesto: rispetto a Python,
l'ecosistema Rust per MLX è ancora acerbo — i binding sono più scarni, gli
esempi meno — quindi c'è più da costruire e verificare partendo dalle primitive.
Quell'attrito è parte del motivo per cui la disciplina di verifica qui sotto ha
contato così tanto.

## Il setup (e il vincolo)

Tutto questo — il motore, i fix, i benchmark — è stato fatto su un **MacBook Air
M1 con 16 GB di memoria unificata**. Non è una nota a piè di pagina: il vincolo
hardware ha plasmato il lavoro. Il bug più insidioso riguardava proprio il
mantenere basso il conteggio dei buffer Metal vivi su memoria limitata;
"funziona" doveva significare "funziona su questa macchina", non su una
workstation.

Lavoro prevalentemente con **OpenCode** e **Pi**, usando i modelli AI **gratuiti**
di OpenCode e i **modelli gratuiti offerti da NVIDIA** — nessun abbonamento a
modelli a pagamento. La leva non è uno strumento costoso; è guidare bene questi
agenti e tenerli a un'asticella dura e verificabile per il "fatto".

## Cosa ha significato "verificare"

L'ingegneria interessante non è stata scrivere codice — è stata scegliere
obiettivi oggettivi e infalsificabili, e tenere il lavoro a quel livello:

- **Parità token-esatta con `mlx_lm`** a temperatura 0. Non "sembra simile" — gli
  *stessi ID di token*. Se ne differisce uno, è un bug.
- **Un benchmark riproducibile**, pubblicato con i numeri, così le affermazioni
  di performance si possono verificare, non credere sulla fiducia.
- **Dichiarare i limiti onestamente**: engine-mlx è una baseline
  correctness-first; il throughput assoluto è ancora sotto `mlx_lm`, e il gap è
  efficienza dei kernel, non overhead del grafo. È scritto nel README.

## Il bug che dimostra il punto

Prima di pubblicare, i benchmark hanno beccato qualcosa di brutto: sotto carico
HTTP sostenuto l'output degenerava in spazzatura ("...") dalla terza richiesta
circa. La tentazione è rattoppare i sintomi. Invece l'ho diagnosticato:

- **Preesisteva** ed era mascherato perché i test resettano lo stato tra le
  esecuzioni mentre il path HTTP no — provato riproducendolo sul codice
  precedente.
- L'errore era `[metal::malloc] Resource limit (499000)` con memoria a solo
  ~4 GB su un limite di ~15 GB — quindi non erano byte, era un **conteggio di
  buffer vivi**.
- La soluzione è arrivata rileggendo un mio design precedente che funzionava e
  allineandomi ad esso: reset e ricostruzione dello stato pulito ad ogni
  richiesta, e liberazione esplicita dei buffer MLX (refcounted) invece del solo
  drop degli handle Rust.

Dopo il fix: stabile su richieste miste 128/512/1024 token e raffiche lunghe,
con numeri sani (~32–40 t/s su Qwen3-1.7B-4bit, ~8% di degradazione sostenuta).

## Perché lo condivido

Il valore non è un benchmark appariscente. È un motore piccolo e auditabile che
dimostra la propria correttezza, dice esattamente dove sta, ed è stato costruito
guidando agenti AI con un'asticella rigorosa per il "fatto". È così che lavoro —
ed è la competenza per cui voglio essere riconosciuto.
