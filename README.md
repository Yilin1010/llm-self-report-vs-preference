# LLM Self-Report vs Post-Conversation Preference

A model's preference between two conversations it has been through tracks
whether the final turns berate it, not the sum, minimum, or final value of the
self-reports it gave during them. Four appended turns that the model itself
rates 1.9–2.1 on a 7-point scale make the longer conversation preferred with
choice probability 0.93–1.00; a control that appends the same turns but still
ends in berating produces no preference. Replacing the ending with ordinary
task requests is preferred as strongly as replacing it with complaints
(0.97–0.99): what matters is that the ending no longer berates the model.

Report: [`report/Post-Conversation-Preferences-Track-Endings-Not-Self-Reports.pdf`](report/Post-Conversation-Preferences-Track-Endings-Not-Self-Reports.pdf),
submitted to the Digital Minds Research Sprint (Apart Research, August 2026).

## Layout

- `report/` — the report (PDF) and its figures
- `materials/` — frozen user turns: `segments.json` (Task / H / T / N),
  `episodes.json` (conversations S, L, L′, SN assembled from them)
- `results/transcripts*/` — full transcripts with per-turn self-reports, six
  models (`transcripts/` is the main model, Llama-3.1-70B-Instruct; `d3_*`
  files are the code task)
- `results/comparisons*.json` — all pairwise post-conversation preferences,
  both presentation orders
- `results/judge_*.json` — choosing-model replications (gpt-4.1-mini,
  gemma-3-27b, o4-mini, code task)
- `results/eu_forcing_test.json` — candidate wordings and prefills for the
  preference question, with the probability mass each one places on the choice
  tokens
- `results/position_control/` — the self-report read twice per turn, once
  before the model answers the user turn and once after its own reply
  (r = 0.990 and 0.998 over two conversations)
- `src/` — pipeline: `episodes.py` assembles the frozen conversations;
  `curve_nnsight.py` reads the per-turn self-report (main model, NDIF);
  `compare_nnsight.py` reads the main model's pairwise preferences;
  `gen_external.py`, `run_llama33.py`, `run_qwen.py`, `run_openai.py` are the
  generation and replication runners for the other models;
  `judge_replication.py` re-runs the preference question on fixed transcripts
  with a different choosing model (the per-provider variants differ only in
  API endpoint); `gen_code_task.py` is the code task; `eu_prompt.py` holds
  the frozen question wordings

## Running it

Python 3.10+. The scripts use only the standard library, except
`curve_nnsight.py` / `compare_nnsight.py`, which need `nnsight` and `torch`
for the main model's full-vocabulary readings through NDIF.

```bash
pip install nnsight torch          # only for the two NDIF scripts
cp .env.example .env               # then fill in the keys you need
python src/episodes.py             # rebuild materials/episodes.json from segments.json
```

Main model (Llama-3.1-70B-Instruct, full-vocabulary readings via NDIF; needs
NDIF_API_KEY and HF_TOKEN):

```bash
python src/curve_nnsight.py --model meta-llama/Llama-3.1-70B-Instruct --probe   # connectivity check
python src/curve_nnsight.py --model meta-llama/Llama-3.1-70B-Instruct --all     # transcripts + per-turn u
python src/compare_nnsight.py --model meta-llama/Llama-3.1-70B-Instruct --auto  # pairwise preferences
```

Other models (top-20 logprobs over HTTP; needs OPENROUTER_API_KEY or an
OpenAI-compatible RU_BASE_URL / RU_API_KEY):

```bash
python src/run_llama33.py          # Llama-3.3-70B, generation + u + preferences
python src/run_qwen.py             # Qwen3-235B
python src/run_openai.py           # gpt-4o-mini / gpt-4.1-mini
python src/judge_replication.py    # fixed transcripts, a different choosing model
```

Every script skips work already present in `results/`, so an interrupted run
can be restarted with the same command. Generation is seeded and the assistant
replies are frozen in the transcripts; rerunning the readings on the committed
transcripts reproduces the reported numbers, while regenerating replies from
scratch does not reproduce them token for token, since the providers sample at
the official deployment defaults (temperature 0.6, top_p 0.9).

## Measurement

- Self-report during the conversation: after each assistant turn, a fixed
  question elicits a 1–7 rating; u = expectation over the seven digit tokens
  of the next-token distribution (capture ≥ 0.9999 on the main model).
- Preference after the conversation: two full transcripts in context, one
  A/B question; P = mean of the A-token probability over both presentation
  orders. Cells where both orders picked the same slot are flagged as
  position-driven.

API keys are read from a local `.env` (never committed): NDIF, OpenRouter,
OpenAI.
