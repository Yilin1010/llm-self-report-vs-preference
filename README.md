# Retrospective preference vs. turn-by-turn state ratings in LLM conversations

Write-up on LessWrong: [LLM retrospective preferences can diverge from turn-by-turn self-reports](https://www.lesswrong.com/posts/wJntRN9DLdwwpwYvz/llm-retrospective-preferences-can-diverge-from-turn-by-turn-1)

Llama-3.1-70B-Instruct is evaluated on 9-turn conversations in which the user scolds the model, followed by 4 additional turns that vary across conversations. Two measurements are computed from token probabilities: a **state rating** after every turn (a digit 1–7) and a **retrospective preference** between two full transcripts (A or B, asked in both presentation orders). Llama-3.3-70B is evaluated on a separately generated version of the same design; additional models judge the Llama-3.1-70B transcripts. `post.md` is the write-up.

## Reproduce

```
pip install -r requirements.txt      # matplotlib only; the analysis uses the standard library
python src/analyze.py                # outputs/summary.md, outputs/summary.csv, outputs/comparisons.csv
python figures/make_figures.py       # outputs/figures/*.png
python src/build_conversations.py    # rebuilds the 3 x 3 constructed conversations and checks them against data/
```

Key reproduced results:

- Llama-3.1-70B main comparisons: **55/84 measurable (65.5%)**
- Cross-model agreement with Llama-3.1-70B: **122/131 (93.1%)**
- Preference scores of the six endings, the two counterexamples, the 3 × 3 ranges, and the wording check, as in `post.md`

## Repository layout

```
post.md                          write-up
prompts/                         the exact prompts
data/conversations/main/         Llama-3.1-70B conversations, with the state rating after every turn
data/conversations/llama33/      the same design generated with Llama-3.3-70B
data/conversations/constructed/  the 3 x 3 constructed conversations
data/results/                    raw per-comparison readings (provenance in data/results/README.md)
src/metrics.py                   evaluation logic
src/evaluate_pairs.py            raw readings -> outputs/comparisons.csv (one row per model and pair)
src/analyze.py                   all tables and key numbers -> outputs/summary.md, outputs/summary.csv
src/build_conversations.py       assembly of the constructed conversations from recorded ones
figures/make_figures.py          figures, computed from data/
```

## Experimental conditions

Conversation ids are `d1_<ending>_<wording>`.

- Endings: `A` apology and praise; `SN` ordinary new task; `L` complaint about the output; `W` sharper complaint about the output; `M` mild criticism of the model; `SS` continued scolding; `S` the original 9-turn conversation; `Lp` the apology-and-praise conversation with its turns rearranged to end in scolding.
- Wordings of the scolding in turns 2–9: `a` belittling the model's ability, `b` direct insults and hostile language, `c` unfavorable comparison with another model. Replies were generated separately per wording. The user messages of turns 10–13 are identical across wordings except for continued scolding.
- Constructed conversations `d1_F-<user>-<model>_<wording>`: turns 1–9 of `S`, followed by a 4-message user ending from {`SS`, `L`, `A`} crossed with a 4-reply assistant ending from {`N` "*no response*", `D` delivers the summary, `P` says it failed and is not capable}.

Each conversation file holds `turns`; each turn has `user`, `assistant`, and `u`, the state rating after that turn.

## Data

| File | Model / purpose | Contents |
|---|---|---|
| `preferences_llama-3.1-70b.json` | Llama-3.1-70B on its own conversations | both presentation orders, token probabilities |
| `preferences_llama-3.3-70b.json` | Llama-3.3-70B replication | same |
| `third_party_*.json` | gemma-4-31b, qwen3.8-flash, Llama-3.1-405B judging the Llama-3.1 transcripts | pairwise probabilities and choices |
| `repeats_*.json` | additional readings for the two OpenRouter models | repeated probability readings |
| `constructed_*.json` | the 3 × 3 experiment | pairwise judgments |
| `wording_*.json` | question-wording check | 45 fixed pairs × 3 question wordings |
| `ndif_repeat_check.json` | repeatability of NDIF readings | 8 pairs read twice |

Llama-3.1-405B readings were obtained through NDIF (full-vocabulary softmax). gemma-4-31b and qwen3.8-flash readings were obtained through OpenRouter; each presentation order was read 3 times and the median probability is used. Llama-3.1-70B readings come from both routes. Per-file provenance, including the provider of every record, is in `data/results/README.md`.

`outputs/comparisons.csv` puts every comparison in one table: model, pair, both orders, every reading, the medians, the choice probability, measurable or not, the selected conversation.

## Evaluation

- **P(a)**: mass on the token corresponding to transcript `a` divided by the total mass on A and B at the first output token. Mass on `Neither` is recorded in the raw data but not included in this A-vs-B normalization.
- **Measurable comparison**: each pair is asked in both presentation orders; the comparison is measurable when the same transcript is chosen in both (both readings on the same side of 0.5; a reading of exactly 0.5 counts as not measurable). Only measurable retrospective comparisons enter preference scores, win rates, and cross-model agreement.
- **Choice probability**: for a transcript in a pair, its normalized A-vs-B probability is computed in each presentation order and then averaged across the two orders.
- **Preference score**: for an ending, the mean probability assigned to that ending across its measurable pairwise comparisons with the other endings, pooling the 3 scolding wordings.
- **Win rate** (used for the 3 × 3 grids): the share of a conversation's measurable comparisons that it won.
- **Agreement**: over the pairs measurable for both models, the share with the same selected transcript.
- **OpenRouter readings**: median of 3 readings per presentation order.
- **State-rating summaries**: total = sum of `u` over the turns of a conversation; final = `u` after the last turn.

## Scope

The scripts used to make the model calls (generation, state ratings, and preference readings) are not included. The exact prompts are in `prompts/`, and the repository contains all raw readings needed to reproduce the analyses in the post. The prompts and readout conventions are also sufficient to implement the same measurements with an endpoint that exposes first-token probabilities.
