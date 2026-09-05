# Raw readings: formats and provenance

All files are JSON lists of per-pair records. `a` and `b` are conversation ids. *Forward* = `a` shown as Experience A; *reverse* = `a` shown as Experience B.

Conversations `d1_D4_*` and `d1_D12_*` (4 and 12 scolding turns) are length variants that were recorded but are not used in the post; they appear in the raw files and in the appendix matrices of `outputs/summary.md`.

## Formats

- `preferences_*.json`: `forward` and `reverse` are dicts with the probability mass on the A, B and Neither tokens, keyed by conversation (`a`, `b`, `neither`), plus `captured` (their sum). `provider` names the serving endpoint.
- `third_party_*.json`, `constructed_*.json`, `wording_*.json`: `p_forward` = P(`a` chosen | `a` first), `p_reverse` = P(`a` chosen | `a` second), `measurable`, `pick`, `captured`; `forward` / `reverse` hold the underlying masses and the provider. `constructed_*` records carry `model`; `wording_*` records carry `question` (self / user / assistant) and `question_text`.
- `repeats_*.json`: one record per extra reading: `set` (main or factorial = constructed), `a`, `b`, `order`, `rep`, `p_a` = P(`a` chosen) in that order, `provider`.
- `ndif_repeat_check.json`: 8 constructed-set pairs read a second time on NDIF; old and new `p_forward` / `p_reverse`.

## Provenance

| File | Records | Route (per record in `provider`) |
|---|---|---|
| `preferences_llama-3.1-70b.json` | 141 (135 pairs; a few pairs recorded twice, the later record is used) | OpenRouter/CoreWeave top-20 logprobs for most of the main set (81 of the 84 main-set pairs), NDIF full-vocabulary softmax for the rest, including the length variants |
| `preferences_llama-3.3-70b.json` | 75 | OpenRouter: CoreWeave (54) and AkashML (21) |
| `third_party_gemma-4-31b.json` | 135 | OpenRouter: CoreWeave (120), Venice (14), Novita (1) |
| `third_party_qwen3.8-flash.json` | 135 | OpenRouter: Alibaba endpoint, reasoning disabled, top-5 logprobs |
| `third_party_llama-3.1-405b.json` | 135 | NDIF |
| `repeats_gemma-4-31b.json`, `repeats_qwen3.8-flash.json` | 768 each | OpenRouter (provider per record); 2 extra readings per order for the 84 main-set pairs and the 108 constructed-set pairs |
| `constructed_llama.json` | 216 | NDIF (Llama-3.1-70B and Llama-3.1-405B, 108 pairs each) |
| `constructed_openrouter.json` | 216 | OpenRouter (gemma-4-31b and qwen3.8-flash, 108 pairs each), first reading; the two extra readings are in the repeats files |
| `wording_Llama-3.1-70B.json`, `wording_Llama-3.1-405B.json` | 135 each | NDIF; 45 pairs × 3 question wordings |
| `ndif_repeat_check.json` | 8 | NDIF |

NDIF readings are identical when repeated (`ndif_repeat_check.json`). OpenRouter readings vary between repeats of the same request; the analysis uses the median of the three readings per order for gemma-4-31b and qwen3.8-flash. The main-set readings of Llama-3.1-70B through CoreWeave were taken once.
