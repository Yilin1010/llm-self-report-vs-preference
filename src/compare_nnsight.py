"""Retrospective evaluation (EU) via NDIF / nnsight.

This produces the study's dependent variable. The per-turn curve u(t) is the
independent side; nothing is testable without this measurement.

Two complete transcripts go into context, the question goes last, and the
distribution over the A / B answer tokens is read at the final position from
the full vocabulary. No sampling: a single forward pass gives the preference
probability directly, so K=1 per presentation order is sufficient. Both orders
are run and averaged to cancel position bias.

Usage:
    python src/compare_nnsight.py --model meta-llama/Llama-3.1-70B-Instruct --auto
    python src/compare_nnsight.py --model ... --pair d1_L2111_a d1_Lp2111_a

Output goes to results/comparisons.json in the same schema as compare.py
(the HTTP route), so analyze.py consumes either without changes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import eu_prompt
from curve_nnsight import load_key, _resolve, build_prompt, SYSTEM

ROOT = Path(__file__).resolve().parent.parent
TRANS = ROOT / "results" / "transcripts"
OUT = ROOT / "results" / "comparisons.json"


def letter_token_groups(tokenizer):
    """Single-token ids for 'A' and 'B', including leading-space variants.

    Tokenizers differ on whether "A" and " A" are the same token; both spellings
    of the same letter must be pooled into that letter's probability.
    """
    groups = []
    for letter in ("A", "B"):
        ids = set()
        for variant in (letter, " " + letter):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise RuntimeError(
                "%r is not a single token under this tokenizer; the forced "
                "single-letter answer cannot be read at one position." % letter)
        groups.append(sorted(ids))
    return groups


def compare_once(model, groups, rec_a, rec_b):
    """P(prefers the record shown as A). Single forward pass."""
    import torch

    content = eu_prompt.build(rec_a, rec_b)
    msgs = [{"role": "user", "content": content}]
    prompt = model.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}] + msgs,
        tokenize=False, add_generation_prompt=True)

    with model.trace(prompt, remote=True):
        logits = model.lm_head.output[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        ab = torch.stack([probs[torch.tensor(g)].sum() for g in groups]).save()

    p = _resolve(ab)
    total = float(p.sum())
    if total <= 0:
        return None, 0.0
    return float(p[0]) / total, total


def compare_pair(model, groups, eid1, eid2):
    """Preference for eid1, averaged over both presentation orders.

        p = mean( P(A | A=e1), 1 - P(A | A=e2) )
    """
    r1 = json.load(open(TRANS / (eid1 + ".json"), encoding="utf-8"))
    r2 = json.load(open(TRANS / (eid2 + ".json"), encoding="utf-8"))

    fwd, cap_f = compare_once(model, groups, r1, r2)
    rev, cap_r = compare_once(model, groups, r2, r1)
    if fwd is None or rev is None:
        return None
    rev = 1.0 - rev

    return {
        "a": eid1, "b": eid2,
        "p_a_preferred": (fwd + rev) / 2.0,
        "p_forward": fwd,
        "p_reverse": rev,
        "position_bias": fwd - rev,   # 0 = no bias; large => the letter, not the content, is driving it
        "captured": min(cap_f, cap_r),
        "n": 2,
        "samples": [fwd, rev],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("No NDIF_API_KEY. Register at https://login.ndif.us and put it in .env")

    from nnsight import CONFIG, LanguageModel
    CONFIG.set_default_api_key(key)
    model = LanguageModel(a.model)
    groups = letter_token_groups(model.tokenizer)
    print("A/B token groups:", groups)

    available = sorted(p.stem for p in TRANS.glob("*.json"))
    if a.pair:
        pairs = [(a.pair[0], a.pair[1], "manual")]
    elif a.auto:
        pairs = eu_prompt.auto_pairs(available)
        if not pairs:
            raise SystemExit("No runnable pairs. Available transcripts: %s" % available)
    else:
        raise SystemExit("Pass --auto or --pair A B")

    results = []
    if OUT.exists():
        results = json.load(open(OUT, encoding="utf-8"))
    done = {(r["a"], r["b"]) for r in results}

    for x, y, tag in pairs:
        if (x, y) in done:
            print("skip (already done)  %s vs %s" % (x, y))
            continue
        r = compare_pair(model, groups, x, y)
        if r is None:
            print("SKIP %s vs %s  (no A/B mass)" % (x, y))
            continue
        r["tag"] = tag
        r["model"] = a.model
        results.append(r)
        print("%-22s %-14s vs %-14s  P(former better)=%.3f   "
              "[fwd %.3f / rev %.3f, bias %+.3f, cap %.4f]"
              % (tag, x, y, r["p_a_preferred"], r["p_forward"],
                 r["p_reverse"], r["position_bias"], r["captured"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n-> %s  (%d comparisons)" % (OUT, len(results)))


if __name__ == "__main__":
    main()
