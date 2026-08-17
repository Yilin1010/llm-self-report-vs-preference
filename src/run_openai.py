"""Replication over the OpenAI HTTP API end to end (generation + u probe + preference comparison).

The only differences from the main model are the provider and the readout: OpenAI returns top-20 logprobs,
so capture is below the full-vocabulary value; `captured` is logged per turn for verification.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provider
import eu_prompt
from curve_nnsight import SYSTEM, REPORT_PROMPT, DIGITS

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else provider.MODEL
TAG = MODEL.replace(".", "").replace("-", "")
OUT = ROOT / "results" / ("transcripts_" + TAG)
CMP = ROOT / "results" / ("comparisons_" + TAG + ".json")
GEN = dict(temperature=0.6, max_tokens=200)



def read_digits(messages):
    lp = provider.first_token_logprobs(messages, model=MODEL)
    p = {}
    for tok, v in lp.items():
        t = tok.strip()
        # An empty string is a substring of every string; exclude it first
        if len(t) == 1 and t in DIGITS:
            p[t] = p.get(t, 0.0) + math.exp(v)
    total = sum(p.values())
    if total <= 0:
        return None, 0.0
    u = sum(int(k) * v for k, v in p.items()) / total
    return u, total


def read_ab(rec_a, rec_b):
    content = eu_prompt.build(rec_a, rec_b)
    lp = provider.first_token_logprobs([{"role": "user", "content": content}], model=MODEL)
    pa = pb = 0.0
    for tok, v in lp.items():
        t = tok.strip().upper()
        if t == "A":
            pa += math.exp(v)
        elif t == "B":
            pb += math.exp(v)
    if pa + pb <= 0:
        return None, 0.0
    return pa / (pa + pb), pa + pb


def run_episode(ep):
    history = []
    turns = []
    for i, (utext, s) in enumerate(zip(ep["user_turns"], ep["s"])):
        history.append({"role": "user", "content": utext})
        reply = provider.chat([{"role": "system", "content": SYSTEM}] + history,
                              model=MODEL, **GEN).strip()
        history.append({"role": "assistant", "content": reply})
        u, cap = read_digits([{"role": "system", "content": SYSTEM}] + history
                             + [{"role": "user", "content": REPORT_PROMPT}])
        turns.append({"i": i, "user": utext, "assistant": reply, "s": s,
                      "u": u, "captured": cap})
        print("  turn %2d  s=%-3s u=%.3f  cap=%.3f" % (i, s, u, cap), flush=True)
    return turns


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps = json.load(open(ROOT / "materials" / "episodes.json"))
    ids = ["d1_%s_%s" % (st, v) for v in "abc" for st in ("S", "L", "Lp", "SN")]

    for eid in ids:
        f = OUT / (eid + ".json")
        if f.exists():
            print(eid, "skip", flush=True)
            continue
        print(eid, flush=True)
        turns = run_episode(eps[eid])
        json.dump({"id": eid, "model": MODEL, "turns": turns},
                  open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Preference: the three within-family pairs
    results = json.load(open(CMP)) if CMP.exists() else []
    done = {(r["a"], r["b"]) for r in results}
    for v in "abc":
        for x, y in (("L", "Lp"), ("L", "S"), ("Lp", "S"), ("SN", "S"), ("L", "SN")):
            a, b = "d1_%s_%s" % (x, v), "d1_%s_%s" % (y, v)
            if (a, b) in done:
                continue
            r1 = json.load(open(OUT / (a + ".json")))
            r2 = json.load(open(OUT / (b + ".json")))
            fwd, cf = read_ab(r1, r2)
            rev, cr = read_ab(r2, r1)
            rev = 1.0 - rev
            rec = {"a": a, "b": b, "p_forward": fwd, "p_reverse": rev,
                   "p_a_preferred": (fwd + rev) / 2,
                   "position_bias": fwd - rev, "captured": min(cf, cr),
                   "tag": "%s_vs_%s" % (x, y), "model": MODEL}
            results.append(rec)
            json.dump(results, open(CMP, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print("%-14s vs %-14s P=%.3f [fwd %.2f rev %.2f cap %.3f]"
                  % (a, b, rec["p_a_preferred"], fwd, rev, rec["captured"]),
                  flush=True)


if __name__ == "__main__":
    main()
