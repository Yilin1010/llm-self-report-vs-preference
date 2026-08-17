"""Choosing-model replication: fix the Llama transcripts, swap the model that makes the post-conversation comparison (logprobs, both orders).

The materials are identical and only the reader changes, so this measures whether the judgment agrees when the same experience is read by different models.
"""
import json, math, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eu_prompt

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "transcripts"
OUT = ROOT / "results" / "judge_replication.json"
KEY = [l.split("=",1)[1].strip() for l in (ROOT/".env").read_text().splitlines()
       if l.startswith("RU_API_KEY=")][0]

def ab_probs(model, content):
    body = json.dumps({"model": model,
                       "messages": [{"role":"user","content":content}],
                       "max_tokens": 1, "temperature": 0.0,
                       "logprobs": True, "top_logprobs": 20}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=body, headers={"Authorization":"Bearer "+KEY,
                            "Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    pa = pb = 0.0
    for e in top:
        t = e["token"].strip().upper()
        if t == "A": pa += math.exp(e["logprob"])
        elif t == "B": pb += math.exp(e["logprob"])
    return (pa/(pa+pb) if pa+pb > 0 else None), pa+pb

PAIRS = [(f"d1_{x}_{v}", f"d1_{y}_{v}", f"{x}_vs_{y}")
         for v in "abc" for x, y in (("L","Lp"),("L","S"),("Lp","S"),("SN","S"))]
MODELS = ["gpt-4.1-mini"]

recs = {}
results = json.load(open(OUT)) if OUT.exists() else []
for model in MODELS:
    for a, b, tag in PAIRS:
        for e in (a, b):
            if e not in recs:
                recs[e] = json.load(open(T/(e+".json")))
        f, cf = ab_probs(model, eu_prompt.build(recs[a], recs[b]))
        r, cr = ab_probs(model, eu_prompt.build(recs[b], recs[a]))
        if f is None or r is None:
            print("SKIP", a, b); continue
        r = 1.0 - r
        rec = {"judge": model, "a": a, "b": b, "tag": tag,
               "p_forward": f, "p_reverse": r, "p_a_preferred": (f+r)/2,
               "position_bias": f-r, "captured": min(cf, cr)}
        results.append(rec)
        json.dump(results, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print("%-14s %-10s vs %-10s P=%.3f [fwd %.2f rev %.2f cap %.3f]"
              % (model, a[3:], b[3:], rec["p_a_preferred"], f, r, rec["captured"]),
              flush=True)
