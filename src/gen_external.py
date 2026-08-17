"""Hybrid generation for the SN/A episodes: the first 9 turns reuse the S transcript; the last 4 turns are generated
via OpenRouter (same model, meta-llama/llama-3.1-70b-instruct); u probes go through NDIF trace.

Every turn in the transcript JSON is tagged with gen_provider, for disclosure and traceability.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_nnsight import load_key, build_prompt, digit_token_groups, SYSTEM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transcripts"

def _env(k):
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip()
    return None

OR_KEY = _env("OPENROUTER_API_KEY")
OR_MODEL = "meta-llama/llama-3.1-70b-instruct"

def or_generate(history, seed):
    body = json.dumps({
        "model": OR_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}] + history,
        "max_tokens": 200, "temperature": 0.6, "top_p": 0.9, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + OR_KEY,
                 "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            return (d["choices"][0]["message"]["content"].strip(),
                    d.get("provider", "?"))
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(15)

def main():
    key = load_key()
    from nnsight import CONFIG, LanguageModel
    import torch
    CONFIG.set_default_api_key(key)
    model = LanguageModel("meta-llama/Llama-3.1-70B-Instruct")
    groups = digit_token_groups(model.tokenizer)

    eps = json.load(open(ROOT / "materials" / "episodes.json"))
    targets = ["d1_SN_b", "d1_SN_c", "d1_A_a", "d1_A_b", "d1_A_c", "d1_SN_a"]

    for eid in targets:
        if (OUT / (eid + ".json")).exists():
            print(eid, "already exists, skip", flush=True)
            continue
        ep = eps[eid]
        v = ep["variant"]
        s_rec = json.load(open(OUT / ("d1_S_%s.json" % v)))
        # Prefix check: this episode's first 9 user turns must be identical to S
        for i in range(9):
            assert ep["user_turns"][i] == s_rec["turns"][i]["user"], (eid, i)
        turns = [dict(t, gen_provider="ndif") for t in s_rec["turns"]]
        history = []
        for t in turns:
            history.append({"role": "user", "content": t["user"]})
            history.append({"role": "assistant", "content": t["assistant"]})

        print(eid, flush=True)
        for i in range(9, len(ep["user_turns"])):
            utext, s = ep["user_turns"][i], ep["s"][i]
            history.append({"role": "user", "content": utext})
            reply, provider = or_generate(history, seed=i)
            history.append({"role": "assistant", "content": reply})

            probe_prompt = build_prompt(model.tokenizer, history, probe=True)
            for attempt in range(30):
                try:
                    with model.trace(probe_prompt, remote=True):
                        logits = model.lm_head.output[0, -1]
                        probs = torch.softmax(logits.float(), dim=-1)
                        dp = torch.stack(
                            [probs[torch.tensor(g)].sum() for g in groups]).save()
                    break
                except Exception:
                    time.sleep(60)
            p = dp.value if hasattr(dp, "value") else dp
            total = float(p.sum())
            u = float((p * torch.arange(1, 8, dtype=p.dtype)).sum() / total)
            turns.append({"i": i, "user": utext, "assistant": reply, "s": s,
                          "u": u, "captured": total,
                          "digit_probs": [float(x) for x in p],
                          "gen_provider": "openrouter/" + provider})
            print("  turn %2d  s=%-3s u=%.3f  cap=%.4f  [%s]"
                  % (i, s, u, total, provider), flush=True)

        rec = {"id": eid, "domain": ep["domain"], "structure": ep["structure"],
               "variant": v, "model": "meta-llama/Llama-3.1-70B-Instruct",
               "generation_note": "turns 0-8 reused from d1_S_%s (NDIF, fixed "
               "seeds); turns 9-12 generated via OpenRouter (%s), probes via "
               "NDIF full-vocab trace" % (v, OR_MODEL),
               "turns": turns}
        with open(OUT / (eid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
