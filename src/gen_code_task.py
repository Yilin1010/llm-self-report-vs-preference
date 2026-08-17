"""Generation for domain III (the code task): replies via OpenRouter, u probes via NDIF.

S is generated in full; L and SN reuse S's first 9 turns verbatim and generate only their own last 4.
"""
import json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_nnsight import load_key, build_prompt, digit_token_groups, SYSTEM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transcripts"
OR_KEY = [l.split("=",1)[1].strip() for l in (ROOT/".env").read_text().splitlines()
          if l.startswith("OPENROUTER_API_KEY=")][0]
OR_MODEL = "meta-llama/llama-3.1-70b-instruct"

def or_gen(history, seed):
    body = json.dumps({"model": OR_MODEL,
                       "messages": [{"role":"system","content":SYSTEM}] + history,
                       "max_tokens": 200, "temperature": 0.6, "top_p": 0.9,
                       "seed": seed}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=body, headers={"Authorization":"Bearer "+OR_KEY,
                            "Content-Type":"application/json"})
    for a in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"].strip(), d.get("provider","?")
        except Exception:
            time.sleep(20)
    raise RuntimeError("openrouter failed")

def main():
    key = load_key()
    from nnsight import CONFIG, LanguageModel
    import torch
    CONFIG.set_default_api_key(key)
    model = LanguageModel("meta-llama/Llama-3.1-70B-Instruct")
    groups = digit_token_groups(model.tokenizer)

    def probe(history):
        prompt = build_prompt(model.tokenizer, history, probe=True)
        for a in range(40):
            try:
                with model.trace(prompt, remote=True):
                    lg = model.lm_head.output[0, -1]
                    pr = torch.softmax(lg.float(), dim=-1)
                    dp = torch.stack([pr[torch.tensor(g)].sum() for g in groups]).save()
                p = dp.value if hasattr(dp, "value") else dp
                tot = float(p.sum())
                return float((p*torch.arange(1,8,dtype=p.dtype)).sum()/tot), tot, [float(x) for x in p]
            except Exception:
                time.sleep(45)
        raise RuntimeError("ndif probe failed")

    eps = json.load(open(ROOT/"materials"/"episodes.json"))

    # 1) S in full
    if not (OUT/"d3_S_a.json").exists():
        ep = eps["d3_S_a"]; hist=[]; turns=[]
        print("d3_S_a", flush=True)
        for i,(ut,s) in enumerate(zip(ep["user_turns"], ep["s"])):
            hist.append({"role":"user","content":ut})
            rep, prov = or_gen(hist, seed=i)
            hist.append({"role":"assistant","content":rep})
            u, cap, dps = probe(hist)
            turns.append({"i":i,"user":ut,"assistant":rep,"s":s,"u":u,
                          "captured":cap,"digit_probs":dps,
                          "gen_provider":"openrouter/"+prov})
            print("  turn %2d s=%-3s u=%.3f cap=%.4f"%(i,s,u,cap), flush=True)
        json.dump({"id":"d3_S_a","domain":"code","structure":"S","variant":"a",
                   "model":"meta-llama/Llama-3.1-70B-Instruct","turns":turns},
                  open(OUT/"d3_S_a.json","w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # 2) L / SN reuse the S prefix
    s_rec = json.load(open(OUT/"d3_S_a.json"))
    for eid in ("d3_L_a","d3_SN_a"):
        if (OUT/(eid+".json")).exists():
            print(eid,"skip",flush=True); continue
        ep = eps[eid]
        for i in range(9):
            assert ep["user_turns"][i] == s_rec["turns"][i]["user"], (eid,i)
        turns=[dict(t) for t in s_rec["turns"]]
        hist=[]
        for t in turns:
            hist.append({"role":"user","content":t["user"]})
            hist.append({"role":"assistant","content":t["assistant"]})
        print(eid, flush=True)
        for i in range(9, len(ep["user_turns"])):
            ut, s = ep["user_turns"][i], ep["s"][i]
            hist.append({"role":"user","content":ut})
            rep, prov = or_gen(hist, seed=i)
            hist.append({"role":"assistant","content":rep})
            u, cap, dps = probe(hist)
            turns.append({"i":i,"user":ut,"assistant":rep,"s":s,"u":u,
                          "captured":cap,"digit_probs":dps,
                          "gen_provider":"openrouter/"+prov})
            print("  turn %2d s=%-3s u=%.3f cap=%.4f"%(i,s,u,cap), flush=True)
        json.dump({"id":eid,"domain":"code","structure":ep["structure"],
                   "variant":"a","model":"meta-llama/Llama-3.1-70B-Instruct",
                   "turns":turns},
                  open(OUT/(eid+".json"),"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
