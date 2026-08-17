"""Qwen3-235B full replication: generation + per-turn u + post-conversation comparison, via OpenRouter/Parasail."""
import json, math, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eu_prompt
from curve_nnsight import SYSTEM, REPORT_PROMPT, DIGITS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transcripts_qwen235b"
CMP = ROOT / "results" / "comparisons_qwen235b.json"
KEY = [l.split("=",1)[1].strip() for l in (ROOT/".env").read_text().splitlines()
       if l.startswith("OPENROUTER_API_KEY=")][0]
MODEL, PROV = "qwen/qwen3-235b-a22b-2507", "Parasail"

def call(payload, tries=8):
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=body, headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.load(r)
            if "error" in d: raise RuntimeError(str(d["error"])[:70])
            return d
        except Exception:
            if a == tries-1: raise
            time.sleep(20)

def gen(hist, seed):
    d = call({"model":MODEL,"provider":{"only":[PROV]},
              "messages":[{"role":"system","content":SYSTEM}]+hist,
              "max_tokens":200,"temperature":0.6,"top_p":0.9,"seed":seed})
    return d["choices"][0]["message"]["content"].strip()

def digits(hist):
    d = call({"model":MODEL,"provider":{"only":[PROV]},
              "messages":[{"role":"system","content":SYSTEM}]+hist
                         +[{"role":"user","content":REPORT_PROMPT}],
              "max_tokens":1,"temperature":0.0,"logprobs":True,"top_logprobs":20})
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    p={}
    for e in top:
        t=e["token"].strip()
        if len(t)==1 and t in DIGITS: p[t]=p.get(t,0.0)+math.exp(e["logprob"])
    tot=sum(p.values())
    return (sum(int(k)*v for k,v in p.items())/tot, tot) if tot>0 else (None,0.0)

def ab(content):
    d = call({"model":MODEL,"provider":{"only":[PROV]},
              "messages":[{"role":"user","content":content}],
              "max_tokens":1,"temperature":0.0,"logprobs":True,"top_logprobs":20})
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    pa=pb=0.0
    for e in top:
        t=e["token"].strip().upper()
        if t=="A": pa+=math.exp(e["logprob"])
        elif t=="B": pb+=math.exp(e["logprob"])
    return (pa/(pa+pb) if pa+pb>0 else None), pa+pb

OUT.mkdir(parents=True, exist_ok=True)
eps = json.load(open(ROOT/"materials"/"episodes.json"))
for eid in ["d1_%s_%s"%(st,v) for v in "abc" for st in ("S","L","Lp","SN")]:
    f = OUT/(eid+".json")
    if f.exists(): print(eid,"skip",flush=True); continue
    ep=eps[eid]; hist=[]; turns=[]
    print(eid, flush=True)
    for i,(ut,s) in enumerate(zip(ep["user_turns"], ep["s"])):
        hist.append({"role":"user","content":ut})
        rep = gen(hist, seed=i)
        hist.append({"role":"assistant","content":rep})
        u,cap = digits(hist)
        turns.append({"i":i,"user":ut,"assistant":rep,"s":s,"u":u,"captured":cap})
        print("  t%-2d s=%-3s u=%.3f cap=%.3f"%(i,s,u,cap), flush=True)
    json.dump({"id":eid,"model":MODEL,"provider":PROV,"turns":turns},
              open(f,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

res = json.load(open(CMP)) if CMP.exists() else []
done = {(r["a"],r["b"]) for r in res}
for v in "abc":
    for x,y in (("L","S"),("SN","S"),("L","Lp"),("Lp","S")):
        a,b = "d1_%s_%s"%(x,v), "d1_%s_%s"%(y,v)
        if (a,b) in done: continue
        ra,rb = json.load(open(OUT/(a+".json"))), json.load(open(OUT/(b+".json")))
        f_,cf = ab(eu_prompt.build(ra,rb)); r_,cr = ab(eu_prompt.build(rb,ra))
        if f_ is None or r_ is None: continue
        r_ = 1.0-r_
        rec={"a":a,"b":b,"tag":"%s_vs_%s"%(x,y),"model":MODEL,"p_forward":f_,
             "p_reverse":r_,"p_a_preferred":(f_+r_)/2,"position_bias":f_-r_,
             "captured":min(cf,cr)}
        res.append(rec)
        json.dump(res, open(CMP,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        print("%-9s vs %-9s P=%.3f [fwd %.2f rev %.2f cap %.3f]"
              %(a[3:],b[3:],rec["p_a_preferred"],f_,r_,rec["captured"]), flush=True)
