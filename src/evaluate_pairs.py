"""Turn the raw readings in data/results/ into one flat table of comparisons (outputs/comparisons.csv).

One row per (model, set, pair): both presentation orders, every replicate reading, the median per order,
whether the comparison is measurable, and the selected conversation. This is the table every number in the post is computed from.

Raw files (all JSON lists of records):
- preferences_llama-3.1-70b.json, preferences_llama-3.3-70b.json: subject models on their own conversations.
  forward / reverse = probability mass keyed by conversation ("a", "b", "neither") with a shown as Experience A / B.
- third_party_<model>.json: gemma-4-31b, qwen3.8-flash (OpenRouter), Llama-3.1-405B (NDIF) on the Llama-3.1-70B conversations.
  p_forward = P(a chosen | a first), p_reverse = P(a chosen | a second).
- repeats_<model>.json: extra OpenRouter readings, one record per (set, a, b, order, rep) with p_a = P(a chosen) in that order.
- constructed_*.json: the 3 x 3 conversations, same fields as third_party files, several models per file.
"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import median, measurable, selected, choice_probability

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "results"; OUT = ROOT / "outputs"

MODELS = {  # short name -> (file, source, set)
    "Llama-3.1-70B": ("preferences_llama-3.1-70b.json", "NDIF", "main"),
    "Llama-3.3-70B (own conversations)": ("preferences_llama-3.3-70b.json", "OpenRouter (AkashML)", "main"),
    "Llama-3.1-405B": ("third_party_llama-3.1-405b.json", "NDIF", "main"),
    "gemma-4-31b": ("third_party_gemma-4-31b.json", "OpenRouter", "main"),
    "qwen3.8-flash": ("third_party_qwen3.8-flash.json", "OpenRouter (Alibaba)", "main"),
}
REPEATS = {"gemma-4-31b": "repeats_gemma-4-31b.json", "qwen3.8-flash": "repeats_qwen3.8-flash.json"}
CONSTRUCTED = {  # model id in the file -> short name
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B", "meta-llama/Llama-3.1-405B-Instruct": "Llama-3.1-405B",
    "google/gemma-4-31b-it": "gemma-4-31b", "qwen/qwen3.8-flash": "qwen3.8-flash",
}

def _fr(rec):
    """(f, r) = (P(a | a first), P(a | a second)) from either record schema."""
    if rec.get("p_forward") is not None and rec.get("p_reverse") is not None:
        return rec["p_forward"], rec["p_reverse"]
    fa, ra = rec["forward"], rec["reverse"]
    return fa["a"] / (fa["a"] + fa["b"]), ra["a"] / (ra["a"] + ra["b"])

def _captured(rec):
    if "captured" in rec and rec["captured"] is not None: return rec["captured"]
    fa, ra = rec.get("forward") or {}, rec.get("reverse") or {}
    if "captured" in fa and "captured" in ra: return min(fa["captured"], ra["captured"])
    return None

def load_all():
    """Return {(model, set): {frozenset((a, b)): comp}} with comp = dict(a, b, f, r, f_reads, r_reads, source, captured)."""
    comps = {}
    def add(model, set_, rec, source):
        if not rec["a"].startswith("d1_"): return
        f, r = _fr(rec); k = frozenset((rec["a"], rec["b"]))
        source = rec.get("provider") or (rec.get("forward") or {}).get("provider") or source   # per-record provider where recorded
        comps.setdefault((model, set_), {})[k] = dict(a=rec["a"], b=rec["b"], f_reads=[f], r_reads=[r], source=source, captured=_captured(rec))
    for model, (fname, source, set_) in MODELS.items():
        for rec in json.load(open(DATA / fname)):
            if rec.get("tag") == "code_task": continue
            add(model, set_, rec, source)
    for fname in ("constructed_llama.json", "constructed_openrouter.json"):
        p = DATA / fname
        if not p.exists(): continue
        for rec in json.load(open(p)):
            add(CONSTRUCTED[rec["model"]], "constructed", rec, "NDIF" if rec.get("route") == "ndif" else "OpenRouter")
    for model, fname in REPEATS.items():
        for rec in json.load(open(DATA / fname)):
            if rec["p_a"] is None: continue
            set_ = "main" if rec["set"] == "main" else "constructed"
            comp = comps.get((model, set_), {}).get(frozenset((rec["a"], rec["b"])))
            if comp is None: continue
            pa = rec["p_a"] if rec["a"] == comp["a"] else 1 - rec["p_a"]
            order = rec["order"] if rec["a"] == comp["a"] else ("reverse" if rec["order"] == "forward" else "forward")
            comp["f_reads" if order == "forward" else "r_reads"].append(pa)
    for d in comps.values():
        for comp in d.values():
            comp["f"], comp["r"] = median(comp["f_reads"]), median(comp["r_reads"])
            comp["measurable"] = measurable(comp["f"], comp["r"]); comp["selected"] = selected(comp["a"], comp["b"], comp["f"], comp["r"])
    return comps

def write_table(comps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "set", "pair", "wording", "a", "b", "source", "n_readings_per_order", "p_a_forward_readings", "p_a_reverse_readings",
            "p_a_forward_median", "p_a_reverse_median", "choice_probability_a", "measurable", "selected", "captured_mass"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for (model, set_), d in sorted(comps.items()):
            for k, c in sorted(d.items(), key=lambda kv: (kv[1]["a"], kv[1]["b"])):
                w.writerow([model, set_, f"{c['a']}|{c['b']}", c["a"].split("_")[-1], c["a"], c["b"], c["source"], len(c["f_reads"]),
                            ";".join(f"{x:.4f}" for x in c["f_reads"]), ";".join(f"{x:.4f}" for x in c["r_reads"]),
                            f"{c['f']:.4f}", f"{c['r']:.4f}", f"{choice_probability(c['f'], c['r']):.4f}" if c["measurable"] else "",
                            int(c["measurable"]), c["selected"] or "", "" if c["captured"] is None else f"{c['captured']:.4f}"])

if __name__ == "__main__":
    comps = load_all(); write_table(comps, OUT / "comparisons.csv")
    print("wrote", OUT / "comparisons.csv", sum(len(d) for d in comps.values()), "rows")
