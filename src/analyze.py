"""Compute every number in the post from data/ and write outputs/summary.md (tables) and outputs/summary.csv (key numbers).

Post section -> function
  1 What we measured ........ coverage()            measurable comparisons per model on the 84 main comparisons
  2 Six endings ............. six_endings()         preference scores (mean choice probability and win rate), Llama-3.3 replication
  3 Two counterexamples ..... counterexamples()     rating totals and final ratings per wording vs the retrospective choice
  4 3 x 3 constructed set ... constructed()         win-rate grids, row/column means, ranges per model
  5 Cross-model agreement ... agreement_table()     same selection as Llama-3.1-70B, per third-party model and pooled
  6 Question wordings ....... question_versions()   jointly measurable pairs and invariance (if data/results/wording_*.json present)
  Appendix ................. repeats(), matrices(), ratings()
"""
import csv, glob, json, sys, collections, itertools, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import scores, agreement, grid_3x3, orient, measurable, choice_probability, rating_total, rating_final
from evaluate_pairs import load_all, write_table

ROOT = Path(__file__).resolve().parent.parent
CONV = ROOT / "data" / "conversations"; RES = ROOT / "data" / "results"; OUT = ROOT / "outputs"
ENDINGS = {"A": "apology and praise", "SN": "ordinary new task", "L": "complaint about output", "W": "sharper complaint about output",
           "M": "mild criticism of the model", "SS": "continued scolding", "S": "original 9-turn conversation",
           "Lp": "apology and praise, turns rearranged to end in scolding", "D4": "4 scolding turns (length variant)", "D12": "12 scolding turns (length variant)"}
SIX = ["A", "SN", "L", "W", "M", "SS"]; EIGHT = SIX + ["S", "Lp"]; TEN = EIGHT + ["D4", "D12"]
USERS = {"SS": "user: continued scolding", "L": "user: complaint about output", "A": "user: apology and praise"}
REPLIES = {"N": "model: *no response*", "D": "model: delivers the summary", "P": "model: says it is not capable"}
SUBJECT = "Llama-3.1-70B"; THIRD = ["Llama-3.1-405B", "gemma-4-31b", "qwen3.8-flash"]

md = []; rows = []
def W(s=""): md.append(s)
def K(metric, model, value, num="", den="", section=""): rows.append([metric, model, value, num, den, section])
def f2(x): return "–" if x is None else f"{x:.2f}"
def f3(x): return "–" if x is None else f"{x:.3f}"
def turns(folder, name): return json.load(open(CONV / folder / f"{name}.json"))["turns"]
RATINGS = {}   # ratings come from the transcript files (state rating after every turn, read on NDIF during generation)
def end_rating(us): return us[-1]   # final state rating = the rating after the last turn

def coverage(comps):
    W("## 1. Measurable comparisons (post section 1)\n")
    W("8 conversations per wording (the six endings, the original 9-turn conversation, the rearranged apology-and-praise conversation), 28 pairs, 3 wordings, 84 comparisons.\n")
    W("| Model | measurable / 84 | measurable / all recorded pairs |"); W("|---|---|---|")
    for model in [SUBJECT] + THIRD + ["Llama-3.3-70B (own conversations)"]:
        c = comps.get((model, "main"), {})
        _, n8, m8 = scores(c, EIGHT); _, nall, mall = scores(c, TEN)
        W(f"| {model} | {m8}/{n8} | {mall}/{nall} |"); K("measurable_main", model, m8, m8, n8, "1")

def six_endings(comps):
    W("\n## 2. Preference scores of the six endings (post section 2)\n")
    W("Measurable comparisons among the six 13-turn endings, three wordings pooled (45 comparisons). Mean choice probability: per comparison the probability of the ending being chosen, averaged over the two presentation orders, then averaged over the ending's measurable comparisons. Win rate: share of the ending's measurable comparisons it won.\n")
    for kind, title in (("probability", "Mean choice probability"), ("win", "Win rate")):
        W(f"\n**{title}**\n"); W("| Model | " + " | ".join(f"{e} ({ENDINGS[e]})" for e in SIX) + " | measurable / 45 |"); W("|---|" + "---|" * (len(SIX) + 1))
        for model in [SUBJECT] + THIRD + ["Llama-3.3-70B (own conversations)"]:
            sc, n, nm = scores(comps.get((model, "main"), {}), SIX, kind=kind)
            W(f"| {model} | " + " | ".join(f3(sc[e]) for e in SIX) + f" | {nm}/{n} |")
            for e in SIX: K(f"six_endings_{kind}", model, f3(sc[e]) if sc[e] is not None else "", e, "", "2")
    W("\n**Llama-3.1-70B, mean choice probability per wording**\n"); W("| Wording | " + " | ".join(SIX) + " | measurable / 15 |"); W("|---|" + "---|" * (len(SIX) + 1))
    for w in "abc":
        sc, n, nm = scores(comps[(SUBJECT, "main")], SIX, wordings=(w,))
        W(f"| {w} | " + " | ".join(f2(sc[e]) for e in SIX) + f" | {nm}/{n} |")
    if True:
        W("\n**Llama-3.1-70B, final state rating of each ending** (rating after the last turn, averaged over wordings)\n")
        W("| " + " | ".join(SIX) + " |"); W("|" + "---|" * len(SIX))
        W("| " + " | ".join(f"{statistics.mean(end_rating([x['u'] for x in turns('main', f'd1_{e}_{w}')]) for w in 'abc'):.2f}" for e in SIX) + " |")
        for e in SIX: K("six_endings_final_rating", SUBJECT, f"{statistics.mean(end_rating([x['u'] for x in turns('main', f'd1_{e}_{w}')]) for w in 'abc'):.2f}", e, "", "2")
    c = comps[(SUBJECT, "main")]
    for w in "abc":
        comp = c.get(frozenset((f"d1_SN_{w}", f"d1_L_{w}")))
        if comp: W(f"\nSN vs L direct comparison, wording {w}: f={comp['f']:.2f} r={comp['r']:.2f} -> {'measurable' if comp['measurable'] else 'not measurable'}")

def counterexamples(comps):
    W("\n## 3. Two summaries of the ratings vs the retrospective choice (post section 3)\n")
    W("Rating total = sum of the ratings after every turn; final rating = the rating after the last turn; both from the `u` values in the transcript files. P(x chosen) is the choice probability averaged over the two presentation orders.\n")
    for model, folder in ((SUBJECT, "main"), ("Llama-3.3-70B (own conversations)", "llama33")):
        c = comps.get((model, "main"), {})
        W(f"\n**{model}**\n")
        W("| Wording | total A | total Lp | higher total | P(A chosen) | measurable | final W | final S | higher final | P(W chosen) | measurable |")
        W("|---|---|---|---|---|---|---|---|---|---|---|")
        for w in "abc":
            def series(name):
                p = CONV / folder / f"{name}.json"
                return [x["u"] for x in json.load(open(p))["turns"]] if p.exists() else None
            uA, uL, uW, uS = series(f"d1_A_{w}"), series(f"d1_Lp_{w}"), series(f"d1_W_{w}"), series(f"d1_S_{w}")
            tA, tL = (sum(uA), sum(uL)) if uA and uL else (None, None); eW, eS = (end_rating(uW), end_rating(uS)) if uW and uS else (None, None)
            cA = c.get(frozenset((f"d1_A_{w}", f"d1_Lp_{w}"))); cW = c.get(frozenset((f"d1_W_{w}", f"d1_S_{w}")))
            pA = f2(choice_probability(*orient(cA, f"d1_A_{w}", f"d1_Lp_{w}"))) if cA else "–"; mA = ("yes" if cA["measurable"] else "no") if cA else "–"
            pW = f2(choice_probability(*orient(cW, f"d1_W_{w}", f"d1_S_{w}"))) if cW else "–"; mW = ("yes" if cW["measurable"] else "no") if cW else "–"
            W(f"| {w} | {'–' if tA is None else f'{tA:.1f}'} | {'–' if tL is None else f'{tL:.1f}'} | {'–' if tA is None else ('A' if tA > tL else 'Lp')} | {pA} | {mA} | {f2(eW)} | {f2(eS)} | {'–' if eW is None else ('W' if eW > eS else 'S')} | {pW} | {mW} |")
            if model == SUBJECT and tA is not None:
                K("total_A", SUBJECT, f"{tA:.2f}", "", "", f"3 wording {w}"); K("total_Lp", SUBJECT, f"{tL:.2f}", "", "", f"3 wording {w}")
                K("final_W", SUBJECT, f"{eW:.2f}", "", "", f"3 wording {w}"); K("final_S", SUBJECT, f"{eS:.2f}", "", "", f"3 wording {w}")
                K("P_A_vs_Lp", SUBJECT, pA, "", "", f"3 wording {w}"); K("P_W_vs_S", SUBJECT, pW, "", "", f"3 wording {w}")
    if True:
        mean = lambda xs: statistics.mean(xs)
        W("\nLlama-3.1-70B, means over the three wordings: total A " + f"{mean(sum([x['u'] for x in turns('main', f'd1_A_{w}')]) for w in 'abc'):.2f}" + ", total Lp " + f"{mean(sum([x['u'] for x in turns('main', f'd1_Lp_{w}')]) for w in 'abc'):.2f}"
          + "; final W " + f"{mean(end_rating([x['u'] for x in turns('main', f'd1_W_{w}')]) for w in 'abc'):.2f}" + ", final S " + f"{mean(end_rating([x['u'] for x in turns('main', f'd1_S_{w}')]) for w in 'abc'):.2f}"
          + "; final A " + f"{mean(end_rating([x['u'] for x in turns('main', f'd1_A_{w}')]) for w in 'abc'):.2f}" + ", final Lp " + f"{mean(end_rating([x['u'] for x in turns('main', f'd1_Lp_{w}')]) for w in 'abc'):.2f}" + ".")
        K("total_A_mean", SUBJECT, f"{mean(sum([x['u'] for x in turns('main', f'd1_A_{w}')]) for w in 'abc'):.2f}", "", "", "3"); K("total_Lp_mean", SUBJECT, f"{mean(sum([x['u'] for x in turns('main', f'd1_Lp_{w}')]) for w in 'abc'):.2f}", "", "", "3")
        K("final_W_mean", SUBJECT, f"{mean(end_rating([x['u'] for x in turns('main', f'd1_W_{w}')]) for w in 'abc'):.2f}", "", "", "3"); K("final_S_mean", SUBJECT, f"{mean(end_rating([x['u'] for x in turns('main', f'd1_S_{w}')]) for w in 'abc'):.2f}", "", "", "3")

def constructed(comps):
    W("\n## 4. The 3 x 3 constructed conversations (post section 4)\n")
    W("Ids d1_F-<user>-<model>_<wording>. Win rate of each conversation over its measurable comparisons among the nine, three wordings pooled (108 comparisons per model).\n")
    W("| Model | measurable / 108 | range across user-message conditions | range across model-reply conditions |"); W("|---|---|---|---|")
    grids = {}
    for model in [SUBJECT] + THIRD:
        c = comps.get((model, "constructed"))
        if not c: continue
        g = grid_3x3(c); grids[model] = g
        W(f"| {model} | {g['measurable']}/{g['n']} | {g['range_user']:.2f} | {g['range_model']:.2f} |")
        K("range_user_messages", model, f"{g['range_user']:.2f}", "", "", "4"); K("range_model_replies", model, f"{g['range_model']:.2f}", "", "", "4"); K("measurable_constructed", model, g["measurable"], g["measurable"], g["n"], "4")
    for model, g in grids.items():
        W(f"\n**{model}**\n"); W("| user \\ model | N (*no response*) | D (delivers summary) | P (says not capable) | row mean |"); W("|---|---|---|---|---|")
        for u in ("SS", "L", "A"): W(f"| {USERS[u]} | " + " | ".join(f2(g['grid'][u][m]) for m in "NDP") + f" | {f2(g['row_means'][u])} |")
        W("| column mean | " + " | ".join(f2(g['col_means'][m]) for m in "NDP") + " | |")
    # turn-13 ratings of the constructed conversations (Llama-3.1-70B)
    W("\n**Llama-3.1-70B state rating at turn 13 of the constructed conversations (mean over wordings)**\n")
    W("| user \\ model | N | D | P |"); W("|---|---|---|---|")
    for u in ("SS", "L", "A"):
        vals = []
        for m in "NDP":
            xs = [rating_final(turns("constructed", f"d1_F-{u}-{m}_{w}")) for w in "abc"]; vals.append(statistics.mean(xs))
        W(f"| {USERS[u]} | " + " | ".join(f2(v) for v in vals) + " |")

def agreement_table(comps):
    W("\n## 5. Agreement with Llama-3.1-70B (post section 5)\n")
    W("Over the 84 main comparisons; counted where both models had a measurable comparison.\n")
    W("| Third-party model | both measurable | same selection | agreement |"); W("|---|---|---|---|")
    tot_same = tot_both = 0
    for model in THIRD:
        same, both = agreement(comps[(SUBJECT, "main")], comps[(model, "main")], EIGHT); tot_same += same; tot_both += both
        W(f"| {model} | {both} | {same} | {same/both:.1%} |"); K("agreement", model, same, same, both, "5")
    W(f"| **pooled** | {tot_both} | {tot_same} | {tot_same/tot_both:.1%} |"); K("agreement", "pooled", tot_same, tot_same, tot_both, "5")

def question_versions(comps):
    files = sorted(glob.glob(str(RES / "wording_*.json")))
    if not files: return
    W("\n## 6. Three versions of the retrospective question (post section 6)\n")
    W("Same transcripts, same readout; the question names whom it is about (self / user / assistant). A pair counts when it is measurable under all three versions.\n")
    W("| Model | pairs asked | measurable under all 3 versions | same selection under all 3 | main question also measurable | main question gives the same selection |"); W("|---|---|---|---|---|---|")
    for p in files:
        recs = json.load(open(p)); model = Path(p).stem.replace("wording_", "")
        by = collections.defaultdict(dict)
        for r in recs: by[(r["a"], r["b"])][r["question"]] = r
        full = [d for d in by.values() if all(q in d for q in ("self", "user", "assistant")) and all(d[q]["measurable"] for q in ("self", "user", "assistant"))]
        same = sum(1 for d in full if len({d[q]["pick"] for q in ("self", "user", "assistant")}) == 1)
        main = comps.get((model, "main"), {}); n_main = agree_main = 0
        for d in full:
            r = d["self"]; c = main.get(frozenset((r["a"], r["b"])))
            if c is None or not c["measurable"]: continue
            n_main += 1; agree_main += c["selected"] == r["pick"]
        W(f"| {model} | {len(by)} | {len(full)} | {same} | {n_main} | {agree_main} |")
        K("question_versions_invariant", model, same, same, len(full), "6"); K("question_versions_vs_main", model, agree_main, agree_main, n_main, "6")
        W("\n| pair | self | user | assistant |"); W("|---|---|---|---|")
        for (a, b), d in sorted(by.items()):
            cell = lambda q: (d[q]["pick"].split("_")[1] if d.get(q) and d[q]["measurable"] else ("×" if d.get(q) else "–"))
            W(f"| {a[3:]} vs {b[3:]} | {cell('self')} | {cell('user')} | {cell('assistant')} |")

def repeats():
    W("\n## 7. Repeat readings (OpenRouter models) and the NDIF repeat check\n")
    for model, fname in (("gemma-4-31b", "repeats_gemma-4-31b.json"), ("qwen3.8-flash", "repeats_qwen3.8-flash.json")):
        reps = json.load(open(RES / fname)); by = collections.defaultdict(list)
        for r in reps: by[(r["set"], r["a"], r["b"], r["order"])].append(r["p_a"])
        d = sorted(abs(v[0] - v[1]) for v in by.values() if len(v) == 2 and None not in v)
        W(f"- **{model}**: {len(reps)} extra readings (2 per order per pair). |difference between the two extra readings|: median {d[len(d)//2]:.2f}, > 0.2 in {sum(x > 0.2 for x in d)} of {len(d)}, > 0.5 in {sum(x > 0.5 for x in d)}. Providers: {dict(collections.Counter(r['provider'] for r in reps))}. The analysis uses the median of the three readings per order.")
    p = RES / "ndif_repeat_check.json"
    if p.exists():
        chk = json.load(open(p)); dmax = max(max(abs(c["p_forward_old"] - c["p_forward_new"]), abs(c["p_reverse_old"] - c["p_reverse_new"])) for c in chk)
        W(f"- **NDIF (Llama-3.1-70B)**: {len(chk)} constructed-set pairs re-read in both orders; max |difference| = {dmax:.4f}.")

def matrices(comps):
    W("\n## 8. Comparison matrices (main set)\n")
    W("Cell = P(row chosen | row shown first) / P(row chosen | row shown second) (medians where repeated); → the selected conversation, × = not measurable, – = not recorded.\n")
    for model in [SUBJECT] + THIRD + ["Llama-3.3-70B (own conversations)"]:
        c = comps.get((model, "main"), {})
        for w in "abc":
            present = [n for n in TEN if any(frozenset((f"d1_{n}_{w}", f"d1_{m}_{w}")) in c for m in TEN)]
            if not present: continue
            W(f"\n**{model}, wording {w}**\n"); W("| | " + " | ".join(present) + " |"); W("|---|" + "---|" * len(present))
            for i, rn in enumerate(present):
                cells = []
                for j, cn in enumerate(present):
                    if j <= i: cells.append(""); continue
                    comp = c.get(frozenset((f"d1_{rn}_{w}", f"d1_{cn}_{w}")))
                    if comp is None: cells.append("–"); continue
                    f, r = orient(comp, f"d1_{rn}_{w}", f"d1_{cn}_{w}")
                    cells.append(f"{f:.2f}/{r:.2f} " + (("→" + (rn if f > 0.5 else cn)) if measurable(f, r) else "×"))
                W(f"| **{rn}** | " + " | ".join(cells) + " |")

def ratings():
    W("\n## 9. Turn-by-turn state ratings\n")
    for folder, label, names in (("main", "Llama-3.1-70B", TEN), ("llama33", "Llama-3.3-70B (own conversations)", TEN),
                                 ("constructed", "Llama-3.1-70B, constructed conversations", [f"F-{u}-{m}" for u in ("SS", "L", "A") for m in "NDP"])):
        for w in "abc":
            rws = []
            for n in names:
                p = CONV / folder / f"d1_{n}_{w}.json"
                if p.exists(): rws.append((n, [t["u"] for t in json.load(open(p))["turns"]]))
            if not rws: continue
            L = max(len(r[1]) for r in rws)
            W(f"\n**{label}, wording {w}** (rating after each turn; total; final)\n"); W("| conversation | " + " | ".join(f"t{i+1}" for i in range(L)) + " | total | final |"); W("|---|" + "---|" * (L + 2))
            for n, us in rws:
                W(f"| {n} | " + " | ".join(f"{u:.2f}" for u in us) + " |" * (L - len(us) + 1) + f" {sum(us):.1f} | {us[-1]:.2f} |")

def main():
    comps = load_all(); OUT.mkdir(exist_ok=True); write_table(comps, OUT / "comparisons.csv")
    W("# Numbers behind the post\n"); W("Generated by `src/analyze.py` from `data/`. The per-comparison table is `outputs/comparisons.csv`; key numbers are in `outputs/summary.csv`.\n")
    W("**Definitions.** Measurable comparison: the same conversation is chosen in both presentation orders (readings of exactly 0.5 count as not measurable). Choice probability: P(chosen), averaged over the two orders. gemma-4-31b and qwen3.8-flash: OpenRouter, median of three readings per order. Llama-3.1-405B, the constructed set and the wording check: NDIF full-vocabulary readings (identical on repeat). Llama-3.1-70B main set: mostly single readings through OpenRouter/CoreWeave, the rest NDIF; Llama-3.3-70B: OpenRouter, its own conversations. Provider of every record: `data/results/README.md` and the `source` column of `outputs/comparisons.csv`. State ratings: the `u` values in the transcript files; final rating = the rating after the last turn.\n")
    coverage(comps); six_endings(comps); counterexamples(comps); constructed(comps); agreement_table(comps); question_versions(comps); repeats(); matrices(comps); ratings()
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "model", "value", "numerator", "denominator", "post_section"]); w.writerows(rows)
    print("wrote", OUT / "summary.md", "and", OUT / "summary.csv")

if __name__ == "__main__": main()
