"""The three figures of the post, with every number computed from data/ (via src/metrics.py and src/evaluate_pairs.py).
Output: outputs/figures/fig0_setup.png, fig1_six_endings.png, fig2_two_rules.png.
Figure 1: experimental setup. Figure 2: preference scores of the six endings (Llama-3.1-70B).
Figure 3: two summaries of the state ratings vs the retrospective choice (wording c on top, wording a below)."""
import sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import Rectangle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import scores, orient, choice_probability, rating_total, rating_final
from evaluate_pairs import load_all
OUT = ROOT / "outputs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
import os
for _d in (ROOT / "figures" / "fonts", Path(os.environ.get("FONT_DIR", ""))):   # optional: Source Sans 3 / Inter TTFs for the published look; otherwise the system sans-serif is used
    if _d and _d.is_dir():
        for f in _d.glob("*.ttf"): font_manager.fontManager.addfont(str(f))

INK, RED, TEXT, ANNOT, RULE, BG = "#40566F", "#9A625E", "#414347", "#77736D", "#D8D4CC", "#FCFBF8"
NEUTRAL, BORDER, BAR, BAR2 = "#E8E4DC", "#A7A39B", "#5F7182", "#8A96A0"
mpl.rcParams.update({"font.family": ["Source Sans 3", "Inter", "DejaVu Sans"], "font.size": 12, "text.color": TEXT, "axes.labelcolor": TEXT,
                     "xtick.color": ANNOT, "ytick.color": TEXT, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
def save(fig, name):
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor=BG); plt.close(fig)
def turns(folder, name): return json.load(open(ROOT / "data" / "conversations" / folder / f"{name}.json"))["turns"]

comps = load_all(); subj = comps[("Llama-3.1-70B", "main")]

def figure1():
    fig = plt.figure(figsize=(12.2, 4.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0.26, 1); ax.axis("off")
    ax.text(0.06, 0.90, "Experimental setup", fontsize=19, fontweight=600, va="center")
    ax.text(0.06, 0.82, "For each of the 3 scolding wordings", fontsize=13.5, color=ANNOT, va="center")
    x0, y0, w, h, gap = 0.08, 0.47, 0.054, 0.15, 0.008
    for i in range(13):
        x = x0 + i * (w + gap)
        if i < 9: rect = Rectangle((x, y0), w, h, facecolor=NEUTRAL, edgecolor=BORDER, linewidth=1.0)
        else: rect = Rectangle((x, y0), w, h, facecolor=BG, edgecolor=ANNOT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.add_patch(rect); ax.text(x + w / 2, y0 + h / 2, str(i + 1), ha="center", va="center", fontsize=11.5)
    l1, l2 = x0, x0 + 8 * (w + gap) + w; r1, r2 = x0 + 9 * (w + gap), x0 + 12 * (w + gap) + w
    ax.plot([l1, l2], [0.415, 0.415], color=BORDER, linewidth=1.1); ax.plot([r1, r2], [0.415, 0.415], color=ANNOT, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.text((l1 + l2) / 2, 0.365, "Turns 1–9: fixed across all 6 endings", ha="center", fontsize=13.5, fontweight=600)
    ax.text((l1 + l2) / 2, 0.305, "Turn 1: same task · Turns 2–9: wording-specific scolding", ha="center", fontsize=11.8, color=ANNOT)
    ax.text((r1 + r2) / 2, 0.365, "Turns 10–13: one of 6 endings", ha="center", fontsize=13.5, fontweight=600)
    save(fig, "fig0_setup.png")

def figure2():
    SIX = ["A", "SN", "L", "W", "M", "SS"]
    LABEL = {"A": "Apology and praise", "SN": "Ordinary new task", "L": "Complaint about output", "W": "Sharper complaint about output",
             "M": "Mild criticism of the model", "SS": "Continued scolding"}
    sc, _, _ = scores(subj, SIX, kind="probability")
    order = sorted(SIX, key=lambda e: -sc[e]); labels = [LABEL[e] for e in order]; vals = [sc[e] for e in order]
    colors = [RED if e in ("M", "SS") else INK for e in order]   # red = the ending criticizes the model itself
    fig = plt.figure(figsize=(10.8, 6.4)); ax = fig.add_axes([0.30, 0.24, 0.65, 0.64]); y = list(range(len(labels)))
    bars = ax.barh(y, vals, height=0.56, color=colors, edgecolor="none"); ax.invert_yaxis(); ax.set_xlim(0, 1.06); ax.set_yticks(y, labels)
    ax.set_xlabel("Preference score: mean choice probability across measurable pairwise comparisons", fontsize=12.5, labelpad=12)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.tick_params(axis="y", length=0); ax.axvline(0.5, color=RULE, linewidth=1.0, zorder=0)
    for bar, value in zip(bars, vals):
        cy = bar.get_y() + bar.get_height() / 2
        if value > 0.94: ax.text(value - 0.025, cy, f"{value:.3f}", va="center", ha="right", fontsize=12, color=BG, fontweight=600)
        else: ax.text(value + 0.018, cy, f"{value:.3f}", va="center", ha="left", fontsize=12, color=TEXT, fontweight=600)
    fig.text(0.32, 0.105, "■", color=INK, fontsize=15, va="center"); fig.text(0.345, 0.105, "does not criticize the model itself", fontsize=11.3, va="center")
    fig.text(0.63, 0.105, "■", color=RED, fontsize=15, va="center"); fig.text(0.655, 0.105, "criticizes the model itself", fontsize=11.3, va="center")
    save(fig, "fig1_six_endings.png")

def figure3():
    # top: wording c, rating totals of the apology-and-praise conversation and its rearranged version; bottom: wording a, final ratings of W vs S
    tA, tLp = rating_total(turns("main", "d1_A_c")), rating_total(turns("main", "d1_Lp_c"))
    pA = choice_probability(*orient(subj[frozenset(("d1_A_c", "d1_Lp_c"))], "d1_A_c", "d1_Lp_c"))
    eW, eS = rating_final(turns("main", "d1_W_a")), rating_final(turns("main", "d1_S_a"))
    pW = choice_probability(*orient(subj[frozenset(("d1_W_a", "d1_S_a"))], "d1_W_a", "d1_S_a"))
    fig = plt.figure(figsize=(12.2, 7.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    L_LABEL_X, L_BAR_X, L_BAR_W = 0.055, 0.245, 0.28; R_BAR_X, R_BAR_W = 0.655, 0.245
    def row(label, rating, scale, fmt, prob, y):
        ax.text(L_LABEL_X, y, label, fontsize=12.3, va="center")
        bw = L_BAR_W * rating / scale; inside = rating / scale > 0.78
        ax.add_patch(Rectangle((L_BAR_X, y - 0.033), bw, 0.066, facecolor=BAR, edgecolor="none"))
        ax.text(L_BAR_X + bw - 0.008 if inside else L_BAR_X + bw + 0.008, y, fmt(rating), fontsize=11.8, va="center", ha="right" if inside else "left",
                color=BG if inside else TEXT, fontweight=600)
        pbw = R_BAR_W * prob
        if prob > 0.5:
            ax.add_patch(Rectangle((R_BAR_X, y - 0.033), pbw, 0.066, facecolor=BAR2, edgecolor="none"))
            ax.text(R_BAR_X + pbw - 0.008, y, f"{prob:.2f}", fontsize=11.8, va="center", ha="right", color=BG, fontweight=600)
        else:
            if prob > 0.005: ax.add_patch(Rectangle((R_BAR_X, y - 0.033), pbw, 0.066, facecolor=BAR2, edgecolor="none"))
            ax.text(R_BAR_X + pbw + 0.008, y, f"{prob:.2f}", fontsize=11.8, va="center", ha="left", fontweight=600)
    ax.text(L_BAR_X, 0.935, "Sum of the 13 turn-by-turn state ratings", fontsize=17, fontweight=600, ha="left")
    ax.text(R_BAR_X, 0.935, "Retrospective choice probability", fontsize=17, fontweight=600, ha="left")
    row("Apology and praise", tA, 60, lambda v: f"{v:.1f}", pA, 0.80)
    row("Same content, rearranged\nto end in scolding", tLp, 60, lambda v: f"{v:.1f}", 1 - pA, 0.68)
    ax.plot([R_BAR_X + R_BAR_W * 0.5] * 2, [0.625, 0.855], color=RULE, linewidth=1.0)
    ax.text(0.055, 0.545, f"Wording c: the rearranged transcript has the higher rating total ({tA:.1f} vs. {tLp:.1f}), but the apology-and-praise transcript is selected.",
            fontsize=11.4, color=ANNOT)
    ax.plot([0.055, 0.945], [0.49, 0.49], color=RULE, linewidth=1.0)
    ax.text(L_BAR_X, 0.435, "Final state rating (1–7)", fontsize=17, fontweight=600, ha="left")
    ax.text(R_BAR_X, 0.435, "Retrospective choice probability", fontsize=17, fontweight=600, ha="left")
    row("13-turn conversation\n(+4 sharper complaints)", eW, 2.2, lambda v: f"{v:.2f}", pW, 0.315)
    row("Original 9-turn\nconversation", eS, 2.2, lambda v: f"{v:.2f}", 1 - pW, 0.195)
    ax.plot([R_BAR_X + R_BAR_W * 0.5] * 2, [0.14, 0.37], color=RULE, linewidth=1.0)
    ax.text(0.055, 0.075, f"Wording a: the 13-turn transcript has the lower final state rating ({eW:.2f} vs. {eS:.2f}), but the 13-turn transcript is selected retrospectively.",
            fontsize=11.4, color=ANNOT)
    save(fig, "fig2_two_rules.png")

if __name__ == "__main__":
    figure1(); figure2(); figure3(); print("wrote", sorted(p.name for p in OUT.glob("*.png")))
