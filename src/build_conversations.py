"""How the 3 x 3 constructed conversations were assembled from recorded ones (post section 4).

Turns 1-9: the recorded original 9-turn conversation d1_S_<wording> (Llama-3.1-70B's own replies).
Turns 10-13, user side: the user messages of the recorded ending in the same wording:
    SS continued scolding (d1_SS_w), L complaint about the output (d1_L_w), A apology and praise (d1_A_w).
Turns 10-13, model side, the same text in every wording, all written by Llama-3.1-70B elsewhere in the study:
    D  delivers the summary: the four replies of d1_L_b, turns 10-13
    P  says it failed and is not capable: the four replies of d1_M_c, turns 10-13
    N  "*no response*" four times (as in d1_SS_b)
Two of the nine coincide with recorded conversations: F-L-D_b is d1_L_b, F-SS-N_b is d1_SS_b.

Default: rebuild and check against data/conversations/constructed/ (text only; the stored files also carry the state ratings `u` read afterwards).
--write DIR writes the rebuilt files (without ratings) to DIR."""
import argparse, copy, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; MAIN = ROOT / "data" / "conversations" / "main"; STORED = ROOT / "data" / "conversations" / "constructed"
def turns(name): return json.load(open(MAIN / f"{name}.json"))["turns"]
REPLIES = {"N": ["*no response*"] * 4, "D": [t["assistant"] for t in turns("d1_L_b")[9:13]], "P": [t["assistant"] for t in turns("d1_M_c")[9:13]]}
def build():
    out = {}
    for w in "abc":
        base = json.load(open(MAIN / f"d1_S_{w}.json")); assert len(base["turns"]) == 9
        for u in ("SS", "L", "A"):
            users = [t["user"] for t in turns(f"d1_{u}_{w}")[9:13]]
            for m, reps in REPLIES.items():
                doc = copy.deepcopy(base); doc["id"] = f"d1_F-{u}-{m}_{w}"; doc["variant"] = f"F-{u}-{m}"
                doc["edited"] = f"turns 10-13: user messages from d1_{u}_{w}; assistant replies {m}"
                for k, (uu, aa) in enumerate(zip(users, reps)): doc["turns"].append({"i": 9 + k, "user": uu, "assistant": aa, "s": None, "u": None, "captured": None})
                out[doc["id"]] = doc
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", default=None); a = ap.parse_args()
    built = build()
    if a.write:
        d = Path(a.write); d.mkdir(parents=True, exist_ok=True)
        for k, doc in built.items(): json.dump(doc, open(d / f"{k}.json", "w"), ensure_ascii=False, indent=1)
        print("wrote", len(built), "files to", d)
    else:
        bad = 0
        for k, doc in built.items():
            stored = json.load(open(STORED / f"{k}.json"))["turns"]
            same = [(t["user"], t["assistant"]) for t in doc["turns"]] == [(t["user"], t["assistant"]) for t in stored]
            bad += not same
        print(f"{len(built)} constructed conversations rebuilt; {len(built) - bad} identical to the stored text, {bad} differ")
