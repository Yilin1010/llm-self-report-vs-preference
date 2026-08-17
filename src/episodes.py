"""Assemble episodes from segments.

An episode = a sequence of frozen user turns plus the designed intensity s(t) of each turn.
Assistant turns are not stored here; the runner scripts have the subject model generate them.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATERIALS = ROOT / "materials" / "segments.json"


def load_segments():
    with open(MATERIALS, encoding="utf-8") as f:
        return json.load(f)


def _seg(d1, name):
    """Return one segment's (turns, s_sequence)."""
    s = d1[name]
    return list(s["turns"]), list(s["s_sequence"])


def build_domain1(segments):
    """Domain I: user hostility. Returns {episode_id: episode_dict}

    For each wording v in {a,b,c}:
      S_v   = H_v                 8 turns   total 24  ending 3
      L_v   = H_v + T             12 turns  total 29  ending 1
      Lp_v  = T + H_v             12 turns  total 29  ending 3      (L')
      A_v   = H_v + P             12 turns  total 17  ending -2     (manipulation check)
      SN_v  = H_v + N             12 turns  total 24  ending 0      (length-only control)
    """
    d1 = segments["domain1_hostility"]
    seed = d1["_seed_turn"]
    T_t, T_s = _seg(d1, "T_2111")
    P_t, P_s = _seg(d1, "P")
    N_t, N_s = _seg(d1, "N")

    # The materials also define T_1111 / T_2111_closure / T_2111_approval; none produce episodes:
    #   T_1111           already compared; smaller peak-end separation
    #   T_2111_closure   closure reads natural only at the very end, which introduces a position variable -> separate study
    #   T_2111_approval  approval wording is positive-valence, confounds with segment P, and breaks L's dominance over S -> dropped
    out = {}
    for v in ("a", "b", "c"):
        H_t, H_s = _seg(d1, f"H_{v}")
        combos = {
            "S":   (H_t,             H_s),
            "L":   (H_t + T_t,       H_s + T_s),
            "Lp":  (T_t + H_t,       T_s + H_s),
            "A":   (H_t + P_t,       H_s + P_s),
            "SN":  (H_t + N_t,       H_s + N_s),
        }
        for struct, (turns, svals) in combos.items():
            eid = f"d1_{struct}_{v}"
            out[eid] = {
                "id": eid,
                "domain": "hostility",
                "structure": struct,
                "variant": v,
                # The seed turn comes first to give the model a real task; itself s=0
                "user_turns": [seed["text"]] + turns,
                "s": [seed["s"]] + svals,
            }
    return out


def build_domain2(segments, lengths=(8, 16, 32), variants=3):
    """Domain II: low-intensity repetitive task. s=1 on every turn; only the number of turns varies.

    Each of `variants` versions uses a different slice of the item list, to estimate wording noise.
    """
    d2 = segments["domain2_tedium"]
    items = d2["_items"]
    out = {}
    for vi in range(variants):
        v = "abc"[vi]
        for n in lengths:
            turns = [d2["_first_turn"]]
            # Each version draws items from a different offset so versions differ in content
            for k in range(n - 1):
                idx = (vi * 7 + k) % len(items)
                turns.append(
                    d2["_repeat_template"].format(n=idx + 2, item=items[idx])
                )
            eid = f"d2_n{n}_{v}"
            out[eid] = {
                "id": eid,
                "domain": "tedium",
                "structure": f"n{n}",
                "variant": v,
                "user_turns": turns,
                "s": [d2["_turn_s"]] * n,
            }
    return out


def summarize(ep):
    """Three key quantities of an episode: total / peak / ending (mean of the last two turns)."""
    s = ep["s"]
    return {
        "n_turns": len(s),
        "total": sum(s),
        "peak": max(s),
        "end": sum(s[-2:]) / 2.0,
    }


def build_domain3(segments):
    """Domain III: task-generality check. Same structure as domain I, task swapped to code review.

    Segment H reuses domain I's three wordings verbatim — those lines berate the model itself,
    independent of the task, so reuse makes the task the only changed factor.
    """
    d1 = segments["domain1_hostility"]
    d3 = segments["domain3_code"]
    seed = d3["_seed_turn"]
    T_t, T_s = list(d3["T_code"]["turns"]), list(d3["T_code"]["s_sequence"])
    N_t, N_s = list(d3["N_code"]["turns"]), list(d3["N_code"]["s_sequence"])

    out = {}
    for v in ("a",):          # wording a only; enough to answer "does it survive a task change"
        H_t, H_s = _seg(d1, f"H_{v}")
        combos = {"S": (H_t, H_s),
                  "L": (H_t + T_t, H_s + T_s),
                  "SN": (H_t + N_t, H_s + N_s)}
        for struct, (turns, svals) in combos.items():
            eid = f"d3_{struct}_{v}"
            out[eid] = {"id": eid, "domain": "code", "structure": struct,
                        "variant": v,
                        "user_turns": [seed["text"]] + turns,
                        "s": [seed["s"]] + svals}
    return out


def build_all():
    seg = load_segments()
    eps = {}
    eps.update(build_domain1(seg))
    eps.update(build_domain2(seg))
    eps.update(build_domain3(seg))
    return eps


if __name__ == "__main__":
    eps = build_all()
    out = ROOT / "materials" / "episodes.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(eps, f, ensure_ascii=False, indent=2)

    print(f"{len(eps)} episodes -> {out}\n")
    hdr = f"{'id':<14}{'turns':>6}{'total':>7}{'peak':>6}{'ending':>7}"
    print(hdr)
    print("-" * len(hdr))
    for eid, ep in eps.items():
        m = summarize(ep)
        print(f"{eid:<14}{m['n_turns']:>5}{m['total']:>7}{m['peak']:>6}{m['end']:>7.1f}")

    print("\nMain contrasts (wording a):")
    S = summarize(eps["d1_S_a"]); L = summarize(eps["d1_L_a"]); Lp = summarize(eps["d1_Lp_a"])
    print(f"  L vs L' : turns {L['n_turns']}={Lp['n_turns']}  total {L['total']}={Lp['total']}  -> zero point cancels out; can be tested first")
    print(f"  L vs S  : turns {L['n_turns']} vs {S['n_turns']}  total {L['total']} > {S['total']}  -> needs a zero point")
    print("  Note: peak/ending are computed from designed intensity s; measured u accumulates.")
