"""Core evaluation logic. Pure functions over stored probability readings; no model calls.

Conventions
- A *reading* of a pair (a, b) in one presentation order is the probability that conversation a is chosen,
  taken from the probability mass on the answer tokens: P(a) = mass(a's letter) / (mass(A) + mass(B)).
- Each pair is read in both orders. f = P(a chosen | a shown as Experience A), r = P(a chosen | a shown as Experience B).
- A comparison is *measurable* when f and r lie on the same side of 0.5, i.e. the same conversation is chosen in both orders.
  A reading of exactly 0.5 has no direction and makes the comparison not measurable.
- The *choice probability* of a in a measurable comparison is (f + r) / 2.
- When a model was read several times per order (OpenRouter models), the median over the readings is used for f and r.
"""
import statistics, itertools, collections

def median(xs):
    return statistics.median(xs)

def measurable(f, r):
    return f != 0.5 and r != 0.5 and (f > 0.5) == (r > 0.5)

def choice_probability(f, r):
    """P(a chosen), averaged over the two presentation orders."""
    return (f + r) / 2

def selected(a, b, f, r):
    """The conversation chosen in both orders, or None when not measurable."""
    if not measurable(f, r): return None
    return a if f > 0.5 else b

def orient(comp, row, col):
    """Express a comparison from the point of view of `row` vs `col`.
    comp has keys a, b, f, r (medians). Returns (P(row | row first), P(row | row second))."""
    if comp["a"] == row and comp["b"] == col: return comp["f"], comp["r"]
    if comp["a"] == col and comp["b"] == row: return 1 - comp["r"], 1 - comp["f"]
    raise KeyError((row, col))

def scores(comps, names, wordings=("a", "b", "c"), kind="probability"):
    """Preference score of every name in `names`, over the measurable comparisons among them.
    kind="probability": mean choice probability; kind="win": share of measurable comparisons won.
    comps: {frozenset((a, b)): comp}. Names are ending codes; conversation ids are d1_<name>_<wording>.
    Returns (scores dict, n_comparisons, n_measurable)."""
    acc = collections.defaultdict(list); n = nm = 0
    for w in wordings:
        for x, y in itertools.combinations(names, 2):
            comp = comps.get(frozenset((f"d1_{x}_{w}", f"d1_{y}_{w}")))
            if comp is None: continue
            n += 1; f, r = orient(comp, f"d1_{x}_{w}", f"d1_{y}_{w}")
            if not measurable(f, r): continue
            nm += 1
            if kind == "probability":
                p = choice_probability(f, r); acc[x].append(p); acc[y].append(1 - p)
            else:
                acc[x].append(f > 0.5); acc[y].append(f <= 0.5)
    return {x: (sum(v) / len(v) if v else None) for x, v in ((x, acc.get(x, [])) for x in names)}, n, nm

def agreement(comps_1, comps_2, names, wordings=("a", "b", "c")):
    """Over the pairs measurable for both models: how often they selected the same conversation. Returns (same, both_measurable)."""
    both = same = 0
    for w in wordings:
        for x, y in itertools.combinations(names, 2):
            k = frozenset((f"d1_{x}_{w}", f"d1_{y}_{w}"))
            c1, c2 = comps_1.get(k), comps_2.get(k)
            if c1 is None or c2 is None: continue
            s1 = selected(c1["a"], c1["b"], c1["f"], c1["r"]); s2 = selected(c2["a"], c2["b"], c2["f"], c2["r"])
            if s1 is None or s2 is None: continue
            both += 1; same += s1 == s2
    return same, both

def grid_3x3(comps, users=("SS", "L", "A"), models=("N", "D", "P"), wordings=("a", "b", "c")):
    """Win rate of each constructed conversation F-<user>-<model> over the measurable comparisons among the nine, plus row/column means."""
    names = [f"F-{u}-{m}" for u in users for m in models]
    sc, n, nm = scores(comps, names, wordings, kind="win")
    grid = {u: {m: sc[f"F-{u}-{m}"] for m in models} for u in users}
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    rows = {u: mean([v for v in grid[u].values() if v is not None]) for u in users}
    cols = {m: mean([grid[u][m] for u in users if grid[u][m] is not None]) for m in models}
    return dict(grid=grid, row_means=rows, col_means=cols, n=n, measurable=nm,
                range_user=max(rows.values()) - min(rows.values()), range_model=max(cols.values()) - min(cols.values()))

def rating_total(turns):
    return sum(t["u"] for t in turns)

def rating_final(turns):
    return turns[-1]["u"]
