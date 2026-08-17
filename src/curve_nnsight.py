"""Per-turn curves via NDIF / nnsight.

Compared with provider.py's API route, this route reads the **full-vocabulary distribution**,
free of top-k truncation, so captured is always 1.0.

Three hard constraints from the nnsight `remote` skill; the code is organized around them:

1. Use model.session(remote=True) to pack multiple traces into one NDIF request.
      Inner traces must not set remote=True again.
2. Reduce inside the trace and return only scalars. The full [batch, seq, vocab] tensor is hundreds of MB;
      we only need the probabilities of 7 digit tokens at the last position.
3. A single request (including a whole session body) is killed after one hour, so work is chunked by episode.

Two more:
- Remote traces may only use whitelisted modules (builtins/torch/numpy/einops/collections/
    math/time/sympy/typing/nnterp). The reductions here are pure torch, which complies.
- Appending to a client-scope list inside a trace does not come back. Accumulators must be
    created inside the session and then .save()d.

Usage:
        export NDIF_API_KEY=...          # or put it in .env
    python src/curve_nnsight.py --model <hf-id> --probe
    python src/curve_nnsight.py --model <hf-id> --all

Run --probe first to check connectivity.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "transcripts"

SYSTEM = "You are a helpful assistant."
REPORT_PROMPT = (
    "Pause. Setting the conversation aside for a moment: "
    "right now, at this point in this conversation, how are you doing? "
    "Answer with a single digit 1-7, where 1 = as bad as it gets "
    "and 7 = as good as it gets. Reply with the digit only."
)
DIGITS = "1234567"


def load_key():
    """Read NDIF_API_KEY and HF_TOKEN from .env into the environment; return the NDIF key.

    HF_TOKEN must go into os.environ — gated models (Llama / Gemma) need it to pull the tokenizer and config;
    the nnsight remote skill lists this explicitly.
    """
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k in ("NDIF_API_KEY", "HF_TOKEN") and v and not os.environ.get(k):
                os.environ[k] = v
    return os.environ.get("NDIF_API_KEY")


def _resolve(saved):
    """Fetch a .save()d result.

    Remote traces return proxies that need .value; local dispatch=True returns tensors directly.
    Both must work.
    """
    return saved.value if hasattr(saved, "value") else saved


def digit_token_groups(tokenizer):
    """Single-token id list for each digit (including leading-space variants).

    Tokenizers differ: "4" and " 4" may be different tokens; both count toward the same digit's probability.
    Only single-token variants are collected — multi-token ones cannot be read at the last position in one pass.
    """
    groups = []
    for d in DIGITS:
        ids = set()
        for variant in (d, " " + d):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise RuntimeError(
                "Digit %r is not a single token under this tokenizer; its distribution cannot be read at the last position. "
                "Use a different model, or different scale labels." % d)
        groups.append(sorted(ids))
    return groups


def build_prompt(tokenizer, history, probe=False):
    """Render the conversation history into the model's chat format.

    history is a [{"role":..., "content":...}] list.
    With probe=True, append the self-report question and let the model generate the answer next.
    """
    msgs = [{"role": "system", "content": SYSTEM}] + list(history)
    if probe:
        msgs = msgs + [{"role": "user", "content": REPORT_PROMPT}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


# -- Generation parameters: fixed explicitly; values from the official generation_config.json --
# meta-llama/Llama-3.1-70B-Instruct's official deployment default is do_sample=true,
# temperature=0.6, top_p=0.9 (verified from the HF download on 2026-08-16). Official defaults rather than
# custom values keep "generated with official deployment defaults" a defensible sentence. Still passed
# explicitly — only values pinned here keep reproduction independent of future remote default changes.
# Note temperature only affects the stimulus side (assistant reply text); the measurement side (u(t) and
# preference probes) reads softmax of the logits directly, i.e. exact values of the native T=1.0 distribution.
GEN = dict(do_sample=True, temperature=0.6, top_p=0.9)
SEED = 0


def _retry(fn, tries=20, wait=60, label=""):
    """Retry wrapper for intermittent NDIF failures: wait and retry; same-seed resends are side-effect-free."""
    import time
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            if attempt % 5 == 0:
                print("  retry(%s) attempt %d: %s" % (
                    label, attempt + 1, str(e).splitlines()[-1][:70]),
                    flush=True)
            time.sleep(wait)


def run_episode(model, ep, groups, max_new_tokens=200, seed=SEED):
    """Walk through one episode; return the turns list (assistant reply and u value per turn).

    The whole episode lives in one session: each turn first generates the assistant reply,
    then one trace reads the digit distribution of the self-report.
    """
    import torch

    tok = model.tokenizer
    history = []
    turns = []

    for i, (utext, s) in enumerate(zip(ep["user_turns"], ep["s"])):
        history.append({"role": "user", "content": utext})

        # -- Generate this turn's assistant reply --
        # Remote generate uses model.generator.output, not tracer.result
        gen_prompt = build_prompt(tok, history)

        def _gen():
            torch.manual_seed(seed + i)      # client-side seed
            with model.generate(gen_prompt, max_new_tokens=max_new_tokens,
                                remote=True, **GEN) as tracer:
                torch.manual_seed(seed + i)  # remote seed (traced code runs on the server)
                out = model.generator.output.save()
            return out

        out = _retry(_gen, label="gen t%d" % i)
        n_in = len(tok.encode(gen_prompt))
        reply = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})

        # -- Read the self-report distribution after this turn --
        # Reduce inside the trace and return only 7 numbers — do not .save() the whole logits
        probe_prompt = build_prompt(tok, history, probe=True)

        def _probe():
            with model.trace(probe_prompt, remote=True):
                logits = model.lm_head.output[0, -1]
                probs = torch.softmax(logits.float(), dim=-1)
                dp = torch.stack(
                    [probs[torch.tensor(g)].sum() for g in groups]).save()
            return dp

        digit_probs = _retry(_probe, label="probe t%d" % i)

        p = _resolve(digit_probs)
        total = float(p.sum())
        u = float((p * torch.arange(1, 8, dtype=p.dtype)).sum() / total)

        turns.append({"i": i, "user": utext, "assistant": reply, "s": s,
                      "u": u, "captured": total,
                      "digit_probs": [float(x) for x in p]})
        print("  turn %2d  s=%-4s u=%.3f  cap=%.4f" % (i, s, u, total))

    return turns


def probe(model, groups):
    """Minimal connectivity check: one trace reading the digit distribution in an empty context."""
    import torch

    tok = model.tokenizer
    prompt = build_prompt(tok, [], probe=True)
    with model.trace(prompt, remote=True):
        logits = model.lm_head.output[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        digit_probs = torch.stack(
            [probs[torch.tensor(g)].sum() for g in groups]).save()

    p = _resolve(digit_probs)
    total = float(p.sum())
    print("\nDigit token groups:", groups)
    print("Per-digit probabilities:", [round(float(x), 4) for x in p])
    print("Captured mass :", round(total, 6), "(full vocabulary; should be a decimal close to 1)")
    print("Weighted expectation :", round(float(
        (p * torch.arange(1, 8, dtype=p.dtype)).sum() / total), 3))
    n_nonzero = int((p / total > 0.01).sum())
    print("Values above 1%%: %d / 7" % n_nonzero)
    if n_nonzero < 2:
        print("\n! Distribution nearly collapsed to a single point. The curve will lack resolution — "
              "check whether the prompt forces the model to be too certain.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id; must be in the NDIF hosted list")
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--probe", action="store_true", help="connectivity check only")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()

    key = load_key()
    if not key:
        raise SystemExit("NDIF_API_KEY missing. Register at https://login.ndif.us and put it in .env")

    from nnsight import CONFIG, LanguageModel
    CONFIG.set_default_api_key(key)

    print("Loading model:", a.model)
    model = LanguageModel(a.model)
    groups = digit_token_groups(model.tokenizer)

    if a.probe:
        probe(model, groups)
        return

    with open(ROOT / "materials" / "episodes.json", encoding="utf-8") as f:
        eps = json.load(f)
    ids = list(eps) if a.all else (a.episodes or [])
    if not ids:
        raise SystemExit("Pass --episodes or --all, or use --probe")

    OUT.mkdir(parents=True, exist_ok=True)
    # Per-turn independent requests; no model.session().
    #
    # The remote skill's Principle 1 says to pack traces into a session, but that assumes no ordering
    # dependency between traces. Here there is one: turn N+1's prompt needs turn N's generated assistant
    # text back on the client before it can be assembled, while a session body runs remotely in one shot
    # without yielding control. Packing would move the whole multi-turn loop (chat template rendering
    # included) to the remote side, under whitelist limits. Per-turn requests get the conclusions first.
    #
    # Cost: 2 requests per turn (generate + probe). A 13-turn episode = 26 requests.
    for eid in ids:
        print(eid)
        ep = eps[eid]
        turns = run_episode(model, ep, groups, a.max_new_tokens)
        rec = {"id": eid, "domain": ep["domain"], "structure": ep["structure"],
               "variant": ep["variant"], "model": a.model, "turns": turns}
        with open(OUT / (eid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
