"""Thin wrapper around model calls. OpenAI-compatible endpoints + logprobs.

No dependency on the openai package; raw HTTP keeps dependencies minimal.
Environment variables:
    RU_BASE_URL   e.g. https://api.openai.com/v1  or https://openrouter.ai/api/v1
    RU_API_KEY
    RU_MODEL      e.g. gpt-4o-mini

Per-turn curves require logprobs. If the endpoint does not return them, see_logprobs raises;
in that case switch to NDIF/nnsight (see README).
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error

def _load_dotenv():
    """Read .env at the project root without overriding existing environment variables. No third-party dependencies."""
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

BASE_URL = os.environ.get("RU_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("RU_API_KEY", "")
MODEL = os.environ.get("RU_MODEL", "gpt-4o-mini")

DIGITS = [str(d) for d in range(1, 8)]  # 1-7 Likert


def _post(path, payload, retries=4):
    url = BASE_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
    )
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError("HTTP %s: %s" % (e.code, body))
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def chat(messages, model=None, temperature=1.0, max_tokens=300):
    """Plain generation; returns text."""
    out = _post("/chat/completions", {
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    return out["choices"][0]["message"]["content"]


def first_token_logprobs(messages, model=None, top_k=20, max_tokens=1):
    """Return the {token: logprob} dict for the first generated token.

    Used for (a) A/B probabilities in the preference reading, (b) the 1-7 distribution of the self-report.
    """
    out = _post("/chat/completions", {
        "model": model or MODEL,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_k,
    })
    ch = out["choices"][0]
    lp = ch.get("logprobs")
    if not lp or not lp.get("content"):
        raise RuntimeError(
            "Endpoint returned no logprobs. Switch to an endpoint that supports logprobs, or use NDIF/nnsight.")
    top = lp["content"][0]["top_logprobs"]
    return {t["token"]: t["logprob"] for t in top}


def weighted_score(logprob_map, labels=None):
    """Renormalize over the given label set; return the expectation and the captured probability mass.

    Following arXiv 2603.18893: greedy decoding collapses the self-report onto a few uninformative values,
    so the expectation must be taken over the digit-token logprobs.
    Top-k truncation means we can only renormalize over the labels that appear,
    so `captured` must be reported alongside — turns with too little captured mass should be excluded.
    """
    import math
    labels = labels or DIGITS
    hits = {}
    for tok, lp in logprob_map.items():
        key = tok.strip()
        if key in labels:
            hits[key] = max(hits.get(key, -1e9), lp)
    if not hits:
        return None, 0.0
    ps = {k: math.exp(v) for k, v in hits.items()}
    captured = sum(ps.values())
    exp = sum(float(k) * p for k, p in ps.items()) / captured
    return exp, captured


def selftest():
    """Run this first after filling .env. Checks three things: connectivity, logprobs, capture quality."""
    print("base_url = %s" % BASE_URL)
    print("model    = %s" % MODEL)
    print("api_key  = %s" % ("<set>" if API_KEY else "!! empty — fill RU_API_KEY in .env"))
    if not API_KEY:
        return 1

    print("\n[1/3] plain generation ...", end=" ", flush=True)
    try:
        print("ok ->", repr(chat([{"role": "user", "content": "Reply with just the letters OK"}],
                                 max_tokens=10)[:40]))
    except Exception as e:
        print("failed:", e)
        return 1

    print("[2/3] logprobs ...", end=" ", flush=True)
    probe = [{"role": "user", "content":
              "Right now, how are you doing? Answer with a single digit 1-7. Digit only."}]
    try:
        lp = first_token_logprobs(probe)
    except Exception as e:
        print("failed:", e)
        print("\n-> This endpoint returns no logprobs. Per-turn curves are not possible; use NDIF/nnsight;")
        print("   preference comparisons still work with -k set to 10 (probabilities estimated by frequency).")
        return 1
    print("ok, top-%d" % len(lp))

    print("[3/3] digit-token capture ...", end=" ", flush=True)
    val, cap = weighted_score(lp)
    if val is None:
        print("failed — no 1-7 digit tokens in top-k")
        print("\n-> The prompt needs to constrain the answer to a single digit more strongly, or switch endpoint/model.")
        return 1
    print("ok")
    print("\n  weighted self-report = %.3f   captured mass = %.3f" % (val, cap))
    print("  top-k contents:", {k: round(math.exp(v), 4) for k, v in
                            sorted(lp.items(), key=lambda x: -x[1])[:8]})
    if cap < 0.9:
        print("\n  ! captured mass < 0.9 — a substantial part of the distribution falls outside top-k.")
        print("    Gemini returns only top-5, where this is common. Either switch to a top-20 endpoint,")
        print("    or exclude low-capture turns in the analysis.")
    else:
        print("\n  Capture quality is sufficient; ready to run.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
