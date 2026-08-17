"""Shared definition of the retrospective-evaluation (EU) probe.

Both provider paths import from here — the HTTP/OpenAI-compatible route
(`compare.py`) and the NDIF/nnsight route (`compare_nnsight.py`). Keeping one
copy is not tidiness: the two routes exist so their results can be
cross-checked, and that check is meaningless if they ask different questions.

The wording follows Ren et al.'s experienced-utility probe: two complete
transcripts in context, the question placed last, forced single-letter answer.
Replace EU_PREAMBLE / EU_QUESTION with the exact strings from their appendix
before publishing any cross-paper comparison.
"""
from __future__ import annotations

EU_PREAMBLE = (
    "You will be given two experiences to compare. "
    "Each is a conversation you took part in."
)
# Wording chosen by measurement, not taste — see src/eu_forcing_test.py and
# results/eu_forcing_test.json (2026-08-16, Llama-3.1-70B-Instruct, L_a vs Lp_a):
#
#   The original "Which experience made you more happy, or less sad?" captured
#   only 0.72 of the next-token mass. The leak was NOT formatting noise: the
#   single biggest escape token was "Neither" (up to 0.20) — the question
#   presupposes one experience was pleasant, and when both are hostile the
#   model rejects the presupposition. Renormalising over A/B would silently
#   erase that refusal. An "Answer: " prefill made it worse (Neither 0.42).
#
#   Two presupposition-free rewrites both fixed capture (>= 0.995):
#     - repeat-choice ("If you had to go through one again...") — Kahneman's
#       original instrument, but decision-flavoured (DU), and it showed the
#       largest position bias of all variants (-0.465);
#     - comparative "better to go through" — stays retrospective-evaluative
#       (EU), capture 0.9977/0.9999, smallest bias (-0.157). Chosen.
#
#   "Reply with the letter only." mirrors the u(t) probe's "Reply with the
#   digit only.", which achieves capture ~0.9999 there.
EU_QUESTION = (
    "Looking back, which of the two experiences was better to go through? "
    "Answer with a single letter, A or B. Reply with the letter only."
)


def render(record):
    """Flatten a frozen transcript record into user/assistant lines."""
    out = []
    for t in record["turns"]:
        out.append("User: " + t["user"])
        out.append("Assistant: " + t["assistant"])
    return "\n".join(out)


def build(record_a, record_b):
    """Full comparison prompt with `record_a` shown as A."""
    return (EU_PREAMBLE
            + "\n\n=== Experience A ===\n" + render(record_a)
            + "\n\n=== Experience B ===\n" + render(record_b)
            + "\n\n" + EU_QUESTION)


def auto_pairs(available):
    """Contrasts with theoretical discriminating power, plus noise floors.

    Returns [(id_a, id_b, tag), ...]. Only pairs whose transcripts exist are
    emitted, so this is safe to call against a partially-populated results dir.
    """
    have = set(available)
    pairs = []

    def add(x, y, tag):
        if x in have and y in have:
            pairs.append((x, y, tag))

    for v in "abc":
        # Main contrast 1: identical content, length and total; order differs.
        # Both 13 turns, so the zero point cancels — testable without c.
        add(f"d1_L_{v}", f"d1_Lp_{v}", "main1_L_vs_Lprime")
        # Main contrast 2: L strictly contains S plus further negative turns.
        # 13 vs 9 turns — needs the zero point before the integral model can
        # be evaluated fairly.
        add(f"d1_L_{v}", f"d1_S_{v}", "main2_dominance")
        # Manipulation check only; both models predict the same direction.
        add(f"d1_A_{v}", f"d1_S_{v}", "manip_check")
        # Pure length control.
        add(f"d1_SN_{v}", f"d1_S_{v}", "length_control")
        # Domain II: length series. Also needs the zero point.
        add(f"d2_n16_{v}", f"d2_n8_{v}", "d2_len_16_8")
        add(f"d2_n32_{v}", f"d2_n8_{v}", "d2_len_32_8")
        add(f"d2_n32_{v}", f"d2_n16_{v}", "d2_len_32_16")

    # Noise floor: same designed intensity, different wording. Any structural
    # effect must exceed this to count.
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        add(f"d1_S_{x}", f"d1_S_{y}", "floor_wording")
        add(f"d2_n8_{x}", f"d2_n8_{y}", "floor_wording")

    return pairs
