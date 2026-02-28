#!/usr/bin/env python3
"""
metrics_b2_decay.py — Phase B2 per-turn steering decay analysis.

Checks whether steering effects (concession rate, response length, hedge rate,
offer gap) decay over the course of a negotiation. Literature (Practitioner's
Field Guide 2026) suggests steering fades after 300-500 tokens.

Reads metrics_b1_enriched.json (from metrics_b1.py).

Run: python metrics_b2_decay.py [--input metrics_b1_enriched.json]

No GPU, no model loading — pure analysis.
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _agent_label(game, speaker):
    return "steered" if speaker == game["steered_role"] else "baseline"


def _utterance_seq(game):
    """Assign each transcript turn its per-agent sequence number (0-indexed).

    Returns list of (agent_label, agent_seq_num) parallel to transcript.
    E.g. for an 8-turn game with steered=seller:
      turn 0: buyer  (baseline) -> baseline_seq=0
      turn 1: seller (steered)  -> steered_seq=0
      turn 2: buyer  (baseline) -> baseline_seq=1
      ...
    """
    counters = {"steered": 0, "baseline": 0}
    result = []
    for turn in game["transcript"]:
        label = _agent_label(game, turn["speaker"])
        result.append((label, counters[label]))
        counters[label] += 1
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-game extraction: align by agent's Nth utterance
# ═══════════════════════════════════════════════════════════════════════════════

def extract_per_utterance(games):
    """For each game, produce per-utterance records keyed by (agent_label, seq).

    Returns list of dicts (one per game), each mapping
    (label, seq) -> {word_count, hedge_per100w, concession_pct_gap, ...}.
    """
    all_records = []
    for game in games:
        m = game["metrics"]
        seq_info = _utterance_seq(game)
        n_turns = len(game["transcript"])

        # Build quick lookup: turn_idx -> concession_pct_gap
        conc_by_turn = {}
        for c in m["concessions"]:
            if not c["is_first_offer"] and c["concession_pct_gap"] is not None:
                conc_by_turn[c["turn_idx"]] = c["concession_pct_gap"]

        records = {}
        for turn_idx in range(n_turns):
            label, seq = seq_info[turn_idx]
            rl = m["response_lengths"][turn_idx]
            hd = m["hedges"][turn_idx]
            records[(label, seq)] = {
                "turn_idx": turn_idx,
                "word_count": rl["word_count"],
                "hedge_per100w": hd["hedge_per100w"],
                "concession_pct_gap": conc_by_turn.get(turn_idx),
            }
        all_records.append(records)
    return all_records


def extract_offer_gap_by_turn(games):
    """Compute offer gap at each absolute turn: |seller_last - buyer_last|.

    Normalised by listing_price to make gaps comparable across games.
    Returns list (one per game) of {turn_idx: normalised_gap}.
    """
    all_gaps = []
    for game in games:
        offers = game["metrics"]["offers"]
        listing = game["listing_price"]
        last = {"seller": None, "buyer": None}
        gaps = {}
        for o in offers:
            last[o["speaker"]] = o["amount"]
            if last["seller"] is not None and last["buyer"] is not None:
                gaps[o["turn_idx"]] = abs(last["seller"] - last["buyer"]) / listing
        all_gaps.append(gaps)
    return all_gaps


def extract_cumulative_tokens(games):
    """Cumulative word count for the steered agent at each of their utterances.

    Returns list (one per game) of [(seq, cum_words), ...].
    """
    all_cum = []
    for game in games:
        seq_info = _utterance_seq(game)
        cum = 0
        points = []
        for turn_idx, (label, seq) in enumerate(seq_info):
            if label == "steered":
                wc = game["metrics"]["response_lengths"][turn_idx]["word_count"]
                cum += wc
                points.append((seq, cum))
        all_cum.append(points)
    return all_cum


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation: mean metric at each agent utterance position
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_by_seq(all_records, metric_key, max_seq=4):
    """Collect metric values at each agent seq position (0..max_seq-1).

    Returns {label: {seq: [values]}} for label in (steered, baseline).
    """
    result = {"steered": {s: [] for s in range(max_seq)},
              "baseline": {s: [] for s in range(max_seq)}}
    for records in all_records:
        for (label, seq), rec in records.items():
            if seq >= max_seq:
                continue
            val = rec[metric_key]
            if val is not None:
                result[label][seq].append(val)
    return result


def aggregate_offer_gap(all_gaps, games, max_turn=8):
    """Collect normalised offer gap at each absolute turn position."""
    by_turn = {t: [] for t in range(max_turn)}
    for gaps in all_gaps:
        for t, g in gaps.items():
            if t < max_turn:
                by_turn[t].append(g)
    return by_turn


def aggregate_cumulative_tokens(all_cum, max_seq=4):
    """Collect cumulative token counts at each steered agent seq position."""
    by_seq = {s: [] for s in range(max_seq)}
    for points in all_cum:
        for seq, cum in points:
            if seq < max_seq:
                by_seq[seq].append(cum)
    return by_seq


# ═══════════════════════════════════════════════════════════════════════════════
# Linear regression per agent
# ═══════════════════════════════════════════════════════════════════════════════

def fit_trend(by_seq):
    """Fit metric ~ seq_number. Returns (slope, intercept, r, p, stderr, n).

    Uses all individual data points, not just means at each position.
    """
    xs, ys = [], []
    for seq, vals in sorted(by_seq.items()):
        for v in vals:
            xs.append(seq)
            ys.append(v)
    if len(xs) < 3:
        return None
    res = stats.linregress(xs, ys)
    return {
        "slope": round(res.slope, 6),
        "intercept": round(res.intercept, 6),
        "r": round(res.rvalue, 4),
        "p": round(res.pvalue, 4),
        "stderr": round(res.stderr, 6),
        "n": len(xs),
        "sig": bool(res.pvalue < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt=".4f"):
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def _table(header, rows):
    col_widths = []
    for i in range(len(header)):
        w = len(str(header[i]))
        for row in rows:
            w = max(w, len(str(row[i])))
        col_widths.append(w + 2)

    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines = [sep]

    def _row(cells):
        parts = []
        for cell, w in zip(cells, col_widths):
            parts.append(f" {str(cell):<{w - 1}}")
        return "|" + "|".join(parts) + "|"

    lines.append(_row(header))
    lines.append(sep)
    for row in rows:
        lines.append(_row(row))
    lines.append(sep)
    return "\n".join(lines)


def print_metric_section(title, by_seq_steered, by_seq_baseline, fmt=".4f"):
    """Print per-seq-position table + regression for one metric."""
    print(f"\n  {title}")
    print("  " + "─" * 74)

    max_seq = max(max(by_seq_steered.keys()), max(by_seq_baseline.keys())) + 1
    header = ["Agent", "Utt #"] + [f"Pos {s}" for s in range(max_seq)] + ["Slope", "p-val", "Sig?"]

    # Means row for steered
    steered_means = []
    for s in range(max_seq):
        vals = by_seq_steered[s]
        steered_means.append(_fmt(np.mean(vals), fmt) if vals else "—")
    trend_s = fit_trend(by_seq_steered)

    # Means row for baseline
    baseline_means = []
    for s in range(max_seq):
        vals = by_seq_baseline[s]
        baseline_means.append(_fmt(np.mean(vals), fmt) if vals else "—")
    trend_b = fit_trend(by_seq_baseline)

    # N row
    steered_ns = [str(len(by_seq_steered.get(s, []))) for s in range(max_seq)]
    baseline_ns = [str(len(by_seq_baseline.get(s, []))) for s in range(max_seq)]

    rows = [
        ["Steered", "mean"] + steered_means + [
            _fmt(trend_s["slope"], ".6f") if trend_s else "—",
            _fmt(trend_s["p"], ".4f") if trend_s else "—",
            "YES" if (trend_s and trend_s["sig"]) else "no",
        ],
        ["", "N"] + steered_ns + ["", "", ""],
        ["Baseline", "mean"] + baseline_means + [
            _fmt(trend_b["slope"], ".6f") if trend_b else "—",
            _fmt(trend_b["p"], ".4f") if trend_b else "—",
            "YES" if (trend_b and trend_b["sig"]) else "no",
        ],
        ["", "N"] + baseline_ns + ["", "", ""],
    ]
    print(_table(header, rows))
    return trend_s, trend_b


def print_report(games, all_records, all_gaps, all_cum):
    """Print full decay analysis report."""
    print("=" * 78)
    print("  METRICS B2 — Per-Turn Steering Decay Analysis")
    print("=" * 78)
    print(f"\n  Input: {len(games)} games")
    print("  Alignment: by steered agent's Nth utterance (not absolute turn)")
    print("  Regression: metric ~ utterance_position, all individual data points")
    print()

    results = {}

    # ── 1. Concession rate by turn ───────────────────────────────────────
    conc = aggregate_by_seq(all_records, "concession_pct_gap")
    ts, tb = print_metric_section(
        "1. CONCESSION RATE (% of gap) BY UTTERANCE POSITION", conc["steered"], conc["baseline"]
    )
    results["concession_pct_gap"] = {"steered_trend": ts, "baseline_trend": tb}
    print()

    # ── 2. Response length by turn ───────────────────────────────────────
    rl = aggregate_by_seq(all_records, "word_count")
    ts, tb = print_metric_section(
        "2. RESPONSE LENGTH (words) BY UTTERANCE POSITION", rl["steered"], rl["baseline"], fmt=".1f"
    )
    results["word_count"] = {"steered_trend": ts, "baseline_trend": tb}
    print()

    # ── 3. Hedge rate by turn ────────────────────────────────────────────
    hd = aggregate_by_seq(all_records, "hedge_per100w")
    ts, tb = print_metric_section(
        "3. HEDGE RATE (per 100w) BY UTTERANCE POSITION", hd["steered"], hd["baseline"]
    )
    results["hedge_per100w"] = {"steered_trend": ts, "baseline_trend": tb}
    print()

    # ── 4. Offer gap by turn ─────────────────────────────────────────────
    gap_by_turn = aggregate_offer_gap(all_gaps, games)
    print(f"\n  4. OFFER GAP (normalised by listing price) BY ABSOLUTE TURN")
    print("  " + "─" * 74)
    print("  (Gap = |seller_last_offer - buyer_last_offer| / listing_price)")
    print("  Requires both sides to have made at least one offer.\n")
    max_t = max(t for t, v in gap_by_turn.items() if v) + 1 if any(gap_by_turn.values()) else 0

    header = ["Turn"] + [str(t) for t in range(max_t)] + ["Slope", "p-val", "Sig?"]
    means = []
    ns = []
    for t in range(max_t):
        vals = gap_by_turn[t]
        means.append(_fmt(np.mean(vals), ".4f") if vals else "—")
        ns.append(str(len(vals)))

    trend_gap = fit_trend({t: v for t, v in gap_by_turn.items() if t < max_t and v})
    rows = [
        ["Mean"] + means + [
            _fmt(trend_gap["slope"], ".6f") if trend_gap else "—",
            _fmt(trend_gap["p"], ".4f") if trend_gap else "—",
            "YES" if (trend_gap and trend_gap["sig"]) else "no",
        ],
        ["N"] + ns + ["", "", ""],
    ]
    print(_table(header, rows))
    results["offer_gap"] = {"trend": trend_gap}
    print()

    # ── 5. Cumulative token count ────────────────────────────────────────
    cum_by_seq = aggregate_cumulative_tokens(all_cum)
    print(f"\n  5. CUMULATIVE TOKENS (steered agent only)")
    print("  " + "─" * 74)
    print("  Literature threshold for steering decay: 300-500 tokens.\n")

    max_seq = max(s for s, v in cum_by_seq.items() if v) + 1 if any(cum_by_seq.values()) else 0
    header = ["Stat"] + [f"Utt {s}" for s in range(max_seq)]
    cum_means = []
    cum_maxs = []
    cum_ns = []
    for s in range(max_seq):
        vals = cum_by_seq[s]
        if vals:
            cum_means.append(_fmt(np.mean(vals), ".1f"))
            cum_maxs.append(_fmt(np.max(vals), ".0f"))
            cum_ns.append(str(len(vals)))
        else:
            cum_means.append("—")
            cum_maxs.append("—")
            cum_ns.append("0")

    rows = [
        ["Mean cum. words"] + cum_means,
        ["Max cum. words"] + cum_maxs,
        ["N"] + cum_ns,
    ]
    print(_table(header, rows))

    # Check 300-500 threshold
    all_final_cum = []
    for points in all_cum:
        if points:
            all_final_cum.append(points[-1][1])
    if all_final_cum:
        n_over_300 = sum(1 for c in all_final_cum if c > 300)
        n_over_500 = sum(1 for c in all_final_cum if c > 500)
        mean_final = np.mean(all_final_cum)
        max_final = np.max(all_final_cum)
        print(f"\n  Final cumulative words (steered agent, last utterance):")
        print(f"    Mean: {mean_final:.1f}   Max: {max_final:.0f}")
        print(f"    >300 words: {n_over_300}/{len(all_final_cum)} games ({n_over_300/len(all_final_cum)*100:.0f}%)")
        print(f"    >500 words: {n_over_500}/{len(all_final_cum)} games ({n_over_500/len(all_final_cum)*100:.0f}%)")
        results["cumulative_tokens"] = {
            "mean_final": round(mean_final, 1),
            "max_final": int(max_final),
            "n_over_300": n_over_300,
            "n_over_500": n_over_500,
            "n_games": len(all_final_cum),
        }
    print()

    # ── Synthesis ────────────────────────────────────────────────────────
    print("─" * 78)
    print("  SYNTHESIS")
    print("─" * 78)

    decay_signals = []
    no_decay_signals = []

    # Check concession decay
    cs = results["concession_pct_gap"]["steered_trend"]
    cb = results["concession_pct_gap"]["baseline_trend"]
    if cs and cs["sig"] and cs["slope"] > 0:
        decay_signals.append(f"Concession rate: steered slope={cs['slope']:+.4f} (p={cs['p']:.4f}) — "
                             f"concessions increase over turns, consistent with steering wearing off")
    elif cs and cs["sig"] and cs["slope"] < 0:
        no_decay_signals.append(f"Concession rate: steered slope={cs['slope']:+.4f} (p={cs['p']:.4f}) — "
                                f"concessions *decrease*, steering may be intensifying")
    else:
        no_decay_signals.append(f"Concession rate: no significant trend (p={cs['p']:.4f})" if cs
                                else "Concession rate: insufficient data")

    # Check response length decay
    ws = results["word_count"]["steered_trend"]
    wb = results["word_count"]["baseline_trend"]
    if ws and ws["sig"]:
        # If steered gets longer over turns, steering (which shortens) may be fading
        direction = "lengthening (decay signal)" if ws["slope"] > 0 else "shortening (no decay)"
        tag = decay_signals if ws["slope"] > 0 else no_decay_signals
        tag.append(f"Response length: steered slope={ws['slope']:+.4f} (p={ws['p']:.4f}) — {direction}")
    else:
        no_decay_signals.append(f"Response length: no significant trend (p={ws['p']:.4f})" if ws
                                else "Response length: insufficient data")

    # Check hedge decay
    hs = results["hedge_per100w"]["steered_trend"]
    if hs and hs["sig"] and hs["slope"] > 0:
        decay_signals.append(f"Hedge rate: steered slope={hs['slope']:+.4f} (p={hs['p']:.4f}) — "
                             f"hedging increases over turns, consistent with decay")
    else:
        no_decay_signals.append(f"Hedge rate: no significant trend (p={hs['p']:.4f})" if hs
                                else "Hedge rate: insufficient data")

    # Check offer gap
    gt = results["offer_gap"]["trend"]
    if gt and gt["sig"] and gt["slope"] < 0:
        no_decay_signals.append(f"Offer gap: slope={gt['slope']:+.4f} (p={gt['p']:.4f}) — "
                                f"gap narrows over turns (expected convergence)")
    elif gt:
        no_decay_signals.append(f"Offer gap: slope={gt['slope']:+.4f} (p={gt['p']:.4f})")

    # Token threshold
    cum = results.get("cumulative_tokens", {})
    if cum.get("n_over_300", 0) == 0:
        no_decay_signals.append(f"Token count: no games exceed 300 words — "
                                f"below literature decay threshold (300-500 tokens)")
    else:
        decay_signals.append(f"Token count: {cum['n_over_300']}/{cum['n_games']} games exceed 300 words")

    print()
    if decay_signals:
        print("  Signals CONSISTENT with steering decay:")
        for s in decay_signals:
            print(f"    - {s}")
    else:
        print("  No signals consistent with steering decay found.")

    print()
    if no_decay_signals:
        print("  Signals AGAINST steering decay / neutral:")
        for s in no_decay_signals:
            print(f"    - {s}")

    # Overall verdict
    print()
    if not decay_signals:
        verdict = ("CONCLUSION: No evidence of steering decay within 8-turn negotiations. "
                    "Steered agent's behavioral differences from baseline are stable across "
                    "utterance positions. This is consistent with the literature threshold "
                    "(300-500 tokens) — our games likely stay below it.")
    elif len(decay_signals) >= 2:
        verdict = ("CONCLUSION: Multiple signals suggest steering effects may decay over "
                    "the course of negotiation. Check if games exceed the 300-500 token "
                    "literature threshold.")
    else:
        verdict = ("CONCLUSION: Mixed evidence. One metric shows a decay-like trend but "
                    "others do not. Likely not a major concern for 8-turn negotiations.")
    print(f"  {verdict}")
    print()
    print("=" * 78)

    results["verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Phase B2 per-turn steering decay analysis.")
    parser.add_argument("--input", default=str(_dir / "metrics_b1_enriched.json"))
    parser.add_argument("--output", default=str(_dir / "metrics_b2_decay.json"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found. Run metrics_b1.py first.", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    games = data["games"]
    if not games:
        print("No games found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(games)} games from {path}")

    all_records = extract_per_utterance(games)
    all_gaps = extract_offer_gap_by_turn(games)
    all_cum = extract_cumulative_tokens(games)

    results = print_report(games, all_records, all_gaps, all_cum)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
