#!/usr/bin/env python3
"""
metrics_b1.py — Phase B1 per-turn negotiation metrics.

Reads results.json (from apply_steering.py), extracts rich per-turn metrics,
and outputs:
  - Structured text report to stdout
  - metrics_b1_enriched.json  (per-game data + new metric fields)
  - metrics_b1_summary.json   (summary tables)

Run: python metrics_b1.py [--input results.json]

No GPU, no model loading — pure analysis of existing results.
"""

import json
import re
import sys
import argparse
from pathlib import Path

import numpy as np

# ── Hedge words / phrases ────────────────────────────────────────────────────

HEDGE_PHRASES = [
    "could consider", "not sure", "kind of", "sort of", "i think",
    "i guess", "i suppose", "a little",
]
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "somewhat"}

# ── Dollar amount regex ──────────────────────────────────────────────────────
# Matches: $1,234.56  $500  $1234  1,234  etc.
# Excludes DEAL= prices (handled separately).
DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Offer trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def parse_offers(transcript):
    """Extract (turn_idx, speaker, amount) for every dollar mention."""
    offers = []
    for turn_idx, turn in enumerate(transcript):
        speaker = turn["speaker"]
        text = turn["utterance"]
        for m in DOLLAR_RE.finditer(text):
            val = float(m.group(1).replace(",", ""))
            if val > 0:
                offers.append((turn_idx, speaker, val))
    return offers


def last_offer_by(offers, speaker, up_to_turn):
    """Return the most recent offer by `speaker` before `up_to_turn`."""
    for turn_idx, spk, amt in reversed(offers):
        if spk == speaker and turn_idx < up_to_turn:
            return amt
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Concession rate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_concessions(offers):
    """Per-offer concession in absolute $ and as % of remaining gap.

    Concession = how much this agent moved toward the other side since
    their own previous offer.  Positive = moved toward opponent.
    % gap = concession / |own_prev - opponent_last| (if both exist).
    """
    concessions = []
    for i, (turn_idx, speaker, amount) in enumerate(offers):
        opponent = "buyer" if speaker == "seller" else "seller"
        own_prev = last_offer_by(offers, speaker, turn_idx)
        opp_last = last_offer_by(offers, opponent, turn_idx + 1)

        if own_prev is None:
            # First offer by this speaker — no concession to compute
            concessions.append({
                "turn_idx": turn_idx, "speaker": speaker, "amount": amount,
                "concession_abs": None, "concession_pct_gap": None,
                "is_first_offer": True,
            })
            continue

        # Concession = movement toward opponent's position.
        # Seller wants high prices, so concession = own_prev - amount (positive = dropped price).
        # Buyer wants low prices, so concession = amount - own_prev (positive = raised price).
        if speaker == "seller":
            concession = own_prev - amount
        else:
            concession = amount - own_prev

        # % of remaining gap
        pct_gap = None
        if opp_last is not None:
            gap = abs(own_prev - opp_last)
            if gap > 0:
                pct_gap = concession / gap

        concessions.append({
            "turn_idx": turn_idx, "speaker": speaker, "amount": amount,
            "concession_abs": round(concession, 2),
            "concession_pct_gap": round(pct_gap, 4) if pct_gap is not None else None,
            "is_first_offer": False,
        })
    return concessions


# ═══════════════════════════════════════════════════════════════════════════════
# 3. First-offer distance
# ═══════════════════════════════════════════════════════════════════════════════

def first_offer_distance(offers, seller_target, buyer_target, listing_price):
    """Distance of each side's first offer from their private target, as % of listing."""
    result = {}
    for speaker, target in [("seller", seller_target), ("buyer", buyer_target)]:
        first = None
        for _, spk, amt in offers:
            if spk == speaker:
                first = amt
                break
        if first is not None and listing_price > 0:
            result[speaker] = {
                "first_offer": first,
                "target": target,
                "distance_abs": round(abs(first - target), 2),
                "distance_pct_listing": round(abs(first - target) / listing_price, 4),
            }
        else:
            result[speaker] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-turn response length
# ═══════════════════════════════════════════════════════════════════════════════

def response_lengths(transcript):
    """Word count per utterance."""
    lengths = []
    for turn in transcript:
        wc = len(turn["utterance"].split())
        lengths.append({
            "speaker": turn["speaker"],
            "word_count": wc,
        })
    return lengths


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Hedge word count
# ═══════════════════════════════════════════════════════════════════════════════

def count_hedges(text):
    """Count hedge words/phrases in text."""
    lower = text.lower()
    count = 0
    for phrase in HEDGE_PHRASES:
        count += lower.count(phrase)
    for word in HEDGE_WORDS:
        # Match whole words only
        count += len(re.findall(r"\b" + re.escape(word) + r"\b", lower))
    return count


def hedge_counts(transcript):
    """Per-utterance hedge counts, raw and per-100-words."""
    results = []
    for turn in transcript:
        text = turn["utterance"]
        raw = count_hedges(text)
        wc = len(text.split())
        per100 = (raw / wc * 100) if wc > 0 else 0.0
        results.append({
            "speaker": turn["speaker"],
            "hedge_raw": raw,
            "hedge_per100w": round(per100, 2),
            "word_count": wc,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring recomputation (results.json lacks clamped/raw fields)
# ═══════════════════════════════════════════════════════════════════════════════

def recompute_scores(game):
    """Recompute raw/clamped scores from agreed_price and targets."""
    if not game["agreed"]:
        return {"clamped": False, "raw_seller_score": 0.0, "raw_buyer_score": 0.0}

    span = game["seller_target"] - game["buyer_target"]
    if span <= 0:
        return {"clamped": False, "raw_seller_score": 0.5, "raw_buyer_score": 0.5}

    raw_seller = (game["agreed_price"] - game["buyer_target"]) / span
    raw_buyer = (game["seller_target"] - game["agreed_price"]) / span
    clamped_seller = max(0.0, min(1.0, raw_seller))
    clamped_buyer = max(0.0, min(1.0, raw_buyer))
    clamped = (clamped_seller != raw_seller) or (clamped_buyer != raw_buyer)

    return {
        "clamped": clamped,
        "raw_seller_score": round(raw_seller, 4),
        "raw_buyer_score": round(raw_buyer, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-game analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_game(game):
    """Run all metrics on a single game. Returns enriched game dict."""
    transcript = game["transcript"]
    offers = parse_offers(transcript)
    concessions = compute_concessions(offers)
    first_dist = first_offer_distance(
        offers, game["seller_target"], game["buyer_target"], game["listing_price"]
    )
    resp_lens = response_lengths(transcript)
    hedges = hedge_counts(transcript)
    scores = recompute_scores(game)

    return {
        "offers": offers,
        "concessions": concessions,
        "first_offer_distance": first_dist,
        "response_lengths": resp_lens,
        "hedges": hedges,
        **scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _stats(values):
    """Mean, std, median, min, max for a list of numbers."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    a = np.array(values, dtype=float)
    return {
        "n": len(a),
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 4),
        "min": round(float(np.min(a)), 4),
        "max": round(float(np.max(a)), 4),
    }


def _agent_label(game, speaker):
    """Is this speaker the steered or baseline agent in this game?"""
    if speaker == game["steered_role"]:
        return "steered"
    return "baseline"


# ═══════════════════════════════════════════════════════════════════════════════
# Summary builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(games, enriched):
    """Build summary tables across all games, split by role and clamping."""
    summary = {}

    # ── Clamping report ──────────────────────────────────────────────────
    n_clamped = sum(1 for e in enriched if e["clamped"])
    summary["clamping"] = {
        "total_games": len(games),
        "n_clamped": n_clamped,
        "pct_clamped": round(n_clamped / len(games) * 100, 1) if games else 0,
    }

    # ── Response length by agent type and role ───────────────────────────
    rl = {"steered": [], "baseline": [], "steered_as_seller": [], "steered_as_buyer": [],
          "baseline_as_seller": [], "baseline_as_buyer": []}
    for game, enr in zip(games, enriched):
        for entry in enr["response_lengths"]:
            label = _agent_label(game, entry["speaker"])
            rl[label].append(entry["word_count"])
            rl[f"{label}_as_{entry['speaker']}"].append(entry["word_count"])
    summary["response_length"] = {k: _stats(v) for k, v in rl.items()}

    # ── Hedge counts by agent type and role ──────────────────────────────
    hd = {"steered": [], "baseline": [], "steered_as_seller": [], "steered_as_buyer": [],
          "baseline_as_seller": [], "baseline_as_buyer": []}
    for game, enr in zip(games, enriched):
        for entry in enr["hedges"]:
            label = _agent_label(game, entry["speaker"])
            hd[label].append(entry["hedge_per100w"])
            hd[f"{label}_as_{entry['speaker']}"].append(entry["hedge_per100w"])
    summary["hedge_per100w"] = {k: _stats(v) for k, v in hd.items()}

    # ── Concession rates by agent type and role ──────────────────────────
    conc_abs = {"steered": [], "baseline": [], "steered_as_seller": [], "steered_as_buyer": [],
                "baseline_as_seller": [], "baseline_as_buyer": []}
    conc_pct = {"steered": [], "baseline": [], "steered_as_seller": [], "steered_as_buyer": [],
                "baseline_as_seller": [], "baseline_as_buyer": []}
    for game, enr in zip(games, enriched):
        for entry in enr["concessions"]:
            if entry["is_first_offer"]:
                continue
            label = _agent_label(game, entry["speaker"])
            if entry["concession_abs"] is not None:
                conc_abs[label].append(entry["concession_abs"])
                conc_abs[f"{label}_as_{entry['speaker']}"].append(entry["concession_abs"])
            if entry["concession_pct_gap"] is not None:
                conc_pct[label].append(entry["concession_pct_gap"])
                conc_pct[f"{label}_as_{entry['speaker']}"].append(entry["concession_pct_gap"])
    summary["concession_abs"] = {k: _stats(v) for k, v in conc_abs.items()}
    summary["concession_pct_gap"] = {k: _stats(v) for k, v in conc_pct.items()}

    # ── First-offer distance by agent type and role ──────────────────────
    fod = {"steered": [], "baseline": [], "steered_as_seller": [], "steered_as_buyer": [],
           "baseline_as_seller": [], "baseline_as_buyer": []}
    for game, enr in zip(games, enriched):
        for speaker in ("seller", "buyer"):
            dist_info = enr["first_offer_distance"].get(speaker)
            if dist_info is None:
                continue
            label = _agent_label(game, speaker)
            fod[label].append(dist_info["distance_pct_listing"])
            fod[f"{label}_as_{speaker}"].append(dist_info["distance_pct_listing"])
    summary["first_offer_distance_pct"] = {k: _stats(v) for k, v in fod.items()}

    # ── Advantage split by clamped/unclamped ─────────────────────────────
    clamped_adv = [g["advantage"] for g, e in zip(games, enriched) if e["clamped"]]
    unclamped_adv = [g["advantage"] for g, e in zip(games, enriched) if not e["clamped"]]
    summary["advantage_by_clamping"] = {
        "clamped": _stats(clamped_adv),
        "unclamped": _stats(unclamped_adv),
        "all": _stats([g["advantage"] for g in games]),
    }

    # ── Advantage split by steered role ──────────────────────────────────
    seller_adv = [g["advantage"] for g in games if g["steered_role"] == "seller"]
    buyer_adv = [g["advantage"] for g in games if g["steered_role"] == "buyer"]
    summary["advantage_by_role"] = {
        "steered_as_seller": _stats(seller_adv),
        "steered_as_buyer": _stats(buyer_adv),
    }

    # ── Offer count per game ─────────────────────────────────────────────
    summary["offers_per_game"] = _stats([len(e["offers"]) for e in enriched])

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt=".4f"):
    """Format a value, handling None."""
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def _table(header, rows, col_widths=None):
    """Simple ASCII table."""
    if col_widths is None:
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


def _stats_row(label, s):
    """Format a _stats dict as a table row."""
    return [
        label, s["n"],
        _fmt(s["mean"]), _fmt(s["std"]),
        _fmt(s["median"]), _fmt(s["min"]), _fmt(s["max"]),
    ]


def print_report(games, enriched, summary):
    """Print structured text report to stdout."""
    print("=" * 78)
    print("  METRICS B1 — Per-Turn Negotiation Analysis")
    print("=" * 78)
    print(f"\n  Input: {len(games)} games, all agreed")
    print()

    header = ["Group", "N", "Mean", "Std", "Median", "Min", "Max"]

    # ── 1. Offer trajectory summary ──────────────────────────────────────
    print("─" * 78)
    print("  1. OFFER TRAJECTORY")
    print("─" * 78)
    print(f"  Offers per game: {_fmt(summary['offers_per_game']['mean'], '.1f')} "
          f"± {_fmt(summary['offers_per_game']['std'], '.1f')} "
          f"(range {_fmt(summary['offers_per_game']['min'], '.0f')}"
          f"–{_fmt(summary['offers_per_game']['max'], '.0f')})")
    print()
    print("  Note: regex extracts ALL dollar amounts per utterance. If an agent")
    print("  quotes the opponent's price before countering, both are captured.")
    print()

    # Sample trajectories (first 5)
    print("  SAMPLE TRAJECTORIES (first 5 games)")
    for i, (game, enr) in enumerate(zip(games[:5], enriched[:5])):
        gid = game.get("game_id", i)
        role = game["steered_role"]
        adv = game["advantage"]
        clamp_tag = " [CLAMPED]" if enr["clamped"] else ""
        print(f"\n  Game {gid} | steered={role} | adv={adv:+.3f}{clamp_tag}")
        print(f"  Listing=${game['listing_price']:.0f}  "
              f"S_target=${game['seller_target']:.0f}  "
              f"B_target=${game['buyer_target']:.0f}")
        for turn_idx, speaker, amount in enr["offers"]:
            label = _agent_label(game, speaker)
            print(f"    Turn {turn_idx}: {speaker:6s} ({label:8s}) -> ${amount:,.0f}")
    print()

    # ── 2. Concession rate (absolute) ────────────────────────────────────
    print("─" * 78)
    print("  2a. CONCESSION RATE — Absolute ($)")
    print("─" * 78)
    ca = summary["concession_abs"]
    rows = [
        _stats_row("Steered", ca["steered"]),
        _stats_row("Baseline", ca["baseline"]),
        _stats_row("Steered-as-Seller", ca["steered_as_seller"]),
        _stats_row("Steered-as-Buyer", ca["steered_as_buyer"]),
        _stats_row("Baseline-as-Seller", ca["baseline_as_seller"]),
        _stats_row("Baseline-as-Buyer", ca["baseline_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 2b. Concession rate (% of gap) ───────────────────────────────────
    print("─" * 78)
    print("  2b. CONCESSION RATE — % of Remaining Gap")
    print("─" * 78)
    cp = summary["concession_pct_gap"]
    rows = [
        _stats_row("Steered", cp["steered"]),
        _stats_row("Baseline", cp["baseline"]),
        _stats_row("Steered-as-Seller", cp["steered_as_seller"]),
        _stats_row("Steered-as-Buyer", cp["steered_as_buyer"]),
        _stats_row("Baseline-as-Seller", cp["baseline_as_seller"]),
        _stats_row("Baseline-as-Buyer", cp["baseline_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 3. First-offer distance ──────────────────────────────────────────
    print("─" * 78)
    print("  3. FIRST-OFFER DISTANCE (as % of listing price)")
    print("─" * 78)
    fo = summary["first_offer_distance_pct"]
    rows = [
        _stats_row("Steered", fo["steered"]),
        _stats_row("Baseline", fo["baseline"]),
        _stats_row("Steered-as-Seller", fo["steered_as_seller"]),
        _stats_row("Steered-as-Buyer", fo["steered_as_buyer"]),
        _stats_row("Baseline-as-Seller", fo["baseline_as_seller"]),
        _stats_row("Baseline-as-Buyer", fo["baseline_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 4. Response length ───────────────────────────────────────────────
    print("─" * 78)
    print("  4. RESPONSE LENGTH (words per utterance)")
    print("─" * 78)
    rl = summary["response_length"]
    rows = [
        _stats_row("Steered", rl["steered"]),
        _stats_row("Baseline", rl["baseline"]),
        _stats_row("Steered-as-Seller", rl["steered_as_seller"]),
        _stats_row("Steered-as-Buyer", rl["steered_as_buyer"]),
        _stats_row("Baseline-as-Seller", rl["baseline_as_seller"]),
        _stats_row("Baseline-as-Buyer", rl["baseline_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 5. Hedge words ───────────────────────────────────────────────────
    print("─" * 78)
    print("  5. HEDGE WORDS (per 100 words)")
    print("─" * 78)
    hd = summary["hedge_per100w"]
    rows = [
        _stats_row("Steered", hd["steered"]),
        _stats_row("Baseline", hd["baseline"]),
        _stats_row("Steered-as-Seller", hd["steered_as_seller"]),
        _stats_row("Steered-as-Buyer", hd["steered_as_buyer"]),
        _stats_row("Baseline-as-Seller", hd["baseline_as_seller"]),
        _stats_row("Baseline-as-Buyer", hd["baseline_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 6. Role-separated advantage ──────────────────────────────────────
    print("─" * 78)
    print("  6. ADVANTAGE BY STEERED ROLE")
    print("─" * 78)
    abr = summary["advantage_by_role"]
    rows = [
        _stats_row("Steered=Seller", abr["steered_as_seller"]),
        _stats_row("Steered=Buyer", abr["steered_as_buyer"]),
    ]
    print(_table(header, rows))
    print()

    # ── 7. Clamping report ───────────────────────────────────────────────
    c = summary["clamping"]
    print("─" * 78)
    print("  7. CLAMPING REPORT")
    print("─" * 78)
    print(f"  Total games:    {c['total_games']}")
    print(f"  Clamped games:  {c['n_clamped']} ({c['pct_clamped']}%)")
    print(f"  Unclamped:      {c['total_games'] - c['n_clamped']}")
    print()

    abc = summary["advantage_by_clamping"]
    rows = [
        _stats_row("Clamped", abc["clamped"]),
        _stats_row("Unclamped", abc["unclamped"]),
        _stats_row("All", abc["all"]),
    ]
    print(_table(header, rows))
    print()

    print("=" * 78)
    print("  Report complete.")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Phase B1 per-turn negotiation metrics.")
    parser.add_argument("--input", default=str(_dir.parent / "results" / "firmness_alpha20_50games.json"),
                        help="Input results JSON file")
    parser.add_argument("--enriched_out", default=str(_dir / "metrics_b1_enriched.json"))
    parser.add_argument("--summary_out", default=str(_dir / "metrics_b1_summary.json"))
    args = parser.parse_args()

    # Load
    path = Path(args.input)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    games = data["games"]
    if not games:
        print("No games found in input.", file=sys.stderr)
        sys.exit(1)

    # Only analyse agreed games (non-agreed have no offers to parse)
    agreed_games = [g for g in games if g["agreed"]]
    print(f"Loaded {len(games)} games ({len(agreed_games)} agreed) from {path}")

    # Analyse each game
    enriched = []
    enriched_games = []
    for game in agreed_games:
        enr = analyse_game(game)
        enriched.append(enr)
        # Build enriched game record (original fields + new metrics)
        enriched_game = dict(game)
        enriched_game["metrics"] = {
            "offers": [
                {"turn_idx": t, "speaker": s, "amount": a}
                for t, s, a in enr["offers"]
            ],
            "concessions": enr["concessions"],
            "first_offer_distance": enr["first_offer_distance"],
            "response_lengths": enr["response_lengths"],
            "hedges": enr["hedges"],
            "clamped": enr["clamped"],
            "raw_seller_score": enr["raw_seller_score"],
            "raw_buyer_score": enr["raw_buyer_score"],
        }
        enriched_games.append(enriched_game)

    # Build summary
    summary = build_summary(agreed_games, enriched)

    # Print report
    print_report(agreed_games, enriched, summary)

    # Save enriched per-game data
    enriched_path = Path(args.enriched_out)
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump({"games": enriched_games}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved enriched data → {enriched_path}")

    # Save summary
    summary_path = Path(args.summary_out)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary      → {summary_path}")


if __name__ == "__main__":
    main()
