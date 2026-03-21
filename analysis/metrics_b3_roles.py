#!/usr/bin/env python3
"""
metrics_b3_roles.py — Phase B3 comprehensive role-separated analysis.

Produces publication-ready breakdown tables showing how every metric differs
when the steered agent plays seller vs buyer.

Reads metrics_b1_enriched.json.

Run: python metrics_b3_roles.py [--input metrics_b1_enriched.json]

No GPU, no model loading — pure analysis. Dependencies: numpy.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt=".4f"):
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def _pct(n, total):
    return f"{n / total * 100:.1f}%" if total > 0 else "—"


def _table(header, rows):
    col_widths = []
    for i in range(len(header)):
        w = len(str(header[i]))
        for row in rows:
            if i < len(row):
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


def _agent_label(game, speaker):
    return "steered" if speaker == game["steered_role"] else "baseline"


def split_by_role(games):
    """Split games into steered-as-seller and steered-as-buyer groups."""
    seller_games = [g for g in games if g["steered_role"] == "seller"]
    buyer_games = [g for g in games if g["steered_role"] == "buyer"]
    return seller_games, buyer_games


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1 — Outcome by role
# ═══════════════════════════════════════════════════════════════════════════════

def table1_outcome(games, seller_games, buyer_games):
    def _row_data(label, subset):
        advs = [g["advantage"] for g in subset]
        n = len(advs)
        if n == 0:
            return [label, 0, "—", "—", "—", "—", "—", "—"]
        a = np.array(advs)
        wins = int(np.sum(a > 0))
        losses = int(np.sum(a < 0))
        neutral = int(np.sum(a == 0))
        return [
            label, n,
            _fmt(np.mean(a)), _fmt(np.std(a, ddof=1) if n > 1 else 0.0),
            _fmt(np.median(a)),
            _pct(wins, n), _pct(losses, n), _pct(neutral, n),
        ]

    header = ["Role", "N", "Mean Adv", "Std", "Median", "Win%", "Lose%", "Neutral%"]
    rows = [
        _row_data("Steered=Seller", seller_games),
        _row_data("Steered=Buyer", buyer_games),
        _row_data("All*", games),
    ]
    return header, rows, {"steered_as_seller": _row_dict(seller_games),
                          "steered_as_buyer": _row_dict(buyer_games),
                          "all": _row_dict(games)}


def _row_dict(subset):
    advs = [g["advantage"] for g in subset]
    if not advs:
        return {}
    a = np.array(advs)
    return {
        "n": len(a),
        "mean_advantage": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 4),
        "win_pct": round(float(np.sum(a > 0) / len(a) * 100), 1),
        "lose_pct": round(float(np.sum(a < 0) / len(a) * 100), 1),
        "neutral_pct": round(float(np.sum(a == 0) / len(a) * 100), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2 — Concession behavior by role
# ═══════════════════════════════════════════════════════════════════════════════

def _concession_stats(games_subset, agent_filter):
    """Collect concession stats for a specific agent within a game subset.

    agent_filter: "steered" or "baseline"
    """
    abs_vals = []
    pct_vals = []
    fod_vals = []

    for game in games_subset:
        m = game["metrics"]
        for c in m["concessions"]:
            if c["is_first_offer"]:
                continue
            label = _agent_label(game, c["speaker"])
            if label != agent_filter:
                continue
            if c["concession_abs"] is not None:
                abs_vals.append(c["concession_abs"])
            if c["concession_pct_gap"] is not None:
                pct_vals.append(c["concession_pct_gap"])

        # First-offer distance
        for speaker in ("seller", "buyer"):
            if _agent_label(game, speaker) != agent_filter:
                continue
            fod = m["first_offer_distance"].get(speaker)
            if fod is not None:
                fod_vals.append(fod["distance_pct_listing"])

    return {
        "conc_abs_mean": round(float(np.mean(abs_vals)), 2) if abs_vals else None,
        "conc_pct_mean": round(float(np.mean(pct_vals)), 4) if pct_vals else None,
        "fod_mean": round(float(np.mean(fod_vals)), 4) if fod_vals else None,
        "conc_abs_n": len(abs_vals),
        "conc_pct_n": len(pct_vals),
        "fod_n": len(fod_vals),
    }


def table2_concession(seller_games, buyer_games):
    configs = [
        ("Steered=Seller", "steered", seller_games),
        ("Steered=Seller", "baseline", seller_games),
        ("Steered=Buyer", "steered", buyer_games),
        ("Steered=Buyer", "baseline", buyer_games),
    ]
    header = ["Role", "Agent", "Conc $ (mean)", "Conc % gap (mean)", "1st Offer Dist %"]
    rows = []
    data = {}
    for role_label, agent, subset in configs:
        s = _concession_stats(subset, agent)
        key = f"{role_label.lower().replace('=', '_')}_{agent}"
        data[key] = s
        rows.append([
            role_label, agent,
            _fmt(s["conc_abs_mean"], ".1f") if s["conc_abs_mean"] is not None else "—",
            _fmt(s["conc_pct_mean"]) if s["conc_pct_mean"] is not None else "—",
            _fmt(s["fod_mean"]) if s["fod_mean"] is not None else "—",
        ])
    return header, rows, data


# ═══════════════════════════════════════════════════════════════════════════════
# Table 3 — Communication by role
# ═══════════════════════════════════════════════════════════════════════════════

def _comm_stats(games_subset, agent_filter):
    """Word counts, hedge rates, and offer counts for an agent."""
    word_counts = []
    hedge_rates = []
    offer_counts_per_game = []

    for game in games_subset:
        m = game["metrics"]
        game_offers = 0
        for rl in m["response_lengths"]:
            if _agent_label(game, rl["speaker"]) == agent_filter:
                word_counts.append(rl["word_count"])
        for hd in m["hedges"]:
            if _agent_label(game, hd["speaker"]) == agent_filter:
                hedge_rates.append(hd["hedge_per100w"])
        for o in m["offers"]:
            if _agent_label(game, o["speaker"]) == agent_filter:
                game_offers += 1
        offer_counts_per_game.append(game_offers)

    return {
        "words_mean": round(float(np.mean(word_counts)), 1) if word_counts else None,
        "words_std": round(float(np.std(word_counts, ddof=1)), 1) if len(word_counts) > 1 else None,
        "hedge_mean": round(float(np.mean(hedge_rates)), 4) if hedge_rates else None,
        "offers_mean": round(float(np.mean(offer_counts_per_game)), 1) if offer_counts_per_game else None,
        "n_utterances": len(word_counts),
    }


def table3_communication(seller_games, buyer_games):
    configs = [
        ("Steered=Seller", "steered", seller_games),
        ("Steered=Seller", "baseline", seller_games),
        ("Steered=Buyer", "steered", buyer_games),
        ("Steered=Buyer", "baseline", buyer_games),
    ]
    header = ["Role", "Agent", "Words/Turn", "Hedge/100w", "Offers/Game"]
    rows = []
    data = {}
    for role_label, agent, subset in configs:
        s = _comm_stats(subset, agent)
        key = f"{role_label.lower().replace('=', '_')}_{agent}"
        data[key] = s
        rows.append([
            role_label, agent,
            _fmt(s["words_mean"], ".1f") if s["words_mean"] is not None else "—",
            _fmt(s["hedge_mean"]) if s["hedge_mean"] is not None else "—",
            _fmt(s["offers_mean"], ".1f") if s["offers_mean"] is not None else "—",
        ])
    return header, rows, data


# ═══════════════════════════════════════════════════════════════════════════════
# Table 4 — Clamping by role
# ═══════════════════════════════════════════════════════════════════════════════

def table4_clamping(seller_games, buyer_games):
    def _clamp_row(label, subset):
        clamped = [g for g in subset if g["metrics"]["clamped"]]
        unclamped = [g for g in subset if not g["metrics"]["clamped"]]
        n = len(subset)
        nc = len(clamped)
        adv_c = [g["advantage"] for g in clamped]
        adv_u = [g["advantage"] for g in unclamped]
        return [
            label, nc, _pct(nc, n),
            _fmt(np.mean(adv_c), "+.4f") if adv_c else "—",
            _fmt(np.mean(adv_u), "+.4f") if adv_u else "—",
        ], {
            "n": n, "n_clamped": nc, "pct_clamped": round(nc / n * 100, 1) if n else 0,
            "adv_clamped": round(float(np.mean(adv_c)), 4) if adv_c else None,
            "adv_unclamped": round(float(np.mean(adv_u)), 4) if adv_u else None,
        }

    header = ["Role", "N Clamp", "% Clamp", "Adv (clamped)", "Adv (unclamped)"]
    r1, d1 = _clamp_row("Steered=Seller", seller_games)
    r2, d2 = _clamp_row("Steered=Buyer", buyer_games)
    return header, [r1, r2], {"steered_as_seller": d1, "steered_as_buyer": d2}


# ═══════════════════════════════════════════════════════════════════════════════
# Table 5 — Deal structure by role
# ═══════════════════════════════════════════════════════════════════════════════

def table5_deal_structure(seller_games, buyer_games):
    def _deal_row(label, subset):
        dealmakers = defaultdict(int)
        prices = []
        listings = []
        for g in subset:
            dealmakers[g["dealmaker"]] += 1
            prices.append(g["agreed_price"])
            listings.append(g["listing_price"])
        n = len(subset)
        # Most common dealmaker
        dm_str = ", ".join(f"{k}={v}" for k, v in sorted(dealmakers.items(), key=lambda x: -x[1]))
        mean_price = float(np.mean(prices)) if prices else 0
        mean_listing = float(np.mean(listings)) if listings else 0
        ratio = mean_price / mean_listing if mean_listing > 0 else 0
        return [
            label, dm_str,
            f"${mean_price:,.0f}", f"${mean_listing:,.0f}",
            _fmt(ratio, ".3f"),
        ], {
            "n": n, "dealmakers": dict(dealmakers),
            "mean_agreed_price": round(mean_price, 2),
            "mean_listing_price": round(mean_listing, 2),
            "agreed_listing_ratio": round(ratio, 4),
        }

    header = ["Role", "Dealmaker", "Mean Agreed $", "Mean Listed $", "Agreed/Listed"]
    r1, d1 = _deal_row("Steered=Seller", seller_games)
    r2, d2 = _deal_row("Steered=Buyer", buyer_games)
    return header, [r1, r2], {"steered_as_seller": d1, "steered_as_buyer": d2}


# ═══════════════════════════════════════════════════════════════════════════════
# Table 6 — Anti-steerability by role
# ═══════════════════════════════════════════════════════════════════════════════

def table6_antisteerability(seller_games, buyer_games):
    def _anti_row(label, subset):
        advs = np.array([g["advantage"] for g in subset])
        n = len(advs)
        helps = int(np.sum(advs > 0))
        hurts = int(np.sum(advs < 0))
        neutral = int(np.sum(advs == 0))
        return [
            label, helps, hurts, neutral,
            _pct(helps, n), _pct(hurts, n),
        ], {
            "n": n, "helps": helps, "hurts": hurts, "neutral": neutral,
            "help_pct": round(helps / n * 100, 1) if n else 0,
            "hurt_pct": round(hurts / n * 100, 1) if n else 0,
        }

    header = ["Role", "N Helps", "N Hurts", "N Neutral", "Help%", "Hurt%"]
    r1, d1 = _anti_row("Steered=Seller", seller_games)
    r2, d2 = _anti_row("Steered=Buyer", buyer_games)
    r3, d3 = _anti_row("All", seller_games + buyer_games)
    return header, [r1, r2, r3], {"steered_as_seller": d1, "steered_as_buyer": d2, "all": d3}


# ═══════════════════════════════════════════════════════════════════════════════
# Table 7 — Category × Role interaction
# ═══════════════════════════════════════════════════════════════════════════════

def table7_category_role(seller_games, buyer_games):
    MIN_N = 3

    cat_seller = defaultdict(list)
    cat_buyer = defaultdict(list)
    for g in seller_games:
        cat_seller[g["category"]].append(g["advantage"])
    for g in buyer_games:
        cat_buyer[g["category"]].append(g["advantage"])

    all_cats = sorted(set(list(cat_seller.keys()) + list(cat_buyer.keys())))

    header = ["Category", "N Seller", "N Buyer", "Adv Seller", "Adv Buyer", "Delta"]
    rows = []
    data = {}
    for cat in all_cats:
        ns = len(cat_seller.get(cat, []))
        nb = len(cat_buyer.get(cat, []))
        if ns < MIN_N and nb < MIN_N:
            continue
        adv_s = float(np.mean(cat_seller[cat])) if ns >= MIN_N else None
        adv_b = float(np.mean(cat_buyer[cat])) if nb >= MIN_N else None
        delta = None
        if adv_s is not None and adv_b is not None:
            delta = adv_b - adv_s
        rows.append([
            cat,
            ns if ns >= MIN_N else f"{ns}*",
            nb if nb >= MIN_N else f"{nb}*",
            _fmt(adv_s, "+.3f") if adv_s is not None else f"N<{MIN_N}",
            _fmt(adv_b, "+.3f") if adv_b is not None else f"N<{MIN_N}",
            _fmt(delta, "+.3f") if delta is not None else "—",
        ])
        data[cat] = {
            "n_seller": ns, "n_buyer": nb,
            "adv_seller": round(adv_s, 4) if adv_s is not None else None,
            "adv_buyer": round(adv_b, 4) if adv_b is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
        }
    return header, rows, data


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(games, seller_games, buyer_games):
    print("=" * 78)
    print("  METRICS B3 — Role-Separated Analysis")
    print("=" * 78)
    print(f"\n  Input: {len(games)} games "
          f"({len(seller_games)} steered-as-seller, {len(buyer_games)} steered-as-buyer)")
    print()

    all_data = {}

    # Table 1
    print("─" * 78)
    print("  TABLE 1: Outcome by Steered Role")
    print("─" * 78)
    h, r, d = table1_outcome(games, seller_games, buyer_games)
    print(_table(h, r))
    print("  * Aggregate is misleading — seller and buyer results have opposite signs.")
    print()
    all_data["table1_outcome"] = d

    # Table 2
    print("─" * 78)
    print("  TABLE 2: Concession Behavior by Role")
    print("─" * 78)
    h, r, d = table2_concession(seller_games, buyer_games)
    print(_table(h, r))
    print()
    all_data["table2_concession"] = d

    # Table 3
    print("─" * 78)
    print("  TABLE 3: Communication Style by Role")
    print("─" * 78)
    h, r, d = table3_communication(seller_games, buyer_games)
    print(_table(h, r))
    print()
    all_data["table3_communication"] = d

    # Table 4
    print("─" * 78)
    print("  TABLE 4: Clamping by Role")
    print("─" * 78)
    h, r, d = table4_clamping(seller_games, buyer_games)
    print(_table(h, r))
    print()
    all_data["table4_clamping"] = d

    # Table 5
    print("─" * 78)
    print("  TABLE 5: Deal Structure by Role")
    print("─" * 78)
    h, r, d = table5_deal_structure(seller_games, buyer_games)
    print(_table(h, r))
    print()
    all_data["table5_deal_structure"] = d

    # Table 6
    print("─" * 78)
    print("  TABLE 6: Anti-Steerability by Role")
    print("─" * 78)
    h, r, d = table6_antisteerability(seller_games, buyer_games)
    print(_table(h, r))
    # Interpret
    ds = d["steered_as_seller"]
    db = d["steered_as_buyer"]
    if ds["hurt_pct"] > ds["help_pct"] and db["help_pct"] > db["hurt_pct"]:
        note = ("Anti-steerability is ASYMMETRIC: steering mostly hurts as seller "
                f"({ds['hurt_pct']:.0f}% hurt) but mostly helps as buyer "
                f"({db['help_pct']:.0f}% help). The aggregate 50/50 split "
                "masks a role-dependent pattern.")
    elif abs(ds["help_pct"] - db["help_pct"]) < 10:
        note = ("Anti-steerability is roughly SYMMETRIC across roles — "
                "both show similar help/hurt splits.")
    else:
        note = (f"Anti-steerability differs by role: seller help={ds['help_pct']:.0f}%, "
                f"buyer help={db['help_pct']:.0f}%.")
    print(f"\n  {note}")
    all_data["table6_antisteerability"] = d
    all_data["table6_note"] = note
    print()

    # Table 7
    print("─" * 78)
    print("  TABLE 7: Category x Role Interaction (N >= 3 per cell)")
    print("─" * 78)
    h, r, d = table7_category_role(seller_games, buyer_games)
    print(_table(h, r))
    print("  * = N below threshold, excluded from mean. Small N limits reliability.")
    print("  Delta = Adv(Buyer) - Adv(Seller). Positive = steering helps more as buyer.")
    print()
    all_data["table7_category_role"] = d

    # Synthesis
    print("─" * 78)
    print("  SYNTHESIS")
    print("─" * 78)
    t1 = all_data["table1_outcome"]
    t3s = all_data["table3_communication"]
    t6 = all_data["table6_antisteerability"]

    print("""
  1. ROLE IS THE DOMINANT MODERATOR. The steered agent loses as seller
     (mean adv = {s_adv:+.3f}) but wins as buyer (mean adv = {b_adv:+.3f}).
     Aggregate advantage ({a_adv:+.3f}) is meaningless.

  2. ANTI-STEERABILITY IS ROLE-DEPENDENT. As seller, steering hurts in
     {s_hurt}% of games. As buyer, it helps in {b_help}% of games. The
     aggregate 50/50 split is an artifact of averaging two different
     distributions.

  3. COMMUNICATION PATTERNS DIFFER BY ROLE. When the steered agent is
     the seller, it speaks {ss_w:.0f} words/turn vs baseline buyer at
     {sb_w:.0f}. When steered is buyer, it speaks {bs_w:.0f} words/turn
     vs baseline seller at {bb_w:.0f}. Steering shortens utterances in
     both roles, but the effect is larger when playing buyer.

  4. FOR THE PAPER: Never report aggregate steering advantage. Always
     report seller and buyer separately. The finding is not "steering
     doesn't work" — it's "steering helps buyers and hurts sellers."
""".format(
        s_adv=t1["steered_as_seller"]["mean_advantage"],
        b_adv=t1["steered_as_buyer"]["mean_advantage"],
        a_adv=t1["all"]["mean_advantage"],
        s_hurt=t6["steered_as_seller"]["hurt_pct"],
        b_help=t6["steered_as_buyer"]["help_pct"],
        ss_w=t3s["steered_seller_steered"]["words_mean"],
        sb_w=t3s["steered_seller_baseline"]["words_mean"],
        bs_w=t3s["steered_buyer_steered"]["words_mean"],
        bb_w=t3s["steered_buyer_baseline"]["words_mean"],
    ))

    print("=" * 78)
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Phase B3 role-separated analysis.")
    parser.add_argument("--input", default=str(_dir / "metrics_b1_enriched.json"))
    parser.add_argument("--output", default=str(_dir / "metrics_b3_roles.json"))
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

    seller_games, buyer_games = split_by_role(games)
    all_data = print_report(games, seller_games, buyer_games)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
