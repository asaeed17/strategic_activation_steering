#!/usr/bin/env python3
"""
Phase C: Analytical Expected Payoff vs Observed Payoff

Computes the "framing effect" — how much text style (not just demand numbers)
affects outcomes. Uses baseline acceptance data to build an empirical acceptance
curve, then computes analytical expected payoff for each steered config.

E[payoff] = mean_steered_demand × P(accept | mean_steered_demand)
framing_effect = observed_payoff - E[payoff]_analytical

NOTE: The acceptance curve is built from baseline data across all 18 configs.
Since all baselines use the same unsteered model at temp=0, they should be
identical (modulo pool variation). We also build a STEERED acceptance curve
from steered-arm data to compare.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/moiz/Documents/code/comp0087_snlp_cwk")
UG_DIR = BASE_DIR / "results/ultimatum/confirmatory/ug"
UG_V2_DIR = BASE_DIR / "results/ultimatum/confirmatory_v2/ug_pos_empathy"
OUTPUT_FILE = BASE_DIR / "results/ultimatum/phase_c_analytical_payoff.json"

BIN_WIDTH = 5  # percentage points per bin


def load_games(directory):
    """Load all JSON result files from a directory."""
    results = {}
    for f in sorted(directory.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        results[f.stem] = data
    return results


def extract_acceptance_data(all_results, arm="baseline"):
    """
    From the specified arm of each game, extract (demand_pct, accepted) pairs.
    demand_pct = proposer_share / pool * 100
    """
    pairs = []
    for config_name, data in all_results.items():
        for game in data["games"]:
            g = game[arm]
            pool = game["pool"]
            if g.get("parse_error") or pool == 0:
                continue
            proposer_share = g["proposer_share"]
            demand_pct = proposer_share / pool * 100
            accepted = 1 if g["agreed"] else 0
            pairs.append((demand_pct, accepted))
    return pairs


def build_acceptance_curve(pairs, bin_width=5):
    """
    Bin by demand% (in bin_width pp bins), compute P(accept) per bin.
    Returns dict: {bin_center: (p_accept, n)}
    """
    bins = defaultdict(list)
    for demand_pct, accepted in pairs:
        bin_center = round(demand_pct / bin_width) * bin_width
        bins[bin_center].append(accepted)

    curve = {}
    for bc in sorted(bins):
        vals = bins[bc]
        curve[bc] = (np.mean(vals), len(vals))
    return curve


def interpolate_acceptance(curve, demand_pct):
    """
    Linear interpolation of acceptance probability from the empirical curve.
    If demand_pct is outside the curve range, use nearest bin.
    """
    centers = sorted(curve.keys())
    if demand_pct <= centers[0]:
        return curve[centers[0]][0]
    if demand_pct >= centers[-1]:
        return curve[centers[-1]][0]

    # Find the two surrounding bins
    for i in range(len(centers) - 1):
        if centers[i] <= demand_pct <= centers[i + 1]:
            lo, hi = centers[i], centers[i + 1]
            p_lo = curve[lo][0]
            p_hi = curve[hi][0]
            t = (demand_pct - lo) / (hi - lo) if hi != lo else 0
            return p_lo + t * (p_hi - p_lo)
    return curve[centers[-1]][0]


def extract_steered_data(data):
    """
    From the STEERED arm, extract mean demand_pct, observed acceptance rate,
    and observed payoff_pct.
    """
    demands = []
    payoffs = []
    accepted = []
    for game in data["games"]:
        st = game["steered"]
        pool = game["pool"]
        if st.get("parse_error") or pool == 0:
            continue
        demand_pct = st["proposer_share"] / pool * 100
        demands.append(demand_pct)
        accepted.append(1 if st["agreed"] else 0)
        if st["agreed"]:
            payoffs.append(st["proposer_payoff"] / pool * 100)
        else:
            payoffs.append(0.0)

    return {
        "mean_demand": np.mean(demands),
        "observed_accept_rate": np.mean(accepted),
        "observed_payoff": np.mean(payoffs),
        "n": len(demands),
    }


def extract_per_game_acceptance(data, arm="steered"):
    """Extract per-game (demand_pct, accepted) pairs from one config."""
    pairs = []
    for game in data["games"]:
        g = game[arm]
        pool = game["pool"]
        if g.get("parse_error") or pool == 0:
            continue
        demand_pct = g["proposer_share"] / pool * 100
        accepted = 1 if g["agreed"] else 0
        pairs.append((demand_pct, accepted))
    return pairs


def config_label(filename):
    """Parse filename into a readable config label."""
    parts = filename.split("_")
    dim = parts[0]
    layer = parts[2]
    alpha = parts[3].replace("a", "α=")
    return f"{dim} {layer} {alpha}"


def main():
    # ── Load all game data ───────────────────────────────────────────
    print("Loading game data...")
    ug_results = load_games(UG_DIR)
    ug_v2_results = load_games(UG_V2_DIR)

    all_results = {**ug_results, **ug_v2_results}
    print(f"  Loaded {len(all_results)} configs "
          f"({len(ug_results)} confirmatory + {len(ug_v2_results)} pos_empathy)")

    # ── Diagnostic: raw baseline demand distribution ─────────────────
    print("\n=== DIAGNOSTIC: Baseline demand distribution ===")
    bl_pairs = extract_acceptance_data(all_results, arm="baseline")
    bl_demands = [d for d, _ in bl_pairs]
    print(f"  Total baseline observations: {len(bl_pairs)}")
    print(f"  Demand range: [{min(bl_demands):.1f}%, {max(bl_demands):.1f}%]")
    print(f"  Mean demand: {np.mean(bl_demands):.1f}%")
    print(f"  Median demand: {np.median(bl_demands):.1f}%")

    # Detailed distribution
    demand_counts = defaultdict(int)
    for d, _ in bl_pairs:
        demand_counts[round(d, 1)] += 1
    print(f"  Unique demand values: {len(demand_counts)}")
    print(f"  Top 10 demand values:")
    for val, cnt in sorted(demand_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {val:.1f}%: {cnt} observations")

    # ── Diagnostic: raw baseline acceptance by demand ────────────────
    print("\n=== DIAGNOSTIC: Baseline acceptance by exact demand ===")
    bl_by_demand = defaultdict(list)
    for d, a in bl_pairs:
        bl_by_demand[round(d, 1)].append(a)
    for val in sorted(bl_by_demand):
        vals = bl_by_demand[val]
        if len(vals) >= 5:  # Only show if enough data
            print(f"    demand={val:.1f}%: P(accept)={np.mean(vals):.3f} (n={len(vals)})")

    # ── Build acceptance curve from baseline data ────────────────────
    # Use finer bins (2.5pp) for regions with more data
    print("\n=== Building acceptance curves ===")

    # Standard 5pp bins
    curve_5pp = build_acceptance_curve(bl_pairs, BIN_WIDTH)

    # Also build from ALL data (baseline + steered pooled) for comparison
    all_pairs = bl_pairs.copy()
    for config_name, data in all_results.items():
        all_pairs.extend(extract_per_game_acceptance(data, arm="steered"))

    curve_all = build_acceptance_curve(all_pairs, BIN_WIDTH)

    print(f"\nTABLE 1a: Empirical Acceptance Curve (BASELINE only, 5pp bins)")
    print(f"{'='*50}")
    print(f"{'Demand Bin (%)':>15} | {'P(accept)':>10} | {'n':>5}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*5}")
    curve_output_baseline = []
    for bc in sorted(curve_5pp):
        p_acc, n = curve_5pp[bc]
        print(f"{bc:>14.0f}% | {p_acc:>10.3f} | {n:>5}")
        curve_output_baseline.append({"bin_center": bc, "p_accept": round(p_acc, 4), "n": n})

    print(f"\nTABLE 1b: Empirical Acceptance Curve (ALL data pooled, 5pp bins)")
    print(f"{'='*50}")
    print(f"{'Demand Bin (%)':>15} | {'P(accept)':>10} | {'n':>5}")
    print(f"{'-'*15}-+-{'-'*10}-+-{'-'*5}")
    curve_output_all = []
    for bc in sorted(curve_all):
        p_acc, n = curve_all[bc]
        print(f"{bc:>14.0f}% | {p_acc:>10.3f} | {n:>5}")
        curve_output_all.append({"bin_center": bc, "p_accept": round(p_acc, 4), "n": n})

    # ── Build a BETTER acceptance curve ──────────────────────────────
    # Problem: baseline clusters at ~50%, so we have almost no data above 55%.
    # Solution: Use steered data to fill in the acceptance curve at higher demands.
    # BUT this conflates framing with numbers. So we do BOTH analyses.

    # Approach 1: Baseline-only curve (sparse at high demands)
    # Approach 2: Pooled curve (baseline + steered, more data but conflated)
    # Approach 3: Per-game analytical — for each steered game, look up the
    #   acceptance probability for THAT exact demand in the pooled curve,
    #   compute E[payoff] per game, then average.

    # Let's also do the analysis per-game rather than using mean demand.
    # E[payoff] = E[demand × P(accept|demand)] (per-game average)
    # vs mean_demand × P(accept|mean_demand) (mean-level, ignores Jensen's inequality)

    # ── Compute analytical vs observed payoff per config ─────────────
    print(f"\n{'='*110}")
    print("TABLE 2: Analytical vs Observed Payoff (using POOLED acceptance curve)")
    print(f"{'='*110}")
    header = (f"{'Config':<28} | {'Demand':>7} | {'Obs':>5} | {'P(acc)':>6} | "
              f"{'E[pay]':>7} | {'E[pay]':>7} | {'Obs':>7} | {'Framing':>7} | {'Interpretation'}")
    print(header)
    subhdr = (f"{'':<28} | {'mean%':>7} | {'acc%':>5} | {'curve':>6} | "
              f"{'mean':>7} | {'game':>7} | {'pay%':>7} | {'pp':>7} |")
    print(subhdr)
    print("-" * 110)

    results_table = []
    for config_name in sorted(all_results):
        data = all_results[config_name]
        label = config_label(config_name)

        steered = extract_steered_data(data)
        mean_demand = steered["mean_demand"]

        # Method 1: Interpolate at mean demand (pooled curve)
        p_accept_at_mean = interpolate_acceptance(curve_all, mean_demand)
        e_payoff_at_mean = mean_demand * p_accept_at_mean

        # Method 2: Per-game analytical payoff (pooled curve)
        steered_games = extract_per_game_acceptance(data, arm="steered")
        per_game_analytical = []
        for demand_pct, _ in steered_games:
            p_acc = interpolate_acceptance(curve_all, demand_pct)
            per_game_analytical.append(demand_pct * p_acc)
        e_payoff_per_game = np.mean(per_game_analytical) if per_game_analytical else 0

        observed_payoff = steered["observed_payoff"]
        observed_accept = steered["observed_accept_rate"]

        # Use per-game method as primary (accounts for demand variance)
        framing_effect = observed_payoff - e_payoff_per_game

        if framing_effect > 2:
            interp = "FRAMING HELPS"
        elif framing_effect < -2:
            interp = "FRAMING HURTS"
        else:
            interp = "NUMBERS ONLY"

        print(f"{label:<28} | {mean_demand:>6.1f}% | {observed_accept*100:>4.0f}% | "
              f"{p_accept_at_mean:>6.3f} | {e_payoff_at_mean:>6.1f}% | "
              f"{e_payoff_per_game:>6.1f}% | {observed_payoff:>6.1f}% | "
              f"{framing_effect:>+6.1f}pp | {interp}")

        results_table.append({
            "config": config_name,
            "label": label,
            "steered_mean_demand_pct": round(mean_demand, 2),
            "observed_accept_rate": round(observed_accept, 4),
            "p_accept_analytical_at_mean": round(p_accept_at_mean, 4),
            "e_payoff_at_mean_demand_pct": round(e_payoff_at_mean, 2),
            "e_payoff_per_game_pct": round(e_payoff_per_game, 2),
            "observed_payoff_pct": round(observed_payoff, 2),
            "framing_effect_pp": round(framing_effect, 2),
            "interpretation": interp,
            "n_games": steered["n"],
        })

    # ── Now repeat with BASELINE-only curve for comparison ───────────
    print(f"\n{'='*110}")
    print("TABLE 3: Same analysis with BASELINE-only acceptance curve (sparse at high demand)")
    print(f"{'='*110}")
    print(f"{'Config':<28} | {'Demand':>7} | {'P(acc)':>6} | {'E[pay]':>7} | "
          f"{'Obs pay':>7} | {'Framing':>7} | {'Note'}")
    print("-" * 110)

    for config_name in sorted(all_results):
        data = all_results[config_name]
        label = config_label(config_name)
        steered = extract_steered_data(data)
        mean_demand = steered["mean_demand"]
        steered_games = extract_per_game_acceptance(data, arm="steered")

        per_game_bl = []
        for demand_pct, _ in steered_games:
            p_acc = interpolate_acceptance(curve_5pp, demand_pct)
            per_game_bl.append(demand_pct * p_acc)
        e_payoff_bl = np.mean(per_game_bl)
        framing_bl = steered["observed_payoff"] - e_payoff_bl

        # Note about data sparsity
        n_above_55 = sum(1 for d, _ in steered_games if d > 55)
        note = f"{n_above_55}/{len(steered_games)} games above 55%"

        print(f"{label:<28} | {mean_demand:>6.1f}% | "
              f"{interpolate_acceptance(curve_5pp, mean_demand):>6.3f} | "
              f"{e_payoff_bl:>6.1f}% | {steered['observed_payoff']:>6.1f}% | "
              f"{framing_bl:>+6.1f}pp | {note}")

    # ── Deep dive: WHERE does framing hurt? ──────────────────────────
    print(f"\n{'='*80}")
    print("DEEP DIVE: Per-game comparison for firmness L10 α=7.0")
    print("(demand_pct, P(accept) from curve, analytical payoff, actual payoff)")
    print(f"{'='*80}")

    data = all_results.get("firmness_proposer_L10_a7.0_paired_n100", {})
    if data:
        steered_games = []
        for game in data.get("games", []):
            st = game["steered"]
            pool = game["pool"]
            if st.get("parse_error") or pool == 0:
                continue
            demand_pct = st["proposer_share"] / pool * 100
            agreed = st["agreed"]
            actual_payoff = st["proposer_payoff"] / pool * 100 if agreed else 0.0
            p_acc_curve = interpolate_acceptance(curve_all, demand_pct)
            analytical_payoff = demand_pct * p_acc_curve
            steered_games.append((demand_pct, agreed, actual_payoff, p_acc_curve, analytical_payoff))

        # Show distribution of demands and where rejections happen
        rejections = [(d, a, ap, pc, anp) for d, a, ap, pc, anp in steered_games if not a]
        print(f"  Total games: {len(steered_games)}")
        print(f"  Rejections: {len(rejections)}")
        if rejections:
            print(f"  Rejection demands: {[f'{d:.0f}%' for d, _, _, _, _ in rejections]}")

        # Show by demand bucket
        by_demand = defaultdict(lambda: {"agreed": 0, "total": 0, "payoffs": []})
        for d, a, ap, pc, anp in steered_games:
            bucket = round(d / 5) * 5
            by_demand[bucket]["total"] += 1
            if a:
                by_demand[bucket]["agreed"] += 1
            by_demand[bucket]["payoffs"].append(ap)

        print(f"\n  {'Demand bucket':>15} | {'Accept':>8} | {'Mean payoff':>11} | {'n':>4}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*11}-+-{'-'*4}")
        for bucket in sorted(by_demand):
            bd = by_demand[bucket]
            rate = bd["agreed"] / bd["total"]
            mean_p = np.mean(bd["payoffs"])
            print(f"  {bucket:>14}% | {rate:>7.1%} | {mean_p:>10.1f}% | {bd['total']:>4}")

    # ── Analysis: Is the issue that rejections happen at HIGH demands? ──
    print(f"\n{'='*80}")
    print("ANALYSIS: Acceptance rate by demand level (ALL steered data pooled)")
    print(f"{'='*80}")

    all_steered = []
    for config_name, data in all_results.items():
        for game in data["games"]:
            st = game["steered"]
            pool = game["pool"]
            if st.get("parse_error") or pool == 0:
                continue
            demand_pct = st["proposer_share"] / pool * 100
            agreed = st["agreed"]
            actual_payoff = st["proposer_payoff"] / pool * 100 if agreed else 0.0
            all_steered.append((demand_pct, agreed, actual_payoff, config_name))

    by_bucket = defaultdict(lambda: {"agreed": 0, "total": 0, "payoffs": []})
    for d, a, ap, _ in all_steered:
        bucket = round(d / 5) * 5
        by_bucket[bucket]["total"] += 1
        if a:
            by_bucket[bucket]["agreed"] += 1
        by_bucket[bucket]["payoffs"].append(ap)

    print(f"  {'Demand bucket':>15} | {'P(accept)':>10} | {'Mean payoff':>11} | "
          f"{'Analytical':>10} | {'Diff':>7} | {'n':>5}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}-+-{'-'*5}")
    for bucket in sorted(by_bucket):
        bd = by_bucket[bucket]
        rate = bd["agreed"] / bd["total"]
        mean_p = np.mean(bd["payoffs"])
        analytical = bucket * rate  # actual acceptance at this demand
        diff = mean_p - analytical
        print(f"  {bucket:>14}% | {rate:>9.3f} | {mean_p:>10.1f}% | "
              f"{analytical:>9.1f}% | {diff:>+6.1f}pp | {bd['total']:>5}")

    # ── CORRECTED ANALYSIS ───────────────────────────────────────────
    # The real question: use the SAME curve for both analytical and observed.
    # The "framing effect" should compare:
    #   - What a NUMBERS-ONLY responder would do (baseline curve applied to steered demands)
    #   - What the ACTUAL responder does when reading steered text
    #
    # A negative framing effect means: the steered text makes the responder
    # MORE LIKELY to reject than a pure number-based response would predict.
    # This could be because:
    #   (a) Steered proposer text is aggressive/off-putting
    #   (b) The acceptance curve from baseline is too generous (baseline text is polite)

    print(f"\n{'='*110}")
    print("TABLE 4: CORRECTED ANALYSIS — Baseline curve vs Actual steered acceptance")
    print("  (Positive = steered text helps, Negative = steered text hurts acceptance)")
    print(f"{'='*110}")
    header = (f"{'Config':<28} | {'Demand':>7} | {'Curve':>6} | {'Actual':>6} | "
              f"{'Acc':>5} | {'E[pay]':>7} | {'Obs':>7} | {'Framing':>7} | {'Interpretation'}")
    print(header)
    subhdr = (f"{'':<28} | {'mean%':>7} | {'P(acc)':>6} | {'P(acc)':>6} | "
              f"{'diff':>5} | {'curve':>7} | {'pay%':>7} | {'pp':>7} |")
    print(subhdr)
    print("-" * 110)

    final_results = []
    for config_name in sorted(all_results):
        data = all_results[config_name]
        label = config_label(config_name)
        steered = extract_steered_data(data)
        mean_demand = steered["mean_demand"]
        observed_accept = steered["observed_accept_rate"]
        observed_payoff = steered["observed_payoff"]

        # Per-game analytical payoff using POOLED curve
        steered_games = extract_per_game_acceptance(data, arm="steered")
        per_game_analytical = []
        per_game_p_accept = []
        for demand_pct, _ in steered_games:
            p_acc = interpolate_acceptance(curve_all, demand_pct)
            per_game_analytical.append(demand_pct * p_acc)
            per_game_p_accept.append(p_acc)
        e_payoff_per_game = np.mean(per_game_analytical)
        mean_p_accept_curve = np.mean(per_game_p_accept)

        framing_effect = observed_payoff - e_payoff_per_game
        acc_diff = observed_accept - mean_p_accept_curve

        if framing_effect > 2:
            interp = "FRAMING HELPS"
        elif framing_effect < -2:
            interp = "FRAMING HURTS"
        else:
            interp = "NUMBERS ONLY"

        print(f"{label:<28} | {mean_demand:>6.1f}% | "
              f"{mean_p_accept_curve:>6.3f} | {observed_accept:>6.3f} | "
              f"{acc_diff:>+4.0f}% | "
              f"{e_payoff_per_game:>6.1f}% | {observed_payoff:>6.1f}% | "
              f"{framing_effect:>+6.1f}pp | {interp}")

        final_results.append({
            "config": config_name,
            "label": label,
            "steered_mean_demand_pct": round(mean_demand, 2),
            "observed_accept_rate": round(observed_accept, 4),
            "curve_accept_rate": round(mean_p_accept_curve, 4),
            "accept_rate_diff": round(acc_diff, 4),
            "e_payoff_per_game_pct": round(e_payoff_per_game, 2),
            "observed_payoff_pct": round(observed_payoff, 2),
            "framing_effect_pp": round(framing_effect, 2),
            "interpretation": interp,
            "n_games": steered["n"],
        })

    # ── Summary statistics ───────────────────────────────────────────
    framing_effects = [r["framing_effect_pp"] for r in final_results]
    acc_diffs = [r["accept_rate_diff"] for r in final_results]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean framing effect: {np.mean(framing_effects):+.1f}pp")
    print(f"  Median framing effect: {np.median(framing_effects):+.1f}pp")
    print(f"  Std framing effect: {np.std(framing_effects):.1f}pp")
    print(f"  Range: [{min(framing_effects):+.1f}pp, {max(framing_effects):+.1f}pp]")
    print(f"  Mean acceptance diff: {np.mean(acc_diffs):+.3f}")
    n_helps = sum(1 for f in framing_effects if f > 2)
    n_hurts = sum(1 for f in framing_effects if f < -2)
    n_neutral = sum(1 for f in framing_effects if -2 <= f <= 2)
    print(f"  FRAMING HELPS: {n_helps}/{len(framing_effects)}")
    print(f"  FRAMING HURTS: {n_hurts}/{len(framing_effects)}")
    print(f"  NUMBERS ONLY:  {n_neutral}/{len(framing_effects)}")

    # ── Breakdown by dimension ───────────────────────────────────────
    print(f"\n  By dimension:")
    dims = defaultdict(list)
    for r in final_results:
        dim = r["config"].split("_")[0]
        dims[dim].append(r["framing_effect_pp"])
    for dim in sorted(dims):
        vals = dims[dim]
        print(f"    {dim}: mean={np.mean(vals):+.1f}pp, "
              f"range=[{min(vals):+.1f}, {max(vals):+.1f}]")

    # ── Breakdown by layer ───────────────────────────────────────────
    print(f"\n  By layer:")
    layers = defaultdict(list)
    for r in final_results:
        parts = r["config"].split("_")
        layer = parts[2]
        layers[layer].append(r["framing_effect_pp"])
    for layer in sorted(layers):
        vals = layers[layer]
        print(f"    {layer}: mean={np.mean(vals):+.1f}pp, "
              f"range=[{min(vals):+.1f}, {max(vals):+.1f}]")

    # ── Decomposition: demand effect + acceptance effect ─────────────
    print(f"\n{'='*60}")
    print("DECOMPOSITION: Demand shift vs Acceptance shift")
    print(f"{'='*60}")
    print("  For each config, total payoff change = demand_effect + acceptance_effect")
    print("  demand_effect = (steered_demand - baseline_demand) × baseline_accept_rate")
    print("  acceptance_effect = steered_demand × (observed_accept - curve_accept)")
    print()

    # Get baseline average stats
    bl_data = extract_acceptance_data(all_results, arm="baseline")
    bl_demands_all = [d for d, _ in bl_data]
    bl_accept_all = [a for _, a in bl_data]
    mean_bl_demand = np.mean(bl_demands_all)
    mean_bl_accept = np.mean(bl_accept_all)
    mean_bl_payoff = np.mean([d * a for d, a in bl_data])

    print(f"  Baseline mean demand: {mean_bl_demand:.1f}%")
    print(f"  Baseline mean acceptance: {mean_bl_accept:.3f}")
    print(f"  Baseline mean payoff: {mean_bl_payoff:.1f}%")
    print()

    print(f"{'Config':<28} | {'Δdemand':>8} | {'Dem eff':>8} | {'Acc eff':>8} | "
          f"{'Total':>8} | {'Obs Δ':>8}")
    print("-" * 85)
    for r in final_results:
        config_name = r["config"]
        data = all_results[config_name]
        steered = extract_steered_data(data)
        demand_shift = steered["mean_demand"] - mean_bl_demand
        demand_effect = demand_shift * mean_bl_accept
        acceptance_effect = steered["mean_demand"] * (steered["observed_accept_rate"] - r["curve_accept_rate"])
        total_analytical = demand_effect + acceptance_effect
        obs_delta = steered["observed_payoff"] - mean_bl_payoff

        print(f"{r['label']:<28} | {demand_shift:>+7.1f}pp | {demand_effect:>+7.1f}pp | "
              f"{acceptance_effect:>+7.1f}pp | {total_analytical:>+7.1f}pp | {obs_delta:>+7.1f}pp")

    # ── LEAVE-ONE-OUT ANALYSIS ────────────────────────────────────────
    # The pooled curve includes steered data from the very config being
    # evaluated, which biases framing_effect toward zero. Fix: for each
    # config, build the curve from ALL data EXCEPT that config's steered arm.
    print(f"\n{'='*110}")
    print("TABLE 5: LEAVE-ONE-OUT — Acceptance curve excludes target config's steered data")
    print(f"{'='*110}")
    print(f"{'Config':<28} | {'Demand':>7} | {'Curve':>6} | {'Actual':>6} | "
          f"{'E[pay]':>7} | {'Obs':>7} | {'Framing':>7} | {'Interpretation'}")
    print(f"{'':<28} | {'mean%':>7} | {'P(acc)':>6} | {'P(acc)':>6} | "
          f"{'LOO':>7} | {'pay%':>7} | {'pp':>7} |")
    print("-" * 110)

    loo_results = []
    for config_name in sorted(all_results):
        data = all_results[config_name]
        label = config_label(config_name)
        steered = extract_steered_data(data)
        mean_demand = steered["mean_demand"]
        observed_accept = steered["observed_accept_rate"]
        observed_payoff = steered["observed_payoff"]

        # Build LOO curve: all baseline data + steered data from OTHER configs
        loo_pairs = extract_acceptance_data(all_results, arm="baseline")
        for other_name, other_data in all_results.items():
            if other_name != config_name:
                loo_pairs.extend(extract_per_game_acceptance(other_data, arm="steered"))
        loo_curve = build_acceptance_curve(loo_pairs, BIN_WIDTH)

        # Per-game analytical payoff using LOO curve
        steered_games = extract_per_game_acceptance(data, arm="steered")
        per_game_analytical = []
        per_game_p_accept = []
        for demand_pct, _ in steered_games:
            p_acc = interpolate_acceptance(loo_curve, demand_pct)
            per_game_analytical.append(demand_pct * p_acc)
            per_game_p_accept.append(p_acc)
        e_payoff_loo = np.mean(per_game_analytical)
        mean_p_accept_loo = np.mean(per_game_p_accept)

        framing_loo = observed_payoff - e_payoff_loo

        if framing_loo > 2:
            interp = "FRAMING HELPS"
        elif framing_loo < -2:
            interp = "FRAMING HURTS"
        else:
            interp = "NUMBERS ONLY"

        print(f"{label:<28} | {mean_demand:>6.1f}% | "
              f"{mean_p_accept_loo:>6.3f} | {observed_accept:>6.3f} | "
              f"{e_payoff_loo:>6.1f}% | {observed_payoff:>6.1f}% | "
              f"{framing_loo:>+6.1f}pp | {interp}")

        loo_results.append({
            "config": config_name,
            "label": label,
            "steered_mean_demand_pct": round(mean_demand, 2),
            "observed_accept_rate": round(observed_accept, 4),
            "curve_accept_rate_loo": round(mean_p_accept_loo, 4),
            "e_payoff_loo_pct": round(e_payoff_loo, 2),
            "observed_payoff_pct": round(observed_payoff, 2),
            "framing_effect_loo_pp": round(framing_loo, 2),
            "interpretation": interp,
        })

    loo_framings = [r["framing_effect_loo_pp"] for r in loo_results]
    print(f"\n  LOO Summary: mean={np.mean(loo_framings):+.1f}pp, "
          f"median={np.median(loo_framings):+.1f}pp, "
          f"std={np.std(loo_framings):.1f}pp")
    n_helps_loo = sum(1 for f in loo_framings if f > 2)
    n_hurts_loo = sum(1 for f in loo_framings if f < -2)
    n_neutral_loo = sum(1 for f in loo_framings if -2 <= f <= 2)
    print(f"  FRAMING HELPS: {n_helps_loo}/18, HURTS: {n_hurts_loo}/18, NUMBERS ONLY: {n_neutral_loo}/18")

    # ── Key question answer ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("KEY QUESTION: Does text framing help or hurt payoff?")
    print(f"{'='*60}")
    mean_abs_framing = np.mean(np.abs(framing_effects))
    mean_framing = np.mean(framing_effects)
    mean_loo = np.mean(loo_framings)
    mean_abs_loo = np.mean(np.abs(loo_framings))

    print(f"  Pooled curve:  mean framing = {mean_framing:+.1f}pp, |framing| = {mean_abs_framing:.1f}pp")
    print(f"  LOO curve:     mean framing = {mean_loo:+.1f}pp, |framing| = {mean_abs_loo:.1f}pp")
    print()

    # Use LOO as primary (avoids circularity)
    if abs(mean_loo) < 2 and mean_abs_loo < 3:
        print("  CONCLUSION: Payoff improvement from steering is explained ENTIRELY")
        print("  by the demand shift (the numbers). Text framing contributes negligibly.")
        print()
        print("  The responder's accept/reject decision is driven almost exclusively")
        print("  by the numerical offer, not the surrounding text. Steering works by")
        print("  making the proposer ask for more, not by making it ask more persuasively.")
    elif mean_loo < -2:
        print("  CONCLUSION: Steered text HURTS outcomes — the aggressive/unusual")
        print("  framing provokes more rejections than the numbers alone would predict.")
        print("  Steering gains come ENTIRELY from higher demands, net of framing penalty.")
    else:
        print("  CONCLUSION: Steered text HELPS outcomes — the framing makes offers")
        print("  more likely to be accepted than the numbers alone would predict.")

    # ── Save output ──────────────────────────────────────────────────
    output = {
        "description": "Phase C: Analytical vs Observed Payoff — Framing Effect Analysis",
        "method": (
            "E[payoff] = mean over games of [demand_i × P(accept|demand_i)] "
            "where P(accept|demand) comes from the pooled empirical acceptance curve "
            "(all baseline + steered data). "
            "framing_effect = observed_payoff - E[payoff]_analytical. "
            "Negative means steered text reduces acceptance vs what numbers alone predict. "
            "LOO (leave-one-out) analysis excludes target config's steered data from curve."
        ),
        "baseline_stats": {
            "mean_demand_pct": round(mean_bl_demand, 2),
            "mean_accept_rate": round(mean_bl_accept, 4),
            "mean_payoff_pct": round(mean_bl_payoff, 2),
            "n_observations": len(bl_data),
        },
        "acceptance_curve_baseline_only": curve_output_baseline,
        "acceptance_curve_pooled": curve_output_all,
        "n_baseline_observations": len(bl_pairs),
        "n_total_observations": len(all_pairs),
        "configs_pooled": final_results,
        "configs_loo": loo_results,
        "summary_pooled": {
            "mean_framing_effect_pp": round(np.mean(framing_effects), 2),
            "median_framing_effect_pp": round(np.median(framing_effects), 2),
            "std_framing_effect_pp": round(np.std(framing_effects), 2),
            "min_framing_effect_pp": round(min(framing_effects), 2),
            "max_framing_effect_pp": round(max(framing_effects), 2),
            "mean_accept_diff": round(np.mean(acc_diffs), 4),
            "n_framing_helps": n_helps,
            "n_framing_hurts": n_hurts,
            "n_numbers_only": n_neutral,
            "mean_abs_framing_effect_pp": round(mean_abs_framing, 2),
        },
        "summary_loo": {
            "mean_framing_effect_pp": round(np.mean(loo_framings), 2),
            "median_framing_effect_pp": round(np.median(loo_framings), 2),
            "std_framing_effect_pp": round(np.std(loo_framings), 2),
            "min_framing_effect_pp": round(min(loo_framings), 2),
            "max_framing_effect_pp": round(max(loo_framings), 2),
            "n_framing_helps": n_helps_loo,
            "n_framing_hurts": n_hurts_loo,
            "n_numbers_only": n_neutral_loo,
            "mean_abs_framing_effect_pp": round(np.mean(np.abs(loo_framings)), 2),
        },
    }
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
