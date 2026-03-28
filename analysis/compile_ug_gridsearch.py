#!/usr/bin/env python3
"""
compile_exploratory.py — Compile ALL Ultimatum Game results with BH-FDR correction.

Loads results from four data sources:
  1. temp7/           — Paired experiments (neg8dim_12pairs_matched, Qwen 7B, temp=0.7)
  2. abdullah_general_pairs_layers_10_14/ — Unpaired, general-domain pairs, L10/L14
  3. general_damon_12_16_19/             — Unpaired, general-domain pairs, L12/L16/L19
  4. temp03_mindims_v4/                  — Unpaired, game-specific pairs, L10/L14/L18, temp=0.3

For paired data:  paired t-test on per-game demand/acceptance deltas.
For unpaired data: Welch's t-test comparing steered vs baseline distributions.

All p-values collected, then BH-FDR corrected across the full family.

Usage:
    python analysis/compile_exploratory.py [--results_dir results/ultimatum/]
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d_paired(diffs):
    """Cohen's d for paired samples: mean(diffs) / std(diffs)."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2 or np.std(diffs, ddof=1) == 0:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def cohens_d_independent(a, b):
    """Cohen's d for independent samples (pooled std)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Returns array of adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    # Enforce monotonicity (step-up): walk from largest rank down
    idx = np.argsort(ranked)[::-1]
    adjusted_sorted = adjusted[idx]
    for i in range(1, len(adjusted_sorted)):
        if adjusted_sorted[i] > adjusted_sorted[i - 1]:
            adjusted_sorted[i] = adjusted_sorted[i - 1]
    adjusted[idx] = adjusted_sorted
    return np.minimum(adjusted, 1.0)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_paired_temp7(results_dir):
    """Load temp7/ paired experiment files. Returns list of result dicts."""
    temp7_dir = os.path.join(results_dir, "temp7")
    if not os.path.isdir(temp7_dir):
        print(f"  [WARN] {temp7_dir} not found, skipping.")
        return []

    results = []
    for fname in sorted(os.listdir(temp7_dir)):
        if not fname.endswith(".json"):
            continue
        if fname.startswith("baseline"):
            continue

        fpath = os.path.join(temp7_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Must have paired format with config/games
        if "config" not in data or "games" not in data:
            continue

        config = data["config"]
        games = data["games"]
        role = config.get("steered_role", "proposer")
        dim = config.get("dimension", "unknown")
        layers = config.get("layers", [0])
        alpha = config.get("alpha", 0.0)
        layer = layers[0] if layers else 0

        # --- PROPOSER: demand shift ---
        if role == "proposer":
            steered_pcts = []
            baseline_pcts = []
            steered_payoffs = []
            baseline_payoffs = []
            for g in games:
                s = g.get("steered", {})
                b = g.get("baseline", {})
                pool = g.get("pool", 100)
                if pool <= 0:
                    continue
                sp = s.get("proposer_share")
                bp = b.get("proposer_share")
                if sp is None or bp is None:
                    continue
                steered_pcts.append(100.0 * sp / pool)
                baseline_pcts.append(100.0 * bp / pool)
                # Payoff = share if accepted, 0 if rejected
                s_agreed = s.get("agreed", False)
                b_agreed = b.get("agreed", False)
                steered_payoffs.append(100.0 * sp / pool if s_agreed else 0.0)
                baseline_payoffs.append(100.0 * bp / pool if b_agreed else 0.0)

            if len(steered_pcts) < 5:
                continue

            diffs = np.array(steered_pcts) - np.array(baseline_pcts)
            t_stat, p_val = stats.ttest_rel(steered_pcts, baseline_pcts)
            d = cohens_d_paired(diffs)
            delta = float(np.mean(diffs))

            payoff_diffs = np.array(steered_payoffs) - np.array(baseline_payoffs)
            t_pay, p_pay = stats.ttest_rel(steered_payoffs, baseline_payoffs)
            d_pay = cohens_d_paired(payoff_diffs)
            delta_pay = float(np.mean(payoff_diffs))

            results.append({
                "source": "temp7",
                "dim": dim,
                "role": role,
                "layer": layer,
                "alpha": alpha,
                "n": len(steered_pcts),
                "design": "paired",
                "metric": "demand",
                "delta": delta,
                "d": d,
                "p": p_val,
                "payoff_delta": delta_pay,
                "payoff_d": d_pay,
                "payoff_p": p_pay,
                "steered_mean": float(np.mean(steered_pcts)),
                "baseline_mean": float(np.mean(baseline_pcts)),
            })

        # --- RESPONDER: acceptance rate ---
        elif role == "responder":
            steered_accept = []
            baseline_accept = []
            for g in games:
                s = g.get("steered", {})
                b = g.get("baseline", {})
                s_resp = str(s.get("response", "")).lower()
                b_resp = str(b.get("response", "")).lower()
                steered_accept.append(1 if s_resp == "accept" else 0)
                baseline_accept.append(1 if b_resp == "accept" else 0)

            if len(steered_accept) < 5:
                continue

            # Paired: McNemar-like via paired t-test on binary accept
            diffs = np.array(steered_accept) - np.array(baseline_accept)
            t_stat, p_val = stats.ttest_rel(steered_accept, baseline_accept)
            d = cohens_d_paired(diffs)
            delta = float(np.mean(diffs)) * 100  # as percentage points

            results.append({
                "source": "temp7",
                "dim": dim,
                "role": role,
                "layer": layer,
                "alpha": alpha,
                "n": len(steered_accept),
                "design": "paired",
                "metric": "accept_rate",
                "delta": delta,
                "d": d,
                "p": p_val,
                "payoff_delta": np.nan,
                "payoff_d": np.nan,
                "payoff_p": np.nan,
                "steered_mean": float(np.mean(steered_accept)) * 100,
                "baseline_mean": float(np.mean(baseline_accept)) * 100,
            })

    return results


def _extract_demand_pcts(games, role):
    """Extract proposer demand % from unpaired game list."""
    pcts = []
    for g in games:
        pool = g.get("pool", 100)
        if pool <= 0:
            continue
        share = g.get("proposer_share")
        if share is None:
            continue
        if g.get("parse_error", False):
            continue
        pcts.append(100.0 * share / pool)
    return pcts


def _extract_accept_flags(games):
    """Extract binary accept flags from unpaired game list."""
    flags = []
    for g in games:
        if g.get("parse_error", False):
            continue
        decision = str(g.get("decision", "")).lower()
        flags.append(1 if decision == "accept" else 0)
    return flags


def _extract_payoff_pcts(games, role):
    """Extract payoff % from unpaired game list. Payoff = share if agreed, else 0."""
    pcts = []
    for g in games:
        pool = g.get("pool", 100)
        if pool <= 0:
            continue
        if g.get("parse_error", False):
            continue
        agreed = g.get("agreed", False)
        if role == "proposer":
            share = g.get("proposer_share", 0)
        else:
            share = g.get("responder_share", 0)
        pcts.append(100.0 * share / pool if agreed else 0.0)
    return pcts


def load_unpaired_dir(results_dir, subdir, source_label):
    """
    Load unpaired results from a directory with structure:
        subdir/L{layer}/{proposer,responder}/{dim}/games_alpha{X}.json
    Baseline is games_baseline.json in each dim directory.
    Returns list of result dicts.
    """
    base = os.path.join(results_dir, subdir)
    if not os.path.isdir(base):
        print(f"  [WARN] {base} not found, skipping.")
        return []

    results = []
    for layer_dir in sorted(os.listdir(base)):
        if not layer_dir.startswith("L"):
            continue
        layer_match = re.match(r"L(\d+)", layer_dir)
        if not layer_match:
            continue
        layer = int(layer_match.group(1))

        for role in ["proposer", "responder"]:
            role_path = os.path.join(base, layer_dir, role)
            if not os.path.isdir(role_path):
                continue

            for dim in sorted(os.listdir(role_path)):
                dim_path = os.path.join(role_path, dim)
                if not os.path.isdir(dim_path):
                    continue

                # Load baseline
                baseline_path = os.path.join(dim_path, "games_baseline.json")
                if not os.path.isfile(baseline_path):
                    continue
                try:
                    with open(baseline_path) as f:
                        bl_data = json.load(f)
                    bl_games = bl_data.get("games", [])
                except (json.JSONDecodeError, OSError):
                    continue

                if len(bl_games) < 3:
                    continue

                # Load each steered alpha file
                for fname in sorted(os.listdir(dim_path)):
                    if not fname.startswith("games_alpha"):
                        continue
                    if fname == "games_alpha0.0.json":
                        continue  # duplicate of baseline
                    fpath = os.path.join(dim_path, fname)

                    # Parse alpha from filename
                    alpha_match = re.match(r"games_alpha(-?[\d.]+)\.json", fname)
                    if not alpha_match:
                        continue
                    alpha = float(alpha_match.group(1))

                    try:
                        with open(fpath) as f:
                            st_data = json.load(f)
                        st_games = st_data.get("games", [])
                    except (json.JSONDecodeError, OSError):
                        continue

                    if len(st_games) < 3:
                        continue

                    if role == "proposer":
                        steered_vals = _extract_demand_pcts(st_games, role)
                        baseline_vals = _extract_demand_pcts(bl_games, role)
                        metric_name = "demand"

                        steered_pay = _extract_payoff_pcts(st_games, role)
                        baseline_pay = _extract_payoff_pcts(bl_games, role)
                    else:
                        steered_vals = [x * 100 for x in _extract_accept_flags(st_games)]
                        baseline_vals = [x * 100 for x in _extract_accept_flags(bl_games)]
                        metric_name = "accept_rate"

                        steered_pay = _extract_payoff_pcts(st_games, role)
                        baseline_pay = _extract_payoff_pcts(bl_games, role)

                    if len(steered_vals) < 3 or len(baseline_vals) < 3:
                        continue

                    # Welch's t-test (independent samples)
                    t_stat, p_val = stats.ttest_ind(
                        steered_vals, baseline_vals, equal_var=False
                    )
                    d = cohens_d_independent(steered_vals, baseline_vals)
                    delta = float(np.mean(steered_vals) - np.mean(baseline_vals))

                    # Payoff test
                    payoff_delta = np.nan
                    payoff_d = np.nan
                    payoff_p = np.nan
                    if len(steered_pay) >= 3 and len(baseline_pay) >= 3:
                        _, p_pay = stats.ttest_ind(
                            steered_pay, baseline_pay, equal_var=False
                        )
                        payoff_d = cohens_d_independent(steered_pay, baseline_pay)
                        payoff_delta = float(
                            np.mean(steered_pay) - np.mean(baseline_pay)
                        )
                        payoff_p = p_pay

                    results.append({
                        "source": source_label,
                        "dim": dim,
                        "role": role,
                        "layer": layer,
                        "alpha": alpha,
                        "n": len(steered_vals),
                        "n_bl": len(baseline_vals),
                        "design": "unpaired",
                        "metric": metric_name,
                        "delta": delta,
                        "d": d,
                        "p": p_val,
                        "payoff_delta": payoff_delta,
                        "payoff_d": payoff_d,
                        "payoff_p": payoff_p,
                        "steered_mean": float(np.mean(steered_vals)),
                        "baseline_mean": float(np.mean(baseline_vals)),
                    })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compile all UG results with BH-FDR correction."
    )
    parser.add_argument(
        "--results_dir",
        default="results/ultimatum/",
        help="Root directory containing UG result subdirectories.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    print("=" * 80)
    print("ULTIMATUM GAME — EXPLORATORY RESULTS WITH BH-FDR CORRECTION")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # Load all sources
    # ------------------------------------------------------------------
    all_results = []

    print("Loading data sources...")
    r = load_paired_temp7(results_dir)
    print(f"  temp7 (paired, temp=0.7):       {len(r)} configs")
    all_results.extend(r)

    r = load_unpaired_dir(
        results_dir, "abdullah_general_pairs_layers_10_14", "abdullah_general_L10_14"
    )
    print(f"  abdullah_general (L10,L14):      {len(r)} configs")
    all_results.extend(r)

    r = load_unpaired_dir(
        results_dir, "general_damon_12_16_19", "damon_general_L12_16_19"
    )
    print(f"  damon_general (L12,L16,L19):     {len(r)} configs")
    all_results.extend(r)

    r = load_unpaired_dir(
        results_dir, "temp03_mindims_v4", "v4_gamespec_t03"
    )
    print(f"  temp03_mindims_v4 (L10,L14,L18): {len(r)} configs")
    all_results.extend(r)

    print(f"\n  TOTAL configs loaded: {len(all_results)}")

    if not all_results:
        print("\nNo results found. Check --results_dir path.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Collect p-values and apply BH-FDR
    # ------------------------------------------------------------------
    demand_pvals = []
    demand_indices = []
    payoff_pvals = []
    payoff_indices = []

    for i, res in enumerate(all_results):
        p = res["p"]
        if np.isfinite(p):
            demand_pvals.append(p)
            demand_indices.append(i)

        pp = res["payoff_p"]
        if np.isfinite(pp):
            payoff_pvals.append(pp)
            payoff_indices.append(i)

    demand_adj = bh_fdr(np.array(demand_pvals))
    payoff_adj = bh_fdr(np.array(payoff_pvals))

    # Write adjusted p-values back
    for res in all_results:
        res["p_adj"] = np.nan
        res["payoff_p_adj"] = np.nan

    for k, idx in enumerate(demand_indices):
        all_results[idx]["p_adj"] = float(demand_adj[k])
    for k, idx in enumerate(payoff_indices):
        all_results[idx]["payoff_p_adj"] = float(payoff_adj[k])

    # ------------------------------------------------------------------
    # Sort by adjusted p-value
    # ------------------------------------------------------------------
    all_results.sort(key=lambda r: (r["p_adj"] if np.isfinite(r["p_adj"]) else 999))

    # ------------------------------------------------------------------
    # Print demand/accept_rate table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TABLE 1: ALL CONFIGS — sorted by BH-FDR adjusted p-value (demand / accept rate)")
    print("=" * 80)
    header = (
        f"{'Source':<26} {'Dim':<20} {'Role':<10} {'L':>3} {'alpha':>6} "
        f"{'n':>4} {'Design':<8} {'Metric':<12} "
        f"{'Delta%':>7} {'d':>6} {'p_raw':>9} {'p_adj':>9} {'Sig':>4}"
    )
    print(header)
    print("-" * len(header))

    n_demand_sig = 0
    n_payoff_sig = 0

    for res in all_results:
        sig = "***" if res["p_adj"] < 0.001 else (
            "**" if res["p_adj"] < 0.01 else (
                "*" if res["p_adj"] < 0.05 else ""
            )
        )
        if res["p_adj"] < 0.05:
            n_demand_sig += 1

        p_raw_str = f"{res['p']:.2e}" if np.isfinite(res['p']) else "N/A"
        p_adj_str = f"{res['p_adj']:.2e}" if np.isfinite(res['p_adj']) else "N/A"

        print(
            f"{res['source']:<26} {res['dim']:<20} {res['role']:<10} "
            f"{res['layer']:>3} {res['alpha']:>6.1f} "
            f"{res['n']:>4} {res['design']:<8} {res['metric']:<12} "
            f"{res['delta']:>+7.1f} {res['d']:>+6.2f} "
            f"{p_raw_str:>9} {p_adj_str:>9} {sig:>4}"
        )

    # ------------------------------------------------------------------
    # Print payoff table (only configs with finite payoff p-values)
    # ------------------------------------------------------------------
    payoff_results = [r for r in all_results if np.isfinite(r["payoff_p_adj"])]
    payoff_results.sort(
        key=lambda r: r["payoff_p_adj"] if np.isfinite(r["payoff_p_adj"]) else 999
    )

    if payoff_results:
        print("\n" + "=" * 80)
        print("TABLE 2: PAYOFF EFFECTS — sorted by BH-FDR adjusted p-value")
        print("=" * 80)
        header2 = (
            f"{'Source':<26} {'Dim':<20} {'Role':<10} {'L':>3} {'alpha':>6} "
            f"{'n':>4} {'Pay_D%':>7} {'pay_d':>6} {'pay_p':>9} {'pay_adj':>9} {'Sig':>4}"
        )
        print(header2)
        print("-" * len(header2))

        for res in payoff_results:
            sig = "***" if res["payoff_p_adj"] < 0.001 else (
                "**" if res["payoff_p_adj"] < 0.01 else (
                    "*" if res["payoff_p_adj"] < 0.05 else ""
                )
            )
            if res["payoff_p_adj"] < 0.05:
                n_payoff_sig += 1

            pp_raw = f"{res['payoff_p']:.2e}" if np.isfinite(res['payoff_p']) else "N/A"
            pp_adj = (
                f"{res['payoff_p_adj']:.2e}"
                if np.isfinite(res['payoff_p_adj'])
                else "N/A"
            )

            print(
                f"{res['source']:<26} {res['dim']:<20} {res['role']:<10} "
                f"{res['layer']:>3} {res['alpha']:>6.1f} "
                f"{res['n']:>4} "
                f"{res['payoff_delta']:>+7.1f} {res['payoff_d']:>+6.2f} "
                f"{pp_raw:>9} {pp_adj:>9} {sig:>4}"
            )

    # ------------------------------------------------------------------
    # Summary counts
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total configs tested:                           {len(all_results)}")
    print(
        f"  Demand/accept tests with finite p:              {len(demand_pvals)}"
    )
    print(f"  Demand/accept significant at q<0.05 (BH-FDR):  {n_demand_sig}")
    print(f"  Payoff tests with finite p:                     {len(payoff_pvals)}")
    print(f"  Payoff significant at q<0.05 (BH-FDR):         {n_payoff_sig}")

    # Breakdown by source
    print("\n  By source:")
    sources = sorted(set(r["source"] for r in all_results))
    for src in sources:
        src_results = [r for r in all_results if r["source"] == src]
        src_sig = sum(1 for r in src_results if r["p_adj"] < 0.05)
        print(f"    {src:<30} {len(src_results):>4} configs, {src_sig:>3} significant")

    # Breakdown by role
    print("\n  By role:")
    for role in ["proposer", "responder"]:
        role_results = [r for r in all_results if r["role"] == role]
        role_sig = sum(1 for r in role_results if r["p_adj"] < 0.05)
        print(f"    {role:<30} {len(role_results):>4} configs, {role_sig:>3} significant")

    # Breakdown by dimension
    print("\n  By dimension:")
    dims = sorted(set(r["dim"] for r in all_results))
    for dim in dims:
        dim_results = [r for r in all_results if r["dim"] == dim]
        dim_sig = sum(1 for r in dim_results if r["p_adj"] < 0.05)
        if dim_sig > 0 or len(dim_results) >= 3:
            print(
                f"    {dim:<30} {len(dim_results):>4} configs, "
                f"{dim_sig:>3} significant"
            )

    # ------------------------------------------------------------------
    # Layer gradient: mean |d| per layer
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LAYER GRADIENT — mean |Cohen's d| per layer")
    print("=" * 80)
    layers = sorted(set(r["layer"] for r in all_results))
    print(f"  {'Layer':>6}  {'n_configs':>10}  {'mean|d|':>8}  {'n_sig':>6}  {'sig_rate':>9}")
    print(f"  {'-----':>6}  {'--------':>10}  {'-------':>8}  {'-----':>6}  {'--------':>9}")
    for layer in layers:
        lr = [r for r in all_results if r["layer"] == layer]
        mean_abs_d = float(np.mean([abs(r["d"]) for r in lr]))
        n_sig = sum(1 for r in lr if r["p_adj"] < 0.05)
        sig_rate = n_sig / len(lr) if lr else 0
        print(
            f"  L{layer:>4}  {len(lr):>10}  {mean_abs_d:>8.3f}  {n_sig:>6}  "
            f"{sig_rate:>8.1%}"
        )

    # ------------------------------------------------------------------
    # Layer x Role gradient
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LAYER x ROLE GRADIENT — mean |d| per layer per role")
    print("=" * 80)
    print(
        f"  {'Layer':>6}  {'Role':<10}  {'n':>4}  {'mean|d|':>8}  "
        f"{'n_sig':>6}  {'mean_delta':>11}"
    )
    print(
        f"  {'-----':>6}  {'----':<10}  {'--':>4}  {'-------':>8}  "
        f"{'-----':>6}  {'----------':>11}"
    )
    for layer in layers:
        for role in ["proposer", "responder"]:
            lr = [
                r for r in all_results
                if r["layer"] == layer and r["role"] == role
            ]
            if not lr:
                continue
            mean_abs_d = float(np.mean([abs(r["d"]) for r in lr]))
            n_sig = sum(1 for r in lr if r["p_adj"] < 0.05)
            mean_delta = float(np.mean([r["delta"] for r in lr]))
            print(
                f"  L{layer:>4}  {role:<10}  {len(lr):>4}  {mean_abs_d:>8.3f}  "
                f"{n_sig:>6}  {mean_delta:>+11.1f}"
            )

    # ------------------------------------------------------------------
    # Top effects (significant after FDR)
    # ------------------------------------------------------------------
    sig_results = [r for r in all_results if r["p_adj"] < 0.05]
    if sig_results:
        sig_results.sort(key=lambda r: -abs(r["d"]))
        print("\n" + "=" * 80)
        print(f"TOP EFFECTS (q<0.05 after BH-FDR, sorted by |d|) — {len(sig_results)} total")
        print("=" * 80)
        for i, res in enumerate(sig_results[:30]):
            print(
                f"  {i+1:>3}. {res['source']:<26} {res['dim']:<20} "
                f"{res['role']:<10} L{res['layer']} a={res['alpha']:>5.1f}  "
                f"delta={res['delta']:>+6.1f}%  d={res['d']:>+5.2f}  "
                f"q={res['p_adj']:.3e}"
            )
        if len(sig_results) > 30:
            print(f"  ... and {len(sig_results) - 30} more.")

    # ------------------------------------------------------------------
    # Comparison: raw vs corrected significance
    # ------------------------------------------------------------------
    n_raw_sig = sum(
        1 for r in all_results if np.isfinite(r["p"]) and r["p"] < 0.05
    )
    print("\n" + "=" * 80)
    print("RAW vs CORRECTED SIGNIFICANCE")
    print("=" * 80)
    print(f"  Raw p<0.05:        {n_raw_sig} / {len(demand_pvals)}")
    print(f"  BH-FDR q<0.05:     {n_demand_sig} / {len(demand_pvals)}")
    print(
        f"  Survived correction: {n_demand_sig}/{n_raw_sig} "
        f"({100*n_demand_sig/max(n_raw_sig,1):.0f}%)"
    )

    n_raw_pay = sum(
        1 for r in all_results
        if np.isfinite(r["payoff_p"]) and r["payoff_p"] < 0.05
    )
    print(f"\n  Payoff raw p<0.05:   {n_raw_pay} / {len(payoff_pvals)}")
    print(f"  Payoff BH-FDR q<0.05: {n_payoff_sig} / {len(payoff_pvals)}")

    print("\n" + "=" * 80)
    print("DONE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
