#!/usr/bin/env python3
"""
analysis/analyse_ultimatum.py — Post-GPU analysis for Ultimatum Game steering experiments.

CPU-only. Reads JSON results from results/ultimatum/, produces:
- Paired t-tests / McNemar's tests per config
- Effect size (Cohen's d) comparison table
- Dose-response curves (alpha vs effect)
- Behavioral metrics comparison
- Prompt engineering vs activation steering comparison table

Usage:
  python analysis/analyse_ultimatum.py --results_dir results/ultimatum/
  python analysis/analyse_ultimatum.py --results_dir results/ultimatum/ --stage 1
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def load_results(results_dir: Path) -> List[Dict]:
    """Load all JSON result files from the directory."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        data["_filename"] = f.name
        results.append(data)
    return results


def classify_result(data: Dict) -> str:
    """Classify a result file as baseline, paired, or single."""
    config = data.get("config", {})
    if config.get("dimension") is None:
        return "baseline"
    if config.get("paired"):
        return "paired"
    return "single"


# ---------------------------------------------------------------------------
# Prompt engineering baselines (from playground experiments on Gemini)
# ---------------------------------------------------------------------------

PROMPT_BASELINES = {
    # dimension -> {role -> {metric -> value}}
    "firmness": {
        "proposer": {"delta_proposer_pct": 6.6, "accept_rate": 1.0, "mean_payoff_pct": 69.1},
        "responder": {"delta_accept_rate": -0.60},  # 10% → 70% rejection = -60% accept
    },
    "anchoring": {
        "proposer": {"delta_proposer_pct": 20.6, "accept_rate": 0.40, "mean_payoff_pct": 31.4},
    },
    "empathy": {
        "proposer": {"delta_proposer_pct": -10.3, "accept_rate": 1.0, "mean_payoff_pct": 52.2},
        "responder": {"delta_accept_rate": 0.10},  # 90% → 100%
    },
    "batna_awareness": {
        "proposer": {"delta_proposer_pct": 15.5, "accept_rate": 0.50, "mean_payoff_pct": 35.7},
    },
}


def print_separator(title: str = "") -> None:
    print(f"\n{'=' * 70}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 70}")


def analyse_stage1(results: List[Dict]) -> None:
    """Analyse Stage 1 quick scan results."""
    paired = [r for r in results if classify_result(r) == "paired"]
    baselines = [r for r in results if classify_result(r) == "baseline"]

    if not paired:
        print("No paired results found for Stage 1 analysis.")
        return

    # Baseline stats
    if baselines:
        b = baselines[0]["summary"]
        print_separator("BASELINE")
        print(f"  Mean proposer %: {b.get('mean_proposer_pct', '?')}% "
              f"(std={b.get('std_proposer_pct', '?')}%)")
        print(f"  Accept rate: {b.get('accept_rate', '?')}")

    # Summary table: all configs sorted by p-value
    print_separator("STAGE 1: ALL CONFIGS (sorted by p-value)")

    rows = []
    for r in paired:
        c = r["config"]
        s = r["summary"]
        dim = c.get("dimension", "?")
        role = c.get("steered_role", "?")
        layers = c.get("layers", [])
        alpha = c.get("alpha", 0)

        if role == "proposer":
            delta = s.get("delta_proposer_pct", 0)
            p_val = s.get("paired_ttest_p", 1.0)
            d_val = s.get("cohens_d", 0)
            n = s.get("n_usable_pairs", 0)
            metric_str = f"Δ={delta:+.1f}%"
        else:
            delta = s.get("delta_accept_rate", 0)
            p_val = s.get("mcnemar_p", 1.0)
            d_val = 0  # no Cohen's d for McNemar
            n = s.get("n_usable_pairs", 0)
            metric_str = f"Δacc={delta:+.1%}"

        rows.append((p_val, dim, role, layers, alpha, metric_str, d_val, n, r["_filename"]))

    rows.sort(key=lambda x: x[0])

    print(f"  {'Dim':<20s} {'Role':<10s} {'L':>3s} {'α':>5s} {'Effect':>12s} "
          f"{'d':>6s} {'p':>8s} {'n':>4s} {'Sig':>4s}")
    print(f"  {'-'*20} {'-'*10} {'-'*3} {'-'*5} {'-'*12} {'-'*6} {'-'*8} {'-'*4} {'-'*4}")

    for p_val, dim, role, layers, alpha, metric_str, d_val, n, fname in rows:
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "." if p_val < 0.10 else ""
        layer_str = ",".join(str(l) for l in layers)
        print(f"  {dim:<20s} {role:<10s} {layer_str:>3s} {alpha:>5.0f} {metric_str:>12s} "
              f"{d_val:>6.2f} {p_val:>8.4f} {n:>4d} {sig:>4s}")

    # Go/no-go decision
    significant = [r for r in rows if r[0] < 0.10]
    print_separator("GO/NO-GO DECISION")
    if significant:
        print(f"  {len(significant)} config(s) with p < 0.10. PROCEED to Stage 2.")
        print(f"  Top config: {significant[0][1]} {significant[0][2]} "
              f"L{significant[0][3]} α={significant[0][4]}")
    else:
        print(f"  NO configs with p < 0.10. STOP — report negative result.")
        print(f"  'Steering vectors do not transfer to UG at tested configs.'")


def analyse_comparison(results: List[Dict]) -> None:
    """Compare activation steering vs prompt engineering."""
    paired = [r for r in results if classify_result(r) == "paired"]

    if not paired:
        return

    print_separator("PROMPT ENGINEERING vs ACTIVATION STEERING")
    print(f"  {'Dimension':<20s} {'Role':<10s} {'Prompt Δ':>10s} {'Steer Δ':>10s} "
          f"{'Steer d':>8s} {'Steer p':>8s} {'Match?':>7s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")

    # Group by (dimension, role), take best config (lowest p)
    from collections import defaultdict
    best_by_dimrole = defaultdict(lambda: (1.0, None, None))

    for r in paired:
        c = r["config"]
        s = r["summary"]
        dim = c.get("dimension", "?")
        role = c.get("steered_role", "?")
        key = (dim, role)

        if role == "proposer":
            p_val = s.get("paired_ttest_p", 1.0)
            delta = s.get("delta_proposer_pct", 0)
            d_val = s.get("cohens_d", 0)
        else:
            p_val = s.get("mcnemar_p", 1.0)
            delta = s.get("delta_accept_rate", 0)
            d_val = 0

        if p_val < best_by_dimrole[key][0]:
            best_by_dimrole[key] = (p_val, delta, d_val)

    for (dim, role), (p_val, steer_delta, steer_d) in sorted(best_by_dimrole.items()):
        prompt_data = PROMPT_BASELINES.get(dim, {}).get(role, {})
        if role == "proposer":
            prompt_delta = prompt_data.get("delta_proposer_pct", None)
            prompt_str = f"{prompt_delta:+.1f}%" if prompt_delta is not None else "N/A"
            steer_str = f"{steer_delta:+.1f}%"
        else:
            prompt_delta = prompt_data.get("delta_accept_rate", None)
            prompt_str = f"{prompt_delta:+.1%}" if prompt_delta is not None else "N/A"
            steer_str = f"{steer_delta:+.1%}"

        # Direction match
        if prompt_delta is not None and steer_delta != 0:
            match = "YES" if (prompt_delta > 0) == (steer_delta > 0) else "NO"
        else:
            match = "?"

        print(f"  {dim:<20s} {role:<10s} {prompt_str:>10s} {steer_str:>10s} "
              f"{steer_d:>8.2f} {p_val:>8.4f} {match:>7s}")


def analyse_dose_response(results: List[Dict]) -> None:
    """Analyse dose-response (alpha vs effect) per dimension."""
    paired = [r for r in results if classify_result(r) == "paired"
              and r["config"].get("steered_role") == "proposer"]

    if not paired:
        return

    from collections import defaultdict
    by_dim_layer = defaultdict(list)

    for r in paired:
        c = r["config"]
        s = r["summary"]
        dim = c["dimension"]
        layers = tuple(c["layers"])
        alpha = c["alpha"]
        delta = s.get("delta_proposer_pct", 0)
        by_dim_layer[(dim, layers)].append((alpha, delta))

    print_separator("DOSE-RESPONSE (alpha vs Δ proposer %)")

    for (dim, layers), points in sorted(by_dim_layer.items()):
        if len(points) < 3:
            continue
        points.sort()
        alphas = [p[0] for p in points]
        deltas = [p[1] for p in points]

        # Spearman correlation
        if len(set(alphas)) > 1:
            rho, sp_p = stats.spearmanr(alphas, deltas)
        else:
            rho, sp_p = 0, 1.0

        layer_str = ",".join(str(l) for l in layers)
        monotonic = "MONOTONIC" if sp_p < 0.10 and rho > 0.5 else \
                    "INV-MONO" if sp_p < 0.10 and rho < -0.5 else "FLAT"

        print(f"\n  {dim} L{layer_str}: ρ={rho:+.2f} (p={sp_p:.3f}) [{monotonic}]")
        for a, d in points:
            bar = "+" * max(0, int(d / 2)) if d > 0 else "-" * max(0, int(-d / 2))
            print(f"    α={a:>5.1f}  Δ={d:+6.1f}%  {bar}")


def analyse_behavioral(results: List[Dict]) -> None:
    """Analyse behavioral metrics (word count, hedging, fairness language)."""
    paired = [r for r in results if classify_result(r) == "paired"
              and r["config"].get("steered_role") == "proposer"]

    if not paired:
        return

    print_separator("BEHAVIORAL METRICS (proposer-steered configs)")
    print(f"  {'Dimension':<20s} {'L':>3s} {'α':>5s} "
          f"{'S_wc':>5s} {'B_wc':>5s} {'S_hdg':>6s} {'B_hdg':>6s} "
          f"{'S_fair':>6s} {'B_fair':>6s}")
    print(f"  {'-'*20} {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for r in paired:
        c = r["config"]
        s = r["summary"]
        dim = c["dimension"]
        layer_str = ",".join(str(l) for l in c["layers"])
        alpha = c["alpha"]

        print(f"  {dim:<20s} {layer_str:>3s} {alpha:>5.0f} "
              f"{s.get('steered_mean_word_count', 0):>5.1f} "
              f"{s.get('baseline_mean_word_count', 0):>5.1f} "
              f"{s.get('steered_mean_hedge_count', 0):>6.2f} "
              f"{s.get('baseline_mean_hedge_count', 0):>6.2f} "
              f"{s.get('steered_mean_fairness_count', 0):>6.2f} "
              f"{s.get('baseline_mean_fairness_count', 0):>6.2f}")


def main():
    p = argparse.ArgumentParser(description="Analyse Ultimatum Game steering results.")
    p.add_argument("--results_dir", default="results/ultimatum/",
                   help="Directory with result JSON files.")
    p.add_argument("--stage", type=int, default=None,
                   help="Focus on specific stage (1 or 2). Default: all.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files from {results_dir}")

    n_baseline = sum(1 for r in results if classify_result(r) == "baseline")
    n_paired = sum(1 for r in results if classify_result(r) == "paired")
    n_single = sum(1 for r in results if classify_result(r) == "single")
    print(f"  Baseline: {n_baseline}  Paired: {n_paired}  Single: {n_single}")

    if args.stage == 1:
        analyse_stage1(results)
    else:
        analyse_stage1(results)
        analyse_comparison(results)
        analyse_dose_response(results)
        analyse_behavioral(results)


if __name__ == "__main__":
    main()
