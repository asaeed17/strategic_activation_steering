#!/usr/bin/env python3
"""
analysis/analyse_resource_exchange.py — Post-GPU analysis for Resource Exchange Game.

CPU-only. Reads JSON results from results/resource_exchange/qwen2.5-32b-gptq/,
pairs steered games with baseline by config index, runs paired t-tests and
computes Cohen's d.

Usage:
    python analysis/analyse_resource_exchange.py \
        --results_dir results/resource_exchange/qwen2.5-32b-gptq
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(path: Path) -> Dict:
    """Load a single run's results.json."""
    with open(path) as f:
        return json.load(f)


def parse_config_name(name: str) -> Dict:
    """Parse directory name like 'firmness_P1_L23_a20' into components."""
    m = re.match(r"(\w+)_P(\d+)_L(\d+)_a(\d+)", name)
    if not m:
        return {}
    return {
        "dimension": m.group(1),
        "player": int(m.group(2)),
        "layer": int(m.group(3)),
        "alpha": int(m.group(4)),
    }


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------

def paired_analysis(
    baseline_games: List[Dict],
    steered_games: List[Dict],
    steered_player: int = 1,
) -> Dict:
    """Compute paired differences and statistics.

    Games are paired by index (same RESOURCE_CONFIGS ordering).
    """
    n = min(len(baseline_games), len(steered_games))
    bl_scores = np.array([g[f"p{steered_player}_score"] for g in baseline_games[:n]])
    st_scores = np.array([g[f"p{steered_player}_score"] for g in steered_games[:n]])

    bl_trades = np.array([g["trades_completed"] for g in baseline_games[:n]])
    st_trades = np.array([g["trades_completed"] for g in steered_games[:n]])

    bl_lengths = np.array([g["game_length"] for g in baseline_games[:n]])
    st_lengths = np.array([g["game_length"] for g in steered_games[:n]])

    # Paired differences
    score_diffs = st_scores - bl_scores
    mean_diff = np.mean(score_diffs)
    std_diff = np.std(score_diffs, ddof=1)

    # Paired t-test
    if std_diff > 0:
        t_stat, p_value = stats.ttest_rel(st_scores, bl_scores)
        cohens_d = mean_diff / std_diff
    else:
        t_stat, p_value, cohens_d = 0.0, 1.0, 0.0

    return {
        "n": n,
        "baseline_mean_score": float(np.mean(bl_scores)),
        "steered_mean_score": float(np.mean(st_scores)),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "baseline_acceptance": float(np.mean(bl_trades > 0)),
        "steered_acceptance": float(np.mean(st_trades > 0)),
        "baseline_mean_length": float(np.mean(bl_lengths)),
        "steered_mean_length": float(np.mean(st_lengths)),
    }


# ---------------------------------------------------------------------------
# Behavioral metrics
# ---------------------------------------------------------------------------

def extract_player_metrics(games: List[Dict], player: int) -> Dict:
    """Extract mean behavioral metrics for a specific player across games."""
    word_counts = []
    hedge_counts = []
    fairness_counts = []
    for g in games:
        for t in g.get("transcript", []):
            if t.get("player") == player and t.get("metrics"):
                m = t["metrics"]
                word_counts.append(m.get("word_count", 0))
                hedge_counts.append(m.get("hedge_count", 0))
                fairness_counts.append(m.get("fairness_count", 0))
    return {
        "mean_word_count": float(np.mean(word_counts)) if word_counts else 0,
        "mean_hedge_count": float(np.mean(hedge_counts)) if hedge_counts else 0,
        "mean_fairness_count": float(np.mean(fairness_counts)) if fairness_counts else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path,
                        default=Path("results/resource_exchange/qwen2.5-32b-gptq"))
    parser.add_argument("--player", type=int, default=1,
                        help="Which player to analyse (steered player)")
    args = parser.parse_args()

    results_dir = args.results_dir
    player = args.player

    # Load baseline
    baseline_path = results_dir / "baseline" / "results.json"
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found at {baseline_path}")
        return
    baseline = load_run(baseline_path)
    bl_games = baseline["games"]
    print(f"Baseline: {len(bl_games)} games, "
          f"mean score={np.mean([g[f'p{player}_score'] for g in bl_games]):.1f}, "
          f"acceptance={np.mean([g['trades_completed'] > 0 for g in bl_games]):.0%}")
    print()

    # Find all steered configs
    configs = []
    for d in sorted(results_dir.iterdir()):
        if d.name == "baseline" or not d.is_dir():
            continue
        rpath = d / "results.json"
        if not rpath.exists():
            continue
        info = parse_config_name(d.name)
        if not info:
            continue
        info["path"] = rpath
        info["dir_name"] = d.name
        configs.append(info)

    if not configs:
        print("No steered configs found.")
        return

    # Run analysis for each config
    all_results = []
    for cfg in configs:
        steered = load_run(cfg["path"])
        st_games = steered["games"]
        result = paired_analysis(bl_games, st_games, player)
        result.update(cfg)
        result.pop("path", None)

        # Behavioral metrics
        bl_metrics = extract_player_metrics(bl_games, player)
        st_metrics = extract_player_metrics(st_games, player)
        result["bl_word_count"] = bl_metrics["mean_word_count"]
        result["st_word_count"] = st_metrics["mean_word_count"]
        result["bl_hedge_count"] = bl_metrics["mean_hedge_count"]
        result["st_hedge_count"] = st_metrics["mean_hedge_count"]
        result["bl_fairness_count"] = bl_metrics["mean_fairness_count"]
        result["st_fairness_count"] = st_metrics["mean_fairness_count"]

        all_results.append(result)

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------
    print("=" * 110)
    print(f"{'Config':<30} {'BL':>6} {'ST':>6} {'Δ':>7} {'d':>6} {'p':>8} {'sig':>4} {'BL%':>5} {'ST%':>5}")
    print("=" * 110)

    # Group by dimension
    dims = sorted(set(r["dimension"] for r in all_results))
    sig_count = 0
    for dim in dims:
        dim_results = sorted(
            [r for r in all_results if r["dimension"] == dim],
            key=lambda r: (r["layer"], r["alpha"]),
        )
        for r in dim_results:
            sig = ""
            if r["p_value"] < 0.001:
                sig = "***"
            elif r["p_value"] < 0.01:
                sig = "**"
            elif r["p_value"] < 0.05:
                sig = "*"

            if sig:
                sig_count += 1

            label = f"{dim} L{r['layer']} α={r['alpha']}"
            print(
                f"{label:<30} "
                f"{r['baseline_mean_score']:>6.1f} "
                f"{r['steered_mean_score']:>6.1f} "
                f"{r['mean_diff']:>+7.1f} "
                f"{r['cohens_d']:>+6.2f} "
                f"{r['p_value']:>8.4f} "
                f"{sig:>4} "
                f"{r['baseline_acceptance']:>5.0%} "
                f"{r['steered_acceptance']:>5.0%}"
            )
        print("-" * 110)

    print()
    print(f"Significant (p<0.05): {sig_count}/{len(all_results)}")

    # ---------------------------------------------------------------------------
    # BH-FDR correction
    # ---------------------------------------------------------------------------
    p_values = np.array([r["p_value"] for r in all_results])
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    bh_threshold = np.array([(i + 1) / n_tests * 0.05 for i in range(n_tests)])
    sorted_p = p_values[sorted_idx]
    bh_sig = sorted_p <= bh_threshold
    # Find largest k where p_(k) <= k/m * 0.05
    if np.any(bh_sig):
        max_k = np.max(np.where(bh_sig)[0])
        bh_significant = sorted_idx[:max_k + 1]
    else:
        bh_significant = np.array([], dtype=int)

    print(f"Significant after BH-FDR (α=0.05): {len(bh_significant)}/{n_tests}")

    if len(bh_significant) > 0:
        print("\nBH-FDR significant configs:")
        for idx in bh_significant:
            r = all_results[idx]
            print(f"  {r['dimension']} L{r['layer']} α={r['alpha']}: "
                  f"Δ={r['mean_diff']:+.1f}, d={r['cohens_d']:+.2f}, p={r['p_value']:.4f}")

    # ---------------------------------------------------------------------------
    # Best configs per dimension
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Best config per dimension (by |Cohen's d|):")
    print("=" * 80)
    for dim in dims:
        dim_results = [r for r in all_results if r["dimension"] == dim]
        best = max(dim_results, key=lambda r: abs(r["cohens_d"]))
        print(f"  {dim:<15} L{best['layer']} α={best['alpha']}: "
              f"Δ={best['mean_diff']:+.1f}, d={best['cohens_d']:+.2f}, "
              f"p={best['p_value']:.4f}, accept={best['steered_acceptance']:.0%}")

    # ---------------------------------------------------------------------------
    # Layer comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Mean |d| by layer (across dims and alphas):")
    print("=" * 80)
    layers = sorted(set(r["layer"] for r in all_results))
    for layer in layers:
        layer_results = [r for r in all_results if r["layer"] == layer]
        mean_abs_d = np.mean([abs(r["cohens_d"]) for r in layer_results])
        n_sig = sum(1 for r in layer_results if r["p_value"] < 0.05)
        print(f"  L{layer}: mean |d|={mean_abs_d:.2f}, "
              f"significant={n_sig}/{len(layer_results)}")

    # ---------------------------------------------------------------------------
    # Alpha comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Mean |d| by alpha (across dims and layers):")
    print("=" * 80)
    alphas = sorted(set(r["alpha"] for r in all_results))
    for alpha in alphas:
        alpha_results = [r for r in all_results if r["alpha"] == alpha]
        mean_abs_d = np.mean([abs(r["cohens_d"]) for r in alpha_results])
        n_sig = sum(1 for r in alpha_results if r["p_value"] < 0.05)
        print(f"  α={alpha}: mean |d|={mean_abs_d:.2f}, "
              f"significant={n_sig}/{len(alpha_results)}")

    # ---------------------------------------------------------------------------
    # Behavioral metrics summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Behavioral metrics (steered player, mean across games):")
    print("=" * 80)
    print(f"{'Config':<30} {'Words':>8} {'Hedges':>8} {'Fair':>8}")
    print("-" * 60)
    print(f"{'baseline':<30} "
          f"{all_results[0]['bl_word_count']:>8.1f} "
          f"{all_results[0]['bl_hedge_count']:>8.2f} "
          f"{all_results[0]['bl_fairness_count']:>8.2f}")
    for dim in dims:
        dim_results = sorted(
            [r for r in all_results if r["dimension"] == dim],
            key=lambda r: (r["layer"], r["alpha"]),
        )
        for r in dim_results:
            label = f"{dim} L{r['layer']} α={r['alpha']}"
            print(f"{label:<30} "
                  f"{r['st_word_count']:>8.1f} "
                  f"{r['st_hedge_count']:>8.2f} "
                  f"{r['st_fairness_count']:>8.2f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
