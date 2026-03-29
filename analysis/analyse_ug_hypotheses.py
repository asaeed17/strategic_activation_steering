#!/usr/bin/env python3
"""
analysis/analyse_confirmatory.py -- Confirmatory analysis for Ultimatum Game steering experiments.

Pre-registered hypotheses (5 tests, BH-FDR corrected):
  H1: Empathy shifts proposer demand downward    (paired t-test, one-sided)
  H2: Firmness shifts proposer demand upward      (paired t-test, one-sided)
  H3: Empathy improves proposer expected payoff   (paired t-test, one-sided)
  H4: Firmness does NOT improve proposer payoff   (TOST equivalence, |delta| < epsilon)
  H5: Firmness demand shift is equal in UG and DG (TOST equivalence, |UG - DG| < epsilon)

Data format (paired JSON files from ultimatum_game.py):
  {
    "config": {"dimension": ..., "layers": [...], "alpha": ..., "steered_role": "proposer", "paired": true, ...},
    "summary": {...},
    "games": [
      {"game_id": 0, "pool": 131,
       "steered": {"proposer_share": 90, "agreed": true, "proposer_payoff": 90, ...},
       "baseline": {"proposer_share": 75, "agreed": true, "proposer_payoff": 75, ...},
       "parse_error": null},
      ...
    ]
  }

Usage:
  python analysis/analyse_confirmatory.py --results_dir results/ultimatum/temp7
  python analysis/analyse_confirmatory.py --results_dir results/ultimatum/temp7 --dg_dir results/dictator/temp7
  python analysis/analyse_confirmatory.py --results_dir results/ultimatum/temp7 --epsilon 3.0

CPU-only. Deps: numpy, scipy.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Confirmatory config grid
# ---------------------------------------------------------------------------

CONFIRMATORY_CONFIGS = {
    # (dimension, game_type) -> [(layer, alpha), ...]
    # NOTE: empathy uses NEGATIVE alpha (demand DOWN = anti-empathy direction)
    ("empathy", "ultimatum"): [
        (10, -3.0), (10, -7.0), (10, -10.0),
        (12, -3.0), (12, -7.0), (12, -10.0),
    ],
    ("firmness", "ultimatum"): [
        (10, 3.0), (10, 7.0), (10, 10.0),
        (12, 3.0), (12, 7.0), (12, 10.0),
    ],
    ("firmness", "dictator"): [
        (10, 3.0), (10, 7.0), (10, 10.0),
        (12, 3.0), (12, 7.0), (12, 10.0),
    ],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_confirmatory_results(
    results_dir: Path,
    configs: List[Tuple[str, int, float]],
) -> Dict[Tuple[str, int, float], Dict]:
    """Load result files matching specific (dimension, layer, alpha) configs.

    Searches for files matching naming convention:
      {dim}_proposer_L{layer}_a{alpha}_paired_n*.json

    Returns dict keyed by (dimension, layer, alpha) -> parsed JSON.
    """
    loaded = {}
    for dim, layer, alpha in configs:
        # Try common naming patterns (with and without dg_ prefix)
        patterns = [
            f"{dim}_proposer_L{layer}_a{alpha}_paired_n*.json",
            f"{dim}_proposer_L{layer}_a{alpha:.1f}_paired_n*.json",
            f"dg_{dim}_proposer_L{layer}_a{alpha}_paired_n*.json",
            f"dg_{dim}_proposer_L{layer}_a{alpha:.1f}_paired_n*.json",
        ]
        found = False
        for pat in patterns:
            matches = sorted(results_dir.glob(pat))
            if matches:
                # Take the file with the largest n (last alphabetically)
                fpath = matches[-1]
                with open(fpath) as fh:
                    data = json.load(fh)
                data["_filepath"] = str(fpath)
                loaded[(dim, layer, alpha)] = data
                found = True
                break

        if not found:
            # Fallback: scan all JSON files and match on config fields
            for fpath in sorted(results_dir.glob("*.json")):
                try:
                    with open(fpath) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                cfg = data.get("config", {})
                if (cfg.get("dimension") == dim
                        and cfg.get("layers") == [layer]
                        and abs(cfg.get("alpha", -999) - alpha) < 0.01
                        and cfg.get("steered_role") == "proposer"
                        and cfg.get("paired")):
                    data["_filepath"] = str(fpath)
                    loaded[(dim, layer, alpha)] = data
                    break

    return loaded


def load_all_paired_results(results_dir: Path) -> List[Dict]:
    """Load all paired proposer-steered result files from a directory tree.

    Walks subdirectories to handle both flat and nested layouts.
    """
    results = []
    for fpath in sorted(results_dir.rglob("*.json")):
        try:
            with open(fpath) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        cfg = data.get("config", {})
        if cfg.get("paired") and cfg.get("steered_role") == "proposer":
            data["_filepath"] = str(fpath)
            results.append(data)
    return results


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _filter_usable_games(games: List[Dict]) -> List[Dict]:
    """Keep games with no parse errors and valid data in both arms."""
    usable = []
    for g in games:
        if g.get("parse_error"):
            continue
        s = g.get("steered", {})
        b = g.get("baseline", {})
        if s.get("parse_error") or b.get("parse_error"):
            continue
        pool = g.get("pool", 0)
        if pool <= 0:
            continue
        # Both arms must have valid proposer_share
        if s.get("proposer_share") is None or b.get("proposer_share") is None:
            continue
        usable.append(g)
    return usable


def compute_demand_shift(games: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired demand percentages (proposer_share / pool * 100).

    Returns (steered_pct, baseline_pct) arrays aligned by game_id.
    """
    usable = _filter_usable_games(games)
    steered_pct = []
    baseline_pct = []
    for g in usable:
        pool = g["pool"]
        steered_pct.append(g["steered"]["proposer_share"] / pool * 100)
        baseline_pct.append(g["baseline"]["proposer_share"] / pool * 100)
    return np.array(steered_pct), np.array(baseline_pct)


def compute_payoff_delta(games: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired payoff percentages (proposer_payoff / pool * 100).

    Payoff = proposer_share if agreed, else 0. Uses the payoff field directly.
    Returns (steered_pct, baseline_pct) arrays aligned by game_id.
    """
    usable = _filter_usable_games(games)
    steered_pct = []
    baseline_pct = []
    for g in usable:
        pool = g["pool"]
        steered_pct.append(g["steered"].get("proposer_payoff", 0) / pool * 100)
        baseline_pct.append(g["baseline"].get("proposer_payoff", 0) / pool * 100)
    return np.array(steered_pct), np.array(baseline_pct)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def one_sided_paired_ttest(
    steered: np.ndarray,
    baseline: np.ndarray,
    alternative: str = "greater",
) -> Tuple[float, float, float]:
    """Paired t-test with one-sided alternative.

    Args:
        steered: steered condition values
        baseline: baseline condition values
        alternative: "greater" (steered > baseline) or "less" (steered < baseline)

    Returns:
        (t_statistic, p_value, cohens_d)
    """
    diff = steered - baseline
    n = len(diff)
    if n < 2:
        return np.nan, 1.0, 0.0

    t_stat, p_two = stats.ttest_rel(steered, baseline)

    if alternative == "greater":
        p_val = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    elif alternative == "less":
        p_val = p_two / 2 if t_stat < 0 else 1 - p_two / 2
    else:
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")

    # Cohen's d for paired samples (d_z)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    return float(t_stat), float(p_val), float(d)


def tost_equivalence(
    steered: np.ndarray,
    baseline: np.ndarray,
    epsilon: float,
) -> Tuple[float, float, float, float, float, bool]:
    """Two One-Sided Tests (TOST) for equivalence.

    Tests H0: |mean(steered - baseline)| >= epsilon
    vs    H1: |mean(steered - baseline)| < epsilon

    Args:
        steered: steered condition values
        baseline: baseline condition values
        epsilon: equivalence margin (same units as data, e.g. % of pool)

    Returns:
        (t_lower, p_lower, t_upper, p_upper, p_tost, equivalent)
        where p_tost = max(p_lower, p_upper), equivalent = p_tost < 0.05
    """
    diff = steered - baseline
    n = len(diff)
    if n < 2:
        return np.nan, 1.0, np.nan, 1.0, 1.0, False

    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    df = n - 1

    if se == 0:
        # Perfect equality
        equiv = abs(mean_diff) < epsilon
        return 0.0, 0.0, 0.0, 0.0, 0.0, equiv

    # Lower test: H0: mean_diff <= -epsilon  =>  t_lower = (mean_diff + epsilon) / se
    t_lower = (mean_diff + epsilon) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)  # one-sided: P(T > t) under H0

    # Upper test: H0: mean_diff >= epsilon  =>  t_upper = (mean_diff - epsilon) / se
    t_upper = (mean_diff - epsilon) / se
    p_upper = stats.t.cdf(t_upper, df)  # one-sided: P(T < t) under H0

    # TOST p-value is the larger of the two
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    return float(t_lower), float(p_lower), float(t_upper), float(p_upper), float(p_tost), equivalent


def bh_fdr_correction(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: list of raw p-values

    Returns:
        list of adjusted p-values (same order as input)
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    # BH procedure: p_adj[i] = min(p[i] * m / rank, 1.0), enforced monotonicity
    prev_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p_val = indexed[rank_idx]
        rank = rank_idx + 1  # 1-based rank
        adj = min(p_val * m / rank, 1.0)
        adj = min(adj, prev_adj)  # enforce monotonicity
        adjusted[orig_idx] = adj
        prev_adj = adj

    return adjusted


# ---------------------------------------------------------------------------
# Dose-response analysis
# ---------------------------------------------------------------------------

def analyse_dose_response(results_by_config: Dict[Tuple[str, int, float], Dict]) -> None:
    """For each dimension x layer, compute Spearman rho of alpha vs demand shift."""
    by_dim_layer = defaultdict(list)

    for (dim, layer, alpha), data in results_by_config.items():
        games = data.get("games", [])
        steered, baseline = compute_demand_shift(games)
        if len(steered) == 0:
            continue
        mean_shift = float(np.mean(steered - baseline))
        by_dim_layer[(dim, layer)].append((alpha, mean_shift, len(steered)))

    if not by_dim_layer:
        print("  No dose-response data available.")
        return

    for (dim, layer), points in sorted(by_dim_layer.items()):
        if len(points) < 3:
            continue
        points.sort(key=lambda x: x[0])
        alphas = [p[0] for p in points]
        shifts = [p[1] for p in points]

        if len(set(alphas)) > 1:
            rho, sp_p = stats.spearmanr(alphas, shifts)
        else:
            rho, sp_p = 0.0, 1.0

        tag = "MONOTONIC" if sp_p < 0.10 and rho > 0.5 else \
              "INV-MONO" if sp_p < 0.10 and rho < -0.5 else "FLAT"

        print(f"\n  {dim} L{layer}: rho={rho:+.3f} (p={sp_p:.3f}) [{tag}]")
        for a, s, n in points:
            bar = "+" * max(0, int(s / 2)) if s > 0 else "-" * max(0, int(-s / 2))
            print(f"    alpha={a:>5.1f}  shift={s:+6.1f}%  n={n:>3d}  {bar}")


# ---------------------------------------------------------------------------
# UG vs DG comparison (H5)
# ---------------------------------------------------------------------------

def analyse_ug_vs_dg(
    ug_results: Dict[Tuple[str, int, float], Dict],
    dg_results: Dict[Tuple[str, int, float], Dict],
    epsilon: float,
) -> List[Dict]:
    """Compare firmness demand shifts between UG and DG conditions.

    For each matching (layer, alpha) config, compute paired demand shifts
    in both games and run TOST equivalence test.

    Returns list of result dicts for inclusion in the hypothesis table.
    """
    results = []

    # Find matching firmness configs
    ug_firmness = {(l, a): d for (dim, l, a), d in ug_results.items() if dim == "firmness"}
    dg_firmness = {(l, a): d for (dim, l, a), d in dg_results.items() if dim == "firmness"}

    common_keys = sorted(set(ug_firmness.keys()) & set(dg_firmness.keys()))

    if not common_keys:
        print("  No matching firmness configs found in both UG and DG.")
        return results

    for layer, alpha in common_keys:
        ug_data = ug_firmness[(layer, alpha)]
        dg_data = dg_firmness[(layer, alpha)]

        ug_s, ug_b = compute_demand_shift(ug_data.get("games", []))
        dg_s, dg_b = compute_demand_shift(dg_data.get("games", []))

        if len(ug_s) == 0 or len(dg_s) == 0:
            continue

        ug_shift = ug_s - ug_b  # per-game shifts
        dg_shift = dg_s - dg_b

        ug_mean = float(np.mean(ug_shift))
        dg_mean = float(np.mean(dg_shift))

        # Use unpaired two-sample TOST since games are different across conditions
        # But if n is matched, we can also report the difference of means
        n_ug, n_dg = len(ug_shift), len(dg_shift)

        # Two-sample TOST: test whether mean(UG_shift) - mean(DG_shift) is within epsilon
        diff_mean = ug_mean - dg_mean
        se_diff = np.sqrt(np.var(ug_shift, ddof=1) / n_ug + np.var(dg_shift, ddof=1) / n_dg)
        df_approx = min(n_ug, n_dg) - 1  # conservative df (Welch would be better but this suffices)

        if se_diff > 0:
            t_lower = (diff_mean + epsilon) / se_diff
            p_lower = 1 - stats.t.cdf(t_lower, df_approx)
            t_upper = (diff_mean - epsilon) / se_diff
            p_upper = stats.t.cdf(t_upper, df_approx)
            p_tost = max(p_lower, p_upper)
        else:
            t_lower, p_lower, t_upper, p_upper = 0.0, 0.0, 0.0, 0.0
            p_tost = 0.0

        equivalent = p_tost < 0.05

        results.append({
            "layer": layer, "alpha": alpha,
            "ug_mean_shift": ug_mean, "dg_mean_shift": dg_mean,
            "diff": diff_mean,
            "n_ug": n_ug, "n_dg": n_dg,
            "t_lower": t_lower, "p_lower": p_lower,
            "t_upper": t_upper, "p_upper": p_upper,
            "p_tost": p_tost, "equivalent": equivalent,
        })

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    print(f"\n{'=' * 78}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 78}")


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def print_hypothesis_table(rows: List[Dict], adjusted_p: List[float]) -> None:
    """Print the main hypothesis results table."""
    print_separator("CONFIRMATORY HYPOTHESES (BH-FDR corrected)")

    header = (f"  {'H':>3s}  {'Hypothesis':<45s}  {'Stat':>7s}  {'p_raw':>8s}  "
              f"{'p_adj':>8s}  {'d/eq':>7s}  {'Sig':>4s}  {'n':>4s}")
    print(header)
    print(f"  {'-'*3}  {'-'*45}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*4}")

    for i, row in enumerate(rows):
        p_adj = adjusted_p[i] if i < len(adjusted_p) else row["p_raw"]
        sig = sig_stars(p_adj)

        if row["test_type"] == "ttest":
            stat_str = f"t={row['t']:.2f}"
            d_str = f"d={row['d']:.2f}"
        elif row["test_type"] == "tost":
            stat_str = f"t={row.get('t_tost', 0):.2f}"
            d_str = "EQUIV" if row.get("equivalent") else "NOT-EQ"
        else:
            stat_str = "?"
            d_str = "?"

        print(f"  {row['label']:>3s}  {row['description']:<45s}  {stat_str:>7s}  "
              f"{row['p_raw']:>8.4f}  {p_adj:>8.4f}  {d_str:>7s}  {sig:>4s}  {row['n']:>4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Confirmatory analysis for Ultimatum Game steering experiments."
    )
    p.add_argument("--results_dir", default="results/ultimatum/temp7",
                   help="Directory with UG paired result JSON files.")
    p.add_argument("--dg_dir", default=None,
                   help="Directory with Dictator Game result JSON files (for H5).")
    p.add_argument("--epsilon", type=float, default=5.0,
                   help="Equivalence margin for TOST tests, in %% of pool (default: 5.0).")
    p.add_argument("--alpha_list", type=float, nargs="+", default=[3.0, 7.0, 10.0],
                   help="Alpha values for firmness (default: 3.0 7.0 10.0).")
    p.add_argument("--empathy_alpha_list", type=float, nargs="+", default=[-3.0, -7.0, -10.0],
                   help="Alpha values for empathy (default: -3.0 -7.0 -10.0, negative = demand DOWN).")
    p.add_argument("--layer_list", type=int, nargs="+", default=[10, 12],
                   help="Layers to include (default: 10 12).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Build confirmatory config list for UG (dimension-specific alpha signs)
    ug_configs = []
    for layer in args.layer_list:
        for alpha in args.empathy_alpha_list:
            ug_configs.append(("empathy", layer, alpha))
        for alpha in args.alpha_list:
            ug_configs.append(("firmness", layer, alpha))

    # Load UG results
    print(f"Loading UG results from {results_dir} ...")
    ug_results = load_confirmatory_results(
        results_dir,
        ug_configs,
    )
    print(f"  Loaded {len(ug_results)} / {len(ug_configs)} confirmatory configs.")
    for key in sorted(ug_results.keys()):
        n_games = len(ug_results[key].get("games", []))
        n_usable = len(_filter_usable_games(ug_results[key].get("games", [])))
        print(f"    {key[0]} L{key[1]} a={key[2]}: {n_usable}/{n_games} usable games")

    if not ug_results:
        print("ERROR: No confirmatory results found. Check --results_dir and file naming.")
        return

    # Load DG results (optional, for H5)
    dg_results = {}
    if args.dg_dir:
        dg_dir = Path(args.dg_dir)
        if dg_dir.exists():
            dg_configs = [("firmness", l, a) for l in args.layer_list for a in args.alpha_list]  # firmness only for DG
            print(f"\nLoading DG results from {dg_dir} ...")
            dg_results = load_confirmatory_results(dg_dir, dg_configs)
            print(f"  Loaded {len(dg_results)} / {len(dg_configs)} DG configs.")
        else:
            print(f"WARNING: DG directory not found: {dg_dir}. H5 will be skipped.")

    # -----------------------------------------------------------------------
    # Select best config per hypothesis (highest n, then lowest p from prior)
    # For pre-registration: use the FIRST matching config, not data-snooped best
    # -----------------------------------------------------------------------

    def _best_config(dim: str, results: Dict) -> Optional[Tuple[str, int, float]]:
        """Select config with largest n for a given dimension."""
        candidates = [(k, d) for k, d in results.items() if k[0] == dim]
        if not candidates:
            return None
        # Pick by largest usable n
        best_key = max(candidates, key=lambda x: len(_filter_usable_games(x[1].get("games", []))))[0]
        return best_key

    # -----------------------------------------------------------------------
    # Run H1-H5
    # -----------------------------------------------------------------------

    hypothesis_rows = []

    # H1: Empathy shifts proposer demand downward (steered < baseline)
    empathy_key = _best_config("empathy", ug_results)
    if empathy_key:
        data = ug_results[empathy_key]
        s, b = compute_demand_shift(data["games"])
        t, pv, d = one_sided_paired_ttest(s, b, alternative="less")
        hypothesis_rows.append({
            "label": "H1", "test_type": "ttest",
            "description": f"Empathy demand DOWN (L{empathy_key[1]} a={empathy_key[2]})",
            "t": t, "p_raw": pv, "d": d, "n": len(s),
            "config": empathy_key,
        })
    else:
        hypothesis_rows.append({
            "label": "H1", "test_type": "ttest",
            "description": "Empathy demand DOWN [NO DATA]",
            "t": 0, "p_raw": 1.0, "d": 0, "n": 0, "config": None,
        })

    # H2: Firmness shifts proposer demand upward (steered > baseline)
    firmness_key = _best_config("firmness", ug_results)
    if firmness_key:
        data = ug_results[firmness_key]
        s, b = compute_demand_shift(data["games"])
        t, pv, d = one_sided_paired_ttest(s, b, alternative="greater")
        hypothesis_rows.append({
            "label": "H2", "test_type": "ttest",
            "description": f"Firmness demand UP (L{firmness_key[1]} a={firmness_key[2]})",
            "t": t, "p_raw": pv, "d": d, "n": len(s),
            "config": firmness_key,
        })
    else:
        hypothesis_rows.append({
            "label": "H2", "test_type": "ttest",
            "description": "Firmness demand UP [NO DATA]",
            "t": 0, "p_raw": 1.0, "d": 0, "n": 0, "config": None,
        })

    # H3: Empathy improves proposer expected payoff (steered > baseline)
    if empathy_key:
        data = ug_results[empathy_key]
        s, b = compute_payoff_delta(data["games"])
        t, pv, d = one_sided_paired_ttest(s, b, alternative="greater")
        hypothesis_rows.append({
            "label": "H3", "test_type": "ttest",
            "description": f"Empathy payoff UP (L{empathy_key[1]} a={empathy_key[2]})",
            "t": t, "p_raw": pv, "d": d, "n": len(s),
            "config": empathy_key,
        })
    else:
        hypothesis_rows.append({
            "label": "H3", "test_type": "ttest",
            "description": "Empathy payoff UP [NO DATA]",
            "t": 0, "p_raw": 1.0, "d": 0, "n": 0, "config": None,
        })

    # H4: Firmness does NOT improve proposer payoff (TOST equivalence)
    if firmness_key:
        data = ug_results[firmness_key]
        s, b = compute_payoff_delta(data["games"])
        t_lo, p_lo, t_up, p_up, p_tost, equiv = tost_equivalence(s, b, args.epsilon)
        hypothesis_rows.append({
            "label": "H4", "test_type": "tost",
            "description": f"Firmness payoff EQUIV |d|<{args.epsilon}% (L{firmness_key[1]} a={firmness_key[2]})",
            "t_tost": t_lo if p_lo > p_up else t_up,
            "p_raw": p_tost, "equivalent": equiv, "n": len(s),
            "config": firmness_key,
            "_detail": {"t_lower": t_lo, "p_lower": p_lo, "t_upper": t_up, "p_upper": p_up},
        })
    else:
        hypothesis_rows.append({
            "label": "H4", "test_type": "tost",
            "description": f"Firmness payoff EQUIV |d|<{args.epsilon}% [NO DATA]",
            "t_tost": 0, "p_raw": 1.0, "equivalent": False, "n": 0, "config": None,
        })

    # H5: Firmness demand shift equal in UG and DG (TOST equivalence)
    if dg_results:
        h5_results = analyse_ug_vs_dg(ug_results, dg_results, args.epsilon)
        if h5_results:
            # Pick the config with largest combined n
            best_h5 = max(h5_results, key=lambda x: x["n_ug"] + x["n_dg"])
            hypothesis_rows.append({
                "label": "H5", "test_type": "tost",
                "description": (f"Firmness UG=DG shift |d|<{args.epsilon}% "
                                f"(L{best_h5['layer']} a={best_h5['alpha']})"),
                "t_tost": best_h5["t_lower"] if best_h5["p_lower"] > best_h5["p_upper"] else best_h5["t_upper"],
                "p_raw": best_h5["p_tost"], "equivalent": best_h5["equivalent"],
                "n": best_h5["n_ug"] + best_h5["n_dg"],
                "config": ("firmness", best_h5["layer"], best_h5["alpha"]),
                "_detail": best_h5,
            })
        else:
            hypothesis_rows.append({
                "label": "H5", "test_type": "tost",
                "description": f"Firmness UG=DG shift [NO MATCHING CONFIGS]",
                "t_tost": 0, "p_raw": 1.0, "equivalent": False, "n": 0, "config": None,
            })
    else:
        hypothesis_rows.append({
            "label": "H5", "test_type": "tost",
            "description": f"Firmness UG=DG shift [NO DG DATA]",
            "t_tost": 0, "p_raw": 1.0, "equivalent": False, "n": 0, "config": None,
        })

    # -----------------------------------------------------------------------
    # BH-FDR correction
    # -----------------------------------------------------------------------

    raw_p = [row["p_raw"] for row in hypothesis_rows]
    adj_p = bh_fdr_correction(raw_p)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------

    print_hypothesis_table(hypothesis_rows, adj_p)

    # Supplementary detail
    print_separator("SUPPLEMENTARY: PER-HYPOTHESIS DETAIL")

    for i, row in enumerate(hypothesis_rows):
        p_adj = adj_p[i]
        print(f"\n  --- {row['label']}: {row['description']} ---")
        cfg = row.get("config")
        if cfg:
            print(f"  Config: dimension={cfg[0]}, layer={cfg[1]}, alpha={cfg[2]}")

        if row["test_type"] == "ttest":
            print(f"  Test: one-sided paired t-test")
            direction = "less" if "DOWN" in row["description"] else "greater"
            print(f"  Alternative: steered {direction} than baseline")
            print(f"  t = {row['t']:.4f}, p_raw = {row['p_raw']:.6f}, p_adj = {p_adj:.6f}")
            print(f"  Cohen's d_z = {row['d']:.3f}")
            print(f"  n = {row['n']} paired games")
        elif row["test_type"] == "tost":
            detail = row.get("_detail", {})
            print(f"  Test: TOST equivalence (epsilon = {args.epsilon}%)")
            if detail:
                print(f"  Lower bound: t = {detail.get('t_lower', 0):.4f}, "
                      f"p = {detail.get('p_lower', 1):.6f}")
                print(f"  Upper bound: t = {detail.get('t_upper', 0):.4f}, "
                      f"p = {detail.get('p_upper', 1):.6f}")
            print(f"  p_TOST = {row['p_raw']:.6f}, p_adj = {p_adj:.6f}")
            print(f"  Equivalent: {'YES' if row.get('equivalent') else 'NO'}")
            print(f"  n = {row['n']}")

        verdict = "SUPPORTED" if p_adj < 0.05 else "NOT SUPPORTED"
        print(f"  Verdict (alpha=0.05, FDR-corrected): {verdict}")

    # -----------------------------------------------------------------------
    # Dose-response (all loaded configs)
    # -----------------------------------------------------------------------

    print_separator("DOSE-RESPONSE (all confirmatory configs)")
    analyse_dose_response(ug_results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    n_supported = sum(1 for i, row in enumerate(hypothesis_rows) if adj_p[i] < 0.05)
    n_total = len(hypothesis_rows)

    print_separator("SUMMARY")
    print(f"  {n_supported} / {n_total} hypotheses supported at alpha=0.05 (BH-FDR corrected).")
    print(f"  Equivalence margin (epsilon): {args.epsilon}% of pool.")
    print(f"  UG results: {results_dir}")
    if args.dg_dir:
        print(f"  DG results: {args.dg_dir}")
    else:
        print(f"  DG results: not provided (H5 skipped).")

    # Quick sanity: report raw vs adjusted for transparency
    print(f"\n  Raw vs adjusted p-values:")
    for i, row in enumerate(hypothesis_rows):
        flag = " <-- FLIPPED" if (raw_p[i] < 0.05) != (adj_p[i] < 0.05) else ""
        print(f"    {row['label']}: raw={raw_p[i]:.4f}  adj={adj_p[i]:.4f}{flag}")


if __name__ == "__main__":
    main()
