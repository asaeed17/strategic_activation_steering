#!/usr/bin/env python3
"""
analysis/statistical_hardening.py -- Comprehensive statistical hardening for UG/DG results.

Reads all existing experiment results and produces a complete statistical report
with bootstrap CIs, Cohen's d CIs, TOST equivalence tests, and BH-FDR correction.

CPU-only. Deps: numpy, scipy.

Usage:
  python analysis/statistical_hardening.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UG_DIR = PROJECT_ROOT / "results" / "ultimatum" / "confirmatory" / "ug"
UG_POS_EMPATHY_DIR = PROJECT_ROOT / "results" / "ultimatum" / "confirmatory_v2" / "ug_pos_empathy"
DG_FIRMNESS_DIR = PROJECT_ROOT / "results" / "ultimatum" / "confirmatory_v2" / "dg"
DG_EMPATHY_DIR = PROJECT_ROOT / "results" / "ultimatum" / "confirmatory_v2" / "dg_empathy"
ACCEPTANCE_CURVE_FILE = PROJECT_ROOT / "results" / "ultimatum" / "acceptance_curve" / "acceptance_curve.json"
PHASE_C_FILE = PROJECT_ROOT / "results" / "ultimatum" / "phase_c_analytical_payoff.json"

OUTPUT_FILE = PROJECT_ROOT / "results" / "ultimatum" / "statistical_hardening.json"

N_BOOTSTRAP = 10_000
RNG_SEED = 42
TOST_EPSILON = 5.0  # percentage points


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result_file(fpath: Path) -> Optional[Dict]:
    """Load a single result JSON file, returning None on failure."""
    try:
        with open(fpath) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return None


def load_directory(dirpath: Path) -> List[Dict]:
    """Load all valid paired result JSON files from a directory."""
    results = []
    if not dirpath.exists():
        print(f"  WARNING: directory not found: {dirpath}")
        return results
    for fpath in sorted(dirpath.glob("*.json")):
        data = load_result_file(fpath)
        if data and "games" in data and "config" in data:
            data["_filepath"] = str(fpath)
            results.append(data)
    return results


def config_label(data: Dict) -> str:
    """Create a human-readable label from config dict."""
    cfg = data.get("config", {})
    dim = cfg.get("dimension", "?")
    layers = cfg.get("layers", [])
    alpha = cfg.get("alpha", 0)
    game = cfg.get("game", "ultimatum")
    layer_str = f"L{layers[0]}" if layers else "L?"
    prefix = "DG " if game == "dictator" else ""
    return f"{prefix}{dim} {layer_str} a={alpha}"


def config_sort_key(data: Dict) -> Tuple:
    """Sort key: game type, dimension, layer, alpha."""
    cfg = data.get("config", {})
    game = 0 if cfg.get("game", "ultimatum") == "ultimatum" else 1
    dim = cfg.get("dimension", "")
    layers = cfg.get("layers", [0])
    alpha = cfg.get("alpha", 0)
    return (game, dim, layers[0] if layers else 0, alpha)


# ---------------------------------------------------------------------------
# Per-game metric extraction
# ---------------------------------------------------------------------------

def filter_usable_games(games: List[Dict]) -> List[Dict]:
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
        if s.get("proposer_share") is None or b.get("proposer_share") is None:
            continue
        usable.append(g)
    return usable


def extract_demand_pct(games: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired demand percentages (proposer_share / pool * 100)."""
    usable = filter_usable_games(games)
    steered = np.array([g["steered"]["proposer_share"] / g["pool"] * 100 for g in usable])
    baseline = np.array([g["baseline"]["proposer_share"] / g["pool"] * 100 for g in usable])
    return steered, baseline


def extract_payoff_pct(games: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract paired payoff percentages (proposer_payoff / pool * 100)."""
    usable = filter_usable_games(games)
    steered = np.array([g["steered"].get("proposer_payoff", 0) / g["pool"] * 100 for g in usable])
    baseline = np.array([g["baseline"].get("proposer_payoff", 0) / g["pool"] * 100 for g in usable])
    return steered, baseline


def extract_acceptance(games: List[Dict]) -> Tuple[int, int, int, int]:
    """Extract acceptance counts: (steered_accept, steered_total, baseline_accept, baseline_total)."""
    usable = filter_usable_games(games)
    s_accept = sum(1 for g in usable if g["steered"].get("agreed", False))
    b_accept = sum(1 for g in usable if g["baseline"].get("agreed", False))
    n = len(usable)
    return s_accept, n, b_accept, n


def count_unique_offer_pairs(games: List[Dict]) -> Tuple[int, int]:
    """Count games where steered_offer != baseline_offer. Returns (n_different, n_total)."""
    usable = filter_usable_games(games)
    n_diff = 0
    for g in usable:
        s_share = g["steered"]["proposer_share"]
        b_share = g["baseline"]["proposer_share"]
        if s_share != b_share:
            n_diff += 1
    return n_diff, len(usable)


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                 rng: np.random.Generator = None) -> Tuple[float, float]:
    """Bootstrap percentile CI for the mean of paired differences."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diff[idx])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d_z for paired samples."""
    if len(diff) < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def cohens_d_ci_bootstrap(diff: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                          rng: np.random.Generator = None) -> Tuple[float, float]:
    """Bootstrap CI for Cohen's d_z."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    boot_d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = diff[idx]
        sd = np.std(sample, ddof=1)
        boot_d[i] = np.mean(sample) / sd if sd > 0 else 0.0
    lo = np.percentile(boot_d, 100 * alpha / 2)
    hi = np.percentile(boot_d, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def paired_ttest_twosided(steered: np.ndarray, baseline: np.ndarray) -> Tuple[float, float]:
    """Two-sided paired t-test. Returns (t_stat, p_value)."""
    if len(steered) < 2:
        return np.nan, 1.0
    t_stat, p_val = stats.ttest_rel(steered, baseline)
    return float(t_stat), float(p_val)


def tost_equivalence(
    steered: np.ndarray,
    baseline: np.ndarray,
    epsilon: float,
) -> Tuple[float, float, float, float, float, bool]:
    """Two One-Sided Tests (TOST) for equivalence.

    Returns (t_lower, p_lower, t_upper, p_upper, p_tost, equivalent).
    """
    diff = steered - baseline
    n = len(diff)
    if n < 2:
        return np.nan, 1.0, np.nan, 1.0, 1.0, False

    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    df = n - 1

    if se == 0:
        equiv = abs(mean_diff) < epsilon
        return 0.0, 0.0, 0.0, 0.0, 0.0, equiv

    # Lower test: H0: mean_diff <= -epsilon
    t_lower = (mean_diff + epsilon) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    # Upper test: H0: mean_diff >= epsilon
    t_upper = (mean_diff - epsilon) / se
    p_upper = stats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    return float(t_lower), float(p_lower), float(t_upper), float(p_upper), float(p_tost), equivalent


def bh_fdr_correction(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev_adj = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p_val = indexed[rank_idx]
        rank = rank_idx + 1
        adj = min(p_val * m / rank, 1.0)
        adj = min(adj, prev_adj)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


# ---------------------------------------------------------------------------
# Per-config comprehensive analysis
# ---------------------------------------------------------------------------

def analyse_config(data: Dict, rng: np.random.Generator) -> Dict:
    """Compute all statistics for one config. Returns a results dict."""
    games = data.get("games", [])
    label = config_label(data)
    cfg = data.get("config", {})

    # Demand
    s_demand, b_demand = extract_demand_pct(games)
    demand_diff = s_demand - b_demand
    n_demand = len(demand_diff)

    demand_mean = float(np.mean(demand_diff)) if n_demand > 0 else np.nan
    demand_ci = bootstrap_ci(demand_diff, rng=rng) if n_demand >= 2 else (np.nan, np.nan)
    demand_d = cohens_d_paired(demand_diff)
    demand_d_ci = cohens_d_ci_bootstrap(demand_diff, rng=rng) if n_demand >= 2 else (np.nan, np.nan)
    demand_t, demand_p = paired_ttest_twosided(s_demand, b_demand)

    # Payoff
    s_payoff, b_payoff = extract_payoff_pct(games)
    payoff_diff = s_payoff - b_payoff
    n_payoff = len(payoff_diff)

    payoff_mean = float(np.mean(payoff_diff)) if n_payoff > 0 else np.nan
    payoff_ci = bootstrap_ci(payoff_diff, rng=rng) if n_payoff >= 2 else (np.nan, np.nan)
    payoff_d = cohens_d_paired(payoff_diff)
    payoff_d_ci = cohens_d_ci_bootstrap(payoff_diff, rng=rng) if n_payoff >= 2 else (np.nan, np.nan)
    payoff_t, payoff_p = paired_ttest_twosided(s_payoff, b_payoff)

    # Acceptance
    s_acc, s_tot, b_acc, b_tot = extract_acceptance(games)

    # Unique offer pairs
    n_different, n_total = count_unique_offer_pairs(games)

    return {
        "label": label,
        "config": {
            "dimension": cfg.get("dimension"),
            "game": cfg.get("game", "ultimatum"),
            "layers": cfg.get("layers"),
            "alpha": cfg.get("alpha"),
        },
        "n_usable": n_demand,
        "demand": {
            "mean_pp": demand_mean,
            "ci_95": list(demand_ci),
            "cohens_d": demand_d,
            "d_ci_95": list(demand_d_ci),
            "t_stat": demand_t,
            "p_value": demand_p,
        },
        "payoff": {
            "mean_pp": payoff_mean,
            "ci_95": list(payoff_ci),
            "cohens_d": payoff_d,
            "d_ci_95": list(payoff_d_ci),
            "t_stat": payoff_t,
            "p_value": payoff_p,
        },
        "acceptance": {
            "steered_accept": s_acc,
            "steered_total": s_tot,
            "steered_rate": s_acc / s_tot if s_tot > 0 else np.nan,
            "baseline_accept": b_acc,
            "baseline_total": b_tot,
            "baseline_rate": b_acc / b_tot if b_tot > 0 else np.nan,
        },
        "unique_offer_pairs": {
            "n_different": n_different,
            "n_total": n_total,
            "pct_different": n_different / n_total * 100 if n_total > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# TOST for empathy DG nullification
# ---------------------------------------------------------------------------

def analyse_empathy_dg_nullification(dg_empathy_results: List[Dict]) -> List[Dict]:
    """Run TOST on empathy DG configs to bound 'no effect' claim."""
    tost_results = []
    for data in dg_empathy_results:
        label = config_label(data)
        games = data.get("games", [])
        s_demand, b_demand = extract_demand_pct(games)
        if len(s_demand) < 2:
            continue
        diff = s_demand - b_demand
        mean_diff = float(np.mean(diff))
        t_lo, p_lo, t_up, p_up, p_tost, equiv = tost_equivalence(s_demand, b_demand, TOST_EPSILON)
        tost_results.append({
            "label": label,
            "mean_diff_pp": mean_diff,
            "n": len(diff),
            "t_lower": t_lo,
            "p_lower": p_lo,
            "t_upper": t_up,
            "p_upper": p_up,
            "p_tost": p_tost,
            "equivalent": equiv,
            "epsilon": TOST_EPSILON,
        })
    return tost_results


# ---------------------------------------------------------------------------
# TOST for framing effect (Phase C)
# ---------------------------------------------------------------------------

def analyse_framing_effect_tost(phase_c_data: Dict) -> Dict:
    """Run TOST on framing_effect_loo_pp values to bound 'framing ~ 0' claim."""
    configs_loo = phase_c_data.get("configs_loo", [])
    if not configs_loo:
        return {"error": "No configs_loo found in Phase C data"}

    framing_values = np.array([c["framing_effect_loo_pp"] for c in configs_loo])
    n = len(framing_values)
    mean_fe = float(np.mean(framing_values))
    sd_fe = float(np.std(framing_values, ddof=1))

    # One-sample TOST: is |mean framing effect| < epsilon?
    se = sd_fe / np.sqrt(n) if n > 1 else np.inf
    df = n - 1

    if se == 0 or n < 2:
        equiv = abs(mean_fe) < TOST_EPSILON
        return {
            "mean_framing_pp": mean_fe, "sd": sd_fe, "n": n,
            "p_tost": 0.0 if equiv else 1.0, "equivalent": equiv,
            "epsilon": TOST_EPSILON,
        }

    # Lower: H0: mu <= -epsilon
    t_lower = (mean_fe + TOST_EPSILON) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    # Upper: H0: mu >= epsilon
    t_upper = (mean_fe - TOST_EPSILON) / se
    p_upper = stats.t.cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    return {
        "mean_framing_pp": mean_fe,
        "sd": sd_fe,
        "n": n,
        "t_lower": float(t_lower),
        "p_lower": float(p_lower),
        "t_upper": float(t_upper),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "equivalent": equivalent,
        "epsilon": TOST_EPSILON,
        "per_config": [
            {"config": c["config"], "framing_pp": c["framing_effect_loo_pp"]}
            for c in configs_loo
        ],
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def fmt(val, width=7, decimals=2) -> str:
    """Format a float, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return " " * (width - 2) + "NA"
    return f"{val:>{width}.{decimals}f}"


def sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


def print_sep(title: str = "") -> None:
    print(f"\n{'=' * 100}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 100}")


def print_config_table(config_results: List[Dict], all_p_adjusted: Dict[str, float]) -> None:
    """Print the comprehensive per-config table."""
    print_sep("SECTION 2: PER-CONFIG COMPREHENSIVE TABLE (all 24+ UG configs + DG configs)")

    # Header
    print(f"\n  {'Config':<32s}  {'n':>3s}  {'Diff%':>5s}  |  "
          f"{'Demand':>7s}  {'95% CI':>14s}  {'d':>5s}  {'d 95% CI':>12s}  {'p':>8s}  {'p_adj':>8s}  {'Sig':>4s}  |  "
          f"{'Payoff':>7s}  {'95% CI':>14s}  {'d':>5s}  {'p':>8s}  {'p_adj':>8s}  {'Sig':>4s}  |  "
          f"{'Acc_s':>5s}  {'Acc_b':>5s}")
    print(f"  {'-' * 32}  {'-' * 3}  {'-' * 5}  |  "
          f"{'-' * 7}  {'-' * 14}  {'-' * 5}  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 4}  |  "
          f"{'-' * 7}  {'-' * 14}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 4}  |  "
          f"{'-' * 5}  {'-' * 5}")

    for r in config_results:
        label = r["label"]
        n = r["n_usable"]
        uniq = r["unique_offer_pairs"]
        pct_diff = uniq["pct_different"]

        dm = r["demand"]
        pm = r["payoff"]
        ac = r["acceptance"]

        # Get adjusted p-values from the lookup
        demand_key = f"demand|{label}"
        payoff_key = f"payoff|{label}"
        dp_adj = all_p_adjusted.get(demand_key, dm["p_value"])
        pp_adj = all_p_adjusted.get(payoff_key, pm["p_value"])

        ci_str = f"[{fmt(dm['ci_95'][0], 5, 1)},{fmt(dm['ci_95'][1], 5, 1)}]"
        d_ci_str = f"[{fmt(dm['d_ci_95'][0], 4, 2)},{fmt(dm['d_ci_95'][1], 4, 2)}]"
        p_ci_str = f"[{fmt(pm['ci_95'][0], 5, 1)},{fmt(pm['ci_95'][1], 5, 1)}]"

        s_rate = f"{ac['steered_rate'] * 100:.0f}%" if not np.isnan(ac.get('steered_rate', np.nan)) else "NA"
        b_rate = f"{ac['baseline_rate'] * 100:.0f}%" if not np.isnan(ac.get('baseline_rate', np.nan)) else "NA"

        print(f"  {label:<32s}  {n:>3d}  {pct_diff:>4.0f}%  |  "
              f"{fmt(dm['mean_pp'], 7, 2)}  {ci_str:>14s}  {fmt(dm['cohens_d'], 5, 2)}  {d_ci_str:>12s}  "
              f"{fmt(dm['p_value'], 8, 5)}  {fmt(dp_adj, 8, 5)}  {sig_stars(dp_adj):>4s}  |  "
              f"{fmt(pm['mean_pp'], 7, 2)}  {p_ci_str:>14s}  {fmt(pm['cohens_d'], 5, 2)}  "
              f"{fmt(pm['p_value'], 8, 5)}  {fmt(pp_adj, 8, 5)}  {sig_stars(pp_adj):>4s}  |  "
              f"{s_rate:>5s}  {b_rate:>5s}")


def print_tost_section(empathy_dg_tost: List[Dict], framing_tost: Dict,
                       all_p_adjusted: Dict[str, float]) -> None:
    """Print TOST equivalence test results."""
    print_sep("SECTION 3: TOST EQUIVALENCE TESTS FOR NULL CLAIMS")

    # Empathy DG nullification
    print(f"\n  3a. Empathy DG nullification (epsilon = {TOST_EPSILON} pp)")
    print(f"  Claim: empathy steering has no meaningful effect on demand in Dictator Game.\n")
    if empathy_dg_tost:
        print(f"  {'Config':<35s}  {'Mean diff':>9s}  {'n':>3s}  "
              f"{'t_lo':>7s}  {'p_lo':>8s}  {'t_up':>7s}  {'p_up':>8s}  "
              f"{'p_TOST':>8s}  {'p_adj':>8s}  {'Result':>8s}")
        print(f"  {'-' * 35}  {'-' * 9}  {'-' * 3}  "
              f"{'-' * 7}  {'-' * 8}  {'-' * 7}  {'-' * 8}  "
              f"{'-' * 8}  {'-' * 8}  {'-' * 8}")
        for t in empathy_dg_tost:
            key = f"tost_dg_empathy|{t['label']}"
            p_adj = all_p_adjusted.get(key, t["p_tost"])
            equiv_adj = p_adj < 0.05
            result = "EQUIV" if equiv_adj else "NOT-EQ"
            print(f"  {t['label']:<35s}  {t['mean_diff_pp']:>+9.2f}  {t['n']:>3d}  "
                  f"{t['t_lower']:>7.2f}  {t['p_lower']:>8.5f}  {t['t_upper']:>7.2f}  {t['p_upper']:>8.5f}  "
                  f"{t['p_tost']:>8.5f}  {p_adj:>8.5f}  {result:>8s}")
    else:
        print("  No empathy DG data available.")

    # Framing effect
    print(f"\n  3b. Framing effect TOST (epsilon = {TOST_EPSILON} pp)")
    print(f"  Claim: framing effect is negligible (steering works through numbers, not framing).\n")
    if "error" not in framing_tost:
        ft = framing_tost
        key = "tost_framing"
        p_adj = all_p_adjusted.get(key, ft["p_tost"])
        equiv_adj = p_adj < 0.05
        print(f"  Mean framing effect: {ft['mean_framing_pp']:+.2f} pp (SD = {ft['sd']:.2f}, n = {ft['n']})")
        print(f"  t_lower = {ft['t_lower']:.4f}, p_lower = {ft['p_lower']:.6f}")
        print(f"  t_upper = {ft['t_upper']:.4f}, p_upper = {ft['p_upper']:.6f}")
        print(f"  p_TOST = {ft['p_tost']:.6f}, p_adj = {p_adj:.6f}")
        print(f"  Result: {'EQUIVALENT (|framing| < ' + str(TOST_EPSILON) + ' pp)' if equiv_adj else 'NOT EQUIVALENT'}")

        print(f"\n  Per-config framing effects:")
        for pc in ft.get("per_config", []):
            bar = "+" * max(0, int(abs(pc["framing_pp"]) * 5)) if pc["framing_pp"] > 0 else \
                  "-" * max(0, int(abs(pc["framing_pp"]) * 5))
            print(f"    {pc['config']:<50s}  {pc['framing_pp']:>+6.2f} pp  {bar}")
    else:
        print(f"  {framing_tost['error']}")


def print_fdr_summary(all_tests: List[Dict], all_p_raw: List[float], all_p_adj: List[float]) -> None:
    """Print BH-FDR correction summary."""
    print_sep("SECTION 4: BH-FDR CORRECTION SUMMARY")

    n_total = len(all_tests)
    n_sig_raw = sum(1 for p in all_p_raw if p < 0.05)
    n_sig_adj = sum(1 for p in all_p_adj if p < 0.05)

    print(f"\n  Total tests: {n_total}")
    print(f"  Significant at alpha=0.05 (raw):      {n_sig_raw}")
    print(f"  Significant at alpha=0.05 (BH-FDR):   {n_sig_adj}")
    print(f"  Tests lost after correction:           {n_sig_raw - n_sig_adj}")

    # Show any flipped results
    flipped = [(t, raw, adj) for t, raw, adj in zip(all_tests, all_p_raw, all_p_adj)
               if (raw < 0.05) != (adj < 0.05)]
    if flipped:
        print(f"\n  Flipped by FDR correction:")
        for t, raw, adj in flipped:
            direction = "sig->non-sig" if raw < 0.05 else "non-sig->sig"
            print(f"    {t['label']:<50s}  raw={raw:.5f}  adj={adj:.5f}  [{direction}]")
    else:
        print(f"\n  No tests flipped by FDR correction.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    print_sep("STATISTICAL HARDENING REPORT")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")
    print(f"  TOST epsilon: {TOST_EPSILON} pp")
    print(f"  RNG seed: {RNG_SEED}")

    # -----------------------------------------------------------------------
    # Section 1: Load all data
    # -----------------------------------------------------------------------
    print_sep("SECTION 1: DATA LOADING")

    ug_data = load_directory(UG_DIR)
    print(f"  UG confirmatory (neg empathy):  {len(ug_data)} files from {UG_DIR}")

    ug_pos_data = load_directory(UG_POS_EMPATHY_DIR)
    print(f"  UG positive empathy:            {len(ug_pos_data)} files from {UG_POS_EMPATHY_DIR}")

    dg_firmness_data = load_directory(DG_FIRMNESS_DIR)
    print(f"  DG firmness:                    {len(dg_firmness_data)} files from {DG_FIRMNESS_DIR}")

    dg_empathy_data = load_directory(DG_EMPATHY_DIR)
    print(f"  DG empathy:                     {len(dg_empathy_data)} files from {DG_EMPATHY_DIR}")

    # Acceptance curve
    acceptance_data = None
    if ACCEPTANCE_CURVE_FILE.exists():
        acceptance_data = load_result_file(ACCEPTANCE_CURVE_FILE)
        print(f"  Acceptance curve:               loaded ({len(acceptance_data.get('results', []))} observations)")
    else:
        print(f"  Acceptance curve:               NOT FOUND")

    # Phase C
    phase_c_data = None
    if PHASE_C_FILE.exists():
        phase_c_data = load_result_file(PHASE_C_FILE)
        n_loo = len(phase_c_data.get("configs_loo", []))
        print(f"  Phase C analytical payoff:      loaded ({n_loo} configs)")
    else:
        print(f"  Phase C analytical payoff:      NOT FOUND")

    # Combine all UG configs
    all_ug = ug_data + ug_pos_data
    all_dg = dg_firmness_data + dg_empathy_data
    all_configs = all_ug + all_dg

    # Sort for consistent output
    all_configs.sort(key=config_sort_key)
    all_ug.sort(key=config_sort_key)

    total_games = sum(len(d.get("games", [])) for d in all_configs)
    total_usable = sum(len(filter_usable_games(d.get("games", []))) for d in all_configs)
    print(f"\n  Total configs: {len(all_configs)} ({len(all_ug)} UG + {len(all_dg)} DG)")
    print(f"  Total games:   {total_games} ({total_usable} usable)")

    # -----------------------------------------------------------------------
    # Section 2: Per-config analysis
    # -----------------------------------------------------------------------
    print(f"\n  Computing per-config statistics (bootstrap n={N_BOOTSTRAP})...", flush=True)

    config_results = []
    for data in all_configs:
        result = analyse_config(data, rng)
        config_results.append(result)

    # -----------------------------------------------------------------------
    # Section 3: TOST equivalence tests
    # -----------------------------------------------------------------------

    # 3a: Empathy DG nullification
    empathy_dg_tost = analyse_empathy_dg_nullification(dg_empathy_data)

    # 3b: Framing effect
    framing_tost = analyse_framing_effect_tost(phase_c_data) if phase_c_data else {"error": "No Phase C data"}

    # -----------------------------------------------------------------------
    # Section 4: BH-FDR correction over ALL p-values
    # -----------------------------------------------------------------------

    # Collect all p-values with labels
    all_tests = []
    all_p_raw = []

    # Demand tests
    for r in config_results:
        label = f"demand|{r['label']}"
        p = r["demand"]["p_value"]
        all_tests.append({"label": label, "type": "demand_ttest"})
        all_p_raw.append(p if not np.isnan(p) else 1.0)

    # Payoff tests
    for r in config_results:
        label = f"payoff|{r['label']}"
        p = r["payoff"]["p_value"]
        all_tests.append({"label": label, "type": "payoff_ttest"})
        all_p_raw.append(p if not np.isnan(p) else 1.0)

    # TOST: empathy DG
    for t in empathy_dg_tost:
        label = f"tost_dg_empathy|{t['label']}"
        all_tests.append({"label": label, "type": "tost_empathy_dg"})
        all_p_raw.append(t["p_tost"])

    # TOST: framing
    if "error" not in framing_tost:
        all_tests.append({"label": "tost_framing", "type": "tost_framing"})
        all_p_raw.append(framing_tost["p_tost"])

    # Apply BH-FDR
    all_p_adj = bh_fdr_correction(all_p_raw)

    # Build lookup for adjusted p-values
    p_adj_lookup = {}
    for i, t in enumerate(all_tests):
        p_adj_lookup[t["label"]] = all_p_adj[i]

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------

    print_config_table(config_results, p_adj_lookup)
    print_tost_section(empathy_dg_tost, framing_tost, p_adj_lookup)
    print_fdr_summary(all_tests, all_p_raw, all_p_adj)

    # -----------------------------------------------------------------------
    # Section 5: Key summary statistics
    # -----------------------------------------------------------------------
    print_sep("SECTION 5: KEY SUMMARY STATISTICS")

    # Separate UG and DG results
    ug_results = [r for r in config_results if r["config"]["game"] == "ultimatum"]
    dg_results = [r for r in config_results if r["config"]["game"] == "dictator"]

    # UG demand significance
    ug_demand_sig = sum(1 for r in ug_results
                        if p_adj_lookup.get(f"demand|{r['label']}", 1.0) < 0.05)
    print(f"\n  UG configs with significant demand shift (BH-FDR p<0.05): {ug_demand_sig}/{len(ug_results)}")

    # UG payoff significance
    ug_payoff_sig = sum(1 for r in ug_results
                        if p_adj_lookup.get(f"payoff|{r['label']}", 1.0) < 0.05)
    print(f"  UG configs with significant payoff shift (BH-FDR p<0.05): {ug_payoff_sig}/{len(ug_results)}")

    # Effect size ranges for UG
    ug_demand_d = [r["demand"]["cohens_d"] for r in ug_results if not np.isnan(r["demand"]["cohens_d"])]
    ug_payoff_d = [r["payoff"]["cohens_d"] for r in ug_results if not np.isnan(r["payoff"]["cohens_d"])]
    if ug_demand_d:
        print(f"  UG demand Cohen's d range: [{min(ug_demand_d):.2f}, {max(ug_demand_d):.2f}]")
    if ug_payoff_d:
        print(f"  UG payoff Cohen's d range: [{min(ug_payoff_d):.2f}, {max(ug_payoff_d):.2f}]")

    # Unique offer pair stats
    ug_unique = [r["unique_offer_pairs"]["pct_different"] for r in ug_results]
    if ug_unique:
        print(f"  UG unique offer pairs: mean={np.mean(ug_unique):.1f}%, "
              f"min={min(ug_unique):.0f}%, max={max(ug_unique):.0f}%")

    # DG summary
    if dg_results:
        dg_demand_d = [r["demand"]["cohens_d"] for r in dg_results if not np.isnan(r["demand"]["cohens_d"])]
        print(f"\n  DG configs: {len(dg_results)}")
        if dg_demand_d:
            print(f"  DG demand Cohen's d range: [{min(dg_demand_d):.2f}, {max(dg_demand_d):.2f}]")

    # Dose-response check: for each dimension x layer x sign, is |d| monotonically increasing with |alpha|?
    print(f"\n  Dose-response monotonicity check (by dimension x layer x alpha-sign):")
    from collections import defaultdict
    dr_groups = defaultdict(list)
    for r in ug_results:
        cfg = r["config"]
        alpha = cfg["alpha"]
        # Separate positive and negative alpha for empathy
        sign_label = "neg" if alpha < 0 else "pos"
        key = (cfg["dimension"], cfg["layers"][0] if cfg["layers"] else 0, sign_label)
        dr_groups[key].append((abs(alpha), r["demand"]["cohens_d"]))

    for (dim, layer, sign), points in sorted(dr_groups.items()):
        points.sort(key=lambda x: x[0])
        alphas = [p[0] for p in points]
        d_vals = [abs(p[1]) for p in points]
        if len(alphas) >= 3:
            rho, sp_p = stats.spearmanr(alphas, d_vals)
            monotonic = rho > 0.5 and sp_p < 0.10
            tag = "MONOTONIC" if monotonic else "NOT MONOTONIC"
            alpha_d_str = ", ".join([f"|a|={a:.0f}->|d|={d:.2f}" for a, d in zip(alphas, d_vals)])
            sign_str = f"({sign} alpha)" if dim == "empathy" else ""
            print(f"    {dim} L{layer} {sign_str}: rho={rho:+.3f} (p={sp_p:.3f}) [{tag}]  {alpha_d_str}")

    # -----------------------------------------------------------------------
    # Section 6: Acceptance curve summary (if available)
    # -----------------------------------------------------------------------
    if acceptance_data:
        print_sep("SECTION 6: ACCEPTANCE CURVE SUMMARY")
        results_list = acceptance_data.get("results", [])
        if results_list:
            from collections import Counter
            by_level = defaultdict(list)
            for r in results_list:
                offer_pct = r.get("offer_pct", 0)
                resp = r.get("response", "").upper()
                accepted = resp in ("ACCEPT", "ACCEPTED")
                by_level[offer_pct].append(accepted)

            print(f"\n  {'Offer %':>8s}  {'Accept':>6s}  {'Total':>5s}  {'Rate':>6s}  {'95% Wilson CI':>16s}")
            print(f"  {'-' * 8}  {'-' * 6}  {'-' * 5}  {'-' * 6}  {'-' * 16}")

            for level in sorted(by_level.keys()):
                accepts = by_level[level]
                n_acc = sum(accepts)
                n_tot = len(accepts)
                rate = n_acc / n_tot if n_tot > 0 else 0
                # Wilson score interval
                if n_tot > 0:
                    z = 1.96
                    denom = 1 + z ** 2 / n_tot
                    centre = (rate + z ** 2 / (2 * n_tot)) / denom
                    spread = z * np.sqrt((rate * (1 - rate) + z ** 2 / (4 * n_tot)) / n_tot) / denom
                    ci_lo = max(0, centre - spread)
                    ci_hi = min(1, centre + spread)
                else:
                    ci_lo, ci_hi = 0, 0
                print(f"  {level * 100:>7.0f}%  {n_acc:>6d}  {n_tot:>5d}  {rate:>5.1%}  [{ci_lo:.3f}, {ci_hi:.3f}]")

            # Non-monotonicity test: is 80% acceptance rate < 60% acceptance rate?
            rates = {}
            for level in sorted(by_level.keys()):
                accepts = by_level[level]
                rates[level] = sum(accepts) / len(accepts) if accepts else 0

            if 0.6 in rates and 0.8 in rates:
                print(f"\n  Non-monotonicity check: rate(60%) = {rates[0.6]:.3f}, rate(80%) = {rates[0.8]:.3f}")
                if rates[0.8] < rates[0.6]:
                    print(f"  CONFIRMED: acceptance drops for overly generous offers ({rates[0.8]:.3f} < {rates[0.6]:.3f})")
                else:
                    print(f"  NOT confirmed: no drop for generous offers")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    output = {
        "description": "Statistical hardening report for UG/DG steering experiments",
        "params": {
            "n_bootstrap": N_BOOTSTRAP,
            "tost_epsilon_pp": TOST_EPSILON,
            "rng_seed": RNG_SEED,
        },
        "config_results": config_results,
        "empathy_dg_tost": empathy_dg_tost,
        "framing_tost": framing_tost if "error" not in (framing_tost or {}) else None,
        "fdr_correction": {
            "n_tests": len(all_tests),
            "tests": [
                {"label": t["label"], "type": t["type"], "p_raw": all_p_raw[i], "p_adj": all_p_adj[i]}
                for i, t in enumerate(all_tests)
            ],
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
