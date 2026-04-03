#!/usr/bin/env python3
"""
Comprehensive analysis of the final 7B grid experiment.
360 configs: 10 dims x 9 layers x 4 alphas x n=50.

Produces:
  1. Completeness check (expect exactly 360 files)
  2. Baseline consistency check (all baselines ~50-65%)
  3. Heat maps: 10 dims x 9 layers for alpha=+7 and alpha=-7
  4. Top 20 configs by |d|
  5. Per-dimension summary (peak layer, peak alpha, max d, direction)
  6. Dose-response for top 5 dims at their peak layer
  7. Comparison with landscape screen results (if available)
  8. Anomaly detection

Usage:
  python analysis/analyse_final_grid.py
  python analysis/analyse_final_grid.py --results_dir results/ultimatum/final_7b_grid
  python analysis/analyse_final_grid.py --landscape_dir results/ultimatum/landscape_screen_v2
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

DIMENSIONS = [
    "firmness", "empathy", "anchoring", "greed", "narcissism",
    "fairness_norm", "composure", "flattery", "spite", "undecidedness",
]
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
ALPHAS = None  # Auto-detected from results


def load_all_results(results_dir: Path) -> tuple:
    """Load all JSON results into a nested dict: results[dim][layer][alpha] = data.
    Also returns the set of alphas found."""
    results = defaultdict(lambda: defaultdict(dict))
    alphas_found = set()
    files = sorted(results_dir.glob("*.json"))

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            cfg = data["config"]
            dim = cfg["dimension"]
            layer = cfg["layers"][0]
            alpha = cfg["alpha"]
            results[dim][layer][alpha] = data
            alphas_found.add(alpha)
        except Exception as e:
            print(f"  WARNING: Failed to load {fp.name}: {e}")

    return results, sorted(alphas_found)


def check_completeness(results: dict) -> tuple:
    """Check that all 360 configs are present."""
    present = 0
    missing = []
    for dim in DIMENSIONS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                if dim in results and layer in results[dim] and alpha in results[dim][layer]:
                    present += 1
                else:
                    missing.append((dim, layer, alpha))
    return present, missing


def check_baselines(results: dict) -> list:
    """Check baseline consistency across all configs."""
    baselines = []
    anomalies = []
    for dim in DIMENSIONS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                data = results.get(dim, {}).get(layer, {}).get(alpha)
                if data is None:
                    continue
                s = data.get("summary", {})
                b_mean = s.get("baseline_mean_proposer_pct")
                if b_mean is not None:
                    baselines.append(b_mean)
                    if b_mean < 40 or b_mean > 75:
                        anomalies.append((dim, layer, alpha, b_mean))
    return baselines, anomalies


def build_heatmap_data(results: dict, target_alpha: float) -> dict:
    """Build a 10x9 matrix of Cohen's d for a given alpha."""
    data = {}
    for dim in DIMENSIONS:
        row = []
        for layer in LAYERS:
            d = results.get(dim, {}).get(layer, {}).get(target_alpha)
            if d is not None:
                cohen_d = d.get("summary", {}).get("cohens_d", None)
                row.append(cohen_d)
            else:
                row.append(None)
        data[dim] = row
    return data


def get_top_configs(results: dict, n: int = 20) -> list:
    """Get top N configs by |Cohen's d|."""
    entries = []
    for dim in DIMENSIONS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                data = results.get(dim, {}).get(layer, {}).get(alpha)
                if data is None:
                    continue
                s = data.get("summary", {})
                d_val = s.get("cohens_d")
                if d_val is not None:
                    entries.append({
                        "dimension": dim,
                        "layer": layer,
                        "alpha": alpha,
                        "cohens_d": d_val,
                        "abs_d": abs(d_val),
                        "delta_pct": s.get("delta_proposer_pct"),
                        "p_value": s.get("paired_ttest_p"),
                        "steered_pct": s.get("steered_mean_proposer_pct"),
                        "baseline_pct": s.get("baseline_mean_proposer_pct"),
                        "steered_acc": s.get("steered_accept_rate"),
                        "baseline_acc": s.get("baseline_accept_rate"),
                        "steered_pay": s.get("steered_mean_payoff_pct"),
                        "baseline_pay": s.get("baseline_mean_payoff_pct"),
                    })
    entries.sort(key=lambda x: x["abs_d"], reverse=True)
    return entries[:n]


def per_dimension_summary(results: dict) -> list:
    """Per-dimension summary: peak layer, peak alpha, max d, direction."""
    summaries = []
    for dim in DIMENSIONS:
        best_d = 0
        best_layer = None
        best_alpha = None
        best_entry = None
        n_significant = 0

        for layer in LAYERS:
            for alpha in ALPHAS:
                data = results.get(dim, {}).get(layer, {}).get(alpha)
                if data is None:
                    continue
                s = data.get("summary", {})
                d_val = s.get("cohens_d")
                p_val = s.get("paired_ttest_p")
                if d_val is not None and abs(d_val) > abs(best_d):
                    best_d = d_val
                    best_layer = layer
                    best_alpha = alpha
                    best_entry = s
                if p_val is not None and p_val < 0.05:
                    n_significant += 1

        summaries.append({
            "dimension": dim,
            "peak_layer": best_layer,
            "peak_alpha": best_alpha,
            "max_d": best_d,
            "max_abs_d": abs(best_d),
            "direction": "+" if best_d > 0 else "-" if best_d < 0 else "0",
            "n_significant": n_significant,
            "n_total": 36,  # 9 layers x 4 alphas
            "peak_delta": best_entry.get("delta_proposer_pct") if best_entry else None,
            "peak_steered": best_entry.get("steered_mean_proposer_pct") if best_entry else None,
            "peak_baseline": best_entry.get("baseline_mean_proposer_pct") if best_entry else None,
            "peak_steered_acc": best_entry.get("steered_accept_rate") if best_entry else None,
            "peak_steered_pay": best_entry.get("steered_mean_payoff_pct") if best_entry else None,
        })

    summaries.sort(key=lambda x: x["max_abs_d"], reverse=True)
    return summaries


def dose_response(results: dict, dim: str, layer: int) -> list:
    """Get dose-response data for a dimension at a specific layer."""
    entries = []
    for alpha in sorted(ALPHAS):
        data = results.get(dim, {}).get(layer, {}).get(alpha)
        if data is None:
            entries.append({"alpha": alpha, "d": None, "delta": None, "p": None})
            continue
        s = data.get("summary", {})
        entries.append({
            "alpha": alpha,
            "d": s.get("cohens_d"),
            "delta": s.get("delta_proposer_pct"),
            "p": s.get("paired_ttest_p"),
            "steered_pct": s.get("steered_mean_proposer_pct"),
            "baseline_pct": s.get("baseline_mean_proposer_pct"),
            "steered_acc": s.get("steered_accept_rate"),
            "steered_pay": s.get("steered_mean_payoff_pct"),
        })
    return entries


def detect_anomalies(results: dict) -> list:
    """Detect configs with unusual patterns."""
    anomalies = []
    for dim in DIMENSIONS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                data = results.get(dim, {}).get(layer, {}).get(alpha)
                if data is None:
                    continue
                s = data.get("summary", {})
                cfg = data.get("config", {})

                # Check parse errors
                n_parse = s.get("n_parse_errors", 0)
                n_usable = s.get("n_usable_pairs", 0)
                if n_parse > 5:
                    anomalies.append(f"HIGH PARSE ERRORS: {dim} L{layer} a{alpha}: {n_parse} parse errors, {n_usable} usable")

                # Check if steered acceptance rate is 0
                s_acc = s.get("steered_accept_rate")
                if s_acc is not None and s_acc == 0.0:
                    anomalies.append(f"ZERO STEERED ACCEPTANCE: {dim} L{layer} a{alpha}: steered_acc=0.0%")

                # Check extreme demands
                s_pct = s.get("steered_mean_proposer_pct")
                if s_pct is not None and (s_pct > 95 or s_pct < 5):
                    anomalies.append(f"EXTREME DEMAND: {dim} L{layer} a{alpha}: steered={s_pct}%")

                # Check model/dtype
                if cfg.get("dtype") != "bfloat16":
                    anomalies.append(f"WRONG DTYPE: {dim} L{layer} a{alpha}: {cfg.get('dtype')}")
                if cfg.get("quantize", False):
                    anomalies.append(f"QUANTIZED: {dim} L{layer} a{alpha}")

                # Check elapsed time (way too fast means something broke)
                elapsed = s.get("elapsed_seconds", 0)
                if elapsed < 10 and n_usable > 0:
                    anomalies.append(f"SUSPICIOUSLY FAST: {dim} L{layer} a{alpha}: {elapsed}s for {n_usable} games")

    return anomalies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/ultimatum/final_7b_grid")
    p.add_argument("--landscape_dir", default="results/ultimatum/landscape_screen_v2",
                   help="Landscape screen results for comparison")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"FINAL 7B GRID — COMPREHENSIVE ANALYSIS")
    print(f"Results directory: {results_dir}")
    print(f"{'=' * 80}")

    # Load all results
    global ALPHAS
    results, ALPHAS = load_all_results(results_dir)
    print(f"  Alphas found: {ALPHAS}")
    print(f"  Expected configs: {len(DIMENSIONS)} dims x {len(LAYERS)} layers x {len(ALPHAS)} alphas = {len(DIMENSIONS)*len(LAYERS)*len(ALPHAS)}")

    # ============================================================
    # 1. Completeness check
    # ============================================================
    print(f"\n{'─' * 80}")
    print("1. COMPLETENESS CHECK")
    print(f"{'─' * 80}")
    present, missing = check_completeness(results)
    print(f"  Total files: {present} / 360")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for dim, layer, alpha in missing[:20]:
            print(f"    {dim} L{layer} a{alpha}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    else:
        print("  All 360 configs present.")

    # ============================================================
    # 2. Baseline consistency
    # ============================================================
    print(f"\n{'─' * 80}")
    print("2. BASELINE CONSISTENCY")
    print(f"{'─' * 80}")
    baselines, baseline_anomalies = check_baselines(results)
    if baselines:
        b_arr = np.array(baselines)
        print(f"  N baselines: {len(baselines)}")
        print(f"  Mean: {b_arr.mean():.2f}%  Std: {b_arr.std():.2f}%")
        print(f"  Range: [{b_arr.min():.2f}%, {b_arr.max():.2f}%]")
        print(f"  Expected range: 50-65% (RTX 3090 Ti bf16)")
        if baseline_anomalies:
            print(f"  ANOMALIES ({len(baseline_anomalies)}):")
            for dim, layer, alpha, val in baseline_anomalies[:10]:
                print(f"    {dim} L{layer} a{alpha}: baseline={val:.1f}%")
        else:
            print("  All baselines within expected range.")
    else:
        print("  No baselines loaded.")

    # ============================================================
    # 3. Heat maps
    # ============================================================
    print(f"\n{'─' * 80}")
    print("3. HEAT MAPS (Cohen's d)")
    print(f"{'─' * 80}")
    # Show heatmaps for the two most extreme alphas
    display_alphas = [max(ALPHAS), min(ALPHAS)] if len(ALPHAS) >= 2 else ALPHAS
    for target_alpha in display_alphas:
        print(f"\n  Alpha = {target_alpha:+.0f}")
        heatmap = build_heatmap_data(results, target_alpha)

        # Header
        header = f"  {'Dimension':<16s}"
        for layer in LAYERS:
            header += f" L{layer:>2d}"
        print(header)
        print("  " + "─" * (16 + 5 * len(LAYERS)))

        for dim in DIMENSIONS:
            row = f"  {dim:<16s}"
            for val in heatmap[dim]:
                if val is None:
                    row += "    ."
                else:
                    row += f" {val:+4.1f}" if abs(val) < 10 else f"{val:+5.1f}"
            print(row)

    # ============================================================
    # 4. Top 20 configs by |d|
    # ============================================================
    print(f"\n{'─' * 80}")
    print("4. TOP 20 CONFIGS BY |Cohen's d|")
    print(f"{'─' * 80}")
    top = get_top_configs(results, 20)
    print(f"  {'#':>3s}  {'Dimension':<16s} {'Layer':>5s} {'Alpha':>6s} {'d':>7s} "
          f"{'delta':>7s} {'p':>8s} {'S%':>5s} {'B%':>5s} {'S_acc':>5s} {'S_pay':>5s}")
    print("  " + "─" * 90)
    for i, e in enumerate(top, 1):
        p_str = f"{e['p_value']:.4f}" if e["p_value"] is not None else "?"
        print(f"  {i:3d}  {e['dimension']:<16s} L{e['layer']:>3d}  {e['alpha']:>+5.0f}  "
              f"{e['cohens_d']:>+6.3f}  {e['delta_pct']:>+6.1f}pp  {p_str:>8s}  "
              f"{e['steered_pct']:>4.1f}  {e['baseline_pct']:>4.1f}  "
              f"{e['steered_acc']:.2f}  {e['steered_pay']:>4.1f}" if all(v is not None for v in [
                  e['delta_pct'], e['steered_pct'], e['baseline_pct'], e['steered_acc'], e['steered_pay']
              ]) else f"  {i:3d}  {e['dimension']:<16s} L{e['layer']:>3d}  {e['alpha']:>+5.0f}  "
              f"{e['cohens_d']:>+6.3f}  (incomplete data)")

    # ============================================================
    # 5. Per-dimension summary
    # ============================================================
    print(f"\n{'─' * 80}")
    print("5. PER-DIMENSION SUMMARY")
    print(f"{'─' * 80}")
    dim_summaries = per_dimension_summary(results)
    print(f"  {'Dimension':<16s} {'PeakL':>5s} {'PeakA':>6s} {'max|d|':>7s} {'Dir':>4s} "
          f"{'Sig':>4s} {'Delta':>7s} {'Steered':>7s} {'Base':>7s}")
    print("  " + "─" * 75)
    for ds in dim_summaries:
        print(f"  {ds['dimension']:<16s} L{ds['peak_layer']:>3d}  {ds['peak_alpha']:>+5.0f}  "
              f"{ds['max_abs_d']:>6.3f}  {ds['direction']:>3s}  "
              f"{ds['n_significant']:>3d}  "
              f"{ds['peak_delta']:>+6.1f}pp  "
              f"{ds['peak_steered']:>5.1f}%  "
              f"{ds['peak_baseline']:>5.1f}%"
              if all(v is not None for v in [ds['peak_delta'], ds['peak_steered'], ds['peak_baseline']])
              else f"  {ds['dimension']:<16s}  (no data)")

    # ============================================================
    # 6. Dose-response for top 5 dimensions at peak layer
    # ============================================================
    print(f"\n{'─' * 80}")
    print("6. DOSE-RESPONSE (top 5 dims at peak layer)")
    print(f"{'─' * 80}")
    for ds in dim_summaries[:5]:
        dim = ds["dimension"]
        layer = ds["peak_layer"]
        print(f"\n  {dim} @ L{layer}:")
        dr = dose_response(results, dim, layer)
        print(f"    {'Alpha':>6s} {'d':>7s} {'Delta':>7s} {'p':>8s} {'Steered':>7s} {'Base':>7s} {'Acc':>5s} {'Pay':>5s}")
        for entry in dr:
            if entry["d"] is None:
                print(f"    {entry['alpha']:>+5.0f}  {'?':>7s}")
            else:
                print(f"    {entry['alpha']:>+5.0f}  {entry['d']:>+6.3f}  "
                      f"{entry['delta']:>+6.1f}pp  "
                      f"{entry['p']:.4f}  "
                      f"{entry['steered_pct']:>5.1f}%  "
                      f"{entry['baseline_pct']:>5.1f}%  "
                      f"{entry['steered_acc']:.2f}  "
                      f"{entry['steered_pay']:>4.1f}%"
                      if all(v is not None for v in [
                          entry['delta'], entry['p'], entry['steered_pct'],
                          entry['baseline_pct'], entry['steered_acc'], entry['steered_pay']
                      ]) else f"    {entry['alpha']:>+5.0f}  {entry['d']:>+6.3f}  (partial data)")

    # ============================================================
    # 7. Comparison with landscape screen
    # ============================================================
    print(f"\n{'─' * 80}")
    print("7. COMPARISON WITH LANDSCAPE SCREEN")
    print(f"{'─' * 80}")
    landscape_dir = Path(args.landscape_dir)
    if landscape_dir.exists():
        landscape, _ = load_all_results(landscape_dir)
        if landscape:
            print(f"  Loaded {sum(len(v2) for v in landscape.values() for v2 in v.values())} landscape configs")
            comparisons = []
            for dim in DIMENSIONS:
                for layer in LAYERS:
                    for alpha in ALPHAS:
                        grid_data = results.get(dim, {}).get(layer, {}).get(alpha)
                        land_data = landscape.get(dim, {}).get(layer, {}).get(alpha)
                        if grid_data and land_data:
                            g_d = grid_data.get("summary", {}).get("cohens_d")
                            l_d = land_data.get("summary", {}).get("cohens_d")
                            g_delta = grid_data.get("summary", {}).get("delta_proposer_pct")
                            l_delta = land_data.get("summary", {}).get("delta_proposer_pct")
                            if g_d is not None and l_d is not None:
                                comparisons.append({
                                    "dim": dim, "layer": layer, "alpha": alpha,
                                    "grid_d": g_d, "landscape_d": l_d,
                                    "grid_delta": g_delta, "landscape_delta": l_delta,
                                })

            if comparisons:
                print(f"  Overlapping configs: {len(comparisons)}")
                # Correlation
                grid_ds = [c["grid_d"] for c in comparisons]
                land_ds = [c["landscape_d"] for c in comparisons]
                corr = np.corrcoef(grid_ds, land_ds)[0, 1]
                print(f"  Cohen's d correlation: r = {corr:.3f}")

                # Direction agreement
                agree = sum(1 for g, l in zip(grid_ds, land_ds)
                            if (g > 0 and l > 0) or (g < 0 and l < 0) or (abs(g) < 0.1 and abs(l) < 0.1))
                print(f"  Direction agreement: {agree}/{len(comparisons)} ({agree/len(comparisons)*100:.0f}%)")

                # Show biggest disagreements
                for c in comparisons:
                    c["d_diff"] = abs(c["grid_d"] - c["landscape_d"])
                comparisons.sort(key=lambda x: x["d_diff"], reverse=True)
                print(f"\n  Biggest d discrepancies (grid n=50 vs landscape n=15):")
                print(f"    {'Dimension':<16s} {'L':>3s} {'a':>5s} {'Grid d':>7s} {'Land d':>7s} {'Diff':>6s}")
                for c in comparisons[:10]:
                    print(f"    {c['dim']:<16s} {c['layer']:>3d} {c['alpha']:>+4.0f}  "
                          f"{c['grid_d']:>+6.3f}  {c['landscape_d']:>+6.3f}  {c['d_diff']:>5.3f}")
        else:
            print("  No landscape results loaded.")
    else:
        print(f"  Landscape directory not found: {landscape_dir}")
        print("  Skipping comparison.")

    # ============================================================
    # 8. Anomalies
    # ============================================================
    print(f"\n{'─' * 80}")
    print("8. ANOMALIES & CONCERNS")
    print(f"{'─' * 80}")
    anomalies = detect_anomalies(results)
    if anomalies:
        print(f"  Found {len(anomalies)} anomalies:")
        for a in anomalies:
            print(f"    {a}")
    else:
        print("  No anomalies detected.")

    # ============================================================
    # Machine consistency check
    # ============================================================
    print(f"\n{'─' * 80}")
    print("9. MACHINE CONSISTENCY")
    print(f"{'─' * 80}")
    machines = defaultdict(list)
    for dim in DIMENSIONS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                data = results.get(dim, {}).get(layer, {}).get(alpha)
                if data:
                    machine = data.get("config", {}).get("machine", "unknown")
                    gpu = data.get("config", {}).get("gpu", "unknown")
                    gpu_cc = data.get("config", {}).get("gpu_cc", "unknown")
                    machines[machine].append((dim, layer, alpha, gpu, gpu_cc))

    for machine, configs in sorted(machines.items()):
        gpus = set(c[3] for c in configs)
        ccs = set(c[4] for c in configs)
        dims = sorted(set(c[0] for c in configs))
        print(f"  {machine}: {len(configs)} configs, GPU: {gpus}, CC: {ccs}")
        print(f"    Dimensions: {dims}")

    print(f"\n{'=' * 80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
