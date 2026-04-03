#!/usr/bin/env python3
"""
Compile overnight experiment results from all 3 machines.
Run AFTER pulling results with rsync.

Generates 5 reports:
  1. Tier 2 Validation (13 configs, n=50)
  2. Batch B Rerun (70 configs, n=15)
  3. Layer Gradient Complete (28 configs, n=50)
  4. Tier 3 Confirmation (5 configs, n=100)
  5. Overall Leaderboard (top 10 by |d|)
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def load_results(directory):
    """Load all JSON result files from a directory."""
    results = []
    d = Path(directory)
    if not d.exists():
        print(f"  WARNING: {directory} does not exist")
        return results
    for f in sorted(d.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append(data)
        except Exception as e:
            print(f"  ERROR loading {f}: {e}")
    return results


def extract_key_metrics(data):
    """Extract dimension, layer, alpha, d, p, delta from result data."""
    c = data.get("config", {})
    s = data.get("summary", {})
    return {
        "dimension": c.get("dimension", "?"),
        "layer": c.get("layers", [0])[0],
        "alpha": c.get("alpha", 0),
        "steered_pct": s.get("steered_mean_proposer_pct", "?"),
        "baseline_pct": s.get("baseline_mean_proposer_pct", "?"),
        "delta": s.get("delta_proposer_pct", "?"),
        "d": s.get("cohens_d", "?"),
        "p": s.get("paired_ttest_p", "?"),
        "n_usable": s.get("n_usable_pairs", "?"),
        "n_games": c.get("n_games", "?"),
        "steered_accept": s.get("steered_accept_rate", "?"),
        "baseline_accept": s.get("baseline_accept_rate", "?"),
        "steered_payoff": s.get("steered_mean_payoff_pct", "?"),
        "baseline_payoff": s.get("baseline_mean_payoff_pct", "?"),
        "machine": c.get("machine", "?"),
        "tier": c.get("tier", "?"),
    }


def sig_marker(p):
    if not isinstance(p, (int, float)):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def fmt(val, decimals=2):
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def report_tier2(results):
    """Report 1: Tier 2 Validation"""
    print("\n" + "=" * 80)
    print("REPORT 1: TIER 2 VALIDATION (n=50)")
    print("=" * 80)
    print(f"\n{'Config':<30} {'delta':>8} {'d':>8} {'p':>10} {'sig':>4} {'steered%':>10} {'base%':>8}")
    print("-" * 80)

    metrics = [extract_key_metrics(r) for r in results]
    metrics.sort(key=lambda x: abs(x["d"]) if isinstance(x["d"], (int, float)) else 0, reverse=True)

    for m in metrics:
        config = f"{m['dimension']} L{m['layer']} a={m['alpha']}"
        sig = sig_marker(m["p"])
        print(f"{config:<30} {fmt(m['delta']):>8} {fmt(m['d']):>8} {fmt(m['p'], 4):>10} {sig:>4} {fmt(m['steered_pct']):>10} {fmt(m['baseline_pct']):>8}")

    # Summary
    replicated = [m for m in metrics if isinstance(m["d"], (int, float)) and abs(m["d"]) > 0.5 and isinstance(m["p"], (int, float)) and m["p"] < 0.05]
    print(f"\nReplicated (|d|>0.5, p<0.05): {len(replicated)}/{len(metrics)}")
    for m in replicated:
        print(f"  {m['dimension']} L{m['layer']} a={m['alpha']}: d={fmt(m['d'])}")


def report_batch_b(results, old_results=None):
    """Report 2: Batch B Rerun comparison"""
    print("\n" + "=" * 80)
    print("REPORT 2: BATCH B RERUN (bird hardware, n=15)")
    print("=" * 80)

    # Group by dimension
    by_dim = defaultdict(list)
    for r in results:
        m = extract_key_metrics(r)
        by_dim[m["dimension"]].append(m)

    for dim in ["composure", "anchoring", "flattery", "greed", "undecidedness"]:
        if dim not in by_dim:
            print(f"\n  {dim}: NO DATA")
            continue
        print(f"\n  {dim.upper()}")
        print(f"  {'Layer':>6} {'a=-7 d':>8} {'a=-7 p':>8} {'a=7 d':>8} {'a=7 p':>8} {'a=-7 base%':>12} {'a=7 base%':>10}")
        print(f"  {'-'*66}")

        configs = by_dim[dim]
        by_layer = defaultdict(dict)
        for c in configs:
            key = "neg" if c["alpha"] < 0 else "pos"
            by_layer[c["layer"]][key] = c

        for layer in [6, 8, 10, 12, 14, 18, 20]:
            neg = by_layer[layer].get("neg", {})
            pos = by_layer[layer].get("pos", {})
            nd = fmt(neg.get("d", "")) if neg else ""
            np_ = fmt(neg.get("p", ""), 3) if neg else ""
            pd = fmt(pos.get("d", "")) if pos else ""
            pp = fmt(pos.get("p", ""), 3) if pos else ""
            nb = fmt(neg.get("baseline_pct", "")) if neg else ""
            pb = fmt(pos.get("baseline_pct", "")) if pos else ""
            print(f"  L{layer:>4} {nd:>8} {np_:>8} {pd:>8} {pp:>8} {nb:>12} {pb:>10}")

    # Baseline consistency check
    baselines = [m["baseline_pct"] for m in [extract_key_metrics(r) for r in results]
                 if isinstance(m["baseline_pct"], (int, float))]
    if baselines:
        mean_b = sum(baselines) / len(baselines)
        std_b = (sum((x - mean_b) ** 2 for x in baselines) / max(len(baselines) - 1, 1)) ** 0.5
        print(f"\n  Bird hardware baseline: mean={mean_b:.1f}%, std={std_b:.1f}%")


def report_layer_gradient(results):
    """Report 3: Layer Gradient Complete"""
    print("\n" + "=" * 80)
    print("REPORT 3: LAYER GRADIENT (n=50)")
    print("=" * 80)

    by_dim = defaultdict(list)
    for r in results:
        m = extract_key_metrics(r)
        by_dim[m["dimension"]].append(m)

    for dim in ["firmness", "empathy"]:
        if dim not in by_dim:
            print(f"\n  {dim}: NO DATA")
            continue
        print(f"\n  {dim.upper()}")
        print(f"  {'Layer':>6} {'a=3 d':>8} {'a=3 p':>8} {'a=3 sig':>8} {'a=10 d':>8} {'a=10 p':>8} {'a=10 sig':>8}")
        print(f"  {'-'*60}")

        configs = by_dim[dim]
        by_layer = defaultdict(dict)
        for c in configs:
            key = f"a{int(c['alpha'])}"
            by_layer[c["layer"]][key] = c

        for layer in [6, 8, 10, 12, 14, 18, 20]:
            a3 = by_layer[layer].get("a3", {})
            a10 = by_layer[layer].get("a10", {})
            d3 = fmt(a3.get("d", "")) if a3 else ""
            p3 = fmt(a3.get("p", ""), 4) if a3 else ""
            s3 = sig_marker(a3.get("p", 1)) if a3 else ""
            d10 = fmt(a10.get("d", "")) if a10 else ""
            p10 = fmt(a10.get("p", ""), 4) if a10 else ""
            s10 = sig_marker(a10.get("p", 1)) if a10 else ""
            print(f"  L{layer:>4} {d3:>8} {p3:>8} {s3:>8} {d10:>8} {p10:>8} {s10:>8}")


def report_tier3(results):
    """Report 4: Tier 3 Confirmation"""
    print("\n" + "=" * 80)
    print("REPORT 4: TIER 3 CONFIRMATION (n=100)")
    print("=" * 80)

    if not results:
        print("  NO DATA YET")
        return

    metrics = [extract_key_metrics(r) for r in results]
    metrics.sort(key=lambda x: abs(x["d"]) if isinstance(x["d"], (int, float)) else 0, reverse=True)

    print(f"\n{'Config':<28} {'delta':>7} {'d':>7} {'p':>10} {'sig':>4} {'steer%':>8} {'base%':>8} {'s_acc':>7} {'b_acc':>7} {'s_pay':>7} {'b_pay':>7}")
    print("-" * 105)

    for m in metrics:
        config = f"{m['dimension']} L{m['layer']} a={m['alpha']}"
        sig = sig_marker(m["p"])
        print(f"{config:<28} {fmt(m['delta']):>7} {fmt(m['d']):>7} {fmt(m['p'], 6):>10} {sig:>4} "
              f"{fmt(m['steered_pct']):>8} {fmt(m['baseline_pct']):>8} "
              f"{fmt(m['steered_accept'], 3):>7} {fmt(m['baseline_accept'], 3):>7} "
              f"{fmt(m['steered_payoff']):>7} {fmt(m['baseline_payoff']):>7}")


def report_leaderboard(all_results):
    """Report 5: Overall Leaderboard"""
    print("\n" + "=" * 80)
    print("REPORT 5: OVERALL LEADERBOARD (top 10 by |d|)")
    print("=" * 80)

    all_metrics = []
    for r in all_results:
        m = extract_key_metrics(r)
        if isinstance(m["d"], (int, float)) and isinstance(m["p"], (int, float)):
            all_metrics.append(m)

    all_metrics.sort(key=lambda x: abs(x["d"]), reverse=True)

    print(f"\n{'#':>3} {'Config':<28} {'n':>5} {'delta':>8} {'d':>8} {'p':>10} {'sig':>4} {'tier':>6}")
    print("-" * 80)

    for i, m in enumerate(all_metrics[:10], 1):
        config = f"{m['dimension']} L{m['layer']} a={m['alpha']}"
        sig = sig_marker(m["p"])
        print(f"{i:>3} {config:<28} {m['n_games']:>5} {fmt(m['delta']):>8} {fmt(m['d']):>8} {fmt(m['p'], 4):>10} {sig:>4} {m['tier']:>6}")


def main():
    base = Path("results/ultimatum")

    print("OVERNIGHT EXPERIMENT RESULTS")
    print(f"Compiled at: {os.popen('date').read().strip()}")
    print()

    # Load all result sets
    tier2 = load_results(base / "tier2_validation")
    tier3 = load_results(base / "tier3_confirmation")
    batch_b = load_results(base / "landscape_screen_v2")
    layer_grad = load_results(base / "layer_gradient_v2")

    print(f"Loaded: T2={len(tier2)}, T3={len(tier3)}, BatchB={len(batch_b)}, LayerGrad={len(layer_grad)}")

    # Generate reports
    if tier2:
        report_tier2(tier2)
    if batch_b:
        report_batch_b(batch_b)
    if layer_grad:
        report_layer_gradient(layer_grad)
    if tier3:
        report_tier3(tier3)

    # Leaderboard from T2 + T3 combined
    all_confirmed = tier2 + tier3
    if all_confirmed:
        report_leaderboard(all_confirmed)

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
