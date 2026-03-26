"""
Summarises all tested configs in results/ultimatum/ and ranks them.
Shows best layers, alphas, dims, and roles — sorted by Cohen's d then p-value.

Usage:
    python best_ultimatum_configs.py
    python best_ultimatum_configs.py --dir results/ultimatum --top 20
"""

import json, os, glob, argparse
import numpy as np
from scipy import stats


def load_file(path):
    with open(path) as f:
        return json.load(f)


def usable_games(games):
    return [
        g for g in games
        if g.get("parse_error") is None
        and g["steered"].get("parse_error") is None
        and g["baseline"].get("parse_error") is None
        and g["steered"].get("proposer_share") is not None
        and g["baseline"].get("proposer_share") is not None
    ]


def get_stats(path):
    data = load_file(path)
    cfg  = data["config"]
    role = cfg.get("steered_role") or data["summary"].get("steered_role") or "proposer"
    games = usable_games(data["games"])
    if len(games) < 5:
        return None  # too few usable games

    key    = "proposer_payoff" if role == "proposer" else "responder_payoff"
    deltas = np.array([(g["steered"][key] - g["baseline"][key]) / g["pool"] for g in games])
    t, p   = stats.ttest_1samp(deltas, 0)
    d      = deltas.mean() / deltas.std(ddof=1) if deltas.std(ddof=1) > 0 else 0.0

    # agreement rate delta
    steered_agreed  = np.mean([g["steered"]["agreed"]  for g in games])
    baseline_agreed = np.mean([g["baseline"]["agreed"] for g in games])

    return {
        "path":           path,
        "dim":            cfg.get("dimension", "?"),
        "role":           role,
        "layer":          cfg.get("layer", cfg.get("layers", "?")),
        "alpha":          cfg.get("alpha", "?"),
        "n":              len(deltas),
        "mean_pct":       deltas.mean() * 100,
        "std_pct":        deltas.std(ddof=1) * 100,
        "median_pct":     np.median(deltas) * 100,
        "t":              t,
        "p":              p,
        "d":              d,
        "agree_delta":    (steered_agreed - baseline_agreed) * 100,
    }


def sig_label(p, d):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "~"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  default="results/ultimatum")
    parser.add_argument("--top",  type=int, default=15, help="How many top configs to show per table")
    parser.add_argument("--role", default=None, help="Filter to 'proposer' or 'responder'")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    files = [f for f in files if "baseline" not in os.path.basename(f)]

    rows = []
    for f in files:
        try:
            s = get_stats(f)
            if s:
                rows.append(s)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")

    if args.role:
        rows = [r for r in rows if r["role"] == args.role]

    if not rows:
        print("No results found.")
        return

    # ── prefer n=100 over n=50 for same config ──────────────────────────────
    best_by_config = {}
    for r in rows:
        key = (r["dim"], r["role"], str(r["layer"]), r["alpha"])
        if key not in best_by_config or r["n"] > best_by_config[key]["n"]:
            best_by_config[key] = r
    rows = list(best_by_config.values())

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 1: All configs ranked by |Cohen's d| (positive effect first)
    # ─────────────────────────────────────────────────────────────────────────
    ranked = sorted(rows, key=lambda r: (-r["mean_pct"], r["p"]))

    print("\n" + "=" * 120)
    print("TABLE 1 — ALL CONFIGS ranked by mean effect (best first), deduplicated to largest n per config")
    print("=" * 120)
    hdr = f"{'dim':<30} {'role':<10} {'L':>4} {'alpha':>6} {'n':>4}  {'mean%':>7} {'std%':>7} {'median%':>8}  {'t':>7} {'p':>9} {'d':>6}  {'agree∆%':>8}  sig"
    print(hdr)
    print("-" * 120)
    for r in ranked:
        sig = sig_label(r["p"], r["d"])
        print(f"{r['dim']:<30} {r['role']:<10} {str(r['layer']):>4} {str(r['alpha']):>6} "
              f"{r['n']:>4}  {r['mean_pct']:>+7.1f} {r['std_pct']:>7.1f} {r['median_pct']:>+8.1f}  "
              f"{r['t']:>7.3f} {r['p']:>9.2e} {r['d']:>6.3f}  {r['agree_delta']:>+8.1f}  {sig}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 2: Best config per dimension
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("TABLE 2 — BEST CONFIG PER DIMENSION (by mean effect)")
    print("=" * 120)
    print(hdr)
    print("-" * 120)
    dims = sorted(set(r["dim"] for r in rows))
    for dim in dims:
        dim_rows = [r for r in rows if r["dim"] == dim]
        best = max(dim_rows, key=lambda r: r["mean_pct"])
        sig = sig_label(best["p"], best["d"])
        print(f"{best['dim']:<30} {best['role']:<10} {str(best['layer']):>4} {str(best['alpha']):>6} "
              f"{best['n']:>4}  {best['mean_pct']:>+7.1f} {best['std_pct']:>7.1f} {best['median_pct']:>+8.1f}  "
              f"{best['t']:>7.3f} {best['p']:>9.2e} {best['d']:>6.3f}  {best['agree_delta']:>+8.1f}  {sig}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 3: Layer breakdown (mean effect averaged across dims/alphas)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("TABLE 3 — LAYER BREAKDOWN (avg mean% across all dims & alphas at that layer)")
    print("=" * 120)
    layers = sorted(set(str(r["layer"]) for r in rows))
    layer_stats = []
    for L in layers:
        sub = [r for r in rows if str(r["layer"]) == L]
        avg_mean = np.mean([r["mean_pct"] for r in sub])
        avg_d    = np.mean([r["d"]        for r in sub])
        n_sig    = sum(1 for r in sub if r["p"] < 0.05)
        layer_stats.append((L, avg_mean, avg_d, n_sig, len(sub)))
    layer_stats.sort(key=lambda x: -x[1])
    print(f"  {'Layer':>6}  {'avg mean%':>10}  {'avg d':>7}  {'n_sig':>6}  {'n_configs':>10}")
    print("  " + "-" * 50)
    for L, avg_mean, avg_d, n_sig, n_cfg in layer_stats:
        print(f"  {str(L):>6}  {avg_mean:>+10.2f}  {avg_d:>7.3f}  {n_sig:>6}  {n_cfg:>10}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 4: Alpha breakdown
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("TABLE 4 — ALPHA BREAKDOWN (avg mean% across all dims & layers at that alpha)")
    print("=" * 120)
    alphas = sorted(set(r["alpha"] for r in rows))
    alpha_stats = []
    for a in alphas:
        sub = [r for r in rows if r["alpha"] == a]
        avg_mean = np.mean([r["mean_pct"] for r in sub])
        avg_d    = np.mean([r["d"]        for r in sub])
        n_sig    = sum(1 for r in sub if r["p"] < 0.05)
        alpha_stats.append((a, avg_mean, avg_d, n_sig, len(sub)))
    alpha_stats.sort(key=lambda x: -x[1])
    print(f"  {'Alpha':>7}  {'avg mean%':>10}  {'avg d':>7}  {'n_sig':>6}  {'n_configs':>10}")
    print("  " + "-" * 50)
    for a, avg_mean, avg_d, n_sig, n_cfg in alpha_stats:
        print(f"  {str(a):>7}  {avg_mean:>+10.2f}  {avg_d:>7.3f}  {n_sig:>6}  {n_cfg:>10}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 5: Significant results only
    # ─────────────────────────────────────────────────────────────────────────
    sig_rows = [r for r in rows if r["p"] < 0.05]
    print("\n" + "=" * 120)
    print(f"TABLE 5 — SIGNIFICANT RESULTS ONLY (p < 0.05)  [{len(sig_rows)} found]")
    print("=" * 120)
    if sig_rows:
        print(hdr)
        print("-" * 120)
        for r in sorted(sig_rows, key=lambda r: r["p"]):
            sig = sig_label(r["p"], r["d"])
            print(f"{r['dim']:<30} {r['role']:<10} {str(r['layer']):>4} {str(r['alpha']):>6} "
                  f"{r['n']:>4}  {r['mean_pct']:>+7.1f} {r['std_pct']:>7.1f} {r['median_pct']:>+8.1f}  "
                  f"{r['t']:>7.3f} {r['p']:>9.2e} {r['d']:>6.3f}  {r['agree_delta']:>+8.1f}  {sig}")
    else:
        print("  (none)")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"  Total configs evaluated : {len(rows)}")
    print(f"  p < 0.05                : {sum(1 for r in rows if r['p'] < 0.05)}")
    print(f"  p < 0.10                : {sum(1 for r in rows if r['p'] < 0.10)}")
    print(f"  Positive mean effect    : {sum(1 for r in rows if r['mean_pct'] > 0)}")
    print(f"  Negative mean effect    : {sum(1 for r in rows if r['mean_pct'] < 0)}")
    overall_best = max(rows, key=lambda r: r["mean_pct"])
    print(f"\n  Single best config      : {overall_best['dim']}  role={overall_best['role']}  "
          f"L={overall_best['layer']}  alpha={overall_best['alpha']}  "
          f"mean={overall_best['mean_pct']:+.1f}%  p={overall_best['p']:.3e}  d={overall_best['d']:.3f}")
    print()


if __name__ == "__main__":
    main()
