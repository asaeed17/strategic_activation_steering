"""
Compares t-test / p-value results between n=50 and n=100 ultimatum game files.

For each dimension+layer+alpha combo that has both a 50-game and 100-game file,
prints the stats side by side.

Usage:
    python compare_50_vs_100.py
    python compare_50_vs_100.py --dir results/ultimatum
"""

import json, os, glob, argparse, re
import numpy as np
from scipy import stats


def load(path):
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
    data  = load(path)
    cfg   = data["config"]
    role  = cfg.get("steered_role") or data["summary"].get("steered_role") or "proposer"
    games = usable_games(data["games"])
    key   = "proposer_payoff" if role == "proposer" else "responder_payoff"

    deltas = [(g["steered"][key] - g["baseline"][key]) / g["pool"] for g in games]
    arr = np.array(deltas)
    t, p = stats.ttest_1samp(arr, 0)
    d = arr.mean() / arr.std(ddof=1) if arr.std(ddof=1) > 0 else 0.0

    return {
        "n":      len(arr),
        "mean":   arr.mean() * 100,
        "std":    arr.std(ddof=1) * 100,
        "median": np.median(arr) * 100,
        "t":      t,
        "p":      p,
        "d":      d,
        "role":   role,
    }


def extract_key(filename):
    """Turn e.g. firmness_proposer_L12_a6.0_paired_n50 → (firmness, proposer, L12, a6.0)"""
    m = re.match(r"(.+)_(proposer|responder)_(L[\d]+)_(a[\d.]+)_paired_n\d+", filename)
    if m:
        return (m.group(1), m.group(2), m.group(3), m.group(4))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/ultimatum")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    files = [f for f in files if "baseline" not in os.path.basename(f)]

    # group by (dim, role, layer, alpha)
    groups = {}
    for f in files:
        name = os.path.basename(f).replace(".json", "")
        key  = extract_key(name)
        if key is None:
            continue
        m = re.search(r"_n(\d+)$", name)
        n = int(m.group(1)) if m else None
        if n not in (50, 100):
            continue
        groups.setdefault(key, {})[n] = f

    # keep only pairs that have both 50 and 100
    pairs = {k: v for k, v in groups.items() if 50 in v and 100 in v}

    if not pairs:
        print("No matching 50/100 pairs found.")
        return

    print(f"Found {len(pairs)} matched pairs (50-game vs 100-game)\n")
    print(f"{'Config':<45}  {'n':>4}  {'mean%':>7}  {'std%':>7}  "
          f"{'median%':>8}  {'t':>7}  {'p':>9}  {'d':>6}  sig")
    print("-" * 110)

    # sort by dimension then layer then alpha
    for key in sorted(pairs):
        dim, role, layer, alpha = key
        label = f"{dim}_{role}_{layer}_{alpha}"

        for n in (50, 100):
            try:
                s = get_stats(pairs[key][n])
                sig = "**" if s["p"] < 0.05 else ("~" if s["p"] < 0.10 else "")
                print(f"  {label:<43}  {s['n']:>4}  {s['mean']:>+7.1f}  {s['std']:>7.1f}  "
                      f"{s['median']:>+8.1f}  {s['t']:>7.3f}  {s['p']:>9.2e}  "
                      f"{s['d']:>6.3f}  {sig}")
            except Exception as e:
                print(f"  {label:<43}  n={n}  ERROR: {e}")

        print()  # blank line between pairs

    # summary: does n=100 consistently give lower p than n=50?
    better_at_100 = 0
    better_at_50  = 0
    for key in pairs:
        try:
            s50  = get_stats(pairs[key][50])
            s100 = get_stats(pairs[key][100])
            if s100["p"] < s50["p"]:
                better_at_100 += 1
            else:
                better_at_50 += 1
        except:
            pass

    total = better_at_100 + better_at_50
    print("=" * 110)
    print(f"\nSUMMARY: n=100 gives lower p-value in {better_at_100}/{total} pairs "
          f"({better_at_100/total*100:.0f}%)")
    print(f"         n=50  gives lower p-value in {better_at_50}/{total} pairs "
          f"({better_at_50/total*100:.0f}%)")

    if better_at_100 > better_at_50:
        print("\n→ More games consistently helps. Variance is the bottleneck, not sample size.")
    else:
        print("\n→ More games is NOT reliably helping. The effect itself may be too noisy to detect.")


if __name__ == "__main__":
    main()
