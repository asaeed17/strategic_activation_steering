"""
Simple validation of ultimatum game steering results.

Checks per file:
  1. Statistical significance — THREE separate tests:
       1a. Offer shift        (paired t-test on proposer_share/pool)
       1b. Acceptance shift   (chi-square on accepted counts)
       1c. Payoff delta       (one-sample t-test, kept for reference but noisy)
  2. Responder rationality  — accepts offers <= 5% of pool (should reject)
  3. Proposer rationality   — proposer keeps < 5% of pool (gives it away)
  4. Say/do mismatch        — text says "small amount" but proposes large offer (or vice versa)
  5. Outlier influence      — how much of the mean delta comes from top 3 games

The headline claim "steering with vector X caused Y% more payoff" is supported
by tests 1a and 1b. The payoff delta (1c) is reported as an observed magnitude
but NOT used for significance — it is zero-inflated by rejections and will
have low power. Use 1a+1b p-values in any written claim.

Usage:
    python validate_ultimatum_results.py
    python validate_ultimatum_results.py --dir results/ultimatum
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


def payoff_delta(game, role):
    key = "proposer_payoff" if role == "proposer" else "responder_payoff"
    s = game["steered"][key] / game["pool"]
    b = game["baseline"][key] / game["pool"]
    return s - b


# ── check 1a: offer shift ─────────────────────────────────────────────────────
# Cleanest test — continuous, no zeroing. Tests whether steering shifts
# how much the proposer keeps, independent of whether it gets accepted.

def check_offer_shift(games):
    steered_offers  = [g["steered"]["proposer_share"]  / g["pool"] for g in games]
    baseline_offers = [g["baseline"]["proposer_share"] / g["pool"] for g in games]

    arr = np.array(steered_offers) - np.array(baseline_offers)
    t, p = stats.ttest_1samp(arr, 0)
    d = arr.mean() / arr.std(ddof=1) if arr.std(ddof=1) > 0 else 0.0

    sig = "SIGNIFICANT" if p < 0.05 else ("TREND" if p < 0.10 else "NOT SIGNIFICANT")
    mean_s = np.mean(steered_offers) * 100
    mean_b = np.mean(baseline_offers) * 100

    print(f"  [1a offer shift]     steered={mean_s:.1f}%  baseline={mean_b:.1f}%  "
          f"delta={arr.mean()*100:+.1f}%")
    print(f"                       t={t:.3f}  p={p:.2e}  d={d:.3f}  → {sig}")

    return p, d, arr.mean()


# ── check 1b: acceptance shift ────────────────────────────────────────────────
# Tests whether steering changes acceptance rate using chi-square.
# Binary outcome — appropriate test, no zero-inflation issue.

def check_acceptance_shift(games):
    s_accepted  = sum(1 for g in games if g["steered"].get("agreed"))
    s_rejected  = len(games) - s_accepted
    b_accepted  = sum(1 for g in games if g["baseline"].get("agreed"))
    b_rejected  = len(games) - b_accepted

    # chi-square on 2x2 contingency table
    contingency = np.array([[s_accepted, s_rejected],
                             [b_accepted, b_rejected]])

    # use Fisher's exact if any cell is small
    if np.any(contingency < 5):
        _, p = stats.fisher_exact(contingency)
        test_name = "Fisher"
    else:
        chi2, p, _, _ = stats.chi2_contingency(contingency, correction=False)
        test_name = "chi2"

    s_rate = s_accepted / len(games)
    b_rate = b_accepted / len(games)
    delta  = s_rate - b_rate

    sig = "SIGNIFICANT" if p < 0.05 else ("TREND" if p < 0.10 else "NOT SIGNIFICANT")
    print(f"  [1b acceptance]      steered={s_rate:.1%}  baseline={b_rate:.1%}  "
          f"delta={delta:+.1%}")
    print(f"                       {test_name}  p={p:.2e}  → {sig}")

    return p, delta


# ── check 1c: payoff delta — permutation test ────────────────────────────────
# Uses a permutation test instead of t-test because payoff is zero-inflated
# by rejections (not normally distributed). The permutation test makes no
# distributional assumptions — it asks: "how often would we see a mean delta
# this large if steering had no effect?" by randomly flipping steered/baseline
# labels within each game pair and recomputing the mean delta.
# n_permutations=10000 gives stable p-values to ~2 decimal places.

def check_payoff_delta(games, role, n_permutations=10000, seed=0):
    rng = np.random.default_rng(seed)
    deltas = np.array([payoff_delta(g, role) for g in games])
    observed_mean = deltas.mean()
    median = np.median(deltas)

    # Each game pair can have its steered/baseline labels flipped independently.
    # Flipping a pair negates its delta. Count how often a random permutation
    # produces a mean as extreme as observed.
    flip_signs = rng.choice([-1, 1], size=(n_permutations, len(deltas)))
    permuted_means = (flip_signs * deltas).mean(axis=1)

    if observed_mean >= 0:
        p_one_tail = float((permuted_means >= observed_mean).mean())
    else:
        p_one_tail = float((permuted_means <= observed_mean).mean())
    p_value = min(p_one_tail * 2, 1.0)  # two-tailed

    sig = "SIGNIFICANT" if p_value < 0.05 else ("TREND" if p_value < 0.10 else "NOT SIGNIFICANT")
    print(f"  [1c payoff delta]    mean={observed_mean*100:+.1f}%  median={median*100:+.1f}%  "
          f"std={deltas.std(ddof=1)*100:.1f}%")
    print(f"                       permutation test (n={n_permutations})  "
          f"p={p_value:.2e}  → {sig}")

    if abs(observed_mean) > 0 and abs(median) < abs(observed_mean) * 0.5:
        print(f"                       ⚠ mean >> median — result may be driven by outliers")

    return p_value, observed_mean


# ── check 2: responder rationality ───────────────────────────────────────────

def check_responder_rationality(games):
    bad = []
    for g in games:
        for side in ("steered", "baseline"):
            s = g[side]
            if s.get("agreed") and s.get("responder_share") is not None:
                pct = s["responder_share"] / g["pool"]
                if pct <= 0.05:
                    bad.append((g["game_id"], side, s["responder_share"], g["pool"], round(pct*100,1)))

    n_steered  = sum(1 for _, side, *_ in bad if side == "steered")
    n_baseline = sum(1 for _, side, *_ in bad if side == "baseline")
    if not bad:
        print(f"  [responder rationality]  OK — no irrational accepts (<=5% of pool)")
    else:
        print(f"  [responder rationality]  ⚠ {len(bad)} irrational accepts  "
              f"(steered={n_steered}, baseline={n_baseline})")
        if n_baseline > n_steered:
            print(f"    → mostly baseline failures: inflates steered advantage artificially")
        elif n_steered > n_baseline:
            print(f"    → mostly steered failures: steering causing worse responder behaviour")
        for gid, side, share, pool, pct in bad[:5]:
            print(f"    game {gid} ({side}): accepted {share}/{pool} = {pct}%")
        if len(bad) > 5:
            print(f"    ... and {len(bad)-5} more")

    return len(bad)


# ── check 3: proposer rationality ────────────────────────────────────────────

def check_proposer_rationality(games):
    bad = []
    for g in games:
        for side in ("steered", "baseline"):
            s = g[side]
            if s.get("proposer_share") is not None:
                pct = s["proposer_share"] / g["pool"]
                if pct < 0.05:
                    bad.append((g["game_id"], side, s["proposer_share"], g["pool"], round(pct*100,1)))

    n_steered  = sum(1 for _, side, *_ in bad if side == "steered")
    n_baseline = sum(1 for _, side, *_ in bad if side == "baseline")
    if not bad:
        print(f"  [proposer rationality]   OK — no pathological offers (<5% kept)")
    else:
        print(f"  [proposer rationality]   ⚠ {len(bad)} pathological offers  "
              f"(steered={n_steered}, baseline={n_baseline})")
        if n_baseline > n_steered:
            print(f"    → mostly baseline failures: baseline proposer sometimes gives pool away, inflating delta")
        elif n_steered > n_baseline:
            print(f"    → mostly steered failures: steering causing proposer to give pool away")
        for gid, side, share, pool, pct in bad[:5]:
            print(f"    game {gid} ({side}): kept {share}/{pool} = {pct}%")
        if len(bad) > 5:
            print(f"    ... and {len(bad)-5} more")

    return len(bad)


# ── check 4: say/do mismatch ─────────────────────────────────────────────────

SMALL_WORDS = re.compile(r"\b(small amount|minimal|token|little|tiny|slight)\b", re.I)
LARGE_WORDS = re.compile(r"\b(large|significant(ly)?|substantial|majority|most of)\b", re.I)
FAIR_WORDS  = re.compile(r"\b(fair|equal|50.50|50\/50|even split)\b", re.I)

def check_say_do_mismatch(games):
    mismatches = []
    for g in games:
        for side in ("steered", "baseline"):
            s = g[side]
            text = s.get("proposer_text", "") or ""
            share = s.get("proposer_share")
            if share is None:
                continue
            pct = share / g["pool"]

            if SMALL_WORDS.search(text) and pct > 0.60:
                mismatches.append((g["game_id"], side, "says 'small' but keeps", pct, text[:80]))
            if LARGE_WORDS.search(text) and pct < 0.30:
                mismatches.append((g["game_id"], side, "says 'large/majority' but keeps", pct, text[:80]))
            if FAIR_WORDS.search(text) and pct > 0.70:
                mismatches.append((g["game_id"], side, "says 'fair/equal' but keeps", pct, text[:80]))

    n_steered  = sum(1 for _, side, *_ in mismatches if side == "steered")
    n_baseline = sum(1 for _, side, *_ in mismatches if side == "baseline")
    if not mismatches:
        print(f"  [say/do mismatch]        OK — no contradictions found")
    else:
        print(f"  [say/do mismatch]        ⚠ {len(mismatches)} contradictions  "
              f"(steered={n_steered}, baseline={n_baseline})")
        if n_baseline > n_steered:
            print(f"    → mostly baseline: baseline model is incoherent, not a steering artefact")
        elif n_steered > n_baseline:
            print(f"    → mostly steered: steering is causing incoherent reasoning")
        else:
            print(f"    → equal on both sides: likely a base model behaviour, not steering-specific")
        for gid, side, msg, pct, txt in mismatches[:5]:
            print(f"    game {gid} ({side}): {msg} {pct:.0%}")
            print(f"      text: \"{txt.strip()}\"")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches)-5} more")

    return len(mismatches)


# ── check 5: outlier influence ───────────────────────────────────────────────

def check_outlier_influence(games, role):
    deltas = sorted(
        [(payoff_delta(g, role), g["game_id"]) for g in games],
        key=lambda x: -abs(x[0])
    )
    arr = np.array([d for d, _ in deltas])
    total_mean = arr.mean()
    if total_mean == 0:
        print(f"  [outlier influence]      mean delta is 0, skipping")
        return

    top3_mean = np.array([d for d, _ in deltas[:3]]).mean()
    rest_mean  = np.array([d for d, _ in deltas[3:]]).mean() if len(deltas) > 3 else 0
    top3_ids   = [gid for _, gid in deltas[:3]]

    pct_from_top3 = (top3_mean * 3) / (total_mean * len(arr)) * 100

    flag = "⚠ HIGH" if pct_from_top3 > 50 else "OK"
    print(f"  [outlier influence]      top 3 games contribute {pct_from_top3:.0f}% of total mean delta  → {flag}")
    print(f"    top 3 game IDs: {top3_ids}  |  rest mean: {rest_mean*100:+.1f}%")

    if rest_mean <= 0 and total_mean > 0:
        print(f"    ⚠ Without top 3 games, mean delta is {rest_mean*100:+.1f}% — effect may not generalise")


# ── verdict helper ────────────────────────────────────────────────────────────

def build_claim(offer_p, offer_delta, accept_p, accept_delta, payoff_delta_mean,
                payoff_p, dimension, alpha, role):
    """
    Construct the plain-English claim supported by all three tests.
    Payoff significance now comes from the permutation test.
    """
    lines = []
    lines.append(f"  CLAIM: Steering '{dimension}' at α={alpha} ({role})")

    offer_sig  = offer_p  < 0.05
    accept_sig = accept_p < 0.05
    payoff_sig = payoff_p < 0.05

    if offer_sig:
        direction = "higher" if offer_delta > 0 else "lower"
        lines.append(f"    • Significantly shifted proposer offers {direction} "
                     f"by {offer_delta*100:+.1f}% of pool  (p={offer_p:.2e})")
    else:
        lines.append(f"    • No significant shift in proposer offers  (p={offer_p:.2e})")

    if accept_sig:
        direction = "increased" if accept_delta > 0 else "decreased"
        lines.append(f"    • Significantly {direction} acceptance rate "
                     f"by {accept_delta:+.1%}  (p={accept_p:.2e})")
    else:
        lines.append(f"    • No significant shift in acceptance rate  (p={accept_p:.2e})")

    if payoff_sig:
        direction = "increased" if payoff_delta_mean > 0 else "decreased"
        lines.append(f"    • Payoff significantly {direction} "
                     f"by {payoff_delta_mean*100:+.1f}% of pool  "
                     f"(permutation test p={payoff_p:.2e})")
    else:
        lines.append(f"    • No significant payoff shift  "
                     f"(permutation test p={payoff_p:.2e}, observed {payoff_delta_mean*100:+.1f}%)")

    return "\n".join(lines)


# ── per-file runner ───────────────────────────────────────────────────────────

def analyse(path):
    data  = load(path)
    cfg   = data["config"]
    summ  = data["summary"]
    role  = cfg.get("steered_role") or summ.get("steered_role") or "proposer"
    games = usable_games(data["games"])

    print(f"\n{'='*70}")
    print(f"FILE  : {os.path.basename(path)}")
    print(f"CONFIG: dim={cfg['dimension']}  layer={cfg['layers']}  "
          f"alpha={cfg['alpha']}  role={role}  n={cfg['n_games']}")
    s_pay = summ.get('steered_mean_payoff_pct') or summ.get('steered_mean_responder_payoff_pct')
    b_pay = summ.get('baseline_mean_payoff_pct') or summ.get('baseline_mean_responder_payoff_pct')
    s_acc = summ.get('steered_accept_rate')
    b_acc = summ.get('baseline_accept_rate')
    acc_str = f"{s_acc:.0%} vs {b_acc:.0%}" if s_acc is not None and b_acc is not None else "n/a"
    print(f"SUMMARY: steered_payoff={s_pay:.1f}%  baseline_payoff={b_pay:.1f}%  accept: {acc_str}")
    print()

    offer_p,  offer_d,  offer_delta  = check_offer_shift(games)
    accept_p, accept_delta           = check_acceptance_shift(games)
    payoff_p, payoff_mean            = check_payoff_delta(games, role)
    print()
    n_irrat_resp = check_responder_rationality(games)
    n_patho_prop = check_proposer_rationality(games)
    n_mismatch   = check_say_do_mismatch(games)
    print()
    check_outlier_influence(games, role)

    print()
    print(build_claim(offer_p, offer_delta, accept_p, accept_delta,
                      payoff_mean, payoff_p, cfg['dimension'], cfg['alpha'], role))

    # verdict: pass if ANY of the three tests is significant
    issues = []
    if offer_p >= 0.05 and accept_p >= 0.05 and payoff_p >= 0.05:
        issues.append("no significant effect on offer, acceptance, or payoff")
    if n_irrat_resp > len(games) * 0.05:
        issues.append(f"responder irrational in {n_irrat_resp} games")
    if n_patho_prop > len(games) * 0.05:
        issues.append(f"proposer pathological in {n_patho_prop} games")
    if n_mismatch > 5:
        issues.append(f"{n_mismatch} say/do mismatches")

    print()
    if not issues:
        print(f"  VERDICT: PASS — at least one clean effect is significant")
    else:
        print(f"  VERDICT: FLAG — {' | '.join(issues)}")

    return offer_p < 0.05 or accept_p < 0.05 or payoff_p < 0.05


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/ultimatum")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    files = [f for f in files if "baseline" not in os.path.basename(f)]

    passed = []
    for f in files:
        try:
            ok = analyse(f)
            if ok:
                passed.append(os.path.basename(f))
        except Exception as e:
            print(f"\nERROR {os.path.basename(f)}: {e}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(passed)}/{len(files)} files passed (offer, acceptance, or payoff p<0.05)")
    for f in passed:
        print(f"  ✓ {f}")


if __name__ == "__main__":
    main()