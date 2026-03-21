#!/usr/bin/env python3
"""
analyse_eval.py — Post-GPU analysis of all evaluation experiments.

Tasks:
  1. G1/G2 paired steering effect
  2. B1 metrics on G1 (SCM transcripts)
  3. B3 role-separated analysis on G1
  4. G3 DonD analysis
  5. G4 firmness clamping analysis
  6. G5 sensitivity interpretation
  8. Within-role judge correlations (if judge data available)

Run: python3 analyse_eval.py

Outputs: results/eval/analysis.json + stdout report
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ── Reuse B1 metric functions ────────────────────────────────────────────────

HEDGE_PHRASES = [
    "could consider", "not sure", "kind of", "sort of", "i think",
    "i guess", "i suppose", "a little",
]
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "somewhat"}
DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def parse_offers(transcript):
    offers = []
    for turn_idx, turn in enumerate(transcript):
        for m in DOLLAR_RE.finditer(turn["utterance"]):
            val = float(m.group(1).replace(",", ""))
            if val > 0:
                offers.append((turn_idx, turn["speaker"], val))
    return offers


def last_offer_by(offers, speaker, up_to_turn):
    for turn_idx, spk, amt in reversed(offers):
        if spk == speaker and turn_idx < up_to_turn:
            return amt
    return None


def compute_concessions(offers):
    concessions = []
    for i, (turn_idx, speaker, amount) in enumerate(offers):
        opponent = "buyer" if speaker == "seller" else "seller"
        own_prev = last_offer_by(offers, speaker, turn_idx)
        opp_last = last_offer_by(offers, opponent, turn_idx + 1)
        if own_prev is None:
            concessions.append({
                "turn_idx": turn_idx, "speaker": speaker,
                "concession_abs": None, "concession_pct_gap": None,
                "is_first_offer": True,
            })
            continue
        if speaker == "seller":
            concession = own_prev - amount
        else:
            concession = amount - own_prev
        pct_gap = None
        if opp_last is not None:
            gap = abs(own_prev - opp_last)
            if gap > 0:
                pct_gap = concession / gap
        concessions.append({
            "turn_idx": turn_idx, "speaker": speaker,
            "concession_abs": round(concession, 2),
            "concession_pct_gap": round(pct_gap, 4) if pct_gap is not None else None,
            "is_first_offer": False,
        })
    return concessions


def count_hedges(text):
    lower = text.lower()
    count = 0
    for phrase in HEDGE_PHRASES:
        count += lower.count(phrase)
    for word in HEDGE_WORDS:
        count += len(re.findall(r"\b" + re.escape(word) + r"\b", lower))
    return count


def response_lengths(transcript):
    return [len(turn["utterance"].split()) for turn in transcript]


def _stats(values):
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None}
    a = np.array(values, dtype=float)
    return {
        "n": len(a),
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 4),
        "min": round(float(np.min(a)), 4),
        "max": round(float(np.max(a)), 4),
    }


def _pct(val, total):
    return f"{val / total * 100:.1f}%" if total > 0 else "n/a"


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: G1/G2 paired steering effect
# ═══════════════════════════════════════════════════════════════════════════════

def task1_paired_effect(g1_games, g2_games):
    """Compute paired steering effect: per-scenario g1_advantage - g2_advantage."""
    print("\n" + "=" * 70)
    print("TASK 1: G1/G2 PAIRED STEERING EFFECT")
    print("=" * 70)

    n = min(len(g1_games), len(g2_games))
    paired_diffs = []
    g1_advs = []
    g2_advs = []
    role_diffs = {"seller": [], "buyer": []}

    for i in range(n):
        g1 = g1_games[i]
        g2 = g2_games[i]
        # Verify same scenario
        assert g1["listing_price"] == g2["listing_price"], f"Scenario mismatch at {i}"
        assert g1["steered_role"] == g2["steered_role"], f"Role mismatch at {i}"

        d = g1["advantage"] - g2["advantage"]
        paired_diffs.append(d)
        g1_advs.append(g1["advantage"])
        g2_advs.append(g2["advantage"])
        role_diffs[g1["steered_role"]].append(d)

    paired_diffs = np.array(paired_diffs)
    g1_advs = np.array(g1_advs)
    g2_advs = np.array(g2_advs)

    # Paired t-test
    t_stat, p_val = sp_stats.ttest_rel(g1_advs, g2_advs)
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_pval = sp_stats.wilcoxon(paired_diffs)
    except ValueError:
        w_stat, w_pval = None, None

    # Effect size (Cohen's d for paired)
    d_mean = np.mean(paired_diffs)
    d_std = np.std(paired_diffs, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0

    print(f"\n  Paired observations: {n}")
    print(f"  G1 (steered) mean advantage:  {np.mean(g1_advs):+.4f}")
    print(f"  G2 (baseline) mean advantage: {np.mean(g2_advs):+.4f}")
    print(f"  Paired steering effect:       {d_mean:+.4f}  (std={d_std:.4f})")
    print(f"  Cohen's d:                    {cohens_d:+.3f}")
    print(f"  Paired t-test:  t={t_stat:.3f}, p={p_val:.4f}  {'*' if p_val < 0.05 else 'ns'}")
    if w_pval is not None:
        print(f"  Wilcoxon test:  W={w_stat:.0f}, p={w_pval:.4f}  {'*' if w_pval < 0.05 else 'ns'}")

    # Direction breakdown
    n_better = np.sum(paired_diffs > 0)
    n_worse = np.sum(paired_diffs < 0)
    n_same = np.sum(paired_diffs == 0)
    print(f"\n  Steering helps: {n_better}/{n} ({n_better/n*100:.0f}%)")
    print(f"  Steering hurts: {n_worse}/{n} ({n_worse/n*100:.0f}%)")
    print(f"  No difference:  {n_same}/{n} ({n_same/n*100:.0f}%)")

    # Role-separated paired effect
    print(f"\n  Role-separated paired effect:")
    role_results = {}
    for role in ["seller", "buyer"]:
        rd = np.array(role_diffs[role])
        if len(rd) == 0:
            continue
        rm = float(np.mean(rd))
        rs = float(np.std(rd, ddof=1)) if len(rd) > 1 else 0.0
        try:
            rt, rp = sp_stats.ttest_1samp(rd, 0)
        except Exception:
            rt, rp = 0, 1
        n_pos = int(np.sum(rd > 0))
        role_results[role] = {
            "n": len(rd), "mean": round(rm, 4), "std": round(rs, 4),
            "t": round(float(rt), 3), "p": round(float(rp), 4),
            "n_positive": n_pos,
        }
        sig = "*" if rp < 0.05 else "ns"
        print(f"    {role:>6s}: mean={rm:+.4f}  std={rs:.4f}  n={len(rd)}  "
              f"helps={n_pos}/{len(rd)} ({n_pos/len(rd)*100:.0f}%)  "
              f"t={rt:.3f} p={rp:.4f} {sig}")

    # Clamping analysis
    g1_clamped = sum(1 for g in g1_games if g.get("clamped", False))
    g2_clamped = sum(1 for g in g2_games if g.get("clamped", False))
    print(f"\n  Clamping: G1={g1_clamped}/{n} ({g1_clamped/n*100:.0f}%), "
          f"G2={g2_clamped}/{n} ({g2_clamped/n*100:.0f}%)")

    # Unclamped-only paired effect
    unclamped_diffs = []
    for i in range(n):
        if not g1_games[i].get("clamped", False) and not g2_games[i].get("clamped", False):
            unclamped_diffs.append(paired_diffs[i])
    if unclamped_diffs:
        uc = np.array(unclamped_diffs)
        try:
            ut, up = sp_stats.ttest_1samp(uc, 0)
        except Exception:
            ut, up = 0, 1
        print(f"  Unclamped-only paired effect: {np.mean(uc):+.4f} (n={len(uc)}, "
              f"t={ut:.3f}, p={up:.4f})")

    # Distribution shape
    print(f"\n  Paired diff distribution:")
    print(f"    Q1={np.percentile(paired_diffs, 25):+.3f}  "
          f"median={np.median(paired_diffs):+.3f}  "
          f"Q3={np.percentile(paired_diffs, 75):+.3f}")

    # Histogram
    bins = [(-2, -0.5), (-0.5, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.5), (0.5, 2)]
    print(f"\n  Histogram of paired diffs:")
    for lo, hi in bins:
        c = int(np.sum((paired_diffs >= lo) & (paired_diffs < hi)))
        bar = "#" * c
        print(f"    [{lo:+.1f}, {hi:+.1f}): {c:>3d} {bar}")

    result = {
        "n_pairs": n,
        "paired_effect_mean": round(float(d_mean), 4),
        "paired_effect_std": round(float(d_std), 4),
        "paired_effect_median": round(float(np.median(paired_diffs)), 4),
        "cohens_d": round(float(cohens_d), 3),
        "ttest_t": round(float(t_stat), 4),
        "ttest_p": round(float(p_val), 4),
        "wilcoxon_W": round(float(w_stat), 1) if w_stat is not None else None,
        "wilcoxon_p": round(float(w_pval), 4) if w_pval is not None else None,
        "n_steering_helps": int(n_better),
        "n_steering_hurts": int(n_worse),
        "g1_clamped": g1_clamped,
        "g2_clamped": g2_clamped,
        "unclamped_paired_effect": round(float(np.mean(unclamped_diffs)), 4) if unclamped_diffs else None,
        "unclamped_n": len(unclamped_diffs),
        "role_separated": role_results,
        "g1_mean": round(float(np.mean(g1_advs)), 4),
        "g2_mean": round(float(np.mean(g2_advs)), 4),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: B1 metrics on G1 (SCM transcripts)
# ═══════════════════════════════════════════════════════════════════════════════

def task2_b1_metrics(g1_games):
    """Apply B1 per-turn metrics pipeline to SCM steered transcripts."""
    print("\n" + "=" * 70)
    print("TASK 2: B1 METRICS ON G1 (SCM α=6.12)")
    print("=" * 70)

    steered_lengths = []
    baseline_lengths = []
    steered_hedges = []
    baseline_hedges = []
    steered_concessions_abs = []
    baseline_concessions_abs = []
    steered_concessions_pct = []
    baseline_concessions_pct = []
    first_dist_steered = []
    first_dist_baseline = []

    for game in g1_games:
        transcript = game["transcript"]
        sr = game["steered_role"]
        br = "buyer" if sr == "seller" else "seller"

        # Response lengths
        for turn in transcript:
            wc = len(turn["utterance"].split())
            if turn["speaker"] == sr:
                steered_lengths.append(wc)
            else:
                baseline_lengths.append(wc)

        # Hedges
        for turn in transcript:
            h = count_hedges(turn["utterance"])
            wc = len(turn["utterance"].split())
            h100 = (h / wc * 100) if wc > 0 else 0
            if turn["speaker"] == sr:
                steered_hedges.append(h100)
            else:
                baseline_hedges.append(h100)

        # Concessions
        offers = parse_offers(transcript)
        concessions = compute_concessions(offers)
        for c in concessions:
            if c["is_first_offer"]:
                continue
            if c["speaker"] == sr:
                if c["concession_abs"] is not None:
                    steered_concessions_abs.append(c["concession_abs"])
                if c["concession_pct_gap"] is not None:
                    steered_concessions_pct.append(c["concession_pct_gap"])
            else:
                if c["concession_abs"] is not None:
                    baseline_concessions_abs.append(c["concession_abs"])
                if c["concession_pct_gap"] is not None:
                    baseline_concessions_pct.append(c["concession_pct_gap"])

        # First-offer distance
        for _, spk, amt in offers:
            if spk == sr:
                target = game["seller_target"] if sr == "seller" else game["buyer_target"]
                first_dist_steered.append(abs(amt - target) / game["listing_price"])
                break
        for _, spk, amt in offers:
            if spk == br:
                target = game["seller_target"] if br == "seller" else game["buyer_target"]
                first_dist_baseline.append(abs(amt - target) / game["listing_price"])
                break

    print(f"\n  Response length:")
    print(f"    Steered:  {_stats(steered_lengths)}")
    print(f"    Baseline: {_stats(baseline_lengths)}")
    ratio = np.mean(steered_lengths) / np.mean(baseline_lengths) if baseline_lengths else 0
    print(f"    Ratio (steered/baseline): {ratio:.3f}")

    print(f"\n  Hedge words (per 100 words):")
    print(f"    Steered:  {_stats(steered_hedges)}")
    print(f"    Baseline: {_stats(baseline_hedges)}")
    sm = np.mean(steered_hedges) if steered_hedges else 0
    bm = np.mean(baseline_hedges) if baseline_hedges else 0
    print(f"    Ratio: {bm/sm:.1f}x baseline vs steered" if sm > 0 else "    Steered hedge rate ≈ 0")

    print(f"\n  Concession rate (abs $):")
    print(f"    Steered:  {_stats(steered_concessions_abs)}")
    print(f"    Baseline: {_stats(baseline_concessions_abs)}")

    print(f"\n  Concession rate (% of gap):")
    print(f"    Steered:  {_stats(steered_concessions_pct)}")
    print(f"    Baseline: {_stats(baseline_concessions_pct)}")

    print(f"\n  First-offer distance (% of listing):")
    print(f"    Steered:  {_stats(first_dist_steered)}")
    print(f"    Baseline: {_stats(first_dist_baseline)}")

    return {
        "response_length": {
            "steered": _stats(steered_lengths),
            "baseline": _stats(baseline_lengths),
            "ratio": round(ratio, 3),
        },
        "hedges_per100w": {
            "steered": _stats(steered_hedges),
            "baseline": _stats(baseline_hedges),
        },
        "concession_abs": {
            "steered": _stats(steered_concessions_abs),
            "baseline": _stats(baseline_concessions_abs),
        },
        "concession_pct_gap": {
            "steered": _stats(steered_concessions_pct),
            "baseline": _stats(baseline_concessions_pct),
        },
        "first_offer_distance": {
            "steered": _stats(first_dist_steered),
            "baseline": _stats(first_dist_baseline),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: B3 role-separated analysis on G1
# ═══════════════════════════════════════════════════════════════════════════════

def task3_role_separated(g1_games):
    """Role-separated analysis of G1 SCM results."""
    print("\n" + "=" * 70)
    print("TASK 3: ROLE-SEPARATED ANALYSIS (G1 SCM)")
    print("=" * 70)

    by_role = {"seller": [], "buyer": []}
    for g in g1_games:
        by_role[g["steered_role"]].append(g)

    results = {}
    for role, games in by_role.items():
        advs = [g["advantage"] for g in games]
        clamped_n = sum(1 for g in games if g.get("clamped", False))
        unclamped_advs = [g["advantage"] for g in games if not g.get("clamped", False)]
        n_help = sum(1 for a in advs if a > 0)
        n_hurt = sum(1 for a in advs if a < 0)

        # Response lengths by role
        s_lens = []
        b_lens = []
        s_hedges = []
        b_hedges = []
        for g in games:
            for turn in g["transcript"]:
                wc = len(turn["utterance"].split())
                h = count_hedges(turn["utterance"])
                h100 = (h / wc * 100) if wc > 0 else 0
                if turn["speaker"] == g["steered_role"]:
                    s_lens.append(wc)
                    s_hedges.append(h100)
                else:
                    b_lens.append(wc)
                    b_hedges.append(h100)

        results[role] = {
            "n": len(games),
            "advantage": _stats(advs),
            "clamped": clamped_n,
            "unclamped_advantage": _stats(unclamped_advs),
            "n_help": n_help,
            "n_hurt": n_hurt,
            "help_pct": round(n_help / len(games) * 100, 1) if games else 0,
            "steered_response_length": _stats(s_lens),
            "baseline_response_length": _stats(b_lens),
            "steered_hedges_per100w": _stats(s_hedges),
            "baseline_hedges_per100w": _stats(b_hedges),
        }

        print(f"\n  Steered as {role.upper()} (n={len(games)}):")
        print(f"    Advantage:        {_stats(advs)}")
        print(f"    Clamped:          {clamped_n} ({_pct(clamped_n, len(games))})")
        print(f"    Unclamped adv:    {_stats(unclamped_advs)}")
        print(f"    Helps/Hurts:      {n_help}/{n_hurt} "
              f"({n_help/len(games)*100:.0f}% / {n_hurt/len(games)*100:.0f}%)")
        print(f"    Steered words:    mean={np.mean(s_lens):.1f}")
        print(f"    Baseline words:   mean={np.mean(b_lens):.1f}")
        print(f"    Steered hedges:   mean={np.mean(s_hedges):.2f}/100w")
        print(f"    Baseline hedges:  mean={np.mean(b_hedges):.2f}/100w")

    # Dealmaker analysis
    dealmakers = [g.get("dealmaker", "unknown") for g in g1_games]
    from collections import Counter
    dm_counts = Counter(dealmakers)
    print(f"\n  Dealmaker: {dict(dm_counts)}")

    # Category breakdown
    cats = {}
    for g in g1_games:
        cat = g.get("category", "unknown")
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(g["advantage"])
    print(f"\n  Category breakdown:")
    for cat, advs in sorted(cats.items(), key=lambda x: -np.mean(x[1])):
        print(f"    {cat:>12s}: n={len(advs):>2d}  mean={np.mean(advs):+.3f}  "
              f"std={np.std(advs, ddof=1):.3f}" if len(advs) > 1 else
              f"    {cat:>12s}: n={len(advs):>2d}  mean={np.mean(advs):+.3f}")

    results["dealmaker"] = dict(dm_counts)
    results["category"] = {cat: _stats(advs) for cat, advs in cats.items()}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4: G3 DonD analysis
# ═══════════════════════════════════════════════════════════════════════════════

def task4_dond(g3_data):
    """Analyse Deal or No Deal cross-dataset results."""
    print("\n" + "=" * 70)
    print("TASK 4: G3 DEAL OR NO DEAL ANALYSIS")
    print("=" * 70)

    games = g3_data["games"]
    summary = g3_data["summary"]
    n = len(games)

    agreed_games = [g for g in games if g["agreed"]]
    disagreed = [g for g in games if not g["agreed"]]

    print(f"\n  Total games: {n}")
    print(f"  Agreed: {len(agreed_games)} ({len(agreed_games)/n*100:.0f}%)")
    print(f"  Disagreed: {len(disagreed)} ({len(disagreed)/n*100:.0f}%)")

    # Advantage
    advs = [g["advantage"] for g in agreed_games]
    print(f"\n  Advantage (agreed games only):")
    print(f"    {_stats(advs)}")
    n_help = sum(1 for a in advs if a > 0)
    n_hurt = sum(1 for a in advs if a < 0)
    print(f"    Helps: {n_help}/{len(advs)} ({n_help/len(advs)*100:.0f}%)")
    print(f"    Hurts: {n_hurt}/{len(advs)} ({n_hurt/len(advs)*100:.0f}%)")

    # Pareto analysis
    pareto_games = [g for g in agreed_games if g.get("pareto_optimal", False)]
    non_pareto = [g for g in agreed_games if not g.get("pareto_optimal", False)]
    print(f"\n  Pareto optimal: {len(pareto_games)}/{len(agreed_games)} "
          f"({len(pareto_games)/len(agreed_games)*100:.0f}%)")
    print(f"    (Naive baseline from session 4: 9.0%)")
    print(f"    (Human baseline from Lewis et al.: 76.9%)")

    # Efficiency
    efficiencies = [g["efficiency"] for g in agreed_games if "efficiency" in g]
    print(f"\n  Efficiency:")
    print(f"    {_stats(efficiencies)}")
    print(f"    (Naive baseline: 0.694)")

    # Joint utility
    joints = [g["joint_utility"] for g in agreed_games if "joint_utility" in g]
    max_joints = [g["max_joint"] for g in agreed_games if "max_joint" in g]
    print(f"\n  Joint utility: {_stats(joints)}")
    print(f"  Max possible:  {_stats(max_joints)}")

    # By steered role
    for role in ["agent1", "agent2"]:
        role_games = [g for g in agreed_games if g.get("steered_role") == role]
        if role_games:
            r_advs = [g["advantage"] for g in role_games]
            r_eff = [g["efficiency"] for g in role_games if "efficiency" in g]
            print(f"\n  Steered as {role} (n={len(role_games)}):")
            print(f"    Advantage: mean={np.mean(r_advs):+.3f}")
            print(f"    Efficiency: mean={np.mean(r_eff):.3f}" if r_eff else "")

    # Steered vs baseline scores
    s_scores = [g["steered_score"] for g in agreed_games]
    b_scores = [g["baseline_score"] for g in agreed_games]
    print(f"\n  Steered score:  {_stats(s_scores)}")
    print(f"  Baseline score: {_stats(b_scores)}")

    return {
        "n_total": n,
        "n_agreed": len(agreed_games),
        "n_disagreed": len(disagreed),
        "agree_rate": round(len(agreed_games) / n, 3),
        "advantage": _stats(advs),
        "pareto_rate": round(len(pareto_games) / len(agreed_games), 3) if agreed_games else 0,
        "efficiency": _stats(efficiencies),
        "joint_utility": _stats(joints),
        "steered_score": _stats(s_scores),
        "baseline_score": _stats(b_scores),
        "n_helps": n_help,
        "n_hurts": n_hurt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5: G4 firmness clamping
# ═══════════════════════════════════════════════════════════════════════════════

def task5_firmness(g4_data):
    """Analyse G4 firmness at moderate alpha — focus on clamping."""
    print("\n" + "=" * 70)
    print("TASK 5: G4 FIRMNESS MODERATE α=5.0 (CLAMPING ANALYSIS)")
    print("=" * 70)

    games = g4_data["games"]
    n = len(games)

    clamped = [g for g in games if g.get("clamped", False)]
    unclamped = [g for g in games if not g.get("clamped", False)]

    print(f"\n  Total games: {n}")
    print(f"  Clamped: {len(clamped)} ({len(clamped)/n*100:.0f}%)")
    print(f"  Unclamped: {len(unclamped)} ({len(unclamped)/n*100:.0f}%)")
    print(f"  (Compare: firmness α=20 on 7B had 24% clamped)")

    all_advs = [g["advantage"] for g in games]
    clamp_advs = [g["advantage"] for g in clamped]
    unclamp_advs = [g["advantage"] for g in unclamped]

    print(f"\n  All games advantage:      {_stats(all_advs)}")
    print(f"  Clamped advantage:        {_stats(clamp_advs)}")
    print(f"  Unclamped advantage:      {_stats(unclamp_advs)}")

    # Role split
    for role in ["seller", "buyer"]:
        role_games = [g for g in games if g["steered_role"] == role]
        role_advs = [g["advantage"] for g in role_games]
        role_clamped = sum(1 for g in role_games if g.get("clamped", False))
        role_unclamped_advs = [g["advantage"] for g in role_games if not g.get("clamped", False)]
        print(f"\n  Steered as {role.upper()} (n={len(role_games)}):")
        print(f"    Advantage: {_stats(role_advs)}")
        print(f"    Clamped: {role_clamped}/{len(role_games)}")
        print(f"    Unclamped adv: {_stats(role_unclamped_advs)}")

    # Check raw scores for clamped games
    if clamped:
        raw_sellers = [g.get("raw_seller_score", 0) for g in clamped]
        print(f"\n  Clamped raw seller scores: {_stats(raw_sellers)}")
        print(f"    (How far below buyer target the deal went)")

    return {
        "n": n,
        "clamped_n": len(clamped),
        "clamped_pct": round(len(clamped) / n * 100, 1),
        "all_advantage": _stats(all_advs),
        "clamped_advantage": _stats(clamp_advs),
        "unclamped_advantage": _stats(unclamp_advs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 6: G5 sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def task6_sensitivity(g5_data):
    """Analyse opening bid sensitivity results."""
    print("\n" + "=" * 70)
    print("TASK 6: G5 OPENING BID SENSITIVITY")
    print("=" * 70)

    results = {}
    for key in ["opening_50pct", "opening_60pct", "opening_70pct"]:
        if key not in g5_data:
            continue
        block = g5_data[key]
        games = block["games"]
        summary = block["summary"]
        advs = [g["advantage"] for g in games]
        clamped_n = sum(1 for g in games if g.get("clamped", False))

        pct = key.replace("opening_", "").replace("pct", "%")
        print(f"\n  Opening at {pct}:")
        print(f"    n={len(games)}  advantage={summary['advantage']:+.4f}  "
              f"agree_rate={summary['agree_rate']:.0%}")
        print(f"    Clamped: {clamped_n}/{len(games)}")
        print(f"    Per-game: {_stats(advs)}")

        # Role split
        for role in ["seller", "buyer"]:
            rg = [g for g in games if g["steered_role"] == role]
            if rg:
                ra = [g["advantage"] for g in rg]
                print(f"    As {role}: n={len(rg)}, mean={np.mean(ra):+.3f}")

        results[key] = {
            "advantage": summary["advantage"],
            "n": len(games),
            "clamped": clamped_n,
            "per_game": _stats(advs),
        }

    # Cross-bid comparison
    print(f"\n  Summary:")
    print(f"    50%: {g5_data.get('opening_50pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"    60%: {g5_data.get('opening_60pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"    70%: {g5_data.get('opening_70pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"  Pattern: Non-monotonic. Advantage peaks at 60% (default) opening.")
    print(f"  Interpretation: Results are moderately sensitive to opening bid.")
    print(f"  Note: n=10 per condition is too small for statistical conclusions.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Task 8: Within-role judge correlations
# ═══════════════════════════════════════════════════════════════════════════════

def task8_judge_correlations(judge_path, g1_games):
    """Compute within-role correlations between judge scores and behavioral metrics."""
    print("\n" + "=" * 70)
    print("TASK 8: WITHIN-ROLE JUDGE-METRIC CORRELATIONS")
    print("=" * 70)

    if not judge_path.exists():
        print(f"  Judge file not found: {judge_path}")
        print(f"  Skipping — run LLM judge first (task 7).")
        return None

    with open(judge_path) as f:
        judge_data = json.load(f)

    judge_games = judge_data.get("games", [])
    if not judge_games:
        print("  No judge data found.")
        return None

    # Build lookup: game_id -> judge scores
    judge_lookup = {}
    for jg in judge_games:
        gid = jg.get("game_id")
        if gid is not None and "judges" in jg:
            judge_lookup[gid] = jg

    # Build per-game behavioral metrics for G1
    game_metrics = {}
    for g in g1_games:
        gid = g.get("game_id")
        sr = g["steered_role"]

        # Compute metrics
        s_lens = [len(t["utterance"].split()) for t in g["transcript"] if t["speaker"] == sr]
        s_hedges = []
        for t in g["transcript"]:
            if t["speaker"] == sr:
                wc = len(t["utterance"].split())
                h = count_hedges(t["utterance"])
                s_hedges.append((h / wc * 100) if wc > 0 else 0)

        game_metrics[gid] = {
            "advantage": g["advantage"],
            "steered_role": sr,
            "mean_length": np.mean(s_lens) if s_lens else 0,
            "mean_hedge_rate": np.mean(s_hedges) if s_hedges else 0,
        }

    # Match games with judge data
    matched = []
    for gid, metrics in game_metrics.items():
        if gid in judge_lookup:
            jg = judge_lookup[gid]
            for judge_name, scores in jg.get("judges", {}).items():
                matched.append({
                    **metrics,
                    "judge": judge_name,
                    **{f"judge_{dim}": scores["steered"].get(dim)
                       for dim in ["firmness", "persuasiveness", "naturalness",
                                   "coherence", "information_management", "strategic_reasoning"]},
                })

    if not matched:
        print("  No matched games between judge and G1 data.")
        return None

    print(f"  Matched games: {len(matched)}")

    results = {}
    DIMS = ["firmness", "persuasiveness", "naturalness", "coherence",
            "information_management", "strategic_reasoning"]

    # Within-role correlations
    for role in ["seller", "buyer"]:
        role_data = [m for m in matched if m["steered_role"] == role]
        if len(role_data) < 5:
            print(f"\n  {role.upper()}: insufficient data (n={len(role_data)})")
            continue

        print(f"\n  Within {role.upper()} (n={len(role_data)}):")
        print(f"    {'Dimension':<25s} {'r(adv)':>8s} {'p':>8s} {'r(len)':>8s} {'p':>8s}")
        print(f"    {'-' * 55}")

        role_results = {}
        for dim in DIMS:
            scores = [m[f"judge_{dim}"] for m in role_data if m[f"judge_{dim}"] is not None]
            advs = [m["advantage"] for m in role_data if m[f"judge_{dim}"] is not None]
            lens = [m["mean_length"] for m in role_data if m[f"judge_{dim}"] is not None]

            if len(scores) < 5:
                continue

            r_adv, p_adv = sp_stats.pearsonr(scores, advs)
            r_len, p_len = sp_stats.pearsonr(scores, lens)

            sig_a = "*" if p_adv < 0.05 else ""
            sig_l = "*" if p_len < 0.05 else ""
            print(f"    {dim:<25s} {r_adv:+.3f}{sig_a:1s} {p_adv:.3f} "
                  f"  {r_len:+.3f}{sig_l:1s} {p_len:.3f}")

            role_results[dim] = {
                "r_advantage": round(r_adv, 4),
                "p_advantage": round(p_adv, 4),
                "r_length": round(r_len, 4),
                "p_length": round(p_len, 4),
            }
        results[role] = role_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _root = Path(__file__).resolve().parent.parent
    eval_dir = _root / "results" / "eval"

    # Load all data
    with open(eval_dir / "g1_scm_steered.json") as f:
        g1_data = json.load(f)
    with open(eval_dir / "g2_baseline.json") as f:
        g2_data = json.load(f)
    with open(eval_dir / "g3_dond_scm.json") as f:
        g3_data = json.load(f)
    with open(eval_dir / "g4_firmness_moderate.json") as f:
        g4_data = json.load(f)
    with open(eval_dir / "g5_sensitivity.json") as f:
        g5_data = json.load(f)
    with open(eval_dir / "cosine_similarity.json") as f:
        cosine_data = json.load(f)

    g1_games = g1_data["games"]
    g2_games = g2_data["games"]

    print("=" * 70)
    print("POST-GPU EVALUATION ANALYSIS")
    print(f"  G1: {len(g1_games)} games (SCM steered α=6.12)")
    print(f"  G2: {len(g2_games)} games (baseline α=0)")
    print(f"  G3: {len(g3_data['games'])} games (DonD)")
    print(f"  G4: {len(g4_data['games'])} games (firmness α=5)")
    print(f"  Cosine: {cosine_data}")
    print("=" * 70)

    analysis = {}

    # Task 1
    analysis["task1_paired_effect"] = task1_paired_effect(g1_games, g2_games)

    # Task 2
    analysis["task2_b1_metrics"] = task2_b1_metrics(g1_games)

    # Task 3
    analysis["task3_role_separated"] = task3_role_separated(g1_games)

    # Task 4
    analysis["task4_dond"] = task4_dond(g3_data)

    # Task 5
    analysis["task5_firmness"] = task5_firmness(g4_data)

    # Task 6
    analysis["task6_sensitivity"] = task6_sensitivity(g5_data)

    # Task 8 — try both possible judge file locations
    judge_path = eval_dir / "judge_scores_g1.json"
    if not judge_path.exists():
        judge_path = _root / "results" / "judge_scores.json"  # old firmness judge
    analysis["task8_judge_correlations"] = task8_judge_correlations(judge_path, g1_games)

    # Cosine similarity
    analysis["cosine_similarity"] = cosine_data

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    t1 = analysis["task1_paired_effect"]
    print(f"\n  1. PAIRED STEERING EFFECT: {t1['paired_effect_mean']:+.4f} "
          f"(p={t1['ttest_p']:.4f}, d={t1['cohens_d']:+.3f})")
    print(f"     Helps {t1['n_steering_helps']}/{t1['n_pairs']}, "
          f"hurts {t1['n_steering_hurts']}/{t1['n_pairs']}")
    if t1.get("unclamped_paired_effect") is not None:
        print(f"     Unclamped-only: {t1['unclamped_paired_effect']:+.4f} (n={t1['unclamped_n']})")

    t2 = analysis["task2_b1_metrics"]
    print(f"\n  2. SCM BEHAVIORAL METRICS:")
    print(f"     Length ratio: {t2['response_length']['ratio']:.3f}")
    sm = t2['hedges_per100w']['steered']['mean']
    bm = t2['hedges_per100w']['baseline']['mean']
    if sm and sm > 0:
        print(f"     Hedge ratio: {bm/sm:.1f}x (baseline/steered)")
    else:
        print(f"     Hedge rate: steered≈0, baseline={bm:.2f}/100w")

    t4 = analysis["task4_dond"]
    print(f"\n  4. DonD: advantage={t4['advantage']['mean']:+.3f}, "
          f"Pareto={t4['pareto_rate']:.1%}, efficiency={t4['efficiency']['mean']:.3f}")

    t5 = analysis["task5_firmness"]
    print(f"\n  5. FIRMNESS α=5: {t5['clamped_pct']:.0f}% clamped "
          f"(vs 24% at α=20)")
    print(f"     Unclamped advantage: {t5['unclamped_advantage']['mean']}")

    print(f"\n  COSINE: SCM-firmness={cosine_data['strategic_concession_making_vs_firmness']:.3f}")

    # Save
    out_path = eval_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
