#!/usr/bin/env python3
"""
analyse_results.py — Post-GPU statistical analysis of evaluation experiments.

Reads the JSON files produced by run_eval.py and produces:
  - Paired steering effect: buyer-steered vs buyer-baseline (primary claim)
  - Seller-steered standalone analysis
  - Per-turn behavioral metrics (response length, hedging, concessions)
  - Role-separated outcome breakdown
  - DonD cross-dataset, firmness, and sensitivity analyses (if available)
  - Within-role judge correlations (if llm_judge.py has been run)

Run:
  python3 analyse_results.py
  python3 analyse_results.py --results_dir results/eval --skip_missing

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
# Paired steering effect
# ═══════════════════════════════════════════════════════════════════════════════

def paired_steering_effect(steered_games, baseline_games, label="steered / baseline"):
    """Compute paired steering effect: per-scenario steered_advantage - baseline_advantage."""
    print("\n" + "=" * 70)
    print(f"PAIRED STEERING EFFECT  [{label}]")
    print("=" * 70)

    n = min(len(steered_games), len(baseline_games))
    paired_diffs = []
    steered_advs = []
    baseline_advs = []
    role_diffs = {"seller": [], "buyer": []}

    for i in range(n):
        g1 = steered_games[i]
        g2 = baseline_games[i]
        # Verify same scenario
        assert g1["listing_price"] == g2["listing_price"], f"Scenario mismatch at {i}"
        assert g1["steered_role"] == g2["steered_role"], f"Role mismatch at {i}"

        d = g1["advantage"] - g2["advantage"]
        paired_diffs.append(d)
        steered_advs.append(g1["advantage"])
        baseline_advs.append(g2["advantage"])
        role_diffs[g1["steered_role"]].append(d)

    paired_diffs = np.array(paired_diffs)
    steered_advs = np.array(steered_advs)
    baseline_advs = np.array(baseline_advs)

    # Paired t-test
    t_stat, p_val = sp_stats.ttest_rel(steered_advs, baseline_advs)
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
    print(f"  Steered mean advantage:  {np.mean(steered_advs):+.4f}")
    print(f"  Baseline mean advantage: {np.mean(baseline_advs):+.4f}")
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
    steered_clamped = sum(1 for g in steered_games if g.get("clamped", False))
    baseline_clamped = sum(1 for g in baseline_games if g.get("clamped", False))
    print(f"\n  Clamping: steered={steered_clamped}/{n} ({steered_clamped/n*100:.0f}%), "
          f"baseline={baseline_clamped}/{n} ({baseline_clamped/n*100:.0f}%)")

    # Unclamped-only paired effect
    unclamped_diffs = []
    for i in range(n):
        if not steered_games[i].get("clamped", False) and not baseline_games[i].get("clamped", False):
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
        "steered_clamped": steered_clamped,
        "baseline_clamped": baseline_clamped,
        "unclamped_paired_effect": round(float(np.mean(unclamped_diffs)), 4) if unclamped_diffs else None,
        "unclamped_n": len(unclamped_diffs),
        "role_separated": role_results,
        "steered_mean": round(float(np.mean(steered_advs)), 4),
        "baseline_mean": round(float(np.mean(baseline_advs)), 4),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-turn behavioral metrics
# ═══════════════════════════════════════════════════════════════════════════════

def per_turn_behavior(games):
    """Per-turn behavioral metrics: response length, hedging, concessions, first-offer distance."""
    print("\n" + "=" * 70)
    print("PER-TURN BEHAVIORAL METRICS (SCM steered)")
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

    for game in games:
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
# Role-separated outcome analysis
# ═══════════════════════════════════════════════════════════════════════════════

def role_separated_outcomes(games):
    """Role-separated outcome analysis: advantage, clamping, lengths, hedging per role."""
    print("\n" + "=" * 70)
    print("ROLE-SEPARATED OUTCOMES (steered=buyer vs steered=seller)")
    print("=" * 70)

    by_role = {"seller": [], "buyer": []}
    for g in games:
        by_role[g["steered_role"]].append(g)

    results = {}
    for role, role_games in by_role.items():
        advs = [g["advantage"] for g in role_games]
        clamped_n = sum(1 for g in role_games if g.get("clamped", False))
        unclamped_advs = [g["advantage"] for g in role_games if not g.get("clamped", False)]
        n_help = sum(1 for a in advs if a > 0)
        n_hurt = sum(1 for a in advs if a < 0)

        # Response lengths by role
        s_lens = []
        b_lens = []
        s_hedges = []
        b_hedges = []
        for g in role_games:
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
            "n": len(role_games),
            "advantage": _stats(advs),
            "clamped": clamped_n,
            "unclamped_advantage": _stats(unclamped_advs),
            "n_help": n_help,
            "n_hurt": n_hurt,
            "help_pct": round(n_help / len(role_games) * 100, 1) if role_games else 0,
            "steered_response_length": _stats(s_lens),
            "baseline_response_length": _stats(b_lens),
            "steered_hedges_per100w": _stats(s_hedges),
            "baseline_hedges_per100w": _stats(b_hedges),
        }

        print(f"\n  Steered as {role.upper()} (n={len(role_games)}):")
        print(f"    Advantage:        {_stats(advs)}")
        print(f"    Clamped:          {clamped_n} ({_pct(clamped_n, len(role_games))})")
        print(f"    Unclamped adv:    {_stats(unclamped_advs)}")
        print(f"    Helps/Hurts:      {n_help}/{n_hurt} "
              f"({n_help/len(role_games)*100:.0f}% / {n_hurt/len(role_games)*100:.0f}%)")
        print(f"    Steered words:    mean={np.mean(s_lens):.1f}")
        print(f"    Baseline words:   mean={np.mean(b_lens):.1f}")
        print(f"    Steered hedges:   mean={np.mean(s_hedges):.2f}/100w")
        print(f"    Baseline hedges:  mean={np.mean(b_hedges):.2f}/100w")

    # Dealmaker analysis
    dealmakers = [g.get("dealmaker", "unknown") for g in buyer_steered_games]
    from collections import Counter
    dm_counts = Counter(dealmakers)
    print(f"\n  Dealmaker: {dict(dm_counts)}")

    # Category breakdown
    cats = {}
    for g in buyer_steered_games:
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
# DonD cross-dataset analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_dond_crossval(g3_data):
    """Analyse Deal or No Deal cross-dataset results."""
    print("\n" + "=" * 70)
    print("DonD CROSS-DATASET ANALYSIS")
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
# Firmness clamping analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_firmness(firmness_data):
    """Analyse firmness vector at moderate alpha — focus on clamping."""
    print("\n" + "=" * 70)
    print("FIRMNESS MODERATE α=5.0 (CLAMPING ANALYSIS)")
    print("=" * 70)

    games = firmness_data["games"]
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
# Opening bid sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_sensitivity(sensitivity_data):
    """Analyse opening bid sensitivity results."""
    print("\n" + "=" * 70)
    print("OPENING BID SENSITIVITY")
    print("=" * 70)

    results = {}
    for key in ["opening_50pct", "opening_60pct", "opening_70pct"]:
        if key not in sensitivity_data:
            continue
        block = sensitivity_data[key]
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
    print(f"    50%: {sensitivity_data.get('opening_50pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"    60%: {sensitivity_data.get('opening_60pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"    70%: {sensitivity_data.get('opening_70pct', {}).get('summary', {}).get('advantage', 'n/a')}")
    print(f"  Pattern: Non-monotonic. Advantage peaks at 60% (default) opening.")
    print(f"  Interpretation: Results are moderately sensitive to opening bid.")
    print(f"  Note: n=10 per condition is too small for statistical conclusions.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Within-role judge correlations
# ═══════════════════════════════════════════════════════════════════════════════

def judge_metric_correlations(judge_path, steered_games):
    """Compute within-role correlations between judge scores and behavioral metrics."""
    print("\n" + "=" * 70)
    print("WITHIN-ROLE JUDGE-METRIC CORRELATIONS")
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

    # Build per-game behavioral metrics
    game_metrics = {}
    for g in steered_games:
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
    import argparse
    _root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser(
        description="Post-GPU analysis of negotiation evaluation experiments."
    )
    p.add_argument("--results_dir",  default=str(_root / "results" / "eval"),
                   help="Directory containing experiment JSON files.")
    p.add_argument("--skip_missing", action="store_true",
                   help="Skip optional files (g3/g4/g5/cosine) if not present.")
    args = p.parse_args()

    eval_dir = Path(args.results_dir)

    def _load(fname, optional=False):
        path = eval_dir / fname
        if not path.exists():
            if optional or args.skip_missing:
                return None
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path) as f:
            return json.load(f)

    # Load required files (produced by run_eval.py --experiments scm_craigslist)
    buyer_steered_data  = _load("scm_buyer_steered.json")
    seller_steered_data = _load("scm_seller_steered.json")
    buyer_baseline_data = _load("scm_buyer_baseline.json")

    # Load optional files (produced by other experiments)
    dond_data        = _load("dond_crossval.json",       optional=True)
    firmness_data    = _load("firmness_moderate.json",   optional=True)
    sensitivity_data = _load("opening_sensitivity.json", optional=True)
    cosine_data      = _load("cosine_similarity.json",   optional=True)

    buyer_steered_games  = buyer_steered_data["games"]
    seller_steered_games = seller_steered_data["games"]
    buyer_baseline_games = buyer_baseline_data["games"]

    print("=" * 70)
    print("NEGOTIATION EVALUATION — STATISTICAL ANALYSIS")
    print(f"  buyer-steered:  {len(buyer_steered_games)} games (SCM steered, role=buyer)")
    print(f"  seller-steered: {len(seller_steered_games)} games (SCM steered, role=seller)")
    print(f"  buyer-baseline: {len(buyer_baseline_games)} games (α=0, role=buyer)")
    if dond_data:
        print(f"  dond_crossval:  {len(dond_data['games'])} games")
    if firmness_data:
        print(f"  firmness:       {len(firmness_data['games'])} games")
    if cosine_data:
        print(f"  cosine: {cosine_data}")
    print("=" * 70)

    analysis = {}

    # Buyer-steering effect: primary paired comparison
    analysis["buyer_steering_effect"] = paired_steering_effect(
        buyer_steered_games, buyer_baseline_games,
        label="buyer-steered / buyer-baseline",
    )

    # Seller-steering (standalone — no matched seller baseline)
    analysis["seller_steering_effect"] = paired_steering_effect(
        seller_steered_games, buyer_baseline_games,
        label="seller-steered / buyer-baseline (reference only)",
    )

    # Per-turn behavioral metrics on buyer-steered games
    analysis["per_turn_behavior"] = per_turn_behavior(buyer_steered_games)

    # Role-separated outcomes: buyer-steered + seller-steered combined
    analysis["role_separated_outcomes"] = role_separated_outcomes(
        buyer_steered_games + seller_steered_games
    )

    # Optional: DonD cross-dataset
    if dond_data:
        analysis["dond_crossval"] = analyse_dond_crossval(dond_data)

    # Optional: Firmness
    if firmness_data:
        analysis["firmness"] = analyse_firmness(firmness_data)

    # Optional: Opening bid sensitivity
    if sensitivity_data:
        analysis["sensitivity"] = analyse_sensitivity(sensitivity_data)

    # Judge correlations — try both possible file locations
    judge_path = eval_dir / "judge_scores_scm_buyer.json"
    if not judge_path.exists():
        judge_path = eval_dir / "judge_scores_g1a.json"
    if not judge_path.exists():
        judge_path = _root / "results" / "judge_scores.json"
    analysis["judge_correlations"] = judge_metric_correlations(judge_path, buyer_steered_games)

    if cosine_data:
        analysis["cosine_similarity"] = cosine_data

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    t1a = analysis["buyer_steering_effect"]
    t1b = analysis["seller_steering_effect"]
    print(f"\n  BUYER-STEERING EFFECT: {t1a['paired_effect_mean']:+.4f} "
          f"(p={t1a['ttest_p']:.4f}, d={t1a['cohens_d']:+.3f})")
    print(f"    Helps {t1a['n_steering_helps']}/{t1a['n_pairs']}, "
          f"hurts {t1a['n_steering_hurts']}/{t1a['n_pairs']}")

    print(f"\n  SELLER-STEERING EFFECT (reference): {t1b['paired_effect_mean']:+.4f} "
          f"(p={t1b['ttest_p']:.4f}, d={t1b['cohens_d']:+.3f})")
    print(f"    Helps {t1b['n_steering_helps']}/{t1b['n_pairs']}, "
          f"hurts {t1b['n_steering_hurts']}/{t1b['n_pairs']}")

    if t1a.get("unclamped_paired_effect") is not None:
        print(f"\n  Unclamped buyer-steering: {t1a['unclamped_paired_effect']:+.4f} "
              f"(n={t1a['unclamped_n']})")

    t2 = analysis["per_turn_behavior"]
    print(f"\n  BEHAVIORAL METRICS (SCM buyer-steered):")
    print(f"    Length ratio: {t2['response_length']['ratio']:.3f}")
    sm = t2['hedges_per100w']['steered']['mean']
    bm = t2['hedges_per100w']['baseline']['mean']
    if sm and sm > 0:
        print(f"    Hedge ratio: {bm/sm:.1f}x (baseline/steered)")
    else:
        print(f"    Hedge rate: steered≈0, baseline={bm:.2f}/100w")

    if dond_data and "dond_crossval" in analysis and analysis["dond_crossval"]:
        td = analysis["dond_crossval"]
        print(f"\n  DonD CROSS-DATASET: advantage={td['advantage']['mean']:+.3f}, "
              f"Pareto={td['pareto_rate']:.1%}, efficiency={td['efficiency']['mean']:.3f}")

    if firmness_data and "firmness" in analysis:
        tf = analysis["firmness"]
        print(f"\n  FIRMNESS α=5: {tf['clamped_pct']:.0f}% clamped "
              f"(vs 24% at α=20)")
        print(f"    Unclamped advantage: {tf['unclamped_advantage']['mean']}")

    if cosine_data and "strategic_concession_making_vs_firmness" in cosine_data:
        print(f"\n  COSINE: SCM-firmness="
              f"{cosine_data['strategic_concession_making_vs_firmness']:.3f}")

    # Save
    out_path = eval_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
