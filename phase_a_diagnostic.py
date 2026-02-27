#!/usr/bin/env python3
"""
Phase A Diagnostic — Foundation Fixes for P4 Evaluation

A1: CraigslistBargains data audit (overlapping targets, bad prices, category distribution)
A2: Existing results diagnostic (role breakdown, response length, confound checks)
A3: score_deal() analysis (impact of overlapping targets on scores)

Run: python phase_a_diagnostic.py
No GPU needed — pure data analysis on existing files.
"""

import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

# ---------------------------------------------------------------------------
# A1: CraigslistBargains Data Audit
# ---------------------------------------------------------------------------

RAW_URLS = {
    "train": "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "validation": "https://worksheets.codalab.org/rest/bundles/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
}


def fetch_json(url: str) -> list:
    print(f"  Downloading {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        chunks = []
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))


def parse_all_scenarios(raw: list) -> list:
    """Parse ALL scenarios without filtering — we want to see what gets dropped."""
    scenarios = []
    parse_failures = 0
    no_kbs = 0
    no_roles = 0
    no_targets = 0

    for i, entry in enumerate(raw):
        try:
            kbs = entry.get("scenario", {}).get("kbs", [])
            if len(kbs) < 2:
                no_kbs += 1
                continue

            p0 = kbs[0].get("personal", {})
            p1 = kbs[1].get("personal", {})

            if "seller" in str(p0.get("Role", "")).lower():
                seller_p, seller_kb = p0, kbs[0]
                buyer_p = p1
            elif "seller" in str(p1.get("Role", "")).lower():
                seller_p, seller_kb = p1, kbs[1]
                buyer_p = p0
            else:
                no_roles += 1
                continue

            seller_target_raw = seller_p.get("Target")
            buyer_target_raw = buyer_p.get("Target")
            if seller_target_raw is None or buyer_target_raw is None:
                no_targets += 1
                continue

            seller_target = float(seller_target_raw)
            buyer_target = float(buyer_target_raw)

            item = seller_kb.get("item", {})

            def _unwrap(v):
                if isinstance(v, list):
                    return v[0] if v else ""
                return v

            listing_price = float(_unwrap(item.get("Price", -1)) or -1)
            title = str(_unwrap(item.get("Title", ""))).strip()
            description = str(_unwrap(item.get("Description", ""))).strip()
            category = str(_unwrap(item.get("Category", ""))).strip()

            scenarios.append({
                "idx": i,
                "title": title,
                "description": description,
                "category": category,
                "listing_price": listing_price,
                "seller_target": seller_target,
                "buyer_target": buyer_target,
            })

        except (KeyError, ValueError, TypeError):
            parse_failures += 1
            continue

    return scenarios, {
        "total_entries": len(raw),
        "parse_failures": parse_failures,
        "no_kbs": no_kbs,
        "no_roles": no_roles,
        "no_targets": no_targets,
        "parsed_ok": len(scenarios),
    }


def audit_dataset(split: str):
    print(f"\n{'='*70}")
    print(f"A1: DATA AUDIT — CraigslistBargains '{split}' split")
    print(f"{'='*70}")

    raw = fetch_json(RAW_URLS[split])
    scenarios, parse_stats = parse_all_scenarios(raw)

    print(f"\n--- Parse Statistics ---")
    for k, v in parse_stats.items():
        print(f"  {k:20s}: {v}")

    # Classify scenarios
    overlapping = []  # seller_target < buyer_target
    zero_negative_price = []
    zero_negative_target = []
    no_title = []
    valid = []

    for s in scenarios:
        issues = []
        if s["listing_price"] <= 0:
            zero_negative_price.append(s)
            issues.append("bad_price")
        if s["seller_target"] <= 0 or s["buyer_target"] <= 0:
            zero_negative_target.append(s)
            issues.append("bad_target")
        if s["seller_target"] < s["buyer_target"]:
            overlapping.append(s)
            issues.append("overlapping")
        if not s["title"]:
            no_title.append(s)
            issues.append("no_title")
        if not issues:
            valid.append(s)

    n = len(scenarios)
    print(f"\n--- Scenario Quality ---")
    print(f"  Total parsed scenarios          : {n}")
    print(f"  Overlapping targets (S < B)     : {len(overlapping)} ({100*len(overlapping)/n:.1f}%)")
    print(f"  Zero/negative listing price     : {len(zero_negative_price)} ({100*len(zero_negative_price)/n:.1f}%)")
    print(f"  Zero/negative target            : {len(zero_negative_target)} ({100*len(zero_negative_target)/n:.1f}%)")
    print(f"  Missing title                   : {len(no_title)} ({100*len(no_title)/n:.1f}%)")
    print(f"  Clean (no issues)               : {len(valid)} ({100*len(valid)/n:.1f}%)")

    # Overlapping target details
    if overlapping:
        print(f"\n--- Overlapping Target Details (seller_target < buyer_target) ---")
        gaps = [s["buyer_target"] - s["seller_target"] for s in overlapping]
        print(f"  Count                           : {len(overlapping)}")
        print(f"  Overlap gap: mean=${mean(gaps):.0f}, median=${median(gaps):.0f}, max=${max(gaps):.0f}")
        cats = Counter(s["category"] for s in overlapping)
        print(f"  By category: {dict(cats)}")

        # What fraction would apply_steering.py's loader keep?
        kept_by_loader = [s for s in overlapping
                          if s["listing_price"] > 0 and s["seller_target"] > 0
                          and s["buyer_target"] > 0 and s["title"]]
        print(f"  Of these, {len(kept_by_loader)} would PASS apply_steering.py's filter")
        print(f"  → These get scored as (0.5, 0.5) by score_deal() — silent draws")

    # Category distribution (clean scenarios only)
    print(f"\n--- Category Distribution (clean scenarios) ---")
    cats = Counter(s["category"] for s in valid)
    for cat, count in cats.most_common():
        print(f"  {cat:30s}: {count:5d} ({100*count/len(valid):.1f}%)")

    # Price distribution by category
    print(f"\n--- Listing Price Stats by Category (clean scenarios) ---")
    by_cat = defaultdict(list)
    for s in valid:
        by_cat[s["category"]].append(s["listing_price"])
    print(f"  {'Category':30s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
    for cat in sorted(by_cat.keys()):
        prices = by_cat[cat]
        print(f"  {cat:30s} {len(prices):5d} ${mean(prices):7.0f} ${median(prices):7.0f} "
              f"${min(prices):7.0f} ${max(prices):7.0f}")

    # Target spread analysis (clean scenarios)
    print(f"\n--- Target Spread Stats (clean scenarios) ---")
    spreads = [s["seller_target"] - s["buyer_target"] for s in valid]
    pct_spreads = [(s["seller_target"] - s["buyer_target"]) / s["listing_price"] * 100
                   for s in valid if s["listing_price"] > 0]
    print(f"  Absolute spread (S-B): mean=${mean(spreads):.0f}, median=${median(spreads):.0f}, "
          f"std=${stdev(spreads):.0f}, min=${min(spreads):.0f}, max=${max(spreads):.0f}")
    print(f"  As % of listing price: mean={mean(pct_spreads):.1f}%, median={median(pct_spreads):.1f}%, "
          f"std={stdev(pct_spreads):.1f}%")

    # Seller target vs listing price
    print(f"\n--- Seller Target vs Listing Price (clean scenarios) ---")
    ratios = [s["seller_target"] / s["listing_price"] for s in valid if s["listing_price"] > 0]
    print(f"  seller_target / listing_price: mean={mean(ratios):.3f}, median={median(ratios):.3f}, "
          f"min={min(ratios):.3f}, max={max(ratios):.3f}")

    # Buyer target vs listing price
    ratios_b = [s["buyer_target"] / s["listing_price"] for s in valid if s["listing_price"] > 0]
    print(f"  buyer_target  / listing_price: mean={mean(ratios_b):.3f}, median={median(ratios_b):.3f}, "
          f"min={min(ratios_b):.3f}, max={max(ratios_b):.3f}")

    return scenarios, valid, overlapping


# ---------------------------------------------------------------------------
# A2: Existing Results Diagnostic
# ---------------------------------------------------------------------------

def diagnose_results(results_path: str):
    print(f"\n{'='*70}")
    print(f"A2: RESULTS DIAGNOSTIC — {results_path}")
    print(f"{'='*70}")

    with open(results_path) as f:
        data = json.load(f)

    games = data["games"]
    agreed = [g for g in games if g["agreed"]]
    n = len(games)
    na = len(agreed)

    print(f"\n--- Basic Stats ---")
    print(f"  Total games: {n}, Agreed: {na} ({100*na/n:.1f}%)")

    if not agreed:
        print("  No agreed games to analyze.")
        return

    # Role breakdown
    seller_games = [g for g in agreed if g["steered_role"] == "seller"]
    buyer_games = [g for g in agreed if g["steered_role"] == "buyer"]

    print(f"\n--- Role Breakdown ---")
    if seller_games:
        adv_s = [g["advantage"] for g in seller_games]
        print(f"  Steered as SELLER: {len(seller_games)} games")
        print(f"    Advantage: mean={mean(adv_s):+.4f}, median={median(adv_s):+.4f}, "
              f"std={stdev(adv_s):.4f}")
        print(f"    Range: [{min(adv_s):+.4f}, {max(adv_s):+.4f}]")
        wins = sum(1 for a in adv_s if a > 0)
        print(f"    Win rate: {wins}/{len(adv_s)} ({100*wins/len(adv_s):.1f}%)")

    if buyer_games:
        adv_b = [g["advantage"] for g in buyer_games]
        print(f"  Steered as BUYER: {len(buyer_games)} games")
        print(f"    Advantage: mean={mean(adv_b):+.4f}, median={median(adv_b):+.4f}, "
              f"std={stdev(adv_b):.4f}")
        print(f"    Range: [{min(adv_b):+.4f}, {max(adv_b):+.4f}]")
        wins = sum(1 for a in adv_b if a > 0)
        print(f"    Win rate: {wins}/{len(adv_b)} ({100*wins/len(adv_b):.1f}%)")

    # Overlapping targets in results
    print(f"\n--- Overlapping Targets in Results ---")
    overlap_games = [g for g in agreed if g["seller_target"] < g["buyer_target"]]
    equal_games = [g for g in agreed if g["seller_target"] == g["buyer_target"]]
    print(f"  Overlapping (S < B): {len(overlap_games)}")
    print(f"  Equal (S == B)     : {len(equal_games)}")
    if overlap_games:
        for g in overlap_games:
            print(f"    Game {g['game_id']}: seller_target=${g['seller_target']:.0f}, "
                  f"buyer_target=${g['buyer_target']:.0f}, "
                  f"agreed=${g['agreed_price']:.0f}, "
                  f"scores=({g['seller_score']}, {g['buyer_score']})")

    # Response length analysis (token proxy: word count)
    print(f"\n--- Response Length Analysis (word count per utterance) ---")
    steered_lengths = []
    baseline_lengths = []
    for g in agreed:
        sr = g["steered_role"]
        for turn in g["transcript"]:
            wc = len(turn["utterance"].split())
            if turn["speaker"] == sr:
                steered_lengths.append(wc)
            else:
                baseline_lengths.append(wc)

    if steered_lengths and baseline_lengths:
        print(f"  Steered agent:  mean={mean(steered_lengths):.1f} words, "
              f"median={median(steered_lengths):.0f}, std={stdev(steered_lengths):.1f}")
        print(f"  Baseline agent: mean={mean(baseline_lengths):.1f} words, "
              f"median={median(baseline_lengths):.0f}, std={stdev(baseline_lengths):.1f}")
        print(f"  Ratio (steered/baseline): {mean(steered_lengths)/mean(baseline_lengths):.3f}")

    # Per-game response length vs advantage
    print(f"\n--- Response Length vs Advantage (per game) ---")
    game_steered_len = []
    game_baseline_len = []
    game_adv = []
    for g in agreed:
        sr = g["steered_role"]
        s_wc = [len(t["utterance"].split()) for t in g["transcript"] if t["speaker"] == sr]
        b_wc = [len(t["utterance"].split()) for t in g["transcript"] if t["speaker"] != sr]
        if s_wc and b_wc:
            game_steered_len.append(mean(s_wc))
            game_baseline_len.append(mean(b_wc))
            game_adv.append(g["advantage"])

    if game_adv:
        # Simple correlation
        n_g = len(game_adv)
        mean_adv = mean(game_adv)
        mean_slen = mean(game_steered_len)
        mean_ratio = mean(sl / bl for sl, bl in zip(game_steered_len, game_baseline_len))

        # Pearson correlation (steered length vs advantage)
        cov = sum((game_steered_len[i] - mean_slen) * (game_adv[i] - mean_adv) for i in range(n_g)) / n_g
        std_s = stdev(game_steered_len) if n_g > 1 else 1
        std_a = stdev(game_adv) if n_g > 1 else 1
        corr = cov / (std_s * std_a) if std_s > 0 and std_a > 0 else 0
        print(f"  Pearson corr(steered_word_count, advantage) = {corr:.4f}")
        print(f"  Mean steered/baseline length ratio = {mean_ratio:.3f}")

    # Category breakdown
    print(f"\n--- Advantage by Category ---")
    by_cat = defaultdict(list)
    for g in agreed:
        by_cat[g["category"]].append(g["advantage"])
    print(f"  {'Category':30s} {'N':>4s} {'Mean Adv':>10s} {'Std':>8s} {'Win%':>6s}")
    for cat in sorted(by_cat.keys()):
        advs = by_cat[cat]
        wins = sum(1 for a in advs if a > 0)
        s = stdev(advs) if len(advs) > 1 else 0
        print(f"  {cat:30s} {len(advs):4d} {mean(advs):+10.4f} {s:8.4f} {100*wins/len(advs):5.1f}%")

    # Price range breakdown
    print(f"\n--- Advantage by Listing Price Bucket ---")
    buckets = {"<$100": [], "$100-500": [], "$500-2000": [], "$2000-10000": [], ">$10000": []}
    for g in agreed:
        p = g["listing_price"]
        if p < 100:
            buckets["<$100"].append(g["advantage"])
        elif p < 500:
            buckets["$100-500"].append(g["advantage"])
        elif p < 2000:
            buckets["$500-2000"].append(g["advantage"])
        elif p < 10000:
            buckets["$2000-10000"].append(g["advantage"])
        else:
            buckets[">$10000"].append(g["advantage"])
    print(f"  {'Bucket':20s} {'N':>4s} {'Mean Adv':>10s} {'Std':>8s}")
    for bucket, advs in buckets.items():
        if advs:
            s = stdev(advs) if len(advs) > 1 else 0
            print(f"  {bucket:20s} {len(advs):4d} {mean(advs):+10.4f} {s:8.4f}")

    # Target spread vs advantage
    print(f"\n--- Advantage by Target Spread ---")
    small_spread = []
    medium_spread = []
    large_spread = []
    for g in agreed:
        spread = g["seller_target"] - g["buyer_target"]
        if spread < 50:
            small_spread.append((spread, g["advantage"]))
        elif spread < 200:
            medium_spread.append((spread, g["advantage"]))
        else:
            large_spread.append((spread, g["advantage"]))

    for label, data_list in [("Small (<$50)", small_spread), ("Medium ($50-200)", medium_spread),
                               ("Large (>$200)", large_spread)]:
        if data_list:
            advs = [d[1] for d in data_list]
            s = stdev(advs) if len(advs) > 1 else 0
            print(f"  {label:20s}: N={len(data_list)}, mean_adv={mean(advs):+.4f}, std={s:.4f}")

    # Dealmaker analysis
    print(f"\n--- Who Finalizes the Deal ---")
    seller_deals = [g for g in agreed if g["dealmaker"] == "seller"]
    buyer_deals = [g for g in agreed if g["dealmaker"] == "buyer"]
    print(f"  Seller finalizes: {len(seller_deals)} ({100*len(seller_deals)/na:.1f}%)")
    print(f"  Buyer finalizes:  {len(buyer_deals)} ({100*len(buyer_deals)/na:.1f}%)")
    if seller_deals:
        print(f"    When seller deals: mean advantage = {mean(g['advantage'] for g in seller_deals):+.4f}")
    if buyer_deals:
        print(f"    When buyer deals:  mean advantage = {mean(g['advantage'] for g in buyer_deals):+.4f}")

    # Anti-steerability check (Issue 6)
    print(f"\n--- Anti-Steerability Check (Issue 6) ---")
    advantages = [g["advantage"] for g in agreed]
    positive = [a for a in advantages if a > 0]
    negative = [a for a in advantages if a < 0]
    zero = [a for a in advantages if a == 0]
    print(f"  Games where steering HELPS:  {len(positive)} ({100*len(positive)/na:.1f}%)")
    print(f"  Games where steering HURTS:  {len(negative)} ({100*len(negative)/na:.1f}%)")
    print(f"  Games with zero advantage:   {len(zero)} ({100*len(zero)/na:.1f}%)")
    print(f"  Advantage distribution:")
    print(f"    mean={mean(advantages):+.4f}, median={median(advantages):+.4f}, "
          f"std={stdev(advantages):.4f}")
    print(f"    min={min(advantages):+.4f}, max={max(advantages):+.4f}")

    # Quartile breakdown
    sorted_adv = sorted(advantages)
    q1 = sorted_adv[len(sorted_adv) // 4]
    q2 = sorted_adv[len(sorted_adv) // 2]
    q3 = sorted_adv[3 * len(sorted_adv) // 4]
    print(f"    Q1={q1:+.4f}, Q2(median)={q2:+.4f}, Q3={q3:+.4f}")

    # Advantage distribution buckets
    print(f"\n--- Advantage Distribution (agreed games) ---")
    hist = {"<-0.3": 0, "-0.3 to -0.1": 0, "-0.1 to 0.0": 0,
            "0.0 to 0.1": 0, "0.1 to 0.3": 0, ">0.3": 0}
    for a in advantages:
        if a < -0.3:
            hist["<-0.3"] += 1
        elif a < -0.1:
            hist["-0.3 to -0.1"] += 1
        elif a < 0.0:
            hist["-0.1 to 0.0"] += 1
        elif a < 0.1:
            hist["0.0 to 0.1"] += 1
        elif a < 0.3:
            hist["0.1 to 0.3"] += 1
        else:
            hist[">0.3"] += 1
    for bucket, count in hist.items():
        bar = "#" * count
        print(f"  {bucket:16s}: {count:3d} {bar}")


# ---------------------------------------------------------------------------
# A2b: Fast Search Results Diagnostic
# ---------------------------------------------------------------------------

def diagnose_fast_search():
    print(f"\n{'='*70}")
    print(f"A2b: FAST SEARCH RESULTS DIAGNOSTIC")
    print(f"{'='*70}")

    # Try to read from damon branch via git
    import subprocess

    # Stage 1
    print(f"\n--- Stage 1: Grid Search (probe_alpha=4.25, 5 games each) ---")
    try:
        s1_raw = subprocess.check_output(
            ["git", "show", "origin/damon:results/fast_run/stage1_results.json"],
            cwd="/Users/moiz/Documents/code/comp0087_snlp_cwk"
        )
        s1 = json.loads(s1_raw)
        print(f"  {'Dimension':35s} {'Method':10s} {'Layer':15s} {'Adv':>8s} {'Agree':>6s}")
        for e in sorted(s1, key=lambda x: -x["advantage"]):
            print(f"  {e['dimension']:35s} {e['method']:10s} {e['layer_preset']:15s} "
                  f"{e['advantage']:+8.4f} {e['agree_rate']:6.1%}")
    except Exception as ex:
        print(f"  Could not read stage1: {ex}")

    # Stage 2
    print(f"\n--- Stage 2: TPE Alpha Search (20 trials each, 5 games per trial) ---")
    try:
        s2_raw = subprocess.check_output(
            ["git", "show", "origin/damon:results/fast_run/stage2_results.json"],
            cwd="/Users/moiz/Documents/code/comp0087_snlp_cwk"
        )
        s2 = json.loads(s2_raw)
        for e in s2:
            trials = e["alpha_trials"]
            advs = [t["advantage"] for t in trials]
            positive = sum(1 for a in advs if a > 0)
            print(f"\n  Config: {e['dimension']} / {e['method']} / {e['layer_preset']} (layers {e['layer_indices']})")
            print(f"    Best alpha: {e['best_alpha']:.3f} → advantage: {e['advantage']:+.4f}")
            print(f"    Positive trials: {positive}/{len(trials)} ({100*positive/len(trials):.0f}%)")
            print(f"    Advantage across trials: mean={mean(advs):+.4f}, std={stdev(advs):.4f}")
            print(f"    Alpha range tested: [{min(t['alpha'] for t in trials):.2f}, "
                  f"{max(t['alpha'] for t in trials):.2f}]")

            # Check if advantage is monotonic with alpha (signal of overfitting)
            sorted_trials = sorted(trials, key=lambda t: t["alpha"])
            alphas = [t["alpha"] for t in sorted_trials]
            trial_advs = [t["advantage"] for t in sorted_trials]
            # Simple: correlation between alpha and advantage
            n_t = len(alphas)
            mean_a = mean(alphas)
            mean_adv = mean(trial_advs)
            cov = sum((alphas[i] - mean_a) * (trial_advs[i] - mean_adv) for i in range(n_t)) / n_t
            std_alpha = stdev(alphas) if n_t > 1 else 1
            std_adv = stdev(trial_advs) if n_t > 1 else 1
            corr = cov / (std_alpha * std_adv) if std_alpha > 0 and std_adv > 0 else 0
            print(f"    Corr(alpha, advantage) = {corr:.3f} (>0.5 suggests monotonic, "
                  f"~0 suggests noisy peak)")
    except Exception as ex:
        print(f"  Could not read stage2: {ex}")

    # Stage 3
    print(f"\n--- Stage 3: Validation (20 games each) ---")
    for rank in [1, 2]:
        try:
            s3_raw = subprocess.check_output(
                ["git", "show", f"origin/damon:results/fast_run/stage3_rank0{rank}.json"],
                cwd="/Users/moiz/Documents/code/comp0087_snlp_cwk"
            )
            s3 = json.loads(s3_raw)
            print(f"\n  Rank {rank}: {s3['dimension']} / {s3['method']} / {s3['layer_preset']}")
            print(f"    Alpha: {s3['alpha']:.3f}")
            print(f"    Advantage: {s3['advantage']:+.4f} ({s3['n_games']} games)")
            print(f"    Agree rate: {s3['agree_rate']:.1%}")
            print(f"    Steered score: {s3['steered_score']:.4f}")
        except Exception as ex:
            print(f"  Could not read stage3_rank0{rank}: {ex}")

    # Key observation: S2 vs S3 dropoff
    print(f"\n--- S2 → S3 Dropoff Analysis ---")
    print(f"  strategic_concession_making: S2 best=+0.5253 → S3=+0.3738 (drop: {0.5253-0.3738:.4f})")
    print(f"  anchoring:                  S2 best=+0.3538 → S3=+0.0102 (drop: {0.3538-0.0102:.4f})")
    print(f"  anchoring collapsed from +35% to +1% — classic overfitting to 5-game noise")
    print(f"  strategic_concession_making held up better but still dropped 29%")


# ---------------------------------------------------------------------------
# A3: score_deal() Analysis
# ---------------------------------------------------------------------------

def analyze_score_deal():
    print(f"\n{'='*70}")
    print(f"A3: score_deal() ANALYSIS")
    print(f"{'='*70}")

    results_path = "/Users/moiz/Documents/code/comp0087_snlp_cwk/results.json"
    with open(results_path) as f:
        data = json.load(f)

    games = data["games"]
    agreed = [g for g in games if g["agreed"]]

    print(f"\n--- Current score_deal() Behavior ---")
    print(f"  Formula: seller_score = (agreed - buyer_target) / (seller_target - buyer_target)")
    print(f"  Clamps to [0,1]. Returns (0.5, 0.5) for span <= 0.")

    # Check for edge cases in actual results
    print(f"\n--- Edge Cases in results.json ---")
    span_zero = []
    span_negative = []
    score_clamped = []
    score_sum_off = []

    for g in agreed:
        span = g["seller_target"] - g["buyer_target"]
        if span == 0:
            span_zero.append(g)
        elif span < 0:
            span_negative.append(g)

        # Check if scores were clamped
        if g["agreed_price"] is not None:
            raw_s = (g["agreed_price"] - g["buyer_target"]) / span if span > 0 else 0
            raw_b = (g["seller_target"] - g["agreed_price"]) / span if span > 0 else 0
            if raw_s < 0 or raw_s > 1 or raw_b < 0 or raw_b > 1:
                score_clamped.append({
                    "game_id": g["game_id"],
                    "agreed": g["agreed_price"],
                    "seller_target": g["seller_target"],
                    "buyer_target": g["buyer_target"],
                    "raw_seller_score": raw_s,
                    "raw_buyer_score": raw_b,
                    "clamped_seller": g["seller_score"],
                    "clamped_buyer": g["buyer_score"],
                })

        # Check sum
        total = g["seller_score"] + g["buyer_score"]
        if abs(total - 1.0) > 0.01:
            score_sum_off.append(g)

    print(f"  Span = 0 (equal targets)   : {len(span_zero)}")
    print(f"  Span < 0 (crossed targets) : {len(span_negative)}")
    print(f"  Scores clamped (deal outside target range): {len(score_clamped)}")
    if score_clamped:
        print(f"\n  --- Clamped Score Details ---")
        for c in score_clamped:
            print(f"    Game {c['game_id']}: agreed=${c['agreed']:.0f}, "
                  f"targets=[S:${c['seller_target']:.0f}, B:${c['buyer_target']:.0f}], "
                  f"raw=({c['raw_seller_score']:.3f}, {c['raw_buyer_score']:.3f}), "
                  f"clamped=({c['clamped_seller']}, {c['clamped_buyer']})")
    print(f"  Score sum != 1.0           : {len(score_sum_off)}")
    if score_sum_off:
        for g in score_sum_off:
            print(f"    Game {g['game_id']}: sum={g['seller_score']+g['buyer_score']:.4f}")

    # What clamping does to the advantage distribution
    if score_clamped:
        print(f"\n--- Impact of Clamping on Advantage ---")
        print(f"  {len(score_clamped)} games had scores clamped. In these games:")
        for c in score_clamped:
            clamped_adv = c["clamped_seller"] - c["clamped_buyer"]
            raw_adv_s = c["raw_seller_score"] - c["raw_buyer_score"]
            print(f"    Game {c['game_id']}: raw_adv={raw_adv_s:+.3f} → clamped_adv={clamped_adv:+.3f} "
                  f"(distortion={clamped_adv - raw_adv_s:+.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Phase A Diagnostic — P4 Foundation Fixes")
    print("=" * 70)

    # A1: Data audit
    all_scenarios, valid, overlapping = audit_dataset("train")
    # Also check validation split
    val_scenarios, val_valid, val_overlapping = audit_dataset("validation")

    # A2: Results diagnostic
    results_path = "/Users/moiz/Documents/code/comp0087_snlp_cwk/results.json"
    if Path(results_path).exists():
        diagnose_results(results_path)
    else:
        print(f"\nresults.json not found at {results_path}")

    # A2b: Fast search results
    diagnose_fast_search()

    # A3: score_deal() analysis
    if Path(results_path).exists():
        analyze_score_deal()

    print(f"\n{'='*70}")
    print(f"PHASE A DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
