#!/usr/bin/env python3
"""
run_eval.py  —  P4 GPU Evaluation Suite
=========================================

Runs all GPU experiments in a single model-load session.
Designed for local GPU execution with incremental saves (also works
headless on remote instances via nohup).

Prerequisites:
  - Steering vectors in vectors/{model_alias}/mean_diff/
    (run extract_vectors.py first if needed)
  - Dependencies: torch, transformers, numpy

Usage:
  # Run all experiments:
  nohup python run_eval.py --model qwen2.5-3b --all 2>&1 | tee results/eval/run.log &

  # Run only the primary SCM experiment (default):
  python run_eval.py --model qwen2.5-3b --experiments scm_craigslist

  # Run multiple experiments:
  python run_eval.py --model qwen2.5-3b --experiments scm_craigslist dond_crossval

  # Check progress:
  tail -f results/eval/run.log

Experiments:
  scm_craigslist  SCM fixed-role: buyer-steered + seller-steered + buyer-baseline (3×50 games)
  dond_crossval   DonD cross-dataset — 30 games, SCM vector on Deal or No Deal
  firmness        Firmness vector at moderate alpha — 30 games
  sensitivity     Opening bid sensitivity — 30 games (10 each at 50%, 60%, 70%)
  cosine          Vector cosine similarity check (no GPU inference, just numpy)

Output files:
  scm_buyer_steered.json    steered agent plays buyer (pairs with scm_buyer_baseline)
  scm_seller_steered.json   steered agent plays seller
  scm_buyer_baseline.json   baseline (alpha=0), buyer role (paired control for scm_buyer_steered)
  scm_scenarios_pinned.json exact scenarios used, for reproducibility
  dond_crossval.json        Deal-or-No-Deal cross-dataset results
  firmness_moderate.json    firmness vector results
  opening_sensitivity.json  opening-bid sensitivity results

After completion (CPU only):
  python analysis/turn_metrics.py    # per-turn behavior enrichment
  python analysis/role_analysis.py   # role-separated tables
  python analysis/analyse_results.py # paired statistical analysis
  python llm_judge.py                # qualitative judge
"""

import json
import random
import logging
import argparse
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Allow imports from the project root (one level up from analysis/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract_vectors import MODELS, HF_TOKEN
from apply_steering import (
    load_craigslist,
    load_direction_vectors,
    run_game,
    summarise,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_eval")


# ─── Experiment configurations ────────────────────────────────────────────
# Identified by P4 B4 analysis as the configs worth evaluating.

SCM_CONFIG = {
    "dimension": "strategic_concession_making",
    "method":    "mean_diff",
    "layers":    [18],       # middle layer on 36-layer model
    "alpha":     6.124,      # best from S2 TPE
}

FIRMNESS_CONFIG = {
    "dimension": "firmness",
    "method":    "mean_diff",
    "layers":    [27],       # late layer on 36-layer model
    "alpha":     5.0,        # moderate alpha from S2 dose-response
}

N_CRAIGSLIST     = 50    # G1/G2 paired comparison
N_DOND           = 30    # G3 cross-dataset
N_FIRMNESS       = 30    # G4 firmness rerun
N_SENSITIVITY    = 10    # G5 per opening-bid level


# ─── Helpers ──────────────────────────────────────────────────────────────

def save_results(games, label, alpha, output_dir, extra_meta=None):
    """Save per-game results with summary. Called after each game for safety."""
    summary = summarise(games, alpha) if games else {}
    out = {
        "label":     label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "alpha": alpha,
            "n_games": len(games),
        },
        "summary": summary,
        "games":   games,
    }
    if extra_meta:
        out["meta"] = extra_meta
    path = output_dir / f"{label}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def load_vectors_safe(vectors_dir, model_alias, dimension, method, layers):
    """Load vectors with clear error on failure."""
    try:
        return load_direction_vectors(
            vectors_dir=Path(vectors_dir),
            model_alias=model_alias,
            dimension=dimension,
            method=method,
            layer_indices=layers,
        )
    except FileNotFoundError as e:
        log.error("Vectors not found: %s", e)
        log.error("Run extract_vectors.py first, or copy vectors/ to this machine.")
        return None


def run_single_game(model, tokenizer, scenario, role, dvecs, alpha,
                    opening_bid_pct=0.6):
    """Run one Craigslist game. Wraps run_game with the steered/baseline split."""
    return run_game(
        model=model, tokenizer=tokenizer, scenario=scenario,
        dvecs_seller=dvecs if role == "seller" else None,
        alpha_seller=alpha if role == "seller" else 0.0,
        dvecs_buyer=dvecs  if role == "buyer"  else None,
        alpha_buyer=alpha  if role == "buyer"  else 0.0,
        steered_role=role,
        max_new_tokens=120,
        temperature=0.7,
        opening_bid_pct=opening_bid_pct,
    )


# ─── SCM Craigslist: buyer-steered + seller-steered + buyer-baseline ─────

def run_scm_craigslist(model, tokenizer, scenarios, scm_dvecs, alpha, output_dir):
    """
    Run three fixed-role batches on the same scenarios (interleaved):
      buyer-steered:  steered=BUYER  (all N games) → scm_buyer_steered.json
      seller-steered: steered=SELLER (all N games) → scm_seller_steered.json
      buyer-baseline: α=0, role=BUYER              → scm_buyer_baseline.json

    Interleaved so each completed triplet is a matched pair.
    buyer-steered vs buyer-baseline = clean buyer-steering effect.
    seller-steered stands alone as the seller-steering result.
    """
    log.info("=" * 70)
    log.info("SCM Craigslist: role-fixed batches (%d scenarios × 3 batches)",
             len(scenarios))
    log.info("  SCM alpha=%.3f, layers=%s", alpha, SCM_CONFIG["layers"])
    log.info("=" * 70)

    buyer_steered_games, seller_steered_games, buyer_baseline_games = [], [], []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        try:
            # buyer-steered: steered agent plays buyer
            r_buy = run_single_game(model, tokenizer, sc, "buyer", scm_dvecs, alpha)
            r_buy["game_id"] = i
            buyer_steered_games.append(r_buy)

            # seller-steered: steered agent plays seller
            r_sell = run_single_game(model, tokenizer, sc, "seller", scm_dvecs, alpha)
            r_sell["game_id"] = i
            seller_steered_games.append(r_sell)

            # buyer-baseline: same role as buyer-steered, alpha=0
            r_base = run_single_game(model, tokenizer, sc, "buyer", None, 0.0)
            r_base["game_id"] = i
            buyer_baseline_games.append(r_base)

        except Exception as e:
            log.error("SCM Craigslist triplet %d failed: %s\n%s", i, e, traceback.format_exc())
            continue

        # Incremental save after each triplet
        save_results(buyer_steered_games,  "scm_buyer_steered",  alpha, output_dir)
        save_results(seller_steered_games, "scm_seller_steered", alpha, output_dir)
        save_results(buyer_baseline_games, "scm_buyer_baseline", 0.0,   output_dir)

        elapsed = time.time() - t0
        mpd_buy  = r_buy.get("midpoint_deviation",  float("nan"))
        mpd_sell = r_sell.get("midpoint_deviation", float("nan"))
        mpd_base = r_base.get("midpoint_deviation", float("nan"))
        log.info(
            "Triplet %2d/%d [%5.0fs] | "
            "buy adv=%+.3f mpd=%+.3f | sell adv=%+.3f mpd=%+.3f | base adv=%+.3f mpd=%+.3f",
            i + 1, len(scenarios), elapsed,
            r_buy["advantage"],  mpd_buy,
            r_sell["advantage"], mpd_sell,
            r_base["advantage"], mpd_base,
        )

    # Paired analysis: buyer-steered vs buyer-baseline (both metrics)
    if buyer_steered_games and buyer_baseline_games:
        paired_adv = [r1["advantage"] - r2["advantage"]
                      for r1, r2 in zip(buyer_steered_games, buyer_baseline_games)
                      if r1["agreed"] and r2["agreed"]]
        paired_mpd = [r1["midpoint_deviation"] - r2["midpoint_deviation"]
                      for r1, r2 in zip(buyer_steered_games, buyer_baseline_games)
                      if r1["agreed"] and r2["agreed"]
                      and r1.get("midpoint_deviation") is not None
                      and r2.get("midpoint_deviation") is not None]
        if paired_adv:
            log.info("=" * 70)
            log.info("BUYER STEERING EFFECT (N=%d agreed pairs):", len(paired_adv))
            log.info("  Δadvantage:        Mean=%+.4f  Std=%.4f  (clamped, noisy)",
                     np.mean(paired_adv), np.std(paired_adv, ddof=1))
            if paired_mpd:
                log.info("  Δmidpoint_dev:     Mean=%+.4f  Std=%.4f  (clamp-immune, preferred)",
                         np.mean(paired_mpd), np.std(paired_mpd, ddof=1))
            log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("SCM Craigslist complete: %d triplets in %.0fs",
             len(buyer_steered_games), elapsed)
    return buyer_steered_games, seller_steered_games, buyer_baseline_games


# ─── Firmness Craigslist: buyer-steered + seller-steered + buyer-baseline ─

def run_firmness_craigslist(model, tokenizer, scenarios, firm_dvecs, alpha, output_dir):
    """
    Same 3-batch fixed-role design as run_scm_craigslist, but with the
    firmness vector (layer 27, alpha 5.0).
    Outputs: firmness_buyer_steered.json, firmness_seller_steered.json,
             firmness_buyer_baseline.json, firmness_scenarios_pinned.json
    """
    log.info("=" * 70)
    log.info("Firmness Craigslist: role-fixed batches (%d scenarios × 3 batches)",
             len(scenarios))
    log.info("  Firmness alpha=%.3f, layers=%s", alpha, FIRMNESS_CONFIG["layers"])
    log.info("=" * 70)

    buyer_steered_games, seller_steered_games, buyer_baseline_games = [], [], []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        try:
            r_buy  = run_single_game(model, tokenizer, sc, "buyer", firm_dvecs, alpha)
            r_buy["game_id"] = i
            buyer_steered_games.append(r_buy)

            r_sell = run_single_game(model, tokenizer, sc, "seller", firm_dvecs, alpha)
            r_sell["game_id"] = i
            seller_steered_games.append(r_sell)

            r_base = run_single_game(model, tokenizer, sc, "buyer", None, 0.0)
            r_base["game_id"] = i
            buyer_baseline_games.append(r_base)

        except Exception as e:
            log.error("Firmness Craigslist triplet %d failed: %s\n%s", i, e, traceback.format_exc())
            continue

        save_results(buyer_steered_games,  "firmness_buyer_steered",  alpha, output_dir)
        save_results(seller_steered_games, "firmness_seller_steered", alpha, output_dir)
        save_results(buyer_baseline_games, "firmness_buyer_baseline", 0.0,   output_dir)

        elapsed = time.time() - t0
        mpd_buy  = r_buy.get("midpoint_deviation",  float("nan"))
        mpd_sell = r_sell.get("midpoint_deviation", float("nan"))
        mpd_base = r_base.get("midpoint_deviation", float("nan"))
        log.info(
            "Triplet %2d/%d [%5.0fs] | "
            "buy adv=%+.3f mpd=%+.3f | sell adv=%+.3f mpd=%+.3f | base adv=%+.3f mpd=%+.3f",
            i + 1, len(scenarios), elapsed,
            r_buy["advantage"],  mpd_buy,
            r_sell["advantage"], mpd_sell,
            r_base["advantage"], mpd_base,
        )

    # Paired analysis
    if buyer_steered_games and buyer_baseline_games:
        paired_adv = [r1["advantage"] - r2["advantage"]
                      for r1, r2 in zip(buyer_steered_games, buyer_baseline_games)
                      if r1["agreed"] and r2["agreed"]]
        paired_mpd = [r1["midpoint_deviation"] - r2["midpoint_deviation"]
                      for r1, r2 in zip(buyer_steered_games, buyer_baseline_games)
                      if r1["agreed"] and r2["agreed"]
                      and r1.get("midpoint_deviation") is not None
                      and r2.get("midpoint_deviation") is not None]
        if paired_adv:
            log.info("=" * 70)
            log.info("FIRMNESS BUYER STEERING EFFECT (N=%d agreed pairs):", len(paired_adv))
            log.info("  Δadvantage:    Mean=%+.4f  Std=%.4f", np.mean(paired_adv), np.std(paired_adv, ddof=1))
            if paired_mpd:
                log.info("  Δmidpoint_dev: Mean=%+.4f  Std=%.4f  (preferred)",
                         np.mean(paired_mpd), np.std(paired_mpd, ddof=1))
            log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("Firmness Craigslist complete: %d triplets in %.0fs",
             len(buyer_steered_games), elapsed)
    return buyer_steered_games, seller_steered_games, buyer_baseline_games


# ─── DonD cross-dataset validation ───────────────────────────────────────

def run_dond_crossval(model, tokenizer, scm_dvecs, alpha, n_games, output_dir):
    """
    Test whether SCM vector improves Pareto efficiency on multi-issue
    negotiation (Deal or No Deal). Cross-dataset falsifiability check.
    """
    log.info("=" * 70)
    log.info("DonD cross-dataset validation (%d games)", n_games)
    log.info("=" * 70)

    try:
        from deal_or_no_deal import load_dealornodeal, run_game_dond, summarise_dond
    except ImportError:
        log.error("deal_or_no_deal.py not found — skipping G3")
        return []

    scenarios = load_dealornodeal(split="selfplay", num_samples=n_games)
    games = []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        steered_role = "agent1" if i % 2 == 0 else "agent2"
        dvecs_a1 = scm_dvecs if steered_role == "agent1" else None
        alpha_a1 = alpha     if steered_role == "agent1" else 0.0
        dvecs_a2 = scm_dvecs if steered_role == "agent2" else None
        alpha_a2 = alpha     if steered_role == "agent2" else 0.0

        try:
            result = run_game_dond(
                model, tokenizer, sc,
                dvecs_a1, alpha_a1, dvecs_a2, alpha_a2,
                steered_role=steered_role,
            )
            result["game_id"] = i
            games.append(result)
        except Exception as e:
            log.warning("G3 game %d failed: %s", i, e)
            continue

        # Incremental save
        summary = summarise_dond(games, alpha)
        out = {
            "label": "dond_crossval",
            "timestamp": datetime.now().isoformat(),
            "config": {"alpha": alpha, "n_games": len(games)},
            "summary": summary,
            "games": games,
        }
        with open(output_dir / "dond_crossval.json", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        agreed = result.get("agreed", False)
        pareto = result.get("pareto_optimal", False) if agreed else False
        log.info(
            "dond game %2d/%d [%5.0fs] | %s | pareto=%s | adv=%+.3f",
            i + 1, n_games, elapsed,
            "deal" if agreed else "NODEAL",
            pareto, result.get("advantage", 0),
        )

    if games:
        summary = summarise_dond(games, alpha)
        log.info("=" * 70)
        log.info("DonD SUMMARY: agree=%.0f%% pareto=%.0f%% efficiency=%.3f adv=%+.4f",
                 summary["agree_rate"] * 100, summary["pareto_rate"] * 100,
                 summary["avg_efficiency"], summary["advantage"])
        log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("DonD cross-dataset complete: %d games in %.0fs", len(games), elapsed)
    return games


# ─── Firmness at moderate alpha ───────────────────────────────────────────

def run_firmness_moderate(model, tokenizer, firm_dvecs, alpha, n_games, output_dir,
                          seed):
    """
    Firmness at alpha≈5 on 3B (vs the original alpha=20 on 7B).
    Checks: lower clamping rate? Same role pattern? Different behavioral profile?
    """
    log.info("=" * 70)
    log.info("Firmness moderate alpha (alpha=%.1f, %d games)", alpha, n_games)
    log.info("=" * 70)

    # Independent scenario sample (different seed from SCM batches)
    rng = random.Random(seed + 100)
    all_scenarios = _load_all_craigslist()
    scenarios = rng.sample(all_scenarios, min(n_games, len(all_scenarios)))
    roles = ["seller" if i % 2 == 0 else "buyer" for i in range(len(scenarios))]

    games = []
    t0 = time.time()

    for i, (sc, role) in enumerate(zip(scenarios, roles)):
        try:
            r = run_single_game(model, tokenizer, sc, role, firm_dvecs, alpha)
            r["game_id"] = i
            games.append(r)
        except Exception as e:
            log.warning("firmness game %d failed: %s", i, e)
            continue

        save_results(games, "firmness_moderate", alpha, output_dir)

        elapsed = time.time() - t0
        log.info(
            "firmness game %2d/%d [%5.0fs] | adv=%+.3f | %s | clamped=%s",
            i + 1, n_games, elapsed,
            r["advantage"],
            "deal" if r["agreed"] else "NODEAL",
            r.get("clamped", "?"),
        )

    elapsed = time.time() - t0
    log.info("Firmness moderate complete: %d games in %.0fs", len(games), elapsed)
    return games


# ─── Opening bid sensitivity ──────────────────────────────────────────────

def run_opening_sensitivity(model, tokenizer, scm_dvecs, alpha, output_dir, seed):
    """
    Does the opening bid percentage (50/60/70%) change the outcome?
    Same 10 scenarios across all 3 percentages for paired comparison.
    """
    log.info("=" * 70)
    log.info("Opening bid sensitivity (10 games x 3 percentages)")
    log.info("=" * 70)

    rng = random.Random(seed + 200)
    all_scenarios = _load_all_craigslist()
    scenarios = rng.sample(all_scenarios, min(N_SENSITIVITY, len(all_scenarios)))
    roles = ["seller" if i % 2 == 0 else "buyer" for i in range(len(scenarios))]

    all_results = {}
    t0 = time.time()

    for pct in [0.5, 0.6, 0.7]:
        label = f"opening_{int(pct * 100)}pct"
        games = []

        for i, (sc, role) in enumerate(zip(scenarios, roles)):
            try:
                r = run_single_game(
                    model, tokenizer, sc, role, scm_dvecs, alpha,
                    opening_bid_pct=pct,
                )
                r["game_id"] = i
                r["opening_bid_pct"] = pct
                games.append(r)
            except Exception as e:
                log.warning("G5 %s game %d failed: %s", label, i, e)
                continue

        summary = summarise(games, alpha)
        all_results[label] = {"summary": summary, "games": games}

        elapsed = time.time() - t0
        log.info(
            "G5 %s [%5.0fs] | adv=%+.4f | agree=%.0f%%",
            label, elapsed,
            summary["advantage"], summary["agree_rate"] * 100,
        )

    with open(output_dir / "opening_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info("Opening sensitivity complete in %.0fs", elapsed)
    return all_results


# ─── Vector cosine similarity check ──────────────────────────────────────

def run_cosine_check(vectors_dir, model_alias):
    """
    Non-GPU check: are SCM, firmness, and anchoring vectors similar?
    If cosine similarity is high, they may encode the same direction.
    """
    log.info("=" * 70)
    log.info("COSINE SIMILARITY CHECK (no GPU needed)")
    log.info("=" * 70)

    dims = ["strategic_concession_making", "firmness", "anchoring"]
    layer_map = {
        "strategic_concession_making": 18,
        "firmness":                    27,
        "anchoring":                   18,
    }

    vectors = {}
    for dim in dims:
        layer = layer_map[dim]
        try:
            dvecs = load_direction_vectors(
                vectors_dir=Path(vectors_dir),
                model_alias=model_alias,
                dimension=dim,
                method="mean_diff",
                layer_indices=[layer],
            )
            vectors[dim] = dvecs[layer]
        except FileNotFoundError:
            log.warning("Vector for %s/layer%d not found — skipping", dim, layer)

    if len(vectors) < 2:
        log.warning("Need at least 2 vectors for comparison. Found: %s",
                     list(vectors.keys()))
        return {}

    results = {}
    dims_found = list(vectors.keys())
    for i, d1 in enumerate(dims_found):
        for d2 in dims_found[i + 1:]:
            v1 = vectors[d1].flatten().astype(np.float64)
            v2 = vectors[d2].flatten().astype(np.float64)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            key = f"{d1}_vs_{d2}"
            results[key] = round(float(cos_sim), 4)
            log.info("  cos(%s, %s) = %.4f", d1, d2, cos_sim)

    # Also check within-dimension split-half stability if all-layers file exists
    for dim in dims_found:
        all_path = Path(vectors_dir) / model_alias / "mean_diff" / f"{dim}_all_layers.npy"
        if all_path.exists():
            all_vecs = np.load(all_path)
            layer = layer_map.get(dim)
            if layer and layer < len(all_vecs):
                v = all_vecs[layer].flatten().astype(np.float64)
                v_stored = vectors[dim].flatten().astype(np.float64)
                match = np.dot(v, v_stored) / (np.linalg.norm(v) * np.linalg.norm(v_stored))
                log.info("  %s layer%d all-vs-single consistency: %.4f",
                         dim, layer, match)

    log.info("Interpretation:")
    log.info("  cos > 0.8  → vectors likely encode the same direction")
    log.info("  cos 0.3-0.8 → partially overlapping")
    log.info("  cos < 0.3  → genuinely different directions")

    return results


# ─── Scenario loading (cached) ────────────────────────────────────────────

_craigslist_cache = None
MIN_SPAN = 100   # passed to load_craigslist; removes near-random low-span scenarios


def _load_all_craigslist(split="train"):
    """Load all Craigslist scenarios (cached, single download)."""
    global _craigslist_cache
    if _craigslist_cache is None:
        log.info("Downloading Craigslist %s split (one-time)...", split)
        # Load with large num_samples to get everything, seeded for determinism
        old_state = random.getstate()
        random.seed(0)  # fixed seed for the full load
        _craigslist_cache = load_craigslist(split=split, num_samples=99999,
                                            min_span=MIN_SPAN)
        random.setstate(old_state)
        log.info("Cached %d Craigslist scenarios.", len(_craigslist_cache))
    return _craigslist_cache


# ─── CLI + Main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="P4 GPU evaluation suite — run all experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", choices=list(MODELS.keys()), default="qwen2.5-3b")
    _root = str(Path(__file__).resolve().parent.parent)
    p.add_argument("--vectors_dir", required=True,
                   help="Path to vectors dir for this variant, e.g. vectors/neg8dim_12pairs_matched/negotiation")
    p.add_argument("--output_dir", default=str(Path(_root) / "results" / "eval"))
    p.add_argument("--experiments", nargs="+",
                   choices=["scm_craigslist", "firmness_craigslist", "dond_crossval",
                             "firmness", "sensitivity", "cosine"],
                   default=None,
                   help="Which experiments to run (default: scm_craigslist). "
                        "scm_craigslist=SCM buyer+seller+baseline batches; "
                        "firmness_craigslist=firmness buyer+seller+baseline batches; "
                        "dond_crossval=Deal-or-No-Deal cross-dataset; "
                        "firmness=firmness vector moderate alpha (old 30-game design); "
                        "sensitivity=opening-bid sensitivity; "
                        "cosine=vector similarity check (no GPU).")
    p.add_argument("--all", action="store_true",
                   help="Run all experiments.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "validation"], default="train",
                   help="Craigslist split. 'validation' recommended for paper.")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        experiments = {"scm_craigslist", "firmness_craigslist", "dond_crossval",
                       "firmness", "sensitivity", "cosine"}
    elif args.experiments:
        experiments = set(args.experiments)
    else:
        # Default: run the primary SCM experiment
        experiments = {"scm_craigslist"}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = MODELS[args.model]
    hf_token = HF_TOKEN if model_cfg.requires_token else None

    # ─── Print run plan ───────────────────────────────────────────────
    plan = []
    total_games = 0
    if "cosine" in experiments:
        plan.append("cosine          Vector similarity check (no inference)")
    if "scm_craigslist" in experiments:
        plan.append(f"scm_craigslist  SCM buyer-steered + seller-steered + baseline: "
                    f"{N_CRAIGSLIST * 3} games")
        total_games += N_CRAIGSLIST * 3
    if "dond_crossval" in experiments:
        plan.append(f"dond_crossval   DonD cross-dataset: {N_DOND} games")
        total_games += N_DOND
    if "firmness_craigslist" in experiments:
        plan.append(f"firmness_craigslist  Firmness buyer+seller+baseline: "
                    f"{N_CRAIGSLIST * 3} games")
        total_games += N_CRAIGSLIST * 3
    if "firmness" in experiments:
        plan.append(f"firmness        Firmness vector moderate alpha: {N_FIRMNESS} games")
        total_games += N_FIRMNESS
    if "sensitivity" in experiments:
        plan.append(f"sensitivity     Opening-bid sensitivity: {N_SENSITIVITY * 3} games")
        total_games += N_SENSITIVITY * 3

    log.info("=" * 70)
    log.info("P4 EVALUATION RUN")
    log.info("  Started:  %s", datetime.now().isoformat())
    log.info("  Model:    %s (%s)", args.model, model_cfg.hf_id)
    log.info("  Output:   %s/", output_dir)
    log.info("  Seed:     %d", args.seed)
    log.info("  Split:    %s", args.split)
    for p in plan:
        log.info("  [x] %s", p)
    log.info("  Total inference games: ~%d", total_games)
    log.info("=" * 70)

    t_start = time.time()

    # ─── Cosine check (before model load, no GPU needed) ──────────────
    cosine_results = None
    if "cosine" in experiments:
        cosine_results = run_cosine_check(args.vectors_dir, model_cfg.alias)
        if cosine_results:
            with open(output_dir / "cosine_similarity.json", "w") as f:
                json.dump(cosine_results, f, indent=2)

    # If only cosine was requested, exit
    if experiments == {"cosine"}:
        log.info("Only cosine check requested — done.")
        return

    # ─── Load vectors ─────────────────────────────────────────────────
    scm_dvecs = None
    firm_dvecs = None

    needs_scm = experiments & {"scm_craigslist", "dond_crossval", "sensitivity"}
    if needs_scm:
        scm_dvecs = load_vectors_safe(
            args.vectors_dir, model_cfg.alias,
            SCM_CONFIG["dimension"], SCM_CONFIG["method"], SCM_CONFIG["layers"],
        )
        if scm_dvecs is None:
            log.error("SCM vectors required but not found. Exiting.")
            sys.exit(1)
        log.info("Loaded SCM vectors: layers=%s", list(scm_dvecs.keys()))

    if experiments & {"firmness_craigslist", "firmness"}:
        firm_dvecs = load_vectors_safe(
            args.vectors_dir, model_cfg.alias,
            FIRMNESS_CONFIG["dimension"], FIRMNESS_CONFIG["method"],
            FIRMNESS_CONFIG["layers"],
        )
        if firm_dvecs is None:
            log.warning("Firmness vectors not found — firmness experiments will be skipped.")
            experiments.discard("firmness_craigslist")
            experiments.discard("firmness")

    # ─── Load model (single load for all experiments) ─────────────────
    log.info("Loading model: %s ...", model_cfg.hf_id)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id, token=hf_token, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id, token=hf_token,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    log.info("Model loaded. Device: %s", next(model.parameters()).device)

    # ─── Pin scenarios ─────────────────────────────────────────────────
    if experiments & {"scm_craigslist", "firmness_craigslist", "sensitivity"}:
        rng_main = random.Random(args.seed)
        all_sc = _load_all_craigslist(split=args.split)

        if "scm_craigslist" in experiments:
            scm_scenarios = rng_main.sample(all_sc, min(N_CRAIGSLIST, len(all_sc)))

            # Save pinned scenarios for reproducibility
            with open(output_dir / "scm_scenarios_pinned.json", "w") as f:
                json.dump({
                    "seed":      args.seed,
                    "split":     args.split,
                    "min_span":  MIN_SPAN,
                    "n":         len(scm_scenarios),
                    "scenarios": scm_scenarios,
                    "buyer_steered_role":  "buyer",
                    "seller_steered_role": "seller",
                    "baseline_role":       "buyer",
                }, f, indent=2, ensure_ascii=False)
            log.info("Pinned %d SCM scenarios (seed=%d, min_span=%d)",
                     len(scm_scenarios), args.seed, MIN_SPAN)

        if "firmness_craigslist" in experiments:
            # Use a different seed offset so scenarios don't overlap with SCM
            rng_firm = random.Random(args.seed + 1)
            firmness_scenarios = rng_firm.sample(all_sc, min(N_CRAIGSLIST, len(all_sc)))
            with open(output_dir / "firmness_scenarios_pinned.json", "w") as f:
                json.dump({
                    "seed":      args.seed + 1,
                    "split":     args.split,
                    "min_span":  MIN_SPAN,
                    "n":         len(firmness_scenarios),
                    "scenarios": firmness_scenarios,
                }, f, indent=2, ensure_ascii=False)
            log.info("Pinned %d firmness scenarios (seed=%d, min_span=%d)",
                     len(firmness_scenarios), args.seed + 1, MIN_SPAN)

    # ─── Run experiments ──────────────────────────────────────────────
    results_summary = {}

    # SCM Craigslist: primary fixed-role batches
    if "scm_craigslist" in experiments:
        buyer_steered, seller_steered, buyer_baseline = run_scm_craigslist(
            model, tokenizer, scm_scenarios,
            scm_dvecs, SCM_CONFIG["alpha"], output_dir,
        )
        buy_agreed  = [g for g in buyer_steered  if g["agreed"]]
        sell_agreed = [g for g in seller_steered if g["agreed"]]
        base_agreed = [g for g in buyer_baseline if g["agreed"]]
        results_summary["scm_buyer"] = {
            "n": len(buyer_steered), "agreed": len(buy_agreed),
            "advantage": np.mean([g["advantage"] for g in buy_agreed]) if buy_agreed else 0,
        }
        results_summary["scm_seller"] = {
            "n": len(seller_steered), "agreed": len(sell_agreed),
            "advantage": np.mean([g["advantage"] for g in sell_agreed]) if sell_agreed else 0,
        }
        results_summary["scm_baseline"] = {
            "n": len(buyer_baseline), "agreed": len(base_agreed),
            "advantage": np.mean([g["advantage"] for g in base_agreed]) if base_agreed else 0,
        }

    # Firmness Craigslist: primary fixed-role batches
    if "firmness_craigslist" in experiments and firm_dvecs:
        firm_buy, firm_sell, firm_base = run_firmness_craigslist(
            model, tokenizer, firmness_scenarios,
            firm_dvecs, FIRMNESS_CONFIG["alpha"], output_dir,
        )
        fb_agreed   = [g for g in firm_buy  if g["agreed"]]
        fs_agreed   = [g for g in firm_sell if g["agreed"]]
        fbl_agreed  = [g for g in firm_base if g["agreed"]]
        results_summary["firmness_buyer"] = {
            "n": len(firm_buy), "agreed": len(fb_agreed),
            "advantage":        np.mean([g["advantage"]        for g in fb_agreed]) if fb_agreed else 0,
            "midpoint_dev":     np.mean([g["midpoint_deviation"] for g in fb_agreed
                                         if g.get("midpoint_deviation") is not None]) if fb_agreed else 0,
        }
        results_summary["firmness_seller"] = {
            "n": len(firm_sell), "agreed": len(fs_agreed),
            "advantage":        np.mean([g["advantage"]        for g in fs_agreed]) if fs_agreed else 0,
            "midpoint_dev":     np.mean([g["midpoint_deviation"] for g in fs_agreed
                                         if g.get("midpoint_deviation") is not None]) if fs_agreed else 0,
        }
        results_summary["firmness_baseline"] = {
            "n": len(firm_base), "agreed": len(fbl_agreed),
            "advantage":        np.mean([g["advantage"]        for g in fbl_agreed]) if fbl_agreed else 0,
            "midpoint_dev":     np.mean([g["midpoint_deviation"] for g in fbl_agreed
                                         if g.get("midpoint_deviation") is not None]) if fbl_agreed else 0,
        }

    # DonD cross-dataset validation
    if "dond_crossval" in experiments:
        dond_games = run_dond_crossval(
            model, tokenizer, scm_dvecs, SCM_CONFIG["alpha"],
            N_DOND, output_dir,
        )
        dond_agreed = [g for g in dond_games if g["agreed"]]
        results_summary["dond_crossval"] = {
            "n": len(dond_games), "agreed": len(dond_agreed),
            "advantage": np.mean([g["advantage"] for g in dond_agreed]) if dond_agreed else 0,
            "pareto_rate": (sum(1 for g in dond_agreed if g["pareto_optimal"]) / len(dond_agreed)) if dond_agreed else 0,
        }

    # Firmness at moderate alpha
    if "firmness" in experiments and firm_dvecs:
        firmness_games = run_firmness_moderate(
            model, tokenizer, firm_dvecs, FIRMNESS_CONFIG["alpha"],
            N_FIRMNESS, output_dir, args.seed,
        )
        firm_agreed = [g for g in firmness_games if g["agreed"]]
        results_summary["firmness"] = {
            "n": len(firmness_games), "agreed": len(firm_agreed),
            "advantage": np.mean([g["advantage"] for g in firm_agreed]) if firm_agreed else 0,
            "clamped_rate": (sum(1 for g in firm_agreed if g.get("clamped")) / len(firm_agreed)) if firm_agreed else 0,
        }

    # Opening bid sensitivity
    if "sensitivity" in experiments:
        sensitivity_results = run_opening_sensitivity(
            model, tokenizer, scm_dvecs, SCM_CONFIG["alpha"],
            output_dir, args.seed,
        )
        results_summary["sensitivity"] = {
            pct_label: data["summary"].get("advantage", 0)
            for pct_label, data in sensitivity_results.items()
        }

    # ─── Final summary ────────────────────────────────────────────────
    elapsed = time.time() - t_start

    log.info("")
    log.info("=" * 70)
    log.info("ALL EXPERIMENTS COMPLETE")
    log.info("  Elapsed: %.0f seconds (%.1f minutes)", elapsed, elapsed / 60)
    log.info("  Output:  %s/", output_dir)
    log.info("-" * 70)

    for exp, summary in results_summary.items():
        if isinstance(summary, dict) and "advantage" in summary:
            extras = "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in summary.items()
                if k not in ("advantage",)
            )
            log.info("  %-18s  adv=%+.4f  %s", exp, summary["advantage"], extras)
        else:
            log.info("  %-18s  %s", exp, summary)

    log.info("=" * 70)

    # Save run metadata
    meta = {
        "model":          args.model,
        "seed":           args.seed,
        "split":          args.split,
        "started":        datetime.fromtimestamp(t_start).isoformat(),
        "completed":      datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "experiments":    sorted(experiments),
        "results_summary": results_summary,
    }
    if cosine_results:
        meta["cosine_similarity"] = cosine_results
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info("Metadata saved to %s/run_meta.json", output_dir)
    log.info("Next steps (CPU, no GPU needed):")
    log.info("  python analysis/turn_metrics.py   # per-turn behavior (reads scm_buyer_steered.json)")
    log.info("  python analysis/role_analysis.py  # role-separated tables (reads turn_metrics_enriched.json)")
    log.info("  python analysis/analyse_results.py # paired statistics")
    log.info("  python llm_judge.py               # qualitative judge on scm_buyer_steered")


if __name__ == "__main__":
    main()
