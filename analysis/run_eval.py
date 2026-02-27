#!/usr/bin/env python3
"""
run_eval.py  —  P4 GPU Evaluation Suite
=========================================

Runs all GPU experiments in a single model-load session.
Designed for headless AWS execution (nohup-safe, incremental saves).

Prerequisites:
  - Steering vectors in vectors/{model_alias}/mean_diff/
    (run extract_vectors.py first if needed)
  - Dependencies: torch, transformers, numpy

Usage:
  # Run all experiments:
  nohup python run_eval.py --model qwen2.5-3b --all 2>&1 | tee results/eval/run.log &

  # Run only the critical pair (G1 + G2):
  python run_eval.py --model qwen2.5-3b --experiments g1g2

  # Run specific experiments:
  python run_eval.py --model qwen2.5-3b --experiments g1g2 g3

  # Check progress:
  tail -f results/eval/run.log

Experiments:
  g1g2   SCM steered + controlled baseline — 50 paired games (interleaved)
  g3     DonD cross-dataset — 30 games, SCM vector on Deal or No Deal
  g4     Firmness moderate alpha — 30 games, firmness/L27/alpha=5.0
  g5     Opening bid sensitivity — 30 games (10 each at 50%, 60%, 70%)
  cosine Vector cosine similarity check (no GPU inference, just numpy)

After completion:
  Results are in results/eval/*.json. Run the analysis pipeline:
    python metrics_b1.py        (adapt to read g1/g2 results)
    python llm_judge.py         (judge the g1 transcripts)
    python metrics_b3_roles.py  (role-separated analysis)
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


# ─── G1+G2: SCM steered vs controlled baseline (paired) ──────────────────

def run_g1_g2_paired(model, tokenizer, scenarios, roles, scm_dvecs, alpha,
                     output_dir):
    """
    Run G1 (SCM steered) and G2 (baseline alpha=0) interleaved on the
    same scenarios with the same role assignments.

    Interleaving ensures that if the script is interrupted, we always have
    N paired observations rather than N steered + 0 baselines.

    The paired design lets us compute:
      steering_effect[i] = g1_advantage[i] - g2_advantage[i]
    which isolates steering from role/scenario effects.
    """
    log.info("=" * 70)
    log.info("G1+G2: SCM steered vs controlled baseline (%d paired games)",
             len(scenarios))
    log.info("  SCM alpha=%.3f, layers=%s", alpha, SCM_CONFIG["layers"])
    log.info("=" * 70)

    g1_games, g2_games = [], []
    t0 = time.time()

    for i, (sc, role) in enumerate(zip(scenarios, roles)):
        try:
            # G1: steered
            r1 = run_single_game(model, tokenizer, sc, role, scm_dvecs, alpha)
            r1["game_id"] = i
            g1_games.append(r1)

            # G2: baseline (same scenario, same role label, alpha=0 both agents)
            r2 = run_single_game(model, tokenizer, sc, role, None, 0.0)
            r2["game_id"] = i
            g2_games.append(r2)

        except Exception as e:
            log.error("G1+G2 pair %d failed: %s\n%s", i, e, traceback.format_exc())
            continue

        # Incremental save after each pair
        save_results(g1_games, "g1_scm_steered", alpha, output_dir)
        save_results(g2_games, "g2_baseline", 0.0, output_dir)

        elapsed = time.time() - t0
        log.info(
            "Pair %2d/%d [%5.0fs] | G1 adv=%+.3f (%s) | G2 adv=%+.3f | %s",
            i + 1, len(scenarios), elapsed,
            r1["advantage"], "deal" if r1["agreed"] else "NODEAL",
            r2["advantage"],
            sc["title"][:50],
        )

    # Final save with paired analysis metadata
    if g1_games and g2_games:
        paired = []
        for r1, r2 in zip(g1_games, g2_games):
            if r1["agreed"] and r2["agreed"]:
                paired.append(r1["advantage"] - r2["advantage"])
        if paired:
            mean_effect = np.mean(paired)
            std_effect = np.std(paired, ddof=1)
            log.info("=" * 70)
            log.info("PAIRED STEERING EFFECT (N=%d agreed pairs):", len(paired))
            log.info("  Mean: %+.4f  Std: %.4f  (>0 means steering helps)",
                     mean_effect, std_effect)
            log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("G1+G2 complete: %d pairs in %.0fs", len(g1_games), elapsed)
    return g1_games, g2_games


# ─── G3: Deal or No Deal cross-dataset ────────────────────────────────────

def run_g3_dond(model, tokenizer, scm_dvecs, alpha, n_games, output_dir):
    """
    Test whether SCM vector improves Pareto efficiency on multi-issue
    negotiation. This is the falsifiability test for Issue 8.
    """
    log.info("=" * 70)
    log.info("G3: DonD cross-dataset validation (%d games)", n_games)
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
            "label": "g3_dond_scm",
            "timestamp": datetime.now().isoformat(),
            "config": {"alpha": alpha, "n_games": len(games)},
            "summary": summary,
            "games": games,
        }
        with open(output_dir / "g3_dond_scm.json", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        agreed = result.get("agreed", False)
        pareto = result.get("pareto_optimal", False) if agreed else False
        log.info(
            "G3 game %2d/%d [%5.0fs] | %s | pareto=%s | adv=%+.3f",
            i + 1, n_games, elapsed,
            "deal" if agreed else "NODEAL",
            pareto, result.get("advantage", 0),
        )

    if games:
        summary = summarise_dond(games, alpha)
        log.info("=" * 70)
        log.info("G3 SUMMARY: agree=%.0f%% pareto=%.0f%% efficiency=%.3f adv=%+.4f",
                 summary["agree_rate"] * 100, summary["pareto_rate"] * 100,
                 summary["avg_efficiency"], summary["advantage"])
        log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("G3 complete: %d games in %.0fs", len(games), elapsed)
    return games


# ─── G4: Firmness at moderate alpha ───────────────────────────────────────

def run_g4_firmness(model, tokenizer, firm_dvecs, alpha, n_games, output_dir,
                    seed):
    """
    Firmness at alpha≈5 on 3B (vs the original alpha=20 on 7B).
    Checks: lower clamping rate? Same role pattern? Different behavioral profile?
    """
    log.info("=" * 70)
    log.info("G4: Firmness moderate alpha (alpha=%.1f, %d games)", alpha, n_games)
    log.info("=" * 70)

    # Independent scenario sample (different seed from G1/G2)
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
            log.warning("G4 game %d failed: %s", i, e)
            continue

        save_results(games, "g4_firmness_moderate", alpha, output_dir)

        elapsed = time.time() - t0
        log.info(
            "G4 game %2d/%d [%5.0fs] | adv=%+.3f | %s | clamped=%s",
            i + 1, n_games, elapsed,
            r["advantage"],
            "deal" if r["agreed"] else "NODEAL",
            r.get("clamped", "?"),
        )

    elapsed = time.time() - t0
    log.info("G4 complete: %d games in %.0fs", len(games), elapsed)
    return games


# ─── G5: Opening bid sensitivity ──────────────────────────────────────────

def run_g5_sensitivity(model, tokenizer, scm_dvecs, alpha, output_dir, seed):
    """
    Test Issue 4: does the opening bid (50/60/70%) change the result?
    Same 10 scenarios across all 3 percentages for paired comparison.
    """
    log.info("=" * 70)
    log.info("G5: Opening bid sensitivity (10 games x 3 percentages)")
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

    with open(output_dir / "g5_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info("G5 complete in %.0fs", elapsed)
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


def _load_all_craigslist(split="train"):
    """Load all Craigslist scenarios (cached, single download)."""
    global _craigslist_cache
    if _craigslist_cache is None:
        log.info("Downloading Craigslist %s split (one-time)...", split)
        # Load with large num_samples to get everything, seeded for determinism
        old_state = random.getstate()
        random.seed(0)  # fixed seed for the full load
        _craigslist_cache = load_craigslist(split=split, num_samples=99999)
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
    p.add_argument("--vectors_dir", default=str(Path(_root) / "vectors"))
    p.add_argument("--output_dir", default=str(Path(_root) / "results" / "eval"))
    p.add_argument("--experiments", nargs="+",
                   choices=["g1g2", "g3", "g4", "g5", "cosine"],
                   default=None,
                   help="Which experiments to run. Omit for --all.")
    p.add_argument("--all", action="store_true",
                   help="Run all experiments (g1g2 + g3 + g4 + g5 + cosine).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "validation"], default="train",
                   help="Craigslist split. 'validation' recommended for paper.")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        experiments = {"g1g2", "g3", "g4", "g5", "cosine"}
    elif args.experiments:
        experiments = set(args.experiments)
    else:
        # Default: run the critical pair
        experiments = {"g1g2"}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = MODELS[args.model]
    hf_token = HF_TOKEN if model_cfg.requires_token else None

    # ─── Print run plan ───────────────────────────────────────────────
    plan = []
    total_games = 0
    if "cosine" in experiments:
        plan.append("cosine  Vector similarity check (no inference)")
    if "g1g2" in experiments:
        plan.append(f"g1g2    SCM paired: {N_CRAIGSLIST} steered + {N_CRAIGSLIST} baseline = {N_CRAIGSLIST * 2} games")
        total_games += N_CRAIGSLIST * 2
    if "g3" in experiments:
        plan.append(f"g3      DonD cross-dataset: {N_DOND} games")
        total_games += N_DOND
    if "g4" in experiments:
        plan.append(f"g4      Firmness moderate alpha: {N_FIRMNESS} games")
        total_games += N_FIRMNESS
    if "g5" in experiments:
        plan.append(f"g5      Opening bid sensitivity: {N_SENSITIVITY * 3} games")
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

    needs_scm = experiments & {"g1g2", "g3", "g5"}
    if needs_scm:
        scm_dvecs = load_vectors_safe(
            args.vectors_dir, model_cfg.alias,
            SCM_CONFIG["dimension"], SCM_CONFIG["method"], SCM_CONFIG["layers"],
        )
        if scm_dvecs is None:
            log.error("SCM vectors required but not found. Exiting.")
            sys.exit(1)
        log.info("Loaded SCM vectors: layers=%s", list(scm_dvecs.keys()))

    if "g4" in experiments:
        firm_dvecs = load_vectors_safe(
            args.vectors_dir, model_cfg.alias,
            FIRMNESS_CONFIG["dimension"], FIRMNESS_CONFIG["method"],
            FIRMNESS_CONFIG["layers"],
        )
        if firm_dvecs is None:
            log.warning("Firmness vectors not found — G4 will be skipped.")
            experiments.discard("g4")

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

    # ─── Pin G1/G2 scenarios ──────────────────────────────────────────
    if "g1g2" in experiments or "g5" in experiments:
        rng_main = random.Random(args.seed)
        all_sc = _load_all_craigslist(split=args.split)

        if "g1g2" in experiments:
            g1g2_scenarios = rng_main.sample(all_sc, min(N_CRAIGSLIST, len(all_sc)))
            g1g2_roles = ["seller" if i % 2 == 0 else "buyer"
                          for i in range(len(g1g2_scenarios))]

            # Save pinned scenarios for reproducibility
            with open(output_dir / "g1g2_scenarios_pinned.json", "w") as f:
                json.dump({
                    "seed": args.seed,
                    "split": args.split,
                    "n": len(g1g2_scenarios),
                    "scenarios": g1g2_scenarios,
                    "roles": g1g2_roles,
                }, f, indent=2, ensure_ascii=False)
            log.info("Pinned %d G1/G2 scenarios (seed=%d)", len(g1g2_scenarios), args.seed)

    # ─── Run experiments ──────────────────────────────────────────────
    results_summary = {}

    # G1+G2: SCM paired comparison (highest priority)
    if "g1g2" in experiments:
        g1_games, g2_games = run_g1_g2_paired(
            model, tokenizer, g1g2_scenarios, g1g2_roles,
            scm_dvecs, SCM_CONFIG["alpha"], output_dir,
        )
        g1_agreed = [g for g in g1_games if g["agreed"]]
        g2_agreed = [g for g in g2_games if g["agreed"]]
        results_summary["g1"] = {
            "n": len(g1_games), "agreed": len(g1_agreed),
            "advantage": np.mean([g["advantage"] for g in g1_agreed]) if g1_agreed else 0,
        }
        results_summary["g2"] = {
            "n": len(g2_games), "agreed": len(g2_agreed),
            "advantage": np.mean([g["advantage"] for g in g2_agreed]) if g2_agreed else 0,
        }

    # G3: DonD cross-dataset
    if "g3" in experiments:
        g3_games = run_g3_dond(
            model, tokenizer, scm_dvecs, SCM_CONFIG["alpha"],
            N_DOND, output_dir,
        )
        g3_agreed = [g for g in g3_games if g["agreed"]]
        results_summary["g3"] = {
            "n": len(g3_games), "agreed": len(g3_agreed),
            "advantage": np.mean([g["advantage"] for g in g3_agreed]) if g3_agreed else 0,
            "pareto_rate": (sum(1 for g in g3_agreed if g["pareto_optimal"]) / len(g3_agreed)) if g3_agreed else 0,
        }

    # G4: Firmness moderate alpha
    if "g4" in experiments and firm_dvecs:
        g4_games = run_g4_firmness(
            model, tokenizer, firm_dvecs, FIRMNESS_CONFIG["alpha"],
            N_FIRMNESS, output_dir, args.seed,
        )
        g4_agreed = [g for g in g4_games if g["agreed"]]
        results_summary["g4"] = {
            "n": len(g4_games), "agreed": len(g4_agreed),
            "advantage": np.mean([g["advantage"] for g in g4_agreed]) if g4_agreed else 0,
            "clamped_rate": (sum(1 for g in g4_agreed if g.get("clamped")) / len(g4_agreed)) if g4_agreed else 0,
        }

    # G5: Opening bid sensitivity
    if "g5" in experiments:
        g5_results = run_g5_sensitivity(
            model, tokenizer, scm_dvecs, SCM_CONFIG["alpha"],
            output_dir, args.seed,
        )
        results_summary["g5"] = {
            pct_label: data["summary"].get("advantage", 0)
            for pct_label, data in g5_results.items()
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
            log.info("  %-6s  adv=%+.4f  %s", exp, summary["advantage"], extras)
        else:
            log.info("  %-6s  %s", exp, summary)

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
    log.info("Next steps:")
    log.info("  1. Run: python metrics_b1.py   (adapt to read g1 results)")
    log.info("  2. Run: python llm_judge.py    (judge g1 transcripts)")
    log.info("  3. Compare g1 vs g2 for paired steering effect")


if __name__ == "__main__":
    main()
