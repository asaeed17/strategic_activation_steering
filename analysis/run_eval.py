#!/usr/bin/env python3
"""
run_eval.py  —  P4 GPU Evaluation Suite
=========================================

Runs gridsearch-driven evaluation in a single model-load session.
Alpha and layers are read automatically from stage2_results.json for each
dimension — no hardcoded configs. Only dimensions that pass the gridsearch
filter (pen_delta >= 0.15, dir_ok) are evaluated.

Designed for local GPU execution with incremental saves (also works
headless on remote instances via nohup).

Prerequisites:
  - Steering vectors in vectors/{model_alias}/mean_diff/
    (run extract_vectors.py first if needed)
  - Gridsearch results from hyperparameter search
  - Dependencies: torch, transformers, numpy

Usage:
  nohup python run_eval.py hyperparameter_results/gridsearch_neg15dim_12pairs_matched \\
      --model qwen2.5-3b \\
      --vectors_dir vectors/neg15dim_12pairs_matched/negotiation \\
      --output_dir results/eval/gridsearch_qwen2.5-3b \\
      2>&1 | tee results/eval/gridsearch_qwen2.5-3b.log &

  # Check progress:
  tail -f results/eval/gridsearch_qwen2.5-3b.log

Output (per dimension that passes the filter):
  {dim}/scenarios_pinned.json         exact scenarios used
  {dim}/{dim}_buyer_steered.json      steered agent plays buyer
  {dim}/{dim}_seller_steered.json     steered agent plays seller
  {dim}/{dim}_buyer_baseline.json     alpha=0 buyer (paired control)
  {dim}/{dim}_seller_baseline.json    alpha=0 seller (paired control)
  gridsearch_manifest.json            which dims ran and at what config
  gridsearch_eval_summary.json        avg_delta/avg_agree matching gridsearch metric
  run_meta.json                       run metadata

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
progress_log = logging.getLogger("run_eval.progress")


N_CRAIGSLIST = 50    # games per dimension (buyer-steered + seller-steered + buyer-baseline + seller-baseline)

# ─── Gridsearch scoring constants (must match visualise_results.ipynb) ────────
GS_LAMBDA           = 0.01   # penalty per unit of |alpha|
GS_MIN_AGREE_RATE   = 0.60   # filter alphas below this before picking winner
GS_MIN_DELTA_TO_RUN = 0.15   # penalised delta must exceed this to run eval
# Dimensions where positive alpha = more of the concept (flag if winner is negative)
GS_EXPECTED_POSITIVE = {
    "active_listening", "empathy", "rapport_building",
    "emotional_regulation", "interest_based_reasoning",
    "assertiveness", "reframing", "value_creation",
    "patience", "information_gathering",
}


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


def load_gridsearch_config(gridsearch_dir, model_alias):
    """
    Read stage2_results.json for every dimension found under
    gridsearch_dir / model_alias / <dimension> / stage2_results.json
    and apply the same three-step scoring as visualise_results.ipynb:

      1. Filter alphas with avg_agree_rate < GS_MIN_AGREE_RATE
      2. Pick max penalised_score = avg_delta - GS_LAMBDA * |alpha|
      3. Flag direction mismatch for GS_EXPECTED_POSITIVE dimensions

    Returns a dict keyed by dimension name:
      {
        "alpha":      float,
        "layers":     [int, ...],
        "pen_delta":  float,
        "dir_ok":     bool,
        "run_eval":   bool,   # True iff pen_delta >= GS_MIN_DELTA_TO_RUN and dir_ok
        "agree_rate": float,
      }

    Dimensions with no stage2_results.json are silently skipped.
    """
    gs_path = Path(gridsearch_dir) / model_alias
    if not gs_path.exists():
        log.error("Gridsearch path not found: %s", gs_path)
        return {}

    configs = {}
    for dim_dir in sorted(gs_path.iterdir()):
        if not dim_dir.is_dir():
            continue
        s2_path = dim_dir / "stage2_results.json"
        if not s2_path.exists():
            continue
        with open(s2_path) as f:
            rows = json.load(f)

        # Step 1: filter by agree rate
        filtered = [r for r in rows if r.get("avg_agree_rate", 0) >= GS_MIN_AGREE_RATE]
        pool = filtered if filtered else rows  # fallback if nothing passes

        # Step 2: pick best penalised score
        winner = max(pool, key=lambda r: r["avg_delta"] - GS_LAMBDA * abs(r["alpha"]))
        pen_delta = winner["avg_delta"] - GS_LAMBDA * abs(winner["alpha"])

        # Step 3: direction check
        dim = dim_dir.name
        dir_ok = (winner["alpha"] > 0) if dim in GS_EXPECTED_POSITIVE else True

        should_run = pen_delta >= GS_MIN_DELTA_TO_RUN and dir_ok

        configs[dim] = {
            "alpha":      winner["alpha"],
            "layers":     winner["layer_indices"],
            "pen_delta":  round(pen_delta, 4),
            "dir_ok":     dir_ok,
            "run_eval":   should_run,
            "agree_rate": winner.get("avg_agree_rate", float("nan")),
        }

    return configs


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
        max_new_tokens=60,
        temperature=0.0,
        opening_bid_pct=opening_bid_pct,
    )


# ─── Generic Craigslist: buyer-steered + seller-steered + baselines ───────────

def run_generic_craigslist(model, tokenizer, scenarios, dvecs, alpha,
                           dimension, output_dir):
    """
    Runs four fixed-role batches on the same scenarios (interleaved):
      buyer-steered:   steered=BUYER,  alpha  → {dimension}_buyer_steered.json
      seller-steered:  steered=SELLER, alpha  → {dimension}_seller_steered.json
      buyer-baseline:  steered=BUYER,  alpha=0 → {dimension}_buyer_baseline.json
      seller-baseline: steered=SELLER, alpha=0 → {dimension}_seller_baseline.json

    Matches the gridsearch metric exactly:
      buyer_delta  = buyer_steered_midpt_adv  - buyer_baseline_midpt_adv
      seller_delta = seller_steered_midpt_adv - seller_baseline_midpt_adv
      avg_delta    = (buyer_delta + seller_delta) / 2

    Returns (buyer_steered_games, seller_steered_games,
             buyer_baseline_games, seller_baseline_games)
    """
    prefix = dimension
    log.info("=" * 70)
    log.info("%s Craigslist: role-fixed batches (%d scenarios × 4 batches)",
             dimension, len(scenarios))
    log.info("  alpha=%.3f, layers=%s", alpha, list(dvecs.keys()))
    log.info("=" * 70)

    buyer_steered_games, seller_steered_games = [], []
    buyer_baseline_games, seller_baseline_games = [], []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        try:
            r_buy = run_single_game(model, tokenizer, sc, "buyer", dvecs, alpha)
            r_buy["game_id"] = i
            buyer_steered_games.append(r_buy)

            r_sell = run_single_game(model, tokenizer, sc, "seller", dvecs, alpha)
            r_sell["game_id"] = i
            seller_steered_games.append(r_sell)

            r_base_buy = run_single_game(model, tokenizer, sc, "buyer", None, 0.0)
            r_base_buy["game_id"] = i
            buyer_baseline_games.append(r_base_buy)

            r_base_sell = run_single_game(model, tokenizer, sc, "seller", None, 0.0)
            r_base_sell["game_id"] = i
            seller_baseline_games.append(r_base_sell)

        except Exception as e:
            log.error("%s Craigslist quad %d failed: %s\n%s",
                      dimension, i, e, traceback.format_exc())
            continue

        save_results(buyer_steered_games,   f"{prefix}_buyer_steered",   alpha, output_dir)
        save_results(seller_steered_games,  f"{prefix}_seller_steered",  alpha, output_dir)
        save_results(buyer_baseline_games,  f"{prefix}_buyer_baseline",  0.0,   output_dir)
        save_results(seller_baseline_games, f"{prefix}_seller_baseline", 0.0,   output_dir)

        elapsed = time.time() - t0
        mpd_buy       = r_buy.get("midpoint_deviation",       float("nan"))
        mpd_sell      = r_sell.get("midpoint_deviation",      float("nan"))
        mpd_base_buy  = r_base_buy.get("midpoint_deviation",  float("nan"))
        mpd_base_sell = r_base_sell.get("midpoint_deviation", float("nan"))

        def _run_midpt(games, a, role):
            if not games:
                return float("nan")
            return summarise(games, a).get("by_role", {}).get(role, {}).get(
                "midpoint_advantage", float("nan"))

        rb  = _run_midpt(buyer_steered_games,  alpha, "buyer")
        rs  = _run_midpt(seller_steered_games, alpha, "seller")
        rbb = _run_midpt(buyer_baseline_games, 0.0,   "buyer")
        rbs = _run_midpt(seller_baseline_games, 0.0,  "seller")
        nans = any(v != v for v in [rb, rs, rbb, rbs])
        run_buy_delta  = float("nan") if nans else rb - rbb
        run_sell_delta = float("nan") if nans else rs - rbs
        run_avg_delta  = float("nan") if nans else (run_buy_delta + run_sell_delta) / 2.0

        progress_log.info(
            "Quad %2d/%d [%5.0fs] | "
            "buy_s mpd=%+.3f | sell_s mpd=%+.3f | "
            "buy_b mpd=%+.3f | sell_b mpd=%+.3f | "
            "running buy_delta=%+.4f  sell_delta=%+.4f  avg_delta=%+.4f",
            i + 1, len(scenarios), elapsed,
            mpd_buy, mpd_sell, mpd_base_buy, mpd_base_sell,
            run_buy_delta, run_sell_delta, run_avg_delta,
        )

    def _paired_mpd(steered, baseline):
        return [
            r1["midpoint_deviation"] - r2["midpoint_deviation"]
            for r1, r2 in zip(steered, baseline)
            if r1["agreed"] and r2["agreed"]
            and r1.get("midpoint_deviation") is not None
            and r2.get("midpoint_deviation") is not None
        ]

    log.info("=" * 70)
    for role, steered, baseline in [
        ("BUYER",  buyer_steered_games,  buyer_baseline_games),
        ("SELLER", seller_steered_games, seller_baseline_games),
    ]:
        paired = _paired_mpd(steered, baseline)
        if paired:
            log.info("%s %s STEERING EFFECT (N=%d agreed pairs): "
                     "Δmidpoint_dev Mean=%+.4f  Std=%.4f",
                     dimension, role, len(paired),
                     np.mean(paired), np.std(paired, ddof=1))
    log.info("=" * 70)

    elapsed = time.time() - t0
    log.info("%s Craigslist complete: %d quads in %.0fs",
             dimension, len(buyer_steered_games), elapsed)
    return (buyer_steered_games, seller_steered_games,
            buyer_baseline_games, seller_baseline_games)


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
        description="P4 GPU evaluation suite — gridsearch-driven evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("gridsearch_dir",
                   help="Path to gridsearch results dir "
                        "(e.g. hyperparameter_results/gridsearch_neg15dim_12pairs_matched). "
                        "Alpha and layers are read from stage2_results.json for each dimension.")
    p.add_argument("--model", choices=list(MODELS.keys()), default="qwen2.5-3b")
    p.add_argument("--vectors_dir", required=True,
                   help="Path to vectors dir, e.g. vectors/neg15dim_12pairs_matched/negotiation")
    _root = str(Path(__file__).resolve().parent.parent)
    p.add_argument("--output_dir", default=str(Path(_root) / "results" / "eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "validation"], default="train",
                   help="Craigslist split. 'train' recommended for paper.")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()

    model_cfg = MODELS[args.model]
    hf_token = HF_TOKEN if model_cfg.requires_token else None

    # ─── Load gridsearch configs ───────────────────────────────────────
    gs_configs = load_gridsearch_config(args.gridsearch_dir, model_cfg.alias)
    if not gs_configs:
        log.error("No gridsearch configs found in %s/%s — check path.",
                  args.gridsearch_dir, model_cfg.alias)
        sys.exit(1)

    log.info("=" * 70)
    log.info("GRIDSEARCH AUTO-CONFIG  (%s / %s)", args.gridsearch_dir, model_cfg.alias)
    log.info("  Scoring: penalised_delta = avg_delta - %.3f*|alpha|, "
             "agree_rate >= %.2f, min_delta = %.2f",
             GS_LAMBDA, GS_MIN_AGREE_RATE, GS_MIN_DELTA_TO_RUN)
    log.info("  %-35s %6s  %-12s  %8s  %s", "dimension", "alpha", "layers",
             "pen_delta", "run?")
    for dim, cfg in sorted(gs_configs.items()):
        log.info("  %-35s %+6.1f  %-12s  %+8.3f  %s",
                 dim, cfg["alpha"], str(cfg["layers"]), cfg["pen_delta"],
                 "YES" if cfg["run_eval"] else "SKIP")
    log.info("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress_fh = logging.FileHandler(output_dir / "progress.log")
    _progress_fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    progress_log.addHandler(_progress_fh)
    progress_log.propagate = True  # still shows in main log

    log.info("=" * 70)
    log.info("P4 EVALUATION RUN")
    log.info("  Started:      %s", datetime.now().isoformat())
    log.info("  Model:        %s (%s)", args.model, model_cfg.hf_id)
    log.info("  Gridsearch:   %s", args.gridsearch_dir)
    log.info("  Output:       %s/", output_dir)
    log.info("  Seed:         %d", args.seed)
    log.info("  Split:        %s", args.split)
    n_to_run = sum(1 for cfg in gs_configs.values() if cfg["run_eval"])
    log.info("  Dimensions:   %d total, %d to run, %d to skip",
             len(gs_configs), n_to_run, len(gs_configs) - n_to_run)
    log.info("  Total games:  ~%d", n_to_run * N_CRAIGSLIST * 4)
    log.info("=" * 70)

    t_start = time.time()

    # ─── Load model ───────────────────────────────────────────────────
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

    # ─── Run all valid dimensions ──────────────────────────────────────
    all_sc = _load_all_craigslist(split=args.split)
    gs_results_summary = {}

    manifest = {
        "gridsearch_dir": str(args.gridsearch_dir),
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "scoring": {
            "lambda": GS_LAMBDA,
            "min_agree_rate": GS_MIN_AGREE_RATE,
            "min_delta_to_run": GS_MIN_DELTA_TO_RUN,
        },
        "dimensions": gs_configs,
    }
    with open(output_dir / "gridsearch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for dim, cfg in sorted(gs_configs.items()):
        if not cfg["run_eval"]:
            log.info("SKIP %s  (pen_delta=%.3f  dir_ok=%s)",
                     dim, cfg["pen_delta"], cfg["dir_ok"])
            continue

        dvecs = load_vectors_safe(
            args.vectors_dir, model_cfg.alias,
            dim, "mean_diff", cfg["layers"],
        )
        if dvecs is None:
            log.warning("Vectors not found for %s — skipping.", dim)
            continue

        dim_seed = args.seed + abs(hash(dim)) % 10000
        dim_scenarios = random.Random(dim_seed).sample(
            all_sc, min(N_CRAIGSLIST, len(all_sc))
        )

        dim_outdir = output_dir / dim
        dim_outdir.mkdir(parents=True, exist_ok=True)

        with open(dim_outdir / "scenarios_pinned.json", "w") as f:
            json.dump({
                "dimension": dim, "alpha": cfg["alpha"],
                "layers": cfg["layers"], "pen_delta": cfg["pen_delta"],
                "seed": dim_seed, "split": args.split, "min_span": MIN_SPAN,
                "n": len(dim_scenarios), "scenarios": dim_scenarios,
            }, f, indent=2, ensure_ascii=False)

        progress_log.info("=== %s  alpha=%+.1f  layers=%s ===",
                          dim, cfg["alpha"], cfg["layers"])
        buy_g, sell_g, base_buy_g, base_sell_g = run_generic_craigslist(
            model, tokenizer, dim_scenarios,
            dvecs, cfg["alpha"], dim, dim_outdir,
        )

        # Compute metrics matching the gridsearch exactly:
        #   buyer_delta  = buyer_steered_midpt_adv  - buyer_baseline_midpt_adv
        #   seller_delta = seller_steered_midpt_adv - seller_baseline_midpt_adv
        #   avg_delta    = (buyer_delta + seller_delta) / 2
        def _midpt(games, alpha, role):
            if not games:
                return float("nan")
            return summarise(games, alpha).get("by_role", {}).get(role, {}).get(
                "midpoint_advantage", float("nan"))

        def _agree(games):
            return sum(1 for g in games if g["agreed"]) / len(games) if games else float("nan")

        buy_midpt       = _midpt(buy_g,       cfg["alpha"], "buyer")
        sell_midpt      = _midpt(sell_g,       cfg["alpha"], "seller")
        base_buy_midpt  = _midpt(base_buy_g,  0.0,          "buyer")
        base_sell_midpt = _midpt(base_sell_g, 0.0,          "seller")

        buyer_delta  = buy_midpt  - base_buy_midpt
        seller_delta = sell_midpt - base_sell_midpt
        avg_delta    = (buyer_delta + seller_delta) / 2.0

        buy_agree   = _agree(buy_g)
        sell_agree  = _agree(sell_g)
        avg_agree   = (buy_agree + sell_agree) / 2.0

        def _fmt(v):
            return round(float(v), 4) if not (v != v) else None  # NaN check

        progress_log.info(
            ">>> %s DONE  buy_delta=%+.4f  sell_delta=%+.4f  avg_delta=%+.4f  "
            "agree=%.2f",
            dim, buyer_delta, seller_delta, avg_delta, avg_agree,
        )
        gs_results_summary[dim] = {
            "alpha":                cfg["alpha"],
            "layers":               cfg["layers"],
            "pen_delta_gs":         cfg["pen_delta"],
            "buyer_midpt":          _fmt(buy_midpt),
            "seller_midpt":         _fmt(sell_midpt),
            "buyer_baseline_midpt": _fmt(base_buy_midpt),
            "seller_baseline_midpt":_fmt(base_sell_midpt),
            "buyer_delta":          _fmt(buyer_delta),
            "seller_delta":         _fmt(seller_delta),
            "avg_delta":            _fmt(avg_delta),
            "buyer_agree_rate":     _fmt(buy_agree),
            "seller_agree_rate":    _fmt(sell_agree),
            "avg_agree_rate":       _fmt(avg_agree),
        }

    with open(output_dir / "gridsearch_eval_summary.json", "w") as f:
        json.dump(gs_results_summary, f, indent=2)

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info("GRIDSEARCH EVAL COMPLETE  (%d dimensions, %.0fs total)",
             len(gs_results_summary), elapsed)
    for dim, r in sorted(gs_results_summary.items()):
        log.info("  %-35s alpha=%+.1f  avg_delta=%s  avg_agree=%s  (gs_pen=%.3f)",
                 dim, r["alpha"],
                 f"{r['avg_delta']:+.4f}"   if r["avg_delta"]    is not None else "n/a",
                 f"{r['avg_agree_rate']:.3f}" if r["avg_agree_rate"] is not None else "n/a",
                 r["pen_delta_gs"])
    log.info("=" * 70)

    meta = {
        "model":           args.model,
        "gridsearch_dir":  str(args.gridsearch_dir),
        "seed":            args.seed,
        "split":           args.split,
        "started":         datetime.fromtimestamp(t_start).isoformat(),
        "completed":       datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "results_summary": gs_results_summary,
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Metadata saved to %s/run_meta.json", output_dir)


if __name__ == "__main__":
    main()

