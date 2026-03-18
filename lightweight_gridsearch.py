#!/usr/bin/env python3
"""
lightweight_gridsearch.py

Two-stage grid search focused on a single steering dimension.

Stage 1 — Preset search
    Fix alpha at a probe value (default 5.0). Sweep layer presets
    (early / middle / late). Rank by avg_delta vs baseline.

Stage 2 — Alpha search
    Fix preset to Stage 1 winner. Sweep a discrete set of alphas
    (default: -7, -5, -3, 3, 5, 7). Rank by avg_delta vs baseline.

Metric (identical to fast_search_steering.py):
    buyer_delta  = steered_buyer_midpt_adv  - baseline_buyer_midpt_adv
    seller_delta = steered_seller_midpt_adv - baseline_seller_midpt_adv
    objective    = (buyer_delta + seller_delta) / 2

    Baseline = alpha=0 (both sides unsteered), same scenarios,
    run ONCE per stage and reused across all configs/alphas.

Usage:
    python lightweight_gridsearch.py \\
        --model qwen2.5-3b \\
        --dimension firmness \\
        --vectors_dir vectors/neg8dim_12pairs_matched/negotiation \\
        --use_craigslist \\
        --output_dir results/lightweight_firmness
"""

import json
import logging
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering import load_craigslist, load_direction_vectors, run_game, summarise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer presets (same as fast_search_steering.py)
# ---------------------------------------------------------------------------

LAYER_PRESET_FRACTIONS: Dict[str, List[float]] = {
    "early":        [0.25],
    "middle":       [0.50],
    "late":         [0.75],
    "early_middle": [0.25, 0.50],
    "middle_late":  [0.50, 0.75],
    "spread":       [0.25, 0.50, 0.75],
}


def layers_from_preset(n_layers: int, preset: str) -> List[int]:
    return sorted(set(int(n_layers * f) for f in LAYER_PRESET_FRACTIONS[preset]))


def load_metadata(vectors_dir: Path, model_alias: str) -> Dict:
    path = vectors_dir / model_alias / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found at {path}. Run extract_vectors.py first.")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Game runners (identical logic to fast_search_steering.py)
# ---------------------------------------------------------------------------

def _run_role_games(
    model,
    tokenizer,
    scenarios:      List[Dict],
    dvecs,          # Dict[int, np.ndarray] or None for baseline
    alpha:          float,
    max_new_tokens: int,
    temperature:    float,
    role:           str,
) -> Tuple[float, float]:
    """
    Run all scenarios with the steered agent fixed to `role`.
    The opponent is always unsteered (alpha=0).
    dvecs=None / alpha=0.0 → fully unsteered baseline.
    Returns (steered_midpoint_advantage_mean, agree_rate).
    """
    games = []
    for sc in scenarios:
        result = run_game(
            model=model,
            tokenizer=tokenizer,
            scenario=sc,
            dvecs_seller=dvecs if role == "seller" else None,
            alpha_seller=alpha  if role == "seller" else 0.0,
            dvecs_buyer =dvecs if role == "buyer"   else None,
            alpha_buyer =alpha  if role == "buyer"  else 0.0,
            steered_role=role,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        games.append(result)
    summary = summarise(games, alpha)
    role_s = summary["by_role"][role]
    return role_s["midpoint_advantage"], role_s["agree_rate"]


def run_baseline_games(
    model,
    tokenizer,
    scenarios:      List[Dict],
    max_new_tokens: int,
    temperature:    float,
) -> Tuple[float, float]:
    """
    Run alpha=0 (fully unsteered) games for both roles.
    Returns (buy_baseline_midpt_adv, sell_baseline_midpt_adv).
    Call once per stage and reuse — result is constant for fixed scenarios.
    """
    buy_bl,  _ = _run_role_games(
        model, tokenizer, scenarios, dvecs=None, alpha=0.0,
        max_new_tokens=max_new_tokens, temperature=temperature, role="buyer",
    )
    sell_bl, _ = _run_role_games(
        model, tokenizer, scenarios, dvecs=None, alpha=0.0,
        max_new_tokens=max_new_tokens, temperature=temperature, role="seller",
    )
    return buy_bl, sell_bl


def eval_config(
    model,
    tokenizer,
    scenarios:           List[Dict],
    dvecs:               Dict,
    alpha:               float,
    max_new_tokens:      int,
    temperature:         float,
    buy_baseline_midpt:  float,
    sell_baseline_midpt: float,
) -> Dict:
    """
    Run steered buyer and seller games, subtract baseline, return result dict.
    """
    buy_midpt,  buy_agree  = _run_role_games(
        model, tokenizer, scenarios, dvecs, alpha, max_new_tokens, temperature, "buyer"
    )
    sell_midpt, sell_agree = _run_role_games(
        model, tokenizer, scenarios, dvecs, alpha, max_new_tokens, temperature, "seller"
    )
    buy_delta  = buy_midpt  - buy_baseline_midpt
    sell_delta = sell_midpt - sell_baseline_midpt
    avg_delta  = (buy_delta + sell_delta) / 2.0
    avg_agree  = (buy_agree + sell_agree) / 2.0
    return {
        "alpha":                 alpha,
        "avg_delta":             avg_delta,
        "buyer_delta":           buy_delta,
        "seller_delta":          sell_delta,
        "buyer_midpt":           buy_midpt,
        "seller_midpt":          sell_midpt,
        "buyer_baseline_midpt":  buy_baseline_midpt,
        "seller_baseline_midpt": sell_baseline_midpt,
        "avg_agree_rate":        avg_agree,
        "buyer_agree_rate":      buy_agree,
        "seller_agree_rate":     sell_agree,
    }


# ---------------------------------------------------------------------------
# Stage 1: preset search (fixed alpha)
# ---------------------------------------------------------------------------

def run_stage1(
    model, tokenizer,
    scenarios:      List[Dict],
    dvecs_by_preset: Dict[str, Dict],   # preset → loaded dvecs
    presets:        List[str],
    probe_alpha:    float,
    max_new_tokens: int,
    temperature:    float,
    n_layers:       int,
    output_dir:     Path,
) -> str:
    """
    Sweep presets at fixed probe_alpha. Returns the winning preset name.
    """
    log.info("Stage 1: computing baseline (alpha=0, %d scenarios)...", len(scenarios))
    buy_bl, sell_bl = run_baseline_games(
        model, tokenizer, scenarios, max_new_tokens, temperature
    )
    log.info("  Baseline: buy=%+.4f  sell=%+.4f", buy_bl, sell_bl)

    results = []
    for i, preset in enumerate(presets):
        layers = layers_from_preset(n_layers, preset)
        dvecs  = dvecs_by_preset[preset]
        log.info(
            "S1 [%d/%d] preset=%-13s layers=%s  alpha=%.2f",
            i + 1, len(presets), preset, layers, probe_alpha,
        )
        r = eval_config(
            model, tokenizer, scenarios, dvecs, probe_alpha,
            max_new_tokens, temperature, buy_bl, sell_bl,
        )
        r["preset"]       = preset
        r["layer_indices"] = layers
        results.append(r)
        log.info(
            "  avg_delta=%+.4f  buy_delta=%+.4f  sell_delta=%+.4f  agree=%.0f%%",
            r["avg_delta"], r["buyer_delta"], r["seller_delta"], r["avg_agree_rate"] * 100,
        )

    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    best_preset = results[0]["preset"]

    # Save
    with open(output_dir / "stage1_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Print table
    print("\n" + "=" * 95)
    print(f"STAGE 1 — PRESET SEARCH  (alpha={probe_alpha:.2f}, {len(scenarios)} scenarios)")
    print(f"  Baseline: buy={buy_bl:+.4f}  sell={sell_bl:+.4f}")
    print("=" * 95)
    print(f"{'Rank':>4}  {'Preset':<15}  {'Layers':<12}  "
          f"{'AvgDelta':>9}  {'BuyDelta':>9}  {'SellDelta':>10}  "
          f"{'BuyMidpt':>9}  {'SellMidpt':>10}  {'Agree':>6}")
    print("-" * 95)
    for rank, r in enumerate(results, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['preset']:<15}  {str(r['layer_indices']):<12}  "
            f"{r['avg_delta']:>+9.4f}  {r['buyer_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  "
            f"{r['buyer_midpt']:>+9.4f}  {r['seller_midpt']:>+10.4f}  "
            f"{r['avg_agree_rate']:>5.0%}{marker}"
        )
    print("=" * 95)
    print(f"\nStage 1 winner: preset='{best_preset}'  (layers={results[0]['layer_indices']})")

    return best_preset


# ---------------------------------------------------------------------------
# Stage 2: alpha search (fixed preset)
# ---------------------------------------------------------------------------

def run_stage2(
    model, tokenizer,
    scenarios:      List[Dict],
    dvecs:          Dict,
    preset:         str,
    layer_indices:  List[int],
    alphas:         List[float],
    max_new_tokens: int,
    temperature:    float,
    output_dir:     Path,
) -> Dict:
    """
    Sweep discrete alphas with the fixed best preset. Returns best result dict.
    """
    log.info("Stage 2: computing baseline (alpha=0, %d scenarios)...", len(scenarios))
    buy_bl, sell_bl = run_baseline_games(
        model, tokenizer, scenarios, max_new_tokens, temperature
    )
    log.info("  Baseline: buy=%+.4f  sell=%+.4f", buy_bl, sell_bl)

    results = []
    for i, alpha in enumerate(alphas):
        log.info(
            "S2 [%d/%d] alpha=%+.2f  preset=%s  layers=%s",
            i + 1, len(alphas), alpha, preset, layer_indices,
        )
        r = eval_config(
            model, tokenizer, scenarios, dvecs, alpha,
            max_new_tokens, temperature, buy_bl, sell_bl,
        )
        r["preset"]        = preset
        r["layer_indices"] = layer_indices
        results.append(r)
        log.info(
            "  avg_delta=%+.4f  buy_delta=%+.4f  sell_delta=%+.4f  agree=%.0f%%",
            r["avg_delta"], r["buyer_delta"], r["seller_delta"], r["avg_agree_rate"] * 100,
        )

    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    best = results[0]

    # Save
    with open(output_dir / "stage2_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Print table
    print("\n" + "=" * 95)
    print(f"STAGE 2 — ALPHA SEARCH  (preset='{preset}', layers={layer_indices}, {len(scenarios)} scenarios)")
    print(f"  Baseline: buy={buy_bl:+.4f}  sell={sell_bl:+.4f}")
    print("=" * 95)
    print(f"{'Rank':>4}  {'Alpha':>7}  "
          f"{'AvgDelta':>9}  {'BuyDelta':>9}  {'SellDelta':>10}  "
          f"{'BuyMidpt':>9}  {'SellMidpt':>10}  {'Agree':>6}")
    print("-" * 95)
    for rank, r in enumerate(results, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['alpha']:>+7.2f}  "
            f"{r['avg_delta']:>+9.4f}  {r['buyer_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  "
            f"{r['buyer_midpt']:>+9.4f}  {r['seller_midpt']:>+10.4f}  "
            f"{r['avg_agree_rate']:>5.0%}{marker}"
        )
    print("=" * 95)
    print(f"\nStage 2 winner: alpha={best['alpha']:+.2f}  avg_delta={best['avg_delta']:+.4f}")

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lightweight two-stage grid search for a single steering dimension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",       choices=list(MODELS.keys()), required=True)
    p.add_argument("--dimension",   default="firmness",
                   help="Steering dimension to search over. Default: firmness.")
    p.add_argument("--vectors_dir", default="vectors")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")

    # Presets to sweep in Stage 1
    p.add_argument("--presets", nargs="+",
                   choices=list(LAYER_PRESET_FRACTIONS.keys()),
                   default=["early", "middle", "late"],
                   help="Layer presets to sweep in Stage 1. Default: early middle late.")

    # Alpha
    p.add_argument("--probe_alpha", type=float, default=5.0,
                   help="Fixed alpha used in Stage 1 preset sweep. Default: 5.0.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-7.0, -5.0, -3.0, 3.0, 5.0, 7.0],
                   help="Discrete alphas to try in Stage 2. Default: -7 -5 -3 3 5 7.")

    # Games
    p.add_argument("--s1_games", type=int, default=10,
                   help="Scenarios per preset in Stage 1. Default: 10.")
    p.add_argument("--s2_games", type=int, default=10,
                   help="Scenarios per alpha in Stage 2. Default: 10.")

    # Generation
    p.add_argument("--temperature",     type=float, default=0.0,
                   help="Decoding temperature. 0=greedy. Default: 0.0.")
    p.add_argument("--max_new_tokens",  type=int,   default=60,
                   help="Max tokens per turn. Default: 60.")

    # Misc
    p.add_argument("--dataset_split",  choices=["train", "validation"], default="train")
    p.add_argument("--dtype",          choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--output_dir",     default="results/lightweight")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--use_craigslist", action="store_true", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_cfg = MODELS[args.model]
    hf_token  = HF_TOKEN if model_cfg.requires_token else None
    vectors_dir = Path(args.vectors_dir)

    metadata = load_metadata(vectors_dir, model_cfg.alias)
    n_layers = metadata["n_layers"]
    log.info("Model %s | n_layers=%d", model_cfg.hf_id, n_layers)

    # Load enough scenarios for both stages
    max_games = max(args.s1_games, args.s2_games)
    all_scenarios = load_craigslist(split=args.dataset_split, num_samples=max_games)
    s1_scenarios  = all_scenarios[: args.s1_games]
    s2_scenarios  = all_scenarios[: args.s2_games]
    log.info("Loaded %d scenarios (s1=%d, s2=%d)", len(all_scenarios), len(s1_scenarios), len(s2_scenarios))

    # Pre-load vectors for every preset we'll need
    log.info("Pre-loading vectors: dim=%s  method=%s", args.dimension, args.method)
    dvecs_by_preset: Dict[str, Dict] = {}
    for preset in args.presets:
        layers = layers_from_preset(n_layers, preset)
        try:
            dvecs_by_preset[preset] = load_direction_vectors(
                vectors_dir=vectors_dir, model_alias=model_cfg.alias,
                dimension=args.dimension, method=args.method, layer_indices=layers,
            )
            log.info("  Loaded preset=%-13s layers=%s", preset, layers)
        except FileNotFoundError as exc:
            log.error("Vectors not found for preset=%s: %s", preset, exc)
            sys.exit(1)

    # Load model
    log.info("Loading model: %s", model_cfg.hf_id)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id, token=hf_token, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id, token=hf_token,
        torch_dtype=dtype_map[args.dtype], device_map="auto",
    )
    model.eval()

    total_s1 = len(args.presets) * args.s1_games * 2 + args.s1_games * 2  # +baseline
    total_s2 = len(args.alphas)  * args.s2_games * 2 + args.s2_games * 2  # +baseline

    print("\n" + "=" * 80)
    print(f"LIGHTWEIGHT GRID SEARCH  |  model={args.model}  dim={args.dimension}")
    print(f"  Stage 1: {len(args.presets)} presets × {args.s1_games} scenarios × 2 roles"
          f"  (probe_alpha={args.probe_alpha:+.1f})  ~{total_s1} games")
    print(f"  Stage 2: {len(args.alphas)} alphas × {args.s2_games} scenarios × 2 roles"
          f"  (alphas={args.alphas})  ~{total_s2} games")
    print(f"  Total: ~{total_s1 + total_s2} games  |  temp={args.temperature}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1 — Preset search
    # =========================================================================
    best_preset = run_stage1(
        model=model, tokenizer=tokenizer,
        scenarios=s1_scenarios,
        dvecs_by_preset=dvecs_by_preset,
        presets=args.presets,
        probe_alpha=args.probe_alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_layers=n_layers,
        output_dir=output_dir,
    )

    # =========================================================================
    # STAGE 2 — Alpha search
    # =========================================================================
    best_layers = layers_from_preset(n_layers, best_preset)
    best_dvecs  = dvecs_by_preset[best_preset]

    best = run_stage2(
        model=model, tokenizer=tokenizer,
        scenarios=s2_scenarios,
        dvecs=best_dvecs,
        preset=best_preset,
        layer_indices=best_layers,
        alphas=args.alphas,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=output_dir,
    )

    # Save final summary
    final = {
        "dimension":    args.dimension,
        "method":       args.method,
        "best_preset":  best_preset,
        "best_layers":  best_layers,
        "best_alpha":   best["alpha"],
        "avg_delta":    best["avg_delta"],
        "buyer_delta":  best["buyer_delta"],
        "seller_delta": best["seller_delta"],
        "buyer_midpt":  best["buyer_midpt"],
        "seller_midpt": best["seller_midpt"],
        "avg_agree_rate": best["avg_agree_rate"],
        "run_info": {
            "model":       args.model,
            "s1_games":    args.s1_games,
            "s2_games":    args.s2_games,
            "probe_alpha": args.probe_alpha,
            "alphas":      args.alphas,
            "temperature": args.temperature,
            "seed":        args.seed,
            "timestamp":   datetime.now().isoformat(),
        },
    }
    with open(output_dir / "final_best.json", "w") as fh:
        json.dump(final, fh, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BEST CONFIG")
    print(f"  Dimension : {args.dimension}")
    print(f"  Method    : {args.method}")
    print(f"  Preset    : {best_preset}  →  layers {best_layers}")
    print(f"  Alpha     : {best['alpha']:+.2f}")
    print(f"  AvgDelta  : {best['avg_delta']:+.4f}")
    print(f"  BuyDelta  : {best['buyer_delta']:+.4f}  (midpt {best['buyer_midpt']:+.4f})")
    print(f"  SellDelta : {best['seller_delta']:+.4f}  (midpt {best['seller_midpt']:+.4f})")
    print(f"  Agree     : {best['avg_agree_rate']:.0%}")
    print(f"  Saved to  : {output_dir}/final_best.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
