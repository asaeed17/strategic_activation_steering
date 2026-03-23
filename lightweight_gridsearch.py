#!/usr/bin/env python3
"""
lightweight_gridsearch.py

Two-stage grid search focused on a single steering dimension.

Stage 1 — Preset search
    Fix alpha at a probe value (default 5.0). Sweep layer presets
    (early / middle / late). Rank by avg_delta vs baseline.

Stage 2 — Alpha search
    Fix preset to Stage 1 winner. Either:
      (a) Sweep a discrete set of alphas (default: -7, -5, -3, 3, 5, 7), or
      (b) Use TPE (--use_tpe) for continuous 1D optimisation over [alpha_low, alpha_high].
          TPE is more efficient: ~15-20 trials to converge vs N grid points.

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

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

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
# Stage 2 (two-pass variant): coarse then fine alpha search
# ---------------------------------------------------------------------------

def run_stage2_two_pass(
    model, tokenizer,
    scenarios:      List[Dict],
    dvecs:          Dict,
    preset:         str,
    layer_indices:  List[int],
    coarse_alphas:  List[float],
    max_new_tokens: int,
    temperature:    float,
    output_dir:     Path,
) -> Dict:
    """
    Two-pass alpha search:
      Pass 1 — coarse sweep over coarse_alphas (e.g. [5, 15, 25])
      Pass 2 — fine sweep around the coarse peak (+/- [1, 3, 5], filtered > 0)
    Returns the best result across both passes.
    """
    log.info("Stage 2 (two-pass): computing baseline (alpha=0, %d scenarios)...", len(scenarios))
    buy_bl, sell_bl = run_baseline_games(
        model, tokenizer, scenarios, max_new_tokens, temperature
    )
    log.info("  Baseline: buy=%+.4f  sell=%+.4f", buy_bl, sell_bl)

    def _sweep(alphas: List[float], tag: str) -> List[Dict]:
        results = []
        for i, alpha in enumerate(alphas):
            log.info(
                "S2 %s [%d/%d] alpha=%+.2f  preset=%s  layers=%s",
                tag, i + 1, len(alphas), alpha, preset, layer_indices,
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
        return results

    # Pass 1: coarse
    log.info("--- Two-pass Stage 2: COARSE pass %s ---", coarse_alphas)
    coarse_results = _sweep(coarse_alphas, "coarse")
    coarse_results_sorted = sorted(coarse_results, key=lambda r: r["avg_delta"], reverse=True)
    with open(output_dir / "stage2_coarse_results.json", "w") as fh:
        json.dump(coarse_results_sorted, fh, indent=2)
    best_coarse_alpha = coarse_results_sorted[0]["alpha"]
    log.info("Coarse best: alpha=%+.2f  avg_delta=%+.4f", best_coarse_alpha, coarse_results_sorted[0]["avg_delta"])

    all_results = coarse_results_sorted
    best = all_results[0]
    with open(output_dir / "stage2_results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)

    # Print table
    print("\n" + "=" * 95)
    print(f"STAGE 2 — COARSE ALPHA SEARCH  (preset='{preset}', layers={layer_indices}, {len(scenarios)} scenarios)")
    print(f"  Baseline: buy={buy_bl:+.4f}  sell={sell_bl:+.4f}")
    print(f"  Coarse alphas: {coarse_alphas}  →  best={best_coarse_alpha:+.2f}")
    print("=" * 95)
    print(f"{'Rank':>4}  {'Alpha':>7}  "
          f"{'AvgDelta':>9}  {'BuyDelta':>9}  {'SellDelta':>10}  "
          f"{'BuyMidpt':>9}  {'SellMidpt':>10}  {'Agree':>6}")
    print("-" * 95)
    for rank, r in enumerate(all_results, 1):
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
# Stage 2 (TPE variant): continuous 1D alpha optimisation
# ---------------------------------------------------------------------------

def run_stage2_tpe(
    model, tokenizer,
    scenarios:      List[Dict],
    dvecs:          Dict,
    preset:         str,
    layer_indices:  List[int],
    alpha_low:      float,
    alpha_high:     float,
    n_trials:       int,
    n_startup:      int,
    max_new_tokens: int,
    temperature:    float,
    output_dir:     Path,
    seed:           int,
) -> Dict:
    """
    1D TPE search over alpha in [alpha_low, alpha_high].
    Baseline is computed once before the study and subtracted in the objective.
    More efficient than grid search: TPE focuses evaluations near promising regions.
    """
    log.info("Stage 2 TPE: computing baseline (alpha=0, %d scenarios)...", len(scenarios))
    buy_bl, sell_bl = run_baseline_games(
        model, tokenizer, scenarios, max_new_tokens, temperature
    )
    log.info("  Baseline: buy=%+.4f  sell=%+.4f", buy_bl, sell_bl)

    trial_log = []

    study = optuna.create_study(
        study_name=f"s2_tpe_{preset}",
        direction="maximize",
        sampler=TPESampler(n_startup_trials=n_startup, seed=seed),
    )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", alpha_low, alpha_high)
        r = eval_config(
            model, tokenizer, scenarios, dvecs, alpha,
            max_new_tokens, temperature, buy_bl, sell_bl,
        )
        trial_log.append({**r, "preset": preset, "layer_indices": layer_indices,
                           "trial": trial.number})
        log.info(
            "  TPE trial %2d | alpha=%+6.3f  avg_delta=%+.4f  "
            "buy_delta=%+.4f  sell_delta=%+.4f  agree=%.0f%%",
            trial.number, alpha, r["avg_delta"],
            r["buyer_delta"], r["seller_delta"], r["avg_agree_rate"] * 100,
        )
        return r["avg_delta"]

    study.optimize(objective, n_trials=n_trials)

    # Sort log by avg_delta for the table
    trial_log_sorted = sorted(trial_log, key=lambda r: r["avg_delta"], reverse=True)
    best = trial_log_sorted[0]

    with open(output_dir / "stage2_results.json", "w") as fh:
        json.dump(trial_log_sorted, fh, indent=2)

    print("\n" + "=" * 105)
    print(f"STAGE 2 — TPE ALPHA SEARCH  "
          f"(preset='{preset}', layers={layer_indices}, "
          f"range=[{alpha_low},{alpha_high}], {n_trials} trials, {len(scenarios)} scenarios)")
    print(f"  Baseline: buy={buy_bl:+.4f}  sell={sell_bl:+.4f}")
    print("=" * 105)
    print(f"{'Rank':>4}  {'Alpha':>7}  "
          f"{'AvgDelta':>9}  {'BuyDelta':>9}  {'SellDelta':>10}  "
          f"{'BuyMidpt':>9}  {'SellMidpt':>10}  {'Agree':>6}")
    print("-" * 105)
    for rank, r in enumerate(trial_log_sorted, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['alpha']:>+7.3f}  "
            f"{r['avg_delta']:>+9.4f}  {r['buyer_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  "
            f"{r['buyer_midpt']:>+9.4f}  {r['seller_midpt']:>+10.4f}  "
            f"{r['avg_agree_rate']:>5.0%}{marker}"
        )
    print("=" * 105)
    print(f"\nStage 2 TPE winner: alpha={best['alpha']:+.4f}  avg_delta={best['avg_delta']:+.4f}")

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
    p.add_argument("--vectors_dir", default="vectors/neg8dim_12pairs_matched/negotiation")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")

    # Presets to sweep in Stage 1
    p.add_argument("--presets", nargs="+",
                   choices=list(LAYER_PRESET_FRACTIONS.keys()),
                   default=["early", "middle", "late"],
                   help="Layer presets to sweep in Stage 1. Default: early middle late.")
    p.add_argument("--fixed_layers", nargs="+", type=int, default=None,
                   help="Skip Stage 1 and use these exact layer indices for Stage 2. "
                        "E.g. --fixed_layers 21")

    # Alpha
    p.add_argument("--probe_alpha", type=float, default=5.0,
                   help="Fixed alpha used in Stage 1 preset sweep. Default: 5.0.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                   help="Discrete alphas to try in Stage 2 (grid mode). Default: 5 10 15 20 25 30.")

    # Two-pass alpha search
    p.add_argument("--two_pass", action="store_true",
                   help="Two-pass Stage 2: coarse sweep (--coarse_alphas), then fine sweep "
                        "around the peak. Overrides --alphas for Stage 2.")
    p.add_argument("--coarse_alphas", nargs="+", type=float,
                   default=[5.0, 15.0, 25.0],
                   help="Coarse alpha grid for first pass of --two_pass. Default: 5 15 25.")

    # TPE mode for Stage 2
    p.add_argument("--use_tpe", action="store_true",
                   help="Use TPE (continuous) instead of discrete grid for Stage 2 alpha search.")
    p.add_argument("--alpha_low",  type=float, default=-8.0,
                   help="TPE search lower bound for alpha. Default: -8.0.")
    p.add_argument("--alpha_high", type=float, default=8.0,
                   help="TPE search upper bound for alpha. Default: 8.0.")
    p.add_argument("--n_trials",   type=int,   default=20,
                   help="Number of TPE trials in Stage 2. Default: 20.")
    p.add_argument("--n_startup",  type=int,   default=8,
                   help="Random startup trials before TPE exploits. Default: 8.")

    # Games
    p.add_argument("--s1_games", type=int, default=20,
                   help="Scenarios per preset in Stage 1. Default: 20.")
    p.add_argument("--s2_games", type=int, default=20,
                   help="Scenarios per alpha in Stage 2. Default: 20.")

    # Generation
    p.add_argument("--temperature",     type=float, default=0.0,
                   help="Decoding temperature. 0=greedy. Default: 0.0.")
    p.add_argument("--max_new_tokens",  type=int,   default=60,
                   help="Max tokens per turn. Default: 60.")

    # Scenario filtering
    p.add_argument("--min_span", type=int, default=200,
                   help="Minimum price span (listing_price - buyer_target) to keep a scenario. "
                        "Filters degenerate near-zero-span games. Default: 200.")

    p.add_argument("--output_suffix", type=str, default="",
                   help="Optional suffix appended to the auto-derived output dir. "
                        "E.g. '_v2' → hyperparameter_results/gridsearch_..._v2/...")

    # Misc
    p.add_argument("--dataset_split",  choices=["train", "validation"], default="train")
    p.add_argument("--dtype",          choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--quantize",       action="store_true",
                   help="Load model in 4-bit (bitsandbytes NF4). Reduces VRAM to ~4GB for 7B.")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--use_craigslist", action="store_true", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Derive from vectors_dir: .../neg15dim_12pairs_matched/negotiation
    #  → hyperparameter_results/gridsearch_neg15dim_12pairs_matched/<model>/<dimension>
    vec_set = Path(args.vectors_dir).parts[-2]
    output_dir = Path("hyperparameter_results") / f"gridsearch_{vec_set}{args.output_suffix}" / args.model / args.dimension
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_cfg = MODELS[args.model]
    hf_token  = HF_TOKEN if model_cfg.requires_token else None
    vectors_dir = Path(args.vectors_dir)

    metadata = load_metadata(vectors_dir, model_cfg.alias)
    n_layers = metadata["n_layers"]
    log.info("Model %s | n_layers=%d", model_cfg.hf_id, n_layers)

    # Resolve layers: --fixed_layers skips Stage 1 entirely
    skip_stage1  = args.fixed_layers is not None
    fixed_layers = sorted(set(args.fixed_layers)) if skip_stage1 else None

    # Load enough scenarios for both stages
    max_games = args.s2_games if skip_stage1 else max(args.s1_games, args.s2_games)
    all_scenarios = load_craigslist(split=args.dataset_split, num_samples=max_games, min_span=args.min_span)
    s1_scenarios  = all_scenarios[: args.s1_games]
    s2_scenarios  = all_scenarios[: args.s2_games]
    log.info("Loaded %d scenarios (s2=%d%s)",
             len(all_scenarios), len(s2_scenarios),
             ", Stage 1 skipped" if skip_stage1 else f", s1={len(s1_scenarios)}")

    # Pre-load vectors for every preset we'll need in Stage 1,
    # OR just the fixed layers if Stage 1 is skipped.
    log.info("Pre-loading vectors: dim=%s  method=%s", args.dimension, args.method)
    if skip_stage1:
        try:
            fixed_dvecs = load_direction_vectors(
                vectors_dir=vectors_dir, model_alias=model_cfg.alias,
                dimension=args.dimension, method=args.method, layer_indices=fixed_layers,
            )
            log.info("  Loaded fixed layers=%s", fixed_layers)
        except FileNotFoundError as exc:
            log.error("Vectors not found for layers=%s: %s", fixed_layers, exc)
            sys.exit(1)
        dvecs_by_preset = {}
    else:
        fixed_dvecs = None
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
    log.info("Loading model: %s%s", model_cfg.hf_id, " [4-bit]" if args.quantize else "")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id, token=hf_token, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(token=hf_token, device_map="auto")
    if args.quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        load_kwargs["max_memory"] = {0: "15GiB"}   # pure GPU, no CPU offload

    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, **load_kwargs)
    model.eval()

    total_s2 = len(args.alphas) * args.s2_games * 2 + args.s2_games * 2  # +baseline

    print("\n" + "=" * 80)
    print(f"LIGHTWEIGHT GRID SEARCH  |  model={args.model}  dim={args.dimension}")
    if skip_stage1:
        print(f"  Stage 1: SKIPPED  (fixed layers={fixed_layers})")
    else:
        total_s1 = len(args.presets) * args.s1_games * 2 + args.s1_games * 2
        print(f"  Stage 1: {len(args.presets)} presets × {args.s1_games} scenarios × 2 roles"
              f"  (probe_alpha={args.probe_alpha:+.1f})  ~{total_s1} games")
    if args.use_tpe:
        total_s2 = args.n_trials * args.s2_games * 2 + args.s2_games * 2
        print(f"  Stage 2: TPE {args.n_trials} trials × {args.s2_games} scenarios × 2 roles"
              f"  (range=[{args.alpha_low}, {args.alpha_high}])  ~{total_s2} games")
    elif args.two_pass:
        total_s2 = len(args.coarse_alphas) * args.s2_games * 2 + args.s2_games * 2
        print(f"  Stage 2: coarse-only  alphas={args.coarse_alphas}"
              f"  ~{total_s2} games")
    else:
        print(f"  Stage 2: {len(args.alphas)} alphas × {args.s2_games} scenarios × 2 roles"
              f"  (alphas={args.alphas})  ~{total_s2} games")
    print(f"  temp={args.temperature}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1 — Preset search (skipped if --fixed_layers provided)
    # =========================================================================
    if skip_stage1:
        best_preset = "fixed"
        best_layers = fixed_layers
        best_dvecs  = fixed_dvecs
        log.info("Stage 1 skipped — using fixed layers=%s", best_layers)
    else:
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
        best_layers = layers_from_preset(n_layers, best_preset)
        best_dvecs  = dvecs_by_preset[best_preset]

    # =========================================================================
    # STAGE 2 — Alpha search (two-pass, grid, or TPE)
    # =========================================================================
    if args.two_pass:
        best = run_stage2_two_pass(
            model=model, tokenizer=tokenizer,
            scenarios=s2_scenarios,
            dvecs=best_dvecs,
            preset=best_preset,
            layer_indices=best_layers,
            coarse_alphas=args.coarse_alphas,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_dir=output_dir,
        )
    elif args.use_tpe:
        if not _OPTUNA_AVAILABLE:
            log.error("--use_tpe requires optuna: pip install optuna")
            sys.exit(1)
        best = run_stage2_tpe(
            model=model, tokenizer=tokenizer,
            scenarios=s2_scenarios,
            dvecs=best_dvecs,
            preset=best_preset,
            layer_indices=best_layers,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            n_trials=args.n_trials,
            n_startup=args.n_startup,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_dir=output_dir,
            seed=args.seed,
        )
    else:
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
            "model":        args.model,
            "s1_games":     args.s1_games,
            "s2_games":     args.s2_games,
            "probe_alpha":  args.probe_alpha,
            "alphas":       args.coarse_alphas if args.two_pass else args.alphas,
            "two_pass":     args.two_pass,
            "min_span":     args.min_span,
            "dataset_split": args.dataset_split,
            "temperature":  args.temperature,
            "seed":         args.seed,
            "timestamp":    datetime.now().isoformat(),
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
