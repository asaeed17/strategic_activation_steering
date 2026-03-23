#!/usr/bin/env python3
"""
lightweight_gridsearch_preset_nego.py

Two-stage grid search for the preset negotiation benchmark.

Uses the 40 fixed scenarios in preset_negotiators/scenarios.json — no
random sampling, no CraigslistBargains. The opponent is scripted so all
noise comes from the steered model alone.

Metric:
    seller_norm = total_value_earned / sum(listing_prices_seller)
    buyer_norm  = 1 - (total_value_spent - min_possible) / (max_possible - min_possible)
    combined    = (seller_norm + buyer_norm) / 2

    seller_delta = steered_seller_norm - baseline_seller_norm
    buyer_delta  = steered_buyer_norm  - baseline_buyer_norm
    avg_delta    = (seller_delta + buyer_delta) / 2          ← optimisation target

Stage 1 — Preset search
    Fix alpha at probe_alpha. Sweep layer presets (early / middle / late / ...).
    Rank by avg_delta.

Stage 2 — Alpha search
    Fix preset to Stage 1 winner. Sweep discrete alphas.
    Optionally use --two_pass for a coarse sweep only.

Usage:
    python lightweight_gridsearch_preset_nego.py \\
        --model qwen2.5-3b \\
        --dimension firmness \\
        --vectors_dir vectors/neg8dim_12pairs_matched/negotiation \\
        --output_dir results/preset_gridsearch_firmness
"""

import json
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering_preset_nego import (
    load_preset_scenarios,
    load_direction_vectors,
    run_all_scenarios,
    summarise_preset,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer presets
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
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(
    model,
    tokenizer,
    scenarios:      List[Dict],
    max_new_tokens: int,
    temperature:    float,
) -> Tuple[float, float]:
    """
    Run all scenarios unsteered (alpha=0). Returns (seller_norm, buyer_norm).
    Call once per stage and cache — result is constant for fixed scenarios.
    """
    log.info("Computing baseline (alpha=0, %d scenarios)...", len(scenarios))
    results = run_all_scenarios(
        model, tokenizer, scenarios, dvecs=None, alpha=0.0,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    summary = summarise_preset(results)
    log.info(
        "  Baseline: seller_norm=%+.4f  buyer_norm=%+.4f  combined=%+.4f",
        summary["seller_norm"], summary["buyer_norm"], summary["combined_score"],
    )
    return summary["seller_norm"], summary["buyer_norm"]


# ---------------------------------------------------------------------------
# Single config evaluator
# ---------------------------------------------------------------------------

def eval_config(
    model,
    tokenizer,
    scenarios:            List[Dict],
    dvecs:                Dict,
    alpha:                float,
    max_new_tokens:       int,
    temperature:          float,
    baseline_seller_norm: float,
    baseline_buyer_norm:  float,
) -> Dict:
    """
    Run all scenarios with given dvecs/alpha, subtract baseline, return result dict.
    """
    results = run_all_scenarios(
        model, tokenizer, scenarios, dvecs, alpha,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    summary = summarise_preset(results)

    seller_norm  = summary["seller_norm"]
    buyer_norm   = summary["buyer_norm"]
    seller_delta = seller_norm  - baseline_seller_norm
    buyer_delta  = buyer_norm   - baseline_buyer_norm
    avg_delta    = (seller_delta + buyer_delta) / 2.0

    return {
        "alpha":              alpha,
        "avg_delta":          round(avg_delta,    4),
        "seller_delta":       round(seller_delta, 4),
        "buyer_delta":        round(buyer_delta,  4),
        "seller_norm":        round(seller_norm,  4),
        "buyer_norm":         round(buyer_norm,   4),
        "combined_score":     round(summary["combined_score"], 4),
        "seller_earned":      summary["seller"]["total_earned"],
        "buyer_spent":        summary["buyer"]["total_spent"],
        "seller_deal_rate":   summary["seller"]["deal_rate"],
        "buyer_deal_rate":    summary["buyer"]["deal_rate"],
        "baseline_seller_norm": round(baseline_seller_norm, 4),
        "baseline_buyer_norm":  round(baseline_buyer_norm,  4),
    }


# ---------------------------------------------------------------------------
# Stage 1: preset search
# ---------------------------------------------------------------------------

def run_stage1(
    model, tokenizer,
    scenarios:       List[Dict],
    dvecs_by_preset: Dict[str, Dict],
    presets:         List[str],
    probe_alpha:     float,
    max_new_tokens:  int,
    temperature:     float,
    n_layers:        int,
    output_dir:      Path,
) -> str:
    log.info("Stage 1: preset search  (%d presets, probe_alpha=%+.1f)", len(presets), probe_alpha)
    sell_bl, buy_bl = run_baseline(model, tokenizer, scenarios, max_new_tokens, temperature)

    results = []
    for i, preset in enumerate(presets):
        layers = layers_from_preset(n_layers, preset)
        dvecs  = dvecs_by_preset[preset]
        log.info("S1 [%d/%d] preset=%-13s layers=%s", i + 1, len(presets), preset, layers)
        r = eval_config(
            model, tokenizer, scenarios, dvecs, probe_alpha,
            max_new_tokens, temperature, sell_bl, buy_bl,
        )
        r["preset"]        = preset
        r["layer_indices"] = layers
        results.append(r)
        log.info(
            "  avg_delta=%+.4f  sell_delta=%+.4f  buy_delta=%+.4f  "
            "sell_deal=%.0f%%  buy_deal=%.0f%%",
            r["avg_delta"], r["seller_delta"], r["buyer_delta"],
            r["seller_deal_rate"] * 100, r["buyer_deal_rate"] * 100,
        )

    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    best_preset = results[0]["preset"]

    with open(output_dir / "stage1_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 100)
    print(f"STAGE 1 — PRESET SEARCH  (alpha={probe_alpha:+.1f}, {len(scenarios)} scenarios)")
    print(f"  Baseline: seller_norm={sell_bl:+.4f}  buyer_norm={buy_bl:+.4f}")
    print("=" * 100)
    print(f"{'Rank':>4}  {'Preset':<15}  {'Layers':<14}  "
          f"{'AvgDelta':>9}  {'SellDelta':>10}  {'BuyDelta':>9}  "
          f"{'SellNorm':>9}  {'BuyNorm':>8}  {'SellDeal':>9}  {'BuyDeal':>8}")
    print("-" * 100)
    for rank, r in enumerate(results, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['preset']:<15}  {str(r['layer_indices']):<14}  "
            f"{r['avg_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  {r['buyer_delta']:>+9.4f}  "
            f"{r['seller_norm']:>+9.4f}  {r['buyer_norm']:>+8.4f}  "
            f"{r['seller_deal_rate']:>8.0%}  {r['buyer_deal_rate']:>7.0%}{marker}"
        )
    print("=" * 100)
    print(f"\nStage 1 winner: preset='{best_preset}'  layers={results[0]['layer_indices']}")
    return best_preset


# ---------------------------------------------------------------------------
# Stage 2: alpha search (grid)
# ---------------------------------------------------------------------------

def run_stage2(
    model, tokenizer,
    scenarios:     List[Dict],
    dvecs:         Dict,
    preset:        str,
    layer_indices: List[int],
    alphas:        List[float],
    max_new_tokens: int,
    temperature:   float,
    output_dir:    Path,
) -> Dict:
    log.info("Stage 2: alpha search  (%d alphas)", len(alphas))
    sell_bl, buy_bl = run_baseline(model, tokenizer, scenarios, max_new_tokens, temperature)

    results = []
    for i, alpha in enumerate(alphas):
        log.info("S2 [%d/%d] alpha=%+.2f  preset=%s  layers=%s",
                 i + 1, len(alphas), alpha, preset, layer_indices)
        r = eval_config(
            model, tokenizer, scenarios, dvecs, alpha,
            max_new_tokens, temperature, sell_bl, buy_bl,
        )
        r["preset"]        = preset
        r["layer_indices"] = layer_indices
        results.append(r)
        log.info(
            "  avg_delta=%+.4f  sell_delta=%+.4f  buy_delta=%+.4f",
            r["avg_delta"], r["seller_delta"], r["buyer_delta"],
        )

    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    best = results[0]

    with open(output_dir / "stage2_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 100)
    print(f"STAGE 2 — ALPHA SEARCH  (preset='{preset}', layers={layer_indices}, {len(scenarios)} scenarios)")
    print(f"  Baseline: seller_norm={sell_bl:+.4f}  buyer_norm={buy_bl:+.4f}")
    print("=" * 100)
    print(f"{'Rank':>4}  {'Alpha':>7}  "
          f"{'AvgDelta':>9}  {'SellDelta':>10}  {'BuyDelta':>9}  "
          f"{'SellNorm':>9}  {'BuyNorm':>8}  {'SellDeal':>9}  {'BuyDeal':>8}")
    print("-" * 100)
    for rank, r in enumerate(results, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['alpha']:>+7.2f}  "
            f"{r['avg_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  {r['buyer_delta']:>+9.4f}  "
            f"{r['seller_norm']:>+9.4f}  {r['buyer_norm']:>+8.4f}  "
            f"{r['seller_deal_rate']:>8.0%}  {r['buyer_deal_rate']:>7.0%}{marker}"
        )
    print("=" * 100)
    print(f"\nStage 2 winner: alpha={best['alpha']:+.2f}  avg_delta={best['avg_delta']:+.4f}")
    return best


# ---------------------------------------------------------------------------
# Stage 2: coarse-only two-pass variant
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
    log.info("Stage 2 (two-pass coarse): alphas=%s", coarse_alphas)
    sell_bl, buy_bl = run_baseline(model, tokenizer, scenarios, max_new_tokens, temperature)

    results = []
    for i, alpha in enumerate(coarse_alphas):
        log.info("S2 coarse [%d/%d] alpha=%+.2f", i + 1, len(coarse_alphas), alpha)
        r = eval_config(
            model, tokenizer, scenarios, dvecs, alpha,
            max_new_tokens, temperature, sell_bl, buy_bl,
        )
        r["preset"]        = preset
        r["layer_indices"] = layer_indices
        results.append(r)
        log.info(
            "  avg_delta=%+.4f  sell_delta=%+.4f  buy_delta=%+.4f",
            r["avg_delta"], r["seller_delta"], r["buyer_delta"],
        )

    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    best = results[0]

    with open(output_dir / "stage2_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 100)
    print(f"STAGE 2 — COARSE ALPHA SEARCH  (preset='{preset}', alphas={coarse_alphas})")
    print(f"  Baseline: seller_norm={sell_bl:+.4f}  buyer_norm={buy_bl:+.4f}")
    print("=" * 100)
    print(f"{'Rank':>4}  {'Alpha':>7}  "
          f"{'AvgDelta':>9}  {'SellDelta':>10}  {'BuyDelta':>9}  "
          f"{'SellNorm':>9}  {'BuyNorm':>8}  {'SellDeal':>9}  {'BuyDeal':>8}")
    print("-" * 100)
    for rank, r in enumerate(results, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(
            f"{rank:>4}  {r['alpha']:>+7.2f}  "
            f"{r['avg_delta']:>+9.4f}  {r['seller_delta']:>+10.4f}  {r['buyer_delta']:>+9.4f}  "
            f"{r['seller_norm']:>+9.4f}  {r['buyer_norm']:>+8.4f}  "
            f"{r['seller_deal_rate']:>8.0%}  {r['buyer_deal_rate']:>7.0%}{marker}"
        )
    print("=" * 100)
    print(f"\nCoarse winner: alpha={best['alpha']:+.2f}  avg_delta={best['avg_delta']:+.4f}")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-stage grid search on preset negotiation scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",       choices=list(MODELS.keys()), required=True)
    p.add_argument("--dimension",   default="firmness")
    p.add_argument("--vectors_dir", default="vectors/neg8dim_12pairs_matched/negotiation")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")

    # Stage 1
    p.add_argument("--presets", nargs="+",
                   choices=list(LAYER_PRESET_FRACTIONS.keys()),
                   default=["early", "middle", "late"])
    p.add_argument("--fixed_layers", nargs="+", type=int, default=None,
                   help="Skip Stage 1 and use these exact layer indices.")
    p.add_argument("--probe_alpha", type=float, default=5.0)

    # Stage 2
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    p.add_argument("--two_pass", action="store_true",
                   help="Coarse-only sweep using --coarse_alphas.")
    p.add_argument("--coarse_alphas", nargs="+", type=float,
                   default=[5.0, 15.0, 25.0])

    # Generation
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int,   default=80)

    # Output
    p.add_argument("--output_dir",    default=None,
                   help="Results directory. Auto-derived from vectors_dir if omitted.")
    p.add_argument("--output_suffix", type=str, default="")

    # Model loading
    p.add_argument("--dtype",     choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--quantize",  action="store_true")
    p.add_argument("--seed",      type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        vec_set    = Path(args.vectors_dir).parts[-2]
        output_dir = (
            Path("hyperparameter_results")
            / f"preset_gridsearch_{vec_set}{args.output_suffix}"
            / args.model
            / args.dimension
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg   = MODELS[args.model]
    hf_token    = HF_TOKEN if model_cfg.requires_token else None
    vectors_dir = Path(args.vectors_dir)

    metadata = load_metadata(vectors_dir, model_cfg.alias)
    n_layers = metadata["n_layers"]
    log.info("Model %s | n_layers=%d", model_cfg.hf_id, n_layers)

    skip_stage1  = args.fixed_layers is not None
    fixed_layers = sorted(set(args.fixed_layers)) if skip_stage1 else None

    # All 40 preset scenarios — fixed, no sampling
    scenarios = load_preset_scenarios()
    log.info("Using all %d preset scenarios.", len(scenarios))

    # Pre-load vectors
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

    load_kwargs: Dict = dict(token=hf_token, device_map="auto")
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
        load_kwargs["max_memory"]  = {0: "15GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, **load_kwargs)
    model.eval()

    # Print run summary
    print("\n" + "=" * 80)
    print(f"PRESET NEGOTIATION GRID SEARCH  |  model={args.model}  dim={args.dimension}")
    print(f"  Scenarios: {len(scenarios)} fixed preset scenarios (no random sampling)")
    if skip_stage1:
        print(f"  Stage 1: SKIPPED  (fixed layers={fixed_layers})")
    else:
        print(f"  Stage 1: {len(args.presets)} presets  probe_alpha={args.probe_alpha:+.1f}")
    if args.two_pass:
        print(f"  Stage 2: coarse-only  alphas={args.coarse_alphas}")
    else:
        print(f"  Stage 2: {len(args.alphas)} alphas={args.alphas}")
    print(f"  temp={args.temperature}  max_new_tokens={args.max_new_tokens}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1
    # =========================================================================
    if skip_stage1:
        best_preset = "fixed"
        best_layers = fixed_layers
        best_dvecs  = fixed_dvecs
        log.info("Stage 1 skipped — using fixed layers=%s", best_layers)
    else:
        best_preset = run_stage1(
            model=model, tokenizer=tokenizer,
            scenarios=scenarios,
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
    # STAGE 2
    # =========================================================================
    if args.two_pass:
        best = run_stage2_two_pass(
            model=model, tokenizer=tokenizer,
            scenarios=scenarios,
            dvecs=best_dvecs,
            preset=best_preset,
            layer_indices=best_layers,
            coarse_alphas=args.coarse_alphas,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_dir=output_dir,
        )
    else:
        best = run_stage2(
            model=model, tokenizer=tokenizer,
            scenarios=scenarios,
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
        "seller_delta": best["seller_delta"],
        "buyer_delta":  best["buyer_delta"],
        "seller_norm":  best["seller_norm"],
        "buyer_norm":   best["buyer_norm"],
        "seller_deal_rate": best["seller_deal_rate"],
        "buyer_deal_rate":  best["buyer_deal_rate"],
        "run_info": {
            "model":           args.model,
            "scenarios":       len(scenarios),
            "probe_alpha":     args.probe_alpha,
            "alphas":          args.coarse_alphas if args.two_pass else args.alphas,
            "two_pass":        args.two_pass,
            "temperature":     args.temperature,
            "max_new_tokens":  args.max_new_tokens,
            "seed":            args.seed,
            "timestamp":       datetime.now().isoformat(),
        },
    }
    with open(output_dir / "final_best.json", "w") as fh:
        json.dump(final, fh, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BEST CONFIG")
    print(f"  Dimension    : {args.dimension}")
    print(f"  Method       : {args.method}")
    print(f"  Preset       : {best_preset}  →  layers {best_layers}")
    print(f"  Alpha        : {best['alpha']:+.2f}")
    print(f"  AvgDelta     : {best['avg_delta']:+.4f}")
    print(f"  SellerDelta  : {best['seller_delta']:+.4f}  (norm {best['seller_norm']:+.4f})")
    print(f"  BuyerDelta   : {best['buyer_delta']:+.4f}  (norm {best['buyer_norm']:+.4f})")
    print(f"  SellerDeals  : {best['seller_deal_rate']:.0%}")
    print(f"  BuyerDeals   : {best['buyer_deal_rate']:.0%}")
    print(f"  Saved to     : {output_dir}/final_best.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
