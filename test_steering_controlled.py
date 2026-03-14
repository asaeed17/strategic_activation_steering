#!/usr/bin/env python3
"""
test_steering_controlled.py — Properly controlled steering experiment

Tests steering vectors with proper experimental controls:
1. Single-role experiments (buyer-only or seller-only)
2. True baseline comparison (both agents unsteered vs one steered)
3. Clean measurement of steering effect

Usage:
    python test_steering_controlled.py --model qwen2.5-3b --dimension firmness \
        --role buyer --alpha 5 --num_games 10
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from existing code
from extract_vectors import MODELS, HF_TOKEN
from apply_steering import (
    load_craigslist,
    load_direction_vectors,
    run_game,
    score_deal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_controlled_experiment(
    model,
    tokenizer,
    scenarios: List[Dict],
    steered_role: str,  # "buyer" or "seller"
    dvecs: Dict[int, np.ndarray],
    alpha: float,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
) -> Dict:
    """
    Run a properly controlled experiment comparing:
    - Control: Both agents baseline (no steering)
    - Treatment: One agent steered, other baseline

    This gives us a clean measurement of steering effect for a single role.
    """
    results = []

    for i, scenario in enumerate(scenarios):
        log.info(f"\n{'='*60}")
        log.info(f"Game {i+1}/{len(scenarios)}: {scenario['title'][:50]}")
        log.info(f"Testing: {steered_role.upper()} steering (α={alpha})")

        # Control: Both baseline
        log.info("Running CONTROL (both baseline)...")
        control = run_game(
            model=model,
            tokenizer=tokenizer,
            scenario=scenario,
            dvecs_seller=None,
            alpha_seller=0.0,
            dvecs_buyer=None,
            alpha_buyer=0.0,
            steered_role="none",  # Neither is steered
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Treatment: Steer only the target role
        log.info(f"Running TREATMENT ({steered_role} steered)...")
        if steered_role == "buyer":
            treatment = run_game(
                model=model,
                tokenizer=tokenizer,
                scenario=scenario,
                dvecs_seller=None,
                alpha_seller=0.0,
                dvecs_buyer=dvecs,
                alpha_buyer=alpha,
                steered_role="buyer",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:  # seller
            treatment = run_game(
                model=model,
                tokenizer=tokenizer,
                scenario=scenario,
                dvecs_seller=dvecs,
                alpha_seller=alpha,
                dvecs_buyer=None,
                alpha_buyer=0.0,
                steered_role="seller",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        # Calculate the TRUE steering effect
        # This compares the same role's performance with/without steering
        if control["agreed"] and treatment["agreed"]:
            if steered_role == "buyer":
                control_score = control["buyer_score"]
                treatment_score = treatment["buyer_score"]
            else:
                control_score = control["seller_score"]
                treatment_score = treatment["seller_score"]

            effect = treatment_score - control_score

            log.info(f"CONTROL   {steered_role}: {control_score:.3f}")
            log.info(f"TREATMENT {steered_role}: {treatment_score:.3f}")
            log.info(f"EFFECT: {effect:+.3f} {'✓ HELPED' if effect > 0 else '✗ HURT'}")
        else:
            effect = None
            log.info("One or both games ended in no deal")

        results.append({
            "game_id": i,
            "scenario": scenario["title"],
            "steered_role": steered_role,
            "control": control,
            "treatment": treatment,
            "effect": effect,
        })

    # Summary statistics
    effects = [r["effect"] for r in results if r["effect"] is not None]
    if effects:
        mean_effect = np.mean(effects)
        std_effect = np.std(effects, ddof=1) if len(effects) > 1 else 0
        positive_count = sum(1 for e in effects if e > 0)

        summary = {
            "steered_role": steered_role,
            "dimension": args.dimension,
            "alpha": alpha,
            "num_games": len(scenarios),
            "num_with_effect": len(effects),
            "mean_effect": round(mean_effect, 4),
            "std_effect": round(std_effect, 4),
            "positive_rate": round(positive_count / len(effects), 3) if effects else 0,
            "interpretation": (
                f"Steering {steered_role}s with {args.dimension} (α={alpha}) "
                f"{'HELPS' if mean_effect > 0 else 'HURTS'} by {abs(mean_effect):.3f} on average"
            ),
        }
    else:
        summary = {
            "steered_role": steered_role,
            "dimension": args.dimension,
            "alpha": alpha,
            "num_games": len(scenarios),
            "num_with_effect": 0,
            "error": "No games resulted in deals for both conditions",
        }

    return {
        "summary": summary,
        "games": results,
    }


def main(args):
    # Load model configuration
    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    # Load steering vectors
    log.info(f"Loading {args.dimension} vectors...")
    dvecs = load_direction_vectors(
        vectors_dir=Path(args.vectors_dir),
        model_alias=cfg.alias,
        dimension=args.dimension,
        method=args.method,
        layer_indices=args.layers,
    )

    # Load model
    log.info(f"Loading model: {cfg.hf_id}")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        token=token,
        dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()

    # Load scenarios
    scenarios = load_craigslist(split="train", num_samples=args.num_games)

    # Run controlled experiment
    results = run_controlled_experiment(
        model=model,
        tokenizer=tokenizer,
        scenarios=scenarios,
        steered_role=args.role,
        dvecs=dvecs,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CONTROLLED EXPERIMENT SUMMARY")
    print("=" * 70)
    summary = results["summary"]
    for key, value in summary.items():
        print(f"  {key:20s}: {value}")
    print("=" * 70)

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run properly controlled steering experiments"
    )
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen2.5-3b")
    parser.add_argument("--role", choices=["buyer", "seller"], required=True,
                        help="Which role to steer (buyer or seller)")
    parser.add_argument("--dimension", default="firmness")
    parser.add_argument("--method", choices=["mean_diff", "pca"], default="mean_diff")
    parser.add_argument("--layers", nargs="+", type=int, default=[16])
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--vectors_dir", default="vectors")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    parser.add_argument("--output_file", default="results/controlled_experiment.json")

    args = parser.parse_args()
    main(args)