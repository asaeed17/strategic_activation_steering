#!/usr/bin/env python3
"""
Overnight batch runner for pintail-l.
Tier 2: 13 configs at n=50 (validation of top screen hits)
Tier 3: 5 configs at n=100 (confirmation of top 5)

Loads model ONCE and loops through all configs sequentially.
"""

import argparse
import json
import random
import time
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPU compute capability check
cc = torch.cuda.get_device_capability()
if cc[0] < 8:
    print(f"ERROR: GPU compute capability {cc[0]}.{cc[1]} < 8.0. Need Ampere (RTX 3090 Ti). Aborting.")
    sys.exit(1)
print(f"GPU check passed: compute capability {cc[0]}.{cc[1]}, device={torch.cuda.get_device_name()}")

from extract_vectors import MODELS, HF_TOKEN
from apply_steering import load_direction_vectors
from ultimatum_game import (
    POOL_SIZES,
    run_paired_game,
    summarise_paired,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Tier 2: 13 configs at n=50
# -----------------------------------------------------------------------
TIER2_CONFIGS = [
    # (dimension, layer, alpha)
    ("firmness",      10,  7),   # anchor/sanity check
    ("greed",         12,  7),   # top new finding
    ("greed",         12,  3),   # dose-response
    ("greed",         12, 10),   # dose-response
    ("greed",         10,  7),   # cross-layer
    ("anchoring",     18,  7),   # different-zone peak
    ("anchoring",     18,  3),   # dose-response
    ("anchoring",     18, 10),   # dose-response
    ("narcissism",    14, -7),   # negative-only effect
    ("narcissism",    14, -3),   # negative dose-response
    ("narcissism",    14, -10),  # negative dose-response
    ("flattery",      10, -7),   # anti-flattery validation
    ("fairness_norm", 12,  7),   # why weak vs teammate d=-8.09?
]

# -----------------------------------------------------------------------
# Tier 3: 5 configs at n=100
# -----------------------------------------------------------------------
TIER3_CONFIGS = [
    ("firmness",   10,   7),
    ("greed",      12,   7),
    ("anchoring",  18,   7),
    ("narcissism", 14,  -7),
    ("flattery",   10,  -7),
]


def run_batch(model, tokenizer, configs, n_games, output_dir, cfg, args, tier_label):
    """Run a list of (dim, layer, alpha) configs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(configs)

    for idx, (dim, layer, alpha) in enumerate(configs, 1):
        parts = [dim, "proposer", f"L{layer}", f"a{alpha}", "paired", f"n{n_games}"]
        filename = "_".join(parts) + ".json"
        out_path = out_dir / filename

        # Resume support
        if out_path.exists():
            log.info("[%s %d/%d] SKIP (exists): %s", tier_label, idx, total, filename)
            continue

        log.info("[%s %d/%d] START dim=%s layer=%d alpha=%s n=%d",
                 tier_label, idx, total, dim, layer, alpha, n_games)
        t0 = time.time()

        # Reset seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Load direction vectors
        try:
            dvecs = load_direction_vectors(
                vectors_dir=Path(args.vectors_dir),
                model_alias=cfg.alias,
                dimension=dim,
                method="mean_diff",
                layer_indices=[layer],
            )
        except Exception as e:
            log.error("[%s %d/%d] FAILED to load vectors: %s", tier_label, idx, total, e)
            continue

        # Prepare pools
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(n_games)]
        random.shuffle(pools)

        # Run games
        games = []
        for i in range(n_games):
            pool = pools[i]
            result = run_paired_game(
                model, tokenizer, pool,
                steered_role="proposer",
                direction_vectors=dvecs,
                alpha=alpha,
                temperature=0.0,
                game="ultimatum",
            )
            result["game_id"] = i
            games.append(result)

            # Progress logging every 10 games
            if (i + 1) % 10 == 0 or i == 0:
                s = result["steered"]
                b = result["baseline"]
                if not result.get("parse_error") and s["proposer_share"] is not None:
                    s_pct = s["proposer_share"] / pool * 100
                    b_pct = b["proposer_share"] / pool * 100
                    log.info("  game %d/%d: steered=%.0f%% baseline=%.0f%%", i+1, n_games, s_pct, b_pct)

        elapsed = time.time() - t0

        # Summarise
        summary = summarise_paired(games, "proposer")
        summary["elapsed_seconds"] = round(elapsed, 1)

        config_dict = {
            "game": "ultimatum",
            "model": args.model,
            "dimension": dim,
            "method": "mean_diff",
            "layers": [layer],
            "alpha": alpha,
            "steered_role": "proposer",
            "paired": True,
            "n_games": n_games,
            "variable_pools": True,
            "temperature": 0.0,
            "seed": 42,
            "dtype": "bfloat16",
            "vectors_dir": args.vectors_dir,
            "timestamp": datetime.now().isoformat(),
            "tier": tier_label,
            "machine": "pintail-l",
        }

        with open(out_path, "w") as f:
            json.dump({"config": config_dict, "summary": summary, "games": games}, f, indent=2)

        d_val = summary.get("cohens_d", "?")
        p_val = summary.get("paired_ttest_p", "?")
        delta = summary.get("delta_proposer_pct", "?")
        s_mean = summary.get("steered_mean_proposer_pct", "?")
        b_mean = summary.get("baseline_mean_proposer_pct", "?")
        log.info("[%s %d/%d] DONE dim=%s L=%d a=%s | steered=%.1f%% base=%.1f%% "
                 "delta=%.1fpp d=%.2f p=%.4f | %.1fs",
                 tier_label, idx, total, dim, layer, alpha,
                 s_mean if isinstance(s_mean, (int, float)) else 0,
                 b_mean if isinstance(b_mean, (int, float)) else 0,
                 delta if isinstance(delta, (int, float)) else 0,
                 d_val if isinstance(d_val, (int, float)) else 0,
                 p_val if isinstance(p_val, (int, float)) else 1,
                 elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--vectors_dir", default="vectors/ultimatum_10dim_20pairs_general_matched/negotiation")
    args = p.parse_args()

    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    print(f"\n{'=' * 70}")
    print(f"OVERNIGHT PINTAIL-L: Tier 2 ({len(TIER2_CONFIGS)} configs @ n=50) + Tier 3 ({len(TIER3_CONFIGS)} configs @ n=100)")
    print(f"  Model: {args.model}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 70}\n")

    # Load model ONCE
    log.info("Loading model: %s", cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        token=token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    log.info("Model loaded. Device: %s", next(model.parameters()).device)

    # Tier 2: n=50
    log.info("=" * 50)
    log.info("TIER 2: %d configs at n=50", len(TIER2_CONFIGS))
    log.info("=" * 50)
    run_batch(model, tokenizer, TIER2_CONFIGS, n_games=50,
              output_dir="results/ultimatum/tier2_validation",
              cfg=cfg, args=args, tier_label="T2")

    # Tier 3: n=100
    log.info("=" * 50)
    log.info("TIER 3: %d configs at n=100", len(TIER3_CONFIGS))
    log.info("=" * 50)
    run_batch(model, tokenizer, TIER3_CONFIGS, n_games=100,
              output_dir="results/ultimatum/tier3_confirmation",
              cfg=cfg, args=args, tier_label="T3")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — pintail-l")
    print(f"End: {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
