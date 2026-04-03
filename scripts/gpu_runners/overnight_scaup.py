#!/usr/bin/env python3
"""
Overnight batch runner for scaup-l.
Rerun Batch B screen on bird hardware (RTX 3090 Ti) for consistent baselines.

5 dims x 7 layers x 2 alphas = 70 configs at n=15.
Dimensions: composure, anchoring, flattery, greed, undecidedness
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

DIMENSIONS = ["composure", "anchoring", "flattery", "greed", "undecidedness"]
LAYERS = [6, 8, 10, 12, 14, 18, 20]
ALPHAS = [-7.0, 7.0]
N_GAMES = 15


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--vectors_dir", default="vectors/ultimatum_10dim_20pairs_general_matched/negotiation")
    args = p.parse_args()

    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    configs = [
        (dim, layer, alpha)
        for dim in DIMENSIONS
        for layer in LAYERS
        for alpha in ALPHAS
    ]
    total = len(configs)

    print(f"\n{'=' * 70}")
    print(f"OVERNIGHT SCAUP-L: Batch B Rerun ({total} configs @ n={N_GAMES})")
    print(f"  Model: {args.model}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Layers: {LAYERS}")
    print(f"  Alphas: {ALPHAS}")
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

    out_dir = Path("results/ultimatum/landscape_screen_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (dim, layer, alpha) in enumerate(configs, 1):
        parts = [dim, "proposer", f"L{layer}", f"a{alpha}", "paired", f"n{N_GAMES}"]
        filename = "_".join(parts) + ".json"
        out_path = out_dir / filename

        # Resume support
        if out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", idx, total, filename)
            continue

        log.info("[%d/%d] START dim=%s layer=%d alpha=%s", idx, total, dim, layer, alpha)
        t0 = time.time()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        try:
            dvecs = load_direction_vectors(
                vectors_dir=Path(args.vectors_dir),
                model_alias=cfg.alias,
                dimension=dim,
                method="mean_diff",
                layer_indices=[layer],
            )
        except Exception as e:
            log.error("[%d/%d] FAILED to load vectors: %s", idx, total, e)
            continue

        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(N_GAMES)]
        random.shuffle(pools)

        games = []
        for i in range(N_GAMES):
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

        elapsed = time.time() - t0

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
            "n_games": N_GAMES,
            "variable_pools": True,
            "temperature": 0.0,
            "seed": 42,
            "dtype": "bfloat16",
            "vectors_dir": args.vectors_dir,
            "timestamp": datetime.now().isoformat(),
            "machine": "scaup-l",
        }

        with open(out_path, "w") as f:
            json.dump({"config": config_dict, "summary": summary, "games": games}, f, indent=2)

        d_val = summary.get("cohens_d", "?")
        p_val = summary.get("paired_ttest_p", "?")
        delta = summary.get("delta_proposer_pct", "?")
        s_mean = summary.get("steered_mean_proposer_pct", "?")
        b_mean = summary.get("baseline_mean_proposer_pct", "?")
        log.info("[%d/%d] DONE dim=%s L=%d a=%s | steered=%.1f%% base=%.1f%% "
                 "delta=%.1fpp d=%.2f p=%.4f | %.1fs",
                 idx, total, dim, layer, alpha,
                 s_mean if isinstance(s_mean, (int, float)) else 0,
                 b_mean if isinstance(b_mean, (int, float)) else 0,
                 delta if isinstance(delta, (int, float)) else 0,
                 d_val if isinstance(d_val, (int, float)) else 0,
                 p_val if isinstance(p_val, (int, float)) else 1,
                 elapsed)

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — scaup-l")
    print(f"End: {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
