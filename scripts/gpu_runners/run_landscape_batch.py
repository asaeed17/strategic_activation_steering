#!/usr/bin/env python3
"""
Batch landscape screen runner. Loads the model ONCE and runs all
(dimension, layer, alpha) configs sequentially, saving one JSON per config.

Usage:
  CUDA_VISIBLE_DEVICES=2 python run_landscape_batch.py \
    --dimensions firmness empathy fairness_norm narcissism spite \
    --layers 6 8 10 12 14 18 20 \
    --alphas -7 7 \
    --n_games 15
"""

import argparse
import json
import random
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering import load_direction_vectors

# Import game functions from ultimatum_game
from ultimatum_game import (
    POOL_SIZES,
    run_paired_game,
    summarise_paired,
    print_paired_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--dimensions", nargs="+", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--n_games", type=int, default=15)
    p.add_argument("--vectors_dir", default="vectors/ultimatum_10dim_20pairs_general_matched/negotiation")
    p.add_argument("--output_dir", default="results/ultimatum/landscape_screen")
    p.add_argument("--method", default="mean_diff")
    p.add_argument("--steered_role", default="proposer")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--game", default="ultimatum")
    args = p.parse_args()

    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    # Total configs
    configs = [
        (dim, layer, alpha)
        for dim in args.dimensions
        for layer in args.layers
        for alpha in args.alphas
    ]
    total = len(configs)

    print(f"\n{'=' * 70}")
    print(f"LANDSCAPE BATCH SCREEN")
    print(f"  Model: {args.model}  Configs: {total}")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Layers: {args.layers}")
    print(f"  Alphas: {args.alphas}")
    print(f"  Games per config: {args.n_games}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'=' * 70}\n")

    # --- Load model ONCE ---
    log.info("Loading model: %s", cfg.hf_id)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        token=token,
        device_map="auto",
        torch_dtype=dtype_map[args.dtype],
    )
    model.eval()
    device = next(model.parameters()).device
    log.info("Model loaded. Device: %s", device)

    if str(device) == "cpu":
        log.error("MODEL LOADED TO CPU! This will be extremely slow. Aborting.")
        log.error("Check CUDA_VISIBLE_DEVICES and GPU memory availability.")
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Run configs ---
    for idx, (dim, layer, alpha) in enumerate(configs, 1):
        # Check if already done (resume support)
        parts = [dim, args.steered_role, f"L{layer}", f"a{alpha}", "paired", f"n{args.n_games}"]
        filename = "_".join(parts) + ".json"
        out_path = out_dir / filename
        if out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", idx, total, filename)
            continue

        log.info("[%d/%d] START dim=%s layer=%d alpha=%s", idx, total, dim, layer, alpha)
        t0 = time.time()

        # Reset seeds for reproducibility per config
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Load direction vectors for this config
        dvecs = load_direction_vectors(
            vectors_dir=Path(args.vectors_dir),
            model_alias=cfg.alias,
            dimension=dim,
            method=args.method,
            layer_indices=[layer],
        )

        # Prepare pool sizes
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(args.n_games)]
        random.shuffle(pools)

        # Run games
        games = []
        for i in range(args.n_games):
            pool = pools[i]
            result = run_paired_game(
                model, tokenizer, pool,
                steered_role=args.steered_role,
                direction_vectors=dvecs,
                alpha=alpha,
                temperature=args.temperature,
                game=args.game,
            )
            result["game_id"] = i
            games.append(result)

            s = result["steered"]
            b = result["baseline"]
            if result.get("parse_error"):
                print(f"    Game {i+1:2d}: PARSE_ERROR  pool=${pool}")
            else:
                s_pct = s["proposer_share"] / pool * 100 if s["proposer_share"] is not None else 0
                b_pct = b["proposer_share"] / pool * 100 if b["proposer_share"] is not None else 0
                s_dec = "ACC" if s["agreed"] else "REJ" if s["response"] else "ERR"
                b_dec = "ACC" if b["agreed"] else "REJ" if b["response"] else "ERR"
                print(f"    Game {i+1:2d}: steered={s_pct:.0f}%({s_dec}) "
                      f"baseline={b_pct:.0f}%({b_dec})  pool=${pool}")

        elapsed = time.time() - t0

        # Summarise
        summary = summarise_paired(games, args.steered_role)
        summary["elapsed_seconds"] = round(elapsed, 1)

        # Config
        config = {
            "game": args.game,
            "model": args.model,
            "dimension": dim,
            "method": args.method,
            "layers": [layer],
            "alpha": alpha,
            "steered_role": args.steered_role,
            "paired": True,
            "n_games": args.n_games,
            "variable_pools": True,
            "temperature": args.temperature,
            "seed": args.seed,
            "dtype": args.dtype,
            "vectors_dir": args.vectors_dir,
            "timestamp": datetime.now().isoformat(),
        }

        # Save
        with open(out_path, "w") as f:
            json.dump({"config": config, "summary": summary, "games": games}, f, indent=2)

        # Print summary line
        d_val = summary.get("cohens_d", "?")
        p_val = summary.get("paired_ttest_p", "?")
        delta = summary.get("delta_proposer_pct", "?")
        s_mean = summary.get("steered_mean_proposer_pct", "?")
        b_mean = summary.get("baseline_mean_proposer_pct", "?")
        log.info("[%d/%d] DONE dim=%s L=%d a=%s | steered=%.1f%% base=%.1f%% "
                 "delta=%.1fpp d=%.2f p=%.4f | %.1fs",
                 idx, total, dim, layer, alpha,
                 s_mean if isinstance(s_mean, float) else 0,
                 b_mean if isinstance(b_mean, float) else 0,
                 delta if isinstance(delta, float) else 0,
                 d_val if isinstance(d_val, float) else 0,
                 p_val if isinstance(p_val, float) else 1,
                 elapsed)

    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {total} configs")
    print(f"End: {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
