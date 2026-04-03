#!/usr/bin/env python3
"""
Final 7B grid runner for scoter-l — BATCH 1.
Dimensions: greed, narcissism, fairness_norm (3 dims x 9 layers x 2 alphas = 54 configs @ n=50)

Loads model ONCE and loops through all configs sequentially.
Resume-safe: skips configs where output file already exists.
"""

import argparse
import json
import random
import time
import sys
import socket
import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# GPU safety check — abort if not Ampere or better
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    print("ABORT: No CUDA GPU detected.")
    sys.exit(1)

cc = torch.cuda.get_device_capability()
if cc[0] < 8:
    print(f"ABORT: GPU compute capability {cc[0]}.{cc[1]} < 8.0. "
          f"Need Ampere (RTX 3090 Ti). Got: {torch.cuda.get_device_name()}")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name()
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU OK: {gpu_name}, CC {cc[0]}.{cc[1]}, {gpu_mem:.1f} GB")

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

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
HOSTNAME = socket.gethostname()
DIMENSIONS = ["greed", "narcissism", "fairness_norm"]
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
ALPHAS = [5.0, 15.0]
N_GAMES = 50
OUTPUT_DIR = "results/ultimatum/final_7b_grid"
VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--vectors_dir", default=VECTORS_DIR)
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
    print(f"FINAL 7B GRID — {HOSTNAME}")
    print(f"  {total} configs: {len(DIMENSIONS)} dims x {len(LAYERS)} layers x {len(ALPHAS)} alphas @ n={N_GAMES}")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Layers: {LAYERS}")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Model: {args.model}  dtype: bfloat16  quantize: NO")
    print(f"  GPU: {gpu_name}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'=' * 70}\n")

    # --- Load model ONCE ---
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
    device = next(model.parameters()).device
    log.info("Model loaded. Device: %s  dtype: %s", device, next(model.parameters()).dtype)

    if str(device) == "cpu":
        log.error("MODEL ON CPU — aborting. Check GPU memory.")
        sys.exit(1)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    t_total = time.time()

    for idx, (dim, layer, alpha) in enumerate(configs, 1):
        # Naming convention: {dim}_proposer_L{layer}_a{alpha}_paired_n50.json
        parts = [dim, "proposer", f"L{layer}", f"a{alpha}", "paired", f"n{N_GAMES}"]
        filename = "_".join(parts) + ".json"
        out_path = out_dir / filename

        # Resume support
        if out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", idx, total, filename)
            skipped += 1
            continue

        log.info("[%d/%d] START dim=%s layer=%d alpha=%.1f", idx, total, dim, layer, alpha)
        t0 = time.time()

        # Reset seeds per config for reproducibility
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
            log.error("[%d/%d] FAILED to load vectors for %s L%d: %s", idx, total, dim, layer, e)
            failed += 1
            continue

        # Prepare pools
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(N_GAMES)]
        random.shuffle(pools)

        # Run games
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

            # Progress logging every 10 games
            if (i + 1) % 10 == 0:
                s = result["steered"]
                b = result["baseline"]
                if not result.get("parse_error") and s.get("proposer_share") is not None:
                    s_pct = s["proposer_share"] / pool * 100
                    b_pct = b["proposer_share"] / pool * 100 if b.get("proposer_share") else 0
                    log.info("  game %d/%d: steered=%.0f%% baseline=%.0f%% pool=$%d",
                             i + 1, N_GAMES, s_pct, b_pct, pool)

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
            "n_games": N_GAMES,
            "variable_pools": True,
            "temperature": 0.0,
            "seed": 42,
            "dtype": "bfloat16",
            "quantize": False,
            "vectors_dir": args.vectors_dir,
            "timestamp": datetime.now().isoformat(),
            "machine": HOSTNAME,
            "gpu": gpu_name,
            "gpu_cc": f"{cc[0]}.{cc[1]}",
        }

        with open(out_path, "w") as f:
            json.dump({"config": config_dict, "summary": summary, "games": games}, f, indent=2)

        done += 1
        d_val = summary.get("cohens_d", "?")
        p_val = summary.get("paired_ttest_p", "?")
        delta = summary.get("delta_proposer_pct", "?")
        s_mean = summary.get("steered_mean_proposer_pct", "?")
        b_mean = summary.get("baseline_mean_proposer_pct", "?")
        s_acc = summary.get("steered_accept_rate", "?")
        b_acc = summary.get("baseline_accept_rate", "?")
        s_pay = summary.get("steered_mean_payoff_pct", "?")
        b_pay = summary.get("baseline_mean_payoff_pct", "?")

        log.info("[%d/%d] DONE dim=%s L=%d a=%.1f | "
                 "steered=%.1f%% base=%.1f%% delta=%+.1fpp d=%.3f p=%.4f | "
                 "acc=%.0f%%/%.0f%% pay=%.1f%%/%.1f%% | %.1fs",
                 idx, total, dim, layer, alpha,
                 s_mean if isinstance(s_mean, (int, float)) else 0,
                 b_mean if isinstance(b_mean, (int, float)) else 0,
                 delta if isinstance(delta, (int, float)) else 0,
                 d_val if isinstance(d_val, (int, float)) else 0,
                 p_val if isinstance(p_val, (int, float)) else 1,
                 (s_acc * 100) if isinstance(s_acc, (int, float)) else 0,
                 (b_acc * 100) if isinstance(b_acc, (int, float)) else 0,
                 s_pay if isinstance(s_pay, (int, float)) else 0,
                 b_pay if isinstance(b_pay, (int, float)) else 0,
                 elapsed)

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {HOSTNAME}")
    print(f"  Completed: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"  Total time: {total_elapsed / 3600:.1f} hours")
    print(f"  End: {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
