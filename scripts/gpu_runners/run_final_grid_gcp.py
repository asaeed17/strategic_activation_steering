#!/usr/bin/env python3
"""
GCP batch runner for final α=-5 grid.
10 dimensions × 9 layers × α=-5 × n=50 paired games = 90 configs, 4,500 games.

Loads Qwen 2.5-7B in bfloat16 (NO quantization) ONCE, then runs all configs
sequentially with resume support.

Usage:
  python run_final_grid_gcp.py
"""

import json
import random
import time
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# ---- Configuration ----
MODEL = "qwen2.5-7b"
DIMENSIONS = [
    "firmness", "empathy", "anchoring", "greed", "narcissism",
    "fairness_norm", "composure", "flattery", "spite", "undecidedness",
]
LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
ALPHA = -5.0
N_GAMES = 50
VECTORS_DIR = "vectors/ultimatum_10dim_20pairs_general_matched/negotiation"
OUTPUT_DIR = "results/ultimatum/final_7b_grid"
METHOD = "mean_diff"
STEERED_ROLE = "proposer"
TEMPERATURE = 0.0
SEED = 42
GAME = "ultimatum"


def check_gpu():
    """Verify GPU is available with compute capability >= 8."""
    if not torch.cuda.is_available():
        log.error("No CUDA GPU available! Aborting.")
        sys.exit(1)
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    log.info("GPU: %s  Compute Capability: %s", name, cc)
    if cc[0] < 8:
        log.error("GPU compute capability %s < 8.0. Need Ampere or newer for bfloat16. Aborting.", cc)
        sys.exit(1)
    return cc, name


def main():
    cc, gpu_name = check_gpu()

    cfg = MODELS[MODEL]
    token = HF_TOKEN if cfg.requires_token else None

    # Build config list
    configs = [
        (dim, layer) for dim in DIMENSIONS for layer in LAYERS
    ]
    total = len(configs)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count already done
    done = sum(
        1 for dim, layer in configs
        if (out_dir / f"{dim}_{STEERED_ROLE}_L{layer}_a{ALPHA}_paired_n{N_GAMES}.json").exists()
    )

    print(f"\n{'=' * 70}")
    print(f"FINAL 7B GRID — alpha={ALPHA}")
    print(f"  GPU: {gpu_name} (CC {cc[0]}.{cc[1]})")
    print(f"  Model: {MODEL} (bfloat16, NO quantization)")
    print(f"  Configs: {total} ({done} already done, {total - done} remaining)")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Layers: {LAYERS}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Games per config: {N_GAMES}")
    print(f"  Start: {datetime.now().isoformat()}")
    print(f"{'=' * 70}\n")
    sys.stdout.flush()

    if done == total:
        print("All configs already complete! Nothing to do.")
        return

    # --- Load model ONCE ---
    log.info("Loading model: %s (bfloat16)", cfg.hf_id)
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
    log.info("Model loaded. Device: %s", device)

    if str(device) == "cpu":
        log.error("MODEL LOADED TO CPU! Aborting.")
        sys.exit(1)

    # --- Run all configs ---
    completed = done
    start_time = time.time()

    for idx, (dim, layer) in enumerate(configs, 1):
        filename = f"{dim}_{STEERED_ROLE}_L{layer}_a{ALPHA}_paired_n{N_GAMES}.json"
        out_path = out_dir / filename

        if out_path.exists():
            log.info("[%d/%d] SKIP (exists): %s", idx, total, filename)
            continue

        log.info("[%d/%d] START dim=%s layer=%d alpha=%s", idx, total, dim, layer, ALPHA)
        t0 = time.time()

        # Reset seeds per config
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Load direction vectors
        dvecs = load_direction_vectors(
            vectors_dir=Path(VECTORS_DIR),
            model_alias=cfg.alias,
            dimension=dim,
            method=METHOD,
            layer_indices=[layer],
        )

        # Prepare pool sizes
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(N_GAMES)]
        random.shuffle(pools)

        # Run games
        games = []
        for i in range(N_GAMES):
            pool = pools[i]
            result = run_paired_game(
                model, tokenizer, pool,
                steered_role=STEERED_ROLE,
                direction_vectors=dvecs,
                alpha=ALPHA,
                temperature=TEMPERATURE,
                game=GAME,
            )
            result["game_id"] = i
            games.append(result)

        elapsed = time.time() - t0

        # Summarise
        summary = summarise_paired(games, STEERED_ROLE)
        summary["elapsed_seconds"] = round(elapsed, 1)

        config_dict = {
            "game": GAME,
            "model": MODEL,
            "dimension": dim,
            "method": METHOD,
            "layers": [layer],
            "alpha": ALPHA,
            "steered_role": STEERED_ROLE,
            "paired": True,
            "n_games": N_GAMES,
            "variable_pools": True,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "dtype": "bfloat16",
            "vectors_dir": VECTORS_DIR,
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_name,
            "gpu_cc": f"{cc[0]}.{cc[1]}",
        }

        with open(out_path, "w") as f:
            json.dump({"config": config_dict, "summary": summary, "games": games}, f, indent=2)

        completed += 1
        d_val = summary.get("cohens_d", "?")
        p_val = summary.get("paired_ttest_p", "?")
        delta = summary.get("delta_proposer_pct", "?")
        s_mean = summary.get("steered_mean_proposer_pct", "?")
        b_mean = summary.get("baseline_mean_proposer_pct", "?")

        elapsed_total = time.time() - start_time
        rate = (completed - done) / elapsed_total * 3600 if elapsed_total > 0 else 0
        eta_h = (total - completed) / rate if rate > 0 else 0

        log.info(
            "[%d/%d] DONE dim=%s L=%d a=%s | steered=%.1f%% base=%.1f%% "
            "delta=%.1fpp d=%.2f p=%.4f | %.1fs | ETA %.1fh",
            idx, total, dim, layer, ALPHA,
            s_mean if isinstance(s_mean, float) else 0,
            b_mean if isinstance(b_mean, float) else 0,
            delta if isinstance(delta, float) else 0,
            d_val if isinstance(d_val, float) else 0,
            p_val if isinstance(p_val, float) else 1,
            elapsed, eta_h,
        )
        sys.stdout.flush()

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"ALL DONE — {completed}/{total} configs completed")
    print(f"Total time: {total_time/3600:.1f}h")
    print(f"End: {datetime.now().isoformat()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
