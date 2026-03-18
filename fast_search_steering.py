#!/usr/bin/env python3
"""
fast_search_steering.py

Fast two-stage search for optimal activation steering configs.

Key changes vs the old bayes_search_steering.py
------------------------------------------------
- Stage 1 is a cheap exhaustive grid over categoricals only (dim × method × preset).
  Alpha is fixed at a mid-range probe value just to rank the categorical combos.
  This is faster than random+Hyperband because we skip the per-game intermediate
  reporting overhead and just run N_PROBE_GAMES per combo.

- Stage 2 runs TPE only over alpha, with dim/method/preset fixed to the top-K
  survivors from Stage 1. The categorical space is now trivially small so TPE
  can actually learn.

- Stage 3 validates the top configs with more games at real temperature.

Speed improvements
------------------
- Greedy decoding (temp=0) during search — no sampling overhead
- max_new_tokens=60 during search (enough to see negotiation behaviour)
- Stage 1 games per combo: 5 (was up to 20)
- Stage 2 games per trial: 10 (was up to 50)
- No Hyperband bracket interleaving overhead
- No per-game intermediate reporting

Typical runtime on a 3B model
------------------------------
  Stage 1: 4 dims × 2 methods × 6 presets = 48 combos × 5 games = 240 games  ~30 min
  Stage 2: top 5 combos × 20 alpha trials × 10 games = 1000 games             ~60 min
  Stage 3: top 3 configs × 20 games                  =  60 games              ~10 min
  Total: ~1.5–2 hours

Usage
-----
  python fast_search_steering.py \\
      --model qwen2.5-3b \\
      --use_craigslist \\
      --output_dir results/fast

  # Narrower run (even faster, ~45 min):
  python fast_search_steering.py \\
      --model qwen2.5-3b \\
      --use_craigslist \\
      --dimensions firmness anchoring \\
      --methods mean_diff \\
      --layer_presets middle late \\
      --output_dir results/narrow
"""

import csv
import json
import logging
import argparse
import random
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna is required: pip install optuna", file=sys.stderr)
    sys.exit(1)

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
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Layer helpers
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
# Core game runners
# ---------------------------------------------------------------------------

def _run_role_games(
    model,
    tokenizer,
    scenarios:      List[Dict],
    dvecs:          Dict[int, np.ndarray],
    alpha:          float,
    max_new_tokens: int,
    temperature:    float,
    role:           str,
) -> Tuple[float, float]:
    """
    Run all scenarios with the steered agent fixed to `role`.
    The opponent is always unsteered (alpha=0).
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
    Run alpha=0 (fully unsteered) games for both roles on the given scenarios.
    Returns (buy_baseline_midpt_adv, sell_baseline_midpt_adv).
    These are constant for a fixed scenario set regardless of dim/method/alpha,
    so callers should compute this once and reuse across multiple configs/trials.
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


def run_config(
    model,
    tokenizer,
    scenarios:           List[Dict],
    dvecs:               Dict[int, np.ndarray],
    alpha:               float,
    max_new_tokens:      int,
    temperature:         float,
    buy_baseline_midpt:  float = None,
    sell_baseline_midpt: float = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Run steered buyer and steered seller games, then subtract the alpha=0
    baseline to produce role deltas.

    If buy_baseline_midpt / sell_baseline_midpt are provided they are reused
    (avoids repeating baseline games for every combo in Stage 1).
    Otherwise they are computed here via run_baseline_games.

    Returns:
        avg_delta           — primary objective: mean of buyer_delta + seller_delta
        avg_agree_rate
        buy_steered_midpt   — raw steered-buyer midpoint advantage
        sell_steered_midpt  — raw steered-seller midpoint advantage
        buy_baseline_midpt  — alpha=0 buyer midpoint advantage (for logging)
        sell_baseline_midpt — alpha=0 seller midpoint advantage (for logging)
    """
    buy_midpt,  buy_agree  = _run_role_games(
        model, tokenizer, scenarios, dvecs, alpha, max_new_tokens, temperature, "buyer"
    )
    sell_midpt, sell_agree = _run_role_games(
        model, tokenizer, scenarios, dvecs, alpha, max_new_tokens, temperature, "seller"
    )

    if buy_baseline_midpt is None or sell_baseline_midpt is None:
        buy_baseline_midpt, sell_baseline_midpt = run_baseline_games(
            model, tokenizer, scenarios, max_new_tokens, temperature
        )

    buy_delta  = buy_midpt  - buy_baseline_midpt
    sell_delta = sell_midpt - sell_baseline_midpt
    avg_delta  = (buy_delta  + sell_delta)  / 2.0
    avg_agree  = (buy_agree  + sell_agree)  / 2.0
    return avg_delta, avg_agree, buy_midpt, sell_midpt, buy_baseline_midpt, sell_baseline_midpt


# ---------------------------------------------------------------------------
# Coherence check — skip configs where steering breaks the model
# ---------------------------------------------------------------------------

def is_coherent(advantage: float, agree_rate: float, min_agree: float = 0.2) -> bool:
    """
    A config is incoherent if the model stops reaching agreements entirely.
    This happens when alpha is too high and corrupts the activations.
    """
    return agree_rate >= min_agree


# ---------------------------------------------------------------------------
# Stage 1: exhaustive categorical grid with fixed probe alpha
# ---------------------------------------------------------------------------

def run_stage1(
    model, tokenizer,
    scenarios:     List[Dict],
    vectors_dir:   Path,
    model_alias:   str,
    n_layers:      int,
    dimensions:    List[str],
    methods:       List[str],
    layer_presets: List[str],
    probe_alpha:   float,
    max_new_tokens: int,
    temperature:   float,
    output_dir:    Path,
) -> List[Dict]:
    """
    Run every (dimension, method, layer_preset) combo with a fixed probe_alpha.
    This is fast because:
      - No optimizer overhead
      - Few games per combo (just enough to rank them)
      - Greedy decoding (temperature=0 recommended)

    Returns results sorted by avg_midpoint_advantage (buyer+seller average) descending.
    """
    combos = list(product(dimensions, methods, layer_presets))
    log.info(
        "Stage 1: %d combos × %d scenarios × 2 roles = %d total games (+ 1 baseline run)",
        len(combos), len(scenarios), len(combos) * len(scenarios) * 2,
    )

    # Run baseline once — same scenarios, alpha=0 — reused for all combos.
    log.info("S1: computing alpha=0 baseline...")
    buy_bl, sell_bl = run_baseline_games(
        model=model, tokenizer=tokenizer, scenarios=scenarios,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    log.info("S1 baseline: buy_midpt=%+.4f  sell_midpt=%+.4f", buy_bl, sell_bl)

    results = []
    for i, (dim, method, preset) in enumerate(combos):
        layers = layers_from_preset(n_layers, preset)

        try:
            dvecs = load_direction_vectors(
                vectors_dir=vectors_dir, model_alias=model_alias,
                dimension=dim, method=method, layer_indices=layers,
            )
        except FileNotFoundError:
            log.warning("S1 [%d/%d] no vectors for %s/%s/%s — skipping",
                        i + 1, len(combos), dim, method, preset)
            continue

        avg_delta, avg_agree, buy_midpt, sell_midpt, _, _ = run_config(
            model=model, tokenizer=tokenizer, scenarios=scenarios,
            dvecs=dvecs, alpha=probe_alpha,
            max_new_tokens=max_new_tokens, temperature=temperature,
            buy_baseline_midpt=buy_bl, sell_baseline_midpt=sell_bl,
        )
        buy_delta  = buy_midpt  - buy_bl
        sell_delta = sell_midpt - sell_bl

        coherent = is_coherent(avg_delta, avg_agree)
        result = {
            "dimension":              dim,
            "method":                 method,
            "layer_preset":           preset,
            "layer_indices":          layers,
            "probe_alpha":            probe_alpha,
            "avg_delta":              avg_delta,
            "buyer_delta":            buy_delta,
            "seller_delta":           sell_delta,
            "buyer_midpoint_advantage":  buy_midpt,
            "seller_midpoint_advantage": sell_midpt,
            "buyer_baseline_midpt":   buy_bl,
            "seller_baseline_midpt":  sell_bl,
            "avg_agree_rate":         avg_agree,
            "coherent":               coherent,
        }
        results.append(result)

        log.info(
            "S1 [%d/%d] dim=%-22s method=%-9s preset=%-13s "
            "avg_delta=%+.4f  buy_delta=%+.4f  sell_delta=%+.4f  agree=%.0f%%%s",
            i + 1, len(combos), dim, method, preset,
            avg_delta, buy_delta, sell_delta, avg_agree * 100,
            "" if coherent else "  ← INCOHERENT",
        )

    # Save stage 1 results
    results.sort(key=lambda r: r["avg_delta"], reverse=True)
    out_path = output_dir / "stage1_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Print summary table
    print("\n" + "=" * 125)
    print("STAGE 1 RESULTS  (both roles, ranked by avg delta vs baseline, probe_alpha={:.2f})".format(probe_alpha))
    print(f"  Baseline: buy_midpt={buy_bl:+.4f}  sell_midpt={sell_bl:+.4f}")
    print("=" * 125)
    print(f"{'Rank':>4}  {'Dimension':<22}  {'Method':<9}  {'Preset':<13}  "
          f"{'Layers':<12}  {'AvgDelta':>9}  {'BuyDelta':>9}  {'SellDelta':>10}  {'Agree':>6}  {'OK':>4}")
    print("-" * 125)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:>4}  {r['dimension']:.<22}  {r['method']:.<9}  "
            f"{r['layer_preset']:.<13}  {str(r['layer_indices']):.<12}  "
            f"{r['avg_delta']:>+9.4f}  "
            f"{r['buyer_delta']:>+9.4f}  "
            f"{r['seller_delta']:>+10.4f}  "
            f"{r['avg_agree_rate']:>5.0%}  "
            f"{'✓' if r['coherent'] else '✗':>4}"
        )
    print("=" * 125)

    return results


# ---------------------------------------------------------------------------
# Stage 2: TPE over alpha only, for top-K categorical combos
# ---------------------------------------------------------------------------

def run_stage2(
    model, tokenizer,
    scenarios:        List[Dict],
    vectors_dir:      Path,
    model_alias:      str,
    n_layers:         int,
    top_configs:      List[Dict],   # top-K from stage 1
    alpha_low:        float,
    alpha_high:       float,
    n_trials:         int,
    max_new_tokens:   int,
    temperature:      float,
    n_startup_trials: int,
    output_dir:       Path,
    seed:             int,
    role:             str,          # "buyer" or "seller"
) -> List[Dict]:
    """
    For each top categorical config, run TPE over alpha for a fixed role.
    Called twice from main — once for "buyer", once for "seller" — to find
    the role-specific optimal alpha for each config.
    """
    all_results = []

    for cfg_rank, cfg in enumerate(top_configs, 1):
        dim    = cfg["dimension"]
        method = cfg["method"]
        preset = cfg["layer_preset"]
        layers = layers_from_preset(n_layers, preset)

        log.info(
            "Stage 2 [%d/%d] %s-alpha search | dim=%s method=%s preset=%s",
            cfg_rank, len(top_configs), role, dim, method, preset,
        )

        try:
            dvecs = load_direction_vectors(
                vectors_dir=vectors_dir, model_alias=model_alias,
                dimension=dim, method=method, layer_indices=layers,
            )
        except FileNotFoundError as exc:
            log.error("Stage 2: vectors not found — skipping. %s", exc)
            continue

        # Compute baseline once per config before the TPE loop.
        # baseline is constant across all alpha trials for the same scenarios+role.
        log.info("  S2-%s: computing alpha=0 baseline for %s/%s/%s...", role, dim, method, preset)
        baseline_midpt, _ = _run_role_games(
            model=model, tokenizer=tokenizer, scenarios=scenarios,
            dvecs=None, alpha=0.0,
            max_new_tokens=max_new_tokens, temperature=temperature,
            role=role,
        )
        log.info("  S2-%s baseline midpt_adv=%+.4f", role, baseline_midpt)

        # One TPE study per (role, categorical combo) — 1D alpha search
        study = optuna.create_study(
            study_name=f"s2_{role}_{dim}_{method}_{preset}",
            storage=f"sqlite:///{output_dir / 'stage2.db'}",
            direction="maximize",
            load_if_exists=True,
            sampler=TPESampler(n_startup_trials=n_startup_trials, seed=seed),
        )

        already_done = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, n_trials - already_done)

        def make_objective(dvecs=dvecs, role=role, baseline_midpt=baseline_midpt):
            def objective(trial: optuna.Trial) -> float:
                alpha = trial.suggest_float("alpha", alpha_low, alpha_high)
                midpt_adv, agree = _run_role_games(
                    model=model, tokenizer=tokenizer, scenarios=scenarios,
                    dvecs=dvecs, alpha=alpha,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    role=role,
                )
                delta = midpt_adv - baseline_midpt
                log.info(
                    "  S2-%s trial %2d | alpha=%5.2f  midpt_adv=%+.4f  "
                    "baseline=%+.4f  delta=%+.4f  agree=%.0f%%",
                    role, trial.number, alpha, midpt_adv, baseline_midpt, delta, agree * 100,
                )
                if not is_coherent(delta, agree):
                    log.warning("  S2-%s trial %d incoherent (agree=%.0f%%) — pruning",
                                role, trial.number, agree * 100)
                    raise optuna.TrialPruned()
                # Store the raw midpt_adv as a user attribute so we can recover it later
                trial.set_user_attr("midpoint_advantage_raw", midpt_adv)
                trial.set_user_attr("baseline_midpt", baseline_midpt)
                return delta  # optimise the delta, not raw midpt_adv
            return objective

        if remaining > 0:
            study.optimize(make_objective(), n_trials=remaining)

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            log.warning("Stage 2-%s: no completed trials for %s/%s/%s", role, dim, method, preset)
            continue

        best = max(completed, key=lambda t: t.value)
        result = {
            "role":                       role,
            "dimension":                  dim,
            "method":                     method,
            "layer_preset":               preset,
            "layer_indices":              layers,
            "best_alpha":                 best.params["alpha"],
            "delta":                      best.value,
            "midpoint_advantage_raw":     best.user_attrs.get("midpoint_advantage_raw"),
            "baseline_midpt":             baseline_midpt,
            "n_trials":                   len(completed),
            "alpha_trials": [
                {
                    "alpha":                  t.params["alpha"],
                    "delta":                  t.value,
                    "midpoint_advantage_raw": t.user_attrs.get("midpoint_advantage_raw"),
                }
                for t in sorted(completed, key=lambda t: t.params["alpha"])
            ],
        }
        all_results.append(result)

        log.info(
            "S2-%s [%d/%d] best alpha=%.4f  delta=%+.4f  midpt_raw=%+.4f  (%d trials)",
            role, cfg_rank, len(top_configs),
            best.params["alpha"], best.value,
            result["midpoint_advantage_raw"] or float("nan"),
            len(completed),
        )

    all_results.sort(key=lambda r: r["delta"], reverse=True)
    out_path = output_dir / f"stage2_{role}_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    # Print summary
    print("\n" + "=" * 105)
    print(f"STAGE 2 RESULTS  ({role}-steered TPE alpha search, objective = delta vs baseline)")
    print("=" * 105)
    print(f"{'Rank':>4}  {'Dimension':<22}  {'Method':<9}  {'Preset':<13}  "
          f"{'Best Alpha':>10}  {'Delta':>9}  {'MidptRaw':>9}  {'Baseline':>9}  {'Trials':>6}")
    print("-" * 105)
    for rank, r in enumerate(all_results, 1):
        print(
            f"{rank:>4}  {r['dimension']:.<22}  {r['method']:.<9}  "
            f"{r['layer_preset']:.<13}  {r['best_alpha']:>10.4f}  "
            f"{r['delta']:>+9.4f}  "
            f"{(r['midpoint_advantage_raw'] or 0):>+9.4f}  "
            f"{r['baseline_midpt']:>+9.4f}  "
            f"{r['n_trials']:>6}"
        )
    print("=" * 105)

    return all_results


# ---------------------------------------------------------------------------
# Stage 3: final validation at real temperature
# ---------------------------------------------------------------------------

def run_stage3(
    model, tokenizer,
    scenarios:     List[Dict],
    vectors_dir:   Path,
    model_alias:   str,
    n_layers:      int,
    top_configs:   List[Dict],   # merged buyer+seller configs from stage 2
    max_new_tokens: int,
    temperature:   float,
    output_dir:    Path,
) -> List[Dict]:
    """
    Validate top configs using the role-specific alphas found in Stage 2.
    Each config carries a buyer_alpha and seller_alpha; we run each role with
    its own optimal alpha and report both results. Sorted by avg midpoint
    advantage (buyer + seller) / 2.
    """
    # Compute baseline once for S3 scenarios — reused across all configs.
    log.info("Stage 3: computing alpha=0 baseline (%d scenarios)...", len(scenarios))
    buy_bl, sell_bl = run_baseline_games(
        model=model, tokenizer=tokenizer, scenarios=scenarios,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    log.info("S3 baseline: buy_midpt=%+.4f  sell_midpt=%+.4f", buy_bl, sell_bl)

    results = []
    for rank, cfg in enumerate(top_configs, 1):
        dim          = cfg["dimension"]
        method       = cfg["method"]
        preset       = cfg["layer_preset"]
        buyer_alpha  = cfg["buyer_alpha"]
        seller_alpha = cfg["seller_alpha"]
        layers       = layers_from_preset(n_layers, preset)

        log.info(
            "Stage 3 [%d/%d] dim=%s method=%s preset=%s "
            "buyer_alpha=%.4f seller_alpha=%.4f",
            rank, len(top_configs), dim, method, preset,
            buyer_alpha, seller_alpha,
        )

        try:
            dvecs = load_direction_vectors(
                vectors_dir=vectors_dir, model_alias=model_alias,
                dimension=dim, method=method, layer_indices=layers,
            )
        except FileNotFoundError as exc:
            log.error("Stage 3: missing vectors — skipping. %s", exc)
            continue

        buy_midpt, buy_agree = _run_role_games(
            model=model, tokenizer=tokenizer, scenarios=scenarios,
            dvecs=dvecs, alpha=buyer_alpha,
            max_new_tokens=max_new_tokens, temperature=temperature,
            role="buyer",
        )
        sell_midpt, sell_agree = _run_role_games(
            model=model, tokenizer=tokenizer, scenarios=scenarios,
            dvecs=dvecs, alpha=seller_alpha,
            max_new_tokens=max_new_tokens, temperature=temperature,
            role="seller",
        )

        buy_delta  = buy_midpt  - buy_bl
        sell_delta = sell_midpt - sell_bl
        avg_delta  = (buy_delta + sell_delta) / 2.0

        result = {
            "rank":                      rank,
            "dimension":                 dim,
            "method":                    method,
            "layer_preset":              preset,
            "layer_indices":             layers,
            "buyer_alpha":               buyer_alpha,
            "seller_alpha":              seller_alpha,
            "buyer_midpoint_advantage":  buy_midpt,
            "buyer_baseline_midpt":      buy_bl,
            "buyer_delta":               buy_delta,
            "buyer_agree_rate":          buy_agree,
            "seller_midpoint_advantage": sell_midpt,
            "seller_baseline_midpt":     sell_bl,
            "seller_delta":              sell_delta,
            "seller_agree_rate":         sell_agree,
            "avg_delta":                 avg_delta,
            "n_games":                   len(scenarios),
        }
        results.append(result)
        log.info(
            "S3 [%d/%d] avg_delta=%+.4f  "
            "buy_delta=%+.4f(raw=%+.4f,α=%.2f,agr=%.0f%%)  "
            "sell_delta=%+.4f(raw=%+.4f,α=%.2f,agr=%.0f%%)",
            rank, len(top_configs), avg_delta,
            buy_delta,  buy_midpt,  buyer_alpha,  buy_agree  * 100,
            sell_delta, sell_midpt, seller_alpha, sell_agree * 100,
        )

        out_path = output_dir / f"stage3_rank{rank:02d}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

    results.sort(key=lambda r: r["avg_delta"], reverse=True)

    print("\n" + "=" * 145)
    print(f"STAGE 3 — FINAL VALIDATION  ({len(scenarios)} scenarios, role-specific alphas, temp={temperature})")
    print(f"  Baseline: buy_midpt={buy_bl:+.4f}  sell_midpt={sell_bl:+.4f}")
    print("=" * 145)
    print(f"{'Rank':>4}  {'Dimension':<22}  {'Method':<9}  {'Preset':<13}  "
          f"{'AvgDelta':>9}  {'BuyAlpha':>9}  {'BuyDelta':>9}  {'BuyAgr':>6}  "
          f"{'SellAlpha':>10}  {'SellDelta':>10}  {'SellAgr':>7}")
    print("-" * 145)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:>4}  {r['dimension']:.<22}  {r['method']:.<9}  "
            f"{r['layer_preset']:.<13}  {r['avg_delta']:>+9.4f}  "
            f"{r['buyer_alpha']:>9.4f}  {r['buyer_delta']:>+9.4f}  "
            f"{r['buyer_agree_rate']:>5.0%}  "
            f"{r['seller_alpha']:>10.4f}  {r['seller_delta']:>+10.4f}  "
            f"{r['seller_agree_rate']:>6.0%}"
        )
    print("=" * 145)

    if results:
        best = results[0]
        print(f"\nFINAL BEST CONFIG (by avg delta vs baseline):")
        print(f"  Dimension     : {best['dimension']}")
        print(f"  Method        : {best['method']}")
        print(f"  Preset        : {best['layer_preset']}  →  layers {best['layer_indices']}")
        print(f"  Baseline      : buy_midpt={best['buyer_baseline_midpt']:+.4f}  sell_midpt={best['seller_baseline_midpt']:+.4f}")
        print(f"  Buyer  alpha  : {best['buyer_alpha']:.4f}  →  midpt {best['buyer_midpoint_advantage']:+.4f}  delta {best['buyer_delta']:+.4f}  (agree {best['buyer_agree_rate']:.0%})")
        print(f"  Seller alpha  : {best['seller_alpha']:.4f}  →  midpt {best['seller_midpoint_advantage']:+.4f}  delta {best['seller_delta']:+.4f}  (agree {best['seller_agree_rate']:.0%})")
        print(f"  Avg delta     : {best['avg_delta']:+.4f}")
        print(f"  Validated on {best['n_games']} scenarios per role")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast two-stage steering config search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",       choices=list(MODELS.keys()), required=True)
    p.add_argument("--vectors_dir", default="vectors")
    p.add_argument("--dimensions",  nargs="*",  default=None)
    p.add_argument("--methods", nargs="+", choices=["mean_diff", "pca"],
               default=["mean_diff"])
    p.add_argument("--layer_presets", nargs="+",
                   choices=list(LAYER_PRESET_FRACTIONS.keys()),
                   default=list(LAYER_PRESET_FRACTIONS.keys()))

    # Alpha
    p.add_argument("--alpha_low",   type=float, default=0.5)
    p.add_argument("--alpha_high",  type=float, default=8.0)
    p.add_argument("--probe_alpha", type=float, default=None,
                   help="Alpha used in Stage 1 grid. Defaults to midpoint of range.")

    # Stage 1
    p.add_argument("--s1_games",    type=int, default=5,
                   help="Games per combo in Stage 1 grid. Default: 5.")

    # Stage 2
    p.add_argument("--s2_top_k",    type=int, default=5,
                   help="Top-K combos from S1 to optimise alpha for. Default: 5.")
    p.add_argument("--s2_trials",   type=int, default=20,
                   help="TPE alpha trials per combo in Stage 2. Default: 20.")
    p.add_argument("--s2_games",    type=int, default=10,
                   help="Games per alpha trial in Stage 2. Default: 10.")
    p.add_argument("--n_startup_trials", type=int, default=8,
                   help="Random trials before TPE exploits in S2. Default: 8.")

    # Stage 3
    p.add_argument("--s3_top_n",    type=int, default=3,
                   help="Top N configs from S2 to validate in S3. Default: 3.")
    p.add_argument("--s3_games",    type=int, default=20,
                   help="Games per config in S3 validation. Default: 20.")
    p.add_argument("--eval_temperature", type=float, default=0.7,
                   help="Temperature for S3 validation. Default: 0.7.")

    # General
    p.add_argument("--search_temperature", type=float, default=0.0,
                   help="Temperature for S1/S2 search. 0=greedy. Default: 0.0.")
    p.add_argument("--max_new_tokens",      type=int,   default=60,
                   help="Max tokens per turn during search. Default: 60.")
    p.add_argument("--s3_max_new_tokens",   type=int,   default=120,
                   help="Max tokens per turn during S3 validation. Default: 120.")
    p.add_argument("--dataset_split",  choices=["train", "validation"], default="train")
    p.add_argument("--dtype",          choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--output_dir",     default="fast_results")
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

    probe_alpha = args.probe_alpha if args.probe_alpha else (args.alpha_low + args.alpha_high) / 2

    model_cfg = MODELS[args.model]
    hf_token  = HF_TOKEN if model_cfg.requires_token else None

    vectors_dir = Path(args.vectors_dir)
    metadata    = load_metadata(vectors_dir, model_cfg.alias)
    n_layers    = metadata["n_layers"]
    log.info("Model %s | n_layers=%d", model_cfg.hf_id, n_layers)

    dimensions = args.dimensions if args.dimensions else metadata.get("dimensions", [])
    if not dimensions:
        raise ValueError("No dimensions found. Pass --dimensions or check metadata.json.")

    # Load max scenarios needed
    max_games = max(args.s1_games, args.s2_games, args.s3_games)
    random.seed(args.seed)
    np.random.seed(args.seed)
    all_scenarios = load_craigslist(split=args.dataset_split, num_samples=max_games)
    log.info("Loaded %d scenarios", len(all_scenarios))

    s1_scenarios = all_scenarios[: args.s1_games]
    s2_scenarios = all_scenarios[: args.s2_games]
    s3_scenarios = all_scenarios[: args.s3_games]

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

    n_combos = len(dimensions) * len(args.methods) * len(args.layer_presets)
    total_s1  = n_combos * args.s1_games
    total_s2  = args.s2_top_k * args.s2_trials * args.s2_games * 2  # ×2: buyer + seller
    total_s3  = args.s3_top_n * args.s3_games

    print("\n" + "=" * 80)
    print(f"FAST STEERING SEARCH  |  model={args.model}")
    print(f"  Stage 1: {n_combos} combos × {args.s1_games} games = {total_s1} games  "
          f"(probe_alpha={probe_alpha:.2f}, temp={args.search_temperature})")
    print(f"  Stage 2: {args.s2_top_k} combos × {args.s2_trials} trials × "
          f"{args.s2_games} games = {total_s2} games")
    print(f"  Stage 3: {args.s3_top_n} configs × {args.s3_games} games = {total_s3} games  "
          f"(temp={args.eval_temperature})")
    print(f"  Total games: ~{total_s1 + total_s2 + total_s3}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STAGE 1 — Exhaustive categorical grid
    # =========================================================================
    s1_results = run_stage1(
        model=model, tokenizer=tokenizer,
        scenarios=s1_scenarios,
        vectors_dir=vectors_dir, model_alias=model_cfg.alias, n_layers=n_layers,
        dimensions=dimensions, methods=args.methods, layer_presets=args.layer_presets,
        probe_alpha=probe_alpha,
        max_new_tokens=args.max_new_tokens, temperature=args.search_temperature,
        output_dir=output_dir,
    )

    # Filter to coherent configs only, take top-K
    coherent = [r for r in s1_results if r["coherent"]]
    if not coherent:
        log.error("No coherent configs found in Stage 1. Try lowering --alpha_high.")
        sys.exit(1)

    top_k_configs = coherent[: args.s2_top_k]
    log.info("Stage 1 → 2: passing top %d coherent configs (ranked by avg delta vs baseline)", len(top_k_configs))
    for r in top_k_configs:
        log.info("  %s / %s / %s  avg_delta=%+.4f  buy_delta=%+.4f  sell_delta=%+.4f",
                 r["dimension"], r["method"], r["layer_preset"],
                 r["avg_delta"], r["buyer_delta"], r["seller_delta"])

    # =========================================================================
    # STAGE 2 — TPE over alpha, separately for buyer and seller
    # =========================================================================
    s2_common = dict(
        model=model, tokenizer=tokenizer,
        scenarios=s2_scenarios,
        vectors_dir=vectors_dir, model_alias=model_cfg.alias, n_layers=n_layers,
        top_configs=top_k_configs,
        alpha_low=args.alpha_low, alpha_high=args.alpha_high,
        n_trials=args.s2_trials,
        max_new_tokens=args.max_new_tokens, temperature=args.search_temperature,
        n_startup_trials=args.n_startup_trials,
        output_dir=output_dir,
        seed=args.seed,
    )
    log.info("Stage 2 — buyer alpha search")
    s2_buyer  = run_stage2(**s2_common, role="buyer")
    log.info("Stage 2 — seller alpha search")
    s2_seller = run_stage2(**s2_common, role="seller")

    if not s2_buyer and not s2_seller:
        log.error("No Stage 2 results. Exiting.")
        sys.exit(1)

    # Merge buyer and seller results by (dim, method, preset)
    # Each S3 config carries the role-specific best alpha for each role.
    merged: Dict[tuple, Dict] = {}
    for r in s2_buyer:
        key = (r["dimension"], r["method"], r["layer_preset"])
        merged[key] = {
            "dimension":       r["dimension"],
            "method":          r["method"],
            "layer_preset":    r["layer_preset"],
            "layer_indices":   r["layer_indices"],
            "buyer_alpha":     r["best_alpha"],
            "buyer_delta_s2":  r["delta"],
            "seller_alpha":    None,
            "seller_delta_s2": None,
        }
    for r in s2_seller:
        key = (r["dimension"], r["method"], r["layer_preset"])
        if key not in merged:
            merged[key] = {
                "dimension":       r["dimension"],
                "method":          r["method"],
                "layer_preset":    r["layer_preset"],
                "layer_indices":   r["layer_indices"],
                "buyer_alpha":     None,
                "buyer_delta_s2":  None,
            }
        merged[key]["seller_alpha"]    = r["best_alpha"]
        merged[key]["seller_delta_s2"] = r["delta"]

    # Fill missing alphas with midpoint of range as fallback
    fallback_alpha = (args.alpha_low + args.alpha_high) / 2.0
    for v in merged.values():
        if v["buyer_alpha"]  is None: v["buyer_alpha"]  = fallback_alpha
        if v["seller_alpha"] is None: v["seller_alpha"] = fallback_alpha

    # Rank by average of S2 buyer and seller deltas
    def _avg_s2(v):
        buy  = v["buyer_delta_s2"]  if v["buyer_delta_s2"]  is not None else -999
        sell = v["seller_delta_s2"] if v["seller_delta_s2"] is not None else -999
        return (buy + sell) / 2.0

    top_s3_configs = sorted(merged.values(), key=_avg_s2, reverse=True)[: args.s3_top_n]

    # =========================================================================
    # STAGE 3 — Final validation with role-specific alphas
    # =========================================================================
    s3_results = run_stage3(
        model=model, tokenizer=tokenizer,
        scenarios=s3_scenarios,
        vectors_dir=vectors_dir, model_alias=model_cfg.alias, n_layers=n_layers,
        top_configs=top_s3_configs,
        max_new_tokens=args.s3_max_new_tokens, temperature=args.eval_temperature,
        output_dir=output_dir,
    )

    # Save final summary
    if s3_results:
        final = {
            "best": s3_results[0],
            "all_validated": s3_results,
            "run_info": {
                "model":              args.model,
                "s1_games":           args.s1_games,
                "s2_trials":          args.s2_trials,
                "s2_games":           args.s2_games,
                "s3_games":           args.s3_games,
                "probe_alpha":        probe_alpha,
                "alpha_range":        [args.alpha_low, args.alpha_high],
                "search_temperature": args.search_temperature,
                "eval_temperature":   args.eval_temperature,
                "seed":               args.seed,
                "timestamp":          datetime.now().isoformat(),
            },
        }
        with open(output_dir / "final_best.json", "w") as fh:
            json.dump(final, fh, indent=2)
        log.info("Done. Final best → %s/final_best.json", output_dir)


if __name__ == "__main__":
    main()