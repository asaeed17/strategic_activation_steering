#!/usr/bin/env python3
"""
lightweight_gridsearch_ultimatum.py

Two-stage grid search for the Ultimatum Game with activation steering.

Stage 1: Layer preset search (or skip with --fixed_layers)
Stage 2: Alpha search (coarse then fine)

Scoring metric (Directional Fitness):
  Evaluates the shift against baseline (alpha=0), multiplied by the EXPECTED_DIRECTION.
  Severely penalises alphas that cause model collapse (< 80% valid games).

Usage:
    python lightweight_gridsearch_ultimatum.py \
        --model qwen2.5-7b \
        --dimension firmness \
        --role proposer \
        --vectors_dir vectors/neg15dim_12pairs_matched/negotiation \
        --fixed_layers 10 14 \
        --coarse_alphas 0 -5 5 15 \
        --output_suffix _7b_ult
"""

import json
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats as scipy_stats
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering_preset_nego import load_direction_vectors
from apply_steering_ultimatum import (
    POOL_SIZES,
    RESPONDER_THRESHOLDS,
    run_all_games,
    summarise,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LAYER_PRESET_FRACTIONS: Dict[str, List[float]] = {
    "early":        [0.25],
    "middle":       [0.50],
    "late":         [0.75],
    "early_middle": [0.25, 0.50],
    "middle_late":  [0.50, 0.75],
    "spread":       [0.25, 0.50, 0.75],
}

# ---------------------------------------------------------------------------
# Fitness Scoring Logic
# ---------------------------------------------------------------------------

EXPECTED_DIRECTIONS = {
    "proposer": {
        "firmness": 1, "empathy": -1, "narcissism": 1, "fairness_norm": -1,
        "spite": 1, "anchoring": 1, "batna_awareness": 1, "flattery": -1,
        "undecidedness": -1, "composure": 1
    },
    "responder": {
        "firmness": -1, "empathy": 1, "composure": 1, "fairness_norm": -1,
        "spite": -1, "narcissism": -1, "undecidedness": -1, "anchoring": -1,
        "batna_awareness": 1, "flattery": 1
    }
}

def get_fitness_score(steered_results: List[Dict], baseline_results: List[Dict], role: str, dimension: str) -> float:
    """
    Calculates how well the steering moved the model in the INTENDED direction.
    Penalizes model collapse (parse errors).
    """
    direction = EXPECTED_DIRECTIONS.get(role, {}).get(dimension, 1)

    def get_valid(games):
        return [g for g in games if not g["parse_error"] and g.get("proposer_share") is not None]

    steered_valid = get_valid(steered_results)
    base_valid = get_valid(baseline_results)

    if not steered_valid or not base_valid:
        return -999.0

    # Proposer: Focus on intent (offer %), not payoff (which is ruined by rejections)
    if role == "proposer":
        steered_metric = sum(g["proposer_share"] / g["pool"] * 100 for g in steered_valid) / len(steered_valid)
        base_metric = sum(g["proposer_share"] / g["pool"] * 100 for g in base_valid) / len(base_valid)
    # Responder: Focus on accept rate %
    else:
        steered_metric = sum(1 for g in steered_valid if g["agreed"]) / len(steered_valid) * 100
        base_metric = sum(1 for g in base_valid if g["agreed"]) / len(base_valid) * 100

    raw_shift = steered_metric - base_metric
    directional_shift = raw_shift * direction

    valid_ratio = len(steered_valid) / max(len(steered_results), 1)
    
    # Severe penalty for model collapse
    if valid_ratio < 0.80:
        return -999.0

    return directional_shift * valid_ratio


def layers_from_preset(number_of_layers: int, preset: str) -> List[int]:
    return sorted(set(int(number_of_layers * f) for f in LAYER_PRESET_FRACTIONS[preset]))


def score_results(results: List[Dict], role: str) -> float:
    """Legacy raw score (for logging)."""
    valid_results = [r for r in results if not r["parse_error"]]
    if not valid_results:
        return 0.0
    if role == "proposer":
        return sum(r["proposer_share"] / r["pool"] for r in valid_results) / len(valid_results) * 100
    else:  
        return sum(1 for r in valid_results if r["agreed"]) / len(valid_results) * 100


def compute_full_stats(
    steered_results: List[Dict],
    baseline_results: List[Dict],
    role: str,
) -> Dict:
    def get_valid(games):
        return [g for g in games if not g["parse_error"] and g.get("proposer_share") is not None]

    steered_valid  = get_valid(steered_results)
    baseline_valid = get_valid(baseline_results)
    number_of_paired_games = min(len(steered_valid), len(baseline_valid))

    if number_of_paired_games == 0:
         return {"error": "Not enough valid games to compute stats."}

    steered_offer_pct  = np.array([g["proposer_share"] / g["pool"] * 100 for g in steered_valid[:number_of_paired_games]])
    baseline_offer_pct = np.array([g["proposer_share"] / g["pool"] * 100 for g in baseline_valid[:number_of_paired_games]])

    steered_payoff_pct  = np.array([g["proposer_payoff"] / g["pool"] * 100 for g in steered_valid[:number_of_paired_games]])
    baseline_payoff_pct = np.array([g["proposer_payoff"] / g["pool"] * 100 for g in baseline_valid[:number_of_paired_games]])

    steered_accepted  = np.array([float(g["agreed"]) for g in steered_valid[:number_of_paired_games]])
    baseline_accepted = np.array([float(g["agreed"]) for g in baseline_valid[:number_of_paired_games]])

    offer_deltas  = steered_offer_pct  - baseline_offer_pct
    payoff_deltas = steered_payoff_pct - baseline_payoff_pct

    def permutation_test(delta_array, n_permutations=10000, seed=0):
        rng = np.random.default_rng(seed)
        observed_mean = delta_array.mean()
        if observed_mean == 0:
            return 0.0, 1.0
        flip_signs = rng.choice([-1, 1], size=(n_permutations, len(delta_array)))
        permuted_means = (flip_signs * delta_array).mean(axis=1)
        if observed_mean >= 0:
            p_one_tail = float((permuted_means >= observed_mean).mean())
        else:
            p_one_tail = float((permuted_means <= observed_mean).mean())
        return observed_mean, min(p_one_tail * 2, 1.0) 

    def cohens_d(delta_array):
        std = delta_array.std(ddof=1)
        if std > 0: return float(delta_array.mean() / std)
        return float("inf") if delta_array.mean() != 0 else 0.0

    def chi_square_or_fisher(steered_accept_count, baseline_accept_count, n):
        steered_reject_count  = n - steered_accept_count
        baseline_reject_count = n - baseline_accept_count
        contingency_table = np.array([
            [steered_accept_count,  steered_reject_count],
            [baseline_accept_count, baseline_reject_count],
        ])
        if np.any(contingency_table < 5):
            _, p_value = scipy_stats.fisher_exact(contingency_table)
            test_used = "fishers_exact"
        else:
            chi2_stat, p_value, _, _ = scipy_stats.chi2_contingency(contingency_table, correction=False)
            test_used = "chi_square"
        return float(p_value), test_used

    def r(value, decimal_places=4):
        return round(float(value), decimal_places)

    _, offer_p_value = permutation_test(offer_deltas)
    offer_significant = offer_p_value < 0.05

    steered_accept_count  = int(steered_accepted.sum())
    baseline_accept_count = int(baseline_accepted.sum())
    acceptance_p_value, acceptance_test_used = chi_square_or_fisher(
        steered_accept_count, baseline_accept_count, number_of_paired_games
    )
    acceptance_significant = acceptance_p_value < 0.05

    steered_accept_rate  = steered_accepted.mean()
    baseline_accept_rate = baseline_accepted.mean()

    _, payoff_p_value = permutation_test(payoff_deltas)

    return {
        "number_of_paired_games":           number_of_paired_games,
        "number_of_steered_valid_games":    len(steered_valid),
        "number_of_baseline_valid_games":   len(baseline_valid),
        "number_of_steered_parse_errors":   sum(1 for g in steered_results  if g["parse_error"]),
        "number_of_baseline_parse_errors":  sum(1 for g in baseline_results if g["parse_error"]),

        "offer_shift": {
            "description":                  "Paired t-test on proposer_share/pool. PRIMARY significance test.",
            "steered_mean_offer_pct":        r(steered_offer_pct.mean()),
            "baseline_mean_offer_pct":       r(baseline_offer_pct.mean()),
            "mean_delta_pct":                r(offer_deltas.mean()),
            "std_delta_pct":                 r(offer_deltas.std(ddof=1)),
            "median_delta_pct":              r(float(np.median(offer_deltas))),
            "t_statistic":                   None,
            "p_value":                       r(offer_p_value),
            "cohens_d":                      r(cohens_d(offer_deltas)),
            "significant_at_0_05":           offer_significant,
        },

        "acceptance_shift": {
            "description":                  "Chi-square or Fisher's exact on accept/reject counts. PRIMARY significance test.",
            "test_used":                     acceptance_test_used,
            "steered_accept_count":          steered_accept_count,
            "baseline_accept_count":         baseline_accept_count,
            "steered_accept_rate":           r(steered_accept_rate),
            "baseline_accept_rate":          r(baseline_accept_rate),
            "accept_rate_delta":             r(steered_accept_rate - baseline_accept_rate),
            "p_value":                       r(acceptance_p_value),
            "significant_at_0_05":           acceptance_significant,
        },

        "payoff_delta_reference_only": {
            "description":                  "T-test on payoff/pool. REFERENCE ONLY.",
            "steered_mean_payoff_pct":       r(steered_payoff_pct.mean()),
            "baseline_mean_payoff_pct":      r(baseline_payoff_pct.mean()),
            "observed_mean_delta_pct":       r(payoff_deltas.mean()),
            "std_delta_pct":                 r(payoff_deltas.std(ddof=1)),
            "median_delta_pct":              r(float(np.median(payoff_deltas))),
            "t_statistic":                   None,
            "p_value":                       r(payoff_p_value),
            "cohens_d":                      r(cohens_d(payoff_deltas)),
            "warning":                       "p_value unreliable — do not use for significance claims",
        },
    }

def run_eval(
    model, tokenizer, role: str, direction_vectors: Optional[Dict[int, np.ndarray]],
    alpha: float, pool_sequence: List[int], max_new_tokens: int, temperature: float,
    accept_threshold: float, proposer_fraction: float,
    rulebased: bool = False,
) -> Tuple[float, List[Dict]]:
    results = run_all_games(
        model, tokenizer, role=role, dvecs=direction_vectors, alpha=alpha,
        n_games=len(pool_sequence), pools=pool_sequence, accept_threshold=accept_threshold,
        proposer_fraction=proposer_fraction, max_new_tokens=max_new_tokens,
        temperature=temperature, rulebased=rulebased,
    )
    return score_results(results, role), results


def run_baseline(
    model, tokenizer, role, pool_sequence,
    max_new_tokens, temperature, accept_threshold, proposer_fraction,
    rulebased: bool = False,
):
    log.info("Computing baseline (alpha=0, %d games%s)...",
             len(pool_sequence), ", rulebased" if rulebased else "")
    baseline_score, baseline_results = run_eval(
        model, tokenizer, role, None, 0.0, pool_sequence,
        max_new_tokens, temperature, accept_threshold, proposer_fraction,
        rulebased=rulebased,
    )
    log.info("Baseline computed.")
    return baseline_score, baseline_results


def run_stage1_layer_search(
    model, tokenizer, role, dimension, direction_vectors_cache, layer_presets,
    pool_sequence, max_new_tokens, temperature, accept_threshold,
    proposer_fraction, number_of_layers, baseline_results, fixed_alpha=15.0,
    rulebased: bool = False,
):
    """Find best layer preset at a fixed alpha using Directional Fitness."""
    log.info(f"Stage 1: layer preset search  presets={layer_presets}  alpha={fixed_alpha}")
    best_preset_name = None
    best_fitness = -9999.0
    best_layers = []

    expected_dir = EXPECTED_DIRECTIONS.get(role, {}).get(dimension, 1)
    log.info(f"  -> Optimizing for Expected Direction: {'Increase' if expected_dir == 1 else 'Decrease'} metric")

    for preset_name in layer_presets:
        layers = layers_from_preset(number_of_layers, preset_name)
        direction_vectors = direction_vectors_cache[tuple(layers)]

        _, steered_results = run_eval(
            model, tokenizer, role, direction_vectors, fixed_alpha, pool_sequence,
            max_new_tokens, temperature, accept_threshold, proposer_fraction,
            rulebased=rulebased,
        )

        fitness = get_fitness_score(steered_results, baseline_results, role, dimension)

        log.info(f"  preset={preset_name:<12} layers={layers}  fitness={fitness:.2f}")
        if fitness > best_fitness:
            best_fitness = fitness
            best_preset_name = preset_name
            best_layers = layers

    log.info(f"Stage 1 best: preset={best_preset_name}  layers={best_layers}  fitness={best_fitness:.2f}")
    return best_layers, best_preset_name


def save_game_results(output_dir: Path, tag: str, role: str, alpha: float, layers, results: list):
    output_path = output_dir / f"games_{tag}.json"
    with open(output_path, "w") as file_handle:
        json.dump({
            "tag":    tag, "role":   role, "alpha":  alpha,
            "layers": layers, "games":  results,
        }, file_handle, indent=2)


def run_stage2_alpha_search(
    model, tokenizer, role, dimension, direction_vectors_cache, fixed_layers,
    pool_sequence, coarse_alphas, max_new_tokens, temperature,
    accept_threshold, proposer_fraction, baseline_score, baseline_results, output_dir: Path = None,
    rulebased: bool = False,
):
    """
    Coarse alpha search using Directional Fitness Score.
    """
    log.info(f"Stage 2 (coarse alpha search): alphas={coarse_alphas}")
    scores_per_alpha = {}

    expected_dir = EXPECTED_DIRECTIONS.get(role, {}).get(dimension, 1)
    log.info(f"  -> Optimizing for Expected Direction: {'Increase' if expected_dir == 1 else 'Decrease'} metric")

    best_alpha = coarse_alphas[0]
    best_fitness = -9999.0
    best_score_raw = baseline_score 

    for alpha in coarse_alphas:
        # If alpha is 0, we can just use the baseline
        if alpha == 0.0:
            score = baseline_score
            steered_results = baseline_results
            fitness = 0.0 # Baseline vs baseline is 0 shift
        else:
            direction_vectors = direction_vectors_cache[tuple(fixed_layers)]
            score, steered_results = run_eval(
                model, tokenizer, role, direction_vectors, alpha, pool_sequence,
                max_new_tokens, temperature, accept_threshold, proposer_fraction,
                rulebased=rulebased,
            )
            fitness = get_fitness_score(steered_results, baseline_results, role, dimension)

        full_stats = compute_full_stats(steered_results, baseline_results, role) if alpha != 0.0 else {}

        scores_per_alpha[alpha] = {
            "gridsearch_fitness":     fitness,
            "raw_metric_score":       score,
            **full_stats,
        }
        
        valid_ratio = len([r for r in steered_results if not r["parse_error"]]) / len(steered_results)
        
        log.info(
            f"  alpha={alpha:+.1f} | fitness={fitness:+.2f} | valid={valid_ratio:.0%} | raw={score:.2f}"
        )
        
        if output_dir:
            save_game_results(output_dir, f"alpha{alpha:.1f}", role, alpha, fixed_layers, steered_results)
            
        if fitness > best_fitness:
            best_fitness = fitness
            best_alpha = alpha
            best_score_raw = score

    best_beats_baseline = best_fitness > 0.0

    log.info(
        f"Stage 2 best: alpha={best_alpha:+.1f}  fitness={best_fitness:+.2f}  beats_baseline={best_beats_baseline}"
    )

    return {
        "best_alpha":           best_alpha,
        "best_fitness":         best_fitness,
        "best_raw_score":       best_score_raw,
        "baseline_score":       baseline_score,
        "best_beats_baseline":  best_beats_baseline,
        "layers":               fixed_layers,
        "scores_per_alpha":     scores_per_alpha,
        "baseline_results":     baseline_results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid search: activation steering on Ultimatum Game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",      choices=list(MODELS.keys()), required=True)
    p.add_argument("--dimension",  required=True)
    p.add_argument("--role",       choices=["proposer", "responder"], default="proposer")
    p.add_argument("--vectors_dir", default="vectors/neg15dim_12pairs_matched/negotiation")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")
    p.add_argument("--fixed_layers", nargs="+", type=int, default=None)
    p.add_argument("--presets",      nargs="+",
                   default=["late", "middle_late"],
                   choices=list(LAYER_PRESET_FRACTIONS.keys()))
    p.add_argument("--two_pass",     action="store_true")
    p.add_argument("--coarse_alphas", nargs="+", type=float, default=[0.0, -5.0, 5.0, 15.0])
    p.add_argument("--n_games",           type=int,   default=100)
    p.add_argument("--max_new_tokens",    type=int,   default=150)
    p.add_argument("--temperature",       type=float, default=0.3)
    p.add_argument("--accept_threshold",  type=float, default=0.35)
    p.add_argument("--proposer_fraction", type=float, default=0.60)
    p.add_argument("--fixed_pool",        type=int,   default=None)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--dtype",    choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--output_suffix", default="")
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--rulebased", action="store_true",
                   help="Use rule-based opponents instead of LLM baseline. "
                        "Proposer role: deterministic responder (35%% threshold). "
                        "Responder role: sweeps 10-90%% offers across each pool.")
    return p.parse_args()


def get_number_of_transformer_layers(model) -> int:
    from apply_steering_preset_nego import get_transformer_layers
    return len(get_transformer_layers(model))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model_config = MODELS[args.model]
    hf_token     = HF_TOKEN if model_config.requires_token else None
    vector_set   = Path(args.vectors_dir).parent.name

    if args.output_dir:
        output_dir = Path(args.output_dir) / args.role / args.dimension
    else:
        output_dir = Path(
            f"hyperparameter_results/ultimatum_gridsearch_{vector_set}{args.output_suffix}"
            f"/{args.model}/{args.role}/{args.dimension}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    final_best_path = output_dir / "final_best.json"
    if final_best_path.exists():
        log.info("Already complete — skipping. Delete %s to rerun.", final_best_path)
        return

    log.info("Model %s | dim=%s | role=%s", model_config.hf_id, args.dimension, args.role)

    if args.fixed_pool is not None:
        pool_sequence = [args.fixed_pool] * args.n_games
    else:
        pool_sequence = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(args.n_games)]
        random.shuffle(pool_sequence)

    log.info("Loading model: %s", model_config.hf_id)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id, token=hf_token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict = dict(token=hf_token, device_map="auto")
    if args.quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        load_kwargs["max_memory"]  = {0: "15GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_config.hf_id, **load_kwargs)
    model.eval()

    number_of_layers = get_number_of_transformer_layers(model)
    log.info("number_of_layers=%d", number_of_layers)

    if args.fixed_layers:
        layer_combinations = [tuple(args.fixed_layers)]
    else:
        layer_combinations = [tuple(layers_from_preset(number_of_layers, p)) for p in args.presets]

    direction_vectors_cache: Dict[tuple, Dict] = {}
    for layer_combo in layer_combinations:
        direction_vectors_cache[layer_combo] = load_direction_vectors(
            vectors_dir=Path(args.vectors_dir), model_alias=model_config.alias,
            dimension=args.dimension, method=args.method, layer_indices=list(layer_combo),
        )

    print(f"\n{'=' * 70}")
    print(f"ULTIMATUM GRIDSEARCH  |  model={args.model}  dim={args.dimension}  role={args.role}")
    print(f"  n_games={args.n_games}  coarse_alphas={args.coarse_alphas}"
          f"{'  rulebased=True' if args.rulebased else ''}")
    print(f"{'=' * 70}")

    # Calculate global baseline ONCE
    baseline_score, baseline_results = run_baseline(
        model, tokenizer, args.role, pool_sequence,
        args.max_new_tokens, args.temperature, args.accept_threshold, args.proposer_fraction,
        rulebased=args.rulebased,
    )
    if output_dir:
        save_game_results(output_dir, "baseline", args.role, 0.0, None, baseline_results)

    if args.fixed_layers:
        fixed_layers = args.fixed_layers
    else:
        fixed_layers, best_preset_name = run_stage1_layer_search(
            model, tokenizer, args.role, args.dimension, direction_vectors_cache, args.presets,
            pool_sequence, args.max_new_tokens, args.temperature,
            args.accept_threshold, args.proposer_fraction, number_of_layers, baseline_results,
            rulebased=args.rulebased,
        )

    search_results = run_stage2_alpha_search(
        model, tokenizer, args.role, args.dimension, direction_vectors_cache, fixed_layers,
        pool_sequence, args.coarse_alphas, args.max_new_tokens, args.temperature,
        args.accept_threshold, args.proposer_fraction, baseline_score, baseline_results,
        output_dir=output_dir, rulebased=args.rulebased,
    )

    search_results["dimension"] = args.dimension
    search_results["role"]      = args.role
    search_results["model"]     = args.model
    search_results["method"]    = args.method
    search_results["rulebased"] = args.rulebased
    search_results["timestamp"] = datetime.now().isoformat()

    # Final evaluation at best alpha
    log.info("Final eval: alpha=%+.1f  layers=%s", search_results["best_alpha"], search_results["layers"])
    if search_results["best_alpha"] == 0.0:
        final_score = baseline_score
        final_results = baseline_results
    else:
        direction_vectors_final = direction_vectors_cache[tuple(search_results["layers"])]
        final_score, final_results = run_eval(
            model, tokenizer, args.role, direction_vectors_final, search_results["best_alpha"],
            pool_sequence, args.max_new_tokens, args.temperature,
            args.accept_threshold, args.proposer_fraction,
            rulebased=args.rulebased,
        )

    search_results["final_score"] = final_score
    search_results["final_stats"] = compute_full_stats(final_results, baseline_results, args.role)
    search_results["baseline_summary"] = summarise(baseline_results)
    search_results["final_summary"]    = summarise(final_results)

    save_game_results(output_dir, "final", args.role, search_results["best_alpha"],
                      search_results["layers"], final_results)

    final_summary = summarise(final_results)
    print_summary(final_summary, args.role, search_results["best_alpha"], args.dimension)

    search_results.pop("baseline_results", None)

    with open(final_best_path, "w") as file_handle:
        json.dump(search_results, file_handle, indent=2)
    log.info("Saved: %s", final_best_path)

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()