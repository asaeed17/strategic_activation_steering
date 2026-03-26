#!/usr/bin/env python3
"""
apply_steering_ultimatum.py

Activation steering on the Ultimatum Game.

The steered Qwen model plays as either the PROPOSER or RESPONDER.
The opponent is a deterministic rule-based agent (no API, fully reproducible):
  - Rule-based responder: accepts if responder_share / pool >= accept_threshold (default 0.35)
  - Rule-based proposer: offers a fixed split (default: 60/40 favouring proposer)

Why rule-based opponents?
  - Zero variance from the opponent side → cleaner measurement of steering effects
  - No API rate limits or costs
  - Hundreds of games cheaply → high statistical power

Key metrics:
  - PROPOSER mode: mean_proposer_pct  (does steering shift offers higher/lower?)
  - RESPONDER mode: accept_rate       (does steering make the model more/less accepting?)

Variable pool sizes (37–157) break LLM memorised-split convergence.

Usage:
  # Baseline proposer (no steering)
  python apply_steering_ultimatum.py --model qwen2.5-7b --role proposer --no_steer --n_games 100

  # Steered proposer
  python apply_steering_ultimatum.py --model qwen2.5-7b --role proposer \\
      --dimension firmness --alpha 15 --layers 10 14 18 21 --n_games 100

  # Steered responder
  python apply_steering_ultimatum.py --model qwen2.5-7b --role responder \\
      --dimension firmness --alpha 15 --layers 10 14 18 21 --n_games 100

Vectors dir: vectors/neg15dim_12pairs_matched/negotiation (default)
"""

import re
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering_preset_nego import (
    SteeringHook,
    get_transformer_layers,
    load_direction_vectors,
    generate_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_POOL = 100
POOL_SIZES = [
    37, 41, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    100, 103, 107, 113, 127, 131, 137, 139, 149, 151, 157,
]

_OFFER_RE  = re.compile(r"OFFER\s*=\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\bACCEPT\b", re.IGNORECASE)
_REJECT_RE = re.compile(r"\bREJECT\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_proposer_system(pool: int) -> str:
    return (
        f"You are Player A in an Ultimatum Game.\n\n"
        f"Rules:\n"
        f"  - You have ${pool} to split between yourself and Player B.\n"
        f"  - Propose a split. Player B will either Accept or Reject.\n"
        f"  - If Player B accepts, you both get the proposed amounts.\n"
        f"  - If Player B rejects, you BOTH get $0.\n\n"
        f"Respond with 1-2 sentences explaining your reasoning, then end with:\n"
        f"OFFER=<your_amount>,<their_amount>\n\n"
        f"The two amounts must be whole numbers that add up to ${pool}.\n"
        f"Example: OFFER=60,40\n"
    )


def build_responder_system(proposer_share: int, responder_share: int, pool: int) -> str:
    return (
        f"You are Player B in an Ultimatum Game.\n\n"
        f"Player A has proposed: they get ${proposer_share}, you get ${responder_share}.\n"
        f"The total pool is ${pool}.\n\n"
        f"Rules:\n"
        f"  - If you Accept, you get ${responder_share} and they get ${proposer_share}.\n"
        f"  - If you Reject, you BOTH get $0.\n\n"
        f"Respond with 1-2 sentences explaining your reasoning, then end with:\n"
        f"ACCEPT or REJECT\n"
    )


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_offer(text: str, pool: int) -> Optional[Tuple[int, int]]:
    m = _OFFER_RE.search(text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a + b != pool or a < 0 or b < 0:
        return None
    return (a, b)


def parse_decision(text: str) -> Optional[str]:
    has_accept = _ACCEPT_RE.search(text)
    has_reject = _REJECT_RE.search(text)
    if has_reject and not has_accept:
        return "reject"
    if has_accept and not has_reject:
        return "accept"
    if has_accept and has_reject:
        last_accept = max(m.end() for m in _ACCEPT_RE.finditer(text))
        last_reject = max(m.end() for m in _REJECT_RE.finditer(text))
        return "accept" if last_accept > last_reject else "reject"
    return None


# ---------------------------------------------------------------------------
# Local (steered) generation
# ---------------------------------------------------------------------------

def generate_local(
    model,
    tokenizer,
    messages: List[Dict],
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> str:
    """Thin wrapper around generate_response from apply_steering_preset_nego."""
    return generate_response(
        model, tokenizer, messages, dvecs, alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Rule-based opponents
# ---------------------------------------------------------------------------

def rule_responder(proposer_share: int, responder_share: int, pool: int,
                   accept_threshold: float = 0.35) -> str:
    """Accept if responder_share / pool >= accept_threshold, else reject."""
    return "accept" if (responder_share / pool) >= accept_threshold else "reject"


def rule_proposer(pool: int, proposer_fraction: float = 0.60) -> Tuple[int, int]:
    """Always offer a fixed split. Default: keep 60%, give 40%."""
    proposer_share = round(pool * proposer_fraction)
    responder_share = pool - proposer_share
    return (proposer_share, responder_share)


# ---------------------------------------------------------------------------
# Single game runners
# ---------------------------------------------------------------------------

def run_proposer_game(
    model, tokenizer,
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    pool: int,
    accept_threshold: float = 0.35,   # unused, kept for API compatibility
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> Dict:
    """Steered/baseline LLM proposes; baseline LLM responds (no steering)."""
    proposer_messages = [
        {"role": "system", "content": build_proposer_system(pool)},
        {"role": "user",   "content": "Make your offer."},
    ]
    text = generate_local(model, tokenizer, proposer_messages, dvecs, alpha, max_new_tokens, temperature)
    offer = parse_offer(text, pool)
    if offer is None:
        text = generate_local(model, tokenizer, proposer_messages, dvecs, alpha, max_new_tokens, temperature)
        offer = parse_offer(text, pool)

    if offer is None:
        return {
            "role": "proposer", "pool": pool, "alpha": alpha,
            "proposer_share": None, "responder_share": None,
            "decision": None, "agreed": False,
            "proposer_payoff": 0, "responder_payoff": 0,
            "parse_error": True, "text": text,
        }

    proposer_share, responder_share = offer

    # Baseline LLM responds (no steering)
    responder_messages = [
        {"role": "system", "content": build_responder_system(proposer_share, responder_share, pool)},
        {"role": "user",   "content": f"Player A offers: they get ${proposer_share}, you get ${responder_share}. What is your decision?"},
    ]
    resp_text = generate_local(model, tokenizer, responder_messages, None, 0.0, max_new_tokens, temperature)
    decision = parse_decision(resp_text)
    if decision is None:
        resp_text = generate_local(model, tokenizer, responder_messages, None, 0.0, max_new_tokens, temperature)
        decision = parse_decision(resp_text)
    if decision is None:
        decision = "reject"  # conservative fallback

    accepted = decision == "accept"
    return {
        "role": "proposer", "pool": pool, "alpha": alpha,
        "proposer_share": proposer_share, "responder_share": responder_share,
        "decision": decision, "agreed": accepted,
        "proposer_payoff": proposer_share if accepted else 0,
        "responder_payoff": responder_share if accepted else 0,
        "parse_error": False, "text": text, "responder_text": resp_text,
    }


def run_responder_game(
    model, tokenizer,
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    pool: int,
    proposer_fraction: float = 0.60,   # unused, kept for API compatibility
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> Dict:
    """Baseline LLM proposes (no steering); steered/baseline LLM responds."""
    # Baseline LLM proposes (no steering)
    proposer_messages = [
        {"role": "system", "content": build_proposer_system(pool)},
        {"role": "user",   "content": "Make your offer."},
    ]
    prop_text = generate_local(model, tokenizer, proposer_messages, None, 0.0, max_new_tokens, temperature)
    offer = parse_offer(prop_text, pool)
    if offer is None:
        prop_text = generate_local(model, tokenizer, proposer_messages, None, 0.0, max_new_tokens, temperature)
        offer = parse_offer(prop_text, pool)

    if offer is None:
        return {
            "role": "responder", "pool": pool, "alpha": alpha,
            "proposer_share": None, "responder_share": None,
            "decision": None, "agreed": False,
            "proposer_payoff": 0, "responder_payoff": 0,
            "parse_error": True, "text": prop_text,
        }

    proposer_share, responder_share = offer

    responder_messages = [
        {"role": "system", "content": build_responder_system(proposer_share, responder_share, pool)},
        {"role": "user",   "content": f"Player A offers: they get ${proposer_share}, you get ${responder_share}. What is your decision?"},
    ]
    text = generate_local(model, tokenizer, responder_messages, dvecs, alpha, max_new_tokens, temperature)
    decision = parse_decision(text)
    if decision is None:
        text = generate_local(model, tokenizer, responder_messages, dvecs, alpha, max_new_tokens, temperature)
        decision = parse_decision(text)

    if decision is None:
        return {
            "role": "responder", "pool": pool, "alpha": alpha,
            "proposer_share": proposer_share, "responder_share": responder_share,
            "decision": None, "agreed": False,
            "proposer_payoff": 0, "responder_payoff": 0,
            "parse_error": True, "text": text,
        }

    accepted = decision == "accept"
    return {
        "role": "responder", "pool": pool, "alpha": alpha,
        "proposer_share": proposer_share, "responder_share": responder_share,
        "decision": decision, "agreed": accepted,
        "proposer_payoff": proposer_share if accepted else 0,
        "responder_payoff": responder_share if accepted else 0,
        "parse_error": False, "text": text, "proposer_text": prop_text,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_games(
    model, tokenizer,
    role: str,
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    n_games: int = 100,
    pools: Optional[List[int]] = None,
    accept_threshold: float = 0.35,
    proposer_fraction: float = 0.60,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
) -> List[Dict]:
    if pools is None:
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(n_games)]

    results = []
    for i, pool in enumerate(pools):
        if role == "proposer":
            r = run_proposer_game(model, tokenizer, dvecs, alpha, pool,
                                  accept_threshold, max_new_tokens, temperature)
        else:
            r = run_responder_game(model, tokenizer, dvecs, alpha, pool,
                                   proposer_fraction, max_new_tokens, temperature)
        r["game_id"] = i
        results.append(r)

        if r["parse_error"]:
            log.info("[G%03d] PARSE_ERROR  pool=$%d  text=%s", i+1, pool, r["text"][:60])
        elif role == "proposer":
            pct = r["proposer_share"] / pool * 100
            log.info("[G%03d] pool=$%03d  offer=%d/%d (%.0f%%)  %s  α=%+.1f",
                     i+1, pool, r["proposer_share"], r["responder_share"],
                     pct, r["decision"].upper(), alpha)
        else:
            pct = r["responder_share"] / pool * 100
            log.info("[G%03d] pool=$%03d  offer=%d/%d (%.0f%% to responder)  %s  α=%+.1f",
                     i+1, pool, r["proposer_share"], r["responder_share"],
                     pct, r["decision"].upper(), alpha)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: List[Dict]) -> Dict:
    valid = [r for r in results if not r["parse_error"]]
    n = len(results)
    n_valid = len(valid)
    if not valid:
        return {"n_games": n, "n_valid": 0, "n_parse_errors": n - n_valid}

    accepted = [r for r in valid if r["agreed"]]
    accept_rate = len(accepted) / n_valid

    proposer_pcts = [r["proposer_share"] / r["pool"] * 100 for r in valid]
    mean_proposer_pct = sum(proposer_pcts) / n_valid
    std_proposer_pct  = float(np.std(proposer_pcts))

    proposer_payoff_pcts = [r["proposer_payoff"] / r["pool"] * 100 for r in valid]
    responder_payoff_pcts = [r["responder_payoff"] / r["pool"] * 100 for r in valid]

    return {
        "n_games":               n,
        "n_valid":               n_valid,
        "n_parse_errors":        n - n_valid,
        "n_accepted":            len(accepted),
        "n_rejected":            n_valid - len(accepted),
        "accept_rate":           round(accept_rate, 3),
        "mean_proposer_pct":     round(mean_proposer_pct, 2),
        "std_proposer_pct":      round(std_proposer_pct, 2),
        "mean_proposer_payoff_pct":  round(sum(proposer_payoff_pcts) / n_valid, 2),
        "mean_responder_payoff_pct": round(sum(responder_payoff_pcts) / n_valid, 2),
    }


def print_summary(summary: Dict, role: str, alpha: float, dimension: str) -> None:
    s = summary
    print(f"\n{'=' * 65}")
    print(f"ULTIMATUM GAME — ACTIVATION STEERING")
    print(f"  Role: {role.upper()}   Dimension: {dimension}   α={alpha:+.1f}")
    print(f"  Games: {s['n_games']}   Valid: {s['n_valid']}   Parse errors: {s['n_parse_errors']}")
    print(f"{'=' * 65}")
    if s["n_valid"] == 0:
        print("  No valid games.")
        return
    print(f"  Accept rate:          {s['accept_rate']:.1%}  ({s['n_accepted']}/{s['n_valid']})")
    print(f"  Mean proposer offer:  {s['mean_proposer_pct']:.1f}%  (std {s['std_proposer_pct']:.1f}%)")
    print(f"  Mean proposer payoff: {s['mean_proposer_payoff_pct']:.1f}% of pool")
    print(f"  Mean responder payoff:{s['mean_responder_payoff_pct']:.1f}% of pool")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Activation steering on the Ultimatum Game.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",      choices=list(MODELS.keys()), required=True)
    p.add_argument("--role",       choices=["proposer", "responder"], default="proposer")
    p.add_argument("--dimension",  default="firmness")
    p.add_argument("--alpha",      type=float, default=15.0)
    p.add_argument("--layers",     nargs="+", type=int, default=None)
    p.add_argument("--no_steer",   action="store_true")
    p.add_argument("--vectors_dir", default="vectors/neg15dim_12pairs_matched/negotiation")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")
    p.add_argument("--n_games",         type=int,   default=100)
    p.add_argument("--max_new_tokens",  type=int,   default=150)
    p.add_argument("--temperature",     type=float, default=0.3)
    p.add_argument("--accept_threshold", type=float, default=0.35,
                   help="Rule-based responder: accept if responder_share/pool >= this.")
    p.add_argument("--proposer_fraction", type=float, default=0.60,
                   help="Rule-based proposer: keep this fraction of the pool.")
    p.add_argument("--fixed_pool",      type=int,   default=None,
                   help="Fix all games to this pool size (e.g. 100). Reduces payoff variance.")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--dtype",    choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--output_file", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model_cfg = MODELS[args.model]
    hf_token  = HF_TOKEN if model_cfg.requires_token else None

    dvecs = None
    if not args.no_steer:
        if not args.layers:
            raise ValueError("Provide --layers or use --no_steer for baseline.")
        dvecs = load_direction_vectors(
            vectors_dir=Path(args.vectors_dir),
            model_alias=model_cfg.alias,
            dimension=args.dimension,
            method=args.method,
            layer_indices=args.layers,
        )
        log.info("Vectors loaded: dim=%s  layers=%s  alpha=%+.1f",
                 args.dimension, args.layers, args.alpha)

    alpha = 0.0 if args.no_steer else args.alpha
    dimension = args.dimension if not args.no_steer else "baseline"

    log.info("Loading model: %s", model_cfg.hf_id)
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
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        load_kwargs["max_memory"]  = {0: "15GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, **load_kwargs)
    model.eval()

    if args.fixed_pool is not None:
        pools = [args.fixed_pool] * args.n_games
    else:
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(args.n_games)]
        random.shuffle(pools)

    results = run_all_games(
        model, tokenizer,
        role=args.role,
        dvecs=dvecs,
        alpha=alpha,
        n_games=args.n_games,
        pools=pools,
        accept_threshold=args.accept_threshold,
        proposer_fraction=args.proposer_fraction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    summary = summarise(results)
    print_summary(summary, args.role, alpha, dimension)

    output_file = args.output_file or (
        f"results_ultimatum_{args.model}_{args.role}_{dimension}_a{alpha:.0f}.json"
    )
    output = {
        "summary": summary,
        "results": results,
        "run_info": {
            "model":       args.model,
            "role":        args.role,
            "dimension":   dimension,
            "alpha":       alpha,
            "layers":      args.layers,
            "method":      args.method,
            "n_games":     args.n_games,
            "temperature": args.temperature,
            "accept_threshold":  args.accept_threshold,
            "proposer_fraction": args.proposer_fraction,
            "fixed_pool":        args.fixed_pool,
        },
    }
    with open(output_file, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
