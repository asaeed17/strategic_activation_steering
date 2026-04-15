#!/usr/bin/env python3
"""
ultimatum_game.py — Activation-steered Ultimatum Game / Dictator Game experiments.

GPU steering version: loads a local model (e.g. Qwen 2.5-7B), registers
SteeringHook forward hooks on transformer layers, and runs Ultimatum Games
or Dictator Games with a steered agent vs a baseline (same model, no hooks).

Dictator Game mode (--game dictator): the proposer's offer is automatically
accepted — the responder has no decision. This isolates proposer behavior
without responder acceptance/rejection confounds.

Paired design:
  - Proposer experiments: for each pool size, generate BOTH steered and
    baseline offers, then run each through the unsteered responder.
  - Responder experiments: generate baseline offers first, then feed the
    SAME offers to both steered and unsteered responder.

This eliminates between-game variance and enables paired t-tests / McNemar's.

Usage:
  # Baseline (no steering)
  python ultimatum_game.py --model qwen2.5-7b --n_games 100 --variable_pools \
      --output_dir results/ultimatum/ --paired

  # Steer proposer with firmness
  python ultimatum_game.py --model qwen2.5-7b --dimension firmness \
      --layers 14 --alpha 6 --steered_role proposer --n_games 100 \
      --variable_pools --paired --output_dir results/ultimatum/

  # Steer responder with empathy
  python ultimatum_game.py --model qwen2.5-7b --dimension empathy \
      --layers 14 --alpha 6 --steered_role responder --n_games 100 \
      --variable_pools --paired --output_dir results/ultimatum/
"""

import re
import json
import random
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import MODELS, HF_TOKEN
from apply_steering import SteeringHook, load_direction_vectors, get_transformer_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from playground/run_ultimatum.py)
# ---------------------------------------------------------------------------

DEFAULT_POOL = 100

# 24 original prime-heavy pools (legacy, kept for backward compat)
POOL_SIZES_24 = [
    37, 41, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    100, 103, 107, 113, 127, 131, 137, 139, 149, 151, 157,
]

# 200 structurally diverse pools ($37-$273): primes, evens, odds, mult-of-5/10.
# Avoids primes-only bias (blocks 50/50 splits, unusual tokenization).
# One game per pool at temp=0 gives honest n=200 for paired tests.
POOL_SIZES = [
    37, 38, 39, 40, 41, 42, 43, 44, 47, 48,
    49, 50, 51, 52, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 68, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 85, 86, 87, 89, 90, 91, 92, 93,
    94, 95, 96, 97, 98, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 110, 111, 112, 114, 116, 118,
    119, 120, 123, 124, 126, 128, 131, 132, 133, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
    147, 148, 149, 150, 151, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    179, 180, 181, 182, 183, 185, 186, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 215, 216, 217, 218, 219, 220, 221,
    222, 223, 225, 226, 227, 228, 229, 230, 231, 233,
    235, 237, 238, 239, 240, 241, 242, 243, 245, 246,
    247, 248, 249, 250, 251, 253, 255, 257, 258, 259,
    260, 261, 262, 263, 265, 267, 269, 270, 271, 273,
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def build_proposer_system(pool: int = DEFAULT_POOL, game: str = "ultimatum") -> str:
    if game == "dictator":
        return (
            f"You are Player A in a Dictator Game.\n"
            f"\n"
            f"Rules:\n"
            f"  - You have ${pool} to split between yourself and Player B.\n"
            f"  - Your offer is automatically accepted. Player B cannot reject.\n"
            f"  - You decide how much to keep and how much to give.\n"
            f"\n"
            f"Respond with 1-2 sentences explaining your reasoning, then end with:\n"
            f"OFFER=<your_amount>,<their_amount>\n"
            f"\n"
            f"The two amounts must be whole numbers that add up to ${pool}.\n"
            f"Example: OFFER=60,40\n"
        )
    return (
        f"You are Player A in an Ultimatum Game.\n"
        f"\n"
        f"Rules:\n"
        f"  - You have ${pool} to split between yourself and Player B.\n"
        f"  - Propose a split. Player B will either Accept or Reject.\n"
        f"  - If Player B accepts, you both get the proposed amounts.\n"
        f"  - If Player B rejects, you BOTH get $0.\n"
        f"\n"
        f"Respond with 1-2 sentences explaining your reasoning, then end with:\n"
        f"OFFER=<your_amount>,<their_amount>\n"
        f"\n"
        f"The two amounts must be whole numbers that add up to ${pool}.\n"
        f"Example: OFFER=60,40\n"
    )


def build_responder_system(
    proposer_share: int, responder_share: int, pool: int = DEFAULT_POOL,
    proposer_text: Optional[str] = None,
) -> str:
    text = "You are Player B in an Ultimatum Game.\n\n"
    if proposer_text:
        text += (
            f"Player A said:\n"
            f'"{proposer_text}"\n\n'
        )
    text += (
        f"Player A has proposed: they get ${proposer_share}, you get ${responder_share}.\n"
        f"The total pool is ${pool}.\n"
        f"\n"
        f"Rules:\n"
        f"  - If you Accept, you get ${responder_share} and they get ${proposer_share}.\n"
        f"  - If you Reject, you BOTH get $0.\n"
        f"\n"
        f"Respond with 1-2 sentences explaining your reasoning, "
        f"then state your final decision as a single word: ACCEPT or REJECT.\n"
    )
    return text


# ---------------------------------------------------------------------------
# Prompt enhancements (matching the 8 steering vector dimensions)
# ---------------------------------------------------------------------------

PROPOSER_ENHANCEMENTS = {
    "firmness": (
        "\nStrategy — FIRMNESS:\n"
        "  - State your position clearly and confidently.\n"
        "  - Do not second-guess yourself or hedge.\n"
        "  - You deserve a strong share — stand firm on what you want.\n"
    ),
    "empathy": (
        "\nStrategy — EMPATHY:\n"
        "  - Consider Player B's perspective — they need enough to feel respected.\n"
        "  - Frame your offer as mutually beneficial.\n"
        "  - Show you care about both sides getting a good outcome.\n"
    ),
    "anchoring": (
        "\nStrategy — ANCHORING:\n"
        "  - Set the terms aggressively. Your initial offer frames the entire negotiation.\n"
        "  - Anchor high — demand a large share so even a 'compromise' favors you.\n"
        "  - Player B's expectations will adjust to your anchor.\n"
    ),
    "batna_awareness": (
        "\nStrategy — BATNA AWARENESS:\n"
        "  - Remember: if Player B rejects, you both get $0. Use this leverage.\n"
        "  - Frame your offer as the best option available to Player B.\n"
        "  - Make it clear that rejection hurts them more than it hurts you.\n"
    ),
}

RESPONDER_ENHANCEMENTS = {
    "firmness": (
        "\nStrategy — FIRMNESS:\n"
        "  - Reject any offer that gives you less than 40% of the pool.\n"
        "  - Unfair offers should be punished even at personal cost.\n"
        "  - You'd rather get $0 than accept an insulting split.\n"
    ),
    "empathy": (
        "\nStrategy — EMPATHY:\n"
        "  - Consider that Player A is also trying to do well.\n"
        "  - Even a smaller share is better than $0 for both of you.\n"
        "  - Be understanding of their position.\n"
    ),
    "anchoring": (
        "\nStrategy — ANCHORING:\n"
        "  - Judge this offer against a fair 50/50 baseline.\n"
        "  - Anything below 40% for you is anchored too far in their favor.\n"
        "  - Don't let their aggressive framing shift your standards.\n"
    ),
    "batna_awareness": (
        "\nStrategy — BATNA AWARENESS:\n"
        "  - Your alternative is $0. Any positive amount beats your BATNA.\n"
        "  - But also: Player A's alternative is $0 too. You have equal leverage.\n"
        "  - Reject only if the offer is truly insulting — your walkaway is $0.\n"
    ),
}

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_OFFER_RE = re.compile(r"OFFER\s*=\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\bACCEPT\b", re.IGNORECASE)
_REJECT_RE = re.compile(r"\bREJECT\b", re.IGNORECASE)


def parse_offer(text: str, pool: int = DEFAULT_POOL) -> Optional[Tuple[int, int]]:
    m = _OFFER_RE.search(text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a + b != pool or a < 0 or b < 0:
        return None
    return (a, b)


_INSTRUCTION_ECHO_RE = re.compile(r"\bACCEPT\s+or\s+REJECT\b", re.IGNORECASE)


def parse_response(text: str) -> Optional[str]:
    # Strip echoed instruction phrase ("ACCEPT or REJECT") before parsing —
    # smaller models (3B) echo this literally as deliberation text.
    cleaned = _INSTRUCTION_ECHO_RE.sub("", text)
    has_accept = _ACCEPT_RE.search(cleaned)
    has_reject = _REJECT_RE.search(cleaned)
    if has_reject and not has_accept:
        return "reject"
    if has_accept and not has_reject:
        return "accept"
    if has_accept and has_reject:
        last_accept = max(m.end() for m in _ACCEPT_RE.finditer(cleaned))
        last_reject = max(m.end() for m in _REJECT_RE.finditer(cleaned))
        return "accept" if last_accept > last_reject else "reject"
    return None


# ---------------------------------------------------------------------------
# Behavioral metrics extraction
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(r"\b(maybe|perhaps|might|could|I think|probably|possibly)\b", re.IGNORECASE)
_FAIR_RE = re.compile(r"\b(fair|equal|both|reasonable|mutual|together)\b", re.IGNORECASE)


def extract_behavioral_metrics(text: str) -> Dict:
    if not text:
        return {"word_count": 0, "hedge_count": 0, "fairness_count": 0}
    words = text.split()
    return {
        "word_count": len(words),
        "hedge_count": len(_HEDGE_RE.findall(text)),
        "fairness_count": len(_FAIR_RE.findall(text)),
    }


# ---------------------------------------------------------------------------
# Generation with steering hooks
# ---------------------------------------------------------------------------

def generate_steered(
    model,
    tokenizer,
    messages: List[Dict],
    direction_vectors: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """Generate text with optional steering hooks. If direction_vectors is None
    or alpha is 0, generates without steering (baseline)."""
    device = next(model.parameters()).device
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    hooks: List[SteeringHook] = []
    if direction_vectors and alpha != 0.0:
        layers = get_transformer_layers(model)
        for layer_idx, vec in direction_vectors.items():
            if layer_idx >= len(layers):
                continue
            dt = torch.tensor(vec, dtype=torch.float32, device=device)
            h = SteeringHook(direction=dt, alpha=alpha)
            h.register(layers[layer_idx])
            hooks.append(h)

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch generation with steering hooks
# ---------------------------------------------------------------------------

def generate_steered_batch(
    model,
    tokenizer,
    messages_list: List[List[Dict]],
    direction_vectors: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> List[str]:
    """Batch version of generate_steered. Processes multiple prompts in one
    forward pass. Uses left-padding so all sequences align on the right."""
    device = next(model.parameters()).device
    formatted = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    hooks: List[SteeringHook] = []
    if direction_vectors and alpha != 0.0:
        layers = get_transformer_layers(model)
        for layer_idx, vec in direction_vectors.items():
            if layer_idx >= len(layers):
                continue
            dt = torch.tensor(vec, dtype=torch.float32, device=device)
            h = SteeringHook(direction=dt, alpha=alpha)
            h.register(layers[layer_idx])
            hooks.append(h)

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in hooks:
            h.remove()

    return [
        tokenizer.decode(out_ids[i, input_len:], skip_special_tokens=True).strip()
        for i in range(len(messages_list))
    ]


# ---------------------------------------------------------------------------
# Single game (one condition: steered or baseline)
# ---------------------------------------------------------------------------

def _run_single_game(
    model,
    tokenizer,
    pool: int,
    steered_role: Optional[str],
    direction_vectors: Optional[Dict[int, np.ndarray]],
    alpha: float,
    proposer_enhancement: Optional[str] = None,
    responder_enhancement: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
    fixed_offer: Optional[Tuple[int, int]] = None,
    game: str = "ultimatum",
    text_visible: bool = False,
) -> Dict:
    """Run a single UG or Dictator Game. If fixed_offer is set, skip proposer
    generation and use the given offer (for paired responder experiments).
    If game=='dictator', skip responder entirely and auto-accept the offer.
    If text_visible, the responder sees the proposer's full reasoning text."""

    if fixed_offer is not None:
        proposer_share, responder_share = fixed_offer
        proposer_text = f"[fixed offer] OFFER={proposer_share},{responder_share}"
    else:
        # Build proposer prompt
        proposer_sys = build_proposer_system(pool, game=game)
        if proposer_enhancement and proposer_enhancement in PROPOSER_ENHANCEMENTS:
            proposer_sys += PROPOSER_ENHANCEMENTS[proposer_enhancement]

        proposer_msgs = [
            {"role": "system", "content": proposer_sys},
            {"role": "user", "content": "Make your offer."},
        ]

        # Steer proposer?
        dvecs = direction_vectors if steered_role == "proposer" else None
        a = alpha if steered_role == "proposer" else 0.0

        proposer_text = generate_steered(model, tokenizer, proposer_msgs, dvecs, a,
                                         max_new_tokens, temperature)
        offer = parse_offer(proposer_text, pool)

        if offer is None:
            # Retry once
            proposer_text = generate_steered(model, tokenizer, proposer_msgs, dvecs, a,
                                             max_new_tokens, temperature)
            offer = parse_offer(proposer_text, pool)

        if offer is None:
            return {
                "agreed": False, "proposer_share": None, "responder_share": None,
                "response": None, "proposer_payoff": 0, "responder_payoff": 0,
                "parse_error": "proposer", "proposer_text": proposer_text,
                "responder_text": None, "pool": pool,
                "proposer_metrics": extract_behavioral_metrics(proposer_text),
                "responder_metrics": extract_behavioral_metrics(""),
            }
        proposer_share, responder_share = offer

    # Dictator game: skip responder, auto-accept
    if game == "dictator":
        return {
            "agreed": True,
            "proposer_share": proposer_share,
            "responder_share": responder_share,
            "response": "accept",
            "proposer_payoff": proposer_share,
            "responder_payoff": responder_share,
            "parse_error": None,
            "proposer_text": proposer_text,
            "responder_text": "[dictator game: auto-accept]",
            "pool": pool,
            "proposer_metrics": extract_behavioral_metrics(proposer_text),
            "responder_metrics": extract_behavioral_metrics(""),
        }

    # Build responder prompt
    responder_sys = build_responder_system(
        proposer_share, responder_share, pool,
        proposer_text=proposer_text if text_visible else None,
    )
    if responder_enhancement and responder_enhancement in RESPONDER_ENHANCEMENTS:
        responder_sys += RESPONDER_ENHANCEMENTS[responder_enhancement]

    responder_msgs = [
        {"role": "system", "content": responder_sys},
        {"role": "user", "content": (
            f"Player A offers: they get ${proposer_share}, you get ${responder_share}. "
            f"What is your decision?"
        )},
    ]

    # Steer responder?
    dvecs = direction_vectors if steered_role == "responder" else None
    a = alpha if steered_role == "responder" else 0.0

    responder_text = generate_steered(model, tokenizer, responder_msgs, dvecs, a,
                                      max_new_tokens, temperature)
    decision = parse_response(responder_text)

    if decision is None:
        responder_text = generate_steered(model, tokenizer, responder_msgs, dvecs, a,
                                          max_new_tokens, temperature)
        decision = parse_response(responder_text)

    if decision is None:
        return {
            "agreed": False, "proposer_share": proposer_share,
            "responder_share": responder_share, "response": None,
            "proposer_payoff": 0, "responder_payoff": 0,
            "parse_error": "responder", "proposer_text": proposer_text,
            "responder_text": responder_text, "pool": pool,
            "proposer_metrics": extract_behavioral_metrics(proposer_text),
            "responder_metrics": extract_behavioral_metrics(responder_text),
        }

    accepted = decision == "accept"
    return {
        "agreed": accepted,
        "proposer_share": proposer_share,
        "responder_share": responder_share,
        "response": decision,
        "proposer_payoff": proposer_share if accepted else 0,
        "responder_payoff": responder_share if accepted else 0,
        "parse_error": None,
        "proposer_text": proposer_text,
        "responder_text": responder_text,
        "pool": pool,
        "proposer_metrics": extract_behavioral_metrics(proposer_text),
        "responder_metrics": extract_behavioral_metrics(responder_text),
    }


# ---------------------------------------------------------------------------
# Paired game (steered + baseline on same pool / same offer)
# ---------------------------------------------------------------------------

def run_paired_game(
    model,
    tokenizer,
    pool: int,
    steered_role: str,
    direction_vectors: Dict[int, np.ndarray],
    alpha: float,
    proposer_enhancement: Optional[str] = None,
    responder_enhancement: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
    game: str = "ultimatum",
    text_visible: bool = False,
) -> Dict:
    """Run a paired game: steered condition + baseline condition on the same pool.

    For proposer steering: generate steered offer and baseline offer independently,
    then run each through the unsteered responder.

    For responder steering: generate a baseline offer, then run it through both
    steered and unsteered responder.
    """
    if game == "dictator" and steered_role == "responder":
        raise ValueError("Dictator game has no responder decision — cannot steer responder role.")

    if steered_role == "proposer":
        # Steered proposer
        steered_result = _run_single_game(
            model, tokenizer, pool,
            steered_role="proposer",
            direction_vectors=direction_vectors,
            alpha=alpha,
            proposer_enhancement=proposer_enhancement,
            responder_enhancement=responder_enhancement,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            game=game,
            text_visible=text_visible,
        )
        # Baseline proposer (same pool, no steering)
        baseline_result = _run_single_game(
            model, tokenizer, pool,
            steered_role=None,
            direction_vectors=None,
            alpha=0.0,
            proposer_enhancement=proposer_enhancement,
            responder_enhancement=responder_enhancement,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            game=game,
            text_visible=text_visible,
        )

    elif steered_role == "responder":
        # First: generate baseline offer (no steering)
        baseline_result = _run_single_game(
            model, tokenizer, pool,
            steered_role=None,
            direction_vectors=None,
            alpha=0.0,
            proposer_enhancement=proposer_enhancement,
            responder_enhancement=responder_enhancement,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            game=game,
        )
        # If baseline proposer failed to parse, can't run paired responder
        if baseline_result["parse_error"] == "proposer":
            return {
                "game_id": None, "pool": pool,
                "steered": baseline_result,
                "baseline": baseline_result,
                "parse_error": "proposer",
            }
        # Feed the SAME offer to steered responder
        offer = (baseline_result["proposer_share"], baseline_result["responder_share"])
        steered_result = _run_single_game(
            model, tokenizer, pool,
            steered_role="responder",
            direction_vectors=direction_vectors,
            alpha=alpha,
            proposer_enhancement=proposer_enhancement,
            responder_enhancement=responder_enhancement,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            fixed_offer=offer,
            game=game,
            text_visible=text_visible,
        )
    else:
        raise ValueError(f"steered_role must be 'proposer' or 'responder', got '{steered_role}'")

    return {
        "game_id": None,  # set by caller
        "pool": pool,
        "steered": steered_result,
        "baseline": baseline_result,
        "parse_error": None,
    }


# ---------------------------------------------------------------------------
# Batch paired game (proposer steering only)
# ---------------------------------------------------------------------------

def run_paired_proposer_batch(
    model,
    tokenizer,
    pools: List[int],
    direction_vectors: Dict[int, np.ndarray],
    alpha: float,
    proposer_enhancement: Optional[str] = None,
    responder_enhancement: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
    game: str = "ultimatum",
    text_visible: bool = False,
) -> List[Dict]:
    """Batch version of run_paired_game for proposer steering.
    Runs 4 batched generate calls per batch:
      1. steered proposer (all pools)
      2. baseline proposer (all pools)
      3. steered responders (valid games only)
      4. baseline responders (valid games only)
    Falls back to sequential generate_steered for individual parse-error retries.
    """
    def build_proposer_msgs(pool):
        sys = build_proposer_system(pool, game=game)
        if proposer_enhancement and proposer_enhancement in PROPOSER_ENHANCEMENTS:
            sys += PROPOSER_ENHANCEMENTS[proposer_enhancement]
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": "Make your offer."},
        ]

    def build_responder_msgs(ps, rs, pool, prop_text):
        sys = build_responder_system(
            ps, rs, pool, proposer_text=prop_text if text_visible else None
        )
        if responder_enhancement and responder_enhancement in RESPONDER_ENHANCEMENTS:
            sys += RESPONDER_ENHANCEMENTS[responder_enhancement]
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": (
                f"Player A offers: they get ${ps}, you get ${rs}. "
                f"What is your decision?"
            )},
        ]

    prop_msgs = [build_proposer_msgs(p) for p in pools]

    # --- 1+2: batch generate proposer outputs ---
    steered_prop_texts = generate_steered_batch(
        model, tokenizer, prop_msgs, direction_vectors, alpha, max_new_tokens, temperature
    )
    baseline_prop_texts = generate_steered_batch(
        model, tokenizer, prop_msgs, None, 0.0, max_new_tokens, temperature
    )

    # Parse offers; retry individually on failure
    steered_offers, baseline_offers = [], []
    for i, pool in enumerate(pools):
        s_off = parse_offer(steered_prop_texts[i], pool)
        if s_off is None:
            retry = generate_steered(model, tokenizer, prop_msgs[i], direction_vectors,
                                     alpha, max_new_tokens, temperature)
            steered_prop_texts[i] = retry
            s_off = parse_offer(retry, pool)
        steered_offers.append(s_off)

        b_off = parse_offer(baseline_prop_texts[i], pool)
        if b_off is None:
            retry = generate_steered(model, tokenizer, prop_msgs[i], None,
                                     0.0, max_new_tokens, temperature)
            baseline_prop_texts[i] = retry
            b_off = parse_offer(retry, pool)
        baseline_offers.append(b_off)

    # Dictator game: skip responder
    if game == "dictator":
        results = []
        for i, pool in enumerate(pools):
            def _dg_result(offer, text):
                if offer is None:
                    return {"agreed": False, "proposer_share": None, "responder_share": None,
                            "response": None, "proposer_payoff": 0, "responder_payoff": 0,
                            "parse_error": "proposer", "proposer_text": text,
                            "responder_text": None, "pool": pool,
                            "proposer_metrics": extract_behavioral_metrics(text),
                            "responder_metrics": extract_behavioral_metrics("")}
                ps, rs = offer
                return {"agreed": True, "proposer_share": ps, "responder_share": rs,
                        "response": "accept", "proposer_payoff": ps, "responder_payoff": rs,
                        "parse_error": None, "proposer_text": text,
                        "responder_text": "[dictator game: auto-accept]", "pool": pool,
                        "proposer_metrics": extract_behavioral_metrics(text),
                        "responder_metrics": extract_behavioral_metrics("")}
            results.append({
                "game_id": None, "pool": pool,
                "steered": _dg_result(steered_offers[i], steered_prop_texts[i]),
                "baseline": _dg_result(baseline_offers[i], baseline_prop_texts[i]),
                "parse_error": None,
            })
        return results

    # --- 3+4: batch generate responder outputs ---
    # Build responder messages only for games with valid offers
    s_resp_items: List[Tuple[int, List[Dict]]] = []
    b_resp_items: List[Tuple[int, List[Dict]]] = []
    for i, pool in enumerate(pools):
        if steered_offers[i] is not None:
            ps, rs = steered_offers[i]
            s_resp_items.append((i, build_responder_msgs(ps, rs, pool, steered_prop_texts[i])))
        if baseline_offers[i] is not None:
            pb, rb = baseline_offers[i]
            b_resp_items.append((i, build_responder_msgs(pb, rb, pool, baseline_prop_texts[i])))

    s_resp_texts_raw = generate_steered_batch(
        model, tokenizer, [m for _, m in s_resp_items], None, 0.0, max_new_tokens, temperature
    ) if s_resp_items else []
    b_resp_texts_raw = generate_steered_batch(
        model, tokenizer, [m for _, m in b_resp_items], None, 0.0, max_new_tokens, temperature
    ) if b_resp_items else []

    s_resp_by_idx = {idx: (text, msg) for (idx, msg), text in zip(s_resp_items, s_resp_texts_raw)}
    b_resp_by_idx = {idx: (text, msg) for (idx, msg), text in zip(b_resp_items, b_resp_texts_raw)}

    # Assemble results
    results = []
    for i, pool in enumerate(pools):
        def _make_result(offer, prop_text, resp_by_idx):
            if offer is None:
                return {"agreed": False, "proposer_share": None, "responder_share": None,
                        "response": None, "proposer_payoff": 0, "responder_payoff": 0,
                        "parse_error": "proposer", "proposer_text": prop_text,
                        "responder_text": None, "pool": pool,
                        "proposer_metrics": extract_behavioral_metrics(prop_text),
                        "responder_metrics": extract_behavioral_metrics("")}
            ps, rs = offer
            resp_text, resp_msgs = resp_by_idx[i]
            decision = parse_response(resp_text)
            if decision is None:
                resp_text = generate_steered(model, tokenizer, resp_msgs, None, 0.0,
                                             max_new_tokens, temperature)
                decision = parse_response(resp_text)
            accepted = decision == "accept" if decision else False
            return {"agreed": accepted, "proposer_share": ps, "responder_share": rs,
                    "response": decision,
                    "proposer_payoff": ps if accepted else 0,
                    "responder_payoff": rs if accepted else 0,
                    "parse_error": None if decision else "responder",
                    "proposer_text": prop_text, "responder_text": resp_text, "pool": pool,
                    "proposer_metrics": extract_behavioral_metrics(prop_text),
                    "responder_metrics": extract_behavioral_metrics(resp_text)}

        results.append({
            "game_id": None, "pool": pool,
            "steered": _make_result(steered_offers[i], steered_prop_texts[i], s_resp_by_idx),
            "baseline": _make_result(baseline_offers[i], baseline_prop_texts[i], b_resp_by_idx),
            "parse_error": None,
        })
    return results


# ---------------------------------------------------------------------------
# Unpaired game (baseline only, no steering)
# ---------------------------------------------------------------------------

def run_baseline_game(
    model,
    tokenizer,
    pool: int,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
    game: str = "ultimatum",
    text_visible: bool = False,
) -> Dict:
    result = _run_single_game(
        model, tokenizer, pool,
        steered_role=None,
        direction_vectors=None,
        alpha=0.0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        game=game,
        text_visible=text_visible,
    )
    return {"game_id": None, "pool": pool, "result": result}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_paired(games: List[Dict], steered_role: str) -> Dict:
    """Compute paired statistics for steered vs baseline."""
    from scipy import stats

    valid = [g for g in games if g.get("parse_error") is None]
    n_total = len(games)
    n_valid = len(valid)
    n_parse_errors = n_total - n_valid

    if not valid:
        return {"n_games": n_total, "n_valid": 0, "n_parse_errors": n_parse_errors}

    if steered_role == "proposer":
        # Both steered and baseline must have valid proposer offers
        usable = [g for g in valid
                  if g["steered"]["parse_error"] is None
                  and g["baseline"]["parse_error"] is None
                  and g["steered"]["proposer_share"] is not None
                  and g["baseline"]["proposer_share"] is not None]

        if not usable:
            return {"n_games": n_total, "n_valid": 0, "n_parse_errors": n_parse_errors}

        steered_pcts = [g["steered"]["proposer_share"] / g["pool"] * 100 for g in usable]
        baseline_pcts = [g["baseline"]["proposer_share"] / g["pool"] * 100 for g in usable]
        deltas = [s - b for s, b in zip(steered_pcts, baseline_pcts)]

        steered_accepted = [g for g in usable if g["steered"]["agreed"]]
        baseline_accepted = [g for g in usable if g["baseline"]["agreed"]]

        # Payoffs as pct of pool
        steered_payoffs = [g["steered"]["proposer_payoff"] / g["pool"] * 100 for g in usable]
        baseline_payoffs = [g["baseline"]["proposer_payoff"] / g["pool"] * 100 for g in usable]

        # Paired t-test
        if len(deltas) >= 2:
            t_stat, p_val = stats.ttest_rel(steered_pcts, baseline_pcts)
            mean_delta = sum(deltas) / len(deltas)
            std_delta = (sum((d - mean_delta) ** 2 for d in deltas) / max(len(deltas) - 1, 1)) ** 0.5
            cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0
        else:
            t_stat, p_val, cohens_d = 0.0, 1.0, 0.0

        # Behavioral metrics
        steered_wc = [g["steered"]["proposer_metrics"]["word_count"] for g in usable]
        baseline_wc = [g["baseline"]["proposer_metrics"]["word_count"] for g in usable]
        steered_hedge = [g["steered"]["proposer_metrics"]["hedge_count"] for g in usable]
        baseline_hedge = [g["baseline"]["proposer_metrics"]["hedge_count"] for g in usable]
        steered_fair = [g["steered"]["proposer_metrics"]["fairness_count"] for g in usable]
        baseline_fair = [g["baseline"]["proposer_metrics"]["fairness_count"] for g in usable]

        return {
            "n_games": n_total,
            "n_valid": n_valid,
            "n_usable_pairs": len(usable),
            "n_parse_errors": n_parse_errors,
            "steered_role": steered_role,
            # Proposer share
            "steered_mean_proposer_pct": round(sum(steered_pcts) / len(steered_pcts), 2),
            "baseline_mean_proposer_pct": round(sum(baseline_pcts) / len(baseline_pcts), 2),
            "delta_proposer_pct": round(sum(deltas) / len(deltas), 2),
            "std_delta": round(std_delta if len(deltas) >= 2 else 0, 2),
            # Statistical test
            "paired_ttest_t": round(t_stat, 4),
            "paired_ttest_p": round(p_val, 6),
            "cohens_d": round(cohens_d, 4),
            # Accept rates
            "steered_accept_rate": round(len(steered_accepted) / len(usable), 3),
            "baseline_accept_rate": round(len(baseline_accepted) / len(usable), 3),
            # Payoffs
            "steered_mean_payoff_pct": round(sum(steered_payoffs) / len(usable), 2),
            "baseline_mean_payoff_pct": round(sum(baseline_payoffs) / len(usable), 2),
            # Behavioral
            "steered_mean_word_count": round(sum(steered_wc) / len(usable), 1),
            "baseline_mean_word_count": round(sum(baseline_wc) / len(usable), 1),
            "steered_mean_hedge_count": round(sum(steered_hedge) / len(usable), 2),
            "baseline_mean_hedge_count": round(sum(baseline_hedge) / len(usable), 2),
            "steered_mean_fairness_count": round(sum(steered_fair) / len(usable), 2),
            "baseline_mean_fairness_count": round(sum(baseline_fair) / len(usable), 2),
        }

    elif steered_role == "responder":
        # Both must have valid responder decisions on the SAME offer
        usable = [g for g in valid
                  if g["steered"]["parse_error"] is None
                  and g["baseline"]["parse_error"] is None
                  and g["steered"]["response"] is not None
                  and g["baseline"]["response"] is not None]

        if not usable:
            return {"n_games": n_total, "n_valid": 0, "n_parse_errors": n_parse_errors}

        # McNemar's test: compare accept/reject decisions on same offers
        # a = both accept, b = steered accepts & baseline rejects,
        # c = steered rejects & baseline accepts, d = both reject
        a = sum(1 for g in usable if g["steered"]["agreed"] and g["baseline"]["agreed"])
        b = sum(1 for g in usable if g["steered"]["agreed"] and not g["baseline"]["agreed"])
        c = sum(1 for g in usable if not g["steered"]["agreed"] and g["baseline"]["agreed"])
        d = sum(1 for g in usable if not g["steered"]["agreed"] and not g["baseline"]["agreed"])

        # McNemar's chi-squared (with continuity correction)
        if b + c > 0:
            mcnemar_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)
        else:
            mcnemar_chi2, mcnemar_p = 0.0, 1.0

        steered_accept = sum(1 for g in usable if g["steered"]["agreed"])
        baseline_accept = sum(1 for g in usable if g["baseline"]["agreed"])

        # Offer distribution (same for both since it's the same proposer)
        offer_pcts = [g["steered"]["proposer_share"] / g["pool"] * 100 for g in usable
                      if g["steered"]["proposer_share"] is not None]

        # Payoffs
        steered_r_payoffs = [g["steered"]["responder_payoff"] / g["pool"] * 100 for g in usable]
        baseline_r_payoffs = [g["baseline"]["responder_payoff"] / g["pool"] * 100 for g in usable]

        return {
            "n_games": n_total,
            "n_valid": n_valid,
            "n_usable_pairs": len(usable),
            "n_parse_errors": n_parse_errors,
            "steered_role": steered_role,
            # Accept rates
            "steered_accept_rate": round(steered_accept / len(usable), 3),
            "baseline_accept_rate": round(baseline_accept / len(usable), 3),
            "delta_accept_rate": round((steered_accept - baseline_accept) / len(usable), 3),
            # McNemar's
            "mcnemar_table": {"both_accept": a, "steered_only": b, "baseline_only": c, "both_reject": d},
            "mcnemar_chi2": round(mcnemar_chi2, 4),
            "mcnemar_p": round(mcnemar_p, 6),
            # Offer distribution
            "mean_offer_pct": round(sum(offer_pcts) / len(offer_pcts), 2) if offer_pcts else None,
            # Payoffs
            "steered_mean_responder_payoff_pct": round(sum(steered_r_payoffs) / len(usable), 2),
            "baseline_mean_responder_payoff_pct": round(sum(baseline_r_payoffs) / len(usable), 2),
        }

    return {}


def summarise_baseline(games: List[Dict]) -> Dict:
    """Summarise baseline (no steering) games."""
    valid = [g for g in games if g["result"]["parse_error"] is None]
    n_total = len(games)
    n_valid = len(valid)

    if not valid:
        return {"n_games": n_total, "n_valid": 0}

    pcts = [g["result"]["proposer_share"] / g["pool"] * 100 for g in valid]
    accepted = [g for g in valid if g["result"]["agreed"]]
    mean_pct = sum(pcts) / len(pcts)
    std_pct = (sum((x - mean_pct) ** 2 for x in pcts) / max(len(pcts) - 1, 1)) ** 0.5

    payoffs_p = [g["result"]["proposer_payoff"] / g["pool"] * 100 for g in valid]
    payoffs_r = [g["result"]["responder_payoff"] / g["pool"] * 100 for g in valid]

    return {
        "n_games": n_total,
        "n_valid": n_valid,
        "n_parse_errors": n_total - n_valid,
        "mean_proposer_pct": round(mean_pct, 2),
        "std_proposer_pct": round(std_pct, 2),
        "accept_rate": round(len(accepted) / n_valid, 3),
        "mean_proposer_payoff_pct": round(sum(payoffs_p) / n_valid, 2),
        "mean_responder_payoff_pct": round(sum(payoffs_r) / n_valid, 2),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_paired_summary(summary: Dict, config: Dict) -> None:
    s = summary
    game_label = "DICTATOR GAME" if config.get("game") == "dictator" else "ULTIMATUM GAME"
    print(f"\n{'=' * 70}")
    print(f"{game_label} — ACTIVATION STEERING RESULTS")
    print(f"  Model:      {config['model']}")
    print(f"  Dimension:  {config.get('dimension', 'NONE')}")
    print(f"  Layers:     {config.get('layers', [])}")
    print(f"  Alpha:      {config.get('alpha', 0)}")
    print(f"  Steered:    {config.get('steered_role', 'none')}")
    print(f"  Games:      {s.get('n_games', 0)}  Valid pairs: {s.get('n_usable_pairs', 0)}  "
          f"Parse errors: {s.get('n_parse_errors', 0)}")
    print(f"{'=' * 70}")

    role = config.get("steered_role")
    if role == "proposer":
        print(f"\n  PROPOSER SHARE (% of pool):")
        print(f"    Steered:   {s.get('steered_mean_proposer_pct', '?')}%")
        print(f"    Baseline:  {s.get('baseline_mean_proposer_pct', '?')}%")
        print(f"    Delta:     {s.get('delta_proposer_pct', '?'):+.2f}%")
        print(f"    Paired t:  t={s.get('paired_ttest_t', '?')}, p={s.get('paired_ttest_p', '?')}")
        print(f"    Cohen's d: {s.get('cohens_d', '?')}")
        print(f"\n  ACCEPT RATES:")
        print(f"    Steered:   {s.get('steered_accept_rate', '?'):.1%}")
        print(f"    Baseline:  {s.get('baseline_accept_rate', '?'):.1%}")
        print(f"\n  PAYOFFS (proposer, % of pool):")
        print(f"    Steered:   {s.get('steered_mean_payoff_pct', '?')}%")
        print(f"    Baseline:  {s.get('baseline_mean_payoff_pct', '?')}%")
        print(f"\n  BEHAVIORAL:")
        print(f"    Word count:  steered={s.get('steered_mean_word_count', '?')} "
              f"baseline={s.get('baseline_mean_word_count', '?')}")
        print(f"    Hedging:     steered={s.get('steered_mean_hedge_count', '?')} "
              f"baseline={s.get('baseline_mean_hedge_count', '?')}")
        print(f"    Fairness:    steered={s.get('steered_mean_fairness_count', '?')} "
              f"baseline={s.get('baseline_mean_fairness_count', '?')}")

    elif role == "responder":
        print(f"\n  RESPONDER DECISIONS (same offers):")
        print(f"    Steered accept:   {s.get('steered_accept_rate', '?'):.1%}")
        print(f"    Baseline accept:  {s.get('baseline_accept_rate', '?'):.1%}")
        print(f"    Delta:            {s.get('delta_accept_rate', '?'):+.1%}")
        mt = s.get("mcnemar_table", {})
        print(f"    McNemar: chi2={s.get('mcnemar_chi2', '?')}, p={s.get('mcnemar_p', '?')}")
        print(f"      Both accept={mt.get('both_accept', '?')} "
              f"Steered-only={mt.get('steered_only', '?')} "
              f"Baseline-only={mt.get('baseline_only', '?')} "
              f"Both reject={mt.get('both_reject', '?')}")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ultimatum Game with activation steering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--game", choices=["ultimatum", "dictator"], default="ultimatum",
                   help="Game type. Dictator game skips responder (auto-accept).")
    p.add_argument("--model", choices=list(MODELS.keys()), default="qwen2.5-7b")
    p.add_argument("--vectors_dir", default="vectors/neg8dim_12pairs_matched/negotiation",
                   help="Root dir for steering vectors.")
    p.add_argument("--dimension", default=None,
                   help="Steering dimension (e.g. firmness, anchoring, empathy, batna_awareness). "
                        "None = baseline only.")
    p.add_argument("--method", choices=["mean_diff", "pca", "logreg"], default="mean_diff")
    p.add_argument("--layers", nargs="+", type=int, default=[14],
                   help="Transformer layers to steer (0-indexed).")
    p.add_argument("--alpha", type=float, default=6.0,
                   help="Steering strength.")
    p.add_argument("--steered_role", choices=["proposer", "responder"], default="proposer",
                   help="Which role to steer.")
    p.add_argument("--n_games", type=int, default=50)
    p.add_argument("--variable_pools", action="store_true",
                   help="Use variable pool sizes ($37-$157).")
    p.add_argument("--pool", type=int, default=DEFAULT_POOL,
                   help="Fixed pool size (ignored if --variable_pools).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paired", action="store_true",
                   help="Enable paired design (steered + baseline on same pool/offer).")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--quantize", action="store_true",
                   help="Load model in 4-bit quantization (for GPUs with <24GB VRAM).")
    p.add_argument("--proposer_enhancement", default=None,
                   help="Optional prompt enhancement for proposer (in addition to/instead of steering).")
    p.add_argument("--responder_enhancement", default=None,
                   help="Optional prompt enhancement for responder.")
    p.add_argument("--output_dir", default="results/ultimatum/",
                   help="Save results JSON to this directory.")
    p.add_argument("--text_visible", action="store_true",
                   help="Responder sees proposer's full reasoning text (not just parsed OFFER numbers).")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of games to process in a single batched forward pass. "
                        "Only applies to paired proposer experiments. "
                        "Values of 4-8 typically give best throughput on a single GPU.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_baseline = args.dimension is None
    is_paired = args.paired and not is_baseline

    if args.game == "dictator" and args.steered_role == "responder" and not is_baseline:
        raise SystemExit("ERROR: Dictator game has no responder decision — cannot steer responder role.")

    # --- Load direction vectors (skip if baseline) ---
    dvecs = None
    if not is_baseline:
        log.info("Loading direction vectors: %s / %s / layers=%s",
                 args.dimension, args.method, args.layers)
        dvecs = load_direction_vectors(
            vectors_dir=Path(args.vectors_dir),
            model_alias=cfg.alias,
            dimension=args.dimension,
            method=args.method,
            layer_indices=args.layers,
        )
        log.info("Loaded vectors for %d layers.", len(dvecs))

    # --- Load model ---
    log.info("Loading model: %s", cfg.hf_id)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(token=token, device_map="auto")
    if args.quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif getattr(cfg, "is_gptq", False) or getattr(cfg, "is_awq", False):
        # Pre-quantized models: transformers auto-detects GPTQ/AWQ from model config
        pass
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]

    # Flash Attention 2: faster attention kernel (requires flash-attn package)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id, **load_kwargs, attn_implementation="flash_attention_2"
        )
        log.info("Flash Attention 2 enabled.")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **load_kwargs)
        log.info("Flash Attention 2 not available, using default attention.")
    model.eval()
    log.info("Model loaded. Device: %s", next(model.parameters()).device)

    # torch.compile: fuses ops for ~10-30% speedup (skipped for quantized models)
    if not args.quantize:
        try:
            model = torch.compile(model)
            log.info("torch.compile() applied.")
        except Exception as e:
            log.warning("torch.compile() failed (%s), running in eager mode.", e)

    # --- Prepare pool sizes ---
    if args.variable_pools:
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(args.n_games)]
        random.shuffle(pools)
    else:
        pools = [args.pool] * args.n_games

    # --- Config ---
    config = {
        "game": args.game,
        "model": args.model,
        "dimension": args.dimension,
        "method": args.method,
        "layers": args.layers,
        "alpha": args.alpha,
        "steered_role": args.steered_role if not is_baseline else None,
        "paired": is_paired,
        "n_games": args.n_games,
        "variable_pools": args.variable_pools,
        "pool": args.pool,
        "temperature": args.temperature,
        "seed": args.seed,
        "dtype": args.dtype,
        "vectors_dir": args.vectors_dir,
        "proposer_enhancement": args.proposer_enhancement,
        "responder_enhancement": args.responder_enhancement,
        "text_visible": args.text_visible,
        "timestamp": datetime.now().isoformat(),
    }

    game_label = "DICTATOR GAME" if args.game == "dictator" else "ULTIMATUM GAME"
    print(f"\n{'=' * 70}")
    if is_baseline:
        print(f"{game_label} — BASELINE (no steering)")
    else:
        print(f"{game_label} — ACTIVATION STEERING")
        print(f"  Dimension: {args.dimension}  Alpha: {args.alpha}  Layers: {args.layers}")
        print(f"  Steered role: {args.steered_role}  Paired: {is_paired}")
    print(f"  Model: {args.model}  Games: {args.n_games}  "
          f"Pools: {'variable' if args.variable_pools else f'${args.pool}'}")
    print(f"{'=' * 70}\n")

    # --- Run games ---
    games = []
    t0 = time.time()

    use_batch = args.batch_size > 1 and is_paired and args.steered_role == "proposer"
    if use_batch:
        log.info("Batch mode: batch_size=%d (paired proposer only)", args.batch_size)
        for batch_start in range(0, args.n_games, args.batch_size):
            batch_pools = pools[batch_start:batch_start + args.batch_size]
            batch_results = run_paired_proposer_batch(
                model, tokenizer, batch_pools,
                direction_vectors=dvecs,
                alpha=args.alpha,
                proposer_enhancement=args.proposer_enhancement,
                responder_enhancement=args.responder_enhancement,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                game=args.game,
                text_visible=args.text_visible,
            )
            for j, result in enumerate(batch_results):
                i = batch_start + j
                result["game_id"] = i
                games.append(result)
                s, b, pool = result["steered"], result["baseline"], result["pool"]
                if result.get("parse_error"):
                    print(f"  Game {i+1:3d}: PARSE_ERROR  pool=${pool}")
                else:
                    s_pct = s["proposer_share"] / pool * 100 if s["proposer_share"] is not None else 0
                    b_pct = b["proposer_share"] / pool * 100 if b["proposer_share"] is not None else 0
                    s_dec = "ACC" if s["agreed"] else "REJ" if s["response"] else "ERR"
                    b_dec = "ACC" if b["agreed"] else "REJ" if b["response"] else "ERR"
                    print(f"  Game {i+1:3d}: steered={s_pct:.0f}%({s_dec}) "
                          f"baseline={b_pct:.0f}%({b_dec})  pool=${pool}")

    for i in range(args.n_games if not use_batch else 0):
        pool = pools[i]

        if is_baseline:
            result = run_baseline_game(
                model, tokenizer, pool,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                game=args.game,
                text_visible=args.text_visible,
            )
            result["game_id"] = i
            games.append(result)

            r = result["result"]
            if r["parse_error"]:
                print(f"  Game {i+1:3d}: PARSE_ERROR ({r['parse_error']})  pool=${pool}")
            elif r["agreed"]:
                pct = r["proposer_share"] / pool * 100
                print(f"  Game {i+1:3d}: ACCEPT  {r['proposer_share']}/{r['responder_share']} "
                      f"({pct:.0f}%)  pool=${pool}")
            else:
                print(f"  Game {i+1:3d}: REJECT  {r['proposer_share']}/{r['responder_share']}  pool=${pool}")

        elif is_paired:
            result = run_paired_game(
                model, tokenizer, pool,
                steered_role=args.steered_role,
                direction_vectors=dvecs,
                alpha=args.alpha,
                proposer_enhancement=args.proposer_enhancement,
                responder_enhancement=args.responder_enhancement,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                game=args.game,
                text_visible=args.text_visible,
            )
            result["game_id"] = i
            games.append(result)

            s = result["steered"]
            b = result["baseline"]
            if result.get("parse_error"):
                print(f"  Game {i+1:3d}: PARSE_ERROR  pool=${pool}")
            elif args.steered_role == "proposer":
                s_pct = s["proposer_share"] / pool * 100 if s["proposer_share"] is not None else 0
                b_pct = b["proposer_share"] / pool * 100 if b["proposer_share"] is not None else 0
                s_dec = "ACC" if s["agreed"] else "REJ" if s["response"] else "ERR"
                b_dec = "ACC" if b["agreed"] else "REJ" if b["response"] else "ERR"
                print(f"  Game {i+1:3d}: steered={s_pct:.0f}%({s_dec}) "
                      f"baseline={b_pct:.0f}%({b_dec})  pool=${pool}")
            else:
                s_dec = "ACC" if s["agreed"] else "REJ"
                b_dec = "ACC" if b["agreed"] else "REJ"
                offer_pct = s["proposer_share"] / pool * 100 if s["proposer_share"] else 0
                print(f"  Game {i+1:3d}: offer={offer_pct:.0f}%  "
                      f"steered={s_dec}  baseline={b_dec}  pool=${pool}")

        else:
            # Non-paired steered game (single condition)
            result = _run_single_game(
                model, tokenizer, pool,
                steered_role=args.steered_role,
                direction_vectors=dvecs,
                alpha=args.alpha,
                proposer_enhancement=args.proposer_enhancement,
                responder_enhancement=args.responder_enhancement,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                game=args.game,
                text_visible=args.text_visible,
            )
            games.append({"game_id": i, "pool": pool, "result": result})
            r = result
            if r["parse_error"]:
                print(f"  Game {i+1:3d}: PARSE_ERROR ({r['parse_error']})  pool=${pool}")
            elif r["agreed"]:
                pct = r["proposer_share"] / pool * 100
                print(f"  Game {i+1:3d}: ACCEPT  {r['proposer_share']}/{r['responder_share']} "
                      f"({pct:.0f}%)  pool=${pool}")
            else:
                print(f"  Game {i+1:3d}: REJECT  {r['proposer_share']}/{r['responder_share']}  pool=${pool}")

    elapsed = time.time() - t0

    # --- Summary ---
    if is_paired:
        summary = summarise_paired(games, args.steered_role)
        print_paired_summary(summary, config)
    elif is_baseline:
        summary = summarise_baseline(games)
        print(f"\n{'=' * 70}")
        print(f"BASELINE SUMMARY")
        print(f"  Valid: {summary.get('n_valid', 0)}/{summary.get('n_games', 0)}")
        print(f"  Mean proposer %: {summary.get('mean_proposer_pct', '?')}% "
              f"(std={summary.get('std_proposer_pct', '?')}%)")
        print(f"  Accept rate: {summary.get('accept_rate', '?'):.1%}")
        print(f"  Proposer payoff: {summary.get('mean_proposer_payoff_pct', '?')}%")
        print(f"  Responder payoff: {summary.get('mean_responder_payoff_pct', '?')}%")
        print(f"{'=' * 70}")
    else:
        summary = {"note": "non-paired single condition, see per-game results"}

    summary["elapsed_seconds"] = round(elapsed, 1)

    # --- Save ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game_prefix = "dg_" if args.game == "dictator" else ""
    tv_prefix = "tv_" if args.text_visible else ""
    if is_baseline:
        filename = f"{tv_prefix}{game_prefix}baseline_{args.model}_n{args.n_games}.json"
    else:
        parts = [args.dimension, args.steered_role]
        parts.append(f"L{'_'.join(str(l) for l in args.layers)}")
        parts.append(f"a{args.alpha}")
        if is_paired:
            parts.append("paired")
        if args.text_visible:
            parts.append("tv")
        parts.append(f"n{args.n_games}")
        filename = game_prefix + "_".join(parts) + ".json"

    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump({
            "config": config,
            "summary": summary,
            "games": games,
        }, f, indent=2)

    log.info("Results saved to %s (%.1f sec, %.1f sec/game)",
             out_path, elapsed, elapsed / max(args.n_games, 1))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
