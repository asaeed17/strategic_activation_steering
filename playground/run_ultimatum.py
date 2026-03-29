#!/usr/bin/env python3
"""
playground/run_ultimatum.py — Ultimatum Game task design validation.

Single-turn game: Proposer splits $100, Responder accepts or rejects.
Zero state tracking, no multi-turn drift, no inventory hallucination.
Hundreds of games cheaply → high statistical power.

Experiments:
  A. Prompt steering: does "be aggressive" shift the proposed split?
  B. Responder threshold: does "be firm about fairness" change rejection rates?
  C. Cross-model: do frontier models behave differently from small models?

Usage:
  # Baseline: Gemini vs Gemini
  python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini \
      --n_games 50

  # Aggressive proposer vs baseline responder
  python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini \
      --proposer_enhancement aggressive --n_games 50

  # Fair-minded responder
  python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini \
      --responder_enhancement firmness --n_games 50

  # Cross-model comparison
  python playground/run_ultimatum.py --proposer api:groq --responder api:groq \
      --n_games 50

Agent specs:
  api:gemini            Google Gemini 2.5 Flash (GOOGLE_API_KEY)
  api:gpt4o             OpenAI GPT-4o (OPENAI_API_KEY)
  api:claude            Anthropic Claude Sonnet (ANTHROPIC_API_KEY)
  api:groq              Groq LLaMA 3.3 70B (GROQ_API_KEY)
  api:groq-llama8b      Groq LLaMA 3.1 8B (GROQ_API_KEY)

Prompt enhancements (--proposer_enhancement / --responder_enhancement):
  aggressive    Demand most of the pool for yourself
  fair          Propose an equal or near-equal split
  rational      Accept any offer > $0 (game-theoretic optimal)
  firmness      Reject unfair offers, demand at least 40%
  empathy       Consider the other player's perspective
"""

import json
import re
import sys
import random
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "openai", "google", "google.genai",
               "google.generativeai", "urllib3", "google.auth", "google.api_core"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

DEFAULT_POOL = 100

# 100 structurally diverse pools ($37-$157): primes, evens, odds, mult-of-5/10.
# Avoids primes-only bias (blocks 50/50 splits, unusual tokenization).
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
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def build_proposer_system(pool: int = DEFAULT_POOL) -> str:
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


def build_responder_system(proposer_share: int, responder_share: int, pool: int = DEFAULT_POOL) -> str:
    return (
        f"You are Player B in an Ultimatum Game.\n"
        f"\n"
        f"Player A has proposed: they get ${proposer_share}, you get ${responder_share}.\n"
        f"The total pool is ${pool}.\n"
        f"\n"
        f"Rules:\n"
        f"  - If you Accept, you get ${responder_share} and they get ${proposer_share}.\n"
        f"  - If you Reject, you BOTH get $0.\n"
        f"\n"
        f"Respond with 1-2 sentences explaining your reasoning, then end with:\n"
        f"ACCEPT or REJECT\n"
    )


# ---------------------------------------------------------------------------
# Prompt enhancements
# ---------------------------------------------------------------------------

# Enhancements mapped to the 8 steering vector dimensions (neg8dim):
#   firmness, empathy, scm, anchoring, batna, active_listening, composure, creative_problem_solving
# Plus controls: aggressive, fair, rational

PROPOSER_ENHANCEMENTS = {
    # -- Controls --
    "aggressive": (
        "\nStrategy — AGGRESSIVE:\n"
        "  - Maximize your own payoff. Demand as much as possible.\n"
        "  - Offer the minimum you think Player B will accept.\n"
        "  - You have all the power — Player B gets $0 if they reject.\n"
    ),
    "fair": (
        "\nStrategy — FAIR SPLIT:\n"
        "  - Propose a fair, equal or near-equal split.\n"
        "  - Fairness builds trust and ensures acceptance.\n"
        "  - A guaranteed fair amount is better than risking rejection.\n"
    ),
    # -- Steering dimension analogues --
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
    "scm": (
        "\nStrategy — STRATEGIC CONCESSION:\n"
        "  - Start with a strong position but be willing to give a little.\n"
        "  - Make a calculated concession that signals goodwill without giving too much.\n"
        "  - Your concession should make Player B feel they got a good deal.\n"
    ),
    "anchoring": (
        "\nStrategy — ANCHORING:\n"
        "  - Set the terms aggressively. Your initial offer frames the entire negotiation.\n"
        "  - Anchor high — demand a large share so even a 'compromise' favors you.\n"
        "  - Player B's expectations will adjust to your anchor.\n"
    ),
    "batna": (
        "\nStrategy — BATNA AWARENESS:\n"
        "  - Remember: if Player B rejects, you both get $0. Use this leverage.\n"
        "  - Frame your offer as the best option available to Player B.\n"
        "  - Make it clear that rejection hurts them more than it hurts you.\n"
    ),
    "active_listening": (
        "\nStrategy — ACTIVE LISTENING:\n"
        "  - Acknowledge that Player B has their own interests and concerns.\n"
        "  - Show you've considered what would matter to them.\n"
        "  - Make your offer responsive to their likely perspective.\n"
    ),
    "composure": (
        "\nStrategy — COMPOSURE:\n"
        "  - Stay calm, measured, and rational in your reasoning.\n"
        "  - Don't be emotional or impulsive — think through the payoffs carefully.\n"
        "  - Present your offer with quiet confidence.\n"
    ),
    "creative_problem_solving": (
        "\nStrategy — CREATIVE PROBLEM-SOLVING:\n"
        "  - Think beyond simple greed vs fairness.\n"
        "  - Find a split that maximizes the chance of acceptance while still favoring you.\n"
        "  - Consider what split makes Player B feel they won something.\n"
    ),
}

RESPONDER_ENHANCEMENTS = {
    # -- Controls --
    "rational": (
        "\nStrategy — RATIONAL:\n"
        "  - Any amount greater than $0 is better than rejecting and getting $0.\n"
        "  - Accept any positive offer — the rational choice maximizes your payoff.\n"
        "  - Don't let emotions override logic.\n"
    ),
    # -- Steering dimension analogues --
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
    "scm": (
        "\nStrategy — STRATEGIC CONCESSION:\n"
        "  - Be willing to accept a slightly unfair deal if it means guaranteed money.\n"
        "  - A strategic concession now (accepting less) beats $0.\n"
        "  - Show flexibility to secure a positive outcome.\n"
    ),
    "anchoring": (
        "\nStrategy — ANCHORING:\n"
        "  - Judge this offer against a fair 50/50 baseline.\n"
        "  - Anything below 40% for you is anchored too far in their favor.\n"
        "  - Don't let their aggressive framing shift your standards.\n"
    ),
    "batna": (
        "\nStrategy — BATNA AWARENESS:\n"
        "  - Your alternative is $0. Any positive amount beats your BATNA.\n"
        "  - But also: Player A's alternative is $0 too. You have equal leverage.\n"
        "  - Reject only if the offer is truly insulting — your walkaway is $0.\n"
    ),
    "composure": (
        "\nStrategy — COMPOSURE:\n"
        "  - Stay calm and evaluate the offer purely on the numbers.\n"
        "  - Don't let emotions like spite or indignation drive your decision.\n"
        "  - Make the choice that maximizes your actual payoff.\n"
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
    if a + b != pool:
        return None
    if a < 0 or b < 0:
        return None
    return (a, b)


def parse_response(text: str) -> Optional[str]:
    has_accept = _ACCEPT_RE.search(text)
    has_reject = _REJECT_RE.search(text)
    if has_reject and not has_accept:
        return "reject"
    if has_accept and not has_reject:
        return "accept"
    if has_accept and has_reject:
        # Take whichever appears last (the final decision)
        last_accept = max(m.end() for m in _ACCEPT_RE.finditer(text))
        last_reject = max(m.end() for m in _REJECT_RE.finditer(text))
        return "accept" if last_accept > last_reject else "reject"
    return None


# ---------------------------------------------------------------------------
# API dispatch (reused from playground/run_game.py)
# ---------------------------------------------------------------------------

def generate_api(
    provider: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> str:
    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens + 8000,
                thinking_config=types.ThinkingConfig(thinking_budget=8000),
            ),
        )
        return (response.text or "").strip()

    elif provider == "gpt4o":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    elif provider == "claude":
        import anthropic
        client = anthropic.Anthropic()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        non_system = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system,
            messages=non_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.content[0].text or "").strip()

    elif provider in ("groq", "groq-llama8b"):
        from groq import Groq
        model_id = {
            "groq": "llama-3.3-70b-versatile",
            "groq-llama8b": "llama-3.1-8b-instant",
        }[provider]
        client = Groq()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    else:
        raise ValueError(f"Unknown API provider: {provider}")


def parse_agent_spec(spec: str) -> Tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or parts[0] not in ("api",):
        raise ValueError(f"Invalid agent spec '{spec}'. Use api:<provider> (e.g. api:gemini)")
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def run_game_ultimatum(
    proposer_spec: Tuple[str, str],
    responder_spec: Tuple[str, str],
    proposer_enhancement: Optional[str] = None,
    responder_enhancement: Optional[str] = None,
    pool: int = DEFAULT_POOL,
    temperature: float = 0.7,
    max_tokens: int = 200,
    verbose: bool = False,
    game: str = "ultimatum",
) -> Dict:
    # Build proposer prompt
    proposer_sys = build_proposer_system(pool)
    if proposer_enhancement:
        enh = PROPOSER_ENHANCEMENTS.get(proposer_enhancement)
        if not enh:
            raise ValueError(f"Unknown proposer enhancement '{proposer_enhancement}'. "
                             f"Choose from: {list(PROPOSER_ENHANCEMENTS.keys())}")
        proposer_sys += enh

    proposer_msgs = [
        {"role": "system", "content": proposer_sys},
        {"role": "user", "content": "Make your offer."},
    ]

    # Generate proposer offer (with one retry on parse failure)
    proposer_text = generate_api(proposer_spec[1], proposer_msgs, temperature, max_tokens)
    offer = parse_offer(proposer_text, pool)

    if offer is None:
        # Retry once
        proposer_text = generate_api(proposer_spec[1], proposer_msgs, temperature, max_tokens)
        offer = parse_offer(proposer_text, pool)

    if verbose:
        print(f"  PROPOSER: {proposer_text}")

    if offer is None:
        return {
            "agreed": False,
            "proposer_share": None,
            "responder_share": None,
            "response": None,
            "proposer_payoff": 0,
            "responder_payoff": 0,
            "parse_error": "proposer",
            "proposer_text": proposer_text,
            "responder_text": None,
            "pool": pool,
        }

    proposer_share, responder_share = offer

    # Dictator Game: auto-accept, no responder generation
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
        }

    # Build responder prompt
    responder_sys = build_responder_system(proposer_share, responder_share, pool)
    if responder_enhancement:
        enh = RESPONDER_ENHANCEMENTS.get(responder_enhancement)
        if not enh:
            raise ValueError(f"Unknown responder enhancement '{responder_enhancement}'. "
                             f"Choose from: {list(RESPONDER_ENHANCEMENTS.keys())}")
        responder_sys += enh

    responder_msgs = [
        {"role": "system", "content": responder_sys},
        {"role": "user", "content": f"Player A offers: they get ${proposer_share}, you get ${responder_share}. What is your decision?"},
    ]

    # Generate responder decision (with one retry on parse failure)
    responder_text = generate_api(responder_spec[1], responder_msgs, temperature, max_tokens)
    decision = parse_response(responder_text)

    if decision is None:
        responder_text = generate_api(responder_spec[1], responder_msgs, temperature, max_tokens)
        decision = parse_response(responder_text)

    if verbose:
        print(f"  RESPONDER: {responder_text}")

    if decision is None:
        return {
            "agreed": False,
            "proposer_share": proposer_share,
            "responder_share": responder_share,
            "response": None,
            "proposer_payoff": 0,
            "responder_payoff": 0,
            "parse_error": "responder",
            "proposer_text": proposer_text,
            "responder_text": responder_text,
            "pool": pool,
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
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(results: List[Dict]) -> Dict:
    valid = [r for r in results if r["parse_error"] is None]
    accepted = [r for r in valid if r["agreed"]]
    rejected = [r for r in valid if not r["agreed"]]

    n = len(results)
    n_valid = len(valid)
    n_parse_errors = n - n_valid
    n_accepted = len(accepted)

    if not valid:
        return {
            "n_games": n, "n_valid": 0, "n_parse_errors": n_parse_errors,
            "accept_rate": 0, "mean_proposer_pct": None,
        }

    # Normalize to percentage of pool (comparable across variable pool sizes)
    proposer_pcts = [r["proposer_share"] / r["pool"] * 100 for r in valid]
    mean_pct = sum(proposer_pcts) / len(proposer_pcts)
    sorted_pcts = sorted(proposer_pcts)
    median_pct = sorted_pcts[len(sorted_pcts) // 2]
    std_pct = (sum((x - mean_pct) ** 2 for x in proposer_pcts) / max(len(proposer_pcts) - 1, 1)) ** 0.5

    accept_rate = n_accepted / n_valid if n_valid else 0

    # Unfair / fair by percentage (responder gets < 30% or >= 40%)
    unfair = [r for r in valid if r["responder_share"] / r["pool"] < 0.30]
    fair = [r for r in valid if r["responder_share"] / r["pool"] >= 0.40]
    unfair_accepted = [r for r in unfair if r["agreed"]]
    fair_accepted = [r for r in fair if r["agreed"]]

    # Payoffs as percentage of pool
    proposer_payoff_pcts = [r["proposer_payoff"] / r["pool"] * 100 for r in valid]
    responder_payoff_pcts = [r["responder_payoff"] / r["pool"] * 100 for r in valid]

    return {
        "n_games": n,
        "n_valid": n_valid,
        "n_parse_errors": n_parse_errors,
        "n_accepted": n_accepted,
        "n_rejected": len(rejected),
        "accept_rate": round(accept_rate, 3),
        "mean_proposer_pct": round(mean_pct, 1),
        "median_proposer_pct": round(median_pct, 1),
        "std_proposer_pct": round(std_pct, 1),
        "min_proposer_pct": round(min(proposer_pcts), 1),
        "max_proposer_pct": round(max(proposer_pcts), 1),
        "n_unfair_offers": len(unfair),
        "unfair_accept_rate": round(len(unfair_accepted) / len(unfair), 3) if unfair else None,
        "n_fair_offers": len(fair),
        "fair_accept_rate": round(len(fair_accepted) / len(fair), 3) if fair else None,
        "mean_proposer_payoff_pct": round(sum(proposer_payoff_pcts) / n_valid, 1),
        "mean_responder_payoff_pct": round(sum(responder_payoff_pcts) / n_valid, 1),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_summary(summary: Dict, config: Dict) -> None:
    s = summary
    game_label = config.get("game", "ultimatum").upper() + " GAME"
    print(f"\n{'=' * 70}")
    print(f"{game_label} RESULTS")
    print(f"  Proposer:   {config['proposer']}" +
          (f"  + {config.get('proposer_enhancement', '')}" if config.get('proposer_enhancement') else ""))
    print(f"  Responder:  {config['responder']}" +
          (f"  + {config.get('responder_enhancement', '')}" if config.get('responder_enhancement') else ""))
    pool_desc = "variable" if config.get("variable_pools") else f"${config.get('pool', DEFAULT_POOL)}"
    print(f"  Pool: {pool_desc}   Games: {s['n_games']}   "
          f"Valid: {s['n_valid']}   Parse errors: {s['n_parse_errors']}")
    print(f"{'=' * 70}")

    if s["n_valid"] == 0:
        print("  No valid games.")
        return

    print(f"\n  PROPOSER OFFERS (% of pool):")
    print(f"    Mean:    {s['mean_proposer_pct']:.1f}% / {100 - s['mean_proposer_pct']:.1f}%")
    print(f"    Median:  {s['median_proposer_pct']:.1f}% / {100 - s['median_proposer_pct']:.1f}%")
    print(f"    Std dev: {s['std_proposer_pct']:.1f}%")
    print(f"    Range:   {s['min_proposer_pct']:.1f}%-{s['max_proposer_pct']:.1f}%")

    print(f"\n  RESPONDER DECISIONS:")
    print(f"    Accept rate:  {s['accept_rate']:.1%}  ({s['n_accepted']}/{s['n_valid']})")
    if s["n_unfair_offers"]:
        print(f"    Unfair (<30% to responder): {s['n_unfair_offers']} offers, "
              f"accept rate: {s['unfair_accept_rate']:.1%}")
    if s["n_fair_offers"]:
        print(f"    Fair (>=40% to responder):   {s['n_fair_offers']} offers, "
              f"accept rate: {s['fair_accept_rate']:.1%}")

    print(f"\n  PAYOFFS (% of pool, including 0% for rejections):")
    print(f"    Proposer:     {s['mean_proposer_payoff_pct']:.1f}% avg")
    print(f"    Responder:    {s['mean_responder_payoff_pct']:.1f}% avg")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    all_enhancements = sorted(set(list(PROPOSER_ENHANCEMENTS.keys()) + list(RESPONDER_ENHANCEMENTS.keys())))

    p = argparse.ArgumentParser(
        description="Ultimatum Game task design validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--game", default="ultimatum", choices=["ultimatum", "dictator"],
                   help="Game type. dictator: proposer splits, always accepted (no responder LLM call).")
    p.add_argument("--proposer", default="api:gemini",
                   help="Proposer agent spec. e.g. api:gemini, api:groq")
    p.add_argument("--responder", default="api:gemini",
                   help="Responder agent spec. e.g. api:gpt4o, api:groq")
    p.add_argument("--proposer_enhancement", default=None,
                   choices=list(PROPOSER_ENHANCEMENTS.keys()),
                   help="Prompt enhancement for proposer.")
    p.add_argument("--responder_enhancement", default=None,
                   choices=list(RESPONDER_ENHANCEMENTS.keys()),
                   help="Prompt enhancement for responder.")
    p.add_argument("--n_games", type=int, default=50)
    p.add_argument("--pool", type=int, default=DEFAULT_POOL,
                   help="Fixed pool size (default: 100). Ignored if --variable_pools.")
    p.add_argument("--variable_pools", action="store_true",
                   help="Use variable pool sizes (37-157) per game to break memorized-split convergence.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full text for each game.")
    p.add_argument("--output_dir", default=None,
                   help="Save results JSON to this directory.")
    args = p.parse_args()

    random.seed(args.seed)
    proposer_spec = parse_agent_spec(args.proposer)
    responder_spec = parse_agent_spec(args.responder)

    config = {
        "game": args.game,
        "proposer": args.proposer,
        "responder": args.responder,
        "proposer_enhancement": args.proposer_enhancement,
        "responder_enhancement": args.responder_enhancement,
        "n_games": args.n_games,
        "pool": args.pool,
        "variable_pools": args.variable_pools,
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'=' * 70}")
    game_label = args.game.upper() + " GAME"
    print(f"{game_label} — TASK DESIGN VALIDATION")
    print(f"  Proposer:   {args.proposer}" +
          (f"  + {args.proposer_enhancement}" if args.proposer_enhancement else ""))
    if args.game == "ultimatum":
        print(f"  Responder:  {args.responder}" +
              (f"  + {args.responder_enhancement}" if args.responder_enhancement else ""))
    else:
        print(f"  Responder:  [auto-accept (dictator game)]")
    print(f"  Games: {args.n_games}   Pool: ${args.pool}   Temp: {args.temperature}")
    print(f"{'=' * 70}\n")

    results = []
    t0 = time.time()

    # Cycle through variable pool sizes to break memorized-split convergence
    if args.variable_pools:
        pools = [POOL_SIZES[i % len(POOL_SIZES)] for i in range(args.n_games)]
        random.shuffle(pools)
    else:
        pools = [args.pool] * args.n_games

    for i in range(args.n_games):
        r = run_game_ultimatum(
            proposer_spec=proposer_spec,
            responder_spec=responder_spec,
            proposer_enhancement=args.proposer_enhancement,
            responder_enhancement=args.responder_enhancement,
            pool=pools[i],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            game=args.game,
        )
        r["game_id"] = i
        results.append(r)

        # Per-game status line
        pool_tag = f"  pool=${r['pool']}" if args.variable_pools else ""
        pct = f"({r['proposer_share']/r['pool']*100:.0f}%)" if r["proposer_share"] is not None else ""
        if r["parse_error"]:
            print(f"  Game {i+1:3d}: PARSE_ERROR ({r['parse_error']}){pool_tag}")
        elif r["agreed"]:
            print(f"  Game {i+1:3d}: ACCEPT  offer=${r['proposer_share']},{r['responder_share']} {pct}"
                  f"  payoffs=${r['proposer_payoff']},{r['responder_payoff']}{pool_tag}")
        else:
            print(f"  Game {i+1:3d}: REJECT  offer=${r['proposer_share']},{r['responder_share']} {pct}"
                  f"  payoffs=$0,$0{pool_tag}")

    elapsed = time.time() - t0

    # Summary
    summary = summarise(results)
    summary["elapsed_seconds"] = round(elapsed, 1)
    print_summary(summary, config)

    # Save
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build filename from config
        parts = [args.game, args.proposer.replace(":", "_")]
        if args.proposer_enhancement:
            parts.append(f"p-{args.proposer_enhancement}")
        parts.append(args.responder.replace(":", "_"))
        if args.responder_enhancement:
            parts.append(f"r-{args.responder_enhancement}")
        parts.append(f"n{args.n_games}")
        filename = "_".join(parts) + ".json"

        out_path = out_dir / filename
        with open(out_path, "w") as f:
            json.dump({
                "config": config,
                "summary": summary,
                "games": results,
            }, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
