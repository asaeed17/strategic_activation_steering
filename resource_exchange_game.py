#!/usr/bin/env python3
"""
resource_exchange_game.py

Resource Exchange Game with activation steering, based on NegotiationArena
(Bianchi et al., ICML 2024). Two players with complementary asymmetric
resource bundles (X and Y) trade over multiple rounds to maximise total
resources.

This serves as cross-task validation for steering effects found in the
Ultimatum Game: if dose-response and layer-location findings replicate
on a structurally different (multi-turn, multi-issue) game, the claims
are more generalisable.

Usage:
    # Self-test (no GPU needed)
    python resource_exchange_game.py --test

    # LLM-vs-LLM with steering (GPU required)
    python resource_exchange_game.py --model qwen2.5-7b --dimension firmness \\
        --layers 10 --alpha 7 --steered_player 1 --n_games 50 --paired \\
        --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \\
        --output_dir results/resource_exchange/firmness_L10_a7

    # Rule-based opponent (cheaper gridsearch)
    python resource_exchange_game.py --model qwen2.5-7b --dimension firmness \\
        --layers 10 --alpha 7 --steered_player 1 --n_games 50 --paired \\
        --rulebased --output_dir results/resource_exchange/firmness_L10_a7_rb
"""

import re
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCE_TYPES = ["X", "Y"]
MAX_ROUNDS = 8

# Variable resource configs (analogous to POOL_SIZES in ultimatum_game.py).
# Each entry: (p1_x, p1_y, p2_x, p2_y). Complementary asymmetric bundles
# so that trading is beneficial. Total resources are conserved (zero-sum trades).
RESOURCE_CONFIGS = [
    # NegotiationArena base config
    (25, 5, 5, 25),
    # Variations on asymmetry ratio
    (20, 10, 10, 20),
    (30, 5, 5, 30),
    (15, 5, 5, 15),
    (20, 5, 5, 20),
    (25, 10, 10, 25),
    (30, 10, 10, 30),
    (15, 10, 10, 15),
    (35, 5, 5, 35),
    (40, 5, 5, 40),
    # Moderate asymmetry
    (18, 12, 12, 18),
    (22, 8, 8, 22),
    (16, 14, 14, 16),
    (28, 7, 7, 28),
    (24, 6, 6, 24),
    # Different total quantities
    (10, 3, 3, 10),
    (12, 4, 4, 12),
    (8, 2, 2, 8),
    (35, 10, 10, 35),
    (40, 10, 10, 40),
    # Asymmetric totals (P1 has more overall)
    (25, 10, 5, 20),
    (30, 5, 10, 25),
    (20, 15, 10, 10),
    (15, 5, 10, 25),
    (25, 5, 10, 20),
    # Reversed asymmetry direction
    (5, 25, 25, 5),
    (10, 20, 20, 10),
    (5, 30, 30, 5),
    (10, 25, 25, 10),
    (5, 15, 15, 5),
    # Near-symmetric (control)
    (15, 15, 15, 15),
    (20, 20, 20, 20),
    (14, 16, 16, 14),
    (13, 17, 17, 13),
    (12, 18, 18, 12),
    # Extreme asymmetry
    (30, 2, 2, 30),
    (35, 3, 3, 35),
    (28, 2, 2, 28),
    (40, 3, 3, 40),
    (25, 2, 2, 25),
    # Odd numbers
    (19, 7, 7, 19),
    (23, 9, 9, 23),
    (27, 11, 11, 27),
    (17, 6, 6, 17),
    (21, 3, 3, 21),
    # Small games
    (6, 2, 2, 6),
    (7, 3, 3, 7),
    (9, 1, 1, 9),
    (5, 3, 3, 5),
    (11, 4, 4, 11),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(resources: Dict[str, int]) -> int:
    """Total resource count. Matches NegotiationArena's maximisation goal."""
    return sum(resources.values())


def compute_balance_ratio(resources: Dict[str, int]) -> float:
    """Ratio of min resource to max resource (1.0 = perfectly balanced)."""
    vals = list(resources.values())
    if max(vals) == 0:
        return 0.0
    return min(vals) / max(vals)


# ---------------------------------------------------------------------------
# Action Parsing
# ---------------------------------------------------------------------------

_PROPOSE_RE = re.compile(
    r"PROPOSE_TRADE\s*:\s*SELL\s+(\d+)\s+(X|Y)\s*,\s*BUY\s+(\d+)\s+(X|Y)",
    re.IGNORECASE,
)
_ACCEPT_RE = re.compile(r"\bACCEPT\b", re.IGNORECASE)
_REJECT_RE = re.compile(r"\bREJECT\b", re.IGNORECASE)
_END_RE = re.compile(r"\bEND\b", re.IGNORECASE)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM output into an action dict.

    Returns one of:
        {"type": "propose", "sell_qty": int, "sell_res": str,
         "buy_qty": int, "buy_res": str}
        {"type": "accept"}
        {"type": "reject"}
        {"type": "end"}
        None  (parse failure)
    """
    # Check PROPOSE_TRADE first (most specific pattern)
    m = _PROPOSE_RE.search(text)
    if m:
        sell_qty, sell_res, buy_qty, buy_res = (
            int(m.group(1)), m.group(2).upper(),
            int(m.group(3)), m.group(4).upper(),
        )
        if sell_res == buy_res:
            return None  # cannot trade same resource
        return {
            "type": "propose",
            "sell_qty": sell_qty,
            "sell_res": sell_res,
            "buy_qty": buy_qty,
            "buy_res": buy_res,
        }

    # For ACCEPT/REJECT/END, take the last match (same logic as UG's
    # parse_response which picks whichever appears last when both present)
    last_action = None
    last_pos = -1
    for pattern, action_type in [
        (_ACCEPT_RE, "accept"),
        (_REJECT_RE, "reject"),
        (_END_RE, "end"),
    ]:
        for match in pattern.finditer(text):
            if match.start() > last_pos:
                last_pos = match.start()
                last_action = action_type

    if last_action:
        return {"type": last_action}

    return None


# ---------------------------------------------------------------------------
# Trade Validation & Execution
# ---------------------------------------------------------------------------

def validate_trade(
    proposal: Dict[str, Any],
    seller_resources: Dict[str, int],
    buyer_resources: Dict[str, int],
) -> bool:
    """Check that the proposer can sell and the counterparty can buy."""
    if proposal["type"] != "propose":
        return False
    sell_res, sell_qty = proposal["sell_res"], proposal["sell_qty"]
    buy_res, buy_qty = proposal["buy_res"], proposal["buy_qty"]
    if sell_qty <= 0 or buy_qty <= 0:
        return False
    if seller_resources.get(sell_res, 0) < sell_qty:
        return False
    if buyer_resources.get(buy_res, 0) < buy_qty:
        return False
    return True


def execute_trade(
    proposer_resources: Dict[str, int],
    responder_resources: Dict[str, int],
    proposal: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Apply an accepted trade. Returns updated (proposer, responder) resources."""
    p = dict(proposer_resources)
    r = dict(responder_resources)
    p[proposal["sell_res"]] -= proposal["sell_qty"]
    r[proposal["sell_res"]] += proposal["sell_qty"]
    p[proposal["buy_res"]] += proposal["buy_qty"]
    r[proposal["buy_res"]] -= proposal["buy_qty"]
    return p, r


def extract_behavioral_metrics(text: str) -> Dict[str, Any]:
    """Extract behavioral metrics from LLM response text."""
    words = text.split()
    hedge_pattern = re.compile(
        r"\b(maybe|perhaps|might|could|I think|probably|possibly)\b",
        re.IGNORECASE,
    )
    fairness_pattern = re.compile(
        r"\b(fair|equal|both|reasonable|mutual|together|balanced)\b",
        re.IGNORECASE,
    )
    return {
        "word_count": len(words),
        "hedge_count": len(hedge_pattern.findall(text)),
        "fairness_count": len(fairness_pattern.findall(text)),
    }


# ---------------------------------------------------------------------------
# Rule-Based Opponent
# ---------------------------------------------------------------------------

def rule_based_action(
    own_resources: Dict[str, int],
    opponent_resources: Dict[str, int],
    pending_proposal: Optional[Dict[str, Any]],
    turn: int,
) -> Dict[str, Any]:
    """Deterministic greedy opponent.

    - If a pending proposal exists: ACCEPT if it increases own score, REJECT otherwise.
    - On own initiative: propose the trade that maximises own score gain
      (sell the resource we have most of, buy the one we have least of).
    - END if no beneficial trade exists or if turn >= MAX_ROUNDS - 1.
    """
    own_score = compute_score(own_resources)

    # Respond to pending proposal
    if pending_proposal is not None and pending_proposal["type"] == "propose":
        # From our perspective: opponent proposed to sell us something and buy from us
        # We are the responder: we give buy_res and receive sell_res
        hypothetical_own = dict(own_resources)
        hypothetical_own[pending_proposal["sell_res"]] = (
            hypothetical_own.get(pending_proposal["sell_res"], 0)
            + pending_proposal["sell_qty"]
        )
        hypothetical_own[pending_proposal["buy_res"]] = (
            hypothetical_own.get(pending_proposal["buy_res"], 0)
            - pending_proposal["buy_qty"]
        )
        # Check feasibility
        if hypothetical_own[pending_proposal["buy_res"]] < 0:
            return {"type": "reject"}
        if compute_score(hypothetical_own) > own_score:
            return {"type": "accept"}
        return {"type": "reject"}

    # Own initiative: find best trade to propose
    if turn >= MAX_ROUNDS - 1:
        return {"type": "end"}

    best_proposal = None
    best_gain = 0
    for sell_res in RESOURCE_TYPES:
        for buy_res in RESOURCE_TYPES:
            if sell_res == buy_res:
                continue
            max_sell = own_resources.get(sell_res, 0)
            max_buy = opponent_resources.get(buy_res, 0)
            for qty in range(1, min(max_sell, max_buy) + 1):
                hyp = dict(own_resources)
                hyp[sell_res] -= qty
                hyp[buy_res] += qty
                gain = compute_score(hyp) - own_score
                if gain > best_gain:
                    best_gain = gain
                    best_proposal = {
                        "type": "propose",
                        "sell_qty": qty,
                        "sell_res": sell_res,
                        "buy_qty": qty,
                        "buy_res": buy_res,
                    }

    if best_proposal is not None:
        return best_proposal
    return {"type": "end"}


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

_PLAYER_NAMES = {1: "RED", 2: "BLUE"}


def build_player_system(
    player_id: int,
    own_resources: Dict[str, int],
    opponent_resources: Dict[str, int],
) -> str:
    """System prompt for a player in the Resource Exchange Game."""
    name = _PLAYER_NAMES[player_id]
    opp_name = _PLAYER_NAMES[3 - player_id]
    return (
        f"You are Player {name} in a Resource Exchange Game.\n\n"
        f"You start with: {own_resources['X']} X and {own_resources['Y']} Y.\n"
        f"Player {opp_name} starts with: {opponent_resources['X']} X "
        f"and {opponent_resources['Y']} Y.\n\n"
        f"Goal: Acquire as many total resources as possible. "
        f"More resources in general are always better.\n\n"
        f"Actions (choose exactly one per turn):\n"
        f"  PROPOSE_TRADE: SELL <qty> <X or Y>, BUY <qty> <X or Y>\n"
        f"    You give away <qty> of one resource and receive <qty> of the other.\n"
        f"  ACCEPT — Accept the opponent's last trade proposal.\n"
        f"  REJECT — Reject the opponent's last trade proposal.\n"
        f"  END — End the game. No more trades.\n\n"
        f"Rules:\n"
        f"  - Up to {MAX_ROUNDS} rounds. You alternate turns with Player {opp_name}.\n"
        f"  - You can only ACCEPT or REJECT when responding to a proposal.\n"
        f"  - You can only PROPOSE_TRADE or END on your own initiative.\n"
        f"  - You cannot sell more resources than you currently have.\n\n"
        f"Respond with 1-2 sentences explaining your reasoning, "
        f"then end with your action on a new line.\n"
    )


def build_turn_prompt(
    player_id: int,
    own_resources: Dict[str, int],
    opponent_resources: Dict[str, int],
    turn: int,
    pending_proposal: Optional[Dict[str, Any]] = None,
) -> str:
    """Per-turn user prompt showing current state."""
    own_score = compute_score(own_resources)
    opp_name = _PLAYER_NAMES[3 - player_id]
    lines = [
        f"Round {turn + 1}/{MAX_ROUNDS}.",
        f"Your current resources: {own_resources['X']} X and {own_resources['Y']} Y "
        f"(total: {own_score}).",
    ]
    if pending_proposal is not None and pending_proposal["type"] == "propose":
        lines.append(
            f"\nPlayer {opp_name} proposes: "
            f"SELL {pending_proposal['sell_qty']} {pending_proposal['sell_res']}, "
            f"BUY {pending_proposal['buy_qty']} {pending_proposal['buy_res']}.\n"
            f"Do you ACCEPT or REJECT?"
        )
    else:
        lines.append("\nYour turn. Propose a trade or END the game.")
    return "\n".join(lines)


def format_action(action: Dict[str, Any]) -> str:
    """Format an action dict as the text the player would output."""
    if action["type"] == "propose":
        return (
            f"PROPOSE_TRADE: SELL {action['sell_qty']} {action['sell_res']}, "
            f"BUY {action['buy_qty']} {action['buy_res']}"
        )
    return action["type"].upper()


# ---------------------------------------------------------------------------
# Game Loop
# ---------------------------------------------------------------------------

def _make_resources(config: Tuple[int, int, int, int], player: int) -> Dict[str, int]:
    """Extract player resources from a config tuple (p1_x, p1_y, p2_x, p2_y)."""
    if player == 1:
        return {"X": config[0], "Y": config[1]}
    return {"X": config[2], "Y": config[3]}


def run_single_exchange_game(
    config: Tuple[int, int, int, int],
    steered_player: Optional[int],
    generate_fn=None,
    rulebased: bool = True,
    temperature: float = 0.0,
    max_new_tokens: int = 200,
) -> Dict[str, Any]:
    """Run one resource exchange game.

    Args:
        config: (p1_x, p1_y, p2_x, p2_y) initial resources.
        steered_player: 1 or 2 (which player is steered), or None for baseline.
        generate_fn: Callable(messages, is_steered) -> str. If None, must be rulebased.
        rulebased: If True, the non-steered player uses rule_based_action.
        temperature: For LLM generation.
        max_new_tokens: For LLM generation.

    Returns:
        Result dict with game transcript, final resources, scores, metrics.
    """
    p1_res = _make_resources(config, 1)
    p2_res = _make_resources(config, 2)
    p1_initial = dict(p1_res)
    p2_initial = dict(p2_res)

    # Conversation histories (separate for each player)
    p1_messages = [{"role": "system", "content": build_player_system(1, p1_res, p2_res)}]
    p2_messages = [{"role": "system", "content": build_player_system(2, p2_res, p1_res)}]

    transcript = []
    trades_completed = 0
    pending_proposal = None
    proposing_player = None
    game_ended = False
    parse_errors = 0

    for turn in range(MAX_ROUNDS):
        # Alternate: player 1 goes on even turns, player 2 on odd
        active_player = 1 if turn % 2 == 0 else 2
        active_res = p1_res if active_player == 1 else p2_res
        opponent_res = p2_res if active_player == 1 else p1_res
        active_messages = p1_messages if active_player == 1 else p2_messages
        is_steered = (active_player == steered_player)

        # Determine if responding to a proposal or initiating
        if proposing_player is not None and proposing_player != active_player:
            # Responding to opponent's proposal
            turn_prompt = build_turn_prompt(
                active_player, active_res, opponent_res, turn,
                pending_proposal=pending_proposal,
            )
        else:
            # Own initiative
            turn_prompt = build_turn_prompt(
                active_player, active_res, opponent_res, turn,
            )

        # Generate action
        use_rulebased = rulebased and not is_steered
        if use_rulebased:
            if proposing_player is not None and proposing_player != active_player:
                action = rule_based_action(
                    active_res, opponent_res, pending_proposal, turn,
                )
            else:
                action = rule_based_action(active_res, opponent_res, None, turn)
            text = format_action(action)
        else:
            if generate_fn is None:
                raise ValueError("generate_fn required for LLM players")
            active_messages.append({"role": "user", "content": turn_prompt})
            text = generate_fn(active_messages, is_steered)
            active_messages.append({"role": "assistant", "content": text})
            action = parse_action(text)
            if action is None:
                parse_errors += 1
                transcript.append({
                    "turn": turn,
                    "player": active_player,
                    "text": text,
                    "action": None,
                    "parse_error": True,
                })
                pending_proposal = None
                proposing_player = None
                continue

        metrics = extract_behavioral_metrics(text) if not use_rulebased else {}

        turn_record = {
            "turn": turn,
            "player": active_player,
            "text": text,
            "action": action,
            "parse_error": False,
            "resources_before": dict(active_res),
            "metrics": metrics,
        }

        # Process action
        if action["type"] == "propose":
            if not validate_trade(action, active_res, opponent_res):
                turn_record["invalid_trade"] = True
                pending_proposal = None
                proposing_player = None
            else:
                pending_proposal = action
                proposing_player = active_player

        elif action["type"] == "accept":
            if pending_proposal is not None and proposing_player != active_player:
                # Execute the trade
                prop_res = p1_res if proposing_player == 1 else p2_res
                resp_res = p1_res if active_player == 1 else p2_res
                new_prop, new_resp = execute_trade(prop_res, resp_res, pending_proposal)
                if proposing_player == 1:
                    p1_res, p2_res = new_prop, new_resp
                else:
                    p2_res, p1_res = new_prop, new_resp
                trades_completed += 1
                turn_record["trade_executed"] = True
            pending_proposal = None
            proposing_player = None

        elif action["type"] == "reject":
            pending_proposal = None
            proposing_player = None

        elif action["type"] == "end":
            turn_record["game_ended"] = True
            transcript.append(turn_record)
            game_ended = True
            break

        transcript.append(turn_record)

    return {
        "config": config,
        "p1_initial": p1_initial,
        "p2_initial": p2_initial,
        "p1_final": p1_res,
        "p2_final": p2_res,
        "p1_score": compute_score(p1_res),
        "p2_score": compute_score(p2_res),
        "p1_initial_score": compute_score(p1_initial),
        "p2_initial_score": compute_score(p2_initial),
        "p1_balance": compute_balance_ratio(p1_res),
        "p2_balance": compute_balance_ratio(p2_res),
        "trades_completed": trades_completed,
        "game_length": len(transcript),
        "game_ended_naturally": game_ended,
        "parse_errors": parse_errors,
        "steered_player": steered_player,
        "transcript": transcript,
    }
