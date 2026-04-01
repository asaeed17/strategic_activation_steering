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
MAX_ROUNDS = 8           # 8 total turns (4 per player), matching NegotiationArena
MAX_PROPOSALS = 3        # max proposals per player before forced to ACCEPT or NONE

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
    r"TRADE\s*:\s*GIVE\s+(\d+)\s+(X|Y)\s*,\s*RECEIVE\s+(\d+)\s+(X|Y)",
    re.IGNORECASE,
)
_ACCEPT_RE = re.compile(r"\bACCEPT\b", re.IGNORECASE)
_NONE_RE = re.compile(r"\bNONE\b", re.IGNORECASE)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM output into an action dict.

    Returns one of:
        {"type": "propose", "give_qty": int, "give_res": str,
         "receive_qty": int, "receive_res": str}
        {"type": "accept"}
        {"type": "none"}
        None  (parse failure)
    """
    # Check TRADE first (most specific pattern)
    m = _PROPOSE_RE.search(text)
    if m:
        give_qty, give_res, receive_qty, receive_res = (
            int(m.group(1)), m.group(2).upper(),
            int(m.group(3)), m.group(4).upper(),
        )
        if give_res == receive_res:
            return None  # cannot trade same resource
        return {
            "type": "propose",
            "give_qty": give_qty,
            "give_res": give_res,
            "receive_qty": receive_qty,
            "receive_res": receive_res,
        }

    # For ACCEPT/NONE, take the last match (same logic as UG's
    # parse_response which picks whichever appears last when both present)
    last_action = None
    last_pos = -1
    for pattern, action_type in [
        (_ACCEPT_RE, "accept"),
        (_NONE_RE, "none"),
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

def validate_proposal(
    proposal: Dict[str, Any],
    proposer_resources: Dict[str, int],
) -> bool:
    """Check that the proposer has enough to give.

    Does NOT check responder resources — that is private information the
    proposer should not have access to. Responder feasibility is checked
    at ACCEPT time instead.
    """
    if proposal["type"] != "propose":
        return False
    if proposal["give_qty"] <= 0 or proposal["receive_qty"] <= 0:
        return False
    if proposer_resources.get(proposal["give_res"], 0) < proposal["give_qty"]:
        return False
    return True


def execute_trade(
    proposer_resources: Dict[str, int],
    responder_resources: Dict[str, int],
    proposal: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Apply an accepted trade. Returns updated (proposer, responder) resources.

    Proposer gives give_qty of give_res, receives receive_qty of receive_res.
    Responder gives receive_qty of receive_res, receives give_qty of give_res.
    Quantities can be unequal — this is how players gain/lose total resources.
    """
    p = dict(proposer_resources)
    r = dict(responder_resources)
    # Proposer gives
    p[proposal["give_res"]] -= proposal["give_qty"]
    r[proposal["give_res"]] += proposal["give_qty"]
    # Proposer receives
    p[proposal["receive_res"]] += proposal["receive_qty"]
    r[proposal["receive_res"]] -= proposal["receive_qty"]
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

    - If a pending proposal exists: ACCEPT if it increases or maintains own score.
    - On own initiative: propose a slightly lopsided trade in own favour.
    - NONE if no beneficial trade or proposals exhausted.
    """
    own_score = compute_score(own_resources)

    # Respond to pending proposal
    if pending_proposal is not None and pending_proposal["type"] == "propose":
        # From responder's perspective: we give receive_res and get give_res
        hypothetical_own = dict(own_resources)
        hypothetical_own[pending_proposal["give_res"]] = (
            hypothetical_own.get(pending_proposal["give_res"], 0)
            + pending_proposal["give_qty"]
        )
        hypothetical_own[pending_proposal["receive_res"]] = (
            hypothetical_own.get(pending_proposal["receive_res"], 0)
            - pending_proposal["receive_qty"]
        )
        # Check feasibility
        if hypothetical_own[pending_proposal["receive_res"]] < 0:
            return {"type": "none"}
        if compute_score(hypothetical_own) >= own_score:
            return {"type": "accept"}
        return {"type": "none"}

    # Own initiative: propose a trade (give abundant, receive scarce)
    own_vals = [(res, own_resources.get(res, 0)) for res in RESOURCE_TYPES]
    own_vals.sort(key=lambda x: -x[1])  # descending
    give_res, give_have = own_vals[0]
    receive_res, receive_have = own_vals[-1]
    if give_res != receive_res and give_have > receive_have:
        give_qty = min(
            (give_have - receive_have) // 2,
            give_have,
        )
        # Ask for 1 more than we give (greedy but plausible)
        receive_qty = give_qty + 1
        if give_qty > 0 and receive_qty > 0:
            return {
                "type": "propose",
                "give_qty": give_qty,
                "give_res": give_res,
                "receive_qty": receive_qty,
                "receive_res": receive_res,
            }

    return {"type": "none"}


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

_PLAYER_NAMES = {1: "RED", 2: "BLUE"}


def build_player_system(
    player_id: int,
    own_resources: Dict[str, int],
) -> str:
    """System prompt for a player in the Resource Exchange Game."""
    name = _PLAYER_NAMES[player_id]
    opp_name = _PLAYER_NAMES[3 - player_id]
    return (
        f"You are Player {name} in a Resource Exchange Game.\n\n"
        f"You start with: {own_resources['X']} X and {own_resources['Y']} Y.\n"
        f"You have no knowledge about Player {opp_name}'s resources.\n"
        f"You can decide if you want to disclose your resources in your message.\n\n"
        f"Goal: Acquire as many total resources as possible. "
        f"More resources in general are always better.\n\n"
        f"Actions (choose exactly one per turn):\n"
        f"  TRADE: GIVE <qty> <X or Y>, RECEIVE <qty> <X or Y>\n"
        f"    You give away some of one resource and receive some of the other.\n"
        f"    The quantities do NOT have to be equal.\n"
        f"    You can only trade integer amounts, not decimals.\n"
        f"  ACCEPT — Accept the opponent's last trade proposal.\n"
        f"  NONE — Pass this turn without proposing or accepting.\n\n"
        f"Rules:\n"
        f"  - Up to {MAX_ROUNDS} total turns (alternating with Player {opp_name}).\n"
        f"  - You are allowed at most {MAX_PROPOSALS} proposals of your own. "
        f"After that you can only ACCEPT or NONE.\n"
        f"  - You can only ACCEPT when responding to a proposal.\n"
        f"  - If you ACCEPT, the trade executes and the game ends.\n"
        f"  - If nobody accepts by the end, no trade occurs and both keep initial resources.\n"
        f"  - You cannot give more resources than you currently have.\n\n"
        f"Respond with a 1-2 sentence message to the other player, "
        f"then end with your action on a new line.\n"
    )


def build_turn_prompt(
    player_id: int,
    own_resources: Dict[str, int],
    turn: int,
    opponent_message: Optional[str] = None,
    opponent_action: Optional[Dict[str, Any]] = None,
) -> str:
    """Per-turn user prompt showing current state and opponent's public message."""
    own_score = compute_score(own_resources)
    opp_name = _PLAYER_NAMES[3 - player_id]
    lines = [
        f"Round {turn + 1}/{MAX_ROUNDS}.",
        f"Your current resources: {own_resources['X']} X and {own_resources['Y']} Y "
        f"(total: {own_score}).",
    ]
    if opponent_message:
        lines.append(f"\nPlayer {opp_name} says: {opponent_message}")
    if opponent_action is not None and opponent_action["type"] == "propose":
        lines.append(
            f"\nPlayer {opp_name} proposes: "
            f"GIVE {opponent_action['give_qty']} {opponent_action['give_res']}, "
            f"RECEIVE {opponent_action['receive_qty']} {opponent_action['receive_res']}.\n"
            f"(You would give {opponent_action['receive_qty']} {opponent_action['receive_res']} "
            f"and receive {opponent_action['give_qty']} {opponent_action['give_res']}.)\n"
            f"Do you ACCEPT, propose a new TRADE, or NONE?"
        )
    else:
        lines.append("\nYour turn. Propose a TRADE, or NONE to pass.")
    return "\n".join(lines)


def format_action(action: Dict[str, Any]) -> str:
    """Format an action dict as the text the player would output."""
    if action["type"] == "propose":
        return (
            f"TRADE: GIVE {action['give_qty']} {action['give_res']}, "
            f"RECEIVE {action['receive_qty']} {action['receive_res']}"
        )
    return action["type"].upper()


def extract_message(text: str) -> str:
    """Extract public message from player output (strips action line).

    Players output 1-2 sentences followed by an action keyword on the last
    line.  The message portion (everything before the action line) is what
    the opponent sees next turn.
    """
    lines = text.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if (_PROPOSE_RE.search(lines[i]) or
                _ACCEPT_RE.search(lines[i]) or
                _NONE_RE.search(lines[i])):
            if i > 0:
                return "\n".join(lines[:i]).strip()
            break
    return text.strip()


def rule_based_message(action: Dict[str, Any]) -> str:
    """Generate a brief public message for a rule-based player's action."""
    if action["type"] == "propose":
        return (
            f"I'd like to propose a trade: {action['give_qty']} "
            f"{action['give_res']} for {action['receive_qty']} "
            f"{action['receive_res']}."
        )
    elif action["type"] == "accept":
        return "That works for me. I accept."
    return "I'll pass this turn."


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
    p1_messages = [{"role": "system", "content": build_player_system(1, p1_res)}]
    p2_messages = [{"role": "system", "content": build_player_system(2, p2_res)}]

    transcript = []
    trades_completed = 0
    pending_proposal = None
    proposing_player = None
    game_ended = False
    parse_errors = 0
    proposals_made = {1: 0, 2: 0}
    last_message = {1: None, 2: None}

    for turn in range(MAX_ROUNDS):
        # Alternate: player 1 goes on even turns, player 2 on odd
        active_player = 1 if turn % 2 == 0 else 2
        active_res = p1_res if active_player == 1 else p2_res
        opponent_res = p2_res if active_player == 1 else p1_res
        active_messages = p1_messages if active_player == 1 else p2_messages
        is_steered = (active_player == steered_player)

        # Determine if responding to a proposal or initiating
        opponent_msg = last_message[3 - active_player]
        if proposing_player is not None and proposing_player != active_player:
            turn_prompt = build_turn_prompt(
                active_player, active_res, turn,
                opponent_message=opponent_msg,
                opponent_action=pending_proposal,
            )
        else:
            turn_prompt = build_turn_prompt(
                active_player, active_res, turn,
                opponent_message=opponent_msg,
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

        # Update public message for opponent to see next turn
        if use_rulebased:
            last_message[active_player] = rule_based_message(action)
        else:
            last_message[active_player] = extract_message(text)

        metrics = extract_behavioral_metrics(text) if not use_rulebased else {}

        turn_record = {
            "turn": turn,
            "player": active_player,
            "text": text,
            "message": last_message[active_player],
            "action": action,
            "parse_error": False,
            "resources_before": dict(active_res),
            "metrics": metrics,
        }

        # Process action
        if action["type"] == "propose":
            if proposals_made[active_player] >= MAX_PROPOSALS:
                # Exceeded proposal limit — treat as NONE
                action = {"type": "none"}
                turn_record["action"] = action
                turn_record["proposal_limit_hit"] = True
            elif not validate_proposal(action, active_res):
                turn_record["invalid_trade"] = True
            else:
                pending_proposal = action
                proposing_player = active_player
                proposals_made[active_player] += 1

        elif action["type"] == "accept":
            if pending_proposal is not None and proposing_player != active_player:
                # Check responder can afford their side of the trade
                resp_res_check = p1_res if active_player == 1 else p2_res
                if resp_res_check.get(pending_proposal["receive_res"], 0) < pending_proposal["receive_qty"]:
                    # Responder can't afford — trade fails, turn passes
                    turn_record["accept_failed"] = True
                else:
                    # Execute the accepted trade
                    prop_res = p1_res if proposing_player == 1 else p2_res
                    resp_res = p1_res if active_player == 1 else p2_res
                    new_prop, new_resp = execute_trade(prop_res, resp_res, pending_proposal)
                    if proposing_player == 1:
                        p1_res, p2_res = new_prop, new_resp
                    else:
                        p2_res, p1_res = new_prop, new_resp
                    trades_completed += 1
                    turn_record["trade_executed"] = True
                    transcript.append(turn_record)
                    game_ended = True
                    break

        elif action["type"] == "none":
            pass  # do nothing, turn passes

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


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def summarise(
    games: List[Dict[str, Any]],
    player: int = 1,
) -> Dict[str, Any]:
    """Compute summary statistics for a single run."""
    scores = []
    trades = []
    lengths = []
    balances = []
    total_parse_errors = 0
    accepted = 0
    word_counts = []
    hedge_counts = []
    fairness_counts = []

    for g in games:
        scores.append(g[f"p{player}_score"])
        trades.append(g["trades_completed"])
        lengths.append(g["game_length"])
        balances.append(g[f"p{player}_balance"])
        total_parse_errors += g["parse_errors"]
        if g["trades_completed"] > 0:
            accepted += 1
        for t in g.get("transcript", []):
            m = t.get("metrics", {})
            if m:
                word_counts.append(m.get("word_count", 0))
                hedge_counts.append(m.get("hedge_count", 0))
                fairness_counts.append(m.get("fairness_count", 0))

    n = len(games)
    return {
        "n_games": n,
        "player": player,
        "mean_score": float(np.mean(scores)) if scores else 0,
        "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0,
        "mean_trades": float(np.mean(trades)) if trades else 0,
        "mean_game_length": float(np.mean(lengths)) if lengths else 0,
        "mean_balance": float(np.mean(balances)) if balances else 0,
        "acceptance_rate": accepted / n if n else 0,
        "total_parse_errors": total_parse_errors,
        "mean_word_count": float(np.mean(word_counts)) if word_counts else 0,
        "mean_hedge_count": float(np.mean(hedge_counts)) if hedge_counts else 0,
        "mean_fairness_count": float(np.mean(fairness_counts)) if fairness_counts else 0,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"Resource Exchange — Summary (n={summary['n_games']})")
    print(f"{'=' * 60}")
    print(f"  Mean score:       {summary['mean_score']:.1f} (+/- {summary['std_score']:.1f})")
    print(f"  Mean trades:      {summary['mean_trades']:.1f}")
    print(f"  Mean game length: {summary['mean_game_length']:.1f}")
    print(f"  Acceptance rate:  {summary['acceptance_rate']:.0%}")
    print(f"  Parse errors:     {summary['total_parse_errors']}")
    print(f"  Mean word count:  {summary['mean_word_count']:.1f}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Validate game logic without GPU."""
    print("Running self-tests...")

    # Scoring
    assert compute_score({"X": 10, "Y": 20}) == 30
    assert compute_score({"X": 0, "Y": 0}) == 0

    # Balance
    assert compute_balance_ratio({"X": 10, "Y": 10}) == 1.0
    assert compute_balance_ratio({"X": 5, "Y": 10}) == 0.5
    assert compute_balance_ratio({"X": 0, "Y": 10}) == 0.0

    # Parsing
    assert parse_action("TRADE: GIVE 5 X, RECEIVE 5 Y") == {
        "type": "propose", "give_qty": 5, "give_res": "X",
        "receive_qty": 5, "receive_res": "Y",
    }
    assert parse_action("TRADE: GIVE 10 X, RECEIVE 15 Y") == {
        "type": "propose", "give_qty": 10, "give_res": "X",
        "receive_qty": 15, "receive_res": "Y",
    }
    assert parse_action("I accept this trade. ACCEPT")["type"] == "accept"
    assert parse_action("I'll pass. NONE")["type"] == "none"
    assert parse_action("gibberish with no action") is None
    # Same resource should fail
    assert parse_action("TRADE: GIVE 5 X, RECEIVE 5 X") is None

    # Proposal validation (only checks proposer, not responder)
    proposer = {"X": 10, "Y": 5}
    good_trade = {"type": "propose", "give_qty": 3, "give_res": "X",
                  "receive_qty": 3, "receive_res": "Y"}
    assert validate_proposal(good_trade, proposer) is True
    bad_trade = {"type": "propose", "give_qty": 20, "give_res": "X",
                 "receive_qty": 3, "receive_res": "Y"}
    assert validate_proposal(bad_trade, proposer) is False
    # Responder can't afford — but proposal is still valid (private info)
    big_ask = {"type": "propose", "give_qty": 3, "give_res": "X",
               "receive_qty": 50, "receive_res": "Y"}
    assert validate_proposal(big_ask, proposer) is True

    # Trade execution (unequal: give 3X, receive 5Y → proposer gains 2 net)
    p, r = execute_trade(
        {"X": 10, "Y": 5}, {"X": 5, "Y": 10},
        {"type": "propose", "give_qty": 3, "give_res": "X",
         "receive_qty": 5, "receive_res": "Y"},
    )
    assert p == {"X": 7, "Y": 10}   # gave 3X, got 5Y → net +2
    assert r == {"X": 8, "Y": 5}    # got 3X, gave 5Y → net -2

    # Rule-based opponent — accept improving trade
    own = {"X": 5, "Y": 25}
    opp = {"X": 25, "Y": 5}
    # Opponent proposes: GIVE 5 X, RECEIVE 3 Y → we give 3Y, get 5X → net +2
    proposal = {"type": "propose", "give_qty": 5, "give_res": "X",
                "receive_qty": 3, "receive_res": "Y"}
    action = rule_based_action(own, opp, proposal, turn=1)
    assert action["type"] == "accept", f"Expected accept, got {action}"

    # Rule-based opponent — reject harmful trade
    # Opponent proposes: GIVE 3 X, RECEIVE 5 Y → we give 5Y, get 3X → net -2
    bad_proposal = {"type": "propose", "give_qty": 3, "give_res": "X",
                    "receive_qty": 5, "receive_res": "Y"}
    action = rule_based_action(own, opp, bad_proposal, turn=1)
    assert action["type"] == "none", f"Expected none, got {action}"

    # Full game with rule-based both sides
    result = run_single_exchange_game(
        config=(25, 5, 5, 25),
        steered_player=None,
        generate_fn=None,
        rulebased=True,
    )
    assert result["p1_score"] >= 0
    assert result["p2_score"] >= 0
    assert result["parse_errors"] == 0
    assert len(result["transcript"]) > 0

    print(f"  P1: {result['p1_initial']} -> {result['p1_final']} "
          f"(score {result['p1_initial_score']} -> {result['p1_score']})")
    print(f"  P2: {result['p2_initial']} -> {result['p2_final']} "
          f"(score {result['p2_initial_score']} -> {result['p2_score']})")
    print(f"  Trades: {result['trades_completed']}, "
          f"Length: {result['game_length']}")
    print("All self-tests passed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resource Exchange Game with activation steering",
    )
    p.add_argument("--test", action="store_true",
                   help="Run self-tests (no GPU needed)")
    p.add_argument("--model", default="qwen2.5-7b")
    p.add_argument("--dimension", default=None,
                   help="Steering dimension (e.g., firmness, empathy)")
    p.add_argument("--method", choices=["mean_diff", "pca", "logreg"],
                   default="mean_diff")
    p.add_argument("--layers", nargs="+", type=int, default=[10])
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--steered_player", type=int, choices=[1, 2], default=1)
    p.add_argument("--vectors_dir",
                   default="vectors/ultimatum_10dim_20pairs_general_matched/negotiation")
    p.add_argument("--n_games", type=int, default=50)
    p.add_argument("--rulebased", action="store_true",
                   help="Use rule-based opponent instead of LLM-vs-LLM")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.test:
        _run_self_test()
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    import os
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / ".hf_cache")

    # Lazy imports for GPU mode
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from extract_vectors import MODELS, HF_TOKEN
    from apply_steering_preset_nego import (
        load_direction_vectors, generate_response, get_transformer_layers,
    )

    model_config = MODELS[args.model]
    hf_token = HF_TOKEN if model_config.requires_token else None

    # Load model
    log.info("Loading model: %s", model_config.hf_id)
    if model_config.is_gptq:
        from auto_gptq import AutoGPTQForCausalLM
        n_gpus = torch.cuda.device_count()
        max_memory = {i: "22GiB" for i in range(n_gpus)}
        model = AutoGPTQForCausalLM.from_quantized(
            model_config.hf_id,
            use_safetensors=True,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=False,
        )
    else:
        load_kwargs = dict(token=hf_token, device_map="auto")
        if args.quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_config.hf_id, **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id, token=hf_token)

    # Load direction vectors
    dvecs = None
    if args.dimension and args.alpha != 0.0:
        dvecs = load_direction_vectors(
            vectors_dir=Path(args.vectors_dir),
            model_alias=model_config.alias,
            dimension=args.dimension,
            method=args.method,
            layer_indices=args.layers,
        )
        log.info("Loaded vectors: dim=%s layers=%s method=%s",
                 args.dimension, args.layers, args.method)

    # Build generate function
    def generate_fn(messages, is_steered):
        dv = dvecs if is_steered else None
        a = args.alpha if is_steered else 0.0
        return generate_response(
            model, tokenizer, messages, dv, a,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # Select configs
    configs = RESOURCE_CONFIGS[:args.n_games]
    if len(configs) < args.n_games:
        log.warning("Only %d configs available, requested %d", len(configs), args.n_games)

    # Run games
    steered_player = args.steered_player if args.dimension else None
    games = []
    for i, config in enumerate(configs):
        log.info("[G%03d] config=(%d,%d,%d,%d)", i + 1, *config)
        result = run_single_exchange_game(
            config=config,
            steered_player=steered_player,
            generate_fn=generate_fn,
            rulebased=args.rulebased,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        result["game_id"] = i
        games.append(result)

        sp = args.steered_player
        log.info(
            "[G%03d] dim=%s a=%s P%d: %dX/%dY (score %d, trades %d, len %d)",
            i + 1, args.dimension or "baseline", args.alpha, sp,
            result[f"p{sp}_final"]["X"], result[f"p{sp}_final"]["Y"],
            result[f"p{sp}_score"], result["trades_completed"],
            result["game_length"],
        )

    # Summary
    summary = summarise(games, args.steered_player)
    print_summary(summary)

    # Save results
    output = {
        "config": {
            "game": "resource_exchange",
            "model": args.model,
            "dimension": args.dimension,
            "method": args.method,
            "layers": args.layers,
            "alpha": args.alpha,
            "steered_player": args.steered_player,
            "rulebased": args.rulebased,
            "n_games": len(games),
            "variable_configs": True,
            "max_turns": MAX_ROUNDS,
            "proposals_per_player": MAX_PROPOSALS,
            "temperature": args.temperature,
            "seed": args.seed,
            "vectors_dir": args.vectors_dir,
            "private_info": True,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "games": games,
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        log.info("Results saved to %s", out_file)
    else:
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
