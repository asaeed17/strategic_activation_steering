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
