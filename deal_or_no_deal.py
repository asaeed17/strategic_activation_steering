#!/usr/bin/env python3
"""
deal_or_no_deal.py

Deal or No Deal dataset integration for cross-dataset validation.

Lewis et al. 2017 created a multi-issue negotiation dataset where two agents
split a pool of items (books, hats, balls). Each agent has a private value
function — the same items are worth different amounts to each player. This
makes it possible to measure Pareto efficiency: did the agents find a split
that maximises total value?

This matters because CraigslistBargains is single-issue (price only). If our
`strategic_concession_making` vector really captures multi-issue reasoning
(not just stubbornness), it should improve Pareto efficiency here. If it
doesn't, the +37% on Craigslist is likely just firmness under a different label.

Usage:
    # Standalone test (no model needed)
    python deal_or_no_deal.py

    # Integration with apply_steering.py
    from deal_or_no_deal import load_dealornodeal, run_game_dond, score_deal_dond
"""

import re
import json
import random
import logging
import urllib.request
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

ITEM_NAMES = ["books", "hats", "balls"]

# GitHub raw URLs for the archived Facebook Research repo (Lewis et al. 2017).
# Repo archived 2021-08-28 but raw content still served.
DOND_URLS = {
    "selfplay": "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/selfplay.txt",
    "train":    "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/train.txt",
    "val":      "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/val.txt",
    "test":     "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/test.txt",
}

CACHE_DIR = Path("data/dealornodeal")

MIN_TURNS_BEFORE_DEAL = 3
MAX_TURNS             = 10


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    log.info("Downloading from %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def _fetch_text_cached(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        return cache_path.read_text()

    text = _fetch_text(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    log.info("Cached to %s", cache_path)
    return text


def _parse_context(six_ints: List[int]) -> Tuple[List[int], List[int]]:
    """Parse 6 integers into (counts, values) for 3 item types.

    Format: count0 value0 count1 value1 count2 value2
    """
    counts = [six_ints[0], six_ints[2], six_ints[4]]
    values = [six_ints[1], six_ints[3], six_ints[5]]
    return counts, values


def _parse_selfplay(text: str) -> List[Dict]:
    """Parse selfplay.txt: consecutive line pairs are (agent1, agent2) contexts."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    scenarios = []

    for i in range(0, len(lines) - 1, 2):
        try:
            ints1 = list(map(int, lines[i].split()))
            ints2 = list(map(int, lines[i + 1].split()))
            if len(ints1) != 6 or len(ints2) != 6:
                continue

            counts1, values1 = _parse_context(ints1)
            counts2, values2 = _parse_context(ints2)

            if counts1 != counts2:
                continue

            total1 = sum(c * v for c, v in zip(counts1, values1))
            total2 = sum(c * v for c, v in zip(counts2, values2))
            if total1 != 10 or total2 != 10:
                continue

            scenarios.append({
                "counts":      counts1,
                "values_1":    values1,
                "values_2":    values2,
                "item_names":  list(ITEM_NAMES),
                "total_items": sum(counts1),
                "max_score":   10,
            })
        except (ValueError, IndexError):
            continue

    return scenarios


def _parse_dialogue_file(text: str) -> List[Dict]:
    """Parse train/val/test.txt: extract unique contexts from tagged fields."""
    scenarios = []
    seen = set()

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        try:
            m_input   = re.search(r"<input>\s*(.*?)\s*</input>", line)
            m_partner = re.search(r"<partner_input>\s*(.*?)\s*</partner_input>", line)

            if not m_input or not m_partner:
                continue

            ints1 = list(map(int, m_input.group(1).split()))
            ints2 = list(map(int, m_partner.group(1).split()))

            if len(ints1) != 6 or len(ints2) != 6:
                continue

            counts1, values1 = _parse_context(ints1)
            counts2, values2 = _parse_context(ints2)

            if counts1 != counts2:
                continue

            total1 = sum(c * v for c, v in zip(counts1, values1))
            total2 = sum(c * v for c, v in zip(counts2, values2))
            if total1 != 10 or total2 != 10:
                continue

            key = (tuple(counts1), tuple(values1), tuple(values2))
            if key in seen:
                continue
            seen.add(key)

            scenarios.append({
                "counts":      counts1,
                "values_1":    values1,
                "values_2":    values2,
                "item_names":  list(ITEM_NAMES),
                "total_items": sum(counts1),
                "max_score":   10,
            })
        except (ValueError, IndexError):
            continue

    return scenarios


def load_dealornodeal(
    split: str = "selfplay",
    num_samples: int = 50,
) -> List[Dict]:
    """Load Deal or No Deal scenarios.

    Args:
        split: One of "selfplay", "train", "val", "test".
               selfplay has the most scenarios (8000+) and no historical
               dialogues — ideal for running fresh LLM negotiations.
        num_samples: Number of scenarios to return (randomly sampled).

    Returns:
        List of scenario dicts, each with:
            counts:      [int, int, int]  — items available (books, hats, balls)
            values_1:    [int, int, int]  — agent 1's per-unit values
            values_2:    [int, int, int]  — agent 2's per-unit values
            item_names:  ["books", "hats", "balls"]
            total_items: int              — sum of counts
            max_score:   10               — max utility per agent
    """
    if split not in DOND_URLS:
        raise ValueError(f"Split '{split}' not available. Choose from: {list(DOND_URLS.keys())}")

    cache_path = CACHE_DIR / f"{split}.txt"
    text = _fetch_text_cached(DOND_URLS[split], cache_path)

    if split == "selfplay":
        scenarios = _parse_selfplay(text)
    else:
        scenarios = _parse_dialogue_file(text)

    log.info("Parsed %d unique scenarios from '%s' split.", len(scenarios), split)

    if not scenarios:
        raise RuntimeError(f"No valid scenarios found in '{split}' split.")

    k = min(num_samples, len(scenarios))
    return random.sample(scenarios, k)


# ---------------------------------------------------------------------------
# Pareto Efficiency & Scoring
# ---------------------------------------------------------------------------

def _enumerate_allocations(counts: List[int]):
    """Generate all valid allocations of items between two agents.

    For each item type with count n, agent 1 can take 0..n, agent 2 gets
    the rest. With 5-7 items across 3 types the search space is small
    (at most ~120 combinations).
    """
    ranges = [range(c + 1) for c in counts]
    for combo in product(*ranges):
        picks_1 = list(combo)
        picks_2 = [c - p for c, p in zip(counts, picks_1)]
        yield picks_1, picks_2


def _compute_utility(values: List[int], picks: List[int]) -> int:
    return sum(v * p for v, p in zip(values, picks))


def is_pareto_optimal(
    counts:   List[int],
    values_1: List[int],
    picks_1:  List[int],
    values_2: List[int],
    picks_2:  List[int],
) -> bool:
    """Check if a deal is Pareto optimal (no allocation Pareto-dominates it)."""
    score_1 = _compute_utility(values_1, picks_1)
    score_2 = _compute_utility(values_2, picks_2)

    for cand_1, cand_2 in _enumerate_allocations(counts):
        cand_s1 = _compute_utility(values_1, cand_1)
        cand_s2 = _compute_utility(values_2, cand_2)
        if ((cand_s1 > score_1 and cand_s2 >= score_2) or
                (cand_s1 >= score_1 and cand_s2 > score_2)):
            return False

    return True


def max_joint_utility(counts: List[int], values_1: List[int], values_2: List[int]) -> int:
    """Find the maximum possible sum of both agents' utilities."""
    best = 0
    for p1, p2 in _enumerate_allocations(counts):
        joint = _compute_utility(values_1, p1) + _compute_utility(values_2, p2)
        if joint > best:
            best = joint
    return best


def score_deal_dond(
    picks_1:  List[int],
    picks_2:  List[int],
    scenario: Dict,
) -> Dict:
    """Score a Deal or No Deal outcome.

    Returns:
        score_1:        Agent 1's raw utility (0-10)
        score_2:        Agent 2's raw utility (0-10)
        norm_score_1:   Normalised to [0, 1]
        norm_score_2:   Normalised to [0, 1]
        joint_utility:  score_1 + score_2
        max_joint:      Theoretical maximum joint utility
        efficiency:     joint_utility / max_joint (1.0 = Pareto optimal frontier)
        pareto_optimal: bool
    """
    values_1 = scenario["values_1"]
    values_2 = scenario["values_2"]
    counts   = scenario["counts"]

    s1 = _compute_utility(values_1, picks_1)
    s2 = _compute_utility(values_2, picks_2)
    mj = max_joint_utility(counts, values_1, values_2)
    po = is_pareto_optimal(counts, values_1, picks_1, values_2, picks_2)

    return {
        "score_1":        s1,
        "score_2":        s2,
        "norm_score_1":   round(s1 / 10, 4),
        "norm_score_2":   round(s2 / 10, 4),
        "joint_utility":  s1 + s2,
        "max_joint":      mj,
        "efficiency":     round((s1 + s2) / mj, 4) if mj > 0 else 0.0,
        "pareto_optimal": po,
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _build_dond_system(scenario: Dict, agent_values_key: str) -> str:
    """Build system prompt for a Deal or No Deal agent."""
    counts = scenario["counts"]
    values = scenario[agent_values_key]
    items_desc = ", ".join(
        f"{counts[i]} {ITEM_NAMES[i]} (worth {values[i]} pts each to you)"
        for i in range(3)
    )
    return (
        f"You are negotiating with another person to divide a pool of items.\n"
        f"\n"
        f"Items available: {items_desc}\n"
        f"Your maximum possible points: {scenario['max_score']}\n"
        f"\n"
        f"Your values are SECRET — do not reveal your exact point values.\n"
        f"The other person has DIFFERENT private values for the same items.\n"
        f"\n"
        f"Your goal: maximise YOUR points. Get items worth the most to you.\n"
        f"Strategy: figure out what the other person values less, then trade\n"
        f"items you care less about for items you care more about.\n"
        f"\n"
        f"Rules:\n"
        f"  - Negotiate in natural sentences. Propose splits, explain preferences.\n"
        f"  - Do NOT reveal your exact point values.\n"
        f"  - When ready to finalise, end your message with: DEAL=X,Y,Z\n"
        f"    where X,Y,Z are the number of books,hats,balls YOU take.\n"
        f"    The other person gets the remainder.\n"
        f"  - Example: DEAL=2,1,0 means you take 2 books, 1 hat, 0 balls.\n"
        f"  - Only write DEAL= when you genuinely accept that split.\n"
    )


# ---------------------------------------------------------------------------
# Deal Parsing (DonD-specific)
# ---------------------------------------------------------------------------

def is_deal_dond(text: str) -> bool:
    """Check if text contains a valid DonD deal: DEAL=X,Y,Z."""
    return bool(re.search(r"DEAL\s*=\s*\d+\s*,\s*\d+\s*,\s*\d+", text, re.IGNORECASE))


def parse_deal_allocation(text: str, counts: List[int]) -> Optional[List[int]]:
    """Parse DEAL=X,Y,Z and validate against item counts.

    Returns [books, hats, balls] the dealmaker claims, or None if invalid.
    """
    m = re.search(r"DEAL\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text, re.IGNORECASE)
    if not m:
        return None

    picks = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    for i in range(3):
        if picks[i] < 0 or picks[i] > counts[i]:
            log.warning("Invalid allocation: %s exceeds available %s", picks, counts)
            return None

    return picks


# ---------------------------------------------------------------------------
# Game Loop
# ---------------------------------------------------------------------------

def run_game_dond(
    model,
    tokenizer,
    scenario:       Dict,
    dvecs_agent1:   Optional[Dict[int, np.ndarray]],
    alpha_agent1:   float,
    dvecs_agent2:   Optional[Dict[int, np.ndarray]],
    alpha_agent2:   float,
    steered_role:   str,   # "agent1" or "agent2"
    max_new_tokens: int   = 150,
    temperature:    float = 0.7,
) -> Dict:
    """Run a Deal or No Deal negotiation game.

    Mirrors run_game() in apply_steering.py but for multi-issue bargaining.
    Agent 1 speaks first. Either agent can close with DEAL=X,Y,Z (what they
    take; the other gets the remainder).
    """
    from apply_steering import generate_turn

    agent1_system = _build_dond_system(scenario, "values_1")
    agent2_system = _build_dond_system(scenario, "values_2")

    counts    = scenario["counts"]
    transcript: List[Tuple[str, str]] = []
    deal_picks: Optional[List[int]]  = None
    dealmaker:  Optional[str]        = None

    for turn in range(MAX_TURNS):
        can_finalise = turn >= MIN_TURNS_BEFORE_DEAL

        # ---- agent1's turn ------------------------------------------------
        def _agent1_msgs() -> List[Dict]:
            msgs  = [{"role": "system", "content": agent1_system}]
            lines = [("YOU: " if spk == "agent1" else "THEM: ") + utt
                     for spk, utt in transcript]
            if not lines:
                instr = "You go first. Propose how to split the items."
            elif can_finalise:
                instr = ("Write 1-2 natural sentences, then optionally end "
                         "with DEAL=X,Y,Z if you are ready to close.")
            else:
                instr = (f"Write 1-2 natural sentences making your counter-proposal. "
                         f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
            content = ("\n".join(lines) + "\n\n" if lines else "") + f"Your turn. {instr}"
            msgs.append({"role": "user", "content": content})
            return msgs

        utt_1 = generate_turn(model, tokenizer, _agent1_msgs(),
                              dvecs_agent1, alpha_agent1,
                              max_new_tokens, temperature, can_finalise)
        transcript.append(("agent1", utt_1))
        log.info("[Turn %d] AGENT1 (α=%+.1f): %s", turn + 1, alpha_agent1, utt_1)

        if can_finalise and is_deal_dond(utt_1):
            deal_picks = parse_deal_allocation(utt_1, counts)
            if deal_picks is not None:
                dealmaker = "agent1"
                break

        # ---- agent2's turn ------------------------------------------------
        def _agent2_msgs() -> List[Dict]:
            msgs  = [{"role": "system", "content": agent2_system}]
            lines = [("YOU: " if spk == "agent2" else "THEM: ") + utt
                     for spk, utt in transcript]
            if can_finalise:
                instr = ("Write 1-2 natural sentences, then optionally end "
                         "with DEAL=X,Y,Z if you are ready to close.")
            else:
                instr = (f"Write 1-2 natural sentences making your counter-proposal. "
                         f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
            msgs.append({"role": "user",
                         "content": "\n".join(lines) + f"\n\nYour turn. {instr}"})
            return msgs

        utt_2 = generate_turn(model, tokenizer, _agent2_msgs(),
                              dvecs_agent2, alpha_agent2,
                              max_new_tokens, temperature, can_finalise)
        transcript.append(("agent2", utt_2))
        log.info("[Turn %d] AGENT2 (α=%+.1f): %s", turn + 1, alpha_agent2, utt_2)

        if can_finalise and is_deal_dond(utt_2):
            deal_picks = parse_deal_allocation(utt_2, counts)
            if deal_picks is not None:
                dealmaker = "agent2"
                break

    # ---- score the outcome ------------------------------------------------
    agreed = deal_picks is not None

    if agreed:
        # deal_picks = what the dealmaker claims for themselves
        if dealmaker == "agent1":
            picks_1 = deal_picks
            picks_2 = [c - p for c, p in zip(counts, deal_picks)]
        else:
            picks_2 = deal_picks
            picks_1 = [c - p for c, p in zip(counts, deal_picks)]

        scores = score_deal_dond(picks_1, picks_2, scenario)
        log.info("DEAL: agent1 takes %s (%d pts), agent2 takes %s (%d pts) | "
                 "Pareto=%s  efficiency=%.2f",
                 picks_1, scores["score_1"],
                 picks_2, scores["score_2"],
                 scores["pareto_optimal"], scores["efficiency"])
    else:
        picks_1 = [0, 0, 0]
        picks_2 = [0, 0, 0]
        scores = {
            "score_1": 0, "score_2": 0,
            "norm_score_1": 0.0, "norm_score_2": 0.0,
            "joint_utility": 0,
            "max_joint": max_joint_utility(counts, scenario["values_1"], scenario["values_2"]),
            "efficiency": 0.0, "pareto_optimal": False,
        }
        log.info("No deal after %d turns.", MAX_TURNS)

    steered_score  = scores["norm_score_1"] if steered_role == "agent1" else scores["norm_score_2"]
    baseline_score = scores["norm_score_2"] if steered_role == "agent1" else scores["norm_score_1"]

    return {
        "dataset":          "dealornodeal",
        "agreed":           agreed,
        "dealmaker":        dealmaker,
        "picks_1":          picks_1,
        "picks_2":          picks_2,
        "score_1":          scores["score_1"],
        "score_2":          scores["score_2"],
        "norm_score_1":     scores["norm_score_1"],
        "norm_score_2":     scores["norm_score_2"],
        "joint_utility":    scores["joint_utility"],
        "max_joint":        scores["max_joint"],
        "efficiency":       scores["efficiency"],
        "pareto_optimal":   scores["pareto_optimal"],
        "steered_role":     steered_role,
        "steered_score":    round(steered_score, 4),
        "baseline_score":   round(baseline_score, 4),
        "advantage":        round(steered_score - baseline_score, 4),
        "num_turns":        len(transcript),
        "transcript":       [{"speaker": s, "utterance": u} for s, u in transcript],
        "counts":           counts,
        "values_1":         scenario["values_1"],
        "values_2":         scenario["values_2"],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_dond(results: List[Dict], alpha: float) -> Dict:
    """Summarise Deal or No Deal results, parallel to summarise() in apply_steering."""
    n      = len(results)
    agreed = [r for r in results if r["agreed"]]
    na     = len(agreed)
    pareto = [r for r in agreed if r["pareto_optimal"]]

    return {
        "dataset":           "dealornodeal",
        "num_games":         n,
        "agree_rate":        round(na / n, 3)                                                if n  else 0,
        "steered_score":     round(sum(r["steered_score"]  for r in agreed) / na, 4)         if na else 0,
        "baseline_score":    round(sum(r["baseline_score"] for r in agreed) / na, 4)         if na else 0,
        "advantage":         round(sum(r["advantage"]      for r in agreed) / na, 4)         if na else 0,
        "avg_score_1":       round(sum(r["score_1"]        for r in agreed) / na, 2)         if na else 0,
        "avg_score_2":       round(sum(r["score_2"]        for r in agreed) / na, 2)         if na else 0,
        "avg_joint_utility": round(sum(r["joint_utility"]  for r in agreed) / na, 2)         if na else 0,
        "avg_efficiency":    round(sum(r["efficiency"]     for r in agreed) / na, 4)         if na else 0,
        "pareto_rate":       round(len(pareto) / na, 3)                                      if na else 0,
        "avg_turns":         round(sum(r["num_turns"]      for r in results) / n, 1)         if n  else 0,
        "alpha":             alpha,
        "note":              "steered agent alternates agent1/agent2 each game",
    }


# ---------------------------------------------------------------------------
# Standalone test — verifies data loading, parsing, and scoring without GPU
# ---------------------------------------------------------------------------

def _run_self_test():
    """Download data, compute stats, validate scoring on known examples."""
    print("=" * 70)
    print("Deal or No Deal — Integration Self-Test")
    print("=" * 70)

    # ---- 1. Load scenarios ------------------------------------------------
    print("\n[1] Loading selfplay scenarios...")
    all_scenarios = load_dealornodeal(split="selfplay", num_samples=99999)
    print(f"    Total scenarios: {len(all_scenarios)}")

    total_items = [s["total_items"] for s in all_scenarios]
    print(f"    Items per scenario: min={min(total_items)}, "
          f"max={max(total_items)}, mean={sum(total_items)/len(total_items):.1f}")

    # item count distribution
    from collections import Counter
    item_dist = Counter(tuple(s["counts"]) for s in all_scenarios)
    print(f"    Unique count configs: {len(item_dist)}")
    for config, cnt in item_dist.most_common(5):
        names = [f"{config[i]} {ITEM_NAMES[i]}" for i in range(3)]
        print(f"      {', '.join(names)}: {cnt} scenarios")

    # ---- 2. Load from dialogue file for comparison -------------------------
    print("\n[2] Loading train split (deduplicated)...")
    train_scenarios = load_dealornodeal(split="train", num_samples=99999)
    print(f"    Unique scenarios from train: {len(train_scenarios)}")

    # ---- 3. Scoring validation on a known example --------------------------
    print("\n[3] Scoring validation...")

    # Example from Lewis et al. 2017:
    # Agent 1: 3 books (val=1), 2 hats (val=3), 1 ball (val=1)
    #   -> Total: 3*1 + 2*3 + 1*1 = 10 ✓
    # Agent 2: 3 books (val=2), 2 hats (val=1), 1 ball (val=2)
    #   -> Total: 3*2 + 2*1 + 1*2 = 10 ✓
    # Deal: Agent 1 gets 2 books + 2 hats, Agent 2 gets 1 book + 1 ball
    #   Agent 1 score: 2*1 + 2*3 + 0*1 = 8
    #   Agent 2 score: 1*2 + 0*1 + 1*2 = 4
    test_scenario = {
        "counts":      [3, 2, 1],
        "values_1":    [1, 3, 1],
        "values_2":    [2, 1, 2],
        "item_names":  ITEM_NAMES,
        "total_items": 6,
        "max_score":   10,
    }
    test_picks_1 = [2, 2, 0]  # agent 1 takes 2 books, 2 hats, 0 balls
    test_picks_2 = [1, 0, 1]  # agent 2 takes 1 book, 0 hats, 1 ball

    result = score_deal_dond(test_picks_1, test_picks_2, test_scenario)
    assert result["score_1"] == 8, f"Expected 8, got {result['score_1']}"
    assert result["score_2"] == 4, f"Expected 4, got {result['score_2']}"
    print(f"    Paper example: Agent1={result['score_1']}pts, Agent2={result['score_2']}pts ✓")
    print(f"    Joint utility: {result['joint_utility']} / {result['max_joint']}")
    print(f"    Efficiency: {result['efficiency']}")
    print(f"    Pareto optimal: {result['pareto_optimal']}")

    # (8, 4) IS Pareto optimal: no alternative makes one agent strictly better
    # without making the other worse. Higher joint utility ≠ domination.
    # All allocations with Agent1 ≥ 8 give Agent2 ≤ 4.
    assert result["pareto_optimal"], "Deal (8,4) should be Pareto optimal"
    print("    Correctly identified as Pareto optimal ✓")

    # Test a NON-Pareto deal: Agent1 takes [1,0,0]=1pt, Agent2 takes [2,2,1]=8pts
    # Dominated by [0,2,0]/[3,0,1] = (6,8): Agent1 improves 1→6, Agent2 stays ≥8.
    bad_picks_1 = [1, 0, 0]
    bad_picks_2 = [2, 2, 1]
    bad_result = score_deal_dond(bad_picks_1, bad_picks_2, test_scenario)
    assert bad_result["score_1"] == 1 and bad_result["score_2"] == 8
    assert not bad_result["pareto_optimal"], "Deal (1,8) should NOT be Pareto optimal"
    print(f"    Non-Pareto deal: Agent1={bad_result['score_1']}pts, "
          f"Agent2={bad_result['score_2']}pts → Pareto={bad_result['pareto_optimal']} ✓")

    # Optimal deal: give each agent their most-valued items
    # Agent 1 values hats most (3 pts each). Agent 2 values books (2) and balls (2).
    opt_picks_1 = [0, 2, 0]
    opt_picks_2 = [3, 0, 1]
    opt_result = score_deal_dond(opt_picks_1, opt_picks_2, test_scenario)
    assert opt_result["pareto_optimal"]
    print(f"\n    Optimal split: Agent1={opt_result['score_1']}pts, "
          f"Agent2={opt_result['score_2']}pts")
    print(f"    Joint utility: {opt_result['joint_utility']} / {opt_result['max_joint']}")
    print(f"    Efficiency: {opt_result['efficiency']}")
    print(f"    Pareto optimal: {opt_result['pareto_optimal']} ✓")

    # ---- 4. Parse function validation --------------------------------------
    print("\n[4] Deal parsing validation...")

    tests = [
        ("DEAL=2,1,0",      [3, 2, 1], [2, 1, 0]),
        ("DEAL = 2, 1, 0",  [3, 2, 1], [2, 1, 0]),
        ("deal=0,0,0",       [3, 2, 1], [0, 0, 0]),
        ("Let's go DEAL=3,2,1 ok?", [3, 2, 1], [3, 2, 1]),
        ("DEAL=5,0,0",      [3, 2, 1], None),  # exceeds count
        ("DEAL=200",         [3, 2, 1], None),  # wrong format
        ("no deal here",     [3, 2, 1], None),  # no deal
    ]
    for text, counts, expected in tests:
        result_parse = parse_deal_allocation(text, counts)
        status = "✓" if result_parse == expected else f"✗ (got {result_parse})"
        print(f"    parse('{text}') -> {result_parse} {status}")

    is_deal_tests = [
        ("DEAL=2,1,0", True),
        ("DEAL = 2, 1, 0", True),
        ("DEAL=200", False),
        ("no deal", False),
    ]
    for text, expected in is_deal_tests:
        got = is_deal_dond(text)
        status = "✓" if got == expected else "✗"
        print(f"    is_deal_dond('{text}') -> {got} {status}")

    # ---- 5. Pareto rate across random scenarios ----------------------------
    print("\n[5] Pareto analysis on random scenarios...")
    sample = random.sample(all_scenarios, min(200, len(all_scenarios)))

    pareto_counts = {"total": 0, "pareto": 0}
    efficiency_sum = 0.0
    for sc in sample:
        # Simulate a "fair split" — each agent takes half of each item (rounded)
        fair_1 = [c // 2 for c in sc["counts"]]
        fair_2 = [c - p for c, p in zip(sc["counts"], fair_1)]
        res = score_deal_dond(fair_1, fair_2, sc)
        pareto_counts["total"] += 1
        if res["pareto_optimal"]:
            pareto_counts["pareto"] += 1
        efficiency_sum += res["efficiency"]

    pareto_pct = pareto_counts["pareto"] / pareto_counts["total"] * 100
    avg_eff = efficiency_sum / pareto_counts["total"]
    print(f"    Naive fair-split baseline ({pareto_counts['total']} scenarios):")
    print(f"      Pareto optimal: {pareto_pct:.1f}%")
    print(f"      Avg efficiency: {avg_eff:.3f}")
    print(f"    (Human baseline from paper: 76.9% Pareto optimal)")

    # ---- 6. Max joint utility distribution ---------------------------------
    print("\n[6] Max joint utility distribution...")
    max_joints = [max_joint_utility(s["counts"], s["values_1"], s["values_2"])
                  for s in sample[:100]]
    mj_counter = Counter(max_joints)
    for mj_val in sorted(mj_counter.keys()):
        print(f"      max_joint={mj_val}: {mj_counter[mj_val]} scenarios")

    print("\n" + "=" * 70)
    print("Self-test complete. Module is ready for integration.")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()
