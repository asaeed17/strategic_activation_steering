#!/usr/bin/env python3
"""
apply_steering.py

Quick experiment: does steering a language model toward "firmness"
actually make it a better price negotiator?

I'm using the CraigslistBargains dataset (He et al., 2018) as the arena.
The steered agent alternates roles each game so we don't accidentally
measure a role-specific advantage:
  - even games  → steered = SELLER, baseline = BUYER
  - odd games   → steered = BUYER,  baseline = SELLER

Scoring (only for games that reach a deal):
  seller_score = (agreed - buyer_target)  / (seller_target - buyer_target)
  buyer_score  = (seller_target - agreed) / (seller_target - buyer_target)

Both scores live in [0,1] and sum to 1. 0.5 means the agreed price
landed exactly at the midpoint between the two targets.
steered_score  = whichever of the two scores belongs to the steered agent.
advantage      = mean(steered_score) - mean(baseline_score)

Example run:
  python apply_steering.py \
      --model qwen2.5-7b \
      --dimension firmness \
      --alpha 20 \
      --layers 12 16 20 \
      --use_craigslist \
      --num_samples 50 \
      --output_file results.json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# tweak these if you want negotiations to feel more/less drawn out
MIN_TURNS_BEFORE_DEAL = 3
MAX_TURNS             = 10

# Matches a bare REJECT on its own line — avoids false positives like
# "I reject this offer, but..." which is natural speech.
_REJECT_RE = re.compile(r'^\s*REJECT\s*$', re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

LOCAL_DATA_DIR = Path(__file__).resolve().parent / "craigslist_data"
LOCAL_FILES = {
    "train":      LOCAL_DATA_DIR / "train.json",
    "validation": LOCAL_DATA_DIR / "validation.json",
}


def load_craigslist(split: str = "train", num_samples: int = 50, min_span: int = 100) -> List[Dict]:
    if split not in LOCAL_FILES:
        raise ValueError(f"Split '{split}' not available. Choose from: {list(LOCAL_FILES.keys())}")

    path = LOCAL_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {path}\n"
            f"Expected craigslist_data/{{train,validation}}.json in the project root."
        )
    log.info("Loading dataset from %s ...", path)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    log.info("Loaded %d raw dialogues from '%s' split.", len(raw), split)

    scenarios = []
    n_filtered_span = 0
    for entry in raw:
        try:
            kbs = entry.get("scenario", {}).get("kbs", [])
            if len(kbs) < 2:
                continue

            p0 = kbs[0].get("personal", {})
            p1 = kbs[1].get("personal", {})

            # figure out which kb belongs to the seller
            if "seller" in str(p0.get("Role", "")).lower():
                seller_p, seller_kb = p0, kbs[0]
                buyer_p             = p1
            elif "seller" in str(p1.get("Role", "")).lower():
                seller_p, seller_kb = p1, kbs[1]
                buyer_p             = p0
            else:
                continue

            seller_target_raw = seller_p.get("Target")
            buyer_target_raw  = buyer_p.get("Target")
            if seller_target_raw is None or buyer_target_raw is None:
                continue

            seller_target = float(seller_target_raw)
            buyer_target  = float(buyer_target_raw)

            item = seller_kb.get("item", {})

            # some fields come back as single-element lists for some reason
            def _unwrap(v):
                if isinstance(v, list):
                    return v[0] if v else ""
                return v

            listing_price = float(_unwrap(item.get("Price", -1)) or -1)
            title         = str(_unwrap(item.get("Title",       ""))).strip()
            description   = str(_unwrap(item.get("Description", ""))).strip()
            category      = str(_unwrap(item.get("Category",    ""))).strip()

            # skip anything with obviously bad data
            if listing_price <= 0 or seller_target <= 0 or buyer_target <= 0:
                continue
            if not title:
                continue

            span = seller_target - buyer_target
            if span < min_span:
                n_filtered_span += 1
                continue

            scenarios.append({
                "title":         title,
                "description":   description,
                "category":      category,
                "listing_price": listing_price,
                "seller_target": seller_target,
                "buyer_target":  buyer_target,
            })

        except (KeyError, ValueError, TypeError):
            continue

    log.info("Span filter (<%d): removed %d scenarios.", min_span, n_filtered_span)
    log.info("Found %d valid scenarios after filtering.", len(scenarios))
    if not scenarios:
        raise RuntimeError("No valid scenarios found after filtering — check the local data files.")

    k = min(num_samples, len(scenarios))
    return random.sample(scenarios, k)


# ---------------------------------------------------------------------------
# Steering Hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """Adds alpha * direction to a transformer layer's output activations."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha     = alpha
        self._handle   = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            d = self.direction.to(device=h.device, dtype=h.dtype)
            return (h + self.alpha * d,) + output[1:]
        d = self.direction.to(device=output.device, dtype=output.dtype)
        return output + self.alpha * d

    def register(self, layer_module) -> None:
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        if self._handle:
            self._handle.remove()
            self._handle = None


def get_transformer_layers(model):
    return model.model.layers


# ---------------------------------------------------------------------------
# Direction Vectors
# ---------------------------------------------------------------------------

def load_direction_vectors(
    vectors_dir: Path,
    model_alias: str,
    dimension: str,
    method: str,
    layer_indices: List[int],
) -> Dict[int, np.ndarray]:
    vec_dir = vectors_dir / model_alias / method
    vectors = {}
    for l in layer_indices:
        path = vec_dir / f"{dimension}_layer{l:02d}.npy"
        if not path.exists():
            # maybe they were all saved in a single stacked array
            all_path = vec_dir / f"{dimension}_all_layers.npy"
            if all_path.exists():
                all_vecs = np.load(all_path)
                if l < len(all_vecs):
                    vectors[l] = all_vecs[l]
                    continue
            log.warning("Vector not found: %s (skipping layer %d)", path, l)
            continue
        vectors[l] = np.load(path)
    if not vectors:
        raise FileNotFoundError(
            f"No vectors found for dimension='{dimension}' method='{method}' "
            f"model='{model_alias}' layers={layer_indices}. Run extract_vectors.py first."
        )
    return vectors


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_seller_system(scenario: Dict) -> str:
    return (
        f"You are a seller on Craigslist negotiating the price of an item.\n"
        f"\n"
        f"Item details:\n"
        f"  Title      : {scenario['title']}\n"
        f"  Category   : {scenario['category']}\n"
        f"  Listed at  : ${scenario['listing_price']:.0f}\n"
        f"  Description: {scenario['description'][:300]}\n"
        f"\n"
        f"Your private minimum: you will NOT accept less than ${scenario['seller_target']:.0f}. "
        f"Never reveal this number.\n"
        f"\n"
        f"Strategy:\n"
        f"  - Start at or near your listing price of ${scenario['listing_price']:.0f} and concede slowly.\n"
        f"  - Make arguments for why the item is worth the price (condition, features, demand, alternatives).\n"
        f"  - Never agree to a price below ${scenario['seller_target']:.0f} — walk away if necessary.\n"
        f"\n"
        f"Rules:\n"
        f"  - Always write in natural conversational sentences, like a real Craigslist seller.\n"
        f"  - Do NOT just say a number — always support your position with a reason.\n"
        f"  - When ready to agree on a price, end your message with: DEAL=<price> (e.g. DEAL=450)\n"
        f"  - Only write DEAL= when you genuinely accept that price "
        f"(it must be at or above ${scenario['seller_target']:.0f}).\n"
        f"  - If the offer is below your minimum and the buyer won't move, write REJECT on its own line.\n"
        f"    Use REJECT only when you would truly walk away — not as a bluff. Available from turn 3.\n"
    )


def build_buyer_system(scenario: Dict) -> str:
    return (
        f"You are a buyer on Craigslist negotiating the price of an item.\n"
        f"\n"
        f"Item details:\n"
        f"  Title      : {scenario['title']}\n"
        f"  Category   : {scenario['category']}\n"
        f"  Listed at  : ${scenario['listing_price']:.0f}\n"
        f"  Description: {scenario['description'][:300]}\n"
        f"\n"
        f"Your private maximum: you will NOT pay more than ${scenario['buyer_target']:.0f}. "
        f"Never reveal this number.\n"
        f"\n"
        f"Strategy:\n"
        f"  - Start well below the listing price and concede slowly.\n"
        f"  - Make arguments for why the price should be lower (wear, comparable listings, budget).\n"
        f"  - Never agree to a price above ${scenario['buyer_target']:.0f} — walk away if necessary.\n"
        f"\n"
        f"Rules:\n"
        f"  - Always write in natural conversational sentences, like a real Craigslist buyer.\n"
        f"  - Do NOT just say a number — always support your position with a reason.\n"
        f"  - When ready to agree on a price, end your message with: DEAL=<price> (e.g. DEAL=350)\n"
        f"  - Only write DEAL= when you genuinely accept that price "
        f"(it must be at or below ${scenario['buyer_target']:.0f}).\n"
        f"  - If the price is above your maximum and the seller won't move, write REJECT on its own line.\n"
        f"    Use REJECT only when you would truly walk away — not as a bluff. Available from turn 3.\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def is_deal(text: str) -> bool:
    return bool(re.search(r"DEAL\s*=\s*\$?\d+", text.upper()))


def parse_deal_price(text: str) -> Optional[float]:
    m = re.search(r"DEAL\s*=\s*\$?([\d,]+(?:\.\d+)?)", text.upper())
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def is_reject(text: str) -> bool:
    return bool(_REJECT_RE.search(text))


def generate_turn(
    model,
    tokenizer,
    messages:          List[Dict],
    direction_vectors: Optional[Dict[int, np.ndarray]],
    alpha:             float,
    max_new_tokens:    int   = 120,
    temperature:       float = 0.7,
    can_finalise:      bool  = True,
) -> str:
    device    = next(model.parameters()).device
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    # attach steering hooks if we have vectors for this agent
    hooks: List[SteeringHook] = []
    if direction_vectors and alpha != 0.0:
        layers = get_transformer_layers(model)
        for layer_idx, vec in direction_vectors.items():
            if layer_idx >= len(layers):
                continue
            dt = torch.tensor(vec, dtype=torch.float32, device=device)
            h  = SteeringHook(direction=dt, alpha=alpha)
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
        # always clean up hooks even if generation crashes
        for h in hooks:
            h.remove()

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # some models parrot back the YOU:/THEM: prefixes from the transcript
    text = re.sub(r'^(YOU|THEM)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    # if we're not at the deal-allowed turn yet, strip any premature DEAL= or REJECT
    if not can_finalise:
        text = re.sub(r'DEAL\s*=\s*\$?[\d,]+', '', text, flags=re.IGNORECASE).strip()
        text = _REJECT_RE.sub('', text).strip()

    # REJECT takes priority over DEAL — check line by line
    for line in text.splitlines():
        if is_reject(line):
            return "REJECT"
        if is_deal(line):
            return line.strip()

    # otherwise, return the first non-empty line (keep it concise)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_deal(
    agreed_price:  float,
    seller_target: float,
    buyer_target:  float,
) -> Dict:
    """
    How well did each side do relative to their private target?

    The span between targets is the "prize" being divided:
      seller_score = (agreed - buyer_target)  / span   → 1.0 if seller got their target
      buyer_score  = (seller_target - agreed) / span   → 1.0 if buyer got their target

    Returns a dict with both raw (unclamped) and clamped [0,1] scores,
    plus a flag indicating whether clamping was applied. Raw scores can
    fall outside [0,1] when the agreed price is beyond the target range
    (e.g. buyer pays less than their own target).
    """
    span = seller_target - buyer_target
    if span <= 0:
        return {
            "seller_score": 0.5, "buyer_score": 0.5,
            "raw_seller_score": 0.5, "raw_buyer_score": 0.5,
            "clamped": False, "span": span,
            "midpoint": round((seller_target + buyer_target) / 2.0, 2),
            "midpoint_deviation": 0.0,
        }
    raw_seller = (agreed_price - buyer_target)  / span
    raw_buyer  = (seller_target - agreed_price) / span
    seller_score = max(0.0, min(1.0, raw_seller))
    buyer_score  = max(0.0, min(1.0, raw_buyer))
    clamped = (seller_score != raw_seller) or (buyer_score != raw_buyer)
    midpoint = (seller_target + buyer_target) / 2.0
    midpoint_deviation = round((agreed_price - midpoint) / span, 4)
    return {
        "seller_score":       round(seller_score, 4),
        "buyer_score":        round(buyer_score, 4),
        "raw_seller_score":   round(raw_seller, 4),
        "raw_buyer_score":    round(raw_buyer, 4),
        "clamped":            clamped,
        "span":               round(span, 2),
        "midpoint":           round(midpoint, 2),
        "midpoint_deviation": midpoint_deviation,
    }


# ---------------------------------------------------------------------------
# Game Loop
# ---------------------------------------------------------------------------

def run_game(
    model,
    tokenizer,
    scenario:       Dict,
    dvecs_seller:   Optional[Dict[int, np.ndarray]],
    alpha_seller:   float,
    dvecs_buyer:    Optional[Dict[int, np.ndarray]],
    alpha_buyer:    float,
    steered_role:   str,   # "seller" or "buyer"
    max_new_tokens: int   = 120,
    temperature:    float = 0.7,
    opening_bid_pct: float = 0.6,
) -> Dict:
    seller_system = build_seller_system(scenario)
    buyer_system  = build_buyer_system(scenario)

    transcript:    List[Tuple[str, str]] = []
    agreed_price:  Optional[float]      = None
    dealmaker:     Optional[str]        = None
    walk_away:     bool                 = False
    walk_away_by:  Optional[str]        = None

    # buyer kicks things off with a lowball (default 60% of listing price)
    opening_bid = round(scenario["listing_price"] * opening_bid_pct)
    transcript.append(("buyer", f"Hi, I'm interested in this. Would you take ${opening_bid:.0f}?"))

    for turn in range(MAX_TURNS):
        can_finalise = turn >= MIN_TURNS_BEFORE_DEAL

        # ---- seller's turn --------------------------------------------------
        def _seller_msgs() -> List[Dict]:
            msgs  = [{"role": "system", "content": seller_system}]
            lines = [("YOU: " if spk == "seller" else "THEM: ") + utt
                     for spk, utt in transcript]
            if can_finalise:
                instr = ("Write 1-2 natural sentences, then optionally end with DEAL=<price> "
                         "if you are ready to close.")
            else:
                instr = (f"Write 1-2 natural sentences making your counter-offer. "
                         f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
            msgs.append({"role": "user",
                         "content": "\n".join(lines) + f"\n\nYour turn. {instr}"})
            return msgs

        utt_s = generate_turn(model, tokenizer, _seller_msgs(),
                               dvecs_seller, alpha_seller,
                               max_new_tokens, temperature, can_finalise)
        transcript.append(("seller", utt_s))
        log.info("[Turn %d] SELLER (α=%+.0f): %s", turn + 1, alpha_seller, utt_s)

        if can_finalise and is_reject(utt_s):
            walk_away, walk_away_by = True, "seller"
            log.info("WALK-AWAY by seller at turn %d.", turn + 1)
            break
        if can_finalise and is_deal(utt_s):
            agreed_price = parse_deal_price(utt_s)
            dealmaker    = "seller"
            break

        # ---- buyer's turn ---------------------------------------------------
        def _buyer_msgs() -> List[Dict]:
            msgs  = [{"role": "system", "content": buyer_system}]
            lines = [("YOU: " if spk == "buyer" else "THEM: ") + utt
                     for spk, utt in transcript]
            if can_finalise:
                instr = ("Write 1-2 natural sentences, then optionally end with DEAL=<price> "
                         "if you are ready to close.")
            else:
                instr = (f"Write 1-2 natural sentences making your counter-offer. "
                         f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
            msgs.append({"role": "user",
                         "content": "\n".join(lines) + f"\n\nYour turn. {instr}"})
            return msgs

        utt_b = generate_turn(model, tokenizer, _buyer_msgs(),
                               dvecs_buyer, alpha_buyer,
                               max_new_tokens, temperature, can_finalise)
        transcript.append(("buyer", utt_b))
        log.info("[Turn %d] BUYER  (α=%+.0f): %s", turn + 1, alpha_buyer, utt_b)

        if can_finalise and is_reject(utt_b):
            walk_away, walk_away_by = True, "buyer"
            log.info("WALK-AWAY by buyer at turn %d.", turn + 1)
            break
        if can_finalise and is_deal(utt_b):
            agreed_price = parse_deal_price(utt_b)
            dealmaker    = "buyer"
            break

    # ---- score the outcome --------------------------------------------------
    agreed = agreed_price is not None and not walk_away

    if agreed:
        scores = score_deal(
            agreed_price,
            scenario["seller_target"],
            scenario["buyer_target"],
        )
        seller_score = scores["seller_score"]
        buyer_score  = scores["buyer_score"]
        log.info("DEAL at $%.0f  |  seller=%.3f  buyer=%.3f  raw=(%.3f, %.3f)%s",
                 agreed_price, seller_score, buyer_score,
                 scores["raw_seller_score"], scores["raw_buyer_score"],
                 "  [CLAMPED]" if scores["clamped"] else "")
    else:
        scores = {
            "seller_score": 0.0, "buyer_score": 0.0,
            "raw_seller_score": 0.0, "raw_buyer_score": 0.0,
            "clamped": False, "span": scenario["seller_target"] - scenario["buyer_target"],
            "midpoint": round((scenario["seller_target"] + scenario["buyer_target"]) / 2.0, 2),
            "midpoint_deviation": 0.0,
        }
        seller_score = 0.0
        buyer_score  = 0.0
        if walk_away:
            log.info("Walk-away by %s — no deal.", walk_away_by)
        else:
            log.info("No deal after %d turns.", MAX_TURNS)

    steered_score  = seller_score if steered_role == "seller" else buyer_score
    baseline_score = buyer_score  if steered_role == "seller" else seller_score

    md             = scores.get("midpoint_deviation", 0.0)
    steered_md_adv = round(md * (1 if steered_role == "seller" else -1), 4)

    return {
        "agreed":                       agreed,
        "agreed_price":                 agreed_price,
        "dealmaker":                    dealmaker,
        "walk_away":                    walk_away,
        "walk_away_by":                 walk_away_by,
        "seller_score":                 seller_score,
        "buyer_score":                  buyer_score,
        "raw_seller_score":             scores["raw_seller_score"],
        "raw_buyer_score":              scores["raw_buyer_score"],
        "clamped":                      scores["clamped"],
        "span":                         scores["span"],
        "midpoint":                     scores.get("midpoint", 0.0),
        "midpoint_deviation":           md if agreed else 0.0,
        "steered_midpoint_advantage":   steered_md_adv if agreed else 0.0,
        "steered_role":                 steered_role,
        "steered_score":                round(steered_score,  4),
        "baseline_score":               round(baseline_score, 4),
        "advantage":                    round(steered_score - baseline_score, 4),
        "num_turns":                    len(transcript),
        "transcript":                   [{"speaker": s, "utterance": u} for s, u in transcript],
        "listing_price":                scenario["listing_price"],
        "seller_target":                scenario["seller_target"],
        "buyer_target":                 scenario["buyer_target"],
        "title":                        scenario["title"],
        "category":                     scenario["category"],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: List[Dict], alpha: float) -> Dict:
    n          = len(results)
    agreed     = [r for r in results if r["agreed"]]
    na         = len(agreed)
    walk_aways = [r for r in results if r.get("walk_away")]
    nw         = len(walk_aways)
    walk_by_steered  = sum(1 for r in walk_aways
                           if r.get("walk_away_by") == r.get("steered_role"))
    walk_by_baseline = nw - walk_by_steered

    def _role_summary(role: str) -> Dict:
        rs  = [r for r in results if r.get("steered_role") == role]
        ra  = [r for r in rs if r["agreed"]]
        nr, nra = len(rs), len(ra)
        nrw = sum(1 for r in rs if r.get("walk_away"))
        return {
            "n":                  nr,
            "agree_rate":         round(nra / nr, 3) if nr else 0,
            "advantage":          round(sum(r["advantage"] for r in ra) / nra, 4) if nra else 0,
            "midpoint_advantage": round(sum(r.get("steered_midpoint_advantage", 0) for r in ra) / nra, 4) if nra else 0,
            "clamped_pct":        round(sum(1 for r in ra if r.get("clamped")) / nra, 3) if nra else 0,
            "walk_away_rate":     round(nrw / nr, 3) if nr else 0,
        }

    return {
        # --- existing keys kept for backward compat ---
        "num_games":      n,
        "agree_rate":     round(na / n, 3)                                             if n  else 0,
        "steered_score":  round(sum(r["steered_score"]  for r in agreed) / na, 4)      if na else 0,
        "baseline_score": round(sum(r["baseline_score"] for r in agreed) / na, 4)      if na else 0,
        "advantage":      round(sum(r["advantage"]      for r in agreed) / na, 4)      if na else 0,
        "seller_score":   round(sum(r["seller_score"]   for r in agreed) / na, 4)      if na else 0,
        "buyer_score":    round(sum(r["buyer_score"]    for r in agreed) / na, 4)      if na else 0,
        "avg_price":      round(sum(r["agreed_price"]   for r in agreed) / na, 2)      if na else 0,
        "avg_turns":      round(sum(r["num_turns"]      for r in results) / n,  1)     if n  else 0,
        "alpha":          alpha,
        # --- new keys ---
        "walk_away_rate":               round(nw / n, 3) if n else 0,
        "walk_away_by_steered":         walk_by_steered,
        "walk_away_by_baseline":        walk_by_baseline,
        "midpoint_deviation_mean":      round(sum(r.get("midpoint_deviation", 0) for r in agreed) / na, 4) if na else 0,
        "steered_midpoint_advantage_mean": round(sum(r.get("steered_midpoint_advantage", 0) for r in agreed) / na, 4) if na else 0,
        "by_role": {
            "buyer":  _role_summary("buyer"),
            "seller": _role_summary("seller"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Price negotiation eval: CraigslistBargains + steering vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",          choices=list(MODELS.keys()), default="qwen2.5-7b")
    p.add_argument("--vectors_dir",    default="vectors")
    p.add_argument("--dimension",      default="firmness")
    p.add_argument("--method",         choices=["mean_diff", "pca"], default="mean_diff")
    p.add_argument("--layers",         nargs="+", type=int, default=[12, 16, 20])
    p.add_argument("--alpha",          type=float, default=20.0)
    p.add_argument("--dataset_split",  choices=["train", "validation"], default="train")
    p.add_argument("--num_samples",    type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--dtype",          choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--output_file",    default="results/results.json")
    p.add_argument("--use_craigslist", action="store_true")
    p.add_argument("--min_span",       type=int, default=100,
                   help="Min seller_target - buyer_target (default 100). Small spans produce random scores.")
    p.add_argument("--steered_role",   choices=["buyer", "seller", "alternate"], default="alternate",
                   help="Fix steered agent role for all games, or alternate each game (default).")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    cfg   = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    log.info("Loading direction vectors for '%s' ...", args.dimension)
    dvecs = load_direction_vectors(
        vectors_dir=Path(args.vectors_dir),
        model_alias=cfg.alias,
        dimension=args.dimension,
        method=args.method,
        layer_indices=args.layers,
    )

    log.info("Loading model: %s", cfg.hf_id)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, token=token,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()

    if not args.use_craigslist:
        log.error("--use_craigslist is required. Exiting.")
        return

    scenarios = load_craigslist(split=args.dataset_split, num_samples=args.num_samples, min_span=args.min_span)

    all_results = []
    print("\n" + "=" * 70)
    for i, sc in enumerate(scenarios):
        if args.steered_role == "alternate":
            steered_role = "seller" if i % 2 == 0 else "buyer"
        else:
            steered_role = args.steered_role
        dvecs_seller = dvecs if steered_role == "seller" else None
        alpha_seller = args.alpha if steered_role == "seller" else 0.0
        dvecs_buyer  = dvecs if steered_role == "buyer"  else None
        alpha_buyer  = args.alpha if steered_role == "buyer"  else 0.0

        print(f"\nGame [{i + 1}/{len(scenarios)}]  [{sc['category']}]  "
              f"steered={steered_role.upper()}")
        print(f"  Item    : {sc['title']}")
        print(f"  Listed  : ${sc['listing_price']:.0f}  |  "
              f"Seller wants ≥${sc['seller_target']:.0f}  |  "
              f"Buyer wants ≤${sc['buyer_target']:.0f}")
        midpoint = (sc['seller_target'] + sc['buyer_target']) / 2
        print(f"  Midpoint: ${midpoint:.0f}  "
              f"({'gap' if sc['seller_target'] > sc['buyer_target'] else 'overlap'})")

        result = run_game(
            model=model, tokenizer=tokenizer,
            scenario=sc,
            dvecs_seller=dvecs_seller, alpha_seller=alpha_seller,
            dvecs_buyer=dvecs_buyer,   alpha_buyer=alpha_buyer,
            steered_role=steered_role,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        result["game_id"]   = i
        result["alpha"]     = args.alpha
        result["dimension"] = args.dimension

        if result["walk_away"]:
            status    = f"WALK-AWAY ({result['walk_away_by']})"
            price_str = "n/a"
        elif result["agreed"]:
            status    = "DEAL"
            price_str = f"${result['agreed_price']:.0f}"
        else:
            status    = "NO DEAL"
            price_str = "n/a"
        print(f"  {status} @ {price_str}  |  "
              f"steered={result['steered_score']:.3f}  "
              f"baseline={result['baseline_score']:.3f}  "
              f"advantage={result['advantage']:+.3f}  "
              f"midpt_adv={result.get('steered_midpoint_advantage', 0):+.3f}  "
              f"turns={result['num_turns']}")
        print("-" * 70)
        all_results.append(result)

    summary = summarise(all_results, args.alpha)
    br = summary["by_role"]
    b  = br["buyer"]
    s  = br["seller"]
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print(f"  Agreement rate   : {summary['agree_rate'] * 100:.1f}%     "
          f"Walk-away rate: {summary['walk_away_rate'] * 100:.1f}%  "
          f"(steered={summary['walk_away_by_steered']}, baseline={summary['walk_away_by_baseline']})")
    print(f"  Avg turns        : {summary['avg_turns']}")
    print()
    print(f"  ROLE-SEPARATED (primary):")
    print(f"    Steered=BUYER   N={b['n']:2d}  agree={b['agree_rate']*100:.0f}%  "
          f"adv={b['advantage']:+.3f}  midpt={b['midpoint_advantage']:+.3f}  "
          f"clamped={b['clamped_pct']*100:.0f}%  walkaway={b['walk_away_rate']*100:.0f}%")
    print(f"    Steered=SELLER  N={s['n']:2d}  agree={s['agree_rate']*100:.0f}%  "
          f"adv={s['advantage']:+.3f}  midpt={s['midpoint_advantage']:+.3f}  "
          f"clamped={s['clamped_pct']*100:.0f}%  walkaway={s['walk_away_rate']*100:.0f}%")
    print()
    print(f"  OVERALL (secondary — do not report alone):")
    print(f"    Steered: {summary['steered_score']:.3f}  "
          f"Baseline: {summary['baseline_score']:.3f}  "
          f"Advantage: {summary['advantage']:+.3f}  "
          f"MidptDev: {summary['midpoint_deviation_mean']:+.3f}")
    print("=" * 70)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "games": all_results}, f, indent=2, ensure_ascii=False)
    log.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()