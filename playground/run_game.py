#!/usr/bin/env python3
"""
playground/run_game.py — Task design validation experiments.

Runs Craigslist negotiation games with pluggable agents (local HF model
or API-based frontier model) and optional prompt enhancements.
No torch dependency for API-only mode.

Experiments:
  A. Prompt steering: does telling the model "use anchoring" improve scores?
  B. Frontier ceiling: does GPT-4o beat Qwen 3B? If not, task is broken.

Usage:
  # Exp A: prompt-enhanced buyer vs baseline seller (no GPU, API only)
  python playground/run_game.py --buyer api:gemini --seller api:gemini \
      --buyer_enhance anchoring --num_games 5

  # Exp B: frontier buyer vs local seller
  python playground/run_game.py --buyer api:gpt4o --seller api:gemini \
      --num_games 5

  # Both sides local (needs GPU)
  python playground/run_game.py --buyer local:qwen2.5-3b --seller local:qwen2.5-3b \
      --num_games 3

  # Quick single game with full transcript
  python playground/run_game.py --buyer api:gemini --seller api:gemini \
      --buyer_enhance strategic_concession --num_games 1 --verbose

Agent specs:
  local:<model_key>     Local HF model (needs GPU). e.g. local:qwen2.5-3b
  api:gemini            Google Gemini (GOOGLE_API_KEY)
  api:gpt4o             OpenAI GPT-4o (OPENAI_API_KEY)
  api:claude            Anthropic Claude (ANTHROPIC_API_KEY)
  api:groq              Groq LLaMA 3.3 70B (GROQ_API_KEY)

Prompt enhancements (--buyer_enhance / --seller_enhance):
  anchoring             Aggressive first-offer anchoring
  strategic_concession  Small concessions tied to reasons
  batna                 Reference alternatives / walkaway power
  firmness              Hold position, don't cave under pressure
  empathy               Build rapport, find mutual value
  combo                 All of the above combined
"""

import json
import re
import sys
import random
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrored from apply_steering.py to avoid torch import)
# ---------------------------------------------------------------------------

MIN_TURNS_BEFORE_DEAL = 3
MAX_TURNS = 10
_REJECT_RE = re.compile(r'^\s*REJECT\s*$', re.IGNORECASE | re.MULTILINE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "craigslist_data"


# ---------------------------------------------------------------------------
# Pure-Python functions inlined from apply_steering.py (no torch needed)
# ---------------------------------------------------------------------------

def load_craigslist(split: str = "train", num_samples: int = 50, min_span: int = 100) -> List[Dict]:
    local_files = {
        "train":      LOCAL_DATA_DIR / "train.json",
        "validation": LOCAL_DATA_DIR / "validation.json",
    }
    if split not in local_files:
        raise ValueError(f"Split '{split}' not available. Choose from: {list(local_files.keys())}")

    path = local_files[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {path}\n"
            f"Expected craigslist_data/{{train,validation}}.json in the project root.")
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
            if "seller" in str(p0.get("Role", "")).lower():
                seller_p, seller_kb = p0, kbs[0]
                buyer_p = p1
            elif "seller" in str(p1.get("Role", "")).lower():
                seller_p, seller_kb = p1, kbs[1]
                buyer_p = p0
            else:
                continue
            seller_target_raw = seller_p.get("Target")
            buyer_target_raw = buyer_p.get("Target")
            if seller_target_raw is None or buyer_target_raw is None:
                continue
            seller_target = float(seller_target_raw)
            buyer_target = float(buyer_target_raw)
            item = seller_kb.get("item", {})

            def _unwrap(v):
                return v[0] if isinstance(v, list) and v else v

            listing_price = float(_unwrap(item.get("Price", -1)) or -1)
            title = str(_unwrap(item.get("Title", ""))).strip()
            description = str(_unwrap(item.get("Description", ""))).strip()
            category = str(_unwrap(item.get("Category", ""))).strip()
            if listing_price <= 0 or seller_target <= 0 or buyer_target <= 0:
                continue
            if not title:
                continue
            span = seller_target - buyer_target
            if span < min_span:
                n_filtered_span += 1
                continue
            scenarios.append({
                "title": title, "description": description, "category": category,
                "listing_price": listing_price,
                "seller_target": seller_target, "buyer_target": buyer_target,
            })
        except (KeyError, ValueError, TypeError):
            continue

    log.info("Span filter (<%d): removed %d scenarios.", min_span, n_filtered_span)
    log.info("Found %d valid scenarios after filtering.", len(scenarios))
    if not scenarios:
        raise RuntimeError("No valid scenarios found after filtering.")
    k = min(num_samples, len(scenarios))
    return random.sample(scenarios, k)


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
        f"  - When ready to agree on a price, end your message with DEAL=<price> as the very last thing you write.\n"
        f"    Use only digits — no commas, no $ sign (e.g. DEAL=450 or DEAL=9300, NOT DEAL=9,300).\n"
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
        f"  - When ready to agree on a price, end your message with DEAL=<price> as the very last thing you write.\n"
        f"    Use only digits — no commas, no $ sign (e.g. DEAL=350 or DEAL=8500, NOT DEAL=8,500).\n"
        f"  - Only write DEAL= when you genuinely accept that price "
        f"(it must be at or below ${scenario['buyer_target']:.0f}).\n"
        f"  - If the price is above your maximum and the seller won't move, write REJECT on its own line.\n"
        f"    Use REJECT only when you would truly walk away — not as a bluff. Available from turn 3.\n"
    )


def is_deal(text: str) -> bool:
    return bool(re.search(r"DEAL\s*=\s*\$?\d+", text.upper()))


def is_reject(text: str) -> bool:
    return bool(_REJECT_RE.search(text))


def parse_deal_price(text: str) -> Optional[float]:
    m = re.search(r"DEAL\s*=\s*\$?([\d,]+(?:\.\d+)?)", text.upper())
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def score_deal(agreed_price: float, seller_target: float, buyer_target: float) -> Dict:
    span = seller_target - buyer_target
    if span <= 0:
        return {
            "seller_score": 0.5, "buyer_score": 0.5,
            "raw_seller_score": 0.5, "raw_buyer_score": 0.5,
            "clamped": False, "span": span,
            "midpoint": round((seller_target + buyer_target) / 2.0, 2),
            "midpoint_deviation": 0.0,
        }
    raw_seller = (agreed_price - buyer_target) / span
    raw_buyer = (seller_target - agreed_price) / span
    seller_score = max(0.0, min(1.0, raw_seller))
    buyer_score = max(0.0, min(1.0, raw_buyer))
    clamped = (seller_score != raw_seller) or (buyer_score != raw_buyer)
    midpoint = (seller_target + buyer_target) / 2.0
    midpoint_deviation = round((agreed_price - midpoint) / span, 4)
    return {
        "seller_score": round(seller_score, 4),
        "buyer_score": round(buyer_score, 4),
        "raw_seller_score": round(raw_seller, 4),
        "raw_buyer_score": round(raw_buyer, 4),
        "clamped": clamped,
        "span": round(span, 2),
        "midpoint": round(midpoint, 2),
        "midpoint_deviation": midpoint_deviation,
    }


# ---------------------------------------------------------------------------
# Prompt enhancements
# ---------------------------------------------------------------------------

ENHANCEMENTS = {
    "anchoring": (
        "Negotiation strategy — ANCHORING:\n"
        "  - Make an aggressive first offer to anchor the negotiation in your favor.\n"
        "  - All subsequent offers will be judged relative to your anchor.\n"
        "  - Justify your anchor with specific reasons (condition, comparable prices, urgency).\n"
        "  - Concede slowly from your anchor — each move should be small and reluctant.\n"
    ),
    "strategic_concession": (
        "Negotiation strategy — STRATEGIC CONCESSIONS:\n"
        "  - Make concessions that decrease in size (e.g. $50, then $30, then $10).\n"
        "  - Always tie each concession to a reason ('since you pointed out the scratch...').\n"
        "  - Signal that you're near your limit ('this is really stretching my budget').\n"
        "  - Never concede without getting something in return.\n"
    ),
    "batna": (
        "Negotiation strategy — BATNA AWARENESS:\n"
        "  - Reference your alternatives ('I've seen similar items listed for less').\n"
        "  - Make it clear you can walk away ('I have other options if we can't agree').\n"
        "  - Use your alternatives as leverage without bluffing.\n"
        "  - Be willing to actually walk away if the deal isn't good enough.\n"
    ),
    "firmness": (
        "Negotiation strategy — FIRMNESS:\n"
        "  - State your position clearly and don't waver.\n"
        "  - Repeat your key arguments when challenged rather than caving.\n"
        "  - Use confident language ('This is a fair price because...').\n"
        "  - Don't rush to close — patience is leverage.\n"
    ),
    "empathy": (
        "Negotiation strategy — EMPATHY & RAPPORT:\n"
        "  - Acknowledge the other person's perspective ('I understand you want a good price').\n"
        "  - Find common ground before negotiating numbers.\n"
        "  - Frame offers as mutually beneficial ('This works for both of us because...').\n"
        "  - Be friendly but don't sacrifice your position.\n"
    ),
    "combo": (
        "Negotiation strategy — USE ALL OF THESE:\n"
        "  1. ANCHOR aggressively with your first offer, justify it.\n"
        "  2. CONCEDE slowly — each concession smaller than the last, tied to a reason.\n"
        "  3. Reference ALTERNATIVES ('I've seen similar for less / I have other buyers').\n"
        "  4. Be FIRM — don't waver when challenged, repeat your key arguments.\n"
        "  5. Show EMPATHY — acknowledge their position, frame deals as mutual wins.\n"
        "  6. Signal your LIMIT gradually ('this is really my maximum/minimum').\n"
    ),
}


def enhance_prompt(base_prompt: str, enhancement_key: Optional[str]) -> str:
    if not enhancement_key:
        return base_prompt
    strategy = ENHANCEMENTS.get(enhancement_key)
    if not strategy:
        raise ValueError(f"Unknown enhancement '{enhancement_key}'. "
                         f"Choose from: {list(ENHANCEMENTS.keys())}")
    return base_prompt + "\n" + strategy


# ---------------------------------------------------------------------------
# API agent
# ---------------------------------------------------------------------------

def _postprocess(text: str, can_finalise: bool) -> str:
    """Shared post-processing for all agent outputs."""
    text = text.strip()
    text = re.sub(r'^(YOU|THEM)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    if not can_finalise:
        text = re.sub(r'DEAL\s*=\s*\$?[\d,]+', '', text, flags=re.IGNORECASE).strip()
        text = _REJECT_RE.sub('', text).strip()

    for line in text.splitlines():
        if is_reject(line):
            return "REJECT"
        if is_deal(line):
            return line.strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else text


def generate_turn_api(
    provider: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 200,
    can_finalise: bool = True,
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
                max_output_tokens=max_tokens,
            ),
        )
        text = response.text or ""

    elif provider == "gpt4o":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""

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
        text = response.content[0].text or ""

    elif provider == "groq":
        from groq import Groq
        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""

    else:
        raise ValueError(f"Unknown API provider: {provider}")

    return _postprocess(text, can_finalise)


# ---------------------------------------------------------------------------
# Local agent — lazy imports (torch/transformers only loaded if needed)
# ---------------------------------------------------------------------------

_local_models = {}


def load_local_model(model_key: str):
    if model_key in _local_models:
        return _local_models[model_key]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, str(PROJECT_ROOT))
    from extract_vectors import MODELS, HF_TOKEN

    model_cfg = MODELS[model_key]
    hf_token = HF_TOKEN if model_cfg.requires_token else None

    log.info("Loading local model: %s ...", model_cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.hf_id, token=hf_token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    log.info("Model loaded: %s", model_cfg.hf_id)

    _local_models[model_key] = (model, tokenizer)
    return model, tokenizer


def generate_turn_local(
    model_key: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 200,
    can_finalise: bool = True,
) -> str:
    sys.path.insert(0, str(PROJECT_ROOT))
    from apply_steering import generate_turn
    model, tokenizer = load_local_model(model_key)
    return generate_turn(
        model, tokenizer, messages,
        direction_vectors=None, alpha=0.0,
        max_new_tokens=max_tokens, temperature=temperature,
        can_finalise=can_finalise,
    )


# ---------------------------------------------------------------------------
# Unified game loop
# ---------------------------------------------------------------------------

def parse_agent_spec(spec: str) -> Tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or parts[0] not in ("api", "local"):
        raise ValueError(f"Invalid agent spec '{spec}'. Use api:<provider> or local:<model_key>")
    return parts[0], parts[1]


def agent_generate(
    agent_type: str, agent_key: str,
    messages: List[Dict],
    temperature: float, max_tokens: int, can_finalise: bool,
) -> str:
    if agent_type == "api":
        return generate_turn_api(agent_key, messages, temperature, max_tokens, can_finalise)
    else:
        return generate_turn_local(agent_key, messages, temperature, max_tokens, can_finalise)


def run_game(
    scenario: Dict,
    seller_agent: Tuple[str, str],
    buyer_agent: Tuple[str, str],
    seller_system: str,
    buyer_system: str,
    temperature: float = 0.7,
    max_tokens: int = 200,
    opening_bid_pct: float = 0.6,
    verbose: bool = False,
) -> Dict:
    transcript: List[Tuple[str, str]] = []
    agreed_price: Optional[float] = None
    dealmaker: Optional[str] = None
    walk_away = False
    walk_away_by: Optional[str] = None

    opening_bid = round(scenario["listing_price"] * opening_bid_pct)
    transcript.append(("buyer", f"Hi, I'm interested in this. Would you take ${opening_bid:.0f}?"))

    for turn in range(MAX_TURNS):
        can_finalise = turn >= MIN_TURNS_BEFORE_DEAL

        # ---- seller's turn ----
        lines = [("YOU: " if spk == "seller" else "THEM: ") + utt
                 for spk, utt in transcript]
        if can_finalise:
            instr = ("Write 1-2 natural sentences, then optionally end with DEAL=<price> "
                     "if you are ready to close.")
        else:
            instr = (f"Write 1-2 natural sentences making your counter-offer. "
                     f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
        seller_msgs = [
            {"role": "system", "content": seller_system},
            {"role": "user", "content": "\n".join(lines) + f"\n\nYour turn. {instr}"},
        ]

        utt_s = agent_generate(*seller_agent, seller_msgs, temperature, max_tokens, can_finalise)
        transcript.append(("seller", utt_s))
        if verbose:
            print(f"  [Turn {turn+1}] SELLER: {utt_s}")

        if can_finalise and is_reject(utt_s):
            walk_away, walk_away_by = True, "seller"
            break
        if can_finalise and is_deal(utt_s):
            agreed_price, dealmaker = parse_deal_price(utt_s), "seller"
            break

        # ---- buyer's turn ----
        lines = [("YOU: " if spk == "buyer" else "THEM: ") + utt
                 for spk, utt in transcript]
        if can_finalise:
            instr = ("Write 1-2 natural sentences, then optionally end with DEAL=<price> "
                     "if you are ready to close.")
        else:
            instr = (f"Write 1-2 natural sentences making your counter-offer. "
                     f"Do NOT include DEAL= for {MIN_TURNS_BEFORE_DEAL - turn} more turn(s).")
        buyer_msgs = [
            {"role": "system", "content": buyer_system},
            {"role": "user", "content": "\n".join(lines) + f"\n\nYour turn. {instr}"},
        ]

        utt_b = agent_generate(*buyer_agent, buyer_msgs, temperature, max_tokens, can_finalise)
        transcript.append(("buyer", utt_b))
        if verbose:
            print(f"  [Turn {turn+1}] BUYER:  {utt_b}")

        if can_finalise and is_reject(utt_b):
            walk_away, walk_away_by = True, "buyer"
            break
        if can_finalise and is_deal(utt_b):
            agreed_price, dealmaker = parse_deal_price(utt_b), "buyer"
            break

    # ---- score ----
    agreed = agreed_price is not None and not walk_away
    if agreed:
        scores = score_deal(agreed_price, scenario["seller_target"], scenario["buyer_target"])
    else:
        scores = {
            "seller_score": 0.0, "buyer_score": 0.0,
            "raw_seller_score": 0.0, "raw_buyer_score": 0.0,
            "clamped": False,
            "span": scenario["seller_target"] - scenario["buyer_target"],
            "midpoint": round((scenario["seller_target"] + scenario["buyer_target"]) / 2.0, 2),
            "midpoint_deviation": 0.0,
        }

    return {
        "agreed": agreed,
        "agreed_price": agreed_price,
        "dealmaker": dealmaker,
        "walk_away": walk_away,
        "walk_away_by": walk_away_by,
        "seller_score": scores["seller_score"],
        "buyer_score": scores["buyer_score"],
        "midpoint_deviation": scores["midpoint_deviation"] if agreed else 0.0,
        "clamped": scores["clamped"],
        "span": scores["span"],
        "num_turns": len(transcript),
        "transcript": [{"speaker": s, "utterance": u} for s, u in transcript],
        "listing_price": scenario["listing_price"],
        "seller_target": scenario["seller_target"],
        "buyer_target": scenario["buyer_target"],
        "title": scenario["title"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Task design validation: pluggable agents + prompt enhancements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--buyer", default="api:gemini",
                   help="Buyer agent spec. e.g. api:gemini, local:qwen2.5-3b")
    p.add_argument("--seller", default="api:gemini",
                   help="Seller agent spec. e.g. api:gpt4o, local:qwen2.5-3b")
    p.add_argument("--buyer_enhance", default=None, choices=list(ENHANCEMENTS.keys()),
                   help="Prompt enhancement for buyer.")
    p.add_argument("--seller_enhance", default=None, choices=list(ENHANCEMENTS.keys()),
                   help="Prompt enhancement for seller.")
    p.add_argument("--num_games", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--opening_bid_pct", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="train")
    p.add_argument("--min_span", type=int, default=100)
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full transcript for each game.")
    p.add_argument("--output", default=None,
                   help="Save results JSON to this path.")
    args = p.parse_args()

    random.seed(args.seed)
    buyer_agent = parse_agent_spec(args.buyer)
    seller_agent = parse_agent_spec(args.seller)

    scenarios = load_craigslist(split=args.split, num_samples=args.num_games,
                                min_span=args.min_span)
    k = min(args.num_games, len(scenarios))
    scenarios = scenarios[:k]

    print(f"\n{'='*70}")
    print(f"TASK DESIGN VALIDATION")
    print(f"  Buyer:  {args.buyer}" + (f"  + {args.buyer_enhance}" if args.buyer_enhance else ""))
    print(f"  Seller: {args.seller}" + (f"  + {args.seller_enhance}" if args.seller_enhance else ""))
    print(f"  Games:  {k}   Temp: {args.temperature}   MaxTokens: {args.max_tokens}")
    print(f"{'='*70}\n")

    results = []
    t0 = time.time()

    for i, sc in enumerate(scenarios):
        seller_sys = enhance_prompt(build_seller_system(sc), args.seller_enhance)
        buyer_sys = enhance_prompt(build_buyer_system(sc), args.buyer_enhance)

        if args.verbose:
            print(f"\n--- Game {i+1}/{k}: {sc['title'][:50]} ---")
            print(f"  Listed: ${sc['listing_price']:.0f}  "
                  f"Seller target: ${sc['seller_target']:.0f}  "
                  f"Buyer target: ${sc['buyer_target']:.0f}  "
                  f"Span: ${sc['seller_target'] - sc['buyer_target']:.0f}")

        r = run_game(
            scenario=sc,
            seller_agent=seller_agent,
            buyer_agent=buyer_agent,
            seller_system=seller_sys,
            buyer_system=buyer_sys,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            opening_bid_pct=args.opening_bid_pct,
            verbose=args.verbose,
        )
        results.append(r)

        status = "DEAL" if r["agreed"] else ("WALK" if r["walk_away"] else "TIMEOUT")
        price_str = f"${r['agreed_price']:.0f}" if r["agreed_price"] else "---"
        print(f"  Game {i+1}: {status}  price={price_str}  "
              f"seller={r['seller_score']:.3f}  buyer={r['buyer_score']:.3f}  "
              f"mpd={r['midpoint_deviation']:+.3f}  turns={r['num_turns']}")

    elapsed = time.time() - t0

    # ---- Summary ----
    agreed = [r for r in results if r["agreed"]]
    n_agreed = len(agreed)
    n_walk = sum(1 for r in results if r["walk_away"])

    print(f"\n{'='*70}")
    print(f"SUMMARY  ({k} games, {elapsed:.0f}s)")
    print(f"  Agreed: {n_agreed}/{k}  Walk-away: {n_walk}/{k}  "
          f"Timeout: {k - n_agreed - n_walk}/{k}")

    if agreed:
        seller_scores = [r["seller_score"] for r in agreed]
        buyer_scores = [r["buyer_score"] for r in agreed]
        mpds = [r["midpoint_deviation"] for r in agreed]
        prices = [r["agreed_price"] for r in agreed]

        mean = lambda xs: sum(xs) / len(xs)
        std = lambda xs: (sum((x - mean(xs))**2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5

        print(f"\n  Agreed games only (N={n_agreed}):")
        print(f"    Seller score:  {mean(seller_scores):.3f} +/- {std(seller_scores):.3f}")
        print(f"    Buyer score:   {mean(buyer_scores):.3f} +/- {std(buyer_scores):.3f}")
        print(f"    Midpoint dev:  {mean(mpds):+.3f} +/- {std(mpds):.3f}")
        print(f"    Avg price:     ${mean(prices):.0f}")
        print(f"    Price range:   ${min(prices):.0f} - ${max(prices):.0f}")
    print(f"{'='*70}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "config": {
                    "buyer": args.buyer,
                    "seller": args.seller,
                    "buyer_enhance": args.buyer_enhance,
                    "seller_enhance": args.seller_enhance,
                    "num_games": k,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                },
                "games": results,
            }, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
