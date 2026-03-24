#!/usr/bin/env python3
"""
apply_steering_preset_nego.py

Evaluates activation steering using 40 preset negotiation scenarios from
preset_negotiators/scenarios.json.

Unlike apply_steering.py (which pits two LLMs against each other), here the
opponent is fully scripted — 4 fixed turns per scenario. This removes all
opponent noise: the only variable is the steered model's behaviour.

Scenario split:
  Scenarios  1-20 : steered model = SELLER  (Jake and Sandra are scripted buyers)
  Scenarios 21-40 : steered model = BUYER   (Ray and Nina are scripted sellers)

Scoring:
  Seller scenarios — deal if steered model's final quoted price <= buyer reservation.
    Deal score  = deal_price          (higher = better)
    No-deal score = 0                 (no cash; failed to close)

  Buyer scenarios — deal if steered model's final quoted price >= seller reservation.
    Deal score  = deal_price          (lower = better)
    No-deal score = listing_price     (worst-case: forced to buy at list elsewhere)

Aggregate metrics:
  total_value_earned  = sum of deal prices across seller scenarios
  total_value_spent   = sum of deal prices (+ listing_price penalties) across buyer scenarios
  seller_norm         = total_value_earned / sum(listing_prices for seller scenarios)
  buyer_norm          = 1 - (total_value_spent - min_possible) / (max_possible - min_possible)
  combined_score      = (seller_norm + buyer_norm) / 2

Example run:
  python apply_steering_preset_nego.py \\
      --model qwen2.5-7b \\
      --dimension firmness \\
      --alpha 15 \\
      --layers 12 16 20 \\
      --output_file results_preset.json
"""

import re
import json
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

SCENARIOS_PATH = Path(__file__).resolve().parent / "preset_negotiators" / "scenarios.json"


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_preset_scenarios(path: Path = SCENARIOS_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data["scenarios"]
    log.info("Loaded %d preset scenarios from %s", len(scenarios), path)
    return scenarios


# ---------------------------------------------------------------------------
# Steering Hook  (identical to apply_steering.py)
# ---------------------------------------------------------------------------

class SteeringHook:
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
# Direction Vectors  (identical to apply_steering.py)
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
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    messages: List[Dict],
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> str:
    """
    Generate a single response with optional steering.
    Returns the full decoded response text (no DEAL=/REJECT parsing).
    """
    device    = next(model.parameters()).device
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    hooks: List[SteeringHook] = []
    if dvecs and alpha != 0.0:
        layers = get_transformer_layers(model)
        for layer_idx, vec in dvecs.items():
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
        for h in hooks:
            h.remove()

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Price Extraction
# ---------------------------------------------------------------------------

def extract_price(text: str) -> Optional[float]:
    """
    Extract the last dollar amount mentioned in text.
    Uses the last occurrence since that's typically the current offer/ask.
    """
    amounts = [
        float(a.replace(",", ""))
        for a in re.findall(r'\$\s*([\d,]+(?:\.\d+)?)', text)
    ]
    amounts = [a for a in amounts if a > 10]  # filter out incidental small numbers
    return amounts[-1] if amounts else None


# ---------------------------------------------------------------------------
# Scenario Runner
# ---------------------------------------------------------------------------

def run_preset_scenario(
    model,
    tokenizer,
    scenario: Dict,
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> Dict:
    """
    Run a single preset scenario. The opponent turns are scripted and fixed.
    The steered model responds to each of the 4 turns.
    Returns the full result dict including per-turn prices and deal outcome.
    """
    system_prompt       = scenario["system_prompt"]
    turns               = scenario["turns"]
    listing_price       = scenario["listing_price"]
    opponent_reservation = scenario["opponent_reservation"]
    steered_role        = scenario["steered_role"]
    deal_threshold      = scenario["deal_condition"]["threshold"]

    messages: List[Dict] = [{"role": "system", "content": system_prompt}]
    model_responses: List[str]          = []
    model_prices:    List[Optional[float]] = []

    for i, scripted_turn in enumerate(turns):
        # Add the scripted opponent turn as a user message
        user_content = (
            scripted_turn["text"]
            + "\n\nRemember to state a specific dollar amount in your response."
        )
        messages.append({"role": "user", "content": user_content})

        # Generate model response
        response = generate_response(
            model, tokenizer, messages, dvecs, alpha,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        price = extract_price(response)
        model_responses.append(response)
        model_prices.append(price)

        # Append to conversation so subsequent turns have context
        messages.append({"role": "assistant", "content": response})

        log.info(
            "[S%02d T%d] %s (α=%+.1f) → %s  |  price=$%s",
            scenario["id"], i + 1, steered_role.upper(), alpha,
            response[:90].replace("\n", " "),
            f"{price:.0f}" if price is not None else "?",
        )

    # --- Deal check ----------------------------------------------------------
    final_price = model_prices[-1] if model_prices else None

    if final_price is not None:
        if steered_role == "seller":
            deal = final_price <= deal_threshold
        else:
            deal = final_price >= deal_threshold
    else:
        deal = False

    # --- Scoring -------------------------------------------------------------
    if deal:
        deal_price = final_price
        value      = deal_price  # seller: higher=better; buyer: lower=better
    else:
        deal_price = None
        if steered_role == "seller":
            value = 0.0            # no sale → no cash
        else:
            value = float(listing_price)  # no deal → worst-case: buy at listing

    log.info(
        "[S%02d] %s | final=$%s | threshold=$%d | deal=%s | value=$%.0f",
        scenario["id"], steered_role.upper(),
        f"{final_price:.0f}" if final_price is not None else "?",
        deal_threshold, deal, value,
    )

    return {
        "id":                   scenario["id"],
        "item":                 scenario["item"],
        "character":            scenario["character"],
        "steered_role":         steered_role,
        "listing_price":        listing_price,
        "opponent_reservation": opponent_reservation,
        "deal_threshold":       deal_threshold,
        "final_price":          final_price,
        "deal":                 deal,
        "deal_price":           deal_price,
        "value":                value,
        "model_prices":         model_prices,
        "model_responses":      model_responses,
        "alpha":                alpha,
    }


def run_all_scenarios(
    model,
    tokenizer,
    scenarios: List[Dict],
    dvecs: Optional[Dict[int, np.ndarray]],
    alpha: float,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> List[Dict]:
    results = []
    for sc in scenarios:
        r = run_preset_scenario(
            model, tokenizer, sc, dvecs, alpha, max_new_tokens, temperature
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarise_preset(results: List[Dict]) -> Dict:
    """
    Compute aggregate metrics across all 40 preset scenarios.

    seller_norm : total_earned / sum_listing_prices  (0=no deals, 1=all deals at listing price)
    buyer_norm  : 1 - (total_spent - min_possible) / (max_possible - min_possible)
                  (0=worst case, 1=all deals at reservation price)
    combined    : (seller_norm + buyer_norm) / 2
    """
    seller_r = [r for r in results if r["steered_role"] == "seller"]
    buyer_r  = [r for r in results if r["steered_role"] == "buyer"]

    # --- Seller --------------------------------------------------------------
    seller_deals  = [r for r in seller_r if r["deal"]]
    seller_earned = sum(r["deal_price"] for r in seller_deals)
    seller_max    = sum(r["listing_price"] for r in seller_r)  # best possible
    seller_norm   = seller_earned / seller_max if seller_max > 0 else 0.0

    # --- Buyer ---------------------------------------------------------------
    buyer_deals    = [r for r in buyer_r if r["deal"]]
    buyer_spent    = sum(r["value"] for r in buyer_r)          # includes no-deal penalties
    buyer_min_poss = sum(r["opponent_reservation"] for r in buyer_r)  # best possible
    buyer_max_poss = sum(r["listing_price"] for r in buyer_r)         # worst possible
    buyer_span     = buyer_max_poss - buyer_min_poss
    buyer_norm     = 1.0 - (buyer_spent - buyer_min_poss) / buyer_span if buyer_span > 0 else 0.0

    combined = (seller_norm + buyer_norm) / 2.0

    return {
        "num_scenarios": len(results),
        "alpha":         results[0]["alpha"] if results else 0.0,
        "combined_score": round(combined, 4),
        "seller_norm":    round(seller_norm, 4),
        "buyer_norm":     round(buyer_norm, 4),
        "seller": {
            "n":                   len(seller_r),
            "deal_rate":           round(len(seller_deals) / len(seller_r), 3) if seller_r else 0,
            "total_earned":        round(seller_earned, 2),
            "max_possible_earned": seller_max,
            "avg_deal_price":      round(seller_earned / len(seller_deals), 2) if seller_deals else 0,
            "normalized_score":    round(seller_norm, 4),
        },
        "buyer": {
            "n":                  len(buyer_r),
            "deal_rate":          round(len(buyer_deals) / len(buyer_r), 3) if buyer_r else 0,
            "total_spent":        round(buyer_spent, 2),
            "min_possible_spent": buyer_min_poss,
            "max_possible_spent": buyer_max_poss,
            "avg_deal_price":     round(sum(r["deal_price"] for r in buyer_deals) / len(buyer_deals), 2) if buyer_deals else 0,
            "normalized_score":   round(buyer_norm, 4),
        },
    }


def print_summary(summary: Dict) -> None:
    alpha = summary.get("alpha", 0.0)
    s = summary["seller"]
    b = summary["buyer"]
    print(f"\n{'=' * 75}")
    print(f"PRESET NEGOTIATION RESULTS  (alpha={alpha:+.1f})")
    print(f"{'=' * 75}")
    print(f"  Combined score : {summary['combined_score']:+.4f}  "
          f"(seller_norm={summary['seller_norm']:+.4f}  buyer_norm={summary['buyer_norm']:+.4f})")
    print(f"  Seller  |  deal_rate={s['deal_rate']:.0%}  "
          f"earned=${s['total_earned']:,.0f} / ${s['max_possible_earned']:,.0f}  "
          f"(norm={s['normalized_score']:.4f})")
    print(f"  Buyer   |  deal_rate={b['deal_rate']:.0%}  "
          f"spent=${b['total_spent']:,.0f}  "
          f"[min=${b['min_possible_spent']:,.0f}  max=${b['max_possible_spent']:,.0f}]  "
          f"(norm={b['normalized_score']:.4f})")
    print(f"{'=' * 75}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate steering on 40 preset negotiation scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",      choices=list(MODELS.keys()), required=True)
    p.add_argument("--dimension",  default="firmness",
                   help="Steering dimension. Default: firmness.")
    p.add_argument("--alpha",      type=float, default=15.0,
                   help="Steering strength. Default: 15.0.")
    p.add_argument("--layers",     nargs="+", type=int, default=None,
                   help="Layer indices to steer. Required unless --no_steer.")
    p.add_argument("--no_steer",   action="store_true",
                   help="Run baseline (alpha=0, no vectors loaded).")
    p.add_argument("--vectors_dir", default="vectors/neg8dim_12pairs_matched/negotiation")
    p.add_argument("--method",      choices=["mean_diff", "pca"], default="mean_diff")
    p.add_argument("--scenarios_path", default=str(SCENARIOS_PATH),
                   help="Path to preset scenarios JSON.")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--dtype",          choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    p.add_argument("--quantize",       action="store_true")
    p.add_argument("--output_file",    default="results_preset.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = MODELS[args.model]
    hf_token  = HF_TOKEN if model_cfg.requires_token else None

    scenarios = load_preset_scenarios(Path(args.scenarios_path))

    # Load vectors (unless baseline run)
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
        log.info("Loaded vectors: dim=%s  layers=%s  alpha=%+.1f",
                 args.dimension, args.layers, args.alpha)

    alpha = 0.0 if args.no_steer else args.alpha

    # Load model
    log.info("Loading model: %s%s", model_cfg.hf_id, " [4-bit]" if args.quantize else "")
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
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        load_kwargs["max_memory"]  = {0: "15GiB"}

    model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, **load_kwargs)
    model.eval()

    # Run all scenarios
    results = run_all_scenarios(
        model, tokenizer, scenarios, dvecs, alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    summary = summarise_preset(results)
    print_summary(summary)

    output = {
        "summary": summary,
        "results": results,
        "run_info": {
            "model":       args.model,
            "dimension":   args.dimension,
            "alpha":       alpha,
            "layers":      args.layers,
            "method":      args.method,
            "no_steer":    args.no_steer,
            "temperature": args.temperature,
        },
    }
    with open(args.output_file, "w") as fh:
        json.dump(output, fh, indent=2)
    log.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
