#!/usr/bin/env python3
"""
Phase B: Measure unsteered Qwen 7B responder acceptance curve.

For each of 7 offer levels (20-80% of pool to responder), across all 100
variable pools, present the responder with a fixed offer and record
ACCEPT/REJECT. No steering involved.

Total: 7 levels x 100 pools = 700 calls. ~15-20 min on T4 with 4-bit.
"""

import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Pools (from ultimatum_game.py) ──────────────────────────────────────────
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

OFFER_LEVELS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def build_responder_system(proposer_share: int, responder_share: int, pool: int) -> str:
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


def parse_decision(text: str) -> str:
    """Parse ACCEPT/REJECT from responder text."""
    upper = text.upper()
    # Search from end (last occurrence is the final decision)
    accept_pos = upper.rfind("ACCEPT")
    reject_pos = upper.rfind("REJECT")
    if accept_pos == -1 and reject_pos == -1:
        return "UNCLEAR"
    if accept_pos == -1:
        return "REJECT"
    if reject_pos == -1:
        return "ACCEPT"
    return "ACCEPT" if accept_pos > reject_pos else "REJECT"


def main():
    t0 = time.time()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = Path("results/ultimatum/acceptance_curve")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run ─────────────────────────────────────────────────────────────────
    results = []
    total = len(OFFER_LEVELS) * len(POOL_SIZES)
    done = 0

    for offer_pct in OFFER_LEVELS:
        level_accept = 0
        level_total = 0

        for pool in POOL_SIZES:
            responder_share = round(pool * offer_pct)
            proposer_share = pool - responder_share

            system_prompt = build_responder_system(proposer_share, responder_share, pool)
            user_msg = (
                f"Player A offers: they get ${proposer_share}, you get ${responder_share}. "
                f"What is your decision?"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=1.0,   # temp=0 not supported directly; use do_sample=False
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only new tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            decision = parse_decision(text)

            results.append({
                "offer_pct": offer_pct,
                "pool": pool,
                "proposer_share": proposer_share,
                "responder_share": responder_share,
                "response": decision,
                "text": text,
            })

            if decision == "ACCEPT":
                level_accept += 1
            level_total += 1
            done += 1

            if done % 50 == 0:
                print(f"  [{done}/{total}] offer={offer_pct:.0%} pool=${pool} -> {decision}")

        p_accept = level_accept / level_total
        print(f"  offer_pct={offer_pct:.0%}: P(accept)={p_accept:.3f} ({level_accept}/{level_total})")

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "offer_levels": OFFER_LEVELS,
        "n_pools": len(POOL_SIZES),
        "total_calls": len(results),
        "model": MODEL_ID,
        "results": results,
    }
    out_path = out_dir / "acceptance_curve.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACCEPTANCE CURVE SUMMARY")
    print("=" * 60)
    print(f"{'Offer %':>10} | {'P(accept)':>10} | {'Accept':>7} | {'Reject':>7} | {'Unclear':>7}")
    print("-" * 60)
    for pct in OFFER_LEVELS:
        level_results = [r for r in results if r["offer_pct"] == pct]
        n_accept = sum(1 for r in level_results if r["response"] == "ACCEPT")
        n_reject = sum(1 for r in level_results if r["response"] == "REJECT")
        n_unclear = sum(1 for r in level_results if r["response"] == "UNCLEAR")
        n = len(level_results)
        p = n_accept / n if n > 0 else 0
        print(f"{pct:>9.0%} | {p:>10.3f} | {n_accept:>7} | {n_reject:>7} | {n_unclear:>7}")
    print("=" * 60)
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
