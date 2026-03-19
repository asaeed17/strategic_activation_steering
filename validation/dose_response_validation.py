#!/usr/bin/env python3
"""
dose_response_validation.py — Dose-response validation of steering vectors.

Tests whether scaling steering vectors by increasing alpha produces
monotonic behavioral changes, as rated by LLM judges.  This is the
gold-standard functional validation: if increasing alpha for the
"firmness" vector does NOT produce monotonically more firm responses
(as independently judged), the vector is not encoding firmness.

Three independent stages (can be run separately):
  Stage 1 (GPU):  Generate single-turn negotiation responses at multiple alphas
  Stage 2 (API):  Rate each response via LLM judge on 6 behavioural dimensions
  Stage 3 (CPU):  Analyse scores for monotonicity, specificity, coherence decay

Usage:
  # Stage 1: Generate (GPU required)
  python dose_response_validation.py --stage generate --model qwen2.5-3b

  # Stage 2: Judge (API key required, no GPU)
  python dose_response_validation.py --stage judge --judges gemini

  # Stage 3: Analyse (CPU only)
  python dose_response_validation.py --stage analyze

  # All stages at once
  python dose_response_validation.py --stage all --model qwen2.5-3b --judges gemini

Architecture:
  generate → dose_response_generations.json
           → judge → dose_response_scores.json
                   → analyze → dose_response_analysis.json + plots + report
"""

import json
import os
import re
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

JUDGE_DIMENSIONS = [
    "firmness",
    "persuasiveness",
    "naturalness",
    "coherence",
    "information_management",
    "strategic_reasoning",
]

# Default alpha schedule — from subtle to extreme
DEFAULT_ALPHAS = [0, 1, 2, 5, 10, 20, 50]

# Best layer per steering dimension (from experimental findings in P4)
DEFAULT_LAYERS: Dict[str, int] = {
    "strategic_concession_making": 18,
    "firmness": 16,
    "anchoring": 14,
}

# Which *judge* dimensions each *steering* dimension should primarily affect.
# Used to compute specificity: target dims should change more than off-targets.
DIMENSION_TARGET_MAP: Dict[str, List[str]] = {
    "firmness":                     ["firmness"],
    "anchoring":                    ["strategic_reasoning", "firmness"],
    "strategic_concession_making":  ["strategic_reasoning"],
}


# ───────────────────────────────────────────────────────────────────────
# Fixed evaluation scenarios — same prompts for every alpha level
# so the ONLY variable is the steering strength.
# ───────────────────────────────────────────────────────────────────────

EVAL_SCENARIOS = [
    {
        "id":          "laptop_seller",
        "title":       "Dell XPS 15 Laptop - Great Condition",
        "category":    "electronics",
        "listing_price": 800,
        "description": "2-year old Dell XPS 15, 16GB RAM, 512GB SSD. Minor lid scratches.",
        "role":        "seller",
        "target_price": 650,
        "opening_message": (
            "Hi, I'm interested in the laptop. Would you take $450 for it? "
            "I've seen similar ones going for around that price online."
        ),
    },
    {
        "id":          "bike_buyer",
        "title":       "Trek Mountain Bike - Barely Used",
        "category":    "bike",
        "listing_price": 600,
        "description": "Trek Marlin 7, 2023 model, ridden only 3 times. Includes helmet.",
        "role":        "buyer",
        "target_price": 400,
        "opening_message": (
            "Thanks for your interest! The bike is like new, retail was $850. "
            "I'm pretty firm at $600."
        ),
    },
    {
        "id":          "furniture_seller",
        "title":       "Solid Oak Dining Table with 6 Chairs",
        "category":    "furniture",
        "listing_price": 350,
        "description": "Solid oak table, excellent condition, seats 6. Must sell — moving.",
        "role":        "seller",
        "target_price": 250,
        "opening_message": (
            "I'd like to offer $150 for the table set. I noticed some wear "
            "in the photos and I'd need to arrange my own transport."
        ),
    },
    {
        "id":          "headphones_buyer",
        "title":       "Sony WH-1000XM5 Headphones",
        "category":    "electronics",
        "listing_price": 250,
        "description": "Sony XM5 noise-cancelling, 6 months old, original box and case.",
        "role":        "buyer",
        "target_price": 180,
        "opening_message": (
            "These are available at $250. They retail for $350 and are "
            "practically new with full warranty remaining."
        ),
    },
    {
        "id":          "housing_seller",
        "title":       "Cozy Studio Apartment - Downtown",
        "category":    "housing",
        "listing_price": 1500,
        "description": "Studio apartment, downtown, utilities included, available now.",
        "role":        "seller",
        "target_price": 1300,
        "opening_message": (
            "I'm looking for a place downtown. Would you consider $1100/month? "
            "There are several similar units listed in the area for less."
        ),
    },
    {
        "id":          "phone_buyer",
        "title":       "iPhone 14 Pro - 128GB Unlocked",
        "category":    "phone",
        "listing_price": 700,
        "description": "iPhone 14 Pro, unlocked, 128GB, space black. No damage.",
        "role":        "buyer",
        "target_price": 550,
        "opening_message": (
            "The phone is in perfect condition with AppleCare. "
            "I'm asking $700 firm — well below the retail price of $999."
        ),
    },
]


# ───────────────────────────────────────────────────────────────────────
# Judge system prompt — adapted for single-response evaluation
# ───────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert negotiation analyst evaluating a SINGLE response \
in a Craigslist bargaining scenario.

CRITICAL INSTRUCTIONS:
- Rate based on CONTENT and BEHAVIOR, NOT response length.
  A short, precise response can score higher than a long, rambling one.
- Do NOT reward or penalise based on opening phrases, politeness \
  formulas, or stylistic patterns.  Focus on negotiation substance.
- Each score MUST include a 1-sentence justification citing specific \
  behavior in the response.

RATING DIMENSIONS (1-5 Likert scale):

1. FIRMNESS — How firmly does the responder maintain their position?
   1 = Immediately capitulates or offers huge concessions
   2 = Makes weak counters, concedes easily
   3 = Holds position, makes moderate counter
   4 = Strong position maintenance, only small concessions
   5 = Extremely firm, refuses to move from stated position

2. PERSUASIVENESS — Quality and specificity of arguments
   1 = No arguments, just states a number
   2 = Generic reasoning ("it's a good deal")
   3 = Relevant arguments about the item
   4 = Strong, specific arguments with concrete evidence
   5 = Compelling multi-pronged arguments

3. NATURALNESS — Human-likeness of conversational style
   1 = Robotic, formulaic, obvious AI patterns
   2 = Mostly natural but AI tells present
   3 = Reads like a reasonable human negotiator
   4 = Natural and varied, personality comes through
   5 = Indistinguishable from a skilled human

4. COHERENCE — Logical consistency and relevance to scenario
   1 = Contradicts scenario, nonsensical, or off-topic
   2 = Partially relevant but confused
   3 = Logically consistent and on-topic
   4 = Strong logical flow, well-structured
   5 = Perfect coherence and relevance

5. INFORMATION_MANAGEMENT — Protecting private target price
   1 = Directly reveals private target/limit
   2 = Strongly implies constraints
   3 = Keeps targets private
   4 = Actively misdirects about constraints
   5 = Strategically manages information flow

6. STRATEGIC_REASONING — Tactical sophistication
   1 = No strategy, random/passive response
   2 = Basic haggling
   3 = Some tactical awareness (anchoring, framing)
   4 = Multiple tactics deployed effectively
   5 = Sophisticated adaptive strategy

CALIBRATION:
- "The price is $400. That's firm."
  → Firmness=5, Persuasiveness=1 (firm but no arguments).
  Short length does NOT lower any score.
- Long response with immediate 30% price reduction
  → Firmness=2 (concedes substantially). Length does NOT raise any score.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences:
{
  "firmness": {"score": <1-5>, "justification": "<1 sentence>"},
  "persuasiveness": {"score": <1-5>, "justification": "<1 sentence>"},
  "naturalness": {"score": <1-5>, "justification": "<1 sentence>"},
  "coherence": {"score": <1-5>, "justification": "<1 sentence>"},
  "information_management": {"score": <1-5>, "justification": "<1 sentence>"},
  "strategic_reasoning": {"score": <1-5>, "justification": "<1 sentence>"}
}"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Generate dose responses (GPU required)
# ═══════════════════════════════════════════════════════════════════════

def _build_generation_messages(scenario: Dict) -> List[Dict]:
    """Build chat messages for a single-turn generation from a scenario."""
    if scenario["role"] == "seller":
        system = (
            f"You are selling an item on Craigslist. Your goal is to get "
            f"the highest price possible.\n\n"
            f"Item: {scenario['title']}\n"
            f"Category: {scenario['category']}\n"
            f"Listed at: ${scenario['listing_price']}\n"
            f"Description: {scenario['description']}\n\n"
            f"Your private minimum: ${scenario['target_price']}. "
            f"Never reveal this number.\n\n"
            f"Respond naturally to the buyer's message in 2-3 sentences."
        )
    else:
        system = (
            f"You are buying an item on Craigslist. Your goal is to pay "
            f"the lowest price possible.\n\n"
            f"Item: {scenario['title']}\n"
            f"Category: {scenario['category']}\n"
            f"Listed at: ${scenario['listing_price']}\n"
            f"Description: {scenario['description']}\n\n"
            f"Your private maximum: ${scenario['target_price']}. "
            f"Never reveal this number.\n\n"
            f"Respond naturally to the seller's message in 2-3 sentences."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": scenario["opening_message"]},
    ]


def generate_dose_responses(
    model_alias:    str,
    dimensions:     List[str],
    alphas:         List[float],
    vectors_dir:    str  = "vectors_gpu",
    method:         str  = "mean_diff",
    layer_map:      Optional[Dict[str, int]] = None,
    temperature:    float = 0.1,
    max_new_tokens: int   = 150,
    output_file:    str   = "results/dose_response_generations.json",
    seed:           int   = 42,
) -> Dict:
    """
    Stage 1 — generate single-turn responses at each
    (dimension × alpha × scenario) combination.

    Requires: GPU, torch, transformers.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from apply_steering import (
        SteeringHook, get_transformer_layers, load_direction_vectors,
    )
    from extract_vectors import MODELS, HF_TOKEN

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if layer_map is None:
        layer_map = dict(DEFAULT_LAYERS)

    # ── Load model ───────────────────────────────────────────────────
    model_id = MODELS[model_alias].id
    log.info("Loading model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Prepare output ───────────────────────────────────────────────
    results: Dict[str, Any] = {
        "metadata": {
            "model_alias":    model_alias,
            "model_id":       model_id,
            "method":         method,
            "vectors_dir":    vectors_dir,
            "alphas":         alphas,
            "dimensions":     dimensions,
            "layer_map":      layer_map,
            "temperature":    temperature,
            "seed":           seed,
            "n_scenarios":    len(EVAL_SCENARIOS),
        },
        "generations": [],
    }

    total = len(dimensions) * len(alphas) * len(EVAL_SCENARIOS)
    count = 0

    for dimension in dimensions:
        layer = layer_map.get(dimension, 18)
        log.info("Dimension: %s, Layer: %d", dimension, layer)

        # Load the steering vector for this dimension + layer
        vec_path = Path(vectors_dir) / model_alias / method
        layer_file = vec_path / f"{dimension}_layer{layer:02d}.npy"
        if layer_file.exists():
            vec = np.load(layer_file)
        else:
            all_file = vec_path / f"{dimension}_all_layers.npy"
            if all_file.exists():
                all_vecs = np.load(all_file)
                vec = all_vecs[layer]
            else:
                log.warning(
                    "No vector found for %s layer %d — skipping.",
                    dimension, layer,
                )
                continue

        vec_tensor = torch.tensor(vec, dtype=torch.float32, device=device)
        layers = get_transformer_layers(model)

        for alpha in alphas:
            for scenario in EVAL_SCENARIOS:
                count += 1
                log.info(
                    "  [%d/%d] dim=%s  alpha=%g  scenario=%s",
                    count, total, dimension, alpha, scenario["id"],
                )

                messages = _build_generation_messages(scenario)
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(formatted, return_tensors="pt").to(device)

                # Attach steering hook
                hooks: list = []
                if alpha != 0:
                    h = SteeringHook(direction=vec_tensor, alpha=alpha)
                    h.register(layers[layer])
                    hooks.append(h)

                try:
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else 1.0,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                finally:
                    for h in hooks:
                        h.remove()

                new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(
                    new_tokens, skip_special_tokens=True,
                ).strip()

                results["generations"].append({
                    "dimension":       dimension,
                    "alpha":           alpha,
                    "layer":           layer,
                    "scenario_id":     scenario["id"],
                    "scenario_role":   scenario["role"],
                    "response":        response,
                    "response_length": len(response),
                    "response_words":  len(response.split()),
                })

    # ── Save ─────────────────────────────────────────────────────────
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(
        "Saved %d generations to %s", len(results["generations"]), out_path,
    )
    return results


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Judge dose responses (API required, no GPU)
# ═══════════════════════════════════════════════════════════════════════

def _build_judge_prompt(generation: Dict, scenario: Dict) -> str:
    """Build the user prompt for judging a single response."""
    role = scenario["role"]
    opponent = "buyer" if role == "seller" else "seller"
    return (
        f"SCENARIO:\n"
        f"Item: {scenario['title']}\n"
        f"Category: {scenario['category']}\n"
        f"Listed price: ${scenario['listing_price']}\n"
        f"The responder is the {role}.\n\n"
        f"The {opponent} said:\n\"{scenario['opening_message']}\"\n\n"
        f"The {role} responded:\n\"{generation['response']}\"\n\n"
        f"Rate this {role}'s response on all 6 dimensions."
    )


def _parse_single_response(text: str) -> Optional[Dict]:
    """Parse judge JSON for single-agent rating."""
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        else:
            return None

    scores: Dict[str, Any] = {}
    for dim in JUDGE_DIMENSIONS:
        if dim not in data:
            return None
        entry = data[dim]
        if isinstance(entry, dict) and "score" in entry:
            s = entry["score"]
            if isinstance(s, (int, float)) and 1 <= s <= 5:
                scores[dim] = {
                    "score": int(round(s)),
                    "justification": entry.get("justification", ""),
                }
            else:
                return None
        elif isinstance(entry, (int, float)) and 1 <= entry <= 5:
            scores[dim] = {"score": int(round(entry)), "justification": ""}
        else:
            return None
    return scores


def judge_dose_responses(
    generations_file: str        = "results/dose_response_generations.json",
    judge_names:      List[str]  = None,
    output_file:      str        = "results/dose_response_scores.json",
    sleep_sec:        float      = 1.5,
    thinking:         bool       = False,
) -> Dict:
    """
    Stage 2 — rate each generated response via LLM judge.

    Requires: API key(s) for the chosen judge(s).
    Produces: same JSON structure with 'judge_scores' added to each generation.
    Supports incremental checkpointing (safe to interrupt and resume).
    """
    from llm_judge import JUDGE_FNS

    if judge_names is None:
        judge_names = ["gemini"]

    # Validate API keys
    key_env: Dict[str, Tuple[str, ...]] = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gpt":    ("OPENAI_API_KEY",),
        "llama":  ("GROQ_API_KEY",),
    }
    for j in judge_names:
        keys = key_env[j]
        if not any(os.environ.get(k) for k in keys):
            log.error("No API key for '%s'. Set one of: %s", j, ", ".join(keys))
            sys.exit(1)

    # ── Load generations ─────────────────────────────────────────────
    with open(generations_file, encoding="utf-8") as f:
        data = json.load(f)

    generations = data["generations"]
    scenario_map = {s["id"]: s for s in EVAL_SCENARIOS}

    log.info(
        "Judging %d generations with %s",
        len(generations), ", ".join(judge_names),
    )

    # Randomise order to prevent systematic ordering effects
    indices = list(range(len(generations)))
    random.shuffle(indices)

    for idx_count, idx in enumerate(indices):
        gen = generations[idx]
        scenario = scenario_map.get(gen["scenario_id"])
        if scenario is None:
            log.warning("Unknown scenario %s — skipping", gen["scenario_id"])
            continue

        prompt = _build_judge_prompt(gen, scenario)

        if "judge_scores" not in gen:
            gen["judge_scores"] = {}

        for judge_name in judge_names:
            # Skip if already judged (resume support)
            if judge_name in gen.get("judge_scores", {}):
                continue

            log.info(
                "  [%d/%d] %s | dim=%s  alpha=%g  scenario=%s",
                idx_count + 1, len(indices), judge_name,
                gen["dimension"], gen["alpha"], gen["scenario_id"],
            )

            fn = JUDGE_FNS[judge_name]
            kwargs = {"thinking": thinking} if judge_name == "gemini" else {}

            raw = fn(prompt, JUDGE_SYSTEM_PROMPT, **kwargs)
            parsed = _parse_single_response(raw)

            # Retry once on parse failure
            if parsed is None and raw is not None:
                log.info("    [RETRY] Malformed JSON, retrying ...")
                time.sleep(2)
                raw = fn(prompt, JUDGE_SYSTEM_PROMPT, **kwargs)
                parsed = _parse_single_response(raw)

            if parsed:
                gen["judge_scores"][judge_name] = {
                    dim: parsed[dim]["score"] for dim in JUDGE_DIMENSIONS
                }
            else:
                log.warning("    [FAIL] Could not parse judge response")
                gen["judge_scores"][judge_name] = None

            time.sleep(sleep_sec)

        # Incremental save every 10 items
        if (idx_count + 1) % 10 == 0:
            _save_json(data, output_file)
            log.info(
                "    Checkpoint saved (%d/%d)", idx_count + 1, len(indices),
            )

    # Final save
    _save_json(data, output_file)
    log.info("Saved scored generations to %s", output_file)
    return data


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Analyse dose-response (CPU only, no GPU, no API)
# ═══════════════════════════════════════════════════════════════════════

def analyze_dose_response(
    scores_file: str = "results/dose_response_scores.json",
    output_dir:  str = "results/dose_response",
) -> Dict:
    """
    Stage 3 — analyse dose-response curves.

    For each steering dimension, computes:
      (a)  Spearman ρ of alpha vs each judge dimension  (monotonicity)
      (b)  Specificity ratio:  target-dim change / off-target-dim change
      (c)  Coherence / naturalness degradation curve
      (d)  Response-length confound (does word count track alpha?)
      (e)  Per-dimension verdict: PASS / WEAK_PASS / PARTIAL / FAIL

    Returns a results dict and saves JSON + plots + text report.
    """
    from scipy import stats as sp_stats

    with open(scores_file, encoding="utf-8") as f:
        data = json.load(f)

    generations = data["generations"]
    metadata = data.get("metadata", {})

    # Identify judges used
    judges_used: set = set()
    for gen in generations:
        if gen.get("judge_scores"):
            judges_used.update(
                k for k, v in gen["judge_scores"].items() if v is not None
            )
    if not judges_used:
        log.warning("No valid judge scores in %s", scores_file)
        return {"error": "No valid judge scores found"}

    dimensions = sorted(set(gen["dimension"] for gen in generations))
    alphas = sorted(set(gen["alpha"] for gen in generations))

    results: Dict[str, Any] = {
        "metadata":    metadata,
        "judges_used": sorted(judges_used),
        "alphas":      alphas,
        "dimensions":  {},
    }

    for dimension in dimensions:
        dim_gens = [g for g in generations if g["dimension"] == dimension]
        target_dims = DIMENSION_TARGET_MAP.get(dimension, [])

        dim_result: Dict[str, Any] = {
            "target_judge_dims": target_dims,
            "n_generations":     len(dim_gens),
            "alphas":            alphas,
            "judge_analysis":    {},
        }

        for judge_name in sorted(judges_used):
            judge_result: Dict[str, Any] = {
                "monotonicity":          {},
                "mean_scores_by_alpha":  {},
                "specificity":           {},
                "coherence_degradation": {},
                "length_analysis":       {},
            }

            # ── Mean score per alpha for each judge dim ──────────────
            for jdim in JUDGE_DIMENSIONS:
                alpha_means: List[Optional[float]] = []
                alpha_stds:  List[Optional[float]] = []
                for alpha in alphas:
                    scores = [
                        g["judge_scores"][judge_name][jdim]
                        for g in dim_gens
                        if g["alpha"] == alpha
                        and g.get("judge_scores", {}).get(judge_name)
                        and isinstance(g["judge_scores"][judge_name], dict)
                    ]
                    if scores:
                        alpha_means.append(float(np.mean(scores)))
                        alpha_stds.append(float(np.std(scores)))
                    else:
                        alpha_means.append(None)
                        alpha_stds.append(None)

                judge_result["mean_scores_by_alpha"][jdim] = {
                    "means": alpha_means,
                    "stds":  alpha_stds,
                }

                # Spearman ρ (alpha vs mean score)
                valid = [
                    (a, m) for a, m in zip(alphas, alpha_means)
                    if m is not None
                ]
                if len(valid) >= 3:
                    va, vm = zip(*valid)
                    rho, p_val = sp_stats.spearmanr(va, vm)
                    judge_result["monotonicity"][jdim] = {
                        "spearman_rho": float(rho),
                        "p_value":      float(p_val),
                        "direction":    "increasing" if rho > 0 else "decreasing",
                        "significant":  p_val < 0.05,
                    }
                else:
                    judge_result["monotonicity"][jdim] = {
                        "spearman_rho": None,
                        "p_value":      None,
                        "direction":    "insufficient_data",
                        "significant":  False,
                    }

            # ── Specificity ──────────────────────────────────────────
            target_rhos:    List[float] = []
            offtarget_rhos: List[float] = []
            for jdim in JUDGE_DIMENSIONS:
                mono = judge_result["monotonicity"].get(jdim, {})
                rho  = mono.get("spearman_rho")
                if rho is None:
                    continue
                if jdim in target_dims:
                    target_rhos.append(abs(rho))
                elif jdim not in ("naturalness", "coherence"):
                    # Exclude quality dims from off-target comparison
                    offtarget_rhos.append(abs(rho))

            if target_rhos and offtarget_rhos:
                mean_t  = float(np.mean(target_rhos))
                mean_ot = float(np.mean(offtarget_rhos))
                ratio   = mean_t / max(mean_ot, 0.01)
                judge_result["specificity"] = {
                    "mean_target_rho":    mean_t,
                    "mean_offtarget_rho": mean_ot,
                    "ratio":              round(ratio, 3),
                    "specific":           ratio > 1.5,
                }
            else:
                judge_result["specificity"] = {
                    "mean_target_rho":    None,
                    "mean_offtarget_rho": None,
                    "ratio":              None,
                    "specific":           None,
                }

            # ── Coherence / naturalness degradation ──────────────────
            coh_means = judge_result["mean_scores_by_alpha"].get(
                "coherence", {},
            ).get("means", [])
            nat_means = judge_result["mean_scores_by_alpha"].get(
                "naturalness", {},
            ).get("means", [])

            baseline_coh = (
                coh_means[0] if coh_means and coh_means[0] is not None
                else None
            )
            baseline_nat = (
                nat_means[0] if nat_means and nat_means[0] is not None
                else None
            )

            degradation_alpha = None
            for i, alpha in enumerate(alphas):
                c = coh_means[i] if i < len(coh_means) else None
                n = nat_means[i] if i < len(nat_means) else None
                if (c is not None and c < 2.0) or (n is not None and n < 2.0):
                    degradation_alpha = alpha
                    break

            judge_result["coherence_degradation"] = {
                "baseline_coherence":  (
                    float(baseline_coh) if baseline_coh is not None else None
                ),
                "baseline_naturalness": (
                    float(baseline_nat) if baseline_nat is not None else None
                ),
                "degradation_alpha": degradation_alpha,
                "degrades":          degradation_alpha is not None,
            }

            # ── Response-length confound ─────────────────────────────
            length_means: List[Optional[float]] = []
            for alpha in alphas:
                wcs = [
                    g["response_words"]
                    for g in dim_gens
                    if g["alpha"] == alpha
                ]
                length_means.append(
                    float(np.mean(wcs)) if wcs else None
                )

            valid_lp = [
                (a, m) for a, m in zip(alphas, length_means) if m is not None
            ]
            if len(valid_lp) >= 3:
                vla, vlm = zip(*valid_lp)
                len_rho, len_p = sp_stats.spearmanr(vla, vlm)
            else:
                len_rho, len_p = None, None

            judge_result["length_analysis"] = {
                "mean_words_by_alpha": length_means,
                "length_spearman_rho": (
                    float(len_rho) if len_rho is not None else None
                ),
                "length_p_value": (
                    float(len_p) if len_p is not None else None
                ),
                "length_changes_with_alpha": (
                    abs(len_rho) > 0.5 if len_rho is not None else None
                ),
            }

            dim_result["judge_analysis"][judge_name] = judge_result

        # ── Per-dimension verdict ────────────────────────────────────
        verdicts: List[str] = []
        for judge_name in sorted(judges_used):
            ja = dim_result["judge_analysis"][judge_name]

            target_monotonic = any(
                ja["monotonicity"].get(td, {}).get("significant", False)
                and ja["monotonicity"].get(td, {}).get("spearman_rho", 0) > 0
                for td in target_dims
            )
            specific = ja["specificity"].get("specific", False)
            degrades = ja["coherence_degradation"].get("degrades", False)

            if target_monotonic and specific and not degrades:
                verdicts.append("PASS")
            elif target_monotonic and specific and degrades:
                verdicts.append("PARTIAL")
            elif target_monotonic and not specific:
                verdicts.append("WEAK_PASS")
            elif not target_monotonic:
                verdicts.append("FAIL")
            else:
                verdicts.append("INCONCLUSIVE")

        # Majority vote if multiple judges
        dim_result["verdict"] = (
            verdicts[0] if len(verdicts) == 1
            else max(set(verdicts), key=verdicts.count)
        )

        results["dimensions"][dimension] = dim_result

    # ── Overall summary ──────────────────────────────────────────────
    dim_verdicts = [
        results["dimensions"][d]["verdict"] for d in dimensions
    ]
    pass_ct = sum(1 for v in dim_verdicts if v in ("PASS", "WEAK_PASS"))
    fail_ct = sum(1 for v in dim_verdicts if v == "FAIL")

    if fail_ct == 0:
        overall = "PASS"
    elif pass_ct > 0:
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    results["summary"] = {
        "total_dimensions":  len(dimensions),
        "pass_count":        pass_ct,
        "fail_count":        fail_ct,
        "overall_verdict":   overall,
    }

    # ── Persist ──────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = out_dir / "dose_response_analysis.json"
    _save_json(results, str(analysis_path))
    log.info("Analysis saved to %s", analysis_path)

    # Plots
    try:
        _plot_dose_response(results, out_dir)
    except Exception as e:
        log.warning("Plotting failed: %s", e)

    # Report
    report = format_dose_response_report(results)
    report_path = out_dir / "dose_response_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def _plot_dose_response(results: Dict, output_dir: Path) -> None:
    """Generate dose-response curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimensions = list(results["dimensions"].keys())
    if not dimensions:
        return

    alphas = results["alphas"]
    judge_name = results["judges_used"][0]

    # ── Per-dimension plots ──────────────────────────────────────────
    for dimension in dimensions:
        dim_data = results["dimensions"][dimension]
        ja = dim_data["judge_analysis"].get(judge_name, {})
        target_dims = dim_data["target_judge_dims"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) All judge dimensions vs alpha
        ax1 = axes[0]
        for jdim in JUDGE_DIMENSIONS:
            means = ja.get("mean_scores_by_alpha", {}).get(
                jdim, {},
            ).get("means", [])
            valid = [(a, m) for a, m in zip(alphas, means) if m is not None]
            if valid:
                va, vm = zip(*valid)
                style = "-o" if jdim in target_dims else "--x"
                lw    = 2.5 if jdim in target_dims else 1.0
                ax1.plot(va, vm, style, label=jdim, linewidth=lw)
        ax1.set_xlabel("Alpha (steering strength)")
        ax1.set_ylabel("Judge Score (1-5)")
        ax1.set_title(f"Dose-Response: {dimension}")
        ax1.legend(fontsize=7)
        ax1.set_ylim(0.5, 5.5)
        ax1.grid(True, alpha=0.3)

        # (b) Monotonicity bar chart (Spearman ρ per judge dim)
        ax2 = axes[1]
        rhos, labels, colours = [], [], []
        for jdim in JUDGE_DIMENSIONS:
            mono = ja.get("monotonicity", {}).get(jdim, {})
            rho  = mono.get("spearman_rho", 0)
            if rho is not None:
                rhos.append(rho)
                labels.append(jdim.replace("_", "\n"))
                colours.append(
                    "green" if jdim in target_dims else "steelblue"
                )
        if rhos:
            ax2.bar(range(len(rhos)), rhos, color=colours)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylabel("Spearman \\u03C1")
        ax2.set_title(f"Monotonicity: {dimension}")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3, axis="y")

        # (c) Response length vs alpha
        ax3 = axes[2]
        wc = ja.get("length_analysis", {}).get("mean_words_by_alpha", [])
        valid = [(a, w) for a, w in zip(alphas, wc) if w is not None]
        if valid:
            va, vw = zip(*valid)
            ax3.plot(va, vw, "-s", color="purple", linewidth=2)
        ax3.set_xlabel("Alpha")
        ax3.set_ylabel("Mean Response Words")
        ax3.set_title(f"Length vs Alpha: {dimension}")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            output_dir / f"dose_response_{dimension}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        log.info("Plot saved: dose_response_%s.png", dimension)

    # ── Cross-dimension summary ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(dimensions))
    width = 0.12

    for i, jdim in enumerate(JUDGE_DIMENSIONS):
        rhos_list = []
        for dimension in dimensions:
            mono = (
                results["dimensions"][dimension]
                ["judge_analysis"].get(judge_name, {})
                .get("monotonicity", {}).get(jdim, {})
            )
            rho = mono.get("spearman_rho", 0)
            rhos_list.append(rho if rho is not None else 0)
        ax.bar(
            x + i * width, rhos_list, width,
            label=jdim.replace("_", " "), alpha=0.8,
        )

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(
        [d.replace("_", "\n") for d in dimensions], fontsize=9,
    )
    ax.set_ylabel("Spearman \\u03C1 (alpha vs judge score)")
    ax.set_title(
        "Dose-Response Monotonicity: "
        "Steering Dimension × Judge Dimension"
    )
    ax.legend(fontsize=7, ncol=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        output_dir / "dose_response_summary.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("Summary plot saved: dose_response_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# Human-readable report
# ═══════════════════════════════════════════════════════════════════════

def format_dose_response_report(results: Dict) -> str:
    """Return a formatted text report of the dose-response analysis."""
    lines: List[str] = []

    def _line(s: str = "") -> None:
        lines.append(s)

    _line("=" * 70)
    _line("DOSE-RESPONSE VALIDATION REPORT")
    _line("=" * 70)

    md = results.get("metadata", {})
    _line(f"Model:     {md.get('model_alias', 'unknown')}")
    _line(f"Alphas:    {results.get('alphas', [])}")
    _line(f"Scenarios: {md.get('n_scenarios', '?')}")
    _line(f"Judges:    {', '.join(results.get('judges_used', []))}")
    _line()

    for dim_name, dim_data in results.get("dimensions", {}).items():
        verdict     = dim_data.get("verdict", "UNKNOWN")
        target_dims = dim_data.get("target_judge_dims", [])

        _line("-" * 70)
        _line(
            f"DIMENSION: {dim_name}  "
            f"(target judge dims: {', '.join(target_dims)})"
        )
        _line(f"  Verdict: {verdict}")

        for judge_name, ja in dim_data.get("judge_analysis", {}).items():
            _line(f"\n  Judge: {judge_name}")

            # Monotonicity table
            _line("  Monotonicity (Spearman rho):")
            _line(
                f"    {'Dimension':<25s}  {'rho':>7s}  "
                f"{'p-value':>9s}  {'Sig?':>5s}"
            )
            for jdim in JUDGE_DIMENSIONS:
                mono = ja.get("monotonicity", {}).get(jdim, {})
                rho  = mono.get("spearman_rho")
                p    = mono.get("p_value")
                sig  = "***" if mono.get("significant") else ""
                tgt  = " [TARGET]" if jdim in target_dims else ""
                if rho is not None:
                    _line(
                        f"    {jdim:<25s}  "
                        f"{rho:+7.3f}  {p:9.4f}  {sig:>5s}{tgt}"
                    )
                else:
                    _line(f"    {jdim:<25s}  {'n/a':>7s}")

            # Specificity
            spec = ja.get("specificity", {})
            if spec.get("ratio") is not None:
                _line(
                    f"  Specificity: ratio={spec['ratio']:.2f}  "
                    f"(target |rho|={spec['mean_target_rho']:.3f}, "
                    f"off-target |rho|={spec['mean_offtarget_rho']:.3f})  "
                    f"Specific: {'Yes' if spec['specific'] else 'No'}"
                )

            # Coherence degradation
            coh = ja.get("coherence_degradation", {})
            if coh.get("degradation_alpha") is not None:
                _line(
                    f"  Coherence degrades at alpha={coh['degradation_alpha']}"
                )
            else:
                _line("  No coherence degradation detected")

            # Length
            la = ja.get("length_analysis", {})
            lr = la.get("length_spearman_rho")
            if lr is not None:
                _line(
                    f"  Length correlation: rho={lr:.3f}  "
                    f"(confound: {'Yes' if la.get('length_changes_with_alpha') else 'No'})"
                )

        _line()

    # Overall
    summary = results.get("summary", {})
    _line("=" * 70)
    _line(f"OVERALL VERDICT: {summary.get('overall_verdict', 'UNKNOWN')}")
    _line(
        f"  Pass: {summary.get('pass_count', 0)}/"
        f"{summary.get('total_dimensions', 0)}   "
        f"Fail: {summary.get('fail_count', 0)}/"
        f"{summary.get('total_dimensions', 0)}"
    )
    _line("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════

def _save_json(data: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dose-response validation of steering vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Generate responses (GPU)
  python dose_response_validation.py --stage generate --model qwen2.5-3b

  # Stage 2: Judge responses (API key)
  python dose_response_validation.py --stage judge --judges gemini

  # Stage 3: Analyse results (CPU)
  python dose_response_validation.py --stage analyze

  # All stages
  python dose_response_validation.py --stage all --model qwen2.5-3b --judges gemini
""",
    )

    p.add_argument(
        "--stage", required=True,
        choices=["generate", "judge", "analyze", "all"],
        help="Which pipeline stage to run",
    )
    p.add_argument(
        "--model", default="qwen2.5-3b",
        help="Model alias (default: qwen2.5-3b)",
    )
    p.add_argument(
        "--dimensions", nargs="+",
        default=["firmness", "anchoring", "strategic_concession_making"],
        help="Steering dimensions to evaluate",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=DEFAULT_ALPHAS,
        help=f"Alpha values for the dose curve (default: {DEFAULT_ALPHAS})",
    )
    p.add_argument(
        "--vectors_dir", default="vectors_gpu",
        help="Directory containing steering vectors",
    )
    p.add_argument(
        "--method", default="mean_diff",
        help="Vector extraction method (default: mean_diff)",
    )
    p.add_argument(
        "--judges", nargs="+", default=["gemini"],
        choices=["gemini", "gpt", "llama"],
        help="LLM judges to use for rating",
    )
    p.add_argument(
        "--thinking", action="store_true",
        help="Enable Gemini thinking mode",
    )
    p.add_argument(
        "--generations_file",
        default="results/dose_response_generations.json",
        help="Path to generations JSON",
    )
    p.add_argument(
        "--scores_file",
        default="results/dose_response_scores.json",
        help="Path to scored generations JSON",
    )
    p.add_argument(
        "--output_dir",
        default="results/dose_response",
        help="Output directory for analysis results and plots",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature (default: 0.1 for reproducibility)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--sleep", type=float, default=1.5,
        help="Seconds between API calls (default: 1.5)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.stage in ("generate", "all"):
        generate_dose_responses(
            model_alias=args.model,
            dimensions=args.dimensions,
            alphas=args.alphas,
            vectors_dir=args.vectors_dir,
            method=args.method,
            temperature=args.temperature,
            output_file=args.generations_file,
            seed=args.seed,
        )

    if args.stage in ("judge", "all"):
        judge_dose_responses(
            generations_file=args.generations_file,
            judge_names=args.judges,
            output_file=args.scores_file,
            sleep_sec=args.sleep,
            thinking=args.thinking,
        )

    if args.stage in ("analyze", "all"):
        results = analyze_dose_response(
            scores_file=args.scores_file,
            output_dir=args.output_dir,
        )
        report = format_dose_response_report(results)
        print(report)


if __name__ == "__main__":
    main()
