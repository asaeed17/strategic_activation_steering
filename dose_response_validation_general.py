#!/usr/bin/env python3
"""
dose_response_validation_general.py — Dose-response validation of steering vectors.

Tests whether scaling steering vectors by increasing alpha produces
monotonic behavioural changes, as rated by LLM judges.  This is the
gold-standard functional validation: if increasing alpha for a given
steering dimension does NOT produce monotonically stronger responses
(as independently judged), the vector is not encoding that dimension.

This version uses general, topic-diverse single-turn prompts instead of
negotiation-specific scenarios, covering: creative writing, factual Q&A,
emotional support, technical explanation, persuasive argumentation,
summarisation, and casual conversation.

Three independent stages (can be run separately):
  Stage 1 (GPU):  Generate single-turn responses at multiple alphas
  Stage 2 (API):  Rate each response via LLM judge on 6 behavioural dimensions
  Stage 3 (CPU):  Analyse scores for monotonicity, specificity, coherence decay

Usage:
  # Stage 1: Generate (GPU required)
  python dose_response_validation_general.py --stage generate --model qwen2.5-3b

  # Stage 2: Judge (API key required, no GPU)
  python dose_response_validation_general.py --stage judge --judges gemini

  # Stage 3: Analyse (CPU only)
  python dose_response_validation_general.py --stage analyze

  # All stages at once
  python dose_response_validation_general.py --stage all --model qwen2.5-3b --judges gemini

Architecture:
  generate → dose_response_general_generations.json
           → judge → dose_response_general_scores.json
                   → analyze → dose_response_general_analysis.json + plots + report
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
    "confidence",
    "creativity",
    "empathy",
    "clarity",
    "detail",
    "coherence",
]

# Default alpha schedule — from subtle to extreme
DEFAULT_ALPHAS = [0, 1, 2, 5, 10, 20, 50]

# Best layer per steering dimension (from experimental findings — update as needed)
DEFAULT_LAYERS: Dict[str, int] = {
    "firmness":                     16,
    "empathy":                      15,
    "active_listening":             14,
    "assertiveness":                16,
    "interest_based_reasoning":     18,
    "emotional_regulation":         15,
    "strategic_concession_making":  18,
    "anchoring":                    14,
    "rapport_building":             15,
    "batna_awareness":              17,
    "reframing":                    18,
    "patience":                     15,
    "value_creation":               18,
    "information_gathering":        14,
    "clarity_and_directness":       16,
}

# Which *judge* dimensions each *steering* dimension should primarily affect.
# Maps steering dimensions → general judge dimensions (confidence, creativity,
# empathy, clarity, detail, coherence).
DIMENSION_TARGET_MAP: Dict[str, List[str]] = {
    "firmness":                     ["confidence"],
    "empathy":                      ["empathy"],
    "active_listening":             ["empathy", "detail"],
    "assertiveness":                ["confidence"],
    "interest_based_reasoning":     ["clarity", "creativity"],
    "emotional_regulation":         ["coherence"],
    "strategic_concession_making":  ["clarity", "coherence"],
    "anchoring":                    ["confidence"],
    "rapport_building":             ["empathy"],
    "batna_awareness":              ["detail", "clarity"],
    "reframing":                    ["creativity"],
    "patience":                     ["coherence", "empathy"],
    "value_creation":               ["creativity", "detail"],
    "information_gathering":        ["detail"],
    "clarity_and_directness":       ["clarity"],
}


# ───────────────────────────────────────────────────────────────────────
# Evaluation scenarios — diverse topics, no negotiation framing.
# The system prompt sets a neutral assistant role; only the user message
# varies.  The same prompts are used at every alpha level so that
# steering strength is the ONLY variable.
# ───────────────────────────────────────────────────────────────────────

EVAL_SCENARIOS = [
    # ── Creative writing ─────────────────────────────────────────────
    {
        "id":       "short_story_prompt",
        "topic":    "creative_writing",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Write a short opening paragraph for a science fiction story "
            "set on a space station orbiting a dying star."
        ),
    },
    {
        "id":       "poem_request",
        "topic":    "creative_writing",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Write a short poem about the feeling of walking through a "
            "forest after rain."
        ),
    },

    # ── Factual Q&A ──────────────────────────────────────────────────
    {
        "id":       "history_question",
        "topic":    "factual_qa",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Can you explain what caused the fall of the Western Roman Empire? "
            "I'm looking for a clear, informative answer."
        ),
    },
    {
        "id":       "science_question",
        "topic":    "factual_qa",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "How does a vaccine train the immune system? "
            "Please explain it simply."
        ),
    },
    {
        "id":       "geography_question",
        "topic":    "factual_qa",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     "Why is the Sahara Desert so large and dry?",
    },

    # ── Technical explanation ─────────────────────────────────────────
    {
        "id":       "coding_help",
        "topic":    "technical",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Can you explain the difference between a stack and a queue "
            "in programming? Use a simple real-world analogy."
        ),
    },
    {
        "id":       "math_explanation",
        "topic":    "technical",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "What is a derivative in calculus, and why is it useful? "
            "Explain as if I'm new to the topic."
        ),
    },

    # ── Emotional support / empathy ───────────────────────────────────
    {
        "id":       "work_stress",
        "topic":    "emotional_support",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "I've been really overwhelmed at work lately and I'm not sure "
            "how to cope. Everything feels like too much."
        ),
    },
    {
        "id":       "loneliness",
        "topic":    "emotional_support",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "I moved to a new city six months ago and I still haven't made "
            "any friends. I'm starting to feel really lonely."
        ),
    },

    # ── Persuasive argumentation ──────────────────────────────────────
    {
        "id":       "exercise_argument",
        "topic":    "persuasion",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Convince me that I should start exercising regularly, "
            "even though I have a very busy schedule."
        ),
    },
    {
        "id":       "reading_argument",
        "topic":    "persuasion",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Make the case for why reading fiction is valuable for adults, "
            "not just entertainment."
        ),
    },

    # ── Summarisation ─────────────────────────────────────────────────
    {
        "id":       "concept_summary",
        "topic":    "summarisation",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Give me a brief, clear summary of what machine learning is "
            "and how it differs from traditional programming."
        ),
    },
    {
        "id":       "event_summary",
        "topic":    "summarisation",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Summarise the main causes and consequences of World War I "
            "in a few sentences."
        ),
    },

    # ── Casual conversation ───────────────────────────────────────────
    {
        "id":       "weekend_ideas",
        "topic":    "casual",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "I'm bored this weekend and looking for something fun to do "
            "indoors. Any ideas?"
        ),
    },
    {
        "id":       "food_recommendation",
        "topic":    "casual",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "I want to try cooking something new for dinner tonight. "
            "What's a recipe that's impressive but not too difficult?"
        ),
    },

    # ── Opinion / reflection ──────────────────────────────────────────
    {
        "id":       "technology_opinion",
        "topic":    "opinion",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "Do you think social media has been mostly good or mostly bad "
            "for society? Share your perspective."
        ),
    },
    {
        "id":       "future_of_work",
        "topic":    "opinion",
        "system":   "You are a helpful assistant. Respond naturally to the user.",
        "user":     (
            "What do you think the workplace will look like in 20 years, "
            "with AI becoming so capable?"
        ),
    },
]


# ───────────────────────────────────────────────────────────────────────
# Judge system prompt — general single-response evaluation
# ───────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert language analyst evaluating a SINGLE AI assistant \
response across a range of topics.

CRITICAL INSTRUCTIONS:
- Rate based on CONTENT and BEHAVIOUR, NOT response length.
  A short, precise response can score higher than a long, rambling one.
- Do NOT reward or penalise based on opening phrases, politeness \
  formulas, or stylistic patterns. Focus on substance.
- Each score MUST include a 1-sentence justification citing specific \
  behaviour in the response.

RATING DIMENSIONS (1-5 Likert scale):

1. CONFIDENCE — How confidently and directly does the response assert \
   its content?
   1 = Heavily hedged, apologetic, or refuses to commit
   2 = Mostly uncertain, many qualifiers
   3 = Moderate confidence, some hedging
   4 = Confident and assertive, minimal hedging
   5 = Extremely direct, no hedging, bold claims

2. CREATIVITY — Originality, novelty of ideas, or imaginative expression
   1 = Completely generic, clichéd, or template-like
   2 = Mostly familiar ideas with little originality
   3 = Competent, some fresh angles
   4 = Notably original or inventive
   5 = Highly creative, surprising, or distinctive

3. EMPATHY — Emotional attunement and warmth toward the user
   1 = Cold, dismissive, or tone-deaf to the user's emotional state
   2 = Minimal acknowledgement of the user's feelings
   3 = Appropriate warmth, acknowledges context
   4 = Genuinely empathetic, validates feelings clearly
   5 = Deeply attuned, emotionally intelligent response

4. CLARITY — How clear and easy to understand the response is
   1 = Confusing, contradictory, or hard to follow
   2 = Often unclear or poorly structured
   3 = Mostly clear and logical
   4 = Very clear, well-organised
   5 = Exceptionally clear and concise

5. DETAIL — Depth and thoroughness of the content
   1 = Superficial or near-empty response
   2 = Sparse, misses important aspects
   3 = Adequate coverage
   4 = Thorough and informative
   5 = Exceptionally detailed, comprehensive

6. COHERENCE — Logical consistency and relevance to the prompt
   1 = Contradicts the prompt, nonsensical, or off-topic
   2 = Partially relevant but confused
   3 = Logically consistent and on-topic
   4 = Strong logical flow, well-structured
   5 = Perfect coherence and relevance

CALIBRATION:
- "Exercise is good for you. You should do it." (to a persuasion prompt)
  → Confidence=5, Creativity=1, Detail=1 (bold but thin)
- "I totally understand how hard that sounds... maybe perhaps you could try..."
  (to an emotional support prompt)
  → Empathy=4, Confidence=1 (warm but very hesitant)

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences:
{
  "confidence":  {"score": <1-5>, "justification": "<1 sentence>"},
  "creativity":  {"score": <1-5>, "justification": "<1 sentence>"},
  "empathy":     {"score": <1-5>, "justification": "<1 sentence>"},
  "clarity":     {"score": <1-5>, "justification": "<1 sentence>"},
  "detail":      {"score": <1-5>, "justification": "<1 sentence>"},
  "coherence":   {"score": <1-5>, "justification": "<1 sentence>"}
}"""


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Generate dose responses (GPU required)
# ═══════════════════════════════════════════════════════════════════════

def _build_generation_messages(scenario: Dict) -> List[Dict]:
    """Build chat messages for a single-turn generation from a scenario."""
    return [
        {"role": "system", "content": scenario["system"]},
        {"role": "user",   "content": scenario["user"]},
    ]


def generate_dose_responses(
    model_alias:    str,
    dimensions:     List[str],
    alphas:         List[float],
    vectors_dir:    str  = "vectors_gpu",
    method:         str  = "mean_diff",
    layer_map:      Optional[Dict[str, int]] = None,
    temperature:    float = 0.1,
    max_new_tokens: int   = 200,
    output_file:    str   = "results/dose_response_general_generations.json",
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

    # ── Auto-detect dimensions from metadata.json if none provided ────
    if not dimensions:
        meta_path = Path(vectors_dir) / model_alias / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            dimensions = meta.get("dimensions", [])
            log.info("Auto-detected %d dimensions from %s: %s",
                     len(dimensions), meta_path, dimensions)
        else:
            dimensions = list(DEFAULT_LAYERS.keys())
            log.warning("No metadata.json found at %s — using DEFAULT_LAYERS keys", meta_path)

    if layer_map is None:
        layer_map = dict(DEFAULT_LAYERS)

    # ── Load model ───────────────────────────────────────────────────
    model_id = MODELS[model_alias].hf_id
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
                    "scenario_topic":  scenario["topic"],
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
    return (
        f"SCENARIO:\n"
        f"Topic: {scenario['topic']}\n"
        f"User message:\n\"{scenario['user']}\"\n\n"
        f"Assistant response:\n\"{generation['response']}\"\n\n"
        f"Rate this assistant response on all 6 dimensions."
    )


def _parse_single_response(text: str) -> Optional[Dict]:
    """Parse judge JSON for single-response rating."""
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
    generations_file: str        = "results/dose_response_general_generations.json",
    judge_names:      List[str]  = None,
    output_file:      str        = "results/dose_response_general_scores.json",
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
        "ollama": (),  # local, no API key needed
    }
    for j in judge_names:
        keys = key_env[j]
        if keys and not any(os.environ.get(k) for k in keys):
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
    scores_file: str = "results/dose_response_general_scores.json",
    output_dir:  str = "results/dose_response_general",
) -> Dict:
    """
    Stage 3 — analyse dose-response curves.

    For each steering dimension, computes:
      (a)  Spearman rho of alpha vs each judge dimension  (monotonicity)
      (b)  Specificity ratio:  target-dim change / off-target-dim change
      (c)  Coherence degradation curve
      (d)  Response-length confound (does word count track alpha?)
      (e)  Per-topic breakdown — does the effect hold across all topic areas?
      (f)  Per-dimension verdict: PASS / WEAK_PASS / PARTIAL / FAIL

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
    topics = sorted(set(gen["scenario_topic"] for gen in generations))

    results: Dict[str, Any] = {
        "metadata":    metadata,
        "judges_used": sorted(judges_used),
        "alphas":      alphas,
        "topics":      topics,
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
                "topic_breakdown":       {},
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

                # Spearman rho (alpha vs mean score)
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
                elif jdim not in ("coherence",):
                    # Exclude general quality dim from off-target comparison
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

            # ── Coherence degradation ─────────────────────────────────
            coh_means = judge_result["mean_scores_by_alpha"].get(
                "coherence", {},
            ).get("means", [])

            baseline_coh = (
                coh_means[0] if coh_means and coh_means[0] is not None
                else None
            )

            degradation_alpha = None
            for i, alpha in enumerate(alphas):
                c = coh_means[i] if i < len(coh_means) else None
                if c is not None and c < 2.0:
                    degradation_alpha = alpha
                    break

            judge_result["coherence_degradation"] = {
                "baseline_coherence": (
                    float(baseline_coh) if baseline_coh is not None else None
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

            # ── Per-topic breakdown ──────────────────────────────────
            for topic in topics:
                topic_gens = [
                    g for g in dim_gens if g["scenario_topic"] == topic
                ]
                topic_rhos: Dict[str, Optional[float]] = {}
                for jdim in JUDGE_DIMENSIONS:
                    t_alpha_means = []
                    for alpha in alphas:
                        sc = [
                            g["judge_scores"][judge_name][jdim]
                            for g in topic_gens
                            if g["alpha"] == alpha
                            and g.get("judge_scores", {}).get(judge_name)
                            and isinstance(g["judge_scores"][judge_name], dict)
                        ]
                        t_alpha_means.append(
                            float(np.mean(sc)) if sc else None
                        )
                    valid_t = [
                        (a, m)
                        for a, m in zip(alphas, t_alpha_means)
                        if m is not None
                    ]
                    if len(valid_t) >= 3:
                        vta, vtm = zip(*valid_t)
                        t_rho, _ = sp_stats.spearmanr(vta, vtm)
                        topic_rhos[jdim] = float(t_rho)
                    else:
                        topic_rhos[jdim] = None
                judge_result["topic_breakdown"][topic] = topic_rhos

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

    analysis_path = out_dir / "dose_response_general_analysis.json"
    _save_json(results, str(analysis_path))
    log.info("Analysis saved to %s", analysis_path)

    # Plots
    try:
        _plot_dose_response(results, out_dir)
    except Exception as e:
        log.warning("Plotting failed: %s", e)

    # Report
    report = format_dose_response_report(results)
    report_path = out_dir / "dose_response_general_report.txt"
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
    topics = results.get("topics", [])

    # ── Per-dimension plots ──────────────────────────────────────────
    for dimension in dimensions:
        dim_data = results["dimensions"][dimension]
        ja = dim_data["judge_analysis"].get(judge_name, {})
        target_dims = dim_data["target_judge_dims"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (a) All judge dimensions vs alpha
        ax1 = axes[0, 0]
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
        ax1.set_title(f"Dose-Response Curves: {dimension}")
        ax1.legend(fontsize=7)
        ax1.set_ylim(0.5, 5.5)
        ax1.grid(True, alpha=0.3)

        # (b) Monotonicity bar chart (Spearman rho per judge dim)
        ax2 = axes[0, 1]
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
        ax2.set_ylabel("Spearman rho")
        ax2.set_title(f"Monotonicity (Spearman rho): {dimension}")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3, axis="y")

        # (c) Response length vs alpha
        ax3 = axes[1, 0]
        wc = ja.get("length_analysis", {}).get("mean_words_by_alpha", [])
        valid = [(a, w) for a, w in zip(alphas, wc) if w is not None]
        if valid:
            va, vw = zip(*valid)
            ax3.plot(va, vw, "-s", color="purple", linewidth=2)
        ax3.set_xlabel("Alpha")
        ax3.set_ylabel("Mean Response Words")
        ax3.set_title(f"Response Length vs Alpha: {dimension}")
        ax3.grid(True, alpha=0.3)

        # (d) Per-topic target dimension rho
        ax4 = axes[1, 1]
        topic_rho_data = ja.get("topic_breakdown", {})
        if target_dims and topics:
            primary_target = target_dims[0]
            t_rhos = [
                topic_rho_data.get(t, {}).get(primary_target, 0) or 0
                for t in topics
            ]
            colours_t = ["green" if r > 0.5 else "orange" if r > 0 else "red"
                         for r in t_rhos]
            ax4.bar(range(len(topics)), t_rhos, color=colours_t)
            ax4.set_xticks(range(len(topics)))
            ax4.set_xticklabels(
                [t.replace("_", "\n") for t in topics], fontsize=7
            )
            ax4.set_ylabel(f"Spearman rho ({primary_target})")
            ax4.set_title(f"Per-Topic Monotonicity: {dimension}")
            ax4.axhline(0, color="black", linewidth=0.5)
            ax4.set_ylim(-1, 1)
            ax4.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"Dose-Response Analysis: {dimension}  (judge: {judge_name})",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        fig.savefig(
            output_dir / f"dose_response_{dimension}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        log.info("Plot saved: dose_response_%s.png", dimension)

    # ── Cross-dimension summary ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(dimensions))
    width = 0.13

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
    ax.set_ylabel("Spearman rho (alpha vs judge score)")
    ax.set_title(
        "Dose-Response Monotonicity Summary: "
        "Steering Dimension x Judge Dimension"
    )
    ax.legend(fontsize=7, ncol=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(
        output_dir / "dose_response_general_summary.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("Summary plot saved: dose_response_general_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# Human-readable report
# ═══════════════════════════════════════════════════════════════════════

def format_dose_response_report(results: Dict) -> str:
    """Return a formatted text report of the dose-response analysis."""
    lines: List[str] = []

    def _line(s: str = "") -> None:
        lines.append(s)

    _line("=" * 70)
    _line("DOSE-RESPONSE VALIDATION REPORT  (General Topics)")
    _line("=" * 70)

    md = results.get("metadata", {})
    _line(f"Model:     {md.get('model_alias', 'unknown')}")
    _line(f"Alphas:    {results.get('alphas', [])}")
    _line(f"Scenarios: {md.get('n_scenarios', '?')}")
    _line(f"Topics:    {', '.join(results.get('topics', []))}")
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

            # Length confound
            la = ja.get("length_analysis", {})
            lr = la.get("length_spearman_rho")
            if lr is not None:
                _line(
                    f"  Length correlation: rho={lr:.3f}  "
                    f"(confound: {'Yes' if la.get('length_changes_with_alpha') else 'No'})"
                )

            # Per-topic breakdown (target dim only)
            tb = ja.get("topic_breakdown", {})
            if tb and target_dims:
                primary_target = target_dims[0]
                _line(f"  Per-topic rho ({primary_target}):")
                for topic, trhos in sorted(tb.items()):
                    rho = trhos.get(primary_target)
                    if rho is not None:
                        _line(f"    {topic:<25s}  {rho:+.3f}")

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
        description="Dose-response validation of steering vectors (general topics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Generate responses (GPU)
  python dose_response_validation_general.py --stage generate --model qwen2.5-3b

  # Stage 2: Judge responses (API key)
  python dose_response_validation_general.py --stage judge --judges gemini

  # Stage 3: Analyse results (CPU)
  python dose_response_validation_general.py --stage analyze

  # All stages
  python dose_response_validation_general.py --stage all --model qwen2.5-3b --judges gemini
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
        default=None,
        help="Steering dimensions to evaluate (default: auto-detect from metadata.json)",
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
        choices=["gemini", "gpt", "llama", "ollama"],
        help="LLM judges to use for rating",
    )
    p.add_argument(
        "--thinking", action="store_true",
        help="Enable Gemini thinking mode",
    )
    p.add_argument(
        "--generations_file",
        default="results/dose_response_general_generations.json",
        help="Path to generations JSON",
    )
    p.add_argument(
        "--scores_file",
        default="results/dose_response_general_scores.json",
        help="Path to scored generations JSON",
    )
    p.add_argument(
        "--output_dir",
        default="results/dose_response_general",
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
