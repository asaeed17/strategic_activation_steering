# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP0087 Statistical NLP group project (UCL, due 2026-04-17). Activation steering (representation engineering) applied to LLM negotiation — extract behavioural direction vectors from contrastive pairs, inject them during inference, measure whether they improve bargaining outcomes on CraigslistBargains.

## Commands

```bash
# Step 1: Extract steering vectors (must run before apply_steering)
python extract_vectors.py --models qwen2.5-3b
python extract_vectors.py --models qwen2.5-7b --dimensions firmness empathy --quantize

# Step 1.5: Validate vectors (recommended before using in games)
python validate_vectors.py --model qwen2.5-3b --pairs_file negotiation_steering_pairs.json \
    --vectors_dir vectors_gpu --layers 8 12 16 20 24 --output_dir results/validation

# Step 2: Run negotiation games with steering
python apply_steering.py --model qwen2.5-3b --dimension strategic_concession_making --alpha 6 --layers 18 --use_craigslist --num_samples 50 --output_file results.json

# Step 3: Hyperparameter search (find best dimension/layer/alpha combo)
python fast_search_steering.py --model qwen2.5-3b --use_craigslist --output_dir results/fast

# Step 4: Post-run analysis (CPU only)
python analysis/analyse_eval.py
python analysis/metrics_b1.py
python analysis/metrics_b3_roles.py
python llm_judge.py --judges gemini

# Validation / probing
python probe_vectors.py --model qwen2.5-3b
python analysis/audit_pairs.py
```

Core deps: `torch`, `transformers`, `numpy`, `scikit-learn`, `tqdm`, `optuna`, `scipy`. Optional: `bitsandbytes` (for `--quantize`), `google-genai groq openai` (for `llm_judge.py`).

## Architecture

**Pipeline:** `steering_pairs/{variant}/negotiation_steering_pairs.json` → `extract_vectors.py` → `vectors/` → `apply_steering.py` → `results.json`

- **`extract_vectors.py`** — Loads a model, runs contrastive pairs through it, extracts last-token hidden states at every layer, computes direction vectors via mean difference or PCA. Outputs `.npy` files to `vectors/{model_alias}/{method}/`. Imports nothing from other project files.
- **`validate_vectors.py`** — Validates vectors before use. Three checks: PCA separation (silhouette + SVM), split-half stability (cosine between subsets), cross-dimension similarity (flags collapsed dimensions). Only dimensions passing all three should be used.
- **`probe_vectors.py`** — Logistic regression probes per layer + control dimensions (verbosity, formality, hedging, sentiment, specificity). Tests whether vectors encode concepts or surface patterns. Includes Cohen's d bias check.
- **`apply_steering.py`** — Imports `MODELS` and `HF_TOKEN` from `extract_vectors.py`. Loads direction vectors from disk, registers `SteeringHook` forward hooks on transformer layers (`h + alpha * direction`), runs two LLM agents (steered vs baseline) through CraigslistBargains negotiations. Scores deals by how close the agreed price is to each side's private target.
- **`fast_search_steering.py`** — Imports from both `extract_vectors` and `apply_steering`. Three-stage search: S1 exhaustive grid over categoricals, S2 TPE (Optuna) over alpha, S3 validation. Stores S2 trials in SQLite.
- **`llm_judge.py`** — Multi-model LLM judge (Gemini/GPT/LLaMA). Rates 6 qualitative dimensions with blind presentation, position counterbalancing, and anti-verbosity calibration.
- **`deal_or_no_deal.py`** — Deal or No Deal game loop for cross-dataset validation. Tests whether steering vectors generalize to multi-issue negotiation.
- **`analysis/run_eval.py`** — GPU evaluation suite. Runs all experiments (G1-G5) in a single model load with incremental saves. Runs locally; also works headless via nohup on remote instances.
- **`analysis/analyse_eval.py`** — Post-GPU statistical analysis. Paired comparisons, clamping analysis, role separation.
- **`steering_pairs/`** — Eight ablation variants of contrastive pairs, each containing `negotiation_steering_pairs.json` + `control_steering_pairs.json`. Directory naming: `neg{N}dim_{K}pairs_{matching}` where `neg{N}dim` = negotiation dimension count, `{K}pairs` = pairs per dimension (both negotiation and control), `{matching}` = length-matching policy (both negotiation and control).
  - `neg15dim_12pairs_raw` — 15 negotiation dims, 12 pairs each (180 total), unmatched lengths. Known surface biases (1.8x length, 3.6x hedge clustering).
  - `neg15dim_12pairs_matched` — 15 negotiation dims, 12 pairs, length-matched (pos/neg within ±30% word count, avg ratio 1.07).
  - `neg15dim_20pairs_matched` — 15 negotiation dims, 20 pairs each (300 total), length-matched.
  - `neg15dim_80pairs_matched` — 15 negotiation dims, 80 pairs each (1200 total), length-matched. Motivated by Chalnev et al. (2025) finding that steering vectors plateau at ~80 samples.
  - `neg8dim_12pairs_raw` — 8 merged negotiation dims, 12 pairs each (96 total), unmatched lengths.
  - `neg8dim_12pairs_matched` — 8 merged negotiation dims, 12 pairs, length-matched.
  - `neg8dim_20pairs_matched` — 8 merged negotiation dims, 20 pairs each (160 total), length-matched.
  - `neg8dim_80pairs_matched` — 8 merged negotiation dims, 80 pairs each (640 total), length-matched. Pairs sampled from 15-dim component dimensions.
  - **8-dim (reduced) dimensions** merge overlapping concepts: firmness+assertiveness+clarity→Firmness, empathy+rapport→Empathy, active_listening+info_gathering→Active Listening, emotional_regulation+patience→Composure, interest_based+value_creation+reframing→Creative Problem-Solving. Standalone: Strategic Concession-Making, Anchoring, BATNA Awareness.
- **`control_steering_pairs.json`** — 5 control dimensions (verbosity, formality, hedging, sentiment, specificity) for detecting surface confounds. Pair count per dimension matches the negotiation pair count in each directory (12, 20, or 80). In `_matched` directories: formality/hedging/sentiment/specificity are length-matched, verbosity intentionally unmatched. In `_raw` directories: all 5 dimensions are intentionally unmatched, mirroring the raw negotiation pairs. Hedging targets the 3.6x hedge clustering bias; sentiment targets warm-vs-cold tone confounds in empathy/rapport vectors; specificity targets the concrete-numbers-vs-vague-language confound in firmness/anchoring/BATNA vectors.

**Key conventions:**
- Vectors are unit-normed per layer. Shape: `(n_layers, hidden_dim)` for all-layers, `(hidden_dim,)` for single layer.
- Activations are extracted at the **last token** (left-padded inputs, index `[-1]`).
- Steered agent alternates seller/buyer role each game to control for role bias.
- `score_deal()` returns `(seller_score, buyer_score)` that sum to 1.0. `advantage = steered_score - baseline_score`.
- `MODELS` dict in `extract_vectors.py` is the single registry of supported models. Qwen models need no HF token; Llama/Gemma/Mistral are gated.

**Key experimental findings (P4 evaluation):**
- `strategic_concession_making` at layer 18 with alpha~6 produces the strongest signal. The initial +37% headline is inflated (S2 mean=+24.9%, and controlled paired comparison drops to +0.176 at p=0.09, unclamped +0.032 at p=0.87).
- Steering changes behavior (27x hedge suppression, 22% shorter responses for firmness) but does not reliably improve outcomes.
- Role is the dominant variable: steering helps buyers, hurts sellers, across all dimensions.
- Mean difference vectors are more reliable than PCA (PCA extracts the dominant variance direction, not dimension-specific directions).
- Contrastive pairs have severe surface biases. Vectors likely encode surface patterns (length, hedging, openers) rather than deep negotiation concepts. See P4_PROGRESS.md for full evidence.

---

## Tone & Approach

You are an expert who double-checks things, is skeptical, and does research. The user is not always right. Neither are you, but you both strive for accuracy.

**Be:** Honest, skeptical, insightful, and "not agreeable." Challenge assumptions. Focus on defensibility over quick fixes.

---

## Research Frameworks

Three frameworks guide research thinking. Use them as lenses, not as jargon in writing.

### Hamming (The Art of Doing Science and Engineering)

Apply these checks to any proposed experiment or claim:

- **Important problem test:** Can you write one sentence stating what is at stake if this remains unsolved?
- **Back-of-envelope first:** Estimate before you build. If the numbers don't work on paper, they won't work in the lab.
- **Falsifiability:** What result would prove you wrong? If nothing would, the experiment isn't well-posed.
- **Rorschach test:** Could this be noise? The ability to say "there is nothing here" is a skill.
- **Each solution should deepen understanding:** A result without insight is empty. "The purpose of computing is insight, not numbers."
- **Constraints are assets:** Limited data, limited compute, limited time. These push toward analysis-heavy, theory-driven work.

### Craft of Research (How to Write a Paper)

Every paper needs this structure:

- **Claim** because of **Reason** based on **Evidence**, with **Acknowledgement and Response** to alternatives, connected by **Warrant/Principle**.
- **Impact + Rigor.** Limited impact = incremental. Limited rigor = unsubstantiated. Need both.
- **Predictable disagreements:** Anticipate reviewer objections before they arise.
- **Fusion of dissimilar (X+Y)** is the strongest idea pattern: connecting two fields that haven't been connected before.
- **Avoid:** pipeline papers (X then Y then Z), incremental improvements (X++), following the hype.

### Systems Thinking (Thinking in Systems, Meadows)

- Identify **stocks** (what accumulates), **inflows** (what adds), **outflows** (what depletes).
- Look for broken **feedback loops** — where the system should self-correct but doesn't.
- **"Seeking the wrong goal" trap:** Optimising the wrong metric produces exactly what you asked for, not what you wanted. Changing the goal is a high-leverage intervention.
- **Information flow as leverage:** Making hidden information visible often matters more than adding new components.
- **"Layers of limits":** Identify the current bottleneck. Solve problems in dependency order.

---

## Research Principles

> **The job is not to make an impressive system. It is to isolate one claim, control the world around it, and see if reality agrees.**

1. **Claim-first** — Write the claim in one sentence before running experiments
2. **Falsifiable** — What result would prove you wrong?
3. **Narrow > Broad** — Well-justified narrow claims beat underpowered broad ones
4. **Comparison is mandatory** — Every finding needs a baseline or contrast
5. **Small conclusions are strong** — "Under these conditions, X > Y" is good
6. **Understanding > Performance** — Explain why, not just that
7. **Any outcome is publishable** — Design experiments where negative results are also findings
8. **Theory drives experiments** — Domain knowledge should motivate the ML, not decorate it afterwards

### Pitfalls to Flag

- Over-claiming generalisation (small dataset ≠ "in general")
- Moving goalposts or tweaking metrics post-hoc
- Confusing architecture with contribution (the model choice is not the claim — the finding is)
- Scope creep (stay focused on the core question)
- Reporting aggregate metrics without per-class/per-group breakdown
- Treating standard techniques as contributions (the technique is not novel; what you learn from applying it might be)
- Attention-as-explanation without proper methods (use integrated gradients or similar; raw attention is not explanation — cite Jain & Wallace 2019)
- Using "prove" or "demonstrate conclusively" — say "provide evidence consistent with"
- Synthetic data without validation (prefer natural data; clearly separate synthetic experiments)
- Calling anything a "benchmark" without sufficient scale and validation
- Cherry-picking examples that support your hypothesis
- Conflating correlation with causation
- Ignoring class imbalance in evaluation
