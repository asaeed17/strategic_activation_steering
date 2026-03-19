# Activation Steering for LLM Negotiation

COMP0087 Statistical NLP Group Project, UCL (due 2026-04-17).

Extract behavioural steering vectors from contrastive pairs, inject them into LLMs during inference, and measure whether they change negotiation behaviour and outcomes. Evaluated on CraigslistBargains (price negotiation) and Deal or No Deal (multi-issue negotiation).

## Project Structure

```
.
├── extract_vectors.py          # Step 1: extract steering vectors from contrastive pairs
├── apply_steering.py           # Step 2: run steered negotiation games (CraigslistBargains)
├── fast_search_steering.py     # Step 3: hyperparameter search (dimension/layer/alpha)
├── negotiation_steering_pairs.json  # 180 contrastive pairs across 15 negotiation dimensions
├── control_steering_pairs.json      # 12 control pairs (verbosity + formality)
│
├── validate_vectors.py         # Vector validation: PCA separation, split-half, cross-dim similarity
├── probe_vectors.py            # Logistic regression probes + control dimension bias checks
├── llm_judge.py                # LLM judge evaluation (Gemini/GPT/LLaMA)
├── deal_or_no_deal.py          # Deal or No Deal game loop for cross-dataset testing
│
├── analysis/                   # Post-GPU analysis scripts
│   ├── run_eval.py             # GPU evaluation suite (runs all experiments in one model load)
│   ├── analyse_eval.py         # Post-GPU statistical analysis
│   ├── audit_pairs.py          # Contrastive pair surface bias audit
│   ├── metrics_b1.py           # Per-turn behavioral metrics (hedges, length, concessions)
│   ├── metrics_b2_decay.py     # Steering decay over turns
│   ├── metrics_b3_roles.py     # Role-separated analysis (seller vs buyer)
│   ├── phase_a_diagnostic.py   # Foundation diagnostics (data audit, clamping)
│   └── read_db.py              # Inspect Optuna search database
│
├── vectors_gpu/                # Extracted steering vectors (.npy)
├── results/                    # Experiment results (.json)
│   ├── eval/                   # Controlled paired evaluation (G1-G5)
│   └── fast_run/               # Hyperparameter search outputs
├── probe_results/              # Logistic regression probe results
├── data/                       # Datasets (Deal or No Deal)
│
├── docs/                       # Background documents
│
└── P4_PROGRESS.md              # P4 evaluation findings and analysis log
```

## Setup

```bash
pip install torch transformers numpy scikit-learn tqdm optuna scipy
# Optional:
pip install bitsandbytes   # for --quantize (4-bit inference)
pip install google-genai groq openai  # for llm_judge.py
```

Models: Qwen 2.5-3B and 7B (no HuggingFace token needed). Llama/Gemma/Mistral are gated (set `HF_TOKEN` env var).

## Pipeline

The full pipeline has four stages. Each stage depends on the outputs of the previous one.

### 1. Extract Steering Vectors

```bash
python extract_vectors.py --models qwen2.5-3b
```

Takes contrastive pairs from `negotiation_steering_pairs.json`, runs them through the model, and computes direction vectors at every layer via mean difference and PCA.

Output: `vectors/{model_alias}/{method}/{dimension}_all_layers.npy` — shape `(n_layers, hidden_dim)`.

Options:

- `--models qwen2.5-7b llama3-8b` — multiple models
- `--dimensions firmness empathy` — subset of dimensions
- `--quantize` — 4-bit quantisation to save VRAM
- `--layers 8 12 16 20` — save only specific layers

### 2. Validate Vectors (recommended)

```bash
python validate_vectors.py \
    --model qwen2.5-3b \
    --pairs_file negotiation_steering_pairs.json \
    --vectors_dir vectors_gpu \
    --layers 8 12 16 20 24 \
    --output_dir results/validation
```

Three checks gate whether a vector should be used in experiments:

- **PCA Separation** — silhouette score > 0.3 AND SVM accuracy > 0.7
- **Split-Half Stability** — cosine between first-half and second-half vectors > 0.8
- **Cross-Dimension Similarity** — flags dimension pairs with cosine > 0.5

Only dimensions that pass all three checks should be used for steering.

### 3. Run Negotiation Games

```bash
# Single config
python apply_steering.py \
    --model qwen2.5-3b \
    --dimension strategic_concession_making \
    --alpha 6 \
    --layers 18 \
    --use_craigslist \
    --num_samples 50 \
    --output_file results.json

# Hyperparameter search (finds best dimension/layer/alpha)
python fast_search_steering.py \
    --model qwen2.5-3b \
    --use_craigslist \
    --output_dir results/fast
```

The steered agent alternates seller/buyer roles each game to control for role asymmetry. `advantage = mean(steered_score) - mean(baseline_score)`.

### 4. Analyse Results

Requires a local GPU (the 3B model fits on a single GPU with ~8GB VRAM; use `--quantize` during extraction if tight on memory). For headless/remote runs, wrap with `nohup`.

```bash
# Full evaluation suite (local GPU, runs all experiments in one model load)
python analysis/run_eval.py --model qwen2.5-3b --all

# If running headless (e.g. remote instance):
nohup python analysis/run_eval.py --model qwen2.5-3b --all 2>&1 | tee results/eval/run.log &

# Post-GPU analysis (CPU only, no GPU needed)
python analysis/analyse_eval.py

# Behavioral metrics on game transcripts (CPU)
python analysis/metrics_b1.py
python analysis/metrics_b3_roles.py

# LLM judge (CPU, needs API keys: GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY)
python llm_judge.py --judges gemini
```

## Key Findings So Far

These findings come from P4 evaluation on Qwen 2.5-3B/7B. See `P4_PROGRESS.md` for full evidence and methodology.

### What works

- **Steering changes model behavior.** Firmness at alpha=20 suppresses hedge words by 27x and shortens responses by 22%. SCM at alpha=6 hardens concessions and makes responses 13% longer. These are real, measurable, and qualitatively different effects.
- **Dose-response is real.** For SCM at middle layers, higher alpha produces higher advantage — the best signal that a steering effect is genuine rather than noise.
- **No steering decay.** Our 8-turn negotiations are too short for the 300-500 token decay threshold.

### What doesn't work

- **Outcome advantages are clamping artifacts.** Stripping games where the agreed price falls outside the target range reduces the advantage to near zero (SCM unclamped paired: +0.032, p=0.87).
- **Role is the dominant variable.** Steering helps buyers and hurts sellers across all dimensions. Firmness: seller -27%, buyer +24%. Aggregate scores mask this.
- **Vectors encode surface patterns from biased pairs.** The contrastive pairs have 1.8x length bias, 3.6x hedge clustering in negatives, zero opener overlap. The firmness vector's primary effect is "stop saying maybe/perhaps" — the exact surface pattern the pairs encode.
- **SCM does not generalize to multi-issue negotiation.** Deal or No Deal Pareto rate: 16% (barely above naive 9%, far below human 77%).

### Probes confirm surface confounding (Damon's validation)

Logistic regression probes achieve 1.0 accuracy on verbosity across all layers — the model trivially separates verbose vs concise text. Cohen's d bias between firmness vectors and verbosity ranges from 2.7 to 47.7 across layers, confirming that steering vectors are significantly confounded with surface features like response length.

### Paper framing

The contribution is the **evaluation framework** itself — a methodology that systematically identifies confounds (role effects, clamping, surface pattern encoding, aggregate masking) that inflate apparent steering effects. This is a "how to properly evaluate steering in applied domains" paper.

## Task Ownership

| Task | Owner | Description                                                               | Status      |
| ---- | ----- | ------------------------------------------------------------------------- | ----------- |
| P1   | —     | Improved contrastive pairs (fix surface biases, 80+ pairs/dim)            | TODO        |
| P2   | Sonny | Probing & interpretability (extend Damon's probe work)                    | IN PROGRESS |
| P3   | Damon | Hyperparameter search (fast_search_steering.py)                           | DONE        |
| P4   | Moiz  | Evaluation framework + results & analysis                                 | DONE        |
| P5   | —     | Paper writing                                                             | TODO        |
| P6   | —     | Negative alpha experiments (reverse steering to confirm causal direction) | TODO        |

## Guidance for Next Tasks

Based on P4 findings, here is what matters most for each remaining task.

### P1: Better Contrastive Pairs (highest priority)

The current 180 pairs have severe surface biases that make it impossible to tell whether vectors encode concepts or artifacts. The P4 evaluation cannot produce clean results until the pairs are fixed.

**What to fix:**

- **Length matching.** Current pairs have 1.8x positive/negative length ratio. Match lengths within 10% per pair.
- **Opener diversity.** 35/180 negatives start with "Okay/Sure/Fine", zero positives do. Add these openers to positives too.
- **Hedge balancing.** Hedges are 3.6x more common in negatives. Include hedge phrases in some positives.
- **80+ pairs per dimension.** Current: 12 per dimension. Literature (Hao et al. 2025): diminishing returns at ~80.
- **Validate before using.** Run `validate_vectors.py` on new pairs. All dimensions should pass PCA separation (silhouette > 0.3, SVM > 0.7) and split-half stability (cosine > 0.8).

**How to validate new pairs:**

```bash
# 1. Extract vectors with new pairs
python extract_vectors.py --models qwen2.5-3b

# 2. Validate
python validate_vectors.py \
    --model qwen2.5-3b \
    --pairs_file YOUR_NEW_PAIRS.json \
    --vectors_dir vectors \
    --layers 8 12 16 20 24 \
    --output_dir results/validation

# 3. Check results — only use dimensions that PASS all three checks
cat results/validation/validation_results.json
```

**Surface bias audit on existing pairs:**

```bash
python analysis/audit_pairs.py
```

### P2: Probes & Interpretability

Damon's probe work (`probe_vectors.py`) is a good start. Key extension needed:

- **Control dimensions are essential.** The verbosity control probe scores 1.0 everywhere — the model trivially separates by length. This means probe accuracy alone doesn't prove concept encoding. Always include verbosity and formality as controls.
- **Cohen's d bias check.** Already implemented in Damon's code. The firmness-vs-verbosity bias (2.7 to 47.7) confirms surface confounding. Late layers (28+) have explosive confounding — this explains why middle layers (16-20) work best for steering.
- **Extend with:** permutation tests (Sonny's `validate_vectors.py` on the `validation` branch has this), per-pair consistency (which individual pairs drive the direction?), length-controlled probes (residualise out length before probing).

### P5: Paper Writing

See `P4_PROGRESS.md` sections C3 (synthesis) and D (paper sections). The evidence structure is:

1. **Behavioral changes are real** (B1 metrics: hedge suppression, length changes, concession patterns)
2. **Outcome improvements are not real** (paired comparison p=0.09, unclamped p=0.87)
3. **Why not** — three confounds: role asymmetry, score clamping, surface pattern encoding
4. **Cross-dataset check confirms** — SCM fails on Deal or No Deal (Pareto 16%)
5. **Contribution** — the evaluation methodology, not the steering technique

### P6: Negative Alpha Experiments

Reverse the steering direction (alpha < 0) to confirm causal control. If negative alpha increases hedges and softens concessions, the vector has a genuine directional effect (even if outcomes don't change). This is a strong experiment because it's the cleanest test of whether the vector captures a meaningful direction vs random noise.

```bash
python apply_steering.py \
    --model qwen2.5-3b \
    --dimension firmness \
    --alpha -20 \
    --layers 16 \
    --use_craigslist \
    --num_samples 50 \
    --output_file results/negative_alpha_firmness.json
```

## Technical Conventions

- Vectors are **unit-normed** per layer. Shape: `(n_layers, hidden_dim)` for all-layers, `(hidden_dim,)` for single layer.
- Activations are extracted at the **last token** (left-padded inputs, index `[-1]`).
- `MODELS` dict in `extract_vectors.py` is the single registry of supported models.
- `score_deal()` returns `(seller_score, buyer_score)` summing to 1.0. `advantage = steered - baseline`.
- Steered agent alternates seller/buyer roles each game.
- Mean difference vectors are generally more reliable than PCA (PCA tends to extract the dominant variance direction rather than dimension-specific directions).

## Key Literature

| Paper                         | Relevance                                                                  |
| ----------------------------- | -------------------------------------------------------------------------- |
| Panickssery et al. 2024 (CAA) | Steering magnitudes degrade text quality                                   |
| Hao et al. 2025 (ICLR WS)     | ~80 pairs for diminishing returns; CAE only reliable in-distribution       |
| Tan et al. 2024 (NeurIPS)     | Anti-steerability (~50% opposite); steerability bias from surface patterns |
| He et al. 2018                | CraigslistBargains dataset                                                 |
| Lewis et al. 2017             | Deal or No Deal dataset                                                    |
| Chandra et al. 2021           | CaSiNo dataset (strategy-annotated dialogues)                              |
