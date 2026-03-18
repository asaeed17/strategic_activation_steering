# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP0087 Statistical NLP group project (UCL, due 2026-04-17). Activation steering (representation engineering) applied to LLM negotiation — extract behavioural direction vectors from contrastive pairs, inject them during inference, measure whether they improve bargaining outcomes on CraigslistBargains.

## Commands

```bash
# Step 1: Extract steering vectors for all 8 variants (negotiation + control)
bash run_extraction.sh          # logs to extraction_log.txt
# Or for a single variant:
python extract_vectors.py --models qwen2.5-3b \
    --pairs_file steering_pairs/neg8dim_12pairs_matched/negotiation_steering_pairs.json \
    --output_dir vectors/neg8dim_12pairs_matched/negotiation

# Step 1.5: Validate vectors (requires GPU, --full re-extracts activations)
bash run_validation.sh          # logs to validation_log.txt
# Or for a single variant:
python validate_vectors.py --model qwen2.5-3b --full \
    --negotiation_pairs steering_pairs/neg8dim_12pairs_matched/negotiation_steering_pairs.json \
    --control_pairs steering_pairs/neg8dim_12pairs_matched/control_steering_pairs.json \
    --vectors_dir vectors/neg8dim_12pairs_matched/negotiation \
    --output_dir results/validation/neg8dim_12pairs_matched

# Steps 1+1.5 combined:
bash run_all_extraction.sh      # runs extraction then validation

# Step 2: Run negotiation games with steering
python apply_steering.py --model qwen2.5-3b --dimension strategic_concession_making --alpha 6 --layers 18 --use_craigslist --num_samples 50 --output_file results.json

# Step 3: Hyperparameter search (find best dimension/layer/alpha combo)
python fast_search_steering.py --model qwen2.5-3b --use_craigslist --output_dir results/fast

# Step 4: Post-run analysis (CPU only)
# Run experiments first (GPU):
python analysis/run_eval.py --model qwen2.5-3b --experiments scm_craigslist
# Then analyse (CPU):
python analysis/turn_metrics.py        # per-turn behavior enrichment (→ turn_metrics_enriched.json)
python analysis/role_analysis.py       # role-separated tables (reads turn_metrics_enriched.json)
python analysis/analyse_results.py     # paired statistical analysis
python llm_judge.py --judges gemini    # qualitative judge

# Validation / probing
python probe_vectors.py --model qwen2.5-3b
python analysis/audit_pairs.py
```

Core deps: `torch`, `transformers`, `numpy`, `scikit-learn`, `tqdm`, `optuna`, `scipy`. Optional: `bitsandbytes` (for `--quantize`), `google-genai groq openai` (for `llm_judge.py`).

## Architecture

**Pipeline:** `steering_pairs/{variant}/negotiation_steering_pairs.json` → `extract_vectors.py` → `vectors/` → `apply_steering.py` → `results.json`

- **`extract_vectors.py`** — Loads a model, runs contrastive pairs through it, extracts last-token hidden states at every layer, computes direction vectors via three methods: mean difference, PCA, and logistic regression. Outputs `.npy` files to `vectors/{model_alias}/{method}/`. Imports nothing from other project files.
- **`validate_vectors.py`** — Validates vectors before use. Two modes: `--full` (requires GPU, re-extracts activations, runs Checks 1-7) and `--analyze-only` (CPU-only, reads vectors from disk, runs Checks 1, 1b, 8-12). Key checks: (1) length confound Cohen's d, (1b) vocabulary overlap Jaccard, (2) probe accuracy + permutation test, (3) cosine similarity between steering directions, (4) Cohen's d bias against all 5 control dimensions, (5) per-pair alignment consistency, (6) 1-D steering-direction probe, (7) selectivity + layer recommendations. Control dimension IDs are derived from the control pairs JSON (not hardcoded). Outputs `validation_results.json`, `validation_report.txt`, and per-dimension selectivity plots.
- **`run_extraction.sh`** — Extracts negotiation + control vectors for all 8 steering pair variants. Logs to `extraction_log.txt`.
- **`run_validation.sh`** — Runs `--full` validation for all 8 variants. Logs to `validation_log.txt`.
- **`run_all_extraction.sh`** — Calls `run_extraction.sh` then `run_validation.sh`.
- **`orthogonal_projection.py`** — Projects out control dimensions from negotiation vectors, measures residual norms and re-runs 1-D probes. Two phases: Phase 1 (CPU, `--all-variants`) computes residual norms and cosine changes; Phase 2 (`--probe`, GPU) loads model, extracts hidden states, compares 1-D probe accuracy before/after projection. Results in `results/projection/`.
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

**Key directories:**
- `vectors/{variant}/negotiation/` and `vectors/{variant}/control/` — Extracted `.npy` vectors per variant, with subdirectories per method (`mean_diff/`, `pca/`, `logreg/`).
- `results/validation/{variant}/qwen2.5-3b/{method}/` — Validation reports, JSON results, and plots per variant and extraction method.
- `results/projection/{variant}/{method}/` — Orthogonal projection results per variant and method.
- `results/validation/VALIDATION_RESULTS.md` — Comprehensive cross-variant analysis (mean difference method).
- `results/validation/PCA_VALIDATION_RESULTS.md` — PCA-specific cross-variant analysis + head-to-head comparison with mean difference.
- `results/validation/LOGREG_VALIDATION_RESULTS.md` — Logistic regression cross-variant analysis + three-method comparison (MD vs PCA vs LR).
- `.hf_cache/` — HuggingFace model cache (redirected from `~/.cache/huggingface` via `HF_HOME` to avoid home dir quota limits on UCL machines).

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
- Mean difference vectors are strictly superior to PCA: 1-D probe accuracy 0.903 vs 0.586 (+54%) in best variant, while validity scores and orthogonal projection robustness are identical. PCA's PC1 captures noise/confound variance rather than concept direction at these sample sizes (12-80 pairs in 2560-dim space). See `results/validation/PCA_VALIDATION_RESULTS.md`. Logistic regression closely matches MD: 1-D probe (held-out) 0.891 vs MD's 0.903, with identical validity scores (33/100) and similar projection drops (2.8% vs 2.4%). LR's near-perfect training-data 1-D accuracy (0.998) is tautological — the weight vector IS the separator. See `results/validation/LOGREG_VALIDATION_RESULTS.md`. All three methods agree on traffic lights, scores, and projection robustness, confirming directions are genuine rather than method artifacts.
- Contrastive pairs have severe surface biases. Vectors likely encode surface patterns (length, hedging, openers) rather than deep negotiation concepts. See P4_PROGRESS.md for full evidence.

**Validation ablation findings (8-variant study, see `results/validation/VALIDATION_RESULTS.md`, `PCA_VALIDATION_RESULTS.md`, and `LOGREG_VALIDATION_RESULTS.md`):**
- **Best variant: `neg8dim_12pairs_matched`** (33/100, 1/8 negotiation flat-high probes, 5/8 AMBER). Zero GREEN dimensions in any variant.
- **Length matching is the only effective intervention:** raw→matched = +10-13 points, 75-80% reduction in flat-high probes.
- **Pair scaling hurts:** 12→20→80 pairs worsens scores (33→28→26 for 8dim). More pairs amplifies surface confounds; per-pair alignment degrades (empathy: 0.464→0.343 at 80 pairs with 30/80 outliers). Contradicts naive extrapolation from Chalnev et al. (2025).
- **Dimension merging helps modestly:** 15→8 dims improves best score from 26→33 by reducing concept overlap.
- **All negotiation×control Cohen's d pairs are SEVERE** in every variant. `cos(firmness, hedging) = -0.703`. But see orthogonal projection results below.
- **No variant has recommended layers** (criteria: acc≥0.85 AND |d|≤0.8) except active_listening at 4 layers in neg8dim_80pairs.
- **Selectivity metric is flawed:** penalty term caps at 0.5, so near-perfect probe accuracy at 80 pairs inflates selectivity even though vectors are more confounded.

**Orthogonal projection findings (`orthogonal_projection.py`, `results/projection/`):**
- **Cohen's d overstates contamination.** After projecting out all 5 control dimensions from negotiation vectors and re-running 1-D probes, average accuracy drops only 2.4% (0.843→0.820) for mean difference. 7/8 dimensions retain signal; 2 dimensions actually improve.
- **Result is robust across all 8 variants.** 84/92 dimension×variant tests are GENUINE (91%), 12 PARTIAL SURFACE, 6 IMPROVED. Average drop ranges 1.1-3.7% across variants.
- **Result is robust across all three extraction methods.** Projection drops: MD 2.4%, PCA 2.1%, LR 2.8% in best variant. All three methods extract directions with similar surface overlap, confirming the finding is about the data geometry, not the extraction algorithm.
- **Firmness retains 96.6% of its probe accuracy** despite cos=-0.703 with hedging. The surface overlap was real but irrelevant to the separation signal.
- **Empathy has the largest surface dependence** (6.9% drop for MD, 6.8% for PCA, 3.6% for LR), consistent with its sentiment overlap. Still well above chance. LR's lower empathy drop is offset by higher firmness drop (7.2% vs MD's 3.4%), suggesting LR's max-margin direction "fudges" toward hedging/formality confounds in firmness.
- **clarity_and_directness is the only consistently surface-dependent dimension** (6.3% mean drop, PARTIAL in 3/4 variants). Its meaning genuinely overlaps with hedging and specificity.
- **batna_awareness and reframing are the purest concepts** — cleaning has no effect or improves accuracy across all variants.
- **Interpretation:** The data (pairs) is confounded in surface features, but the extracted steering directions are mostly genuine — they capture conceptual variance beyond surface features. Cohen's d detects data confounds, not direction confounds.

**Extraction method rationale (3 methods: mean_diff, PCA, logreg):**
- **Mean difference** (generative): `mean(pos) - mean(neg)`. Standard CAA (Panickssery et al. 2024). Im & Li 2025 prove this is optimal under pointwise loss.
- **PCA** (variance-based): PC1 of difference vectors. Follows Zou et al. 2023 (RepE). Finds dominant axis of variation; can diverge from concept direction if noise variance dominates.
- **Logistic regression** (discriminative): L2-regularised LR weight vector. Follows Li et al. 2024 (ITI), Zou et al. 2023 (classifier variant). Finds max-margin separation boundary. Known to "fudge" direction when surface confounds are non-orthogonal (Marks & Tegmark 2023). Empirically confirmed: LR's firmness drops 7.2% on projection vs MD's 3.4%, while 1-D held-out probe accuracy is within 1.2% of MD (0.891 vs 0.903). Literature consensus confirmed: MD ≥ LR >> PCA for steering (Im & Li 2025).
- **K-means rejected:** With balanced classes, k-means centroids ≈ class means, so `centroid_1 - centroid_0 ≈ mean_diff`. When clusters don't recover classes, it's worse — Euclidean distance in full activation space is dominated by highest-variance (surface confound) directions. Adds no new lens; conceptually redundant with mean diff.
- **Why run all three (triangulation value):** The methods have different inductive biases (generative vs variance-based vs discriminative), so their agreement/disagreement is informative. Empirically: identical validity scores (33/100), identical traffic lights, and similar projection drops (2.1-2.8%) across all three methods prove data quality is the bottleneck, not algorithmic choice. Disagreements reveal method-specific weaknesses: LR's 7.2% firmness projection drop (vs MD's 3.4%) exposes discriminative confound exploitation; PCA's 0.586 mean 1-D probe accuracy (vs MD's 0.903) exposes noise-sensitivity at small sample sizes. A reviewer cannot dismiss findings as method-dependent when three algorithmically distinct approaches converge.

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
