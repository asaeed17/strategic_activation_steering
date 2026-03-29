# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP0087 Statistical NLP group project (UCL, due 2026-04-17). Activation steering (representation engineering) applied to LLM negotiation — extract behavioural direction vectors from contrastive pairs, inject them during inference, measure whether they improve bargaining outcomes.

## Research Log

**`RESEARCH_LOG.md`** is the single source of truth for all experiments, findings, and design decisions. Use it as the primary reference when writing the paper. **Update it after every significant experiment or finding.**

When the user asks about results, hypotheses, or experimental context, consult the research log first. When new experiments complete, add the results to the log in the same format.

## Project Status (2026-03-29)

> **For detailed findings, exact numbers, and per-config results, see `RESEARCH_LOG.md`.**

**Phase 1 (complete):** CraigslistBargains. Behavioral changes real (27x hedge suppression) but outcomes not significant (p=0.87). Task too noisy. See RESEARCH_LOG Section 2.

**Phase 2 (complete):** Ultimatum Game experiments. 10,000+ games across 6 rounds + acceptance curve + analytical decomposition + text-visibility control. See RESEARCH_LOG Sections 7-11.

**Main findings (Phase 2):**
1. **Steering reliably shifts behavior** — all 24 UG configs significant (p<0.001, d=0.37-1.54), near-perfect dose-response. L14 attenuates (firmness d=0.57, empathy NS).
2. **Dimension×layer×context interaction** — firmness L10 reverses in DG (-5.8pp), firmness L12 persists (+15pp). Empathy DG pattern uncertain: null at α=7 (TOST confirmed) but large effects at other alphas under different model precision (confounded — needs replication).
3. **Empathy vector encodes activation, not valence** — both positive and negative alpha increase demand in UG. Sign modulates tone (acceptance rate), not direction. Best payoff: empathy L10 α=+7 (+17pp) in numbers-only mode.
4. **RLHF creates bidirectional fairness enforcement** — acceptance curve is non-monotonic. Rejects BOTH unfair and generous offers. Baseline sits at 50% demand — the worst spot.
5. **Framing is massively negative when text visible** — numbers-only: framing ≈ 0pp. Text-visible: acceptance crashes by 34-49pp, all payoff gains reverse. "Steering improves payoff" is bounded to numbers-only settings.
6. **General pairs >> game-specific pairs** — 16pp effect vs ~0pp for firmness at L10.
7. **Teammate's L14 peak NOT replicated** — empathy L14 d=0.14 (NS) vs their d=-3.05. Design differences explain discrepancy.

**Current status:** Paper writing. One blocking experiment remains: rerun empathy DG under consistent conditions to resolve precision confound. Teammate working on multi-turn and 32B.

**Key design choices:** Paired design, variable pools ($37-$157), temp=0, Qwen 7B, general-domain pairs, mean difference extraction, 100 games per config, BH-FDR correction. See RESEARCH_LOG Section 4 for rationale.

**Teammate divergences to track:** Different prompt ("Your aim: earn as much as you can"), rule-based responder mode, 32B model. Results not directly comparable to confirmatory — different baselines, no true pairing. See RESEARCH_LOG Section 10.

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

```bash
# Phase 2: Ultimatum Game experiments
python ultimatum_game.py --model qwen2.5-7b --dimension firmness \
    --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
    --layers 10 --alpha 7 --steered_role proposer --game ultimatum \
    --n_games 100 --variable_pools --paired --temperature 0.0 \
    --quantize --output_dir results/ultimatum/my_run

# Confirmatory analysis (CPU, reads JSON results)
python analysis/analyse_confirmatory.py \
    --results_dir results/ultimatum/confirmatory/ug \
    --dg_dir results/ultimatum/confirmatory_v2/dg \
    --empathy_alpha_list -3 -7 -10
```

Core deps: `torch`, `transformers`, `numpy`, `scikit-learn`, `tqdm`, `optuna`, `scipy`. Optional: `bitsandbytes` (for `--quantize`), `google-genai groq openai` (for `llm_judge.py`).

## Architecture

**Pipeline:** `steering_pairs/{variant}/negotiation_steering_pairs.json` → `extract_vectors.py` → `vectors/` → `apply_steering.py` → `results.json`

- **`extract_vectors.py`** — Loads a model, runs contrastive pairs through it, extracts last-token hidden states at every layer, computes direction vectors via three methods: mean difference, PCA, and logistic regression. Outputs `.npy` files to `vectors/{model_alias}/{method}/`. Imports nothing from other project files.
- **`validate_vectors.py`** — Validates vectors before use. Two modes: `--full` (requires GPU, re-extracts activations, runs Checks 1-7) and `--analyze-only` (CPU-only, reads vectors from disk, runs Checks 1, 1b, 8-12). Key checks: (1) length confound Cohen's d, (1b) vocabulary overlap Jaccard, (2) probe accuracy + permutation test, (3) cosine similarity between steering directions, (4) Cohen's d bias against all 5 control dimensions, (5) per-pair alignment consistency, (6) 1-D steering-direction probe, (7) selectivity + layer recommendations. Control dimension IDs are derived from the control pairs JSON (not hardcoded). Outputs `validation_results.json`, `validation_report.txt`, and per-dimension selectivity plots.
- **`run_extraction.sh`** — Extracts vectors for all 8 variants (hardcoded to `qwen2.5-3b`). Sets `HF_HOME=.hf_cache/`. Logs to `extraction_log.txt`.
- **`run_extraction_all_variants.sh`** — Takes model name as argument: `bash run_extraction_all_variants.sh <model>`.
- **`run_extraction_and_validation_all_variants.sh`** — Runs all three extraction methods (mean_diff, PCA, logreg) for all variants with full GPU validation.
- **`run_validation.sh`** — Runs `--full` validation for all 8 variants. Logs to `validation_log.txt`.
- **`run_all_extraction.sh`** — Calls `run_extraction.sh` then `run_validation.sh`.
- **`orthogonal_projection.py`** — Lives in `validation/`. Projects out control dimensions from negotiation vectors, measures residual norms and re-runs 1-D probes. Two phases: Phase 1 (CPU, `--variant X`) computes residual norms and cosine changes; Phase 2 (`--probe`, GPU) loads model, extracts hidden states, compares 1-D probe accuracy before/after projection. Use `--model qwen2.5-7b` for ultimatum variants (default is qwen2.5-3b). Supports `--method {mean_diff,pca,logreg}`. Also handles `ultimatum_steering_pairs.json` naming (falls back from `negotiation_steering_pairs.json`). Results in `results/projection/`.
- **`probe_vectors.py`** — Logistic regression probes per layer + control dimensions (verbosity, formality, hedging, sentiment, specificity). Tests whether vectors encode concepts or surface patterns. Includes Cohen's d bias check.
- **`apply_steering.py`** — Imports `MODELS` and `HF_TOKEN` from `extract_vectors.py`. Loads direction vectors from disk, registers `SteeringHook` forward hooks on transformer layers (`h + alpha * direction`), runs two LLM agents (steered vs baseline) through CraigslistBargains negotiations. Scores deals by how close the agreed price is to each side's private target.
- **`fast_search_steering.py`** — Imports from both `extract_vectors` and `apply_steering`. Three-stage search: S1 exhaustive grid over categoricals, S2 TPE (Optuna) over alpha, S3 validation. Stores S2 trials in SQLite. ~1.5-2h on 3B model.
- **`lightweight_gridsearch.py`** — Simplified 2-stage search for a single dimension. Sweeps layer presets (early/middle/late) then alpha. Same imports as `fast_search_steering.py`.
- **`test_steering_controlled.py`** — Properly controlled experiment: both-baseline vs one-steered. Single-role tests (buyer-only or seller-only). Imports from `extract_vectors` and `apply_steering`.
- **`llm_judge.py`** — Multi-model LLM judge (Gemini/GPT/LLaMA). Rates 6 qualitative dimensions with blind presentation, position counterbalancing, and anti-verbosity calibration. No torch dependency.
- **`deal_or_no_deal.py`** — Deal or No Deal game loop for cross-dataset validation (Lewis et al. 2017). Tests whether steering vectors generalize to multi-issue negotiation. No torch dependency.
- **`playground/run_game.py`** — Task design validation. Pluggable agents (`local:<model_key>` or `api:gemini/gpt4o/claude/groq`) with optional prompt enhancements (anchoring, strategic_concession, batna, firmness, empathy, combo). No torch dependency for API-only mode. Tests whether the negotiation task itself can discriminate agent quality.
- **`analysis/run_eval.py`** — GPU evaluation suite. Runs all experiments (G1-G5) in a single model load with incremental saves. Reads configs from `stage2_results.json` (gridsearch output). Filters dimensions by `pen_delta >= 0.15` and `dir_ok == True`. Works headless via nohup on remote instances.
- **`analysis/analyse_eval.py`** — Post-GPU statistical analysis. Paired comparisons (G1/G2), per-turn metrics (B1), role separation (B3), DonD cross-dataset (G3), firmness clamping (G4), sensitivity (G5).
- **`analysis/turn_metrics.py`** — CPU-only per-turn behavior analysis: offer trajectories, concession rates, hedge rates, response lengths. Output: `turn_metrics_enriched.json`.
- **`analysis/role_analysis.py`** — Role-separated outcome tables. Key insight: never report aggregate alone; steering helps buyers, hurts sellers.
- **`analysis/analyse_results.py`** — Paired steering effect analysis with per-turn behavioral metrics.
- **`analysis/audit_pairs.py`** — Surface bias audit for contrastive pairs. 10 checks (token count, hedging, sentiment, etc.) with Cohen's d effect sizes.
- **`analysis/phase_a_diagnostic.py`** — CraigslistBargains data audit: overlapping targets, bad prices, category distribution. CPU-only.
- **`analysis/metrics_b2_decay.py`** — Steering decay analysis: do effects fade over negotiation course? (Literature: ~300-500 tokens.)
- **`analysis/read_db.py`** — Quick utility to read Optuna Stage 2 TPE trial results from SQLite.
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
  - `ultimatum_10dim_20pairs_matched` — 10 ultimatum game dims, 20 pairs each (200 total), game-specific language (OFFER=X,Y, ACCEPT/REJECT, proposer/responder contexts), length-matched. Dims: firmness, empathy, anchoring, batna_awareness, composure, fairness_norm, flattery, narcissism, spite, undecidedness. Added PR #34.
  - `ultimatum_10dim_20pairs_general_matched` — 10 general-domain dims, 20 pairs each (200 total), diverse contexts (salary, business, real estate, etc.), NO game-specific tokens. Same dims but batna_awareness replaced by greed. **v2.1: all negotiation pairs length-matched** (132 rewritten, ±30% word count, mean ratios 0.94–1.11). Control pairs also length-matched. Added PR #36; pairs fixed on `abdullah-control-general-ultimatum-vectors` branch.
- **`control_steering_pairs.json`** — 5 control dimensions (verbosity, formality, hedging, sentiment, specificity) for detecting surface confounds. Pair count per dimension matches the negotiation pair count in each directory (12, 20, or 80). In `_matched` directories: formality/hedging/sentiment/specificity are length-matched, verbosity intentionally unmatched. In `_raw` directories: all 5 dimensions are intentionally unmatched, mirroring the raw negotiation pairs. Hedging targets the 3.6x hedge clustering bias; sentiment targets warm-vs-cold tone confounds in empathy/rapport vectors; specificity targets the concrete-numbers-vs-vague-language confound in firmness/anchoring/BATNA vectors. The `ultimatum_10dim_20pairs_general_matched` control pairs use general-domain contexts matching the negotiation pairs.

**Key directories:**

- `vectors/{variant}/negotiation/` and `vectors/{variant}/control/` — Extracted `.npy` vectors per variant, with subdirectories per method (`mean_diff/`, `pca/`, `logreg/`).
- `results/validation/{variant}/qwen2.5-3b/{method}/` — Validation reports, JSON results, and plots per variant and extraction method.
- `results/projection/{variant}/{method}/` — Orthogonal projection results per variant and method.
- `results/validation/VALIDATION_RESULTS.md` — Comprehensive cross-variant analysis (mean difference method).
- `results/validation/PCA_VALIDATION_RESULTS.md` — PCA-specific cross-variant analysis + head-to-head comparison with mean difference.
- `results/validation/LOGREG_VALIDATION_RESULTS.md` — Logistic regression cross-variant analysis + three-method comparison (MD vs PCA vs LR).
- `.hf_cache/` — HuggingFace model cache (redirected from `~/.cache/huggingface` via `HF_HOME` to avoid home dir quota limits on UCL machines).
- `playground/results/` — Task design validation experiment outputs.
- `results/eval/` — `run_eval.py` outputs per dimension.
- `results/ultimatum/` — Ultimatum game gridsearch results. `temp03_mindims_v4/L{10,14}/` has per-dimension per-role results with game-specific pairs on Qwen 7B.
- `FINAL_VALIDATION_RESULTS/ultimatum_10dim_20pairs_general_matched/` — Validation results for general-domain ultimatum pairs (v2.1: score 27/100, 8 RED / 2 AMBER).
- `results/projection/ultimatum_10dim_20pairs_general_matched/mean_diff/` — Orthogonal projection results (8/10 GENUINE, 1 PARTIAL, 1 SURFACE-DOMINATED).
- `validation/validate_vectors.py` — Note: validation script lives in `validation/` subdirectory, not project root. Run with `PYTHONPATH=. python validation/validate_vectors.py`.

**Import dependency graph:**

```
extract_vectors.py  ← standalone, no internal imports
  ↓ exports: MODELS, HF_TOKEN
  ├── apply_steering.py
  │     ↓ exports: load_craigslist, load_direction_vectors, run_game, score_deal, summarise
  │     ├── fast_search_steering.py
  │     ├── lightweight_gridsearch.py
  │     └── test_steering_controlled.py
  ├── validate_vectors.py
  ├── orthogonal_projection.py
  └── probe_vectors.py

playground/run_game.py  ← inlined load_craigslist(); no torch for API-only mode
analysis/*.py           ← all CPU-only, read JSON outputs, no internal imports
```

**Key conventions:**

- Vectors are unit-normed per layer. Shape: `(n_layers, hidden_dim)` for all-layers, `(hidden_dim,)` for single layer.
- Activations are extracted at the **last token** (left-padded inputs, index `[-1]`).
- Steered agent alternates seller/buyer role each game to control for role bias.
- `score_deal()` returns `(seller_score, buyer_score)` that sum to 1.0. `advantage = steered_score - baseline_score`.
- `MODELS` dict in `extract_vectors.py` is the single registry of supported models. Qwen models need no HF token; Llama/Gemma/Mistral are gated.

**Key decisions (see RESEARCH_LOG for full rationale):**
- Mean difference extraction (MD >= LR >> PCA). General-domain pairs for Phase 2.
- Role-separated reporting mandatory. Report both demand shift and payoff.
- Qwen 7B primary (3B for Phase 1). Llama 3B produced faulty results.
- Layer location is mechanistic: L10 = style/tone, L12 = reasoning, L14+ = inactive.
- UG chosen over Split-or-Steal (continuous outcome, dose-response possible).
- Variable pools + temp=0 + paired design = honest n=100 per config.

---

## Tone & Approach

You are an expert who double-checks things, is skeptical, and does research. The user is not always right. Neither are you, but you both strive for accuracy.

**Be:** Honest, skeptical, insightful, and "not agreeable." Challenge assumptions. Focus on defensibility over quick fixes.

---

## Research Principles

> **The job is not to make an impressive system. It is to isolate one claim, control the world around it, and see if reality agrees.**

- **Claim-first, falsifiable.** Write the claim before running experiments. What result would prove you wrong?
- **Narrow > Broad.** "Under these conditions, X > Y" beats underpowered broad claims.
- **Understanding > Performance.** Explain why, not just that. Any outcome is publishable if designed right.
- **Pitfalls:** Over-claiming generalisation, moving goalposts post-hoc, cherry-picking, conflating correlation with causation, reporting aggregates without per-group breakdown, scope creep.
