# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP0087 Statistical NLP group project (UCL, due 2026-04-17). Activation steering (representation engineering) applied to LLM negotiation — extract behavioural direction vectors from contrastive pairs, inject them during inference, measure whether they improve bargaining outcomes.

## Project Status (2026-03-27)

**Phase 1 (complete):** Vector extraction, validation, and ablation study across 8 steering pair variants, 3 extraction methods, and orthogonal projection analysis. Vectors successfully capture behavioral dimensions (linear probes confirm). Infrastructure is solid.

**Phase 1 finding (the bottleneck):** Steering has negligible/zero average effect on LLM-vs-LLM CraigslistBargains outcomes. Extensive grid searches on Qwen 3B and 7B confirm this. The dataset/task is the primary bottleneck, not the vectors. LLM-vs-LLM negotiation has too much noise/variance for steering effects to surface reliably.

**Phase 2 (in progress — THE PIVOT):** Redesigning the evaluation task. Ultimatum Game is the primary alternative task.
- **Ultimatum Game** (merged to main via PRs #34, #35, #36): `ultimatum_game.py` with paired design (steered vs baseline on same offers). Two steering pair variants:
  - `ultimatum_10dim_20pairs_matched` — Game-specific pairs (OFFER=X,Y / ACCEPT / REJECT language). 10 dims: firmness, empathy, anchoring, batna_awareness, composure, fairness_norm, flattery, narcissism, spite, undecidedness. Vectors extracted for Qwen 7B.
  - `ultimatum_10dim_20pairs_general_matched` — General-domain pairs (salary, business, real estate contexts, no game tokens). Same 10 dims but batna_awareness replaced by greed. Vectors extracted for Qwen 7B. **v2.1: all 200 negotiation pairs length-matched** (132 pairs rewritten, all within ±30% word count). Control pairs also length-matched. Prior validation (v2.0, pre-fix): 29/100 (all RED) due to length confounds. **Needs re-extraction and re-validation on v2.1 pairs.**
- **Preset negotiation scripts** (merged via PR #32): 40 fixed 4-turn scripts. Grid search results in `hyperparameter_results/`.
- **Playground task validation** (`playground/run_game.py`): Pluggable agents (local HF or API) with prompt enhancements.

**Ultimatum Game results (game-specific pairs, Qwen 7B, L10, 50 paired games):**
- **Proposer steering:** Weak effects. Empathy (alpha=-5): offer drops from 65.4% to 56.0% (p<0.001, d=-0.81). Fairness_norm (alpha=5): offer drops to 56.9% (p<0.001, d=-1.00). Narcissism (alpha=15): offer rises to 70.4% (p<0.001, d=+0.92). Firmness: zero effect.
- **Responder steering:** Dramatic but likely model-breaking. Firmness/spite at alpha=15: acceptance drops from 96% to 8% (p<0.001). Fairness_norm: 92%→12%. These are statistically significant but may reflect incoherent output rather than genuine behavioral steering.
- **Key concern:** Multiple comparisons problem (160 tests without correction). Best alpha selected on same data used for significance testing (double-dipping).

**General pairs validation (2026-03-27, v2.0 pairs — pre-length-matching):**
- Score: 29/100 (all 10 dims RED). Problems: (1) negotiation pairs not length-matched (anchoring 2.33x, flattery 2.04x, undecidedness 2.27x), (2) all probes flat-high at all layers (surface separation), (3) all 50 neg×control Cohen's d pairs SEVERE, (4) zero recommended layers.
- Per-pair alignment is decent (flattery 0.700, anchoring 0.694, undecidedness 0.740) — conceptual contrasts are internally consistent, but directions encode surface features.
- Key cosine overlaps: hedging↔undecidedness=0.710 (undecidedness IS hedging), firmness↔hedging=-0.574 (firmness = anti-hedging), composure↔specificity=0.639.
- **v2.1 fix (2026-03-27):** All 200 negotiation pairs rewritten to ±30% word-count ratio. 132 pairs modified: expanded short negatives (anchoring, undecidedness, flattery, narcissism, empathy) with contextually appropriate detail that does NOT introduce the target trait; expanded short firmness positives with firm reasoning. Mean ratios now 0.94–1.11 across all 10 dims. **Vectors must be re-extracted and re-validated on v2.1 pairs.**

**Current branch:** `main`.

**Next steps:** ~~Length-match the general ultimatum pairs~~ (done, v2.1). Re-extract vectors and re-validate general pairs on v2.1. Run orthogonal projection on both ultimatum variants. Run final steering experiments on whichever variant validates best. Paper writing. Models: Qwen 7B primary (Llama 3-8B extraction in progress).

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
- **`run_extraction.sh`** — Extracts vectors for all 8 variants (hardcoded to `qwen2.5-3b`). Sets `HF_HOME=.hf_cache/`. Logs to `extraction_log.txt`.
- **`run_extraction_all_variants.sh`** — Takes model name as argument: `bash run_extraction_all_variants.sh <model>`.
- **`run_extraction_and_validation_all_variants.sh`** — Runs all three extraction methods (mean_diff, PCA, logreg) for all variants with full GPU validation.
- **`run_validation.sh`** — Runs `--full` validation for all 8 variants. Logs to `validation_log.txt`.
- **`run_all_extraction.sh`** — Calls `run_extraction.sh` then `run_validation.sh`.
- **`orthogonal_projection.py`** — Projects out control dimensions from negotiation vectors, measures residual norms and re-runs 1-D probes. Two phases: Phase 1 (CPU, `--all-variants`) computes residual norms and cosine changes; Phase 2 (`--probe`, GPU) loads model, extracts hidden states, compares 1-D probe accuracy before/after projection. Results in `results/projection/`.
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
- `FINAL_VALIDATION_RESULTS/ultimatum_10dim_20pairs_general_matched/` — Validation results for general-domain ultimatum pairs (score 29/100).
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

**Concrete results (SCM on CraigslistBargains, Qwen 2.5-3B, 50 paired games):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Paired effect (all games) | +0.176, p=0.09, d=0.24 | Not significant at p<0.05 |
| Paired effect (unclamped only, n=20) | +0.032, p=0.87 | Essentially zero |
| Clamped games | 50% (25/50) | Half of all games have extreme scores |
| Seller advantage (n=25) | -0.461 mean, 72% hurt | Steering hurts sellers |
| Buyer advantage (n=25) | +0.780 mean, 92% help | Steering helps buyers |
| Dealmaker | Seller 100% of the time | Structural game asymmetry |

**Behavioral changes (real but not outcome-improving):**

| Dimension | Behavioral Effect | Numbers |
|-----------|-------------------|---------|
| Firmness (alpha=20, 7B) | Hedge suppression | 27x (0.05 vs 1.44 hedges/100w) |
| Firmness | Shorter responses | 22% (23.5 vs 30.1 words) |
| Firmness | Clamping | 24% of games |
| SCM (alpha=6, 3B) | Longer responses | 13% (25.6 vs 22.7 words) |
| SCM | Hedge change | Minimal (1.67 vs 1.99/100w) |
| SCM | Concession hardening | Steered concedes less per move |
| SCM | Clamping | 50% of games |

**Cross-dataset transfer (Deal or No Deal, Lewis et al. 2017):**
SCM vector does NOT generalize. Pareto rate: 16% (naive baseline=9%, human=77%). Advantage: -0.052. The "strategic concession making" label is a misnomer — the vector likely encodes "resist conceding," which is domain-specific price stubbornness on Craigslist, not transferable strategic reasoning.

**LLM judge (Gemini Flash, 50 games):**
Role dominates judge scores. Sellers rated higher on 5/6 dimensions regardless of steering. Only role-independent signal: naturalness degradation at high alpha (firmness). SCM judge scores: steered wins on firmness (+0.74), info_management (+0.51), strategic_reasoning (+0.35). But within-role correlations are inflated by clamped games — unclamped, only info_management survives (r=+0.58).

**Alpha dose-response (S2 TPE trials):**
- SCM/middle/L18: monotonic (r=+0.50, p=0.025). Low alpha (<2): +0.133, high (>5): +0.330. Best signal quality.
- Firmness/late/L27: monotonic but small (r=+0.53, mean +0.117).
- Anchoring/middle/L18: inverted-U, likely noise (r=-0.09, p=0.69).
- Layer matters: SCM/middle=+24.9% vs SCM/late=+6.0%.

**No steering decay detected.** Max cumulative steered tokens = 139 words, well below 300-500 token literature threshold (Practitioner's Field Guide 2026). Our 8-turn negotiations are too short.

**The causal chain (why outcomes don't improve):**
```
Contrastive pairs (surface biases: 1.8x length, 3.6x hedge clustering, zero opener overlap)
  → Vector extraction (encodes mix of surface + genuine conceptual signal)
  → Behavioral changes during generation (REAL: hedge suppression, response length, concession patterns)
  → Outcome scores (CONFOUNDED by: role effects + clamping artifacts)
  → Cross-dataset transfer (FAILS: not strategic reasoning, just price stubbornness)
```

**Why the task is the bottleneck (the pivot rationale):**
1. **Clamping:** 24-70% of games have prices outside target range, producing extreme ±1.0 scores that dominate averages. Strip clamped games → near-zero effect.
2. **Role asymmetry:** Seller finalizes 100% of deals (via `DEAL=`). Firmness is adaptive for buyers (resist overpaying) but maladaptive for sellers (can't close). This is structural, not fixable by better vectors.
3. **LLM-vs-LLM variance:** Two copies of the same model negotiating produces high per-game variance (std=0.72). Need n>>50 to detect d=0.24 effects, but GPU cost is prohibitive.
4. **1D scoring:** Craigslist is price-only. Cannot measure multi-dimensional strategic reasoning. DonD confirms this limitation.

**Proposed paper framing (from P4_PROGRESS.md):**
- **Primary claim:** "Activation steering produces measurable behavioral changes in LLM negotiation agents, but these changes do not translate to statistically significant outcome improvements. Apparent advantages are driven by scoring artifacts (clamping) and role asymmetry."
- **Contribution:** The evaluation framework itself — methodology that identifies confounds inflating apparent steering effects. "How to properly evaluate steering in applied domains" paper, not "steering improves negotiation."

**Key experimental findings (extraction methods):**
- Mean difference vectors are strictly superior to PCA: 1-D probe accuracy 0.903 vs 0.586 (+54%) in best variant. PCA's PC1 captures noise at small sample sizes (12-80 pairs in 2560-dim space).
- Logistic regression closely matches MD: 1-D probe 0.891 vs 0.903. LR's near-perfect training accuracy (0.998) is tautological. LR "fudges" toward confounds (firmness projection drop 7.2% vs MD's 3.4%).
- All three methods agree on validity scores (33/100), traffic lights, and projection drops (2.1-2.8%). Data quality is the bottleneck, not algorithmic choice.
- Literature consensus confirmed: MD >= LR >> PCA for steering (Im & Li 2025).

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

**Known open issues (from `docs/P4_PROGRESS.md`):**
1. **Same-model evaluation contamination.** Both agents share one model instance. Baseline sees steered agent's text and may adapt in-context. Fixing requires separate model instances (2x VRAM). Currently flagged as limitation.
2. **Low statistical power.** n=50 paired scenarios, 25 per role. After removing clamped games, n=20. Insufficient to detect small effects (d=0.2-0.3).
3. **Single model family.** All results on Qwen 2.5 (3B and 7B). No cross-architecture validation.
4. **Contrastive pair quality.** 6 critical surface biases found in audit: 1.8x length ratio, zero opener overlap between pos/neg, perfect ellipsis separation, capitulation word segregation, 3.6x hedge clustering, 5.1x yielding clustering.
5. **Single judge model.** Only Gemini Flash. Inter-model reliability not computed.

**CraigslistBargains dataset properties (from `phase_a_diagnostic.py`):**
- 5,247 train / 597 val scenarios, zero overlapping targets, zero bad prices
- seller_target == listing_price always; buyer_target ~72% of listing
- 6 categories: furniture 25%, housing 20%, bike 18%, car 13%, electronics 13%, phone 11%
- Price ranges: ~$180 (electronics) to ~$10,933 (car)

**Playground experiment results (task design validation, `moiz-task-design` branch):**
- GPT-4o vs Gemini (1 game): Walk-away (no deal). Seller stuck at $4500, buyer gave up. Suggests frontier models are *too* firm — the task may be structurally broken even for strong models.
- LLaMA 8B baseline (10 games via Groq): 8/10 agreed, 1 walk-away, 1 clamped. Buyer scores tend higher (median ~0.65). Sellers still finalize most deals. One game had the seller write `(Note: I'm not ready to close at $9050, so I didn't write DEAL=9050)` — the model literally explained its internal reasoning instead of acting, showing prompt-following issues.

**Extraction method rationale (3 methods: mean_diff, PCA, logreg):**
- **Mean difference** (generative): `mean(pos) - mean(neg)`. Standard CAA (Panickssery et al. 2024). Im & Li 2025 prove this is optimal under pointwise loss.
- **PCA** (variance-based): PC1 of difference vectors. Follows Zou et al. 2023 (RepE). Finds dominant axis of variation; can diverge from concept direction if noise variance dominates.
- **Logistic regression** (discriminative): L2-regularised LR weight vector. Follows Li et al. 2024 (ITI), Zou et al. 2023 (classifier variant). Finds max-margin separation boundary. Known to "fudge" direction when surface confounds are non-orthogonal (Marks & Tegmark 2023). Empirically confirmed: LR's firmness drops 7.2% on projection vs MD's 3.4%, while 1-D held-out probe accuracy is within 1.2% of MD (0.891 vs 0.903). Literature consensus confirmed: MD ≥ LR >> PCA for steering (Im & Li 2025).
- **K-means rejected:** With balanced classes, k-means centroids ≈ class means, so `centroid_1 - centroid_0 ≈ mean_diff`. When clusters don't recover classes, it's worse — Euclidean distance in full activation space is dominated by highest-variance (surface confound) directions. Adds no new lens; conceptually redundant with mean diff.
- **Why run all three (triangulation value):** The methods have different inductive biases (generative vs variance-based vs discriminative), so their agreement/disagreement is informative. Empirically: identical validity scores (33/100), identical traffic lights, and similar projection drops (2.1-2.8%) across all three methods prove data quality is the bottleneck, not algorithmic choice. Disagreements reveal method-specific weaknesses: LR's 7.2% firmness projection drop (vs MD's 3.4%) exposes discriminative confound exploitation; PCA's 0.586 mean 1-D probe accuracy (vs MD's 0.903) exposes noise-sensitivity at small sample sizes. A reviewer cannot dismiss findings as method-dependent when three algorithmically distinct approaches converge.

**Key decisions and rationale:**
1. **Role-separated reporting is mandatory.** Aggregate -1.6% hides seller=-27% vs buyer=+24%. Never report aggregate scores alone.
2. **Clamped games must be flagged separately.** 24-70% of games produce extreme ±1.0 scores. Unclamped analysis is the real signal.
3. **Mean difference is the primary extraction method.** MD >= LR >> PCA empirically and theoretically (Im & Li 2025). Run all three for triangulation, but MD is the workhorse.
4. **Best steering pair variant is `neg8dim_12pairs_matched`.** 33/100 validity score, best among all 8 variants. Length-matching is the only effective intervention; more pairs hurts.
5. **Best steering config: SCM, layer 18 (middle), alpha~6.** Only config with monotonic dose-response. Firmness at L27 works behaviorally but outcomes are clamping artifacts.
6. **The paper is about evaluation methodology, not positive results.** The contribution is the framework that identifies confounds. Negative results are publishable findings.
7. **Phase 2 pivot to controlled tasks.** LLM-vs-LLM has too much variance. Preset scripts or simpler tasks needed to isolate steering signal from game noise.
8. **Qwen 3B/7B only.** Llama 3B was tested but produced faulty results. Qwen models don't require HF tokens.
9. **Anti-steerability is role-dependent, not random.** Reframed from Tan et al. 2024. Firmness is adaptive for buyers, maladaptive for sellers.
10. **Hedge suppression is the primary observable effect.** 27x reduction maps directly from contrastive pair bias to deployment behavior. This is the clearest causal chain.

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
