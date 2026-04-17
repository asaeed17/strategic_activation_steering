# Activation Steering for LLM Negotiation

Extract behavioural steering vectors from contrastive pairs, inject them into LLMs during inference, and measure whether they change negotiation behaviour and outcomes. Evaluated on CraigslistBargains (price negotiation), Ultimatum / Dictator Game (single-shot bargaining), Resource Exchange (multi-turn), and Deal or No Deal (multi-issue).

## Project Structure

```
.
├── Extraction & steering
│   ├── extract_vectors.py                    # Extract vectors (mean diff / PCA / logreg)
│   ├── apply_steering.py                     # CraigslistBargains steered games
│   ├── apply_steering_preset_nego.py         # 40 fixed preset scenarios
│   ├── apply_steering_ultimatum.py           # Ultimatum Game (rule-based opponent)
│   ├── ultimatum_game.py                     # Ultimatum/Dictator (LLM-vs-LLM, paired)
│   ├── resource_exchange_game.py             # Multi-turn resource exchange
│   ├── deal_or_no_deal.py                    # Deal or No Deal cross-dataset test
│   └── run_grid.py                           # Single-model-load grid runner
│
├── Hyperparameter search
│   ├── fast_search_steering.py               # Two-stage TPE search (CraigslistBargains)
│   ├── lightweight_gridsearch.py             # Single-dim grid (CraigslistBargains)
│   ├── lightweight_gridsearch_ultimatum.py   # Ultimatum grid
│   └── lightweight_gridsearch_preset_nego.py # Preset-scenario grid
│
├── Analysis
│   ├── llm_judge.py                          # Multi-model LLM judge (Gemini/GPT/LLaMA)
│   ├── steering_scalar_analysis.py           # Geometric analysis of alpha
│   ├── plot_scalar_safety.py                 # Safety-analysis figures
│   └── analysis/                             # Post-run CPU analyses
│       ├── analyse_eval.py
│       ├── analyse_final_grid.py             # 7B grid heatmaps + top configs
│       ├── analyse_ultimatum.py              # UG paired tests + dose-response
│       ├── analyse_ug_hypotheses.py          # Pre-registered UG hypotheses (BH-FDR)
│       ├── compile_ug_gridsearch.py          # Combine UG results across runs
│       ├── analyse_resource_exchange.py
│       ├── turn_metrics.py                   # Per-turn behavioural metrics
│       ├── role_analysis.py                  # Seller vs buyer breakdown
│       ├── audit_pairs.py                    # Contrastive pair surface bias audit
│       ├── statistical_hardening.py          # Bootstrap CIs, TOST, BH-FDR
│       ├── interpretability.py               # Geometric / contamination analysis
│       ├── plot_results.py                   # Publication figures
│       └── unified_results_analysis.ipynb
│
├── Validation
│   └── validation/
│       ├── validate_vectors.py               # 7-check validation suite
│       ├── probe_vectors.py                  # Logistic regression probes
│       ├── orthogonal_projection.py          # Surface-feature decontamination
│       ├── dose_response_validation.py       # LLM-judge dose-response test
│       └── run_full_validation.py            # Full validation orchestrator
│
├── Orchestration (shell)
│   ├── run_extraction.sh                     # Extract all variants (Qwen 3B)
│   ├── run_extraction_all_variants.sh
│   ├── run_extraction_and_validation_all_variants.sh
│   ├── run_all_methods.sh                    # 6 variants × 3 methods
│   ├── run_3b_grid.sh                        # 5 dims × 5 layers × 4 alphas
│   └── run_3b_empathy.sh                     # Empathy dim only
│
├── Data
│   ├── steering_pairs/{variant}/             # 12 variants of contrastive pairs
│   │   ├── negotiation_steering_pairs.json
│   │   └── control_steering_pairs.json
│   ├── craigslist_data/                      # CraigslistBargains (train + val JSON)
│   └── preset_negotiators/scenarios.json     # 40 fixed negotiation scenarios
│
├── Outputs
│   ├── vectors/{variant}/{negotiation,control}/{method}/   # Extracted .npy
│   ├── results/
│   │   ├── eval/                             # P4 controlled evaluations
│   │   ├── ultimatum/                        # UG/DG grids
│   │   ├── resource_exchange/
│   │   ├── projection/                       # Orthogonal projection
│   │   ├── figures/                          # Publication figures
│   │   └── validation/                       # Per-variant validation reports
│   ├── FINAL_VALIDATION_RESULTS/             # Full-pipeline validation outputs
│   └── hyperparameter_results/               # Optuna DBs + stage outputs
│
└── Misc
    ├── docs/                                 # Background documents
    ├── playground/                           # Minimal test runners
    ├── scripts/                              # GPU runners, launchers, one-offs
    ├── tests/                                # pytest tests
    ├── logs/
    └── old_files/                            # Archived intermediate work
```

### Steering-pair variants

Contrastive pairs live under `steering_pairs/{variant}/`. Naming: `neg{N}dim_{K}pairs_{matching}` where `N` is the negotiation-dimension count, `K` is pairs per dimension, and `matching` is `matched` (length-matched within ±30% word count) or `raw` (unmatched). Ultimatum variants use 10 game-specific dimensions.

| Variant | Dims | Pairs/dim | Lengths |
|---|---|---|---|
| `neg15dim_12pairs_{raw,matched}` | 15 | 12 | raw / matched |
| `neg15dim_{20,80}pairs_matched` | 15 | 20 / 80 | matched |
| `neg8dim_12pairs_{raw,matched}` | 8 (merged) | 12 | raw / matched |
| `neg8dim_{20,80}pairs_matched` | 8 | 20 / 80 | matched |
| `ultimatum_10dim_20pairs_matched` | 10 (UG-specific) | 20 | matched, game-specific tokens |
| `ultimatum_10dim_20pairs_general_matched` | 10 | 20 | matched, general-domain |

## Setup

```bash
pip install torch transformers numpy scikit-learn tqdm optuna scipy
# Optional
pip install bitsandbytes                # 4-bit quantisation
pip install google-genai groq openai    # llm_judge.py
```

Models: Qwen 2.5 (3B, 7B, 32B) — no HuggingFace token needed. Llama / Gemma / Mistral are gated (set `HF_TOKEN`).

`HF_HOME` is redirected to `.hf_cache/` (set by the shell scripts) to avoid UCL home-directory quota limits.

## Pipeline

Four stages. Each depends on the outputs of the previous.

### 1. Extract steering vectors

```bash
# Single variant
python extract_vectors.py \
    --models qwen2.5-3b \
    --pairs_file steering_pairs/neg8dim_12pairs_matched/negotiation_steering_pairs.json \
    --output_dir vectors/neg8dim_12pairs_matched/negotiation

# All 8 variants, Qwen 3B
bash run_extraction.sh
```

Runs contrastive pairs through the model, extracts last-token hidden states at every layer, and computes direction vectors via three methods (mean difference, PCA, logistic regression).

Output: `vectors/{variant}/{negotiation|control}/{method}/{dimension}_all_layers.npy`, shape `(n_layers, hidden_dim)`, unit-normed per layer.

Options:

- `--models qwen2.5-3b qwen2.5-7b` — multiple models
- `--dimensions firmness empathy` — subset of dimensions
- `--quantize` — 4-bit inference (do **not** use for game runs — suppresses steering effects)
- `--layers 8 12 16 20` — save only specific layers

### 2. Validate vectors

```bash
python validation/validate_vectors.py \
    --model qwen2.5-3b --full \
    --negotiation_pairs steering_pairs/neg8dim_12pairs_matched/negotiation_steering_pairs.json \
    --control_pairs steering_pairs/neg8dim_12pairs_matched/control_steering_pairs.json \
    --vectors_dir vectors/neg8dim_12pairs_matched/negotiation \
    --output_dir results/validation/neg8dim_12pairs_matched
```

Seven checks: length confound (Cohen's d), vocabulary Jaccard, probe accuracy + permutation test, cross-dimension cosine, per-pair alignment, 1-D steering-direction probe, selectivity. Outputs `validation_results.json`, `validation_report.txt`, and per-dimension selectivity plots.

Decontamination — project out control-dimension components, re-run probes:

```bash
python validation/orthogonal_projection.py \
    --variant neg8dim_12pairs_matched --method mean_diff
```

`--full` mode requires a GPU (re-extracts activations); `--analyze-only` mode is CPU-only and reads vectors from disk.

### 3. Run negotiation games

**CraigslistBargains:**

```bash
python apply_steering.py \
    --model qwen2.5-3b \
    --dimension strategic_concession_making \
    --alpha 6 --layers 18 \
    --use_craigslist --num_samples 50 \
    --output_file results.json

# Hyperparameter search
python fast_search_steering.py --model qwen2.5-3b --use_craigslist --output_dir results/fast
```

The steered agent alternates seller / buyer roles each game to control for role asymmetry. `advantage = mean(steered_score) - mean(baseline_score)`.

**Ultimatum / Dictator Game:**

```bash
python ultimatum_game.py \
    --model qwen2.5-7b --dimension firmness \
    --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
    --layers 10 --alpha 7 \
    --steered_role proposer --game ultimatum \
    --n_games 50 --variable_pools --paired --temperature 0.0 \
    --output_dir results/ultimatum/final_7b_grid

# Full grid with single model load (recommended for large sweeps)
python run_grid.py --config_file grid_config.json
```

Paired design: steered and baseline play the same game scenarios, giving honest within-pair comparisons.

**Resource Exchange (multi-turn, cross-task):**

```bash
python resource_exchange_game.py \
    --model qwen2.5-32b --dimension firmness \
    --layers 28 --alpha 5 --n_games 50
```

**Deal or No Deal (cross-dataset):**

```bash
python deal_or_no_deal.py \
    --model qwen2.5-3b --dimension strategic_concession_making \
    --alpha 6 --layers 18 --num_samples 50
```

### 4. Analyse results

Post-run analysis is CPU-only and reads game-result JSON files.

```bash
# CraigslistBargains
python analysis/turn_metrics.py          # per-turn behaviour (→ turn_metrics_enriched.json)
python analysis/role_analysis.py         # seller vs buyer tables
python analysis/analyse_eval.py          # paired analysis of P4 evaluation

# Ultimatum Game
python analysis/analyse_ultimatum.py     # paired tests, Cohen's d, dose-response
python analysis/analyse_ug_hypotheses.py # pre-registered hypotheses with BH-FDR
python analysis/compile_ug_gridsearch.py # combine across batches
python analysis/analyse_final_grid.py    # 7B full-grid heatmaps + top configs

# Figures
python analysis/plot_results.py

# LLM judge (needs GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY)
python llm_judge.py --judges gemini

# Unified notebook (CraigslistBargains + UG + Resource Exchange comparison)
jupyter notebook analysis/unified_results_analysis.ipynb
```
