# Activation Steering in Strategic Interactions

COMP0087 Statistical NLP Group Project, UCL (2025-26).

Contrastive activation addition applied to ten behavioural dimensions (greed, firmness, empathy, etc.) in LLM negotiation. Steering vectors are extracted from contrastive prompt pairs, injected at inference, and evaluated on the Ultimatum Game (UG, Qwen 2.5-7B, 450-config grid) and a multi-turn Resource Exchange (RE, Qwen 2.5-32B-GPTQ).

## Key Results

- All 10 dimensions produce significant demand shifts in UG, but payoff improvements are rare and fragile.
- Greed and spite outperform firmness by raising the proposed split without triggering the responder's fairness heuristic.
- In multi-turn RE, only interpersonal dimensions (empathy, narcissism, spite) transfer; intrapersonal ones are neutralised by an adaptive opponent.
- Activation steering disrupts RLHF alignment defaults rather than installing new capabilities.

## Repository Structure

```
.
├── extract_vectors.py              # Step 1: extract steering vectors from contrastive pairs
├── ultimatum_game.py               # Step 2: run Ultimatum Game experiments with steering
├── resource_exchange_game.py        # Step 2b: run Resource Exchange experiments (32B)
│
├── validation/                      # Vector validation pipeline
│   └── validate_vectors.py          #   Full validation suite (probes, selectivity, confound checks)
│
├── analysis/                        # Post-experiment analysis (all CPU-only)
│   ├── analyse_final_grid.py        #   Main UG grid analysis (450 configs)
│   ├── analyse_resource_exchange.py #   RE analysis
│   ├── plot_results.py              #   Generate paper figures
│   ├── interpretability.py          #   Cosine evolution, PCA, logit lens, contamination
│   ├── statistical_hardening.py     #   Statistical robustness checks
│   ├── audit_pairs.py              #   Contrastive pair surface bias audit
│   ├── unified_results_analysis.ipynb  # Multi-paradigm comparison notebook
│   └── ...                          #   Additional analysis scripts
│
├── scripts/                         # GPU job runners and monitoring
│   ├── gpu_runners/                 #   Final grid batch scripts
│   ├── launchers/                   #   Shell launchers for GPU dispatch
│   └── monitoring/                  #   Grid monitoring and result pulling
│
├── steering_pairs/
│   └── ultimatum_10dim_20pairs_general_matched/  # 10 dims x 20 pairs, length-matched
│       ├── negotiation_steering_pairs.json
│       └── control_steering_pairs.json
│
├── vectors/
│   └── ultimatum_10dim_20pairs_general_matched/  # Extracted .npy vectors (mean_diff)
│
├── results/
│   ├── ultimatum/
│   │   ├── final_7b_llm_vs_llm/    # 450-config main grid (10d x 9L x 5a x n=50)
│   │   ├── llm_vs_rulebased/       # Cross-design validation data
│   │   ├── multimodel_3b/          # 3B model comparison
│   │   └── acceptance_curve/       # Unsteered acceptance curve (700 games)
│   ├── resource_exchange/           # RE results (32B, 6 dims x 3 layers)
│   ├── figures/                     # Publication figures (fig1-fig14)
│   ├── interpretability/            # Cosine evolution, logit lens, PCA, contamination
│   └── projection/                  # Orthogonal projection validation
│
├── FINAL_VALIDATION_RESULTS/        # Vector validation reports and plots
│   └── ultimatum_10dim_20pairs_general_matched/
│
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Core: `torch`, `transformers`, `numpy`, `scikit-learn`, `scipy`, `tqdm`.

Models: Qwen 2.5-7B and 32B-GPTQ (no HuggingFace token needed).

## Pipeline

### 1. Extract Steering Vectors

```bash
python extract_vectors.py --models qwen2.5-7b \
    --pairs_file steering_pairs/ultimatum_10dim_20pairs_general_matched/negotiation_steering_pairs.json \
    --output_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation
```

Runs contrastive pairs through the model, extracts last-token hidden states at every layer, computes direction vectors via mean difference. Output: `.npy` files, shape `(n_layers, hidden_dim)`, unit-normed per layer.

### 2. Validate Vectors

```bash
PYTHONPATH=. python validation/validate_vectors.py --model qwen2.5-7b --full \
    --negotiation_pairs steering_pairs/ultimatum_10dim_20pairs_general_matched/negotiation_steering_pairs.json \
    --control_pairs steering_pairs/ultimatum_10dim_20pairs_general_matched/control_steering_pairs.json \
    --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
    --output_dir results/validation
```

Seven checks: length confound (Cohen's d), vocabulary Jaccard, probe accuracy + permutation test, cross-dimension cosine, per-pair alignment, 1-D steering-direction probe, selectivity.

### 3. Run Experiments

**Ultimatum Game (main experiment):**

```bash
python ultimatum_game.py --model qwen2.5-7b --dimension firmness \
    --vectors_dir vectors/ultimatum_10dim_20pairs_general_matched/negotiation \
    --layers 10 --alpha 7 --steered_role proposer --game ultimatum \
    --n_games 50 --variable_pools --paired --temperature 0.0 \
    --output_dir results/ultimatum/final_7b_grid
```

**Resource Exchange (cross-task generalisation):**

```bash
python resource_exchange_game.py \
    --model qwen2.5-32b --dimension firmness \
    --layers 28 --alpha 50 --n_games 50
```

### 4. Analyse Results

```bash
python analysis/analyse_final_grid.py       # Main grid statistics
python analysis/analyse_resource_exchange.py # RE analysis
python analysis/plot_results.py             # Generate paper figures
python analysis/interpretability.py         # Interpretability suite
```

## Design Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Dimensions | 10 (firmness, empathy, greed, spite, anchoring, composure, fairness_norm, flattery, narcissism, undecidedness) | Full behavioural landscape |
| Layers | L4-L20 (9 layers, step 2) | Different dimensions peak at different layers |
| Alphas | {-7, -5, 5, 7, 15} | Sign coverage + dose-response |
| n per config | 50 paired games | Detects d > 0.5 at 80% power |
| Hardware | RTX 3090 Ti, bf16 | Ampere CC 8.6; quantization suppresses effects |
| Temperature | 0.0 | Variable pools provide variance |
| Extraction | Mean difference | Matched/exceeded logistic regression; PCA weaker |

## Authors

Abdullah Saeed, Ahmed Ansari, Ayman Khan, Damon Surrao, Moiz Imran, Sonny Lin — University College London, Department of Computer Science.
