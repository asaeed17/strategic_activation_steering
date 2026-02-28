# Activation Steering for LLM Negotiation — Full Project Context

> **Purpose of this document:** Complete context for an AI assistant working on this codebase. Covers everything that has been built, every design decision made, and the full roadmap of planned work. Read this before touching any file.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Repository Structure](#2-repository-structure)
3. [What Has Been Built](#3-what-has-been-built)
   - 3.1 [negotiation_steering_pairs.json](#31-negotiation_steering_pairsjson)
   - 3.2 [extract_vectors.py](#32-extract_vectorspy)
   - 3.3 [apply_steering.py](#33-apply_steeringpy)
   - 3.4 [fast_search_steering.py](#34-fast_search_steeringpy)
4. [Core Concepts & Design Decisions](#4-core-concepts--design-decisions)
5. [Data Flow End-to-End](#5-data-flow-end-to-end)
6. [Scoring System](#6-scoring-system)
7. [What Remains To Be Built — Full Roadmap](#7-what-remains-to-be-built--full-roadmap)
   - P1 [Extraction Methods](#p1--extraction-methods)
   - P2 [Validation & Probing](#p2--validation--probing)
   - P3 [Application Strategies](#p3--application-strategies)
   - P4 [Metrics & Evaluation](#p4--metrics--evaluation)
   - P5 [Data & Baselines](#p5--data--baselines)
   - P6 [Concept Patching & Interpretability](#p6--concept-patching--interpretability)
8. [Key Conventions & Gotchas](#8-key-conventions--gotchas)
9. [Paper Structure](#9-paper-structure)

---

## 1. Project Goal

The central research question is: **does activation steering — adding a scaled behavioural direction vector to a transformer's residual stream at inference time — measurably improve a language model's performance as a price negotiator?**

The approach follows Contrastive Activation Addition (CAA, Panickssery et al. 2024) and Representation Engineering (RepE, Zou et al. 2023). We identify directions in a model's activation space that correspond to negotiation-relevant traits (firmness, assertiveness, patience, etc.) and inject them during inference. We then measure whether the steered model gets better deals in simulated buyer-seller negotiations drawn from a real-world corpus.

This is not a prompting or fine-tuning project. The model weights are never modified. Steering happens entirely through forward hooks on residual stream activations.

---

## 2. Repository Structure

```
project/
├── negotiation_steering_pairs.json   # Contrastive pair dataset (the training signal)
├── extract_vectors.py                # Pulls direction vectors from models
├── apply_steering.py                 # Runs steered vs unsteered negotiations
├── fast_search_steering.py           # 3-stage hyperparameter search
├── vectors/                          # Output of extract_vectors.py
│   └── {model_alias}/
│       ├── metadata.json
│       ├── mean_diff/
│       │   ├── {dim}_all_layers.npy  # shape: (n_layers, hidden_dim)
│       │   └── {dim}_layer{N:02d}.npy  # shape: (hidden_dim,)
│       └── pca/
│           └── (same structure)
└── results/                          # Output of apply_steering.py / search
    └── results.json
```

Files not yet created but planned:

```
├── probe_vectors.py          # P2: logistic regression probing per layer
├── concept_patching.py       # P6: causal intervention experiments
├── llm_judge.py              # P4: LLM-based transcript evaluation
└── data/
    └── craigslist/           # P5: locally cached dataset splits
```

---

## 3. What Has Been Built

### 3.1 `negotiation_steering_pairs.json`

**What it is:** The foundational dataset. 180 contrastive pairs across 15 negotiation dimensions. Each pair has:

- `context`: a realistic negotiation scenario (e.g. "A buyer challenges your asking price of $85,000")
- `positive`: a response demonstrating the target trait at high quality
- `negative`: a response demonstrating the opposite or an absence of the trait

**The 15 dimensions:**

| ID                            | Name                        | Core Behaviour                                                      |
| ----------------------------- | --------------------------- | ------------------------------------------------------------------- |
| `firmness`                    | Firmness                    | Holding position under pressure; no unnecessary concessions         |
| `empathy`                     | Empathy                     | Acknowledging counterpart feelings without abandoning your position |
| `active_listening`            | Active Listening            | Reflecting and summarising what the other party actually said       |
| `assertiveness`               | Assertiveness               | Direct, confident requests without aggression or over-hedging       |
| `interest_based_reasoning`    | Interest-Based Reasoning    | Asking "why" to find underlying needs, not just positions           |
| `emotional_regulation`        | Emotional Regulation        | Composure when challenged, provoked, or under pressure              |
| `strategic_concession_making` | Strategic Concession-Making | Conditional, reciprocal, decreasing-magnitude concessions           |
| `anchoring`                   | Anchoring                   | Setting advantageous reference points early                         |
| `rapport_building`            | Rapport Building            | Establishing trust and connection with the counterpart              |
| `batna_awareness`             | BATNA Awareness             | Knowing your alternatives; genuine willingness to walk away         |
| `reframing`                   | Reframing                   | Shifting narrative or perspective to open new possibilities         |
| `patience`                    | Patience                    | Using silence and deliberate pacing as negotiation tools            |
| `value_creation`              | Value Creation              | Trading across multiple issues to expand deal space                 |
| `information_gathering`       | Information Gathering       | Probing questions to uncover constraints and priorities             |
| `clarity_and_directness`      | Clarity & Directness        | Unambiguous, explicit asks; no hedging language                     |

**Format:**

```json
{
  "metadata": { "dimensions": 15, "total_pairs": 180, "version": "1.0" },
  "dimensions": [
    {
      "id": "firmness",
      "name": "Firmness",
      "description": "...",
      "pairs": [
        {
          "context": "A buyer challenges your asking price...",
          "positive": "I understand it feels like a stretch, but $85,000 reflects...",
          "negative": "You know what, I hear you — maybe $85,000 is a bit much..."
        },
        ...  // 12 pairs per dimension
      ]
    }
  ]
}
```

**Each dimension currently has exactly 12 pairs.** P1 and P5 are tasked with expanding this to 20+ pairs per dimension.

---

### 3.2 `extract_vectors.py`

**What it does:** Loads a model, runs the contrastive pairs through it, captures hidden states, computes direction vectors for each dimension at every layer, and saves them to disk.

**Model registry (`MODELS` dict):** Five families supported. The alias is the subdirectory name used for all saved vectors.

| Key            | HuggingFace ID                        | Alias        | Token Required |
| -------------- | ------------------------------------- | ------------ | -------------- |
| `qwen2.5-7b`   | Qwen/Qwen2.5-7B-Instruct              | qwen2.5-7b   | No             |
| `qwen2.5-3b`   | Qwen/Qwen2.5-3B-Instruct              | qwen2.5-3b   | No             |
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B-Instruct            | qwen2.5-1.5b | No             |
| `llama3-8b`    | meta-llama/Meta-Llama-3.1-8B-Instruct | llama3-8b    | Yes            |
| `llama3-3b`    | meta-llama/Llama-3.2-3B-Instruct      | llama3-3b    | Yes            |
| `gemma2-9b`    | google/gemma-2-9b-it                  | gemma2-9b    | Yes            |
| `gemma2-2b`    | google/gemma-2-2b-it                  | gemma2-2b    | Yes            |
| `mistral-7b`   | mistralai/Mistral-7B-Instruct-v0.3    | mistral-7b   | Yes            |

**HF_TOKEN:** Read from `os.environ.get("HF_TOKEN", None)` or hardcoded at the top of the file.

**`format_sample(context, response, tokenizer, config)`:**

- Wraps the pair in the model's native chat template: context → user turn, response → assistant turn
- Falls back to `"{context}\n\nResponse: {response}"` if the template fails
- This means the last token in the sequence is the last token of the assistant response, which is where we read the hidden state

**`extract_hidden_states(model, tokenizer, texts, batch_size=4)`:**

- Left-pads all sequences (tokenizer `padding_side='left'`) so that index `[-1]` always lands on the final real token regardless of sequence length — critical for batched extraction
- Runs `model(**enc, output_hidden_states=True)` in `torch.no_grad()`
- `out.hidden_states` is a tuple of length `n_layers + 1`: `[0]` is the embedding output, `[1:]` are transformer block outputs
- Collects `h[:, -1, :]` from each block → shape per batch: `(B, n_layers, hidden_dim)`
- Returns `np.ndarray` of shape `(N, n_layers, hidden_dim)`, dtype float32, on CPU

**`compute_mean_diff(pos, neg)`:**

```
direction_l = normalise( mean(pos_l) - mean(neg_l) )
```

Shape in: `(N, n_layers, H)` × 2. Shape out: `(n_layers, H)`. Each layer's vector is unit-normalised independently.

**`compute_pca_direction(pos, neg)`:**

```
diffs_l = [pos_i_l - neg_i_l  for each pair i]   # shape (N, H)
direction_l = PCA(diffs_l).components_[0]
```

Sign is resolved by checking `dot(pc1, mean_diff_l)` — flipped if negative. Falls back to using the single difference vector if N < 2.

**Output files per model per method:**

- `{dim}_all_layers.npy` — shape `(n_layers, H)`, always saved
- `{dim}_layer{N:02d}.npy` — shape `(H,)`, saved for each layer in `target_layers` (default: all layers)
- `metadata.json` — records `hf_id`, `alias`, `n_layers`, `hidden_dim`, list of dimension IDs, methods, saved layers

**Currently extracted (confirmed from metadata.json):**

- Model: `Qwen/Qwen2.5-3B-Instruct` (alias: `qwen2.5-3b`)
- 36 layers, hidden_dim = 2048
- All 15 dimensions extracted with both `mean_diff` and `pca`
- All 36 layers saved

**CLI flags:**

```bash
python extract_vectors.py \
  --models qwen2.5-3b qwen2.5-7b \
  --dimensions firmness empathy \
  --layers 8 16 24 \
  --quantize \           # 4-bit NF4 via bitsandbytes
  --sim_matrix \         # print cosine similarity matrix after extraction
  --sim_layer 16
```

**`print_similarity_matrix`:** Diagnostic utility. Loads all per-layer .npy files for a given method and layer, computes pairwise cosine similarities. High similarity between two dimensions suggests they may be partially redundant in this model.

---

### 3.3 `apply_steering.py`

**What it does:** Loads direction vectors and a model, draws negotiation scenarios from CraigslistBargains, and runs steered vs. unsteered agents against each other. Saves full transcripts and scores to JSON.

**`SteeringHook` class:**

```python
class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float): ...
    def hook_fn(self, module, input, output):
        # output is (hidden_state, ...) tuple or just tensor
        # adds alpha * direction to hidden_state
    def register(self, layer_module) -> None:
        self._handle = layer_module.register_forward_hook(self.hook_fn)
    def remove(self) -> None:
        self._handle.remove()
```

Hooks are attached to `model.model.layers[layer_idx]` (accessed via `get_transformer_layers(model)`). They are registered just before `model.generate()` and removed in a `finally` block — this is safe even if generation raises an exception.

**`load_direction_vectors(vectors_dir, model_alias, dimension, method, layer_indices)`:**

- Loads per-layer .npy files for the specified layer indices
- Falls back to slicing `{dim}_all_layers.npy` if individual files are missing
- Returns `Dict[int, np.ndarray]` mapping layer index → vector

**CraigslistBargains loading (`load_craigslist`):**

- Fetches parsed JSON from CodaLab URLs (two splits: `train`, `validation`)
- **The URLs are live-fetched every run — they could go down. See P5.**
- Parses each entry to extract: `title`, `description`, `category`, `listing_price`, `seller_target`, `buyer_target`
- Filters out entries with missing/zero prices or missing titles
- Returns a random sample of `num_samples` scenarios

**System prompts:**

- `build_seller_system(scenario)`: instructs the model it is selling the item, gives item details, states the private minimum (seller_target), tells it to write in conversational sentences and end with `DEAL=<price>` when ready
- `build_buyer_system(scenario)`: same structure for the buyer role; private maximum is buyer_target; opens with lowball framing

**`generate_turn(model, tokenizer, messages, direction_vectors, alpha, ...)`:**

- Formats messages with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- Attaches steering hooks for the specified layers if `direction_vectors` is not None and `alpha != 0`
- Calls `model.generate()` with `do_sample=(temperature > 0)`
- Strips `YOU:` / `THEM:` prefixes that some models parrot back
- If `can_finalise=False`, removes any `DEAL=` lines the model produced prematurely
- Returns the first non-empty line, or the full stripped text if no line breaks

**`run_game(model, tokenizer, scenario, dvecs_seller, alpha_seller, dvecs_buyer, alpha_buyer, steered_role, ...)`:**

- Buyer always opens: `"Hi, I'm interested in this. Would you take ${listing_price * 0.6:.0f}?"`
- `MIN_TURNS_BEFORE_DEAL = 3` — DEAL signals are suppressed until turn 3
- `MAX_TURNS = 10` — game ends at no-deal if this is reached
- Seller goes first after the opening bid, then alternates
- The `steered_role` parameter records which agent (seller or buyer) is steered, so the correct score gets attributed
- Returns a dict with: `agreed`, `agreed_price`, `dealmaker`, `seller_score`, `buyer_score`, `steered_role`, `steered_score`, `baseline_score`, `advantage`, `num_turns`, `transcript`, and all scenario fields

**Role alternation:** In `main()`, `steered_role = "seller" if i % 2 == 0 else "buyer"`. This ensures the steered agent plays each role roughly equally, preventing role-specific bias from inflating or deflating the advantage metric.

**`score_deal(agreed_price, seller_target, buyer_target)`:**

```
span = seller_target - buyer_target
seller_score = (agreed_price - buyer_target) / span     → 1.0 if seller got their target
buyer_score  = (seller_target - agreed_price) / span    → 1.0 if buyer got their target
```

Both clamped to [0, 1]. They sum to 1.0. Score of 0.5 = deal at the exact midpoint. If `span <= 0` (targets overlap), returns (0.5, 0.5).

**`summarise(results, alpha)`:** Aggregates over completed deals only. Key fields: `agree_rate`, `steered_score`, `baseline_score`, `advantage`, `avg_price`, `avg_turns`.

**CLI example:**

```bash
python apply_steering.py \
  --model qwen2.5-7b \
  --dimension firmness \
  --alpha 20 \
  --layers 12 16 20 \
  --use_craigslist \
  --num_samples 50 \
  --output_file results.json
```

`--use_craigslist` is required (it's a guard against accidentally running without data).

---

### 3.4 `fast_search_steering.py`

**What it does:** Searches over the four-dimensional config space (dimension × method × layer_preset × alpha) to find the combination that maximises negotiation advantage. Three-stage pipeline to balance cost and precision.

**Layer presets** (`LAYER_PRESET_FRACTIONS`): Named positions expressed as fractions of model depth. Model-agnostic — the same preset works for a 36-layer or a 32-layer model.

| Preset         | Fractions          | Layers for 36-layer model |
| -------------- | ------------------ | ------------------------- |
| `early`        | [0.25]             | [9]                       |
| `middle`       | [0.50]             | [18]                      |
| `late`         | [0.75]             | [27]                      |
| `early_middle` | [0.25, 0.50]       | [9, 18]                   |
| `middle_late`  | [0.50, 0.75]       | [18, 27]                  |
| `spread`       | [0.25, 0.50, 0.75] | [9, 18, 27]               |

**Stage 1 — Exhaustive Categorical Grid:**

- Iterates every combination of `dimensions × methods × layer_presets`
- Uses a fixed `probe_alpha` (defaults to midpoint of `[alpha_low, alpha_high]`)
- Runs `s1_games` (default: 5) per combo with `search_temperature=0` (greedy)
- Saves all results to `stage1_results.json`, sorted by advantage descending
- Filters incoherent configs (`agree_rate < 0.20`) before passing to Stage 2
- Passes top `s2_top_k` (default: 5) coherent configs to Stage 2

**Stage 2 — TPE Alpha Search:**

- One Optuna study per categorical config (`sqlite:///stage2.db` for persistence/resumability)
- Samples `alpha` in `[alpha_low, alpha_high]` using TPE (Tree-structured Parzen Estimator)
- `n_startup_trials` (default: 8) random trials before TPE exploits
- Prunes incoherent trials via `raise optuna.TrialPruned()`
- Saves `stage2_results.json` with per-combo alpha trial curves
- Passes top `s3_top_n` (default: 3) configs to Stage 3

**Stage 3 — Final Validation:**

- Reruns top configs at `eval_temperature` (default: 0.7) with `s3_max_new_tokens=120`
- Uses `s3_games` (default: 20) scenarios
- Saves `stage3_rank{N:02d}.json` per config and `final_best.json`

**`run_config(model, tokenizer, scenarios, dvecs, alpha, ...)`:** The core inner loop. Runs all scenarios for a fixed config. Returns `(advantage, agree_rate, steered_score)`.

**`is_coherent(advantage, agree_rate, min_agree=0.20)`:** Returns False if agree_rate < 0.20. Used to filter explosive alpha values that cause the model to stop making deals.

**Typical runtime on a 3B model:**

- Stage 1: ~30 min (4 dims × 2 methods × 6 presets × 5 games)
- Stage 2: ~60 min (5 combos × 20 trials × 10 games)
- Stage 3: ~10 min (3 configs × 20 games)

**CLI example:**

```bash
python fast_search_steering.py \
  --model qwen2.5-3b \
  --use_craigslist \
  --dimensions firmness anchoring \
  --methods mean_diff \
  --layer_presets middle late \
  --alpha_low 0.5 \
  --alpha_high 8.0 \
  --s1_games 5 \
  --s2_top_k 5 \
  --s2_trials 20 \
  --s3_games 20 \
  --output_dir results/fast
```

---

## 4. Core Concepts & Design Decisions

### Why left-padding?

Transformer hidden states are computed left-to-right. With right-padding, the last real token is at a variable position in the sequence depending on length. With left-padding, the last real token is always at index `[-1]` regardless of length. This makes batched activation extraction trivially correct — no need to compute attention mask positions.

### Why forward hooks?

Forward hooks allow per-generate-call steering without modifying model weights or architecture. Different agents in the same game can be steered with different directions and alphas. Hooks are always removed in `finally` blocks, so a crash during generation doesn't leave stale hooks that corrupt subsequent calls.

### Why role alternation?

In CraigslistBargains, the seller almost always has a higher target than the buyer. Scoring is relative to private targets. If the steered agent always played seller, a positive advantage could just mean "seller role is easier to optimise for." Alternating ensures the advantage is an average over both roles.

### Why two extraction methods?

Mean Difference is fast and interpretable — it's the straight average shift in representation space. PCA is more robust when individual pairs are noisy (some pairs may reflect multiple dimensions simultaneously). Comparing both tells us whether a dimension is robustly encoded (both methods agree) or fragile.

### Why layer presets rather than specific indices?

The optimal layer varies by model size. A preset like `middle` (0.5 × depth) maps to layer 16 in a 32-layer model and layer 18 in a 36-layer model. This makes search results transferable across model families.

### Why suppress DEAL before turn 3?

Without this, models sometimes accept the buyer's opening lowball immediately. `MIN_TURNS_BEFORE_DEAL = 3` forces at least two rounds of counter-offers, making the negotiation non-trivial and the scoring more meaningful.

### The `advantage` metric

`advantage = mean(steered_score) - mean(baseline_score)` computed over games that reached a deal. 0.0 means no benefit from steering. +0.1 means the steered agent captures 10 percentage points more of the available surplus on average. The maximum possible advantage is +1.0 (steered always gets everything, baseline always gets nothing), but in practice values above +0.2 would be remarkable.

---

## 5. Data Flow End-to-End

```
negotiation_steering_pairs.json
        │
        ▼
extract_vectors.py
  → format pairs with chat template
  → extract hidden states (last token, every layer)
  → compute mean_diff and pca directions
  → save to vectors/{alias}/{method}/{dim}_*.npy
        │
        ▼
fast_search_steering.py   (or apply_steering.py directly)
  → load metadata.json to get n_layers
  → Stage 1: grid search over dim × method × preset at fixed alpha
  → Stage 2: TPE search over alpha for top-K combos
  → Stage 3: final validation at eval temperature
  → save final_best.json
        │
        ▼
apply_steering.py (full eval run with best config)
  → load_craigslist() → N scenarios
  → for each scenario:
      → build seller/buyer system prompts
      → register hooks on steered agent
      → alternate turns (seller first after opening bid)
      → detect DEAL=<price> token
      → score_deal() → seller_score, buyer_score
  → summarise() → advantage, agree_rate, avg_price
  → save results.json
```

---

## 6. Scoring System

The scoring system is designed so that both agents' scores are always interpretable on the same scale and always sum to 1.0.

**Definition:**

```
span = seller_target - buyer_target   (the total available surplus)

seller_score = (agreed_price - buyer_target) / span
buyer_score  = (seller_target - agreed_price) / span
```

**Interpretation:**

- Score = 1.0 → agent got exactly their target price
- Score = 0.5 → deal landed at the exact midpoint of the two private targets
- Score = 0.0 → agent got exactly the counterpart's target (worst possible outcome)
- Scores can technically go outside [0,1] if the agreed price falls outside the target range, but both are clamped

**Edge case:** If `seller_target <= buyer_target` (targets overlap — both parties would be happy with almost any price in the overlap range), the scoring is degenerate, so we return (0.5, 0.5).

**No-deal games** are excluded from the advantage calculation but count toward `agree_rate`. A steered agent that gets great scores but only reaches deals 10% of the time is not useful.

---

## 7. What Remains To Be Built — Full Roadmap

### P1 — Extraction Methods

**Owner:** Person 1. Paper section: Extraction Methods.

#### Add whitened mean difference to `extract_vectors.py`

Standard mean difference picks up noise from general model variance — sentence length effects, token frequency effects, etc. Whitening removes these by decorrelating the activation space before computing the mean shift.

Implementation sketch:

```python
def compute_whitened_mean_diff(pos, neg):
    # pos, neg: (N, n_layers, H)
    # For each layer l:
    all_activations_l = np.concatenate([pos[:, l, :], neg[:, l, :]], axis=0)  # (2N, H)
    cov = np.cov(all_activations_l.T)  # (H, H)
    # Regularise: cov + eps * I
    eigvals, eigvecs = np.linalg.eigh(cov)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-6)) @ eigvecs.T  # whitening matrix
    pos_w = (pos[:, l, :] @ W.T)
    neg_w = (neg[:, l, :] @ W.T)
    direction_l = normalise(pos_w.mean(0) - neg_w.mean(0))
```

Save under `vectors/{alias}/whitened_mean_diff/`.

#### Add contrastive decoding direction to `extract_vectors.py`

Instead of comparing hidden states, compare the model's next-token probability distributions. The direction is derived from what the model is about to say, not from its internal state alone.

Implementation sketch:

```python
# Run model on context only (no response appended)
# Get logits from the last token position
# For each pair: pos_logits - neg_logits in vocab space
# Project back to residual stream space or use as a separate signal
```

#### Add random direction control

Generate a random unit vector for each layer (using `np.random.randn` normalised). Run the full negotiation eval with this as the "direction". If the random direction produces non-trivial advantage, the steering effect is not dimension-specific — it's just the result of perturbing activations in any direction.

- Expected result: random direction → advantage ≈ 0.0, often lower agree_rate
- This is the null hypothesis baseline for all other methods

#### Extract directions from attention heads

Each transformer layer has `n_heads` attention heads, each operating in a subspace of dimension `head_dim = hidden_dim / n_heads`. Access per-head outputs via hooks on `model.model.layers[l].self_attn`.

```python
# Hook on attention output projection (before it's added back to residual)
# Reshape to (batch, seq, n_heads, head_dim)
# Extract last-token, per-head activations
# Compute mean_diff per head
```

Produces `(n_layers, n_heads, head_dim)` arrays. Lets us ask: is firmness encoded in a specific head at a specific layer?

#### Extract directions from naturalistic generation

Run the model freely on negotiation prompts (no canned pairs). Collect activations at tokens where the model spontaneously produces firm language (e.g. tokens following "I won't go below", "my price is firm") vs. yielding language ("I can probably", "maybe we could"). The direction from naturalistic behaviour may generalise better.

#### Unsupervised direction via SVD on negotiation transcripts

Collect 200+ negotiation utterances (from CraigslistBargains or synthetic generation). Extract hidden states for all of them. Run SVD on the resulting matrix. Inspect whether the top singular vectors correspond to interpretable dimensions. This is fully unsupervised — no pairs required.

#### Expand `negotiation_steering_pairs.json` to 20+ pairs per dimension

Current: 12 pairs per dimension. Target: 20+ per dimension. More pairs → better direction estimates → stronger steering. Coordinate with P5 (split dimensions between the two contributors to avoid duplication).

---

### P2 — Validation & Probing

**Owner:** Person 2. Paper section: Related Work + Vector Validation.

#### Write `probe_vectors.py`

Trains a logistic regression classifier per layer to predict whether a text is `positive` or `negative` for a given dimension, using the saved hidden states.

```python
# For each dimension d, layer l:
#   X = np.concatenate([pos_hiddens[:, l, :], neg_hiddens[:, l, :]], axis=0)
#   y = np.array([1]*N + [0]*N)
#   clf = LogisticRegression(max_iter=1000).fit(X, y)
#   accuracy[d][l] = cross_val_score(clf, X, y, cv=5).mean()
# Plot: accuracy vs layer, one curve per dimension
```

Output: accuracy curves showing which layers encode each dimension. Tells the search pipeline where to concentrate steering.

Expected finding: accuracy rises in middle layers, peaks somewhere between 0.5–0.75 of depth, then may drop slightly in the final layers (which are more output-focused).

#### Add control probes for verbosity and formality

If the "firmness" vector is really just encoding response length or register (formal vs. informal), any observed negotiation advantage is an artefact. Build two control probes:

- **Verbosity probe:** classify short vs. long responses (split by median token count)
- **Formality probe:** classify formal vs. casual language (use a simple lexical heuristic or a small classifier)

If the firmness direction has high cosine similarity with the verbosity or formality direction, the extraction method needs rethinking.

---

### P3 — Application Strategies

**Owner:** Person 3. Paper section: Application Strategies.

#### Add `DynamicAlphaScheduler` to `apply_steering.py`

A real negotiator modulates firmness dynamically — pushes hard when the opponent pushes, softens when making an offer. Constant alpha is a blunt instrument.

```python
class DynamicAlphaScheduler:
    def __init__(self, base_alpha, schedule: str = "ramp"):
        # schedules: "ramp" (increases over turns), "reactive" (increases when opponent pushes back)

    def get_alpha(self, turn: int, transcript: list) -> float:
        if self.schedule == "ramp":
            return self.base_alpha * (1 + 0.1 * turn)
        elif self.schedule == "reactive":
            last_opp_utt = transcript[-1]["utterance"] if transcript else ""
            if any(w in last_opp_utt.lower() for w in ["can't", "won't", "too high", "too low"]):
                return self.base_alpha * 1.5
            return self.base_alpha
```

Pass `scheduler` to `run_game()` and call `scheduler.get_alpha(turn, transcript)` before each `generate_turn()` call.

#### Add `TokenSelectiveHook` to `apply_steering.py`

Current hook fires on every token. Steering is most meaningful at tokens where the model commits to a position (numeric price tokens, modal verbs like "will", "can't"). Steering on filler tokens ("the", "a") is wasteful and disrupts fluency.

```python
class TokenSelectiveHook(SteeringHook):
    PRICE_PATTERN = re.compile(r'\$?\d+')
    COMMITMENT_TOKENS = {"will", "won't", "can't", "must", "final", "firm"}

    def hook_fn(self, module, input, output):
        # Only apply if the current generation step is producing a "commitment" token
        # This requires tracking the current token being generated — hook into the logits processor
        # or use a custom stopping criterion that records which tokens triggered steering
```

Implementation note: the standard forward hook fires on every layer pass regardless of which token is being generated. To make it token-selective, you need to either (a) use a `LogitsProcessor` that tracks the current token type and enables/disables the hook, or (b) generate one token at a time and check after each step.

#### Add `ProjectionHook` to `apply_steering.py`

The additive hook `h + alpha * d` inflates the norm of `h` when alpha is large, which pushes activations out of the manifold the model was trained on and causes incoherence.

Alternative: project `h` onto the direction `d` and only modify that component.

```python
class ProjectionHook(SteeringHook):
    def hook_fn(self, module, input, output):
        h = output[0]
        d = self.direction.to(h.device, h.dtype)
        # Project h onto d, scale, and add back
        proj = (h * d).sum(dim=-1, keepdim=True) * d   # component along d
        h_modified = h - proj + (proj * self.alpha)     # scale only the d-component
        return (h_modified,) + output[1:]
```

This preserves the norm of `h` while still moving it in the target direction. Allows higher effective alpha without activation collapse.

---

### P4 — Metrics & Evaluation

**Owner:** Person 4. Paper section: Evaluation Framework + Results & Analysis.

#### Add rich behavioural metrics to `run_game()` and `score_deal()`

Current output: price outcome only. Planned additions:

```python
# In run_game() return dict, add:
{
    "offer_trajectory": [opening_bid, offer_2, offer_3, ..., agreed_price],
    "concession_rate": (first_offer - agreed_price) / (first_offer - counterpart_first_offer),
    "hedge_word_count": count of "maybe", "perhaps", "I guess", "possibly" across all utterances,
    "turns_to_deal": num_turns,
    "first_offer_distance": abs(first_offer - listing_price) / listing_price,
    "anchor_strength": first_offer relative to final agreed price,
}
```

These metrics tell us _how_ the steered agent is winning — is it anchoring more aggressively? Conceding less per turn? Closing faster? Without these, a positive advantage score is hard to interpret.

#### Write `llm_judge.py`

Takes a `results.json`, sends each transcript to an LLM API (e.g. Claude via `anthropic` SDK), and rates each transcript on three dimensions:

- **Firmness** (1–5): Does the agent hold its position under pressure?
- **Persuasiveness** (1–5): Does the agent make effective arguments for its price?
- **Naturalness** (1–5): Does the agent sound like a real human negotiating?

```python
# For each game transcript:
prompt = f"""
Rate this negotiation transcript on three dimensions (1-5 each):
- Firmness: does Agent A hold its position under pressure?
- Persuasiveness: does Agent A make effective arguments?
- Naturalness: does Agent A sound like a real human negotiator?

Transcript:
{format_transcript(game["transcript"], agent="steered")}

Respond with JSON: {{"firmness": N, "persuasiveness": N, "naturalness": N, "rationale": "..."}}
"""
```

This catches cases where a high price advantage came from degenerate behaviour (e.g. the model just refusing to say anything until the opponent caves) rather than genuine negotiation skill.

---

### P5 — Data & Baselines

**Owner:** Person 5. Paper section: Datasets & Experimental Setup.

#### Download and cache CraigslistBargains locally — URGENT

The current code fetches from CodaLab URLs on every run. These URLs are not guaranteed to remain live. If they go down mid-project, all evaluation stops.

```python
# Add to apply_steering.py and fast_search_steering.py:
CACHE_DIR = Path("data/craigslist")

def load_craigslist(split="train", num_samples=50, force_download=False):
    cache_path = CACHE_DIR / f"{split}.json"
    if cache_path.exists() and not force_download:
        with open(cache_path) as f:
            raw = json.load(f)
    else:
        raw = _fetch_json(RAW_URLS[split])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(raw, f)
    ...
```

#### Run data quality audit

Issues to check for and document:

- `seller_target <= buyer_target` (degenerate scoring, already handled but needs counting)
- `listing_price <= 0` or `seller_target <= 0` or `buyer_target <= 0` (already filtered)
- `listing_price << seller_target` (item listed way below seller's own target — data error)
- Duplicate items (same title appearing multiple times)
- Category distribution (ensure we're not over-sampled in one category)

#### Create 4 domain-balanced JSONL splits

Split the dataset into: cars, electronics, housing (furniture/appliances), bikes/sports.

```python
CATEGORY_MAP = {
    "cars": ["car", "truck", "vehicle", "auto"],
    "electronics": ["phone", "laptop", "computer", "camera", "tv"],
    "housing": ["sofa", "bed", "table", "couch", "furniture", "apartment"],
    "bikes": ["bike", "bicycle", "scooter", "cycle"],
}
```

Save as `data/craigslist/{category}.jsonl`. Sample equally from each when constructing eval sets.

#### Expand `negotiation_steering_pairs.json` alongside P1

Coordinate: P1 takes half the dimensions, P5 takes the other half. Target 20+ pairs per dimension. Focus on diverse contexts (not just price negotiation — also project scope, job offers, lease terms, partnership equity) to improve generalisation.

#### Run human baseline — 10 scenarios each, 4 people

Have 4 human participants each negotiate 10 CraigslistBargains scenarios as either buyer or seller (randomly assigned). Record their offers and final prices. Score using `score_deal()`. This establishes a human-level benchmark: does the steered LLM approach human-level negotiation performance?

This is the most important result for the paper. "Steered LLM approaches human negotiators" is a far stronger claim than "steered LLM beats unsteered LLM."

---

### P6 — Concept Patching & Interpretability

**Owner:** Person 6. Paper section: Discussion & Limitations.

#### Write `concept_patching.py`

Mid-game intervention: overwrite components of the hidden state along the steering direction at a specific turn, then measure how the model's subsequent behaviour changes.

```python
def patch_hidden_state(model, tokenizer, game_state, patch_layer, patch_direction, patch_alpha):
    """
    At a specific game turn, force the hidden state at patch_layer to move
    patch_alpha units in patch_direction. Then continue the game and measure:
    - hedge_word_count in subsequent turns
    - numeric_mention_count (how many price numbers are mentioned)
    - offer_movement (how much the agent's offer changes after patching)
    """
```

This is causal evidence. Probing tells us a direction is _encoded_; patching tells us it _controls behaviour_.

#### Run probe-steering consistency experiment

At various alpha values (e.g. 0, 1, 2, 5, 10, 20), collect hidden states from the steered agent mid-game. Project these onto the firmness direction. Plot: does the projection increase monotonically with alpha? If yes, increasing alpha genuinely increases firmness in the representation. If no, high alpha is adding noise, not steering.

#### Run negative alpha experiment — 30 games

Use `alpha = -best_alpha` with the best-found direction. If the model becomes measurably more yielding (lower steered_score, more hedge words, larger concessions), this is strong evidence the direction is causally controlling firmness.

Expected: negative alpha → steered_score significantly below 0.5 (the agent gives more than it takes).

#### Run probe degradation sweep — alpha 1 to 50

Sweep alpha from 1 to 50 in steps of 5. For each value, measure:

- agree_rate (how often a deal is reached)
- activation norm of steered hidden states (does it blow up?)
- hedge word count (does fluency degrade?)
- steered_score (does advantage first rise then fall?)

This finds the "sweet spot" range and establishes the ceiling. Provides guidance to P3 on what alpha range is realistic for the application strategy experiments.

---

## 8. Key Conventions & Gotchas

### Code conventions

- All vectors are **unit-normalised** before saving. When loading, do not re-normalise unless you have a specific reason.
- `get_transformer_layers(model)` returns `model.model.layers` — this is the correct path for all supported model families (Qwen, Llama, Gemma, Mistral all use this structure).
- Direction vectors are always stored as `float32` on CPU. Cast to model dtype and device inside the hook.
- The `HF_TOKEN` is read from environment; never hardcode tokens in committed code.

### Dataset gotchas

- CraigslistBargains `seller_target` is the **minimum** the seller will accept, not their ideal price. `buyer_target` is the **maximum** the buyer will pay. The listing price is typically higher than `seller_target`.
- `listing_price * 0.6` as the opening bid can sometimes be below `buyer_target`. This is fine — it just means the buyer opened with a bid they're actually happy to exceed.
- Some scenarios have `seller_target < buyer_target` (the targets are compatible — there's already a deal available at any price in the overlap). These produce degenerate scores and are technically valid but worth being aware of.

### Scoring gotchas

- `advantage` is only meaningful over deals that were reached. Always check `agree_rate` alongside `advantage` — a steered agent with advantage=0.8 but agree_rate=0.1 is not useful.
- The scoring does not penalise no-deals. If a steered agent walks away from more deals (because it's firmer), it might have higher `steered_score` but lower `agree_rate`. Both matter.

### Steering gotchas

- Very high alpha corrupts activations and the model stops producing coherent text. Typical safe range for 3B models: alpha 1–15. For 7B models: 1–25. The coherence filter (`agree_rate >= 0.2`) catches this automatically.
- The hook fires on every forward pass including during speculative decoding prefill. This is fine for standard `model.generate()` but may need adjustment if using other generation strategies.
- Hooks must be removed before the model is used for anything else. The `finally` block in `generate_turn()` handles this.

### Running experiments

- Always set `--use_craigslist` flag explicitly. It's required.
- For reproducibility, set `--seed` in `fast_search_steering.py`.
- Stage 2 saves to `stage2.db` — if you re-run with the same output_dir, it will resume from where it left off (Optuna `load_if_exists=True`).

---

## 9. Paper Structure

The project is intended to produce a research paper. Each person (P1–P6) is responsible for one section. The planned structure:

| Section                       | Owner | Status      |
| ----------------------------- | ----- | ----------- |
| Abstract                      | TBD   | Not started |
| Introduction                  | TBD   | Not started |
| Related Work                  | P2    | Not started |
| Datasets & Experimental Setup | P5    | Not started |
| Extraction Methods            | P1    | Not started |
| Vector Validation             | P2    | Not started |
| Application Strategies        | P3    | Not started |
| Evaluation Framework          | P4    | Not started |
| Results & Analysis            | P4    | Not started |
| Discussion & Limitations      | P6    | Not started |
| Conclusion                    | TBD   | Not started |

The key claims the paper needs to establish, in order of importance:

1. Steering with a dimension-specific vector produces higher advantage than steering with a random vector (establishes that the direction is meaningful)
2. Probing accuracy predicts which layers to steer (establishes that the method is principled)
3. The steered agent approaches or matches human-level negotiation performance (establishes the scale of the result)
4. Dynamic and projection-based application strategies outperform constant additive steering (establishes that naive steering is not the ceiling)

---

_Last updated: February 2026. Generated from codebase and task list._
