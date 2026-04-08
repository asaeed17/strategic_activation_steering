# Research Log: Activation Steering for LLM Negotiation

**Project:** COMP0087 Statistical NLP, UCL, due 2026-04-17
**Team:** Moiz, Abdullah, Damon, Ashley
**Model:** Qwen 2.5-7B (primary), Qwen 2.5-3B (Phase 1)
**Method:** Contrastive Activation Addition (mean difference extraction)

This document is the single source of truth for every experiment, finding, and design decision. Update it after every significant result.

---

## Table of Contents

1. [What We Do](#1-what-we-do)
2. [Phase 1: CraigslistBargains](#2-phase-1-craigslistbargains)
3. [The Pivot](#3-the-pivot)
4. [Phase 2: Ultimatum Game Setup](#4-phase-2-ultimatum-game-setup)
5. [Steering Pair Design](#5-steering-pair-design)
6. [Vector Validation](#6-vector-validation)
7. [Exploratory Experiments](#7-exploratory-experiments)
8. [Confirmatory Experiment](#8-confirmatory-experiment)
9. [Follow-Up Experiments](#9-follow-up-experiments)
10. [Teammate Experiments](#10-teammate-experiments)
11. [Consolidated Findings](#11-consolidated-findings)
12. [Paper Narrative](#12-paper-narrative-updated-2026-03-29)
13. [Future Directions](#13-future-directions)
14. [File Reference](#14-file-reference)

---

## 1. What We Do

We extract **steering vectors** from contrastive text pairs (e.g., "firm negotiator" vs "meek negotiator"), then inject them into a language model during inference to change its negotiation behavior. The question: **does changing behavior improve outcomes?**

**Pipeline:**

```
contrastive pairs  -->  extract direction vectors  -->  inject during inference  -->  measure behavior + outcomes
```

**Extraction:** For each behavioral dimension (e.g., firmness), we have ~20 paired texts. We run both through the model, take the last-token hidden state at each layer, and compute the mean difference. This gives a direction vector per layer.

**Injection:** During inference, we add `alpha * direction_vector` to the hidden state at a chosen layer. Alpha controls strength. Positive alpha steers toward the "positive" side of the pair; negative steers away.

---

## 2. Phase 1: CraigslistBargains

**Task:** Two LLMs negotiate over real product listings. One plays buyer, one plays seller. They chat for up to 8 turns, then agree on a price (or fail). Each has a private target price; we score by how close the agreed price is to each target.

**Model:** Qwen 2.5-3B
**Pairs:** `neg8dim_12pairs_matched` (8 merged dimensions, 12 pairs each, length-matched)
**Scale:** ~4,600 paired games, 59 configs, 8 dimensions

### What Worked

Behavioral changes were large and measurable:

| Metric            | Firmness L27 α=20               | SCM L18 α=6 |
| ----------------- | ------------------------------- | ----------- |
| Hedge suppression | 27x (1.44 → 0.05 per 100 words) | Moderate    |
| Response length   | -22%                            | +13%        |
| Dose-response     | Monotonic                       | Monotonic   |

### What Didn't Work

Outcomes were not significant:

| Config             | Advantage | p-value | Why                                  |
| ------------------ | --------- | ------- | ------------------------------------ |
| SCM L18 α=6 (best) | +0.032    | 0.87    | Noise too high                       |
| Firmness L27 α=20  | -0.04     | 0.87    | Helps buyers, hurts sellers; cancels |

### Why It Failed

1. **Clamping:** 24-70% of games produce irrational agreed prices (outside target ranges), yielding extreme ±1.0 scores. Real signal is drowned.
2. **Role asymmetry:** Steering helps buyers (+24%) but hurts sellers (-27%). Aggregating hides this. The game structure forces sellers to finalize deals.
3. **LLM-vs-LLM variance:** std=0.72 on a [-1, 1] scale. Effect sizes of d=0.2 are undetectable at n=50.
4. **1D scoring:** Price is the only outcome variable. No room for behavioral nuance.

### Phase 1 Validation Summary

We tested 8 steering pair variants to understand what makes good contrastive pairs:

| Variant                     | Dims  | Pairs  | Matching   | Score      |
| --------------------------- | ----- | ------ | ---------- | ---------- |
| neg15dim_12pairs_raw        | 15    | 12     | None       | 22/100     |
| neg15dim_12pairs_matched    | 15    | 12     | Length     | 28/100     |
| neg8dim_12pairs_raw         | 8     | 12     | None       | 25/100     |
| **neg8dim_12pairs_matched** | **8** | **12** | **Length** | **33/100** |
| neg15dim_20pairs_matched    | 15    | 20     | Length     | 26/100     |
| neg8dim_20pairs_matched     | 8     | 20     | Length     | 30/100     |
| neg15dim_80pairs_matched    | 15    | 80     | Length     | 24/100     |
| neg8dim_80pairs_matched     | 8     | 80     | Length     | 27/100     |

**Lessons:** Length-matching is the only effective intervention. More pairs hurts (surface confounds accumulate). Dimension merging helps modestly.

**Extraction method comparison:** Mean Diff >= Logistic Regression >> PCA (consistent with Im & Li 2025).

---

## 3. The Pivot

CraigslistBargains has too much noise for steering effects to surface. We need a task with:

- Clean payoff function (not clamped price ratios)
- Low variance (not 8-turn negotiation)
- Paired design (same input for steered and baseline)
- Continuous outcome (not binary accept/reject)

**Chosen task: Ultimatum Game.** One proposer splits a pool, one responder accepts or rejects. If rejected, both get $0. Single turn, clean math, 0% clamping.

**Rejected alternative: Split or Steal.** Binary outcome (cooperate/defect) gives less statistical power. RLHF alignment would push ~100% cooperation, leaving no variance.

---

## 4. Phase 2: Ultimatum Game Setup

### Game Design

```
Proposer (Qwen 7B, possibly steered):
  "You have $X to split. Propose a split. If rejected, both get $0."
  → Outputs: "I propose... OFFER=<your_share>,<their_share>"

Responder (Qwen 7B, never steered):
  "Player A proposes: they get $Y, you get $Z. Accept or Reject?"
  → Outputs: "ACCEPT" or "REJECT"
```

### Paired Design

Each game runs twice with the same pool:

1. **Steered proposer** (alpha > 0, hooks active) vs unsteered responder
2. **Baseline proposer** (alpha = 0, no hooks) vs unsteered responder

Same pool, same seed → enables paired t-test. Eliminates between-game variance.

### Variable Pools

100 diverse pool sizes ($37-$157). Without this, temp=0 produces identical outputs for every game. Variable pools force per-game computation and provide the variance needed for statistical tests.

### Key Design Decisions

| Decision     | Choice                            | Why                                               |
| ------------ | --------------------------------- | ------------------------------------------------- |
| Opponent     | LLM responder (unsteered Qwen 7B) | Rule-based threshold destroys acceptance gradient |
| Temperature  | 0.0 (primary)                     | Deterministic; variable pools provide variance    |
| Sample size  | n=100 per config                  | 100 diverse pools; honest n at temp=0             |
| Extraction   | Mean difference                   | Best empirically (Im & Li 2025)                   |
| Quantization | 4-bit NF4                         | T4 GPU (16 GB VRAM) can't fit fp16 7B model       |

### Dictator Game Variant

Same as Ultimatum Game but the responder always accepts. Tests whether the proposer adjusts behavior when there's no rejection risk.

**Critical implementation detail:** The DG prompt must tell the proposer their offer is auto-accepted. Without this, the proposer generates identical text in UG and DG (we learned this the hard way in Round 1 — see Section 8).

---

## 5. Steering Pair Design

### General-Domain Pairs (Used for All Phase 2 Experiments)

**Variant:** `ultimatum_10dim_20pairs_general_matched` (v2.1)

10 behavioral dimensions, 20 contrastive pairs each. Diverse contexts (salary negotiation, business deals, real estate) with NO game-specific tokens. Length-matched within ±30% word count.

| Dimension | Positive Example (excerpt)                                    | Negative Example (excerpt)                           |
| --------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| Firmness  | "I need at least $85K to consider this role..."               | "Well, I suppose I could consider a lower salary..." |
| Empathy   | "I understand this is a big decision for your family..."      | "The asking price is $400K. Take it or leave it..."  |
| Anchoring | "Based on market research, comparable properties sold for..." | "I'm open to whatever price you think is fair..."    |

5 control dimensions (verbosity, formality, hedging, sentiment, specificity) detect surface confounds.

### Why General Pairs Beat Game-Specific Pairs

| Pair Type                                                      | Firmness Effect (L10, UG) | Why                                                                               |
| -------------------------------------------------------------- | ------------------------- | --------------------------------------------------------------------------------- |
| **Game-specific** (`ultimatum_10dim_20pairs_matched`)          | ~0% demand shift          | Pairs hold OFFER=X,Y constant between pos/neg. Vector learns tone, not action.    |
| **General-domain** (`ultimatum_10dim_20pairs_general_matched`) | +16pp demand shift        | Diverse contexts let the vector capture the abstract concept of "demanding more." |

This is a 16pp difference from the same steering method, same model, same layer. The contrastive pairs are the dominant variable.

---

## 6. Vector Validation

### Probe Accuracy (Linear Classifier on Hidden States)

For each dimension, we train a logistic regression probe on hidden states to classify positive vs negative examples. High accuracy = the model internally represents this dimension.

| Dimension     | Probe Acc | Pattern       | Recommended Layers |
| ------------- | --------- | ------------- | ------------------ |
| Flattery      | 0.941     | Flat-high     | —                  |
| Firmness      | 0.941     | Flat-high     | —                  |
| Undecidedness | 0.954     | Flat-high     | —                  |
| Empathy       | 0.946     | Rising (L5-6) | —                  |
| Fairness_norm | 0.978     | Rising (L7+)  | —                  |
| Anchoring     | 0.875     | Flat-high     | [18, 19, 20]       |
| Composure     | 0.895     | Flat-high     | —                  |
| Narcissism    | 0.869     | Flat-high     | —                  |
| Greed         | 0.813     | Flat-high     | —                  |
| Spite         | 0.805     | Flat-high     | —                  |

Overall validation score: **27/100** (8 RED, 2 AMBER). This sounds bad but overstates the problem — see next section.

### Orthogonal Projection (Do Vectors Capture Real Concepts?)

We project out all 5 control dimensions from each negotiation vector and re-run probes. If accuracy drops sharply, the vector was just capturing surface features (e.g., length, hedging). If it holds, the concept is genuine.

| Dimension     | Before | After | Drop     | Verdict                             |
| ------------- | ------ | ----- | -------- | ----------------------------------- |
| Flattery      | 0.941  | 0.954 | Improved | GENUINE                             |
| Narcissism    | 0.869  | 0.889 | Improved | GENUINE                             |
| Spite         | 0.805  | 0.829 | Improved | GENUINE                             |
| Firmness      | 0.941  | 0.921 | -2.0%    | GENUINE                             |
| Fairness_norm | 0.879  | 0.872 | -0.7%    | GENUINE                             |
| Anchoring     | 0.875  | 0.877 | -0.2%    | GENUINE                             |
| Composure     | 0.895  | 0.867 | -2.8%    | GENUINE                             |
| Greed         | 0.813  | 0.799 | -1.4%    | GENUINE                             |
| Empathy       | 0.864  | 0.811 | -5.4%    | PARTIAL (sentiment overlap)         |
| Undecidedness | 0.954  | 0.804 | -15.1%   | SURFACE-DOMINATED (hedging overlap) |

**8/10 dimensions are GENUINE.** Mean probe drop is only 2.3%. The validation score (27/100) dramatically overstates contamination because Cohen's d detects data confounds, not direction confounds.

### Cosine Similarity Between Dimensions (L10)

Key overlaps that affect experiment interpretation:

| Pair                     | cos @ L10 | cos @ L12 | Implication                               |
| ------------------------ | --------- | --------- | ----------------------------------------- |
| Firmness ↔ Empathy       | -0.287    | -0.349    | Anti-empathy ≈ pro-firmness direction     |
| Empathy ↔ Flattery       | +0.607    | +0.583    | Empathy vector partially encodes flattery |
| Greed ↔ Narcissism       | +0.544    | +0.569    | Cluster together                          |
| Firmness ↔ Undecidedness | -0.373    | -0.414    | Opposite ends of assertiveness axis       |

The firmness-empathy anti-correlation is critical: it means steering "away from empathy" (negative alpha) partially steers "toward firmness."

---

## 7. Exploratory Experiments

Before the confirmatory experiment, teammates ran ~476 configs across multiple batches with various pair types, layers, alphas, and temperatures.

### Exploratory Results Summary (BH-FDR corrected)

| Metric       | Configs Tested | Significant (p < 0.05, corrected) | Rate |
| ------------ | -------------- | --------------------------------- | ---- |
| Demand shift | 476            | 266                               | 56%  |
| Payoff delta | 476            | 134                               | 28%  |

Demand shifts are robust and survive correction. Payoff effects are weaker — many disappear after correction. This is the demand≠payoff divergence that motivated the confirmatory experiment.

### Key Exploratory Findings

1. **Firmness is the strongest dimension** for demand shift across all pair types and layers.
2. **L10 and L12 are the active layers.** L14+ produces near-zero effects.
3. **Temperature matters:** temp=0.7 adds noise; temp=0.0 is cleaner but needs variable pools.
4. **Game-specific pairs produce weaker effects** than general pairs (see Section 5).

---

## 8. Confirmatory Experiment (Round 1)

**Date:** 2026-03-27 to 2026-03-28
**GPU:** g4dn.xlarge (T4 16GB, 4-bit quantized), eu-west-2
**Pairs:** `ultimatum_10dim_20pairs_general_matched` (general-domain, v2.1)
**Design:** 100 diverse pools, temp=0, paired, n=100 per config

### Pre-Registered Hypotheses

| ID  | Claim                                       | Test                      | Direction          |
| --- | ------------------------------------------- | ------------------------- | ------------------ |
| H1  | Empathy steering shifts demand DOWN         | Paired t-test (one-sided) | steered < baseline |
| H2  | Firmness steering shifts demand UP          | Paired t-test (one-sided) | steered > baseline |
| H3  | Empathy steering improves payoff            | Paired t-test (one-sided) | steered > baseline |
| H4  | Firmness steering does NOT improve payoff   | TOST equivalence          | \|delta\| < 5%     |
| H5  | Firmness demand shift is equal in UG and DG | TOST equivalence          | \|UG - DG\| < 5%   |

Correction: Benjamini-Hochberg FDR over H1-H5 at alpha=0.05.

### Configs Run

**UG (12 configs):**

- Firmness: L{10,12} × alpha={3, 7, 10} (positive alpha → demand UP)
- Empathy: L{10,12} × alpha={-3, -7, -10} (negative alpha → expected demand DOWN)

**DG (6 configs):**

- Firmness: L{10,12} × alpha={3, 7, 10}

**Robustness (1 config):**

- Empathy L12 alpha=-7 at temp=0.3

### Results: UG Firmness

| Layer | Alpha | Demand Shift | d    | p      | Acceptance     | Payoff Delta |
| ----- | ----- | ------------ | ---- | ------ | -------------- | ------------ |
| 10    | 3     | +10.1pp      | 1.07 | <0.001 | 88.9% vs 83.8% | +11.8pp      |
| 10    | 7     | +16.1pp      | 1.38 | <0.001 | 88.8% vs 84.7% | +15.9pp      |
| 10    | 10    | +23.1pp      | 1.54 | <0.001 | 83.0% vs 85.1% | +16.5pp      |
| 12    | 3     | +7.6pp       | 0.73 | <0.001 | 91.0% vs 84.0% | +10.5pp      |
| 12    | 7     | +13.2pp      | 1.21 | <0.001 | 88.9% vs 84.8% | +13.1pp      |
| 12    | 10    | +18.7pp      | 1.35 | <0.001 | 85.9% vs 84.8% | +15.4pp      |

All configs have perfectly monotonic dose-response (Spearman rho = 1.0).

### Results: UG Empathy (Negative Alpha)

| Layer | Alpha | Demand Shift | d    | p      | Acceptance     | Payoff Delta |
| ----- | ----- | ------------ | ---- | ------ | -------------- | ------------ |
| 10    | -3    | **+10.8pp**  | 1.18 | <0.001 | 91.9% vs 83.8% | +14.1pp      |
| 10    | -7    | **+12.9pp**  | 1.33 | <0.001 | 90.9% vs 83.8% | +15.0pp      |
| 10    | -10   | **+14.2pp**  | 1.44 | <0.001 | 88.8% vs 83.7% | +14.5pp      |
| 12    | -3    | **+9.2pp**   | 0.87 | <0.001 | 89.0% vs 84.0% | +11.0pp      |
| 12    | -7    | **+11.8pp**  | 1.14 | <0.001 | 90.9% vs 84.8% | +13.7pp      |
| 12    | -10   | **+14.4pp**  | 1.40 | <0.001 | 84.8% vs 84.8% | +11.6pp      |

**Surprise: Demand went UP, not down.** H1 is rejected. Both empathy signs increase demand (see Section 9 for positive alpha results).

### Results: DG (Round 1 — Bug)

Round 1 DG used the same proposer prompt as UG ("Player B will either Accept or Reject"). The proposer didn't know it was a Dictator Game, so it produced **byte-for-byte identical** outputs. Demand shifts were trivially identical to UG. This was a bug. See Section 9 for the fixed DG results.

### Robustness Check

Empathy L12 alpha=-7 at temp=0.0 vs temp=0.3:

| Metric               | temp=0.0 | temp=0.3 |
| -------------------- | -------- | -------- |
| Demand shift         | +11.8pp  | +11.6pp  |
| Cohen's d            | 1.14     | 1.08     |
| Acceptance (steered) | 91%      | 89%      |

Results are robust to temperature.

### Hypothesis Verdicts (Round 1)

| H   | Verdict            | p_adj  | What Happened                              |
| --- | ------------------ | ------ | ------------------------------------------ |
| H1  | **REJECTED**       | 1.0    | Demand went UP (+9.2pp), not down          |
| H2  | **SUPPORTED**      | <0.001 | d=0.73, monotonic dose-response            |
| H3  | **SUPPORTED**      | <0.001 | d=0.45, payoff improved                    |
| H4  | **REJECTED**       | 1.0    | Payoff ALSO improved (+10.5pp)             |
| H5  | **TRIVIALLY TRUE** | <0.001 | Bug: identical prompts → identical outputs |

3/5 supported, but H4 and H5 need reinterpretation (see below).

### Why H4 Failed: The Baseline Is Suboptimal

The baseline Qwen 7B at temp=0 produces only 12 unique proposer texts across 100 games. It offers ~50/50 splits with vague framing like:

> "I propose a fair split to ensure Player B is willing to accept. OFFER=47,47"

The LLM responder **rejects 16% of these near-equal offers** because the vague fairness-signaling text triggers its own fairness-reasoning mode:

> "While the offer is quite close, it's important to consider the principle of fairness and the potential for future interactions. REJECT"

The 16 rejected baseline offers have mean demand of 50.5-51.2% — the responder rejects **near-equal splits** because of text framing, not unfair numbers.

**Any steering** replaces vague text with explicit text ("I propose a split where I keep $X and Player B gets $Y"), which makes the responder more rational. This is why ALL configs improve payoff — not because of dimension-specific reasoning, but because explicit framing > vague framing.

---

## 9. Follow-Up Experiments

### Round 2: Fixed Dictator Game (2026-03-28)

**Fix:** `build_proposer_system(pool, game)` now passes the game type. DG prompt says "Your offer is automatically accepted. Player B cannot reject."

**Baseline change:** DG baseline demand is 59.2% (vs 52.4% UG). The model correctly demands more when told there's no rejection risk.

**Firmness Results (UG vs Fixed DG):**

| Layer  | Alpha | UG Demand Shift | DG Demand Shift | Interpretation |
| ------ | ----- | --------------- | --------------- | -------------- |
| **10** | 3     | +10.1pp         | **-0.5pp**      | No effect      |
| **10** | 7     | +16.1pp         | **-5.8pp**      | **REVERSED**   |
| **10** | 10    | +23.1pp         | **-5.0pp**      | **REVERSED**   |
| **12** | 3     | +7.6pp          | -2.1pp          | Weak/opposite  |
| **12** | 7     | +13.2pp         | **+15.3pp**     | Consistent     |
| **12** | 10    | +18.7pp         | **+25.5pp**     | Even stronger  |

**Finding: L10 firmness reverses direction in DG.** The same vector that means "demand aggressively" in UG means "propose fairly" in DG. The adversarial framing (rejection risk) is what makes L10 interpret firmness as aggression. Without it, the model interprets firmness as principled fairness.

**L12 firmness is context-independent.** It increases demand regardless of game structure. At L12 alpha=10 in DG: demand = 84.7% of pool (d=2.46). The vector encodes "take more" as an abstract disposition.

**This replaces H5 with a much stronger finding:** Steering effects are both layer-dependent AND context-dependent. L10 = context-sensitive style/tone layer. L12 = context-independent behavioral disposition layer.

### Round 3: Positive Empathy (2026-03-28)

**Question:** Does steering TOWARD empathy (positive alpha) decrease demand?

**Results:**

| Layer | Alpha | Demand Shift | d    | Acceptance | Payoff Delta |
| ----- | ----- | ------------ | ---- | ---------- | ------------ |
| 10    | +3    | +10.1pp      | 0.98 | 93.0%      | +14.0pp      |
| 10    | +7    | +11.4pp      | 1.16 | **94.9%**  | **+17.0pp**  |
| 10    | +10   | +11.7pp      | 1.22 | 93.9%      | +16.2pp      |
| 12    | +3    | +3.4pp       | 0.37 | 95.0%      | +9.0pp       |
| 12    | +7    | +5.2pp       | 0.63 | 95.0%      | +10.9pp      |
| 12    | +10   | +5.3pp       | 0.61 | 95.0%      | +11.0pp      |

**Answer: No. Both positive and negative empathy increase demand.** The empathy vector is NOT bidirectional for demand. It encodes "activation/engagement" — steering in either direction makes the proposer more active, which in this task means demanding more.

**But the sign matters for tone.** Positive empathy gets 93-95% acceptance; negative empathy gets 85-92%. The responder perceives positive-empathy text as warmer and accepts more readily, even though the numerical offer is similar.

**Best overall config: Empathy L10 alpha=+7.** Demand +11.4pp, acceptance 94.9%, payoff **+17.0pp**. This is the highest payoff improvement of any config tested. The mechanism: moderate demand increase + highest acceptance rate = best expected value.

### Comparison: Positive vs Negative Empathy at L10

| Alpha | Neg Demand | Pos Demand | Neg Accept | Pos Accept | Neg Payoff | Pos Payoff |
| ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 3     | +10.8pp    | +10.1pp    | 91.9%      | 93.0%      | +14.1pp    | +14.0pp    |
| 7     | +12.9pp    | +11.4pp    | 90.9%      | 94.9%      | +15.0pp    | +17.0pp    |
| 10    | +14.2pp    | +11.7pp    | 88.8%      | 93.9%      | +14.5pp    | +16.2pp    |

Negative alpha pushes demand slightly higher but acceptance lower. Positive alpha is more moderate but better accepted. Net: positive is strategically better.

### Comparison: L10 vs L12 for Positive Empathy

| Alpha | L10 Demand | L12 Demand | L10 Accept | L12 Accept |
| ----- | ---------- | ---------- | ---------- | ---------- |
| 3     | +10.1pp    | +3.4pp     | 93%        | 95%        |
| 7     | +11.4pp    | +5.2pp     | 95%        | 95%        |
| 10    | +11.7pp    | +5.3pp     | 95%        | 95%        |

L10 has 2-3x the demand effect of L12 for empathy. L12 acceptance is slightly higher. This mirrors the firmness finding: L10 amplifies more aggressively.

### Round 4: Empathy DG (2026-03-29)

**Question:** Does empathy also show the L10 context-reversal found for firmness? Or is that firmness-specific?

**Initial result (2 configs, α=7 only, QUANTIZED — later shown to be confounded):**

| Config           | UG Δ demand | DG Δ demand (quantized) | Apparent Pattern |
| ---------------- | ----------- | ----------------------- | ---------------- |
| Firmness L10 α=7 | +16.1pp     | **-5.8pp** (d=-0.65)    | Reversal         |
| Firmness L12 α=7 | +13.2pp     | **+15.3pp** (d=0.96)    | Persistent       |
| Empathy L10 α=+7 | +11.4pp     | **-0.5pp** (p=0.23, NS) | Nullified        |
| Empathy L12 α=+7 | +5.2pp      | **-1.1pp** (d=-0.24)    | Nullified        |

### Round 7: Empathy DG Confound Resolution (2026-03-31)

**The "nullification" was a quantization artifact.** Reran all 6 empathy DG configs on the same machine (mallard-l, RTX 3090 Ti) with same dtype (bfloat16, no quantization). Baselines are now consistent: 77.29-77.44% (0.15pp spread).

| Layer | Alpha | DG Δ demand | d     | p       | DG baseline |
| ----- | ----- | ----------- | ----- | ------- | ----------- |
| 10    | 3     | **-25.0pp** | -3.06 | <1e-300 | 77.4%       |
| 10    | 7     | **-15.0pp** | -1.29 | <1e-300 | 77.4%       |
| 10    | 10    | **-13.5pp** | -1.15 | <1e-300 | 77.3%       |
| 12    | 3     | **-18.6pp** | -1.64 | <1e-300 | 77.4%       |
| 12    | 7     | **-22.7pp** | -2.58 | <1e-300 | 77.3%       |
| 12    | 10    | **-22.8pp** | -2.68 | <1e-300 | 77.3%       |

**Compare quantized vs unquantized at α=7:**

| Config  | Quantized (Round 4) | Unquantized (Round 7) | Quantization effect          |
| ------- | ------------------- | --------------------- | ---------------------------- |
| L10 α=7 | d=-0.12 (NS)        | **d=-1.29** (p<1e-300) | Killed a d=1.29 effect       |
| L12 α=7 | d=-0.24 (marginal)   | **d=-2.58** (p<1e-300) | Killed a d=2.58 effect       |

**4-bit NF4 quantization suppressed a massive steering effect.** The quantized model's DG baseline (59%) vs unquantized (77%) shows the model's "fairness" disposition changes dramatically with precision. Steering vectors extracted from the unquantized model work much better when applied to the unquantized model.

**Empathy REVERSES in DG (not nullified).** All 6 configs show large negative demand shifts (d=-1.15 to -3.06). In UG, empathy increases demand (+5 to +11pp). In DG, empathy decreases demand (-13 to -25pp). This is the same context-reversal pattern as firmness L10.

**Dose-response patterns differ by layer:**
- **L10:** Non-monotonic. Strongest at α=3 (-25pp), weaker at α=7 (-15pp) and α=10 (-14pp). Low alpha drives larger behavioral change.
- **L12:** Monotonic. α=3 (-18.6pp) → α=7 (-22.7pp) → α=10 (-22.8pp), plateauing at α=7.

**Revised two-pattern taxonomy:**

| Pattern         | Where                         | Meaning                                           |
| --------------- | ----------------------------- | ------------------------------------------------- |
| **Reversal**    | Firmness L10, Empathy L10+L12 | Context-dependent: demands more in UG, less in DG |
| **Persistence** | Firmness L12 only             | Context-independent: demands more regardless      |

The "nullification" category is eliminated. Only firmness L12 persists across contexts. Everything else reverses. The key mechanistic question becomes: what makes firmness L12 special?

**Methodological warning: quantization can destroy steering effects.** This is a significant finding for the field. Any activation steering study using quantized models should validate against unquantized baselines. The 4-bit NF4 quantization changed both the model's baseline behavior (59% → 77% demand in DG) and the effectiveness of steering vectors (d=-0.12 → d=-1.29).

### Phase B: Acceptance Curve (2026-03-29)

**What:** Swept 7 offer levels (20-80% to responder) across 100 pools with unsteered LLM responder. 700 calls, temp=0.

| Offer to Responder | P(accept) | n   |
| ------------------ | --------- | --- |
| 20%                | 0.700     | 100 |
| 30%                | **0.940** | 100 |
| 40%                | **0.930** | 100 |
| 50%                | 0.770     | 100 |
| 60%                | 0.910     | 100 |
| 70%                | 0.760     | 100 |
| 80%                | 0.750     | 100 |

**The curve is non-monotonic.** A rational agent would accept more as offers improve. Instead: peak acceptance at 30-40% to responder (proposer keeps 60-70%), with dips at 50% (fairness rounding artifact) AND at 70-80% (rejects generous offers on fairness grounds).

**The 50% dip:** Odd-numbered pools give proposer $1 more via rounding → 86% rejection rate (19/22 odd pools). Exactly equal splits: 0% rejection. The model detects $1 inequality and punishes it.

**The 70-80% rejection:** The model rejects offers that FAVOR it because they violate the 50/50 norm. 24/25 rejections at 80% cite "fairness." RLHF alignment creates bidirectional fairness enforcement.

**Implication:** The optimal proposer demand is 60-70% of pool (offer responder 30-40%), which is exactly where steered proposers land. The baseline at 50% sits in the worst possible spot.

### Phase C: Analytical Payoff Decomposition (2026-03-29)

**What:** Computed E[payoff] = demand × P(accept|demand) analytically using the acceptance curve, compared to observed payoff from LLM-vs-LLM games.

**Result: Framing effect ≈ 0.** Mean framing effect = +0.03pp across 18 configs. 15/18 within ±2pp.

**However, this is trivially true by design:** The responder prompt (`build_responder_system`) only passes the parsed OFFER numbers — the responder never sees the proposer's reasoning text. The framing is stripped before the responder evaluates. So payoff is entirely determined by the numerical demand, not persuasion.

**This means:** In single-turn UG, steering works purely through the numbers (what the model asks for). Whether steering also affects persuasion/framing can only be tested in multi-turn negotiation where both sides see each other's full text.

### Round 5: L14 Adjudication (2026-03-29)

**Question:** Teammate data shows d=-3.05 at L14 for empathy. Does this replicate in our clean setup?

| Dimension | Layer | Alpha | Demand Shift | d    | p      | Acceptance     | Payoff Δ |
| --------- | ----- | ----- | ------------ | ---- | ------ | -------------- | -------- |
| Firmness  | 14    | 7     | +14.2pp      | 0.57 | <0.001 | 84.5% vs 85.6% | +10.5pp  |
| Empathy   | 14    | 7     | +3.3pp       | 0.14 | 0.171  | 86.7% vs 85.7% | +3.7pp   |

**Firmness L14 α=7:** Significant but noisy (d=0.57, std_delta=24.75pp). Compare: L10 d=1.38, L12 d=1.21. L14 is weaker but not dead — effect attenuates with layer depth. High variance suggests some games steered strongly, others not at all.

**Empathy L14 α=7:** Not significant (d=0.14, p=0.17). The empathy vector is effectively inactive at L14.

**Teammate's d=-3.05 NOT replicated.** Our empathy L14 gives d=+0.14 (NS), with the opposite sign. The discrepancy is consistent with the known design differences (different prompt, different alpha selection, different baselines — see Section 10). The teammate result should be interpreted cautiously; it may reflect their "aim" prompt interacting with L14 in ways our prompt does not.

**Layer gradient (firmness UG, α=7):**

| Layer | d    | Interpretation                |
| ----- | ---- | ----------------------------- |
| 10    | 1.38 | Strongest (context-sensitive) |
| 12    | 1.21 | Strong (context-independent)  |
| 14    | 0.57 | Moderate (attenuated)         |

Effects attenuate but don't disappear at L14 for firmness. This is consistent with the exploratory finding that L14+ is mostly inactive — "mostly" rather than "entirely."

### Round 6: Text-Visibility Control (2026-03-29)

**Question:** Steering payoff gains rely on the responder only seeing parsed OFFER numbers. What happens when the responder sees the proposer's full reasoning text?

**Design:** Standard UG with `--text_visible` flag. The responder prompt now includes: `Player A said: "<full proposer output>"` before the parsed offer. Three configs: baseline, firmness L12 α=7, empathy L10 α=+7.

**Results:**

| Config                 | Demand Δ | Accept (steered) | Accept (baseline) | Payoff Δ    |
| ---------------------- | -------- | ---------------- | ----------------- | ----------- |
| TV baseline (no steer) | —        | 91.0%            | —                 | —           |
| TV firmness L12 α=7    | +13.0pp  | **54.5%**        | 90.9%             | **-11.5pp** |
| TV empathy L10 α=+7    | +11.6pp  | **45.5%**        | 90.9%             | **-19.1pp** |

**Comparison with numbers-only mode (same configs):**

| Config           | Numbers-Only Accept | Text-Visible Accept | Numbers-Only Payoff Δ | Text-Visible Payoff Δ |
| ---------------- | ------------------- | ------------------- | --------------------- | --------------------- |
| Firmness L12 α=7 | 88.9%               | **54.5%**           | **+13.1pp**           | **-11.5pp**           |
| Empathy L10 α=+7 | 94.9%               | **45.5%**           | **+17.0pp**           | **-19.1pp**           |

**Finding: Steered text is massively counterproductive.** Demand shifts are similar (~13pp firmness, ~12pp empathy), so steering still changes what the model asks for. But acceptance rates crash by 34-49pp when the responder can read the reasoning. All payoff gains reverse to losses.

**The framing effect is real, large, and negative.** The "framing ≈ 0" result (Section 9, Phase C) was an artifact of the numbers-only design. When text is visible:

- Steered proposers demand more (+11-13pp) — same as before
- But their reasoning text reveals the greediness, triggering rejection
- Net payoff drops by 25-36pp compared to numbers-only mode

**Text visibility also HELPS the baseline.** Baseline acceptance is 91% text-visible vs 84% numbers-only. The baseline's "fair split" reasoning text reassures the responder. Text visibility is a double-edged sword: it helps cooperative offers but punishes greedy ones.

**Empathy text is rejected more than firmness text** (45.5% vs 54.5% acceptance). Speculative explanation: empathy-steered text is perceived as manipulative ("I understand your position... but I should get more"), which triggers stronger rejection than blunt firmness ("I propose keeping $X").

**Implication for the paper narrative:** The "steering improves payoff" claim must be bounded: it improves payoff in numbers-only mode where the opponent cannot evaluate the proposer's reasoning. In any setting with text communication (multi-turn, full-text single-turn), steering likely hurts payoff because the opponent detects and punishes the greedy reasoning. This is an important external validity caveat.

### Round 7: n=200 Replication With Extended Pools (2026-04-01)

**Question:** Do the top 5 empathy UG configs replicate at n=200 with a wider pool range? Does effect size hold when extrapolating beyond the original $37-$157 pool range?

**Config selection:** The 5 configs with the highest payoff delta from Round 3 (positive empathy), ranked by payoff improvement:

| Rank | Config   | Round 3 Payoff Δ | Why Selected                                      |
| ---- | -------- | ---------------- | ------------------------------------------------- |
| 1    | L10 α=7  | +17.0pp          | Best overall config (highest payoff, 95% accept)  |
| 2    | L10 α=10 | +16.2pp          | Strongest demand shift at L10                     |
| 3    | L10 α=3  | +14.0pp          | Lowest alpha with large effect                    |
| 4    | L12 α=10 | +11.0pp          | Best L12 config                                   |
| 5    | L12 α=7  | +10.9pp          | Second-best L12 config                            |

Excluded: L12 α=3 (+9.0pp payoff Δ, weakest of the 6 Round 3 configs, d=0.37).

**Design:** Same setup as Round 3 (positive empathy, proposer steering, paired, temp=0, float16) but with `n_games=200`. The 200-game pool sequence extends the range to $37-$273 (mean $154) vs the original 100-game range of $37-$157 (mean $95). All other parameters identical (same seed, same vectors, same model).

**Results:**

| Layer | Alpha | Demand Δ | d    | Accept (S/B)  | Payoff Δ  |
| ----- | ----- | -------- | ---- | ------------- | --------- |
| 10    | 3     | +3.2pp   | 0.15 | 86.0% / 82.5% | +4.1pp    |
| 10    | 7     | +6.2pp   | 0.29 | 86.0% / 82.5% | +6.5pp    |
| 10    | 10    | +8.5pp   | 0.39 | 84.5% / 82.5% | +7.8pp    |
| 12    | 7     | +5.4pp   | 0.26 | 85.0% / 82.5% | +5.7pp    |
| 12    | 10    | +5.4pp   | 0.26 | 92.0% / 82.5% | +9.4pp    |

**Comparison with Round 3 (n=100, $37-$157 pools):**

| Config   | Round 3 Δ demand | Round 7 Δ demand | Round 3 d | Round 7 d | Round 3 payoff Δ | Round 7 payoff Δ |
| -------- | ---------------- | ---------------- | --------- | --------- | ---------------- | ---------------- |
| L10 α=3  | +10.1pp          | +3.2pp           | 0.98      | 0.15      | +14.0pp          | +4.1pp           |
| L10 α=7  | +11.4pp          | +6.2pp           | 1.13      | 0.29      | +16.3pp          | +6.5pp           |
| L10 α=10 | +11.7pp          | +8.5pp           | 1.19      | 0.39      | +15.6pp          | +7.8pp           |
| L12 α=7  | +5.1pp           | +5.4pp           | 0.63      | 0.26      | +10.9pp          | +5.7pp           |
| L12 α=10 | +5.3pp           | +5.4pp           | 0.61      | 0.26      | +11.1pp          | +9.4pp           |

**Key observations:**

1. **Effects replicate directionally** — all 5 configs show significant positive demand shifts and payoff gains. Direction is 100% consistent.

2. **Magnitude attenuates substantially at L10.** Demand deltas shrink from +10-12pp to +3-9pp; Cohen's d drops from ~1.0-1.2 to ~0.15-0.39. Payoff gains halve from +14-16pp to +4-8pp. The attenuation is concentrated at L10.

3. **L12 is relatively stable.** L12 demand deltas (+5.1-5.3pp → +5.4pp) are nearly identical across pool ranges. The L12 effect appears pool-invariant — it shifts demand by ~5pp regardless of stakes. This is consistent with the "context-independent disposition" interpretation from the DG dissociation.

4. **L10 dose-response steepens.** In the original pools, L10 α=3/7/10 produce similar deltas (+10.1/+11.4/+11.7pp — near-saturated). With extended pools, dose-response separates clearly (+3.2/+6.2/+8.5pp). The larger pool range breaks the ceiling effect and reveals the true dose-response gradient.

5. **Baseline drops with larger pools.** Baseline demand is 49.2% (extended) vs 52.3% (original). With higher-stakes pools, the model defaults closer to even splits — consistent with risk aversion scaling with stakes.

6. **L12 α=10 acceptance anomaly persists.** This config gets 92.0% acceptance (vs 82.5% baseline and 84-86% for other steered configs). The same anomaly appeared in Round 3 (95.0%). L12 α=10 uniquely combines moderate demand increase with high acceptance — mechanism unclear but reproducible.

**Interpretation:** The original n=100 results (pool range $37-$157) overstate the effect for L10 because the model has seen similar stakes in training and steering can exploit learned patterns. Extended pools ($158-$273) are out-of-distribution for the original pool sequence, and L10's context-sensitive mechanism is less effective at unfamiliar stakes. L12's context-independent mechanism is robust to pool range.

**For the paper:** The n=200 extended-pool results are the more conservative and defensible estimates. They represent generalization to unseen pool sizes. The n=100 results are "best case" within the training distribution. Report both, or conservatively use the n=200 numbers. The dose-response at L10 with extended pools (+3.2/+6.2/+8.5pp) is actually cleaner than the original (which was near-saturated).

**Result files:** `results/ultimatum/top5_200games/empathy_proposer_L{10,12}_a{3,7,10}.0_paired_n200.json`

### Statistical Hardening (2026-03-29)

Applied bootstrap CIs (10,000 resamples), TOST equivalence tests (ε=5pp), and BH-FDR correction across all configs.

**Key results:**

- **18/18 UG configs:** Significant for both demand and payoff after FDR correction
- **TOST confirms nulls:** Empathy DG α=7 at both layers bounded within ±5pp (L10 p_tost=7.5e-24, L12 p_tost=5.9e-14). Framing effect bounded within ±5pp across all 18 configs (p_tost=2.8e-11)
- **Dose-response monotonicity:** 5/6 dimension×layer combos are perfectly monotonic (Spearman ρ=1.0). Exception: empathy L12 (ρ=0.87, still monotonic trend but α=7 and α=10 are nearly tied at 5.2pp and 5.3pp)
- **Unique offer diversity:** Mean 84% unique offer pairs across configs. Range: 59% (empathy L12 α=3, weakest config) to 100%. Confirms variable-pool design generates genuine between-game variance
- Full results in `results/ultimatum/statistical_hardening.json`

### Interpretability Suite (2026-04-08)

**Goal:** Add mechanistic evidence without relying on fragile GPU forward-hook analyses. The revised suite is CPU-first and operates on saved vectors plus finalized behavioral JSONs. Practically, each test asks a different question:

- **Cosine evolution:** Are two named dimensions actually the same direction, opposite directions, or something in between?
- **PCA / effective dimensionality:** Do the 10 dimensions span 10 genuinely independent mechanisms, or a smaller shared negotiation subspace?
- **Control contamination vs effect:** Are the strongest behavioral effects just surface confounds like hedging, sentiment, or verbosity?
- **Logit lens:** Can we read off a human-interpretable vocabulary signature from the raw vector itself?

**Files:**

- `results/interpretability/cosine_evolution.json`
- `results/interpretability/pca_analysis.json`
- `results/interpretability/contamination_analysis.json`
- `results/interpretability/logit_lens_results.json`
- `results/interpretability/figures/fig9_cosine_heatmaps.*`
- `results/interpretability/figures/fig10_cosine_evolution.*`
- `results/interpretability/figures/fig11_pca_steering_space.*`
- `results/interpretability/figures/fig12_contamination_vs_effect.*`

#### Test 1: Cosine Similarity Evolution Across Layers

**What it measures:** Pairwise cosine similarity between all 10 negotiation vectors at each layer.

**Practical relevance:** This tells us whether two behavior labels are really distinct mechanisms or just near-duplicates with different names. If two vectors are highly aligned, they may steer the model through similar internal channels. If they are opposed, then one dimension is effectively the inverse of another.

**Tracked pairs across depth:**

| Pair | L4 | L10 | L12 | L20 | Interpretation |
| ---- | -- | --- | --- | --- | -------------- |
| Firmness ↔ Empathy | -0.517 | -0.287 | -0.349 | -0.284 | Consistently opposed, but not identical inverses |
| Greed ↔ Narcissism | +0.546 | +0.544 | +0.569 | +0.598 | Stable cluster, strengthens with depth |
| Empathy ↔ Flattery | +0.766 | +0.607 | +0.583 | +0.446 | Very similar early, then progressively disentangle |
| Firmness ↔ Undecidedness | -0.647 | -0.373 | -0.414 | -0.213 | Strong assertiveness-vs-hesitation opposition |

**Additional layer-specific structure:**

- **L4:** strongest negative pair is firmness ↔ undecidedness (-0.647); strongest positive pair is empathy ↔ flattery (+0.766).
- **L10:** strongest negative pair is fairness_norm ↔ undecidedness (-0.476); strongest positive pair is empathy ↔ flattery (+0.607), with greed ↔ narcissism close behind (+0.544).
- **L12:** same broad geometry as L10, but firmness ↔ empathy becomes slightly more negative and greed ↔ narcissism slightly more positive.
- **L20:** fairness_norm separates sharply from greed (-0.588) and narcissism (-0.436), while greed ↔ narcissism remains the clearest positive cluster (+0.598).

**Interpretation:** The dimensions occupy **distinct but overlapping subspaces** rather than isolated one-hot directions. The layer dependence matters. Empathy and flattery are almost interchangeable early, but not later. Greed and narcissism remain tightly coupled at every layer. Firmness and empathy are anti-correlated, but not so strongly that one can be reduced to the other. This is important for the paper because the DG results already showed that empathy and firmness behave differently despite their negative cosine overlap.

**Most useful claim for the paper:** different behavioral dimensions are geometrically related in systematic ways, and those relations shift with network depth rather than staying fixed.

#### Test 2: Effective Dimensionality (PCA of Steering Space)

**What it measures:** At each layer, stack the 10 negotiation vectors and ask how many principal components are needed to explain 90% of their variance.

**Practical relevance:** This tells us whether the 10 dimensions are really 10 independent steering handles. If the answer were 2-3, then many dimensions would just be aliases of the same latent factor. If the answer were 10, they would be almost fully independent. The true answer is in between.

**Key numbers:**

| Layer | Components for 90% variance | PC1 variance | PC2 variance | Cumulative variance by PC4 |
| ----- | --------------------------- | ------------ | ------------ | -------------------------- |
| 0     | 7                           | 39.0%        | 15.7%        | 76.1%                      |
| 4     | 6                           | 42.0%        | 17.5%        | 79.8%                      |
| 10    | 7                           | 34.3%        | 15.0%        | 73.6%                      |
| 12    | 7                           | 32.4%        | 16.1%        | 73.0%                      |
| 20    | 7                           | 31.7%        | 19.9%        | 72.5%                      |
| 27    | 6                           | 53.6%        | 13.5%        | 84.1%                      |

**L10 PCA scatter (most behaviorally active mid-layer):**

- `firmness` = (-0.714, +0.313)
- `fairness_norm` = (-0.794, -0.346)
- `spite` = (-0.459, -0.553)
- `composure` = (-0.449, +0.459)
- `empathy` = (+0.572, -0.398)
- `flattery` = (+0.468, -0.265)
- `undecidedness` = (+0.662, +0.113)
- `greed` = (+0.291, +0.432)
- `narcissism` = (+0.319, +0.123)
- `anchoring` = (+0.104, +0.123)

**Interpretation:** The steering space is **low-rank but not trivial**. Across most layers, **6-7 principal components** are enough to explain 90% of the variance. So the dimensions are not fully independent, but they also do not collapse to a single “good vs bad negotiator” axis.

The practical picture at L10 is:

- `greed` and `narcissism` sit close together, consistent with their similar behavioral profile.
- `empathy`, `flattery`, and `undecidedness` occupy one side of PC1.
- `firmness`, `fairness_norm`, and `spite` occupy the opposite side.
- `composure` is separated mainly along PC2.
- `anchoring` sits closer to the center, consistent with it being a more specialized strategic dimension rather than a broad style axis.

**Most useful claim for the paper:** the 10 steering dimensions span a **shared negotiation subspace with about 6-7 effective degrees of freedom**, which helps explain why some dimensions cluster behaviorally while others remain distinct.

#### Test 3: Control Contamination vs Behavioral Effect

**What it measures:** For each negotiation dimension at each tested layer, compute the strongest absolute cosine overlap with any control vector (`verbosity`, `formality`, `hedging`, `sentiment`, `specificity`), then compare that overlap to behavioral effect size (`|Cohen's d|`).

**Practical relevance:** This is the direct confound test. If the strongest behavioral effects came from the highest overlap with hedging or sentiment, then the project could be criticized as “steering surface style, not negotiation behavior.” This test asks whether that critique is empirically true.

**Headline result:** it is **not** true.

- Pearson correlation between contamination and `|Cohen's d|`: **-0.094**
- Spearman correlation: **-0.002**

These are both essentially zero. Higher control overlap does **not** predict larger behavioral effects.

**Strongest effects versus their contamination:**

| Config | |d| | Max contamination | Closest control dim | Interpretation |
| ------ | --- | ----------------- | ------------------- | -------------- |
| fairness_norm L4 | 1.370 | 0.356 | sentiment | Very strong effect, only moderate contamination |
| firmness L10 | 1.266 | 0.469 | hedging | Strong effect, contamination present but not dominant |
| greed L14 | 1.262 | 0.232 | specificity | Strong effect with low contamination |
| greed L12 | 1.123 | 0.301 | specificity | Strong effect with low-moderate contamination |
| anchoring L18 | 0.618 | 0.103 | specificity | Clear effect with almost no control overlap |

**Highest contamination cases:**

| Config | Max contamination | Closest control dim | |d| | Interpretation |
| ------ | ----------------- | ------------------- | --- | -------------- |
| undecidedness L10 | 0.819 | hedging | 0.434 | Strong overlap with hedging, only moderate behavioral effect |
| undecidedness L12 | 0.816 | hedging | 0.210 | Very high contamination, weak effect |
| undecidedness L14 | 0.815 | hedging | 0.466 | Same pattern |
| undecidedness L20 | 0.803 | hedging | 0.098 | Near-null effect despite huge contamination |

**Interpretation:** This is exactly the pattern we hoped to see. The dimension with the strongest control contamination is `undecidedness`, and the contaminating control is `hedging`, which is intuitively sensible. But that high overlap does **not** translate into the strongest bargaining effects. Meanwhile, the strongest payoff- and demand-relevant configs (`fairness_norm`, `firmness`, `greed`, `anchoring`) are not the most contaminated.

So the practical conclusion is:

- some dimensions do overlap with surface controls
- this overlap is **dimension-specific**, not universal
- but the strongest steering effects are **not** explained by those overlaps

This materially strengthens the argument against the “it’s just sentiment/hedging/verbosity” critique.

#### Test 4: Logit Lens on Raw Steering Vectors

**What it measures:** Project each saved vector through the model’s final normalization layer and unembedding matrix to see which tokens it promotes or suppresses.

**Practical relevance:** In principle, this should give the most intuitive explanation of a vector: if `firmness` promotes tokens like `keep`, `$80`, or `demand`, then we can say the vector has a direct vocabulary-level signature.

**Observed result:** the probe is **not semantically reliable** in this setup.

Examples:

- `firmness` L10 top tokens: `Composition`, `bondage`, `rai`, `chez`, `wur`
- `greed` L12 top tokens: `Composition`, `Emily`, `chez`, `asm`, `rig`
- `empathy` L10 top tokens: `ellipse`, `rai`, `elial`, `sofort`, `leh`
- `fairness_norm` L4 top tokens: `bondage`, `Composition`, `rai`, `Hindered`, `studs`

The repetition is the key warning sign:

- `Composition` is the top-1 token in **30** dimension-layer cells
- `ellipse` is top-1 in **12**
- `bondage` is top-1 in **11**
- the most common promoted tokens overall are `Emily`, `Composition`, `bondage`, `rai`, `Missile`, `elial`, `sofort`, `chez`

This is not a believable bargaining vocabulary signature. It is almost certainly an artifact of applying the final norm + unembedding to an **isolated direction vector** rather than an in-context residual stream state.

**Conclusion for write-up:** treat logit lens as a **negative or auxiliary result**, not as primary evidence. It is useful because it shows that naive vocabulary projection of raw steering vectors is unstable here. But it should not be used to support semantic claims like “the firmness vector literally encodes money/demand words.”

### Interpretability Conclusion

The interpretability suite supports three strong claims and one explicit non-claim:

1. **The steering dimensions have structured geometry.** They are not arbitrary labels attached to unrelated vectors.
2. **That geometry is layer-dependent.** Relationships such as empathy↔flattery and fairness_norm↔greed shift substantially with depth.
3. **The negotiation space is low-rank but not collapsed.** About 6-7 effective components explain the 10 dimensions.
4. **The strongest behavioral effects are not explained by control overlap.** This is the most practically important interpretability result because it addresses the surface-confound critique directly.

**Non-claim:** The current raw-vector logit lens does **not** yield reliable vocabulary semantics, so we should not over-interpret it.

**Most useful paper takeaway:** the behavioral dimensions are best understood as a **structured, partially overlapping negotiation subspace** whose geometry changes across layers, rather than as independent word-level concepts.

---

## 10. Teammate Experiments (2026-03-28 to 2026-03-29)

### Prompt Change (PRs #50-51)

Teammates added to BOTH prompts in `apply_steering_ultimatum.py`:

> **Proposer:** "Your aim: earn as much as you can from this split — but remember, if Player B rejects, you both get nothing."
> **Responder:** "Your aim: decide whether this offer is worth accepting — if you reject, neither player receives anything."

Also removed `Example: OFFER=60,40`. Our `ultimatum_game.py` retains the old prompt (no "aim", still has example). **Results between the two code paths are not directly comparable** — the "aim" prompt biases baseline behavior and interacts with steering.

### Rule-Based Responder Mode (PR #46)

Added `--rulebased` flag: deterministic responder accepts if responder share ≥ 35% of pool. Also added responder sweep (rule-based proposer offers at 20/40/60/80% thresholds, LLM responder accepts/rejects each).

### 32B Model Support (PR #49)

Added Qwen 2.5-32B (GPTQ/AWQ quantized). 64 layers → test layers 28/32/36 (proportionally ~L10/L12/L14 on 7B). Multi-GPU SSH dispatch across 22 lab machines.

### Teammate Layer Gradient (7B, LLM-vs-LLM, General Pairs)

Empathy proposer steering across layers (their gridsearch best-alpha, their prompt, mixed temp):

| Layer   | Their Δ demand | Their d   | Our Δ demand (α=-7) | Our d |
| ------- | -------------- | --------- | ------------------- | ----- |
| L10     | -10.2pp        | -1.61     | +12.9pp             | +1.33 |
| L12     | -6.2pp         | -0.52     | +11.8pp             | +1.14 |
| **L14** | **-16.1pp**    | **-3.05** | not tested          | —     |
| L16     | +2.5pp         | +0.38     | not tested          | —     |
| L18     | -6.3pp         | -1.07     | not tested          | —     |
| L19     | -3.4pp         | -0.72     | not tested          | —     |
| L20     | -5.0pp         | -0.74     | not tested          | —     |

**Direction discrepancy:** Their empathy goes DOWN, ours goes UP. Caused by different alpha values (gridsearch vs fixed), different prompts ("aim" vs clean), different baselines (62-66% vs 52%), possibly different temperatures.

**L14 peak (d=-3.05)** is notable — strongest effect of any layer. Not tested in our confirmatory design. Non-monotonic gradient: L14 peaks, L16 reverses, L18-20 moderate.

**Comparability note:** Their results use different experimental design (no true pairing, gridsearch alpha selection, "aim" prompt). Useful as directional evidence but not directly comparable to our confirmatory results.

---

## 11. Consolidated Findings

### Finding 1: Steering Reliably Changes Behavior

All 24 UG configs show significant demand shifts (p < 0.001 after BH-FDR correction). Effect sizes range from d=0.37 to d=1.54. Dose-response is perfectly monotonic (Spearman rho = 1.0 for 5/6 dimension×layer combinations; empathy L12 is rho=0.87). This is robust, reproducible, and large. L14 extends the gradient: firmness still significant (d=0.57) but empathy is null (d=0.14, NS).

### Finding 2: Steering Vectors Interact With Task Context in Dimension-Specific Ways

The UG/DG comparison reveals a clean two-pattern taxonomy (updated after Round 7 resolved the quantization confound):

| Pattern         | Where                         | UG effect    | DG effect       | Meaning                                        |
| --------------- | ----------------------------- | ------------ | --------------- | ---------------------------------------------- |
| **Reversal**    | Firmness L10                  | +16pp        | -5.8pp          | Context-dependent (aggression ↔ fairness)      |
| **Reversal**    | Empathy L10                   | +11pp        | -15pp           | Context-dependent (demands more ↔ gives more)  |
| **Reversal**    | Empathy L12                   | +5pp         | -23pp           | Context-dependent (even stronger reversal)     |
| **Persistence** | Firmness L12 only             | +13pp        | +15pp           | Context-independent ("take more" regardless)   |

**The earlier "nullification" of empathy in DG was a quantization artifact** (see Section 9, Round 7). Under consistent bfloat16 conditions, empathy shows massive DG reversal at all alphas (d=-1.15 to -3.06). The quantized runs suppressed a d=1.29-2.58 effect to near-zero.

**Only firmness L12 persists across contexts.** Every other dimension×layer combination reverses direction between UG and DG. This makes firmness L12 the anomaly — it encodes "take more" as an abstract disposition that is insensitive to whether a responder can reject.

**The empathy reversal exceeds firmness in magnitude.** Empathy L12 swings by 28pp between UG (+5pp) and DG (-23pp). Firmness L10 swings by 22pp. Empathy is the most context-sensitive dimension, not the least.

### Finding 3: The Empathy Vector Encodes Activation, Not Valence

The empathy direction vector does not encode "empathic vs selfish." Evidence:

- Both positive and negative alpha increase demand in UG
- cos(firmness, empathy) = -0.287 at L10, -0.349 at L12 — negative empathy ≈ positive firmness
- The sign modulates tone (acceptance rate), not direction (demand)
- 69-72% of per-game offers are identical between firmness α=7 and empathy α=-7

**Counter to the "empathy = noisy firmness" critique:** The empathy and firmness vectors behave differently in DG. Firmness L12 persists (+15pp); empathy L12 reverses (-23pp). If empathy were just collinear firmness, it would persist at L12 too. The opposite DG behavior at L12 is definitive proof they encode different concepts. Furthermore, empathy reverses at BOTH layers in DG while firmness only reverses at L10 — the dimension×layer×context interaction structure is distinct.

### Finding 4: RLHF Creates Bidirectional Fairness Enforcement

The acceptance curve is non-monotonic. The LLM responder rejects not just unfair offers but also generous ones:

| Offer to Responder | P(accept) | Issue                                                              |
| ------------------ | --------- | ------------------------------------------------------------------ |
| 20%                | 0.70      | Too unfair (expected)                                              |
| 30-40%             | 0.93-0.94 | **Sweet spot**                                                     |
| 50%                | 0.77      | **Fairness dip** ($1 rounding triggers 86% rejection on odd pools) |
| 60%                | 0.91      | OK                                                                 |
| 70-80%             | 0.75-0.76 | **Rejects generous offers** (24/25 cite "fairness")                |

RLHF taught the model to enforce 50/50 splits in both directions. It rejects $67/$66 splits ($1 difference) AND rejects offers where it gets 80% of the pool. The baseline proposer at 50% demand sits in the worst possible spot on this curve.

**Asymmetry insight:** The enforcement is not symmetric. The model polices generosity (rejects 80% to responder at 25% rate) more harshly than exploitation (accepts 30% to responder at 94% rate). RLHF alignment makes the model suspicious of "too good" offers but tolerant of clearly disadvantageous ones.

### Finding 5: Framing Is a Major Negative Channel When Text Is Visible

**Numbers-only mode** (Phase C): framing effect = +0.03pp across 18 configs. TOST confirms equivalence within ±5pp (p_tost=2.8e-11). In this mode, payoff improvement is explained entirely by the demand shift.

**Text-visible mode** (Round 6): framing effect is **massive and negative**. The responder sees the proposer's full reasoning text and rejects steered proposals at dramatically higher rates:

| Config           | Numbers-Only Accept | Text-Visible Accept | Framing Penalty |
| ---------------- | ------------------- | ------------------- | --------------- |
| Firmness L12 α=7 | 88.9%               | 54.5%               | -34.4pp         |
| Empathy L10 α=+7 | 94.9%               | 45.5%               | -49.4pp         |

Demand shifts are near-identical across modes (~13pp firmness, ~12pp empathy). Steering changes what the model asks for regardless of visibility. But when the responder can read the steered reasoning, it detects and punishes the greediness. Payoff reverses from +13-17pp to -12-19pp.

**Key nuance:** Text visibility also helps the baseline (91% accept vs 84%), because the baseline's "fair split" reasoning reassures the responder. The framing channel works in both directions: cooperative text helps, greedy text hurts.

**Implication:** The "steering improves payoff" finding is bounded to numbers-only (parsed offers). In any communication-rich setting, steered text is counterproductive. This is the central external validity caveat for the paper.

### Finding 6: Effects Attenuate With Extended Pool Range (L10 > L12)

The n=200 replication with extended pools ($37-$273 vs $37-$157) shows all effects replicate directionally but L10 effects attenuate by ~50%. L12 effects are pool-invariant:

| Config   | Original d (n=100) | Extended d (n=200) | Attenuation |
| -------- | ------------------ | ------------------ | ----------- |
| L10 α=3  | 0.98               | 0.15               | -85%        |
| L10 α=7  | 1.13               | 0.29               | -74%        |
| L10 α=10 | 1.19               | 0.39               | -67%        |
| L12 α=7  | 0.63               | 0.26               | -59%        |
| L12 α=10 | 0.61               | 0.26               | -57%        |

L10's context-sensitive mechanism is less effective at unfamiliar (higher) stakes. L12's demand deltas barely change (+5.1-5.3pp → +5.4pp), consistent with the "context-independent disposition" interpretation. The extended-pool results are the more defensible estimates for the paper. See Section 9, Round 7 for full details.

### Finding 7 (was 6): Variable Pools Are Essential

| Design                 | Unique offers (n=100) | Baseline std |
| ---------------------- | --------------------- | ------------ |
| Fixed $100, temp=0     | 2                     | ~0%          |
| Variable pools, temp=0 | 34                    | 7.0%         |

Without variable pools, no experiment is possible at temp=0. The 100 diverse pool sizes ($37-$157) are the single most important design decision.

### Finding 8 (was 7): The Baseline Runs a Rigid Heuristic, Not a Continuous Distribution

66/100 baseline games produce byte-for-byte identical offers (50/50 split template). The unsteered model isn't sampling from a continuous intent distribution — it's executing a hard-coded "split equally" heuristic. Steering doesn't nudge a smooth distribution; it breaks a rigid algorithm. This means Cohen's d overstates the effect (comparing point mass vs spread distribution). Report effect sizes in both d AND absolute percentage-point shifts.

### Finding 9 (was 8): L14 Extends the Layer Gradient but Does Not Replicate Teammate's Peak

Firmness attenuates from L10 (d=1.38) → L12 (d=1.21) → L14 (d=0.57). Empathy attenuates from L10 (d=1.16) → L12 (d=0.63) → L14 (d=0.14, NS). Both dimensions fade with layer depth, consistent with the exploratory finding that L14+ is mostly inactive.

Teammate's empathy L14 d=-3.05 is NOT replicated (we get d=+0.14 with opposite sign). The discrepancy likely reflects their different prompt ("Your aim: earn as much as you can"), different alpha value (gridsearch-selected), and/or different baselines. L14 is not a special layer for steering; it's just further along the decay gradient.

### Finding 10: The Steering Dimensions Form a Structured, Low-Rank Negotiation Subspace

The new interpretability analyses (Section 9, 2026-04-08) show that the 10 dimensions are neither orthogonal nor redundant. Across layers, the space requires **6-7 principal components** to explain 90% of variance. This means the dimensions share underlying structure, but do not collapse to a single scalar like "niceness" or "aggression."

The geometry is also behaviorally meaningful:

- `greed` and `narcissism` are stably aligned (+0.54 to +0.60 from L4-L20)
- `firmness` and `undecidedness` are stably opposed (-0.65 to -0.21)
- `empathy` and `flattery` are highly aligned early (+0.77) but separate later (+0.45 by L20)
- `firmness` and `empathy` are consistently anti-correlated, but not enough to treat one as a simple sign-flip of the other

**Interpretation:** steering operates in a shared negotiation manifold whose internal geometry changes with layer depth. This is a stronger and more precise claim than "different layers work for different dimensions."

### Finding 11: Surface-Style Overlap Does Not Explain the Strongest Effects

Control contamination has essentially zero relationship with behavioral effect size:

- Pearson r(contamination, |d|) = **-0.094**
- Spearman r(contamination, |d|) = **-0.002**

This is important because it directly answers the confound critique. The strongest effects (`fairness_norm` L4, `firmness` L10, `greed` L12/L14) do **not** occur at the highest contamination levels. The most contaminated dimension is `undecidedness`, driven by overlap with `hedging`, but its behavioral effects are only moderate.

**Interpretation:** some dimensions contain traceable surface-style overlap, but the main negotiation effects are not reducible to hedging, sentiment, specificity, verbosity, or formality.

### Research Council Insights (2026-03-29)

A 4-model research council (GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Grok 4) deliberated on next steps. Key insights not previously identified:

1. **The DG dissociation is the proof that empathy ≠ firmness.** Despite cos=-0.29 overlap, the three distinct DG patterns (reversal, persistence, nullification) mathematically prove the vectors encode different concepts. This is the defense against the collinearity critique.

2. **The acceptance curve asymmetry is a publishable RLHF pathology.** The model polices generosity more harshly than exploitation — a bidirectional but asymmetric fairness norm.

3. **The "nullification" claim for empathy DG rests on thin evidence** (only 2 configs, 1 alpha). Need 4 more empathy DG configs (L10/L12 × α={3,10}) to distinguish "nullified" from "small attenuated effect." ~2 hrs GPU.

4. **A "text visibility" single-turn experiment** would cleanly test framing effects without multi-turn variance. Show the responder the proposer's full text instead of just parsed numbers. 3 configs, ~1 hr GPU.

5. **Narrow the scope to "ultimatum bargaining" not "negotiation."** There's no back-and-forth, no persuasion, no concessions. The paper is about steering effects on LLM decision-making in strategic interactions.

6. **Reframe the core narrative:** "Activation steering mechanically disrupts rigid, suboptimal RLHF heuristics, shifting behavior into an asymmetrical acceptance landscape. Vector semantics are highly context-dependent."

---

## 12. Paper Narrative (Updated 2026-04-02)

### Core Framing

Activation steering reliably changes LLM behavior in strategic interactions. The effect is dimension-specific (10 dims tested, 7 active), layer-specific (different dims peak at different layers), and context-dependent (most effects reverse between UG and DG). Whether steering helps or hurts payoff depends on the communication setting: in numbers-only mode, moderate steering (anchoring L18) achieves optimal payoff; in text-visible mode, steered reasoning triggers rejection and payoff reverses.

### Story Arc

1. **Steering works across all 10 behavioral dimensions.** Full grid (450 configs, n=50) shows all 10 dims active when full alpha range tested. Top by |d|: greed (L14 d=1.88), composure (L10 d=1.56), firmness (L10 d=1.50), fairness_norm (L4 d=1.37). Screen-to-final validation: 91% direction agreement.

2. **Different dimensions peak at different layers, and the dimensions occupy a structured shared subspace.** Firmness peaks at L10, greed at L12, anchoring at L18, narcissism at L14. No single "best layer." The interpretability suite shows this is not a simple depth gradient: the pairwise geometry between dimensions also changes with layer.

3. **Sign asymmetry reveals vector semantics.** Firmness only works at positive α (directional). Narcissism only at negative α (anti-narcissism → generosity). Empathy/flattery respond to both signs (activation, not valence). Greed is strongly unidirectional.

4. **Context reverses most effects.** In DG (no rejection risk), 3/4 dimension×layer combos reverse direction. Only firmness L12 persists. Empathy reverses at both layers (biggest swing: 28pp). "Nullification" was a quantization artifact.

5. **Text-visible framing is massively negative.** Numbers-only: framing ≈ 0pp. Text-visible: acceptance crashes 34-49pp. Steered text reveals greediness (empathy: "I want to keep as much as possible for myself") and the responder punishes it. Anchoring may be the exception — its text frames greed as generosity.

6. **RLHF creates bidirectional fairness enforcement.** Non-monotonic acceptance curve: rejects both unfair AND generous offers. Baseline sits at 50% demand — the worst spot. Steering disrupts this.

7. **Quantization suppresses steering and shifts thresholds.** 4-bit NF4 killed a d=1.29 effect to d=0.12. Also shifted the activation threshold: α=3 works quantized but not unquantized. α=7 converges across precisions.

8. **Greed L12 α=+7 is the Pareto-optimal config.** Best payoff (62.7%) with +23.4pp demand and 74% acceptance. Anchoring L18 is second (53.9%). Firmness demands too aggressively and gets rejected (57-62% acceptance).

### Key Claims (bounded)

- **We claim:** Steering reliably alters behavioral outputs across 7/10 dimensions tested (p < 0.001 at n=50-100).
- **We claim:** The effect is dimension×layer specific — different concepts are encoded at different network depths, within a low-rank shared negotiation subspace.
- **We claim:** RLHF-aligned models have suboptimal strategic defaults that steering can disrupt.
- **We claim:** The framing channel is real and negative — steered reasoning text triggers opponent rejection.
- **We claim:** Quantization suppresses steering effects and shifts activation thresholds.
- **We claim:** The strongest effects are not explained by overlap with surface control dimensions.
- **We claim:** Cross-design agreement with rulebased experiments (r=0.41, p=0.003) validates effect directions.
- **We do NOT claim:** Steering improves negotiation outcomes in general — only in numbers-only mode with specific dimensions (anchoring, greed).
- **We do NOT claim:** Results generalize to 32B — existing 32B data (GPTQ quantized, wrong layers) is inconclusive.

## 13. Future Directions & Experiment Log

### All 7B experiments complete (2026-03-27 to 2026-04-02)

| # | Experiment | Configs | Status |
|---|-----------|---------|--------|
| 1 | Confirmatory UG (firmness+empathy, L10/L12, α={3,7,10}) | 19 @ n=100 | Done (Round 1) |
| 2 | Fixed DG (firmness, L10/L12) | 6 @ n=100 | Done (Round 2) |
| 3 | Positive empathy UG | 6 @ n=100 | Done (Round 3) |
| 4 | Empathy DG (quantized, confounded) | 2 @ n=100 | Done (Round 4) |
| 5 | L14 adjudication | 2 @ n=100 | Done (Round 5) |
| 6 | Text-visibility control | 3 @ n=100 | Done (Round 6) |
| 7 | Empathy DG clean (unquantized) | 6 @ n=100 | Done (Round 7) |
| 8 | Acceptance curve | 700 calls | Done (Phase B) |
| 9 | Layer gradient screen | 18 @ n=15 | Done |
| 10 | 10-dim landscape screen | 140 @ n=15 | Done |
| 11 | Tier 2 validation | 13 @ n=50 | Done |
| 12 | Tier 3 confirmation | 5 @ n=100 | Done |
| 13 | Alpha bridge (α=5,15) | 2 @ n=50 | Done |
| 14 | Alpha-3 check | 1 @ n=100 | Done |
| 15 | **Final grid batch 1** (α={5,15}) | 180 @ n=50 | **Done** |
| 16 | **Final grid batch 2** (α={-7,7}) | 180 @ n=50 | **Done** |
| 17 | **Final grid batch 3** (α={-5}) | 90 @ n=50 | **Done** |

**Final grid complete: 450 configs** in `results/ultimatum/final_7b_llm_vs_llm/` = 10 dims × 9 layers × 5 alphas {-7,-5,5,7,15}.

### Final Grid Key Results (2026-04-03)

**Tiered screen validation:** 91% direction agreement (127/140 matched configs between n=15 screen and n=50 final).

**All 10 dimensions are active** (≥24% configs significant at p<0.05). None truly dead when full alpha range tested.

**Top configs by |d|:**

| Dim | Layer | α | d | Δpp | Accept | Payoff |
|-----|-------|---|------|-----|--------|--------|
| greed | L14 | +15 | +1.88 | +33.5pp | 28% | 28.0% |
| greed | L12 | +15 | +1.83 | +30.0pp | 51% | 48.6% |
| composure | L10 | +15 | +1.56 | +25.5pp | 60% | 53.3% |
| firmness | L10 | +15 | +1.50 | +23.9pp | 60% | 52.0% |
| fairness_norm | L4 | +7 | +1.37 | +23.9pp | 65% | 55.9% |
| firmness | L10 | +7 | +1.27 | +25.6pp | 62% | 53.8% |
| greed | L14 | +7 | +1.26 | +23.5pp | 62% | 53.0% |
| greed | L12 | +7 | +1.12 | +23.4pp | 74% | **62.7%** |

**Best payoff config: greed L12 α=+7 (62.7%)** — surpasses anchoring L18 as the Pareto optimal.

**Dose-response profiles (at peak layer):**
- Firmness L10: threshold at α=5, peaks α=15, directional (neg α null)
- Greed L12: monotonic α=5→7→15, directional
- Anchoring L18: threshold at α=7, plateaus α=15
- Narcissism L14: inverts (neg α decreases demand, pos α=15 increases it)
- Empathy L10: inverts (neg α increases demand, pos α=5 decreases it)

**New findings from full grid:**
- Fairness_norm L4 α=+7 (d=+1.37) increases demand — opposite of L12 (d=-0.87). Layer-dependent sign reversal within same dimension.
- Composure activates only at α=15 (d=1.56 at L10, d=1.49 at L18). Higher threshold than other dims.
- Narcissism L14 inverts cleanly: negative α = generosity, positive α = greed. The vector encodes self/other orientation.

### Must-Do Before Paper Submission

1. **Write the paper.** All experiments complete. 14 days to deadline (2026-04-17). Data: 450 final grid configs + Phase 2 deep dives (DG, text-visible, acceptance curve) + teammate cross-design comparison.

### Nice-to-Have

2. **32B scaling (unquantized).** Requires A100 80GB (GCP or AWS). Full grid costs ~$150. Proportional layers: 7B L10 → 32B L23. Existing 32B data (GPTQ quantized, L28/L32/L36) is inconclusive due to quantization + wrong layers.

3. **Text-visible extension for new dimensions.** Anchoring L18 may survive text-visibility (text frames greed as generosity). Quick test: 3-5 configs.

4. **DG extension for new dimensions.** Does greed reverse in DG like firmness/empathy? Quick test.

### GPU Infrastructure

**bf16 consistency rule:** Only use Ampere+ GPUs (CC ≥ 8.0). See `/Users/moiz/Documents/code/misconception-finetune/AWS_GPU_GUIDE.md`.

| Platform | GPU | CC | Cost | Status |
|----------|-----|-----|------|--------|
| UCL Lab 1.05 | RTX 3090 Ti | 8.6 | Free | Primary. 3-8 birds available |
| AWS g5.xlarge | A10G | 8.6 | $0.59/hr spot | 2 on-demand available, spot quota pending |
| GCP g2-standard | L4 | 8.9 | $0.57/hr spot | 1 GPU only (quota denied for more) |
| **AVOID** | T4, Quadro RTX 6000, V100 | ≤7.5 | — | Changes baselines, kills effects |

---

## 14. File Reference

### Result Directories

### Results Structure

```
results/ultimatum/
├── final_7b_llm_vs_llm/        ← OUR MAIN DATA (use this for the paper)
│   └── 450 JSON files: {dim}_proposer_L{layer}_a{alpha}_paired_n50.json
│       10 dims × 9 layers (L4-L20) × 5 alphas (-7,-5,5,7,15) × n=50
│       All on RTX 3090 Ti, bfloat16, paired, variable pools, temp=0
│
├── deep_dive_experiments/       ← TARGETED EXPERIMENTS (specific findings)
│   ├── dg/                      Firmness Dictator Game reversal (n=100)
│   ├── dg_empathy/              Empathy DG — quantized, showed "nullification" (confounded)
│   ├── dg_empathy_clean/        Empathy DG — unquantized, resolved confound (empathy reverses)
│   ├── text_visible/            Text-visibility: responder sees proposer's reasoning text
│   └── ug_pos_empathy/          Positive empathy alpha (both signs increase demand)
│
├── acceptance_curve/            ← RLHF FAIRNESS CURVE (no steering)
│   └── 7 offer levels × 100 pools = 700 games. Non-monotonic acceptance.
│
├── llm_vs_rulebased/            ← TEAMMATE DATA (cross-design comparison)
│   ├── qwen2.5-7b_*/           7B rulebased: 10 dims, L10-L20, α={-5,5,15}
│   └── qwen2.5-32b-gptq_*/    32B rulebased: 10 dims, L28/L32/L36
│
├── old_results/                 ← ARCHIVED (intermediate data, audit trail)
│   ├── phase2_deep_dive/        Original Rounds 2-6 (before rename)
│   ├── confirmatory/            Round 1 (quantized, AWS)
│   ├── confirmatory_v3/         Round 7 empathy DG clean
│   ├── landscape_screen*/       n=15 screening (superseded by final grid)
│   ├── tier2_validation/        n=50 validation (superseded)
│   ├── tier3_confirmation/      n=100 top 5 (superseded)
│   ├── layer_gradient*/         Layer sweep (superseded)
│   └── alpha*_check/bridge/     Alpha experiments (superseded)
│
├── phase_c_analytical_payoff.json   Framing decomposition (numbers-only ≈ 0)
└── statistical_hardening.json       Bootstrap CIs, TOST, BH-FDR
```

**For the paper, use these 4 sources:**
1. `final_7b_llm_vs_llm/` — all dimension×layer×alpha results
2. `deep_dive_experiments/` — DG context reversal, text-visibility framing, quantization effect
3. `acceptance_curve/` — RLHF fairness enforcement mechanism
4. `llm_vs_rulebased/` — cross-design validation (r=0.41)

### Key Scripts

| Script                                  | Purpose                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| `ultimatum_game.py`                     | GPU steering with paired design, UG/DG modes               |
| `final_grid_scaup.py` / `scoter` / `shoveler` | Batch runners for final grid (load model once)    |
| `final_grid_alpha_neg5.py`              | α=-5 batch runner                                          |
| `analysis/plot_results.py`              | Generate all publication figures                           |
| `analysis/analyse_final_grid.py`        | Final grid analysis (completeness, heat maps, etc.)        |
| `analysis/unified_results_analysis.ipynb` | 3-paradigm Jupyter notebook (LLM-vs-LLM, rulebased, clean) |
| `analysis/analyse_ug_hypotheses.py`     | Original H1-H5 confirmatory analysis                       |
| `extract_vectors.py`                    | Vector extraction (standalone)                             |
| `validation/validate_vectors.py`        | Full validation suite                                      |
| `validation/orthogonal_projection.py`   | Control dimension projection                               |
| `run_acceptance_curve.py`               | Phase B acceptance curve sweep                             |
| `analysis/phase_c_analytical_payoff.py` | Phase C framing decomposition                              |
| `analysis/statistical_hardening.py`     | Bootstrap CIs, TOST, BH-FDR, dose-response monotonicity    |
| `analysis/interpretability.py`          | CPU interpretability suite + optional logit lens           |
| `run_sprint_empathy_dg.sh`              | Empathy DG thin cells (4 configs)                          |
| `run_sprint_l14.sh`                     | L14 adjudication (2 configs)                               |
| `run_sprint_text_vis.sh`                | Text-visibility control (3 configs)                        |
| `run_top5_200games.sh`                  | Top 5 empathy configs at n=200 (extended pools)            |
| `apply_steering_ultimatum.py`           | Teammate steering code (different prompts, rulebased mode) |
| `lightweight_gridsearch_ultimatum.py`   | Teammate gridsearch dispatcher                             |

### Steering Pairs

| Directory                                                 | Dims | Pairs | Used For                    |
| --------------------------------------------------------- | ---- | ----- | --------------------------- |
| `steering_pairs/ultimatum_10dim_20pairs_general_matched/` | 10   | 20    | All Phase 2 experiments     |
| `steering_pairs/ultimatum_10dim_20pairs_matched/`         | 10   | 20    | Exploratory (game-specific) |
| `steering_pairs/neg8dim_12pairs_matched/`                 | 8    | 12    | Phase 1 + early Phase 2     |

---

_Last updated: 2026-04-08 (Interpretability suite added: cosine geometry, PCA steering space, control contamination, logit-lens negative result). Update this document after every significant experiment or finding._
