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

| Config           | UG Δ demand | DG Δ demand             | Pattern    |
| ---------------- | ----------- | ----------------------- | ---------- |
| Firmness L10 α=7 | +16.1pp     | **-5.8pp** (d=-0.65)    | Reversal   |
| Firmness L12 α=7 | +13.2pp     | **+15.3pp** (d=0.96)    | Persistent |
| Empathy L10 α=+7 | +11.4pp     | **-0.5pp** (p=0.23, NS) | Nullified  |
| Empathy L12 α=+7 | +5.2pp      | **-1.1pp** (d=-0.24)    | Nullified  |

**The L10/L12 distinction is NOT a general layer property — it's a dimension×layer interaction.** Firmness has a clean dissociation (L10 reverses, L12 persists). Empathy collapses to zero at BOTH layers in DG. The empathy vector's entire effect is mediated by the presence of a responder who can reject. Remove the threat, and empathy steering does nothing — only 15/100 games had different offers.

Three distinct patterns emerge:

| Pattern           | Where           | Meaning                                              |
| ----------------- | --------------- | ---------------------------------------------------- |
| **Reversal**      | Firmness L10    | Context-dependent style (aggression ↔ fairness)      |
| **Persistence**   | Firmness L12    | Context-independent disposition ("take more")        |
| **Nullification** | Empathy L10+L12 | Only activates in adversarial context (needs threat) |

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

All 24 UG configs show significant demand shifts (p < 0.001 after BH-FDR correction). Effect sizes range from d=0.37 to d=1.54. Dose-response is perfectly monotonic (Spearman rho = 1.0 for all 4 dimension×layer combinations). This is robust, reproducible, and large.

### Finding 2: Steering Vectors Interact With Task Context in Dimension-Specific Ways

Three distinct patterns emerge from the UG/DG comparison:

| Pattern           | Where           | UG effect   | DG effect | Meaning                                         |
| ----------------- | --------------- | ----------- | --------- | ----------------------------------------------- |
| **Reversal**      | Firmness L10    | +16pp       | -5.8pp    | Context-dependent style (aggression ↔ fairness) |
| **Persistence**   | Firmness L12    | +13pp       | +15pp     | Context-independent disposition ("take more")   |
| **Nullification** | Empathy L10+L12 | +5 to +11pp | ~0pp (NS) | Only activates with adversarial threat          |

This is NOT simply "L10 = context-dependent, L12 = context-independent." It's a dimension×layer×context interaction. Firmness has a clean layer dissociation; empathy does not.

**Firmness L10** encodes something the model interprets as "aggression" when there's rejection risk but "principled fairness" without it. **Firmness L12** encodes "take more" regardless. **Empathy** (both layers) only produces effects when a responsive agent is present — it encodes "engagement" that requires someone to engage with.

### Finding 3: The Empathy Vector Encodes Activation, Not Valence

The empathy direction vector does not encode "empathic vs selfish." Evidence:

- Both positive and negative alpha increase demand
- cos(firmness, empathy) = -0.287 at L10, -0.349 at L12 — negative empathy ≈ positive firmness
- The sign modulates tone (acceptance rate), not direction (demand)
- 69-72% of per-game offers are identical between firmness α=7 and empathy α=-7
- Effect nullifies completely in DG — needs a responsive counterpart

**Counter to the "empathy = noisy firmness" critique:** If empathy were just a collinear copy of firmness, it would behave identically in DG (reverse at L10, persist at L12). Instead it nullifies at both layers. The dimension×layer×context interaction is the mathematical proof that they are semantically distinct vectors, despite cos=-0.29 overlap.

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

### Finding 5: Steering Works Through Numbers, Not Framing (In Single-Turn)

Phase C decomposition: framing effect = +0.03pp (±1.4pp std) across 18 configs. Payoff improvement is explained entirely by the demand shift.

**This is a design feature, not a finding:** the responder only sees parsed OFFER=X,Y numbers, never the proposer's reasoning text. The numbers-only design provides clean causal attribution — we know steering changes _what_ the model asks for (behavior), not _how persuasively_ it asks (rhetoric). But it cannot test whether steering also affects communication. A "full-text" single-turn condition (where the responder sees the proposer's complete text) would isolate the framing contribution.

### Finding 6: Variable Pools Are Essential

| Design                 | Unique offers (n=100) | Baseline std |
| ---------------------- | --------------------- | ------------ |
| Fixed $100, temp=0     | 2                     | ~0%          |
| Variable pools, temp=0 | 34                    | 7.0%         |

Without variable pools, no experiment is possible at temp=0. The 100 diverse pool sizes ($37-$157) are the single most important design decision.

### Finding 7: The Baseline Runs a Rigid Heuristic, Not a Continuous Distribution

66/100 baseline games produce byte-for-byte identical offers (50/50 split template). The unsteered model isn't sampling from a continuous intent distribution — it's executing a hard-coded "split equally" heuristic. Steering doesn't nudge a smooth distribution; it breaks a rigid algorithm. This means Cohen's d overstates the effect (comparing point mass vs spread distribution). Report effect sizes in both d AND absolute percentage-point shifts.

### Research Council Insights (2026-03-29)

A 4-model research council (GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6, Grok 4) deliberated on next steps. Key insights not previously identified:

1. **The DG dissociation is the proof that empathy ≠ firmness.** Despite cos=-0.29 overlap, the three distinct DG patterns (reversal, persistence, nullification) mathematically prove the vectors encode different concepts. This is the defense against the collinearity critique.

2. **The acceptance curve asymmetry is a publishable RLHF pathology.** The model polices generosity more harshly than exploitation — a bidirectional but asymmetric fairness norm.

3. **The "nullification" claim for empathy DG rests on thin evidence** (only 2 configs, 1 alpha). Need 4 more empathy DG configs (L10/L12 × α={3,10}) to distinguish "nullified" from "small attenuated effect." ~2 hrs GPU.

4. **A "text visibility" single-turn experiment** would cleanly test framing effects without multi-turn variance. Show the responder the proposer's full text instead of just parsed numbers. 3 configs, ~1 hr GPU.

5. **Narrow the scope to "ultimatum bargaining" not "negotiation."** There's no back-and-forth, no persuasion, no concessions. The paper is about steering effects on LLM decision-making in strategic interactions.

6. **Reframe the core narrative:** "Activation steering mechanically disrupts rigid, suboptimal RLHF heuristics, shifting behavior into an asymmetrical acceptance landscape. Vector semantics are highly context-dependent."

---

## 12. Paper Narrative (Updated 2026-03-29)

### Core Framing

Activation steering mechanically disrupts rigid, suboptimal RLHF heuristics. The baseline model executes a hard-coded "split equally" algorithm that lands in the worst spot on its own acceptance curve. Steering breaks this heuristic, shifting behavior into the region where the responder is most rational. Vector semantics are highly context-dependent: the same vector reverses, persists, or nullifies depending on dimension, layer, and game structure.

### Story Arc

1. **Steering changes what the model asks for, not how it reasons.** All 24 configs significant (d=0.37-1.54), perfect dose-response. The numbers-only design proves the mechanism is behavioral (demand shift), not rhetorical (persuasion).

2. **The same vector produces three distinct patterns across contexts.** Firmness L10 reverses in DG (aggression → fairness). Firmness L12 persists (context-independent disposition). Empathy nullifies entirely (requires adversarial context). This dimension×layer×context interaction is the central mechanistic finding.

3. **RLHF alignment creates a bidirectional, asymmetric fairness norm.** The model rejects both unfair AND generous offers, but polices generosity more harshly than exploitation. The baseline sits at 50% demand — the worst possible spot. Any steering disrupts this suboptimal heuristic.

4. **Empathy encodes activation, not valence.** Both signs increase demand. The DG nullification proves it's semantically distinct from firmness (despite cos=-0.29 overlap). The sign modulates acceptance rate through tone, making positive empathy the strategically optimal config (+17pp).

### Key Claims (bounded)

- **We claim:** Steering reliably alters behavioral outputs in strategic interactions.
- **We claim:** The effect is dimension×layer×context dependent, not a simple layer hierarchy.
- **We claim:** RLHF-aligned models have suboptimal strategic defaults that steering can disrupt.
- **We do NOT claim:** Steering improves "negotiation" (no back-and-forth tested).
- **We do NOT claim:** Steering alters "strategic reasoning" (we changed behavior, not reasoning).
- **We do NOT claim:** The framing/persuasion channel matters (untested — responder only sees numbers).

## 13. Future Directions

### Must-Do Before Paper Submission (~4 hrs GPU total)

1. **Shore up empathy DG "nullification" claim.** Current evidence: only 2 configs (L10 α=+7, L12 α=+7). Run 4 more: L10 α={3,10} + L12 α={3,10} in DG mode. Distinguishes "nullified" from "small attenuated effect." Add TOST equivalence test to formally bound the claim. ~2 hrs GPU.

2. **L14 adjudication.** Teammate data shows d=-3.05 at L14 for empathy (strongest effect of any layer), but with different design. Run firmness + empathy at L14 α=7 in our clean setup. Confirms or denies. ~40 min GPU.

3. **Text-visibility control.** Single-turn UG where responder sees proposer's FULL TEXT (not just parsed numbers). 3 configs: baseline, firmness L12 α=7, empathy L10 α=+7. Tests whether framing matters when the responder can actually see it. ~1 hr GPU.

4. **Statistical hardening (no GPU).** Add confidence intervals to all effect sizes. Add TOST for all "null" claims. Report pp shifts alongside Cohen's d. Count unique offer pairs per config (addresses tied-observation inflation).

### Nice-to-Have (if time permits)

5. **32B scaling.** Teammates running Qwen 32B. If layer-gradient holds proportionally, it's a generalization claim.

6. **Rule-based validation.** Report teammates' rule-based results as independent validation in supplementary.

7. **Prompt comparison.** Small run with "aim" prompt to quantify the effect. Relegate to appendix.

### Not Worth Pursuing

- Multi-model replication (no time, single-model is defensible for a course paper)
- More alpha values (3-point dose-response is sufficient)
- More dimensions (firmness + empathy cover the three key patterns)
- Game-specific pairs (fundamentally flawed design — hold offers constant)

---

## 14. File Reference

### Result Directories

| Path                                                | Contents                                | Round       |
| --------------------------------------------------- | --------------------------------------- | ----------- |
| `results/ultimatum/confirmatory/ug/`                | 12 UG configs (firmness +α, empathy -α) | 1           |
| `results/ultimatum/confirmatory/dg/`                | 6 DG configs (bug: identical prompts)   | 1           |
| `results/ultimatum/confirmatory/robustness/`        | 1 temp=0.3 check                        | 1           |
| `results/ultimatum/confirmatory_v2/dg/`             | 6 DG configs (fixed prompt)             | 2           |
| `results/ultimatum/confirmatory_v2/ug_pos_empathy/` | 6 positive empathy configs              | 3           |
| `results/ultimatum/confirmatory_v2/dg_empathy/`     | 2 empathy DG configs                    | 4           |
| `results/ultimatum/acceptance_curve/`               | Acceptance curve (7 levels × 100 pools) | B           |
| `results/ultimatum/phase_c_analytical_payoff.json`  | Framing effect decomposition            | C           |
| `results/ultimatum/napkin_test/`                    | 3 sanity check configs                  | Pre         |
| `results/ultimatum/llm_vs_llm/`                     | Teammate LLM-vs-LLM exploratory results | Exploratory |
| `results/ultimatum/llm_vs_rulebased/`               | Teammate rule-based results             | Exploratory |

### Key Scripts

| Script                                  | Purpose                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| `ultimatum_game.py`                     | GPU steering with paired design, UG/DG modes               |
| `run_confirmatory.sh`                   | Round 1 runner (19 configs)                                |
| `analysis/analyse_confirmatory.py`      | H1-H5 with BH-FDR, TOST, dose-response                     |
| `analysis/compile_exploratory.py`       | Re-analyzes all exploratory results                        |
| `analyse_napkin.py`                     | Napkin test decision gate                                  |
| `extract_vectors.py`                    | Vector extraction (standalone)                             |
| `validation/validate_vectors.py`        | Full validation suite                                      |
| `validation/orthogonal_projection.py`   | Control dimension projection                               |
| `run_acceptance_curve.py`               | Phase B acceptance curve sweep                             |
| `analysis/phase_c_analytical_payoff.py` | Phase C framing decomposition                              |
| `apply_steering_ultimatum.py`           | Teammate steering code (different prompts, rulebased mode) |
| `lightweight_gridsearch_ultimatum.py`   | Teammate gridsearch dispatcher                             |

### Steering Pairs

| Directory                                                 | Dims | Pairs | Used For                    |
| --------------------------------------------------------- | ---- | ----- | --------------------------- |
| `steering_pairs/ultimatum_10dim_20pairs_general_matched/` | 10   | 20    | All Phase 2 experiments     |
| `steering_pairs/ultimatum_10dim_20pairs_matched/`         | 10   | 20    | Exploratory (game-specific) |
| `steering_pairs/neg8dim_12pairs_matched/`                 | 8    | 12    | Phase 1 + early Phase 2     |

---

_Last updated: 2026-03-29. Update this document after every significant experiment or finding._
