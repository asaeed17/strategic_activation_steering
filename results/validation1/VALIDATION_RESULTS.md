# Steering Vector Validation Results

**Model:** Qwen2.5-3B-Instruct | **Date:** 2026-03-06 | **Mode:** `--full`

8 steering pair variants validated across 5 control dimensions (verbosity, formality, hedging, sentiment, specificity).

---

## 1. Summary Table

| Variant | Score | RED | AMBER | GREEN | Neg flat-high | Ctrl flat-high | Best selectivity |
|---|---|---|---|---|---|---|---|
| **neg8dim_12pairs_matched** | **33/100** | 3 | 5 | 0 | **1/8** | 3/5 | firmness 0.639 |
| neg15dim_20pairs_matched | 28/100 | 10 | 5 | 0 | 5/15 | 4/5 | interest_based 0.782 |
| neg8dim_20pairs_matched | 28/100 | 4 | 4 | 0 | 2/8 | 4/5 | active_listening 0.690 |
| neg15dim_12pairs_matched | 26/100 | 8 | 7 | 0 | 2/15 | 3/5 | interest_based 0.768 |
| neg8dim_80pairs_matched | 26/100 | 6 | 2 | 0 | 6/8 | 5/5 | active_listening 0.782 |
| neg15dim_80pairs_matched | 24/100 | 13 | 2 | 0 | 13/15 | 5/5 | info_gathering 0.826 |
| neg8dim_12pairs_raw | 20/100 | 8 | 0 | 0 | 4/8 | 5/5 | anchoring 0.577 |
| neg15dim_12pairs_raw | 16/100 | 13 | 2 | 0 | 9/15 | 5/5 | anchoring 0.577 |

**Zero negotiation dimensions reach GREEN in any variant.**

---

## 2. Effect of Each Intervention

### 2a. Length Matching (raw -> matched)

The single most effective intervention. Cuts negotiation flat-high probes by 75-80%.

| Comparison | Score | Neg flat-high | Length d (median) |
|---|---|---|---|
| 15dim_12 raw | 16 | 9/15 | SEVERE (d > 1.5) |
| 15dim_12 matched | 26 (+10) | 2/15 | MILD (d < 0.5) |
| 8dim_12 raw | 20 | 4/8 | SEVERE |
| 8dim_12 matched | 33 (+13) | 1/8 | MILD |

### 2b. Pair Scaling (12 -> 20 -> 80, all matched)

**More pairs worsens validation.** Contradicts naive scaling hypothesis.

| Pairs | 15dim score | 15dim neg flat-high | 8dim score | 8dim neg flat-high |
|---|---|---|---|---|
| 12 | 26 | 2/15 | **33** | **1/8** |
| 20 | 28 | 5/15 | 28 | 2/8 |
| 80 | **24** | **13/15** | 26 | **6/8** |

Per-pair alignment degrades at scale:

| Dimension | 12 pairs | 20 pairs | 80 pairs |
|---|---|---|---|
| empathy | 0.464 | 0.411 | **0.343** (30/80 outliers) |
| composure | 0.476 | 0.445 | **0.399** (26/80 outliers) |
| firmness | 0.535 | 0.489 | **0.427** (16/80 outliers) |

### 2c. Dimension Merging (15 -> 8)

Modest improvement at low pair counts from merging overlapping concepts.

| Pairs | 15dim score | 8dim score | Delta |
|---|---|---|---|
| 12 matched | 26 | **33** | +7 |
| 20 matched | 28 | 28 | 0 |
| 80 matched | 24 | 26 | +2 |

---

## 3. Cross-Dimension Cosine Similarity (Bug Fix Results)

After fixing hardcoded `{"verbosity", "formality"}` to use all 5 control dimensions, previously hidden confounds are now visible.

### Highest cosine overlaps with control dimensions

| Negotiation dim | Control dim | Cosine | Variant / Layer |
|---|---|---|---|
| firmness | hedging | **-0.703** | 8dim_80pairs / L27 |
| firmness | sentiment | **-0.692** | 8dim_80pairs / L9 |
| creative_problem_solving | verbosity | 0.651 | 8dim_12pairs_raw / L9 |
| empathy | verbosity | 0.643 | 8dim_12pairs_raw / L18 |
| strategic_concession | sentiment | **-0.640** | 8dim_80pairs / L9 |
| empathy | sentiment | 0.614 | 8dim_12pairs_raw / L18 |
| firmness | formality | 0.656 | 8dim_20pairs / L9 |

**Interpretation:** "Be firm" is 70% anti-aligned with "hedge" in embedding space. This is a surface confound (firm speakers hedge less by definition), not a deep negotiation insight.

---

## 4. Cohen's d Bias (Negotiation vs Control)

**Every negotiation x control combination is SEVERE across all variants.** Even the best variant (neg8dim_12pairs_matched) has 40/40 SEVERE pairs.

Example from neg8dim_12pairs_matched (the best variant):

| Negotiation dim | Worst control | max |d| | Severe layers |
|---|---|---|---|
| active_listening | verbosity | 2.90 | 26/36 |
| anchoring | formality | 4.90 | 35/36 |
| batna_awareness | verbosity | 10.00 | 31/36 |
| composure | verbosity | 4.10 | 24/36 |
| creative_problem_solving | verbosity | 10.47 | 36/36 |
| empathy | hedging | 5.67 | 32/36 |
| firmness | formality | 6.11 | 29/36 |
| strategic_concession | hedging | 4.36 | 36/36 |

---

## 5. Layer Recommendations

Criteria: `accuracy >= 0.85 AND max |Cohen's d| <= 0.8`

Only **one** dimension in **one** variant has any recommended layers:

| Variant | Dimension | Recommended layers |
|---|---|---|
| neg8dim_80pairs_matched | active_listening | [3, 4, 15, 16] |

All other dimension x variant combinations: **RECOMMEND: none found, AVOID: all 36 layers.**

---

## 6. Selectivity Scores

Best selectivity per dimension across all variants:

| Dimension | Best variant | Selectivity | Best layer |
|---|---|---|---|
| information_gathering | 15dim_80pairs | 0.826 | L4 |
| interest_based_reasoning | 15dim_80pairs | 0.825 | L12 |
| active_listening | 8dim_80pairs | 0.782 | L4 |
| emotional_regulation | 15dim_80pairs | 0.682 | L26 |
| empathy | 15dim_20pairs | 0.679 | L6 |
| firmness | 8dim_12pairs | 0.639 | L28 |
| composure | 8dim_12pairs | 0.606 | L28 |
| anchoring | all variants | ~0.50-0.58 | varies |

Note: High selectivity at 80 pairs is misleading. The selectivity formula (`probe_acc - 0.5 * min(max|d|/2, 1)`) caps the penalty at 0.5, so near-perfect probe accuracy (driven by surface features at 80 pairs) inflates selectivity.

---

## 7. Probe Accuracy Patterns

### Negotiation dimension flat-high counts by variant

| Variant | Flat-high dims | Non-flat dims | % flat |
|---|---|---|---|
| neg8dim_12pairs_matched | 1 (anchoring) | 7 | 12.5% |
| neg15dim_12pairs_matched | 2 (anchoring, interest_based) | 13 | 13.3% |
| neg8dim_20pairs_matched | 2 (active_list, anchoring) | 6 | 25% |
| neg15dim_20pairs_matched | 5 | 10 | 33.3% |
| neg8dim_12pairs_raw | 4 | 4 | 50% |
| neg8dim_80pairs_matched | 6 | 2 | 75% |
| neg15dim_12pairs_raw | 9 | 6 | 60% |
| neg15dim_80pairs_matched | 13 | 2 | **86.7%** |

### Control dimension flat-high is expected

Control dimensions are designed to capture surface features. Flat-high probes for controls validate that the control dimensions are working correctly.

| Control dim | Flat-high in matched variants | Notes |
|---|---|---|
| verbosity | 8/8 (always) | By design: length is the signal |
| hedging | 7/8 | Strong surface marker |
| sentiment | 6/8 | Word-level vocab separation |
| formality | 5/8 | Drops out with matching |
| specificity | 4/8 | Most semantically nuanced control |

---

## 8. Vocabulary Overlap

Dimensions with CRITICAL overlap (Jaccard < 0.15 global AND < 0.08 pair):

| Dimension | Global Jaccard | Pair Jaccard | Present in |
|---|---|---|---|
| verbosity | 0.066-0.101 | 0.040-0.046 | All variants |
| sentiment | 0.096-0.120 | 0.066-0.091 | Raw + some matched |

Probe accuracy for these dimensions is meaningless — a bag-of-words classifier would achieve the same separation.

---

## 9. Per-Dimension Traffic Light (Best Variant: neg8dim_12pairs_matched)

| Dimension | Status | Flags |
|---|---|---|
| active_listening | AMBER | cohens_d |
| composure | AMBER | cohens_d |
| empathy | AMBER | cohens_d |
| firmness | AMBER | cohens_d |
| strategic_concession_making | AMBER | cohens_d |
| anchoring | RED | flat_probe, cohens_d |
| batna_awareness | RED | length, cohens_d |
| creative_problem_solving | RED | length, cohens_d |

---

## 10. Key Conclusions

1. **Length matching is the only intervention that reliably improves quality** (+10-13 points, 75-80% flat-high reduction). It should be considered mandatory.

2. **Pair scaling hurts**: More pairs amplifies surface confounds rather than fixing them. At 80 pairs, 87% of 15-dim dimensions show flat-high probes. Per-pair alignment degrades because generated pairs become internally inconsistent. This directly contradicts the naive extrapolation from Chalnev et al. (2025), who found plateaus at ~80 samples for clean binary behavioral signals.

3. **Dimension merging helps modestly**: 15 -> 8 dimensions improves the best score from 26 to 33 by reducing concept overlap.

4. **No variant produces defensibly usable vectors**: Every negotiation dimension has SEVERE Cohen's d bias against every control dimension at nearly every layer. The `firmness x hedging` cosine of -0.703 is a clear smoking gun — the vectors encode surface patterns (firm speakers hedge less) rather than deep negotiation strategy.

5. **The bug fix was material**: Three hardcoded references to `{"verbosity", "formality"}` were hiding confounds with hedging, sentiment, and specificity. After correction, the number of flagged cosine overlaps roughly doubled.

6. **Recommended variant for downstream experiments**: `neg8dim_12pairs_matched` (score 33, 1/8 flat-high, 5/8 AMBER). But use with full awareness that vectors are confounded — any observed effects may be surface-feature driven.
