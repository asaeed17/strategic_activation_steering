# PCA Steering Vector Validation Results

**Model:** Qwen2.5-3B-Instruct | **Method:** PCA (PC1 of difference vectors) | **Date:** 2026-03-18 | **Mode:** `--full`

8 steering pair variants validated across 5 control dimensions (verbosity, formality, hedging, sentiment, specificity). This document parallels `VALIDATION_RESULTS.md` (mean difference method) for direct comparison.

---

## 1. Summary Table

| Variant | Score | RED | AMBER | GREEN | Neg flat-high | Ctrl flat-high | Best selectivity |
|---|---|---|---|---|---|---|---|
| **neg8dim_12pairs_matched** | **33/100** | 3 | 5 | 0 | **1/8** | 3/5 | anchoring 0.786 |
| neg8dim_20pairs_matched | 28/100 | 4 | 4 | 0 | 2/8 | 4/5 | firmness 0.760 |
| neg15dim_20pairs_matched | 28/100 | 4 | 4 | 0 | 2/15 | 4/5 | firmness 0.760 |
| neg8dim_80pairs_matched | 26/100 | 6 | 2 | 0 | 6/8 | 5/5 | anchoring 0.880 |
| neg15dim_80pairs_matched | 26/100 | 6 | 2 | 0 | 6/15 | 5/5 | anchoring 0.880 |
| neg15dim_12pairs_matched | 26/100 | 8 | 7 | 0 | 2/15 | 3/5 | anchoring 0.800 |
| neg8dim_12pairs_raw | 20/100 | 8 | 0 | 0 | 4/8 | 5/5 | anchoring 0.800 |
| neg15dim_12pairs_raw | 20/100 | 8 | 0 | 0 | 4/15 | 5/5 | anchoring 0.800 |

**Zero negotiation dimensions reach GREEN in any variant** — identical to mean difference.

---

## 2. PCA vs Mean Difference: Head-to-Head

The critical comparison: PCA extracts the dominant variance direction (PC1) while mean difference computes the centroid displacement. With confounded data, these can diverge.

### 2a. Validity Scores

| Variant | MD Score | PCA Score | Delta |
|---|---|---|---|
| neg8dim_12pairs_matched | 33 | 33 | 0 |
| neg8dim_20pairs_matched | 28 | 28 | 0 |
| neg15dim_20pairs_matched | 28 | 28 | 0 |
| neg15dim_12pairs_matched | 26 | 26 | 0 |
| neg8dim_80pairs_matched | 26 | 26 | 0 |
| neg15dim_80pairs_matched | 24 | 26 | +2 |
| neg8dim_12pairs_raw | 20 | 20 | 0 |
| neg15dim_12pairs_raw | 16 | 20 | +4 |

**Validity scores are nearly identical.** This is expected: scores are driven by data quality (length confounds, vocab overlap, sample size), not extraction method. Both methods receive the same penalty deductions.

### 2b. 1-D Steering-Direction Probe Accuracy (neg8dim_12pairs_matched)

This is where the methods diverge dramatically. The 1-D probe tests whether projecting hidden states onto the extracted direction separates pos/neg samples.

| Dimension | MD 1D acc | PCA 1D acc | Delta |
|---|---|---|---|
| active_listening | 0.889 | 0.716 | **-0.173** |
| anchoring | 0.969 | 0.575 | **-0.394** |
| batna_awareness | 0.861 | 0.572 | **-0.289** |
| composure | 0.868 | 0.595 | **-0.273** |
| creative_problem_solving | 0.923 | 0.517 | **-0.406** |
| empathy | 0.867 | 0.541 | **-0.326** |
| firmness | 0.921 | 0.617 | **-0.304** |
| strategic_concession_making | 0.924 | 0.551 | **-0.373** |
| **Mean** | **0.903** | **0.586** | **-0.317** |

**Mean difference vectors achieve 54% higher 1-D probe accuracy than PCA.** PCA's mean accuracy (0.586) is only marginally above chance (0.5), indicating PC1 frequently captures noise variance rather than the concept-discriminative direction.

### 2c. Full-Dimension Probe Accuracy Comparison (neg8dim_12pairs_matched)

The full-dimension probe uses all hidden dimensions (not just the extracted direction). This tests data quality, not extraction quality.

| Dimension | MD full-dim | PCA full-dim | Delta |
|---|---|---|---|
| active_listening | 0.870 | 0.870 | 0 |
| anchoring | 0.942 | 0.942 | 0 |
| batna_awareness | 0.854 | 0.854 | 0 |
| composure | 0.842 | 0.842 | 0 |
| creative_problem_solving | 0.904 | 0.904 | 0 |
| empathy | 0.803 | 0.803 | 0 |
| firmness | 0.919 | 0.919 | 0 |
| strategic_concession_making | 0.881 | 0.881 | 0 |

Full-dimension probes are identical (as expected — same hidden states, same labels). The divergence is entirely in which 1-D direction each method selects.

---

## 3. Effect of Each Intervention (PCA)

### 3a. Length Matching (raw -> matched)

| Comparison | Score | Neg flat-high |
|---|---|---|
| 15dim_12 raw | 20 | 4/15 |
| 15dim_12 matched | 26 (+6) | 2/15 |
| 8dim_12 raw | 20 | 4/8 |
| 8dim_12 matched | 33 (+13) | 1/8 |

Length matching improves PCA scores by 6-13 points, consistent with mean difference.

### 3b. Pair Scaling (12 -> 20 -> 80, all matched)

| Pairs | 15dim score | 8dim score |
|---|---|---|
| 12 | 26 | **33** |
| 20 | 28 | 28 |
| 80 | 26 | 26 |

Same pattern as mean difference: more pairs does not help, and actively hurts at 80.

### 3c. Dimension Merging (15 -> 8)

| Pairs | 15dim score | 8dim score | Delta |
|---|---|---|---|
| 12 matched | 26 | **33** | +7 |
| 20 matched | 28 | 28 | 0 |
| 80 matched | 26 | 26 | 0 |

---

## 4. Layer Recommendations (PCA)

Criteria: `accuracy >= 0.85 AND max |Cohen's d| <= 0.8`

PCA finds slightly more recommended layers than mean difference, because lower 1-D probe accuracy means fewer layers exceed the 0.85 threshold in the first place, and those that do tend to be at layers where Cohen's d is also low.

| Variant | Dimension | Recommended layers |
|---|---|---|
| neg8dim_12pairs_matched | active_listening | [3] |
| neg8dim_12pairs_matched | anchoring | [23, 25] |
| neg8dim_12pairs_matched | composure | [26] (from selectivity) |
| neg8dim_80pairs_matched | anchoring | [3] |
| neg8dim_80pairs_matched | batna_awareness | [4] |
| neg8dim_80pairs_matched | empathy | [4, 22, 23] |
| neg8dim_80pairs_matched | strategic_concession_making | [25] |
| neg15dim_20pairs_matched | firmness | [24] |
| neg15dim_80pairs_matched | anchoring | [3] |
| neg15dim_80pairs_matched | batna_awareness | [4] |
| neg15dim_80pairs_matched | empathy | [4, 22, 23] |
| neg15dim_80pairs_matched | strategic_concession_making | [25] |

Compared to mean difference (only active_listening at layers [3,4,15,16] in neg8dim_80pairs), PCA has more recommended layers but this is misleading — PCA's lower probe accuracy gives it more "clean" layers by default (the direction is too weak to trigger the Cohen's d penalty).

---

## 5. Per-Dimension Traffic Light (Best Variant: neg8dim_12pairs_matched, PCA)

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

**Identical traffic light to mean difference.** Status is data-driven, not method-driven.

---

## 6. Orthogonal Projection (PCA)

After projecting out all 5 control dimensions from PCA vectors and re-running 1-D probes.

### Results (neg8dim_12pairs_matched)

| Dimension | Residual norm | Original acc | Cleaned acc | Drop | Verdict |
|---|---|---|---|---|---|
| active_listening | 0.876 | 0.662 | **0.685** | **-0.023** | IMPROVED |
| anchoring | 0.841 | 0.645 | 0.609 | +0.036 | GENUINE |
| batna_awareness | 0.865 | 0.434 | 0.424 | +0.010 | GENUINE |
| composure | 0.843 | 0.620 | 0.587 | +0.033 | GENUINE |
| creative_problem_solving | 0.805 | 0.505 | 0.495 | +0.010 | GENUINE |
| empathy | 0.857 | 0.604 | 0.536 | +0.068 | PARTIAL SURFACE |
| firmness | 0.867 | 0.556 | 0.534 | +0.022 | GENUINE |
| strategic_concession_making | 0.837 | 0.597 | 0.588 | +0.009 | GENUINE |
| **Mean** | **0.849** | **0.578** | **0.557** | **+0.021** | |

### PCA vs MD Projection Comparison (neg8dim_12pairs_matched)

| Dimension | MD drop | PCA drop | Interpretation |
|---|---|---|---|
| active_listening | -0.023 (improved) | -0.023 (improved) | Both methods: cleaning helps |
| anchoring | +0.005 | +0.036 | PCA more surface-dependent |
| batna_awareness | -0.005 (improved) | +0.010 | PCA slightly surface-dependent; MD pure |
| composure | +0.037 | +0.033 | Similar |
| creative_problem_solving | +0.018 | +0.010 | PCA slightly cleaner |
| empathy | +0.069 | +0.068 | Both: most surface-dependent |
| firmness | +0.034 | +0.022 | PCA slightly cleaner |
| strategic_concession_making | +0.046 | +0.009 | PCA much cleaner |
| **Mean drop** | **+0.024** | **+0.021** | Similar overall |

**Projection drops are nearly identical between methods** (~2.1-2.4%). Both methods extract directions with similar surface overlap. The difference is in total discriminative power: MD starts at 0.843 and drops to 0.820; PCA starts at 0.578 and drops to 0.557. The surface component is the same; PCA simply has less signal to begin with.

---

## 7. Why PCA Underperforms

PCA (PC1 of per-pair difference vectors) finds the direction of maximum variance in the difference vector cloud. This fails when:

1. **Noise variance exceeds signal variance.** With 12 pairs, the difference vectors have high variance from pair-specific noise (different sentence structures, topics, vocabulary). PC1 captures this noise rather than the shared concept direction.

2. **Surface confounds create dominant variance.** If some pairs differ primarily in length or hedging (data confound), that variance axis dominates PC1. Mean difference averages out pair-specific noise, exposing the shared signal.

3. **Small sample regime.** With only 12 difference vectors (or 20, or 80), PCA's covariance estimate is unreliable in 2560-dimensional space. The effective sample-to-dimension ratio is 0.005-0.031, far below the regime where PCA is trustworthy.

This is consistent with Im & Li (2025), who prove mean difference is optimal under pointwise loss, and with the general finding in the literature that MD > PCA for steering (Zou et al. 2023 used PCA for reading, not steering).

---

## 8. Key Conclusions

1. **PCA validity scores match mean difference** — both methods face the same data quality issues.

2. **PCA 1-D probe accuracy is ~54% lower than mean difference** (0.586 vs 0.903). PCA's PC1 direction is a poor proxy for the concept-discriminative direction with these sample sizes and confound structure.

3. **Orthogonal projection confirms both methods extract mostly genuine directions** — ~2% average accuracy drop for both. The surface component is small and similar across methods.

4. **The intervention rankings are identical:** length matching > dimension merging > pair scaling (which hurts). No PCA-specific intervention changes these priorities.

5. **PCA should not be used for steering.** Mean difference is strictly superior on probe accuracy while achieving identical validity scores and projection robustness. PCA is informative as a diagnostic (if MD and PCA diverge, the concept is poorly captured) but should not be the steering direction.

6. **For triangulation purposes:** In the best variant (neg8dim_12pairs_matched), PCA agrees with MD on dimension traffic lights (same RED/AMBER assignments) and on which dimension has the most surface dependence (empathy). This cross-method agreement strengthens confidence in the MD results.

---

## 9. Comparison with Mean Difference Summary

| Metric | Mean Difference | PCA | Winner |
|---|---|---|---|
| Best validity score | 33/100 | 33/100 | Tie |
| Best variant | neg8dim_12pairs_matched | neg8dim_12pairs_matched | Same |
| Mean 1D probe (best var) | 0.903 | 0.586 | **MD (+54%)** |
| Projection drop (best var) | 2.4% | 2.1% | Similar |
| Recommended layers (best var) | 0 | 3 | PCA* |
| Traffic lights (best var) | 5A/3R | 5A/3R | Same |

*PCA's "advantage" in recommended layers is an artifact: its lower probe accuracy means fewer layers hit the 0.85 threshold, and those that do tend to be at low-confound layers. This is not a real advantage — it means PCA rarely achieves high enough accuracy to be penalized.
