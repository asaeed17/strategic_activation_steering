# P4 Progress -- Evaluation Framework (Moiz)

## Project Context

COMP0087 Statistical NLP, UCL (due 2026-04-17). Activation steering applied to LLM negotiation on CraigslistBargains. Qwen 2.5-3B/7B. My role: P4 -- Evaluation Framework + Results & Analysis.

## TL;DR -- What We Found

- **Steering changes behavior but not outcomes.** Firmness at alpha=20 suppresses hedges by 27x and shortens responses by 22%. SCM at alpha=6 hardens concessions and makes responses 13% longer. These are real, measurable, and qualitatively different.
- **Outcome advantages are clamping artifacts.** Controlled paired comparison for SCM: +0.176 (p=0.09, not significant). Unclamped: +0.032 (p=0.87, null). Firmness unclamped: seller -0.04, buyer +0.002. Nearly all apparent advantage comes from games where the model agreed to irrational prices.
- **Role is the dominant variable.** Steering helps buyers and hurts sellers across all dimensions. Firmness: seller -27%, buyer +24%. SCM: seller -46%, buyer +78%. Aggregate scores mask opposite distributions.
- **Vectors encode surface patterns from biased pairs.** Contrastive pairs have 1.8x length bias, zero opener overlap, perfect ellipsis separation, 3.6x hedge clustering. Firmness vector suppresses hedges by 27x -- exactly the surface pattern the pairs encoded.
- **SCM does not generalize to multi-issue negotiation.** DonD Pareto rate: 16% (naive=9%, human=77%). The +37% Craigslist headline is domain-specific price stubbornness, not strategic reasoning.
- **The +37% headline was inflated.** S2 distribution mean is +24.9%, median +22.3%. True population effect likely +15-25%.
- **LLM judge captures role, not steering.** Sellers rated higher on 5/6 dimensions regardless of steering. Only clean steering signal: naturalness degradation at high alpha.
- **Contribution reframing:** The paper should be "how to properly evaluate steering in applied domains" (the evaluation framework itself), not "steering improves negotiation."

## Issues Identified

| # | Issue | Status | Summary | Key Evidence |
|---|-------|--------|---------|--------------|
| 1 | Evaluation contamination (same model, both agents) | OPEN | Baseline sees steered text, adapts in-context | Flag as limitation; fix requires 2x VRAM |
| 2 | score_deal() breaks on overlapping targets | CLOSED | Zero overlapping targets in dataset (0/5247 train, 0/597 val) | A1 audit; but 24% clamping is a separate real problem |
| 3 | No verification vectors encode claimed concepts | PARTIAL | Firmness suppresses hedges (surface pattern). SCM untested for PCA/probes | B1: 27x hedge suppression maps to pair bias |
| 4 | Fixed opening bid anchors results | RESOLVED | Results are sensitive: 50% opening=-0.065, 60%=+0.099, 70%=+0.031 | Session 8 G5 (n=10/condition) |
| 5 | 12 pairs per dimension (literature suggests ~80) | OPEN | Vectors may be noisy; split-half stability not checked | Flag as limitation |
| 6 | Anti-steerability (~50% of inputs) | REFRAMED | Not random (Tan et al.) -- systematic role effect. Sellers: 60% hurt. Buyers: 64% help | B3 role-separated analysis |
| 7 | Steerability bias (surface patterns in pairs) | CONFIRMED-SEVERE | 6 surface biases found in all 180 pairs | audit_pairs.py (session 3) |
| 8 | CraigslistBargains is 1D (can't measure strategic concession) | RESOLVED | DonD confirms SCM does not generalize to multi-issue | Session 8 G3: Pareto=16% |

## Phase A: Foundation Diagnostics (DONE)

Script: `phase_a_diagnostic.py` (CPU only).

### A1: Data Audit

CraigslistBargains is 100% clean: 5,247 train / 597 val scenarios, zero overlapping targets, zero bad prices. seller_target == listing_price always. buyer_target ~72% of listing. 6 categories (furniture 25%, housing 20%, bike 18%, car 13%, electronics 13%, phone 11%). Price ranges from ~$180 (electronics) to ~$10,933 (car).

### A2: Existing Results Diagnostic (firmness, alpha=20, 7B, 50 games)

| Finding | Key Numbers |
|---------|-------------|
| Role asymmetry | Seller: -0.273 avg advantage. Buyer: +0.242. Aggregate: -0.016 |
| Seller always finalizes | 100% of deals closed by seller |
| Steered is shorter | 23.5 vs 30.1 words (22% shorter). Shorter correlates with better outcomes (r=-0.56) |
| Anti-steerability | 50% help / 48% hurt / 2% neutral. Bimodal distribution, std=0.682 |
| Score clamping | 24% of games clamped (agreed price below buyer_target). Raw scores as extreme as -6.2 |
| S2 to S3 dropoff | SCM: +52.5% -> +37.4% (-29%). Anchoring: +35.4% -> +1.0% (-97%) |

### A3: score_deal() Analysis

No overlapping targets exist. The real issue is clamping: 24% of games have agreed prices outside the target range, producing extreme scores (+-1.0) that inflate averages. Recommendations: log raw + clamped scores, add `clamped` flag, report fraction affected separately.

## Phase B: Behavioral Metrics (DONE)

### B1: Per-turn Metrics

Script: `metrics_b1.py` (CPU). Outputs: `metrics_b1_enriched.json`, `metrics_b1_summary.json`.

Seven metrics implemented, all role-separated. Key findings on firmness alpha=20:

| Metric | Steered | Baseline | Key Takeaway |
|--------|---------|----------|--------------|
| Hedges/100w | 0.05 | 1.44 | **27x suppression** -- strongest behavioral signal |
| Response length | 23.5 words | 30.1 words | 22% shorter |
| Concession %gap/move | 30.7% | 8.9% | Steered concedes more per move (fewer, bigger moves) |
| First-offer distance | 14.7% | 13.8% | No anchoring effect |
| Clamped games | 24% have std~1.0 | | All-or-nothing, inflate variance |

Hedge suppression maps directly to contrastive pair bias (hedges 3.6x more common in negatives). The firmness vector's primary observable effect is "stop saying maybe/perhaps/I think."

### B2: Steering Decay

Script: `metrics_b2_decay.py`. Output: `metrics_b2_decay.json`.

**No decay detected.** Steered agent gets shorter and less yielding over turns (effects intensify, not fade). Max cumulative steered tokens = 139 words, well below the 300-500 token literature threshold for decay (Practitioner's Field Guide 2026). Our 8-turn negotiations are too short for decay to matter.

| Metric | Steered slope | p | Baseline slope | p |
|--------|--------------|---|----------------|---|
| Concession %gap | -0.191 | 0.0004 | -0.175 | 0.015 |
| Response length | -2.524 | 0.0005 | +0.616 | 0.398 |
| Hedge/100w | -0.006 | 0.820 | +0.311 | 0.003 |

### B3: Role-Separated Analysis

Script: `metrics_b3_roles.py`. Output: `metrics_b3_roles.json`. Produces 7 publication-ready tables.

Core insight: the aggregate 50/50 help/hurt split is an artifact of averaging two opposite distributions. As seller, steering hurts 60% of games. As buyer, it helps 64%. The mechanism: firmness is adaptive for buyers (resist overpaying) but maladaptive for sellers (can't close at high prices -- seller always capitulates via DEAL=).

Clamping is structurally symmetric (6/role, 24% each) but produces opposite extreme scores: -1.0 for seller, +1.0 for buyer. Unclamped games are near-zero in both roles (-0.04 seller, +0.002 buyer).

### B4: Alpha-Curve Analysis (S2 data)

No per-game data exists for SCM -- fast_search only saved summaries. Analysed S2 alpha trials (20 trials x 5 games each).

| Study | Pos% | Mean Adv | r(alpha,adv) | p(r) | Pattern |
|-------|------|----------|-------------|------|---------|
| SCM / middle / L18 | 95% | +0.249 | +0.50 | 0.025 | Monotonic -- likely real |
| Firmness / late / L27 | 95% | +0.117 | +0.53 | 0.016 | Monotonic -- real but small |
| Anchoring / middle / L18 | 80% | +0.072 | -0.09 | 0.69 | Inverted-U -- noise |

SCM/middle alpha breakdown: low (<2) +0.133, med (2-5) +0.191, high (>5) +0.330. Dose-response is the best signal quality indicator. Layer matters: SCM/middle=+24.9% vs SCM/late=+6.0%.

## Phase C: LLM Judge (DONE)

### C1-C2: Judge Design and Results

Script: `llm_judge.py`. Outputs: `judge_scores.json`, `judge_scores_raw.json`.

**Design:** 6 dimensions rated 1-5 Likert (firmness, persuasiveness, naturalness, coherence, information_management, strategic_reasoning). Guardrails: blind presentation (Agent A/B), position counterbalancing, anti-verbosity calibration, structured JSON with justifications, multi-model support (Gemini Flash, GPT, LLaMA 70B via Groq).

**Firmness judge results (50 games):** Role dominates. When steered=seller: steered scores higher on 5/6 dimensions. When steered=buyer: steered scores lower on all 6. The judge is rating the role, not the steering. Only role-independent signal: naturalness degradation (steered sounds less natural in both roles, consistent with Panickssery et al. 2024).

**SCM judge results (50 games, session 8):**

| Dimension | Steered | Baseline | Diff |
|-----------|---------|----------|------|
| firmness | 3.21 | 2.47 | +0.74 |
| persuasiveness | 2.59 | 2.28 | +0.31 |
| naturalness | 1.74 | 1.72 | +0.02 |
| coherence | 2.65 | 2.63 | +0.02 |
| info_management | 3.37 | 2.86 | +0.51 |
| strategic_reasoning | 2.60 | 2.25 | +0.35 |

Role pattern reverses from firmness: SCM-steered buyers score higher than baseline (firmness-steered buyers scored lower). No clear naturalness degradation for SCM (unlike firmness). Position bias is negative (mean -0.37): judge systematically favours Agent B (presented second).

**Within-role judge-advantage correlations (session 8):**

For steered-as-seller (n=25): firmness r=+0.63*, info_management r=+0.61*, strategic_reasoning r=+0.65*, coherence r=+0.57*. But unclamped-only (n=14): only info_management survives (r=+0.58*). Correlations are inflated by clamped games.

For steered-as-buyer (n=25): only info_management significant (r=+0.43*).

### C3: Synthesis -- What Activation Steering Actually Does to Negotiation

This section connects all findings (A1-A3, B1-B4, C1-C2) into the evidence structure for the paper.

#### The causal chain: from contrastive pairs to negotiation outcomes

```
Contrastive pairs (180 pairs, 15 dimensions)
  |  Severe surface biases: 1.8x length, zero opener overlap,
  |  perfect ellipsis separation, 3.6x hedge clustering
  v
Vector extraction (mean-diff, last-token activations)
  |  Vectors encode surface patterns, not deep concepts
  |  Evidence: cos(SCM, firmness) = 0.179 -- different vectors,
  |  but both produce role-dependent outcomes
  v
Behavioral changes during generation (REAL, MEASURABLE)
  |  Firmness alpha=20: 27x hedge suppression, 22% shorter, terser
  |  SCM alpha=6: 13% longer, hardened concessions, minimal hedge change
  |  These are qualitatively different -- not same vector under different label
  v
Outcome scores (CONFOUNDED)
  |  Dominated by: (1) role effects, (2) clamping artifacts
  |  Controlled paired effect: +0.176 (p=0.09, not significant)
  |  Unclamped paired effect: +0.032 (p=0.87, null)
  v
Cross-dataset transfer (FAILS)
  |  DonD Pareto rate: 16% (naive=9%, human=77%)
  |  SCM does not encode strategic reasoning
  v
LLM judge perception (DOMINATED BY ROLE)
    Sellers rated higher on 5/6 dimensions regardless of steering
    Only clean steering signal: naturalness degradation (firmness only)
    Within-role correlations partly inflated by clamped games
```

#### Five core findings, ranked by confidence

**1. Steering changes model behavior. (HIGH confidence)**
Undeniable. Firmness at alpha=20 suppresses hedge words by 27x (0.05 vs 1.44 per 100 words), shortens responses by 22%, eliminates capitulation openers. SCM at alpha=6 produces different changes: slightly longer responses, hardened concession patterns, minimal hedge change. The vectors are doing something -- the question is whether that something helps.

**2. Outcome advantages are driven by clamping, not genuine improvement. (HIGH confidence)**
The most important finding. In both firmness and SCM, stripping clamped games reduces the advantage to near zero:
- Firmness unclamped: seller -0.04, buyer +0.002
- SCM unclamped paired: +0.032 (p=0.87)
- Firmness moderate alpha=5 on 3B: 70% clamped, unclamped +0.045

The "advantage" is not the steered agent being a better negotiator -- it's the steered agent (or its opponent) sometimes agreeing to irrational prices, and the scoring formula turning that irrationality into extreme scores. This is a measurement artifact, not a steering success.

**3. Role is the dominant variable. (HIGH confidence)**
Across all dimensions, models, and alphas:
- Steering helps buyers and hurts sellers
- The game structure requires sellers to finalize (DEAL=) -- sellers capitulate 100% of the time
- Firmness: seller -27%, buyer +24%. SCM: seller -46%, buyer +78%
- The aggregate effect is an average of two opposite distributions
- The LLM judge shows the same pattern: sellers rated higher on firmness/strategy regardless of steering

This is a structural property of the CraigslistBargains game loop + Qwen model, not a steering phenomenon.

**4. Vectors encode surface patterns from biased contrastive pairs. (MODERATE confidence)**
Evidence chain:
- Pairs audit: hedges 3.6x more common in negatives, "Okay/Sure/Fine" open 35/180 negatives and 0/180 positives
- B1: firmness vector suppresses hedges by 27x -- exactly the surface pattern the pairs encoded
- B1: SCM vector does NOT suppress hedges (only 1.2x) despite its own biases (2.06x length ratio)
- Different pair biases produce different behavioral changes (firmness -> hedge suppression, SCM -> concession hardening), suggesting the vectors capture something from the pairs -- possibly a mix of surface patterns and genuine concepts
- Cannot fully disentangle without better-controlled pairs (P1's job)

**5. SCM does not encode multi-issue strategic reasoning. (HIGH confidence)**
The DonD result is the cleanest falsification:
- If SCM encodes strategic reasoning, it should improve Pareto efficiency on multi-issue negotiation
- Pareto rate: 16% (barely above naive 9%, far below human 77%)
- Efficiency: 0.713 (marginally above naive 0.694)
- The Craigslist result (+15.9% uncontrolled, +17.6% paired) is domain-specific price stubbornness
- The label "strategic concession making" is a misnomer -- the vector likely encodes "resist conceding"

#### What the LLM judge adds

The judge is not the load-bearing evidence -- the paired comparison and DonD falsification are. But it adds:

1. **Naturalness degradation signal.** Firmness at alpha=20 degrades naturalness in both roles (-0.72 seller, -0.22 buyer). SCM at alpha=6 does not. Practical implication: alpha=20 is too aggressive; alpha=6 may be a sweet spot.

2. **Within-role seller correlations.** For steered sellers, judge firmness (r=+0.63), strategic_reasoning (r=+0.65), info_management (r=+0.61), coherence (r=+0.57) all predict advantage. But correlations are inflated by clamped games (unclamped: only info_management survives).

3. **Role-perception asymmetry confirmation.** The judge independently confirms that role is the dominant signal. Convergence between behavioral metrics (B3), outcomes (session 8), and qualitative ratings (judge) is the strongest evidence that role effects are pervasive.

#### What the paper should claim

**Primary claim:** "Activation steering produces measurable behavioral changes in LLM negotiation agents, but these changes do not translate to statistically significant outcome improvements in controlled paired comparison. The apparent advantages reported in uncontrolled settings are driven by scoring artifacts (price clamping) and role asymmetry in the game structure."

**Supporting claims:**
- "The steering vectors encode surface patterns from biased contrastive pairs, as evidenced by the direct mapping between pair-level hedge bias and deployment-level hedge suppression."
- "The effectiveness of steering is strongly role-dependent: it helps buyers and hurts sellers, because firmness is adaptive for the defending role but maladaptive for the closing role."
- "The `strategic_concession_making` vector does not generalize to multi-issue negotiation, producing Pareto efficiency barely above naive allocation (16% vs 9%) and far below human performance (77%)."

**Contribution framing:** The contribution is the evaluation framework itself -- the methodology that systematically identifies confounds (role effects, clamping, surface pattern encoding, aggregate masking) that inflate apparent steering effects. This is a "how to properly evaluate steering in applied domains" paper, not a "steering improves negotiation" paper.

#### Open methodological limitations

1. **Same-model evaluation contamination (Issue 1).** Both agents share one model instance. The baseline sees the steered agent's text and may adapt in-context. Fixing requires separate model instances (doubles VRAM).
2. **N is thin.** 50 paired scenarios, 25 per role. After removing clamped games, n=20 unclamped pairs. Power to detect small effects (d=0.2-0.3) is low.
3. **Single model family.** All results are on Qwen 2.5 (3B and 7B).
4. **12 pairs per dimension.** Literature suggests ~80 for reliable vectors.
5. **Single judge model.** Only Gemini Flash. Inter-model reliability not computed.

## Phase D: Paper Sections (TODO)

### D1: Evaluation Framework section
- Describe all metrics and justify each. Describe LLM judge methodology. Present role-separated analysis rationale. Present clamping problem and handling. Reference contrastive pairs audit as motivation.

### D2: Results & Analysis section
- Headline: paired comparison +0.176 (p=0.09, d=0.24). Key robustness check: unclamped +0.032 (null). Role-separated tables for firmness and SCM. Behavioral changes as evidence steering works mechanistically. DonD as falsification. Judge as supporting evidence. Honest framing: negative-but-informative result.

## Next Phase: CaSiNo Integration (TODO)

CaSiNo (Chandra et al. 2021) provides 1,030 campsite negotiation dialogues with 10 human-annotated strategy labels across 4,615 utterances. This addresses the core weakness of Craigslist: all our findings are dataset-dependent (role asymmetry, clamping, 1D scoring).

**Purpose:** Two-pronged. (1) Build better contrastive pairs grounded in CaSiNo's strategy annotations — fixing the surface bias problem (Issue 7) and low pair count (Issue 5). (2) Run steered games on CaSiNo's multi-issue task (food, water, firewood) and evaluate using strategy classification rather than just price outcomes.

**Why CaSiNo over more Craigslist:**
- Strategy annotations (self-need, showing-empathy, promote-coordination, etc.) give validated behavioral labels instead of crude proxies (hedge count, response length)
- Multi-issue symmetric negotiation — no seller/buyer role asymmetry
- Deterministic scoring (High=5pts, Medium=4pts, Low=3pts) — no clamping problem
- Pairs extracted from CaSiNo are in-distribution for CaSiNo games (Hao et al.: "CAE only reliable in-distribution")

**Four phases:**
1. Extract contrastive pairs from CaSiNo annotations (~80+ pairs/dimension for 6-7 strategies with 250+ examples)
2. Extract steering vectors using existing pipeline (no code changes)
3. Build CaSiNo game loop (follows DonD pattern: loader, game loop, scoring, Pareto check)
4. Evaluate: strategy label frequencies (via LLM judge) + outcome points + Pareto efficiency

**Key risks mitigated:** Domain mismatch eliminated (same domain for pairs and games). Pair count meets literature threshold (~80). Split-half stability and PCA separation checks planned before steering.

## Remaining Work

**CaSiNo integration (primary):**
- Build CaSiNo pair extraction script (Phase 1)
- Extract vectors with new pairs, validate with PCA + split-half (Phase 2)
- Build CaSiNo game loop and scoring (Phase 3)
- Run steered games + strategy classification evaluation (Phase 4, needs GPU)

**Other experiments:**
- Negative alpha experiments -- reverse steering to confirm causal control (P6 scope, needs GPU)
- Alpha vs behavioral metrics -- plot alpha vs hedge rate / concession rate instead of alpha vs advantage
- LLM judge as behavioral classifier -- correlate judge scores with alpha, not price outcomes

**Remaining validation:**
- Second judge model -- inter-model reliability with LLaMA 70B or GPT-4o
- Split-half vector stability -- cosine similarity between first vs second half of pairs (Issue 5)
- Same-model contamination -- characterize or flag as limitation (Issue 1)

**Paper writing (Phase D):**
- D1: Evaluation Framework section
- D2: Results & Analysis section

## Key Decisions

1. We are P4 (Metrics & Evaluation), not P1-P3 or P5-P6.
2. Lightweight-first: validate pipeline before building sophisticated metrics.
3. LLM judge via API (Groq/Gemini). Steering needs local GPU; evaluation does not.
4. Role-separated reporting is mandatory. Aggregate -1.6% hides seller=-27% vs buyer=+24%.
5. Coherence is a first-class metric. Degenerate-text "wins" are not results.
6. Clamped games must be flagged and reported separately (24% of firmness, 50% of SCM).
7. Seller finalizes 100% of deals -- structural asymmetry that affects all interpretation.
8. Anti-steerability is real but role-dependent, not random (reframed from Tan et al.).
9. Contrastive pairs have severe surface biases. Every dimension has at least one confound.
10. Hedge suppression (27x) is the primary observable effect of firmness steering -- causal chain from pair bias to deployment behavior.
11. "Anti-steerability" is actually role-dependent effectiveness: firmness is adaptive for buyers, maladaptive for sellers.
12. The +37% SCM headline is inflated. S2 mean=+24.9%, true effect likely +15-25%.
13. Alpha-curve dose-response shape is the best signal quality indicator (monotonic=real, inverted-U=noise).
14. Role dominates LLM judge scores. Only naturalness shows role-independent steering signal.
15. Firmness outcomes are clamping artifacts (unclamped seller -0.04, buyer +0.002).
16. Paired SCM effect is +0.176 (p=0.09, d=0.24) -- not significant at p<0.05.
17. Unclamped paired effect is +0.032 (p=0.87) -- essentially zero.
18. SCM does NOT suppress hedges (1.2x vs firmness 27x). SCM makes agent 13% longer (firmness: 22% shorter). Qualitatively different behavioral changes.
19. SCM does not generalize to multi-issue (DonD Pareto=16%, advantage=-0.052).
20. Firmness at moderate alpha=5 on 3B has WORSE clamping (70%) than alpha=20 on 7B (24%).
21. Within-role judge correlations are useful for sellers, not buyers. Unclamped: only info_management survives.

## Dataset Strategy

| Dataset | Role | Status | Key Strengths |
|---------|------|--------|---------------|
| CraigslistBargains | Primary evaluation | INTEGRATED | Naturalistic, real items/prices, full pipeline |
| Deal or No Deal (Lewis et al. 2017) | Falsification of SCM claim | INTEGRATED | Multi-issue, known utility functions, Pareto-testable |
| CaSiNo (Chandra et al. 2021) | Judge validation | TODO | 1,030 dialogues with strategy annotations mapping to our dimensions |

## Literature

| Paper | Why it matters |
|-------|---------------|
| Panickssery et al. 2024 (CAA) | Large steering magnitudes degrade text quality. Must measure coherence. |
| Hao et al. 2025 (ICLR WS) | ~80 pairs for diminishing returns. CAE only reliable in-distribution. |
| **Tan et al. 2024 (NeurIPS)** | **Anti-steerability (~50% opposite). Steerability bias (surface patterns). Essential for our variance.** |
| Xia et al. 2024 | Buyer harder than seller. Formalizes bargaining as asymmetric game. |
| Wang et al. 2025 | Turn-level evaluation grounded in Theory of Mind. |
| He et al. 2018 (CraigslistBargains) | Dataset paper. |
| Lewis et al. 2017 (Deal or No Deal) | Multi-issue negotiation. Validation dataset. |
| Chandra et al. 2021 (CaSiNo) | Strategy-annotated dialogues. Judge validation. |
| "Sober Look at Steering Vectors" (AF) | Steering degrades performance equivalent to halving pre-training compute. |
| Practitioner's Field Guide 2026 | Steering fades after 300-500 tokens. |
| HAMBA metric (2025) | Consumer Surplus + Negotiation Power + Acquisition Ratio. |

## Session Log

### Session 8 (2026-02-27) -- GPU run + full analysis
Ran all 5 experiments on AWS g4dn.xlarge (Tesla T4). Scripts: `run_eval.py --all`, `analyse_eval.py`, `llm_judge.py`. Total runtime: 40 min. PyTorch downgraded to 2.6.0+cu126 (CUBLAS errors on T4). PCA hit NaN; mean_diff worked fine.
- **G1+G2:** Paired SCM effect +0.176, p=0.09 (ns). Unclamped +0.032, p=0.87 (null). 50% clamped. See Phase C3.
- **G1 B1 metrics:** SCM is 13% longer, 1.2x hedge ratio (vs firmness 27x), negative concession (hardens). See Phase B1.
- **G1 B3 roles:** SCM seller -0.461 (72% hurt), buyer +0.780 (92% help). Even more extreme than firmness. See Phase B3.
- **G3 DonD:** Pareto=16%, efficiency=0.713, advantage=-0.052. SCM fails to generalize. See C3 Finding 5.
- **G4 firmness moderate alpha:** 70% clamped at alpha=5/3B (worse than 24% at alpha=20/7B). Unclamped +0.045.
- **G5 opening bid:** 50%=-0.065, 60%=+0.099, 70%=+0.031. Non-monotonic, sensitive to opening. Issue 4 resolved.
- **G1 judge:** SCM steered scores higher on 5/6 dimensions. Role pattern reverses from firmness. Position bias -0.37.
- **G1 judge correlations:** Seller: 4/6 significant, but unclamped only info_management survives. Buyer: only info_management. See Phase C2.
- **Cosine sims:** SCM-firmness=0.179, SCM-anchoring=0.245, firmness-anchoring=0.223. All moderately different.

### Session 7 (2026-02-27) -- GPU evaluation setup
Created `run_eval.py` (~450 lines). Orchestrates all GPU experiments with single model load, interleaved G1/G2 paired design, incremental per-game saves. See Phase C3 for experiment design table.

### Session 6 (2026-02-27) -- S2 analysis, judge results, synthesis
Completed B4 alpha-curve analysis (see Phase B4). Completed C1/C2 judge on firmness (see Phase C). Identified role dominance in judge scores. Added Decisions 12-15.

### Session 5 (2026-02-27) -- Phase B metrics
Created `metrics_b1.py` (7 per-turn metrics), `metrics_b2_decay.py` (no decay found), `metrics_b3_roles.py` (7 role-separated tables). Key findings: 27x hedge suppression, role-dependent anti-steerability, no steering decay. See Phases B1-B3.

### Session 4 (2026-02-27) -- Deal or No Deal integration
Created `deal_or_no_deal.py`. Full game loop, Pareto scoring, brute-force optimality check. 4,086 selfplay scenarios. Naive Pareto=9%, human=77%. Validated against paper examples. See Dataset Strategy.

### Session 3 (2026-02-27) -- Contrastive pairs audit + judge design
Created `audit_pairs.py`. Found 6 critical surface biases in all 180 pairs: 1.8x length, zero opener overlap, perfect ellipsis separation, capitulation word segregation, 3.6x hedge clustering, 5.1x yielding clustering. Flagged recommendations for P1. Started LLM judge design (C1).

### Session 2 (2026-02-27) -- Phase A diagnostics
Created `phase_a_diagnostic.py`. A1: dataset 100% clean, closed Issue 2. A2: 7 findings from results.json (role asymmetry, seller finalizes, shorter=better, anti-steerability, clamping, category effects, S2-S3 dropoff). A3: clamping is the real issue, not overlapping targets.

### Session 1 (2026-02-27) -- Initial review
Reviewed full codebase and Damon's search results. Identified 8 critical issues. Literature review (12 papers). Created this tracking document. Established lightweight-first strategy.
