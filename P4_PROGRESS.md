# P4 Progress — Metrics & Evaluation (Moiz)

## Project Context

**Course:** COMP0087 Statistical NLP, UCL. Due 2026-04-17.
**Project:** Activation steering (representation engineering) to enhance LLM negotiation on CraigslistBargains.
**Model:** Qwen 2.5-3B/7B on RTX 4070 (12GB VRAM).
**My role:** P4 — Evaluation Framework + Results & Analysis.

## Current State of the Codebase

### What exists
- `extract_vectors.py` — extracts steering vectors via mean-diff and PCA (15 dimensions, 12 pairs each)
- `apply_steering.py` — runs steered vs baseline negotiation games, scores by price split
- `fast_search_steering.py` — 3-stage hyperparameter search (grid + TPE + validation)
- `negotiation_steering_pairs.json` — 180 contrastive pairs across 15 dimensions
- `results.json` — 50 games, firmness, alpha=20 (advantage = -0.016, basically null)
- `results/fast_run/` — search results from Damon's sweep

### Key experimental findings so far
- `strategic_concession_making` / layer 18 / alpha=6.12 gave **+37.4% advantage** in Stage 3 validation (20 games)
- `anchoring` / layer 18 / alpha=4.07 gave +0.010 in validation (washed out from +35% in S2 — likely noise)
- `firmness` at alpha=20 (original run) slightly hurt performance
- All runs: 100% agreement rate, avg 8 turns per game

---

## Critical Issues to Fix Before Metrics Matter

These were identified through literature review and a research-focused review session. Each must be addressed (at least a lightweight pass) before P4 metrics are trustworthy.

### ISSUE 1: Evaluation contamination — agents are not independent
- **Problem:** Same model generates both agents. The baseline sees the steered agent's utterances and adapts in-context. We're measuring "steered vs baseline-that-has-been-influenced-by-steered", not "steered vs independent baseline".
- **Impact:** Advantage scores are inflated. The +37% may partly be a context-contamination artifact.
- **Lightweight fix:** Run a diagnostic comparing same-model games vs a setup where we log the steered agent's offers independently and check if the baseline's behavior shifts in response. At minimum, flag this as a limitation. The "gold standard" fix would be two separate model instances, but that doubles VRAM.
- **Status:** NOT STARTED

### ISSUE 2: score_deal() breaks on overlapping targets
- **Problem:** When seller_target < buyer_target (targets overlap), span is negative. The function clamps to [0,1] but silently produces garbage scores. These pollute averages.
- **Impact:** Unknown fraction of games have crossed targets. Averages may be meaningfully wrong.
- **Lightweight fix:** (a) Audit how many scenarios have overlapping targets. (b) Filter them out or score them separately. (c) Report the fraction excluded.
- **Status:** ✅ RESOLVED — Phase A1 audit found **zero overlapping targets** in both train (0/5247) and validation (0/597). seller_target == listing_price for 100% of scenarios. The span≤0 codepath is dead code for this dataset. **However**, a related issue was found: 24% of games have agreed prices below buyer_target, causing score clamping (see A3 findings).

### ISSUE 3: No verification that vectors encode claimed concepts
- **Problem:** The pipeline extracts a "firmness" direction and immediately uses it to steer. No check that this direction actually encodes firmness vs a correlated artifact (response length, confidence, verbosity).
- **Impact:** Could be steering verbosity and winning because longer responses contain more counter-arguments.
- **Lightweight fix:** (a) Correlate steering with response length — does alpha increase token count? (b) Quick PCA visualization of positive vs negative activations — do they separate? (c) This is really P2's job (probes), but P4 needs a sanity check.
- **Status:** PARTIAL — Phase A2 found that firmness steering at α=20 makes responses 22% **shorter** (23.5 vs 30.1 words), and shorter responses correlate with better outcomes (r=-0.56). This rules out the "verbosity wins" hypothesis for firmness. Still need: (a) same check for strategic_concession_making at α≈6, (b) PCA visualization, (c) check for other surface pattern correlates.

### ISSUE 4: Fixed opening bid anchors all results
- **Problem:** Buyer always opens at 60% of listing price. All results are conditional on this constant.
- **Impact:** Results may not generalize. If we change to 50% or 70%, advantage scores could flip.
- **Lightweight fix:** Run a small sensitivity test (10 games each at 50%, 60%, 70% opening). Check if advantage is stable.
- **Status:** NOT STARTED

### ISSUE 5: 12 pairs per dimension is probably too few
- **Problem:** Literature (Hao et al. 2025) suggests ~80 pairs is where diminishing returns kick in. We have 12.
- **Impact:** Extracted vectors may be noisy. This is P1/P5's job to expand, but P4 should flag it.
- **Lightweight fix:** Just flag as a known limitation. Check cosine similarity between vectors extracted from first 6 pairs vs last 6 pairs — if they're very different, the vector is unstable.
- **Status:** NOT STARTED

### ISSUE 6: Anti-steerability — steering may backfire on ~50% of inputs
- **Problem:** Tan et al. (NeurIPS 2024) found that for many behaviors, steering produces the *opposite* of the intended behavior on ~50% of inputs. They call this "anti-steerability."
- **Impact:** Our huge per-game variance (Game 0: +0.11, Game 1: +0.94) is likely this effect. Reporting only the mean hides that steering may actively hurt on many scenarios.
- **Lightweight fix:** (a) Report variance and std dev alongside means. (b) Compute the fraction of games where steered agent does *worse* than baseline. (c) Characterize which scenarios are anti-steerable (price range? category? target gap?).
- **Status:** ✅ CONFIRMED — Phase A2 found exactly 50% help / 48% hurt / 2% neutral in firmness α=20 results. Distribution is bimodal: games cluster at ±1.0, not near zero. Std=0.682. This is textbook anti-steerability. Still need: (a) same analysis for strategic_concession_making results, (b) characterize which scenario features predict anti-steerability (blocked on getting per-game transcripts for fast_run results).

### ISSUE 7: Steerability bias — vectors may encode surface patterns, not concepts
- **Problem:** Tan et al. found models learn "steerability bias" — superficial token patterns rather than high-level concepts. If all "firm" examples start with "I understand but..." and all "yielding" examples start with "You're right...", the vector encodes sentence-opening patterns, not firmness.
- **Impact:** The steering vector may just be a starts-with-X vector, not a firmness vector.
- **Lightweight fix:** (a) Audit `negotiation_steering_pairs.json` for structural patterns in positive vs negative responses (opening words, sentence length, punctuation). (b) Flag for P1/P2 to address. (c) In our evaluation, check if steered outputs share surface patterns with the positive training examples.
- **Status:** CONFIRMED — SEVERE. See Session 3 audit results below. Every bias Tan et al. warned about is present in our pairs.

### ISSUE 8: CraigslistBargains is one-dimensional — can't measure strategic concession
- **Problem:** CraigslistBargains is single-issue (price only). "Strategic concession making" means trading something you value less for something you value more — impossible when the only variable is money. Our best-performing vector may not be doing what the label says.
- **Impact:** The +37% from `strategic_concession_making` might just be a different flavor of firmness/anchoring, not genuine multi-issue strategic reasoning. We can't distinguish these on a 1D dataset.
- **Lightweight fix:** Flag as major limitation. Consider cross-dataset validation (see Dataset Strategy below).
- **Status:** NOT STARTED

---

## Dataset Strategy

CraigslistBargains is wired up and stays as the primary dataset. But for a defensible paper, we should advocate for cross-dataset evaluation to test whether our vectors generalize.

### CraigslistBargains (primary — keep)
- **Strengths:** Naturalistic language, real items with prices, already integrated in pipeline.
- **Weaknesses:** Single-issue (price only), ~~known data quirks (crossed targets, weird pricing)~~ [A1 audit: no crossed targets, no bad prices — dataset is 100% clean], can't test multi-issue concession strategies.
- **Role:** Main evaluation arena. All metrics built here first.

### Deal or No Deal (strongly recommended — add for validation)
- **Paper:** Lewis et al. 2017 (Facebook AI Research)
- **Strengths:** Multi-issue (books, hats, balls with hidden utility values per player). Can directly measure Pareto efficiency. Known value functions make scoring unambiguous.
- **Weaknesses:** Simpler language patterns (2017 human data). Not "LLM-style" dialogue.
- **Role:** Validation dataset. If `strategic_concession_making` actually captures multi-issue reasoning, it should improve Pareto efficiency here. If it only works on Craigslist, it's just price stubbornness with a fancy label.
- **Effort to integrate:** Moderate — need new scenario loader and scoring function, but game loop is reusable.

### CaSiNo (useful for LLM judge validation)
- **Paper:** Chandra et al. 2021 (campsite resource negotiation)
- **Strengths:** 1,030 dialogues with explicit strategy annotations ("elicit preferences", "show empathy", "make trade-off"). These map directly to our behavioral steering dimensions.
- **Weaknesses:** Small dataset (1,030 dialogues). Campsite-specific.
- **Role:** LLM judge validation. Use CaSiNo's human-annotated strategy labels to test whether our LLM judge can correctly identify strategies. If the judge can't match CaSiNo annotations, it won't reliably evaluate our steered transcripts either.
- **Effort to integrate:** Low — only needed for judge validation, not full game simulation.

**Recommendation to team:** Keep Craigslist as primary. Add Deal or No Deal as a validation dataset to test the "strategic concession" claim. Use CaSiNo annotations to validate the LLM judge. This gives us cross-dataset evidence, which is much stronger than single-dataset results.

---

## P4 Deliverables — Lightweight Pass First, Then Depth

The strategy: build lightweight versions of everything to validate the pipeline is sound, THEN go deep on the real metrics. This way the other task owners (P1-P3, P5-P6) can also benefit from early diagnostics.

### Phase A: Foundation Fixes — COMPLETED 2026-02-27

Script: `phase_a_diagnostic.py` (run with `python phase_a_diagnostic.py`, no GPU needed).

#### A1. Data audit on CraigslistBargains — DONE

**Result: Dataset is cleaner than expected. No overlapping targets at all.**

| Metric | Train | Validation |
|--------|-------|------------|
| Total entries | 5,247 | 597 |
| Parse failures | 0 | 0 |
| Overlapping targets (S < B) | **0 (0.0%)** | **0 (0.0%)** |
| Zero/negative price | 0 | 0 |
| Zero/negative target | 0 | 0 |
| Clean scenarios | 5,247 (100%) | 597 (100%) |

**Dataset structure:**
- `seller_target == listing_price` for 100% of scenarios (ratio = 1.000 exactly)
- `buyer_target ≈ 72% of listing_price` (mean train=0.718, val=0.727)
- Target spread as % of listing price: mean≈28%, std≈16%

**Category distribution (train):**

| Category | N | % | Mean Price | Median |
|----------|---|----|-----------|--------|
| furniture | 1,301 | 24.8% | $296 | $125 |
| housing | 1,064 | 20.3% | $2,448 | $2,250 |
| bike | 940 | 17.9% | $635 | $250 |
| car | 698 | 13.3% | $10,933 | $8,700 |
| electronics | 686 | 13.1% | $180 | $90 |
| phone | 558 | 10.6% | $228 | $150 |

**Implications for Issue 2:** Overlapping targets don't exist in this dataset. The `score_deal()` span≤0 path was never triggered. Issue 2 is **closed as non-applicable** for CraigslistBargains. However, the clamping issue found in A3 is a separate, real problem (see below).

#### A2. Existing results diagnostic — DONE

**results.json: 50 games, firmness, alpha=20, Qwen 2.5-7B**

##### Finding 1: Massive role asymmetry

| Role | N | Mean Advantage | Median | Std | Win Rate |
|------|---|---------------|--------|-----|----------|
| Steered as SELLER | 25 | **-0.273** | -0.136 | 0.576 | 36% |
| Steered as BUYER | 25 | **+0.242** | +0.257 | 0.693 | 64% |
| Aggregate | 50 | -0.016 | +0.030 | 0.682 | 50% |

The aggregate -1.6% hides that the buyer role is structurally advantaged. Steering helps as buyer (+24%) but actively hurts as seller (-27%). **Aggregate reporting is misleading.** Confirms decision #4 (role-separated reporting is mandatory).

##### Finding 2: Seller always finalizes

**100% of deals** are closed by the seller saying `DEAL=<price>`. The buyer never finalizes. This is a structural asymmetry: the seller capitulates to buyer pressure, not the other way around. The game loop structure (seller moves first each turn) may contribute.

##### Finding 3: Negative response length correlation

| Agent | Mean Words | Median | Std |
|-------|-----------|--------|-----|
| Steered (α=20) | 23.5 | 25 | 11.6 |
| Baseline (α=0) | 30.1 | 32 | 11.5 |

- Steered agent is **22% shorter** than baseline (ratio=0.779)
- Pearson corr(steered_word_count, advantage) = **-0.558** (strong negative)
- This is the **opposite** of the verbosity-wins hypothesis. Shorter steered responses do better. At α=20, "firmness" steering appears to make the agent more terse, not more verbose.
- **For Issue 3:** This partially addresses the verbosity confound concern. The vector is not just "talk more." But "talk less and win" could still be a surface pattern rather than genuine firmness.

##### Finding 4: Anti-steerability confirmed (Issue 6)

| Outcome | Count | % |
|---------|-------|---|
| Steering helps (adv > 0) | 25 | 50% |
| Steering hurts (adv < 0) | 24 | 48% |
| Neutral | 1 | 2% |

Advantage distribution: mean=-0.016, **std=0.682**, min=-1.000, max=+1.000, Q1=-0.659, Q3=+0.490.

**Histogram:**
```
  <-0.3        :  19 ###################
  -0.3 to -0.1 :   4 ####
  -0.1 to 0.0  :   1 #
  0.0 to 0.1   :   3 ###
  0.1 to 0.3   :   7 #######
  >0.3         :  16 ################
```

Bimodal. Games tend to be big wins OR big losses, not small shifts. This is consistent with Tan et al.'s anti-steerability: the vector pushes ~half of scenarios in the wrong direction.

##### Finding 5: Score clamping distorts 24% of games

12 of 50 agreed games (24%) have the agreed price **below the buyer's target**. The raw score formula produces values like -2.6 or -6.2, which get clamped to 0.0 (seller) and 1.0 (buyer). Every clamped game becomes advantage = -1.000 for whichever role is seller.

Examples:
- Game 19: agreed=$950, seller_target=$1488, buyer_target=$1339. Raw seller_score=-2.611 → clamped to 0.0
- Game 13: agreed=$1180, seller_target=$1800, buyer_target=$1620. Raw seller_score=-2.444 → clamped to 0.0

**Impact:** Clamping compresses what should be varying degrees of bad deals into a single -1.0 floor. This inflates variance and makes the seller role look worse than it already is. The proper fix: either (a) don't clamp and report raw scores (loses the [0,1] sum-to-1 property), or (b) log both raw and clamped scores and report the fraction affected.

##### Finding 6: Category and price effects (low N, unreliable)

| Category | N | Mean Adv | Std | Win% |
|----------|---|----------|-----|------|
| phone | 5 | -0.633 | 0.514 | 20% |
| bike | 12 | -0.274 | 0.660 | 33% |
| furniture | 6 | +0.043 | 0.448 | 50% |
| electronics | 8 | +0.127 | 0.766 | 50% |
| car | 8 | +0.121 | 0.498 | 63% |
| housing | 11 | +0.312 | 0.765 | 73% |

Small N per category — don't read too much into this. But the phone/bike categories (lower prices, smaller spreads) look bad while housing/car (higher prices, larger spreads) look better.

##### Fast Search Diagnostic (Stage 2 → 3 dropoff)

| Config | S2 Best | S3 Validation | Drop |
|--------|---------|--------------|------|
| strategic_concession_making / middle / α=6.12 | +0.525 | +0.374 | -29% |
| anchoring / middle / α=4.07 | +0.354 | +0.010 | **-97%** |

Anchoring collapsed from +35% to +1% going from 5-game trials to 20-game validation. Classic overfitting to noise. strategic_concession_making held up better (still +37% in validation) but the 29% drop from S2 to S3 suggests the true effect is lower than the headline number.

Stage 2 overall: strategic_concession_making/middle had 95% positive trials (19/20) with mean advantage +0.249. The alpha-advantage correlation was 0.475 — near-monotonic, suggesting higher alpha = more effect (up to a point). Anchoring had only 80% positive trials and near-zero alpha correlation (-0.09), meaning the "best" alpha was just the luckiest draw.

#### A3. score_deal() analysis — DONE

**No overlapping targets exist** in CraigslistBargains (see A1). The `span <= 0` codepath in `score_deal()` is dead code for this dataset.

**But clamping is the real issue.** 24% of games in results.json have the agreed price outside the [buyer_target, seller_target] range (specifically, below buyer_target). The clamp to [0,1] masks the severity — a deal at 50% below buyer_target looks the same as a deal at 5% below.

**Recommendations:**
1. Log raw (unclamped) scores alongside clamped scores for transparency
2. Add a `clamped` boolean flag to each game result
3. Report the fraction of clamped games separately
4. Consider: is "agreed price below buyer target" a model failure (buyer overshoots their own goal)? Or is it the LLM not understanding target constraints? This matters for interpretation.

### Session 3 Work: Contrastive Pairs Audit + LLM Judge Design

#### Contrastive Pairs Steerability Bias Audit (Issue 7) — DONE

Ran `audit_pairs.py` on all 180 contrastive pairs across 15 dimensions. The audit checks for systematic surface-level differences between positive and negative responses that steering vectors could latch onto instead of encoding the intended behavioral trait.

**Script:** `audit_pairs.py` — checks word/char count, sentence count, pronouns (1st/2nd person), hedge words, apologetic language, confident language, yielding words, punctuation (questions, exclamations, ellipsis, dashes), and opening phrase patterns.

##### Global Findings (all 180 pairs)

| Metric | Positive avg | Negative avg | Diff | Ratio | Flagged? |
|--------|-------------|-------------|------|-------|----------|
| Words | 24.76 | 13.72 | +11.03 | 1.80x | **YES** |
| Characters | 135.03 | 70.04 | +64.99 | 1.93x | **YES** |
| Sentences | 1.90 | 1.91 | -0.01 | 0.99x | No |
| First-person pronouns | 2.97 | 2.05 | +0.92 | 1.45x | **YES** |
| Second-person pronouns | 1.10 | 0.54 | +0.56 | 2.02x | **YES** |
| Hedge words | 0.13 | 0.46 | -0.33 | 0.28x | **YES** |
| Yielding words | 0.09 | 0.46 | -0.37 | 0.19x | **YES** |
| Ellipsis (...) total | 0 | 29 | -29 | 0.00x | **YES** |

##### Five Critical Surface Biases

1. **Length bias (1.8x):** Positive responses average 25 words, negatives average 14. Steering vectors will partially encode "be more verbose." Worst: Value Creation (2.87x chars), Interest-Based Reasoning (2.77x), Active Listening (2.75x). Least biased: Assertiveness (1.07x chars).

2. **Zero overlap in opening phrases:** Top-20 positive openers and top-20 negative openers share **zero** phrases. Positives cluster around "I want to" (11x), "What if we" (6x), "I'd like to" (4x). Negatives cluster around "Okay" (19x as opener), "Sure" (7x), "Fine" (5x), "Oh" (4x). A trivial first-word classifier would separate classes with high accuracy.

3. **Ellipsis is a perfect separator:** 0 ellipses across all 180 positive responses. 29 ellipses across negative responses. A vector that detects trailing-off speech patterns ("I'm not sure...") perfectly separates the classes.

4. **Capitulation word segregation:** "Okay" opens 0/180 positives vs 19/180 negatives. "Sure" 0 vs 7. "Fine" 0 vs 5. "Oh" 0 vs 4. Combined: these capitulation starters appear in 35/180 negatives and 0/180 positives. The vector likely encodes "don't start with capitulation words."

5. **Hedge/yielding word clustering:** Hedges are 3.6x more common in negatives (0.46 vs 0.13). Yielding words are 5.1x more common (0.46 vs 0.09). These are shallow lexical features, not deep behavioral concepts.

##### Per-Dimension Length Bias (ranked worst to least)

| Dimension | Char ratio (pos/neg) | Worst bias type |
|-----------|---------------------|-----------------|
| Value Creation | 2.87x | Length + questions (4.5x) |
| Interest-Based Reasoning | 2.77x | Length + second-person (∞) |
| Active Listening | 2.75x | Length + second-person (3x) |
| Reframing | 2.53x | Length + questions (∞) |
| Empathy | 2.35x | Length + first-person (3.3x) |
| Information Gathering | 2.33x | Length + questions (3.2x) |
| Rapport Building | 2.14x | Length + first-person (2.4x) |
| Strategic Concession | 2.06x | Length + yielding (∞) |
| BATNA Awareness | 1.77x | Length + first-person (1.6x) |
| Emotional Regulation | 1.56x | Length + first-person (1.9x) |
| Firmness | 1.64x | Hedges (0.09x) + yielding (0.06x) |
| Anchoring | 1.46x | Hedges (0x) + questions (0x) |
| Patience | 1.24x | Hedges (0x) + yielding (0.08x) |
| Clarity & Directness | 1.53x | Hedges (0x) + confident (∞) |
| Assertiveness | **1.07x** ← least | Hedges (0x) + ellipsis (0x) |

##### Implications for Our Best Result (+37% from strategic_concession_making)

The `strategic_concession_making` dimension has a 2.06x character ratio between positives and negatives. Its positive examples cluster around "I can [verb]" openers (e.g., "I can extend", "I can move", "I can accelerate"), while negatives cluster around "Sure, we can" and "Okay, I can". The steering vector for this dimension likely encodes a mix of:
- "Be longer" (length bias)
- "Start with 'I can [condition]' rather than 'Sure/Okay'" (opener pattern)
- "Don't use yielding words" (yielding bias: 0 vs 0.92)

Whether this accidentally-correct surface pattern leads to genuinely better negotiation outcomes (a happy coincidence) or whether the outcome improvement is also an artifact of these surface patterns is an open question. The Phase A finding that steering makes responses 22% *shorter* in actual gameplay (despite positives being 2x longer in training) is interesting — it suggests the vector isn't simply encoding "be verbose" at deployment time, even if it partly encodes length during extraction.

##### Recommendations for P1 (flagged)

1. **Match response lengths** — positive and negative responses for each pair should be within ±20% word count
2. **Vary opening phrases** — introduce "I want to..." in some negatives and "Okay" in some positives
3. **Add ellipsis to some positives** — break the perfect separation
4. **Cross-contaminate yielding words** — some positives should concede on minor points using "sure" or "okay"
5. **Each dimension should be audited independently** — Assertiveness is well-constructed (1.07x length ratio); Value Creation is poorly constructed (2.87x)

#### LLM Judge Design (Phase C1) — IN PROGRESS

*(To be completed this session)*

### Phase B: Richer Metrics (P4 core task)

#### B1. Add to run_game() / score_deal()
- **Offer trajectory:** Extract all numeric offers from each turn per agent
- **Concession rate:** How much does each agent move per turn (in $ and as % of remaining gap)
- **First-offer distance:** How far is each agent's first offer from their private target
- **Hedge word count:** Count hedging language ("maybe", "perhaps", "I could consider") per turn
- **Turns to deal:** Already tracked, but break down by who finalizes
- **Response length per turn:** Token count, to check for verbosity confound

#### B2. Per-turn steering decay check
- Plot metric values by turn number across all games
- Check if steered agent's firmness/concession rate decays in later turns (literature says steering fades after 300-500 tokens)

#### B3. Role-separated analysis
- All metrics broken down by steered-as-seller vs steered-as-buyer
- Known asymmetry: buyers have structural advantage (Xia et al. 2024). Report separately.

### Phase C: LLM Judge (llm_judge.py)

#### C1. Design judge prompts
- Rate each transcript on: firmness, persuasiveness, naturalness, coherence
- 1-5 Likert scale with rubric definitions
- Judge both the steered and baseline agent in each game

#### C2. Implementation
- Input: results JSON with transcripts
- API: Groq (LLaMA 70B) or similar
- Output: per-game judge scores appended to results
- Include inter-rater reliability check: run judge twice, measure agreement

#### C3. Validate judge against simple metrics
- Does judge's "firmness" rating correlate with concession rate?
- Does judge's "naturalness" rating correlate inversely with alpha?
- If judge scores don't correlate with anything behavioral, the judge prompt needs work

### Phase D: Paper Sections

#### D1. Evaluation Framework section
- Describe all metrics, justify each one
- Describe LLM judge methodology
- Describe role-separated analysis and why it matters

#### D2. Results & Analysis section
- Present all results with proper breakdowns
- Statistical tests (paired t-test or Wilcoxon on per-game advantages)
- Discussion of what steering is actually doing to negotiation behavior

---

## Key Literature for P4

| Paper | Why it matters |
|-------|---------------|
| Panickssery et al. 2024 (CAA) | Large steering magnitudes degrade text quality. Must measure coherence. |
| Hao et al. 2025 (ICLR WS) | ~80 pairs for diminishing returns. CAE only reliable in-distribution. Steering harms perplexity. |
| **Tan et al. 2024 (NeurIPS)** | **Anti-steerability: ~50% of inputs steer opposite direction. Steerability bias: vectors encode surface patterns. Steerability is a dataset property. Essential for understanding our variance.** |
| Xia et al. 2024 (Bargaining benchmark) | Buyer is harder than Seller. Formalize bargaining as asymmetric game. |
| Wang et al. 2025 (Multi-turn bargain) | Turn-level evaluation grounded in Theory of Mind, not just outcome. |
| He et al. 2018 (CraigslistBargains) | Dataset paper. Known data quirks with crossed targets. |
| Lewis et al. 2017 (Deal or No Deal) | Multi-issue negotiation with known value functions. Validation dataset for strategic concession claims. |
| Chandra et al. 2021 (CaSiNo) | Strategy-annotated negotiation dialogues. Use to validate LLM judge. |
| "Sober Look at Steering Vectors" (AF) | Steering degrades performance equivalent to halving pre-training compute. |
| Practitioner's Field Guide 2026 | Steering fades after 300-500 tokens. Multi-vector stacking is fragile. |
| HAMBA metric (2025) | Consumer Surplus + Negotiation Power + Acquisition Ratio. More principled than simple price split. |

---

## Decisions Made

1. **We are P4 (Metrics & Evaluation).** Not P1-P3 or P5-P6.
2. **Lightweight-first strategy:** Validate the pipeline before building sophisticated metrics. No point measuring something broken.
3. **LLM judge via API (Groq).** Steering requires local model access; evaluation does not.
4. **Role-separated reporting is mandatory.** Never report aggregate scores without buyer/seller breakdown. **Confirmed by A2: seller=-27% vs buyer=+24%, aggregate=-1.6% hides everything.**
5. **Coherence is a first-class metric.** An agent that "wins" by producing degenerate text is not a result.
6. **Clamped games must be flagged.** 24% of firmness results have scores clamped to (0,1) due to agreed price below buyer target. These need a `clamped` flag and separate reporting.
7. **Dealmaker asymmetry must be investigated.** Seller finalizes 100% of deals in current results. This is either a game design artifact or a real model behavior — either way it affects interpretation.
8. **Anti-steerability is real, not hypothetical.** 50/48% help/hurt split in firmness results. Must be front-and-center in reporting, not buried in an appendix.
9. **Contrastive pairs have severe surface biases.** Every dimension has at least one confound (length, opener segregation, hedge clustering). Paper must acknowledge this as a limitation. P1 should address before final experiments. LLM judge should be designed to detect whether steered outputs mimic surface patterns from positive training examples.

## Open Questions

- Should we push to use two separate model instances (steered + unsteered) to eliminate context contamination? Or flag it as limitation?
- What alpha values should the final evaluation sweep cover? (Current best: ~6 for strategic_concession_making)
- Should we evaluate on validation split (untouched) rather than train split (used in search)?
- How many games do we need for statistical significance? Current runs use 20-50, which is thin.
- Is `strategic_concession_making` actually doing something different from firmness/anchoring, or is it just a relabeled version of the same direction? Check cosine similarity between these vectors.
- ~~Should we integrate Deal or No Deal ourselves (moderate effort) or propose it to the team as a shared task?~~ **DONE — integrated. See session 4 log.**

---

## Log

### 2026-02-27 (session 4) — Deal or No Deal cross-dataset integration

Created `deal_or_no_deal.py` — complete integration of the Deal or No Deal dataset (Lewis et al. 2017, Facebook AI Research) for cross-dataset validation.

#### What was built

| Component | Description |
|-----------|-------------|
| `load_dealornodeal(split, num_samples)` | Downloads and parses DonD scenarios from archived GitHub repo. Supports "selfplay" (4086 scenarios), "train" (3693 unique), "val", "test" splits. Caches locally to `data/dealornodeal/`. |
| `score_deal_dond(picks_1, picks_2, scenario)` | Computes each agent's utility (0-10), normalised scores, joint utility, efficiency ratio, and Pareto optimality. |
| `is_pareto_optimal(counts, values_1, picks_1, values_2, picks_2)` | Brute-force enumeration of all valid allocations (≤120 per scenario). Checks if any alternative Pareto-dominates the current deal. |
| `max_joint_utility(counts, values_1, values_2)` | Computes theoretical max joint utility for computing efficiency ratio. |
| `run_game_dond(model, tokenizer, scenario, dvecs, alphas, ...)` | Full game loop for multi-issue negotiation. Agent 1 proposes first, agents alternate, DEAL=X,Y,Z format for allocations. Reuses `generate_turn()` from `apply_steering.py` including steering hooks. |
| `summarise_dond(results, alpha)` | Summary stats parallel to `summarise()` in apply_steering: advantage, Pareto rate, efficiency, agreement rate, turns. |
| `_run_self_test()` | Standalone validation: downloads data, validates scoring against paper examples, tests Pareto computation, parse functions, computes baseline stats. |

#### Design decisions

1. **Separate game loop, not a shoehorn.** DonD is multi-issue (split 3 item types) vs Craigslist (single price). Separate `run_game_dond` is cleaner than forcing both through one function. Shared `generate_turn()` (with its steering hooks) is reused as-is.
2. **DEAL=X,Y,Z format.** Agents write `DEAL=2,1,0` (what they take, remainder goes to other). Parallel to Craigslist's `DEAL=$500`. The existing `generate_turn` regex `DEAL\s*=\s*\$?[\d,]+` coincidentally handles both formats.
3. **Lazy import of `generate_turn`.** Inside `run_game_dond`, not at module level. This lets the data loading and scoring functions work without GPU infrastructure (standalone testing).
4. **selfplay split as default.** 4086 exhaustive scenario pairs with no historical dialogues — ideal for running fresh LLM negotiations.
5. **Cached downloads.** Dataset files cached to `data/dealornodeal/` on first download.

#### Dataset stats

| Stat | Value |
|------|-------|
| Selfplay scenarios | 4,086 |
| Unique train scenarios | 3,693 |
| Items per scenario | 5-7 (mean 5.5) |
| Unique count configs | 20 |
| Most common config | 1 book, 1 hat, 3 balls (455 scenarios) |
| Max joint utility range | 10-19 across scenarios |
| Naive fair-split Pareto rate | 9.0% |
| Naive fair-split efficiency | 0.694 |
| Human baseline Pareto rate (paper) | 76.9% |

The gap between naive (9%) and human (76.9%) Pareto rates means there is substantial room for steering to show improvement — or failure.

#### Scoring validation (paper example)

```
Scenario: 3 books (A1:1pt, A2:2pt), 2 hats (A1:3pt, A2:1pt), 1 ball (A1:1pt, A2:2pt)
Paper deal: Agent1 takes [2,2,0]=8pts, Agent2 takes [1,0,1]=4pts
  Joint utility: 12/14, Efficiency: 0.857, Pareto optimal: True ✓
Optimal deal: Agent1 takes [0,2,0]=6pts, Agent2 takes [3,0,1]=8pts
  Joint utility: 14/14, Efficiency: 1.000, Pareto optimal: True ✓
Non-Pareto deal: Agent1 takes [1,0,0]=1pt, Agent2 takes [2,2,1]=8pts
  Dominated by (6,8): Pareto optimal: False ✓
```

#### Why this matters for the paper claim

The +37% from `strategic_concession_making` on CraigslistBargains is ambiguous because Craigslist is single-issue. "Strategic concession" means trading low-value items for high-value items — impossible when the only variable is price. On Deal or No Deal:

- If the vector improves **Pareto efficiency** (agents find better trades), it genuinely encodes multi-issue strategic reasoning.
- If it only improves the **steered agent's raw score** without improving joint efficiency, it's just firmness/anchoring under a different label.
- If it has **no effect** on DonD, the Craigslist result is likely domain-specific stubbornness.

This is the strongest falsifiability test for the paper's core claim.

#### How to run with steering

```python
from deal_or_no_deal import load_dealornodeal, run_game_dond, summarise_dond

scenarios = load_dealornodeal(split="selfplay", num_samples=30)
results = []
for i, sc in enumerate(scenarios):
    steered_role = "agent1" if i % 2 == 0 else "agent2"
    dvecs_a1 = dvecs if steered_role == "agent1" else None
    alpha_a1 = alpha if steered_role == "agent1" else 0.0
    dvecs_a2 = dvecs if steered_role == "agent2" else None
    alpha_a2 = alpha if steered_role == "agent2" else 0.0
    result = run_game_dond(model, tokenizer, sc,
                           dvecs_a1, alpha_a1, dvecs_a2, alpha_a2,
                           steered_role=steered_role)
    results.append(result)
summary = summarise_dond(results, alpha)
```

#### Known limitations

- **DEAL format parsing is strict.** Model must write `DEAL=X,Y,Z` (comma-separated integers). If it generates natural language like "I'll take 2 books and a hat", the deal isn't parsed. The prompt is explicit, but lower-quality models may not follow the format.
- **No two-phase commit.** In the original Lewis et al. protocol, both agents independently report the deal and it only counts if reports match. Our protocol is simpler: the agent who writes DEAL= claims items unilaterally. This favours the dealmaker (they can grab all high-value items). Mitigated by the same agent alternating roles across games.
- **Steering vector domain mismatch.** Vectors were extracted from Craigslist-style negotiation pairs. Applying them to multi-issue DonD scenarios is a transfer test — if they work, it's evidence of generalisation; if they don't, it could be domain mismatch rather than vector failure.

### 2026-02-27 (session 3) — Contrastive pairs audit + LLM judge design
- Created and ran `audit_pairs.py` — systematic surface bias audit of all 180 contrastive pairs
- **Issue 7 CONFIRMED — SEVERE:** 6 flagged biases across all pairs:
  1. Length: positives 1.8x longer than negatives (worst: Value Creation at 2.87x)
  2. Opening phrases: zero overlap between top-20 positive and negative openers
  3. Ellipsis: perfect separator (0 in positives, 29 in negatives)
  4. Capitulation words: "Okay/Sure/Fine/Oh" open 35 negatives, 0 positives
  5. Hedge words: 3.6x more common in negatives
  6. Yielding words: 5.1x more common in negatives
- Flagged recommendations for P1 (pair authors): match lengths, vary openers, cross-contaminate surface features
- Noted that strategic_concession_making (our best result) has 2.06x length ratio and "I can [verb]" vs "Sure/Okay" opener segregation
- Key tension: Phase A found steering makes responses 22% *shorter* at deployment, contradicting the length-bias-in-training hypothesis. The vector may learn length patterns during extraction but express differently during generation.

### 2026-02-27 (session 2) — Phase A completed
- Created and ran `phase_a_diagnostic.py` — comprehensive data audit + results diagnostic
- **A1 Data Audit:** CraigslistBargains is 100% clean. Zero overlapping targets, zero bad prices. seller_target == listing_price always. buyer_target ≈ 72% of listing price. 6 categories, dominated by furniture (25%) and housing (20%). Closed Issue 2 as non-applicable.
- **A2 Results Diagnostic — 7 key findings:**
  1. Massive role asymmetry: steered seller -27% vs steered buyer +24% (aggregate -1.6% hides everything)
  2. Seller finalizes 100% of deals — structural game asymmetry
  3. Steering makes responses 22% shorter, shorter correlates with better outcomes (r=-0.56) — rules out verbosity-wins hypothesis
  4. Anti-steerability confirmed: 50% help / 48% hurt, bimodal distribution, std=0.682
  5. Score clamping affects 24% of games (agreed price below buyer_target)
  6. Category effects exist but N too small to be reliable
  7. Fast search: anchoring collapsed 97% from S2→S3 (overfitting). strategic_concession_making dropped 29% but held at +37%
- **A3 score_deal():** No overlapping targets to fix. Real issue is clamping — 24% of games have raw scores well outside [0,1] that get masked.
- Updated Issue 2 (closed), Issue 3 (partial), Issue 6 (confirmed)
- Added decisions 6-8 (clamped games, dealmaker asymmetry, anti-steerability)

### 2026-02-27 (session 1)
- Reviewed full codebase (extract_vectors.py, apply_steering.py, fast_search_steering.py)
- Reviewed Damon's new search results on origin/damon branch
- Identified 5 critical issues with current evaluation pipeline
- Created this tracking document
- Literature review: 12 papers identified as relevant
- Decision: lightweight validation pass before building P4 metrics
- Added 3 more critical issues from Tan et al. (NeurIPS 2024) review: anti-steerability, steerability bias, CraigslistBargains dimensionality limitation
- Added dataset strategy: keep Craigslist as primary, add Deal or No Deal for validation, use CaSiNo for judge validation
- Total critical issues: 8 (Issues 1-5 pipeline/methodology, Issues 6-8 from literature)
