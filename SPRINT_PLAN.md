# Sprint Plan: Final Ultimatum Game Experiments

**Branch:** `moiz-task-design`
**Date:** 2026-03-27
**Deadline:** 2026-04-17 (paper due)
**GPU:** g5.xlarge (A10G 24GB), ~$1/hr, eu-west-2

---

## Goal

Run a clean, pre-registered confirmatory experiment showing:
1. Activation steering reliably shifts LLM negotiation behavior (demand shift)
2. Behavioral change does not equal strategic improvement (firmness: demand up, payoff flat/down)
3. One exception: empathy steering improves expected payoff (demands less, rejected less)
4. Layer location determines steering quality (L10 = crude amplification, L12 = strategic)
5. Firmness is blind behavioral amplification, not context-sensitive strategy (UG vs DG contrast)

---

## Critical Design Decisions (locked in)

| Decision | Choice | Justification |
|----------|--------|---------------|
| **Steering pairs** | General-domain only (`ultimatum_10dim_20pairs_general_matched`) | Game-specific pairs hold offers constant between pos/neg, so vectors learn tone-not-action. General pairs capture abstract behavioral concepts that transfer to gameplay. Validated: 8/10 dims GENUINE via orthogonal projection. |
| **Opponent** | LLM responder (unsteered Qwen 7B) | Rule-based opponent (threshold=0.35) destroys the acceptance gradient. LLM responder has graded sensitivity to offer fairness, which is needed to show empathy's strategic advantage. Also supports "negotiation" framing in paper. |
| **Pools** | Variable ($37-$157, 24 prime-heavy sizes from POOL_SIZES) | Fixed $100 produces 2 unique baseline texts across 50 games. Variable pools force per-game computation. Essential for any statistical validity. |
| **Pairing** | True pairing (steered + baseline on same pool, same game dict) | Teammate's design used separate files — paired t-tests on unpaired data inflates Type I error. Must share pool and seed. |
| **Temperature** | 0.0 primary, 0.3 robustness check | Variable pools provide the variance. Temperature adds noise and baseline drift (66.6/65.4/63.0 across teammate runs). Temp=0 is deterministic and reproducible. One config at temp=0.3 as robustness check. |
| **Extraction method** | Mean difference | MD >= LR >> PCA (Im & Li 2025). Confirmed across all our variants. |
| **Dimensions** | Empathy (expected positive), Firmness (expected behavioral-only) | Pre-specified from exploratory data. Empathy is the only dim that showed payoff improvement. Firmness is the strongest behavioral effect and the contrast case. |
| **Layers** | L10, L12 | L10 = crude amplification (style/tone), L12 = strategic reasoning preserved. Replicated across 3 experimental batches. L14+ = dead (near-zero effect). |
| **Alphas** | {3, 7, 10} | Three points for dose-response. Avoids alpha=15 (model-breaking: 95/5 every game at L10). Avoids alpha=5 (too close to 3 and 7). |
| **Sample size** | n=150 per config | d=0.3 effect (empathy payoff) needs n=150 to survive BH-FDR correction over 5 primary hypotheses at ~80% power. |

---

## Pre-registered Hypotheses

| ID | Hypothesis | Test | Direction |
|----|-----------|------|-----------|
| H1 | Empathy shifts proposer demand downward | Paired t-test (one-sided) | demand_steered < demand_baseline |
| H2 | Firmness shifts proposer demand upward | Paired t-test (one-sided) | demand_steered > demand_baseline |
| H3 | Empathy improves proposer expected payoff | Paired t-test (one-sided) | payoff_steered > payoff_baseline |
| H4 | Firmness does NOT improve proposer payoff | TOST equivalence test | |payoff_delta| < epsilon |
| H5 | Firmness demand shift is equal in UG and DG | TOST equivalence test | |UG_shift - DG_shift| < epsilon |

Correction: BH-FDR over H1-H5. Dose-response and layer effects are secondary/descriptive.

---

## Task Breakdown

### Phase -1: Prompt Sanity Check (Day 0, local, no GPU)

A zero-cost API check before writing any code. Uses the existing `playground/run_ultimatum.py` (already has variable pools, prompt enhancements for firmness/empathy, and API support via Gemini/Groq). Takes ~10 min and validates the task design itself.

#### T-1.1 — Baseline UG: variable pools, no enhancement, 20 games
- **What:** `python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini --n_games 20`
- **Why:** Confirms variable pools produce offer variance (not the fixed-60/40 problem). Establishes baseline offer distribution and acceptance rate.
- **Depends on:** Nothing
- **Time:** 2 min
- **Parallel:** Yes

#### T-1.2 — Firmness UG: prompt-steered proposer, 20 games
- **What:** `python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini --proposer_enhancement firmness --n_games 20`
- **Why:** Does a firmness prompt shift demand upward? If prompt can't do it, steering likely can't either (at least not in a strategically meaningful way).
- **Depends on:** Nothing
- **Time:** 2 min
- **Parallel:** Yes, parallel with T-1.1, T-1.3

#### T-1.3 — Empathy UG: prompt-steered proposer, 20 games
- **What:** `python playground/run_ultimatum.py --proposer api:gemini --responder api:gemini --proposer_enhancement empathy --n_games 20`
- **Why:** Does empathy prompt shift demand down AND improve payoff (accepted more)? This is the key positive-result hypothesis.
- **Depends on:** Nothing
- **Time:** 2 min
- **Parallel:** Yes, parallel with T-1.1, T-1.2

#### T-1.4 — Firmness DG: prompt-steered proposer, 10 games
- **What:** Requires adding a `--game dictator` flag to `run_ultimatum.py` (small patch: skip responder call, always accept). Then run with firmness enhancement.
- **Why:** Does firmness demand the same in DG as UG? If yes → blind behavioral amplification works even at the prompt level. If demand is higher in DG → the model IS context-sensitive, and the DG contrast will work for steering too.
- **Depends on:** Small code patch (~10 lines)
- **Time:** 5 min (patch + run)
- **Parallel:** After quick patch, parallel with others

**DECISION GATE after T-1:**
- If baseline has <5 unique offer levels across 20 variable-pool games → problem is the model/API, not our infrastructure. Try a different model (Groq LLaMA 8B).
- If firmness prompt doesn't shift demand → the UG task may not discriminate firmness. Consider whether steering can do what prompting can't (possible if steering acts on a different mechanism).
- If empathy doesn't improve payoff at prompt level → the positive-result hypothesis is weaker. Still worth testing with steering (different mechanism), but temper expectations.
- If DG and UG produce identical firmness demands → the DG contrast experiment will likely show the same for steering (blind amplification). Good — confirms the experiment is worth running.
- If DG demand >> UG demand at prompt level → model already has strategic awareness. The interesting question becomes whether steering *preserves* that awareness (L12) or *destroys* it (L10).

---

### Phase 0: Infrastructure & Quick Validation (Day 1, local + GPU)

These tasks fix the codebase and run cheap sanity checks before committing GPU hours to the main experiment.

#### T0.1 — Fix `apply_steering_ultimatum.py` to use variable pools
- **What:** Patch the default pool assignment to use `POOL_SIZES` instead of fixed 100. Ensure pools cycle deterministically from a seed.
- **Why:** Fixed $100 = 2 unique baseline texts. This is the #1 blocker.
- **Depends on:** Nothing
- **Time:** 30 min (local)
- **Parallel:** Yes, independent of T0.2-T0.4

#### T0.2 — Implement true pairing in game runner
- **What:** Modify game runner so steered and baseline share the same pool AND the same LLM responder offer (for responder mode). Output both in a single game dict (like Moiz's original `ultimatum_game.py` design). Baseline run uses alpha=0 with vectors loaded (not a separate no-steer run).
- **Why:** Paired t-tests on unpaired data = inflated Type I error. Current separate-files design is invalid.
- **Depends on:** Nothing
- **Time:** 1-2 hrs (local)
- **Parallel:** Yes, independent of T0.1, T0.3-T0.4

#### T0.3 — Add Dictator Game mode
- **What:** Add `--game {ultimatum,dictator}` flag. In DG mode, proposer makes an offer, responder always accepts (no LLM call needed). Record the same metrics. Minimal code change — skip responder generation, set `agreed=True` always.
- **Why:** DG is the within-dimension control for "blind amplification vs strategic awareness." Methodologically essential per council. Zero extra responder GPU cost.
- **Depends on:** T0.1 (uses same pool infrastructure)
- **Time:** 30 min (local)
- **Parallel:** Can start after T0.1, parallel with T0.2

#### T0.4 — Add analysis script for the confirmatory experiment
- **What:** Script that reads game results and computes: (a) demand shift (paired t-test), (b) payoff delta (paired t-test), (c) acceptance rate shift (McNemar's), (d) BH-FDR over H1-H5, (e) dose-response plots, (f) UG vs DG comparison for H5. Also computes responder acceptance curve from pre-registration data.
- **Why:** Need this before running experiments so we can analyze incrementally.
- **Depends on:** T0.2 (needs to know output format)
- **Time:** 2-3 hrs (local)
- **Parallel:** Can start in parallel with T0.1-T0.3, finish after T0.2

#### T0.5 — Quick smoke test (local or GPU, 10 games)
- **What:** Run 10 games each for firmness L10 alpha=7 and empathy L12 alpha=7, variable pools, temp=0, LLM responder. Check: (a) variable offers? (b) no parse errors? (c) reasonable text output? (d) pairing works?
- **Why:** Catches integration bugs before committing to 3,600+ games. Also reveals if temp=0 + variable pools produces enough variance.
- **Depends on:** T0.1, T0.2
- **Time:** ~15 min on GPU
- **Parallel:** No — gate for Phase 1

**DECISION GATE after T0.5:** Review smoke test outputs manually. If baseline still produces <5 unique offers across 10 variable-pool games at temp=0, switch to temp=0.3 with fixed seeds. If parse error rate >10%, debug prompts before proceeding.

---

### Phase 1: Pre-Registration Characterization (Day 1-2, GPU)

These runs characterize the baseline and opponent before any steered experiments. Results determine whether the main experiment is viable and set the interpretive anchor.

#### T1.1 — Characterize LLM responder acceptance curve
- **What:** Run 200+ games where we systematically vary the proposer's offer from 50/50 to 99/1 (in steps of ~5% of pool), across all 24 pool sizes, temp=0. No steering. Record P(accept) vs proposer_share/pool.
- **Why:** Without this, "improves payoff" is uninterpretable. We need to know the optimal offer (the point where P(accept) starts dropping). This is our interpretive anchor: "the unsteered model demands X, the optimum is Y, empathy-steered demands Z which is closer to Y."
- **Depends on:** T0.1, T0.2
- **Time:** ~30 min GPU (200 single-turn games, only responder generation needed)
- **Parallel:** Yes, independent of T1.2

#### T1.2 — Characterize baseline proposer
- **What:** Run 150 games with unsteered proposer, LLM responder, variable pools, temp=0. Record: distribution of offers by pool size, acceptance rate, mean payoff.
- **Why:** This is the pre-registered baseline. All steered conditions compare against this. Eliminates the drifting-baselines problem (66.6/65.4/63.0).
- **Depends on:** T0.1, T0.2
- **Time:** ~20 min GPU
- **Parallel:** Yes, independent of T1.1

**DECISION GATE after T1.1 + T1.2:**
- Plot the acceptance curve. If it's a step function (e.g., 100% accept above 40%, 0% below), the LLM responder is equivalent to the rule-based opponent and the "graded sensitivity" argument fails. In that case: either (a) add noise to the responder via temp=0.3 for the responder only, or (b) accept the step function and reframe accordingly.
- Check baseline offer distribution. If temp=0 + variable pools still produces <5 unique offer levels, temperature must increase. We'd switch to temp=0.3 with per-game seeds.
- Compute the theoretically optimal offer from the acceptance curve. This becomes the benchmark for H3/H4.

---

### Phase 2: Quick Exploratory Sweep (Day 2, GPU)

A cheap sweep to verify that steering effects exist with the new infrastructure before committing to n=150. This catches problems early.

#### T2.1 — Mini sweep: 4 key configs, n=30 each
- **What:** Run 30 paired games for each of:
  1. Firmness, L10, alpha=7, UG
  2. Firmness, L12, alpha=7, UG
  3. Empathy, L10, alpha=7, UG
  4. Empathy, L12, alpha=7, UG
- **Why:** 30 games is enough to detect d>0.5 effects (which firmness demand shift should be). If we see zero effect on demand for firmness, something is broken. If empathy shows no directional shift, the confirmatory experiment is unlikely to succeed at n=150.
- **Depends on:** T1.1, T1.2 (need baseline characterization first)
- **Time:** ~30 min GPU (4 × 30 = 120 games)
- **Parallel:** All 4 configs can run sequentially in one script (single model load)

**DECISION GATE after T2.1:**
- **Firmness demand shift at L10:** If d < 0.3 → something is wrong with infrastructure. Debug before proceeding. Expected: d > 0.5.
- **Empathy demand shift at L12:** If effect is in the wrong direction (demand UP instead of down) → empathy may not replicate. Consider swapping to a different dimension or adjusting alpha range.
- **L10 vs L12 contrast:** If both layers show identical effects → the layer-location story doesn't hold with general pairs. May need to add L14 to test.
- **Payoff direction for empathy:** Even at n=30, check if the sign is positive. If negative → the positive-payoff story is dead. Reframe paper as pure negative result.
- If all looks good → proceed to Phase 3.

#### T2.2 — Mini DG sweep: firmness only, n=30
- **What:** Run 30 paired Dictator Game games for firmness L10 alpha=7.
- **Why:** Quick check that DG produces different behavior than UG. If firmness demands the same in both → confirms blind amplification. If identical → still a finding. If the model is confused by DG framing → fix prompts.
- **Depends on:** T0.3, T2.1 (run after UG to compare)
- **Time:** ~10 min GPU (no responder generation)
- **Parallel:** Can run immediately after T2.1 (same model loaded)

**DECISION GATE after T2.2:**
- If DG demand ≈ UG demand for firmness → blind amplification confirmed. Proceed with full DG in Phase 3.
- If DG demand >> UG demand → surprising. The vector HAS game-awareness. This changes the paper narrative (in an interesting way). Proceed but adjust hypotheses.
- If DG outputs are broken (parse errors, confused model) → fix DG prompt or drop DG from the final experiment.

---

### Phase 3: Main Confirmatory Experiment (Day 2-4, GPU)

The full experiment. Only run this after Phases 0-2 confirm the infrastructure works and effects exist.

#### T3.1 — Full UG experiment: 2 dims × 2 layers × 3 alphas × n=150
- **What:** Run the confirmatory experiment:
  - Empathy: L10 × {3, 7, 10} + L12 × {3, 7, 10} = 6 configs
  - Firmness: L10 × {3, 7, 10} + L12 × {3, 7, 10} = 6 configs
  - Each config: 150 paired games, variable pools, temp=0, LLM responder
  - Total: 12 configs × 150 = 1,800 paired games
- **Why:** This is the pre-registered confirmatory experiment testing H1-H4.
- **Depends on:** Phase 2 decision gates passed
- **Time:** ~3-4 hrs GPU (each config ~15-20 min with LLM responder at 150 games)
- **Parallel:** Configs must run sequentially (single model load), but can batch overnight via nohup.

#### T3.2 — Full DG experiment: firmness only, 2 layers × 3 alphas × n=150
- **What:** Dictator Game for firmness:
  - Firmness: L10 × {3, 7, 10} + L12 × {3, 7, 10} = 6 configs
  - Each config: 150 games, variable pools, temp=0, deterministic acceptance
  - Total: 6 × 150 = 900 games
- **Why:** Tests H5 (blind amplification). No responder LLM calls needed → much faster.
- **Depends on:** T0.3 (DG mode), Phase 2 decision gate
- **Time:** ~1 hr GPU (no responder generation, ~6 min per config)
- **Parallel:** Can run after T3.1, or interleaved. Could also run in parallel on a second instance if available.

#### T3.3 — Temperature robustness check
- **What:** Re-run one config (empathy L12 alpha=7 UG) at temp=0.3 with fixed seeds, n=150.
- **Why:** Reviewer defense for the temp=0 choice. Shows results are qualitatively similar.
- **Depends on:** T3.1
- **Time:** ~20 min GPU
- **Parallel:** Run after main experiment or interleaved

**Total GPU for Phase 3:** ~5-6 hrs. Can run overnight in one session.

---

### Phase 4: Analysis & Paper (Day 4-7+, local)

#### T4.1 — Run confirmatory analysis
- **What:** Run the analysis script from T0.4 on all Phase 3 results. Compute H1-H5 with BH-FDR correction. Generate dose-response plots, UG vs DG comparison, acceptance curve overlay.
- **Depends on:** T3.1, T3.2, T0.4
- **Time:** 1-2 hrs (local)
- **Parallel:** Start as soon as first results come in

#### T4.2 — Behavioral text analysis
- **What:** Extract from proposer/responder text: word count, hedge rate, fairness language frequency, reasoning patterns. Compare steered vs baseline. This is the "behavioral change is real" evidence.
- **Depends on:** T3.1
- **Time:** 2-3 hrs (local)
- **Parallel:** Yes, parallel with T4.1

#### T4.3 — Compile exploratory results from teammate data
- **What:** Re-analyze ALL existing results (temp7, temp03_mindims_v4, abdullah_general_pairs, damon_12_16_19) with proper correction. These form the exploratory phase of the paper. Report with full BH-FDR over ~130 configs. Expected result: most payoff effects null after correction, demand effects survive.
- **Why:** The paper needs both phases: exploratory (broad sweep, corrected to null on payoff) → confirmatory (targeted experiment on empathy/firmness).
- **Depends on:** Nothing (existing data)
- **Time:** 3-4 hrs (local)
- **Parallel:** Can start immediately, parallel with everything

#### T4.4 — Write results section
- **Depends on:** T4.1, T4.2, T4.3
- **Time:** 2-3 days
- **Parallel:** No — needs all analysis complete

#### T4.5 — Write methods + discussion
- **Depends on:** T4.4 (methods can start earlier)
- **Time:** 2-3 days
- **Parallel:** Methods can be written in parallel with T4.1-T4.3

---

## Task Dependency Graph

```
T-1.1 (baseline UG prompt) ─┐
T-1.2 (firmness UG prompt) ─┼──→ GATE-1 (does task design work?)
T-1.3 (empathy UG prompt)  ─┤         │
T-1.4 (firmness DG prompt) ─┘         ▼

T0.1 (variable pools) ──┬──→ T0.3 (DG mode) ──→ T2.2 (mini DG) ──→ T3.2 (full DG)
                         │
T0.2 (true pairing) ────┼──→ T0.5 (smoke test) ──→ T1.1 (acceptance curve) ──┐
                         │                          T1.2 (baseline char)  ────┤
T0.4 (analysis script) ─┘                                                    │
                                                     GATE ←──────────────────┘
                                                       │
                                                       ▼
                                                    T2.1 (mini UG sweep) ──→ GATE
                                                       │
                                                       ▼
                                                    T3.1 (full UG) ──→ T3.3 (temp robustness)
                                                    T3.2 (full DG)
                                                       │
                                                       ▼
                                              T4.1 (confirmatory analysis)
                                              T4.2 (behavioral text analysis)
T4.3 (compile exploratory) ← can start Day 0        │
                                                     ▼
                                              T4.4 (results) → T4.5 (methods/discussion)
```

**Parallelism summary:**
- **Day 0 (local, no GPU):** T-1.1 + T-1.2 + T-1.3 + T-1.4 in parallel (API calls only, ~10 min total). Review results. T4.3 can also start.
- **Day 1 (local):** T0.1 + T0.2 + T0.4 in parallel. T0.3 after T0.1.
- **Day 1 (GPU):** T0.5 smoke test after T0.1+T0.2.
- **Day 1-2 (GPU):** T1.1 + T1.2 in parallel after smoke test passes.
- **Day 2 (GPU):** T2.1, then T2.2 (sequential, same model load).
- **Day 2-3 (GPU):** T3.1 + T3.2 overnight batch.
- **Day 3 (GPU):** T3.3.
- **Day 4+ (local):** T4.1 + T4.2 in parallel. Then T4.4, T4.5.

---

## Decision Gates Summary

| Gate | After | Key Question | Go Criteria | No-Go Action |
|------|-------|-------------|-------------|--------------|
| **G-1** | T-1.1-T-1.4 | Does the task design work at prompt level? | Variance in offers, firmness shifts demand up, empathy shifts down, DG differs from UG | If no variance: try different API model. If no prompt effect: task may not discriminate — reconsider design. If DG=UG: still proceed (blind amplification is a valid finding). |
| **G0** | T0.5 | Does infrastructure work? | >5 unique offers in 10 games, <10% parse errors, pairing correct | Debug. If temp=0 too degenerate, switch to temp=0.3 + seeds. |
| **G1** | T1.1+T1.2 | Is LLM responder graded? Is baseline stable? | Acceptance curve is NOT a step function. Baseline has >5 offer levels. | If step function: add temp=0.3 to responder only. If baseline degenerate: increase temp globally. |
| **G2** | T2.1 | Do steering effects exist? | Firmness demand d>0.3. Empathy shifts in expected direction. L10≠L12. | If zero effect: infrastructure bug, debug. If wrong direction: reframe or swap dimension. |
| **G3** | T2.2 | Does DG work? | Firmness DG demand ≠ baseline. No parse errors. | If broken: fix DG prompt. If identical to baseline: DG may not add value, consider dropping. |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Empathy payoff doesn't replicate at n=150 | Medium | High | Paper still works as negative result ("behavioral ≠ strategic"). Exploratory phase shows the pattern. Confirmatory failure is itself a finding about effect fragility. |
| Temp=0 too degenerate even with variable pools | Medium | Medium | G0 catches this. Fallback: temp=0.3 with per-game seeds. |
| LLM responder acceptance is a step function | Low-Medium | Medium | G1 catches this. Fallback: temp=0.3 for responder only, or reframe as "quasi-rule-based opponent." |
| GPU instance unavailable or budget exceeded | Low | High | Total GPU: ~8 hrs = ~$8. Well within budget. Instance is already provisioned. |
| General pairs produce weaker effects than teammate's results suggested | Medium | Medium | Teammate's results were inflated by fixed pools. G2 catches this with honest variable-pool measurement. Firmness should still show strong demand shift. |
| Layer gradient doesn't hold with general pairs | Low | Medium | Already replicated across 3 batches with different pair types. G2 checks at n=30. |

---

## Budget

| Phase | GPU Hours | Cost |
|-------|-----------|------|
| Phase 0 (smoke test) | 0.25 | $0.25 |
| Phase 1 (characterization) | 1.0 | $1.00 |
| Phase 2 (exploratory) | 0.75 | $0.75 |
| Phase 3 (confirmatory) | 6.0 | $6.00 |
| **Total** | **~8 hrs** | **~$8** |

---

## Paper Structure (tentative)

1. **Introduction:** Activation steering as behavioral intervention for LLM agents. Gap: does behavioral change translate to strategic improvement?
2. **Background:** Representation engineering, contrastive activation addition, Ultimatum Game as evaluation framework.
3. **Methods:** Vector extraction (mean difference), experimental design (paired UG/DG, variable pools, pre-registered hypotheses), metrics (demand shift, expected payoff, acceptance rate).
4. **Exploratory Phase:** Broad sweep across dimensions/layers/alphas. Massive demand shifts (25+ significant configs). Near-zero payoff effects after correction. Layer gradient. General > specific observation.
5. **Confirmatory Phase:** Pre-registered empathy vs firmness experiment. Empathy: payoff improvement (if replicates). Firmness: demand without payoff. DG contrast: blind amplification.
6. **Discussion:** Behavioral dispositions vs strategic reasoning. Implications for steering as alignment tool. Layer-location as mechanistic finding. Limitations (single model, self-play).
