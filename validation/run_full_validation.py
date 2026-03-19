#!/usr/bin/env python3
"""
run_full_validation.py — Full validation pipeline orchestrator.

Runs all five validation stages for every variant/model combination and
writes results to FINAL_VALIDATION_RESULTS/{variant}/{model}/.

Stages
------
1. validate_vectors.py --full
     In-sample checks 1-7: length confound, probe accuracy + permutation
     test, cosine similarity matrix, Cohen's d bias, per-pair alignment,
     1-D steering direction probe, selectivity & layer recommendation.
     Requires GPU.

2. probe_vectors_v2.py
     Held-out projection probe: the ONLY probe that tests the extracted
     direction on pairs NOT used during vector extraction. Runs on both
     heldout-negotiation and non-negotiation-general pairs.
     Requires GPU.

3. orthogonal_projection.py --probe
     Causal surface-confound test: projects out all 5 control directions
     from each negotiation vector and measures how much probe accuracy
     survives. Distinguishes data confounds from direction confounds.
     Requires GPU.

4. dose_response_validation.py
     Functional validation on negotiation-domain prompts. Tests whether
     scaling alpha monotonically increases the steered behaviour as rated
     by an LLM judge. Gold-standard functional check.
     Stage 4a (generate): requires GPU.
     Stage 4b (judge):    requires LLM API key (Gemini/GPT/LLaMA).
     Stage 4c (analyze):  CPU only.

5. dose_response_validation_general.py
     Same as stage 4 but on topic-diverse general prompts (creative
     writing, Q&A, emotional support, etc.). Tests whether the vector
     generalises beyond negotiation contexts.
     Stage 5a/5b/5c: same requirements as stage 4.

Output structure
----------------
FINAL_VALIDATION_RESULTS/
  {variant}/
    {model}/
      validation_results.json        (stage 1)
      validation_report.txt          (stage 1)
      probe_v2_results.json          (stage 2)
      probe_v2_{dim}.png             (stage 2, one per dimension)
      orthogonal_projection.json     (stage 3)
      dose_response_generations.json (stage 4a)
      dose_response_scores.json      (stage 4b)
      dose_response_analysis/        (stage 4c)
      dose_response_general_generations.json  (stage 5a)
      dose_response_general_scores.json       (stage 5b)
      dose_response_general_analysis/         (stage 5c)

Usage
-----
  # All variants, default model, all stages
  python run_full_validation.py

  # Specific variant and model
  python run_full_validation.py --variants neg8dim_12pairs_matched --models qwen2.5-3b

  # GPU-only stages (no API key needed — skip judge sub-stages)
  python run_full_validation.py --no-judge

  # CPU-only analyze-only (no GPU at all)
  python run_full_validation.py --analyze-only

  # Skip specific top-level stages
  python run_full_validation.py --skip-stage 4 --skip-stage 5

  # Choose LLM judge
  python run_full_validation.py --judges gemini gpt
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

# Project root is one level above this script (validation/)
ROOT = Path(__file__).parent.parent.resolve()

# Validation scripts live alongside this file
VAL_DIR = Path(__file__).parent.resolve()

# ── Configuration ─────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    "neg15dim_12pairs_raw",
    "neg15dim_12pairs_matched",
    "neg15dim_20pairs_matched",
    "neg15dim_80pairs_matched",
    "neg8dim_12pairs_raw",
    "neg8dim_12pairs_matched",
    "neg8dim_20pairs_matched",
    "neg8dim_80pairs_matched",
]

ALL_MODELS = ["qwen2.5-3b"]

OUTPUT_ROOT = ROOT / "FINAL_VALIDATION_RESULTS"

# orthogonal_projection.py hardcodes its output path and ignores --output-dir
PROJECTION_HARDCODED_ROOT = ROOT / "results" / "projection"


# ── Subprocess helper ─────────────────────────────────────────────────────────

def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess command from project root. Returns True on success."""
    log.info("Running %s: %s", label, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        log.error("%s failed (exit %d)", label, result.returncode)
        return False
    return True


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage1(variant: str, model: str, out_dir: Path, analyze_only: bool) -> bool:
    """validate_vectors.py — in-sample validation checks 1-7."""
    mode_flag = "--analyze-only" if analyze_only else "--full"
    cmd = [
        sys.executable, str(VAL_DIR / "validate_vectors.py"),
        mode_flag,
        "--model",             model,
        "--negotiation_pairs", str(ROOT / "steering_pairs" / variant / "negotiation_steering_pairs.json"),
        "--control_pairs",     str(ROOT / "steering_pairs" / variant / "control_steering_pairs.json"),
        "--vectors_dir",       str(ROOT / "vectors" / variant / "negotiation"),
        "--output_dir",        str(out_dir.parent),  # script appends /{model}/ itself
    ]
    return run(cmd, f"Stage 1 [{variant}/{model}]")


def run_stage2(variant: str, model: str, out_dir: Path) -> bool:
    """probe_vectors_v2.py — held-out projection probe."""
    cmd = [
        sys.executable, str(VAL_DIR / "probe_vectors_v2.py"),
        "--model",          model,
        "--vectors_dir",    str(ROOT / "vectors" / variant / "negotiation"),
        "--steering_pairs", str(ROOT / "steering_pairs" / variant / "negotiation_steering_pairs.json"),
        "--control_file",   str(ROOT / "steering_pairs" / variant / "control_steering_pairs.json"),
        "--probe_dir",      str(ROOT / "probing_held_out_contrastive_pairs"),
        "--output_dir",     str(out_dir.parent),  # script appends /{model.alias}/ itself
    ]
    ok = run(cmd, f"Stage 2 [{variant}/{model}]")
    if not ok:
        return False

    # Rename to avoid collision with stage 1's probe_results.json
    src = out_dir / "probe_results.json"
    dst = out_dir / "probe_v2_results.json"
    if src.exists() and not dst.exists():
        src.rename(dst)

    return True


def run_stage3(variant: str, model: str, out_dir: Path) -> bool:
    """orthogonal_projection.py --probe — surface confound causal test."""
    cmd = [
        sys.executable, str(VAL_DIR / "orthogonal_projection.py"),
        "--variant", variant,
        "--probe",
    ]
    # Run from project root so orthogonal_projection.py resolves vectors/ correctly
    ok = run(cmd, f"Stage 3 [{variant}/{model}]")
    if not ok:
        return False

    # Script hardcodes output to results/projection/{variant}/ — copy into our structure
    src = PROJECTION_HARDCODED_ROOT / variant / "orthogonal_projection.json"
    dst = out_dir / "orthogonal_projection.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Copied orthogonal_projection.json → %s", dst)
    else:
        log.warning("orthogonal_projection.json not found at %s", src)

    return True


def run_dose_response(
    script: str,
    variant: str,
    model: str,
    out_dir: Path,
    judges: list[str],
    run_judge: bool,
    gen_file: str,
    scores_file: str,
    label_prefix: str,
) -> bool:
    """
    Run a dose_response_validation script (negotiation or general) in three
    sub-stages: generate → judge → analyze.

    Returns True only if all requested sub-stages succeed.
    """
    ok = True

    # Sub-stage a: generate (GPU)
    cmd_gen = [
        sys.executable, str(VAL_DIR / script),
        "--stage",       "generate",
        "--model",       model,
        "--vectors_dir", str(ROOT / "vectors" / variant / "negotiation"),
        "--output_dir",  str(out_dir),
    ]
    if not run(cmd_gen, f"{label_prefix} generate [{variant}/{model}]"):
        ok = False

    # Sub-stage b: judge (API — skippable)
    if run_judge and ok:
        cmd_judge = [
            sys.executable, str(VAL_DIR / script),
            "--stage",            "judge",
            "--judges",           *judges,
            "--generations_file", str(out_dir / gen_file),
            "--output_dir",       str(out_dir),
        ]
        if not run(cmd_judge, f"{label_prefix} judge [{variant}/{model}]"):
            ok = False

    # Sub-stage c: analyze (CPU)
    if ok:
        cmd_analyze = [
            sys.executable, str(VAL_DIR / script),
            "--stage",       "analyze",
            "--scores_file", str(out_dir / scores_file),
            "--output_dir",  str(out_dir),
        ]
        if not run(cmd_analyze, f"{label_prefix} analyze [{variant}/{model}]"):
            ok = False

    return ok


def run_stage4(
    variant: str, model: str, out_dir: Path,
    judges: list[str], run_judge: bool,
) -> bool:
    """dose_response_validation.py — functional validation, negotiation prompts."""
    return run_dose_response(
        script        = "dose_response_validation.py",
        variant       = variant,
        model         = model,
        out_dir       = out_dir,
        judges        = judges,
        run_judge     = run_judge,
        gen_file      = "dose_response_generations.json",
        scores_file   = "dose_response_scores.json",
        label_prefix  = "Stage 4",
    )


def run_stage5(
    variant: str, model: str, out_dir: Path,
    judges: list[str], run_judge: bool,
) -> bool:
    """dose_response_validation_general.py — functional validation, general prompts."""
    return run_dose_response(
        script        = "dose_response_validation_general.py",
        variant       = variant,
        model         = model,
        out_dir       = out_dir,
        judges        = judges,
        run_judge     = run_judge,
        gen_file      = "dose_response_general_generations.json",
        scores_file   = "dose_response_general_scores.json",
        label_prefix  = "Stage 5",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full validation pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS,
        help="Variants to run (default: all 8)",
    )
    p.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help="Models to run (default: qwen2.5-3b)",
    )
    p.add_argument(
        "--skip-stage", type=int, action="append", default=[], dest="skip_stages",
        metavar="N",
        help="Skip stage N (1-5). Can be specified multiple times.",
    )
    p.add_argument(
        "--analyze-only", action="store_true",
        help="Stage 1 in --analyze-only mode (no GPU). Stages 2, 3, 4, 5 are skipped.",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="Skip the LLM judge sub-stage in stages 4 and 5 (no API key needed). "
             "Generate and analyze sub-stages still run.",
    )
    p.add_argument(
        "--judges", nargs="+", default=["gemini"],
        choices=["gemini", "gpt", "llama", "ollama"],
        help="LLM judge(s) to use for stages 4 and 5 (default: gemini)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    skip = set(args.skip_stages)
    if args.analyze_only:
        skip |= {2, 3, 4, 5}

    run_judge = not args.no_judge

    log.info("Variants : %s", args.variants)
    log.info("Models   : %s", args.models)
    log.info("Stages   : %s", sorted({1, 2, 3, 4, 5} - skip))
    log.info("Judges   : %s", args.judges if run_judge else "skipped (--no-judge)")
    log.info("Output   : %s", OUTPUT_ROOT)

    failures: list[str] = []

    for variant in args.variants:
        for model in args.models:
            out_dir = OUTPUT_ROOT / variant / model
            out_dir.mkdir(parents=True, exist_ok=True)

            label = f"{variant}/{model}"
            log.info("=" * 60)
            log.info("Processing: %s", label)
            log.info("=" * 60)

            if 1 not in skip:
                if not run_stage1(variant, model, out_dir, args.analyze_only):
                    failures.append(f"Stage 1: {label}")

            if 2 not in skip:
                if not run_stage2(variant, model, out_dir):
                    failures.append(f"Stage 2: {label}")

            if 3 not in skip:
                if not run_stage3(variant, model, out_dir):
                    failures.append(f"Stage 3: {label}")

            if 4 not in skip:
                if not run_stage4(variant, model, out_dir, args.judges, run_judge):
                    failures.append(f"Stage 4: {label}")

            if 5 not in skip:
                if not run_stage5(variant, model, out_dir, args.judges, run_judge):
                    failures.append(f"Stage 5: {label}")

    log.info("=" * 60)
    if failures:
        log.error("Pipeline finished with %d failure(s):", len(failures))
        for f in failures:
            log.error("  FAILED: %s", f)
        sys.exit(1)
    else:
        log.info("Pipeline complete. Results in %s/", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
