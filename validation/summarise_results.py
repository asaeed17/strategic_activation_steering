"""
summarise_results.py
--------------------
Read FINAL_VALIDATION_RESULTS/{variant}/{model}/validation_results.json for
every available variant of a model and produce a ranked summary report.

Usage
-----
    python summarise_results.py                          # defaults
    python summarise_results.py --model qwen2.5-3b
    python summarise_results.py --results_dir ../FINAL_VALIDATION_RESULTS --output summary.md
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── per-variant metric extraction ─────────────────────────────────────────────

def extract_metrics(data: dict, probe_v2_data: dict = None) -> dict:
    """Return a flat dict of scalar metrics for one variant/model combo."""
    if probe_v2_data is None:
        probe_v2_data = {}
    control_ids = set(data.get("control_dim_ids", []))

    # ── probe patterns (negotiation dims only) ───────────────────────────────
    probe_means, probe_maxes, flat_high_count = [], [], 0
    for dim_id, v in data.get("probe_patterns", {}).items():
        if dim_id in control_ids:
            continue
        probe_means.append(v["mean_acc"])
        probe_maxes.append(v["max_acc"])
        if v.get("pattern") == "flat_high":
            flat_high_count += 1
    n_neg_dims = len(probe_means)

    # ── pair consistency (negotiation dims only) ─────────────────────────────
    consistency_means, total_outliers, total_pairs = [], 0, 0
    for dim_id, v in data.get("pair_consistency", {}).items():
        if dim_id in control_ids:
            continue
        consistency_means.append(v["overall_mean"])
        total_outliers += v.get("n_outliers", len(v.get("outlier_pairs", [])))
        total_pairs += len(v.get("per_pair_mean_alignment", []))
    # Outlier rate (0-1): fraction of all evaluated pairs that are outliers.
    # Raw count / n_dims would unfairly penalise 80-pair variants 6-7× more
    # than 12-pair variants for the same underlying outlier rate.
    outlier_frac = total_outliers / max(total_pairs, 1)

    # ── selectivity (negotiation dims only) ──────────────────────────────────
    selectivities = []
    for dim_id, v in data.get("selectivity", {}).items():
        if dim_id in control_ids:
            continue
        selectivities.append(v["best_selectivity"])

    # ── steering direction probe (negotiation dims only) ─────────────────────
    one_d_accs = []
    for dim_id, v in data.get("steering_direction_probe", {}).items():
        if dim_id in control_ids:
            continue
        one_d_accs.append(v.get("best_acc", v.get("best_1d", float("nan"))))

    # ── length confound ───────────────────────────────────────────────────────
    severity_counts = {"OK": 0, "MILD": 0, "MODERATE": 0, "SEVERE": 0}
    for dim_id, v in data.get("length_confound", {}).items():
        if dim_id in control_ids:
            continue
        sev = v.get("severity", "OK")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # ── vocabulary overlap ────────────────────────────────────────────────────
    vocab_ok, vocab_concerns = 0, 0
    for dim_id, v in data.get("vocabulary_overlap", {}).items():
        if dim_id in control_ids:
            continue
        if v.get("concern", "OK") == "OK":
            vocab_ok += 1
        else:
            vocab_concerns += 1

    # ── permutation test: fraction of layers that are significant ────────────
    perm_sig_fracs = []
    for dim_id, layers in data.get("permutation_test", {}).items():
        if dim_id in control_ids:
            continue
        sigs = [v["significant"] for v in layers.values()]
        if sigs:
            perm_sig_fracs.append(sum(sigs) / len(sigs))

    # ── Cohen's d bias: fraction of layer-pairs with severe entanglement ─────
    # Using a 0-1 fraction rather than a raw count prevents this from
    # dominating the score (raw count reaches ~900-1700, swamping all positives).
    severe_layer_total, total_layer_pairs = 0, 0
    for _pair_key, v in data.get("cohens_d_bias", {}).items():
        sl = v.get("severe_layers", [])
        severe_layer_total += len(sl) if isinstance(sl, list) else int(sl)
        total_layer_pairs += len(v.get("cohens_d_per_layer", []))
    severe_bias_frac = severe_layer_total / max(total_layer_pairs, 1)

    # ── Stage 2: held-out projection probe (probe_v2_results.json) ───────────
    # Best-layer projection accuracy on pairs NOT used during vector extraction.
    # nan when probe_v2_results.json was not produced for this variant.
    probe_v2_heldout_accs = []
    for v in probe_v2_data.values():
        per_layer = v.get("datasets", {}).get("held-out negotiation", {}).get("projection_acc", [])
        if per_layer:
            probe_v2_heldout_accs.append(max(per_layer))

    return {
        "n_neg_dims":             n_neg_dims,
        "probe_mean_acc":         _mean(probe_means),
        "probe_max_acc":          _mean(probe_maxes),
        "flat_high_dims":         flat_high_count,
        "consistency_mean":       _mean(consistency_means),
        "outlier_pairs_total":    total_outliers,   # raw count (for display)
        "outlier_frac":           outlier_frac,     # 0-1 rate  (for scoring)
        "selectivity_mean":       _mean(selectivities),
        "one_d_acc_mean":         _mean(one_d_accs),
        "length_severe":          severity_counts.get("SEVERE", 0),
        "length_moderate":        severity_counts.get("MODERATE", 0),
        "vocab_concerns":         vocab_concerns,
        "perm_sig_frac":          _mean(perm_sig_fracs),
        "severe_bias_layers":     severe_layer_total,  # raw count (for display)
        "severe_bias_frac":       severe_bias_frac,    # 0-1 frac  (for scoring)
        "probe_v2_heldout_acc":   _mean(probe_v2_heldout_accs),
        "n_warnings":             len(data.get("warnings", [])),
        "overall_assessment":     data.get("overall_assessment", ""),
    }


# ── composite score ────────────────────────────────────────────────────────────

def composite_score(m: dict) -> float:
    """
    Higher = better quality steering vectors.

    Positive contributors (all scaled 0–1):
      probe_v2_heldout_acc  weight 2.5 — held-out projection probe (Stage 2);
                                         skipped (0 contribution) if unavailable
      selectivity_mean      weight 3   — most diagnostic of in-sample vector quality
      consistency_mean      weight 2   — pairs agree on direction
      perm_sig_frac         weight 2   — concept actually encoded (sanity check)
      one_d_acc_mean        weight 1   — 1-D separability along steering direction

    Negative contributors (all 0–1 fractions or per-dim rates):
      flat_high_dims / n    weight 2   — surface-feature artefact
      outlier_frac          weight 2   — fraction of all pairs that are outliers;
                                         normalised so 12- and 80-pair variants are
                                         comparable on the same scale
      length_severe / n     weight 2   — confounded by token length
      length_moderate / n   weight 1
      vocab_concerns / n    weight 1   — lexical leakage
      severe_bias_frac      weight 1.5 — fraction of layer-pairs with severe
                                         control-dimension entanglement (0–1);
                                         previously a raw count that dominated
    """
    n = max(m["n_neg_dims"], 1)
    score  = 3.0 * m["selectivity_mean"]
    score += 2.0 * m["consistency_mean"]
    score += 2.0 * m["perm_sig_frac"]
    score += 1.0 * m["one_d_acc_mean"]

    # Stage 2 held-out probe — only add when the file was present
    v2 = m["probe_v2_heldout_acc"]
    if not math.isnan(v2):
        score += 2.5 * v2

    score -= 2.0 * (m["flat_high_dims"] / n)
    score -= 2.0 * m["outlier_frac"]          # 0-1 rate, not raw count / n_dims
    score -= 2.0 * (m["length_severe"] / n)
    score -= 1.0 * (m["length_moderate"] / n)
    score -= 1.0 * (m["vocab_concerns"] / n)
    score -= 1.5 * m["severe_bias_frac"]      # 0-1 fraction, not raw count × 0.05

    return round(score, 4)


def score_to_assessment(score: float) -> str:
    """
    Derive a summary-level assessment from the composite score.

    The Stage 1 overall_assessment is NOT used here because its n_severe
    threshold is trivially exceeded by cohens_d_bias alone (every neg×control
    pair with even one severe layer adds 1, so 40 pairs → 40 >> threshold of 3),
    causing every variant to be labelled CONCERNING regardless of how well the
    vectors actually perform on held-out probes or 1-D steering accuracy.

    Thresholds (max achievable score ≈ 10.5 with all metrics at 1.0):
      ≥ 7.0  GOOD      — strong across all checks
      ≥ 5.0  ACCEPTABLE — solid vectors with documented caveats
      ≥ 3.5  MIXED     — meaningful concerns; use recommended layers only
      < 3.5  POOR      — significant confounds; rework contrastive pairs
    """
    if score >= 7.0:
        return "GOOD — strong evidence of vector quality across all checks."
    if score >= 5.0:
        return "ACCEPTABLE — solid vectors with documented caveats; use recommended layers."
    if score >= 3.5:
        return "MIXED — meaningful concerns present; use recommended layers only."
    return "POOR — significant confounds detected; rework contrastive pairs before use."


# ── report formatting ─────────────────────────────────────────────────────────

def render_report(rows: list, model: str) -> str:
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)

    lines = [
        f"# Steering Vector Quality Summary — {model}",
        "",
        "Variants ranked by composite quality score (higher = better).",
        "Score weights: V2-probe × 2.5 (held-out, skipped if absent), selectivity × 3,",
        "consistency × 2, permutation significance × 2, 1-D probe acc × 1;",
        "penalties for flat-high probe, outlier rate, length confound, vocab overlap,",
        "and severe-bias fraction (all penalties normalised to 0–1 scale).",
        "",
        "---",
        "",
    ]

    # ── ranking table ─────────────────────────────────────────────────────────
    col_w = [4, 30, 7, 9, 11, 12, 10, 10, 11, 9, 10, 12]
    header = (
        f"{'Rank':<{col_w[0]}}  "
        f"{'Variant':<{col_w[1]}}  "
        f"{'Score':>{col_w[2]}}  "
        f"{'Selectiv':>{col_w[3]}}  "
        f"{'Consist':>{col_w[4]}}  "
        f"{'PermSig%':>{col_w[5]}}  "
        f"{'1D-Acc':>{col_w[6]}}  "
        f"{'V2Probe':>{col_w[7]}}  "
        f"{'FlatHigh':>{col_w[8]}}  "
        f"{'OutlRate':>{col_w[9]}}  "
        f"{'LenSev':>{col_w[10]}}  "
        f"{'SevBias%':>{col_w[11]}}"
    )
    sep = "-" * len(header)
    lines += [f"```", header, sep]

    for rank, r in enumerate(rows, 1):
        m = r["metrics"]
        v2 = m["probe_v2_heldout_acc"]
        v2_str = f"{v2:>{col_w[7]}.3f}" if not math.isnan(v2) else f"{'n/a':>{col_w[7]}}"
        lines.append(
            f"{rank:<{col_w[0]}}  "
            f"{r['variant']:<{col_w[1]}}  "
            f"{r['score']:>{col_w[2]}.4f}  "
            f"{m['selectivity_mean']:>{col_w[3]}.3f}  "
            f"{m['consistency_mean']:>{col_w[4]}.3f}  "
            f"{m['perm_sig_frac']:>{col_w[5]}.1%}  "
            f"{m['one_d_acc_mean']:>{col_w[6]}.3f}  "
            f"{v2_str}  "
            f"{m['flat_high_dims']:>{col_w[8]}}  "
            f"{m['outlier_frac']:>{col_w[9]}.1%}  "
            f"{m['length_severe']:>{col_w[10]}}  "
            f"{m['severe_bias_frac']:>{col_w[11]}.1%}"
        )

    lines += [f"```", ""]

    # ── per-variant detail ────────────────────────────────────────────────────
    lines += ["---", "", "## Per-Variant Detail", ""]
    for rank, r in enumerate(rows, 1):
        m = r["metrics"]
        v2 = m["probe_v2_heldout_acc"]
        v2_str = f"{v2:.3f}" if not math.isnan(v2) else "n/a (Stage 2 not run)"
        lines += [
            f"### {rank}. {r['variant']}  (score: {r['score']})",
            "",
            f"- **Negotiation dims**: {m['n_neg_dims']}",
            f"- **Probe accuracy (in-sample)**: mean {m['probe_mean_acc']:.3f}, max {m['probe_max_acc']:.3f}"
            + (f"  ⚠ {m['flat_high_dims']} flat-high dim(s)" if m["flat_high_dims"] else ""),
            f"- **Held-out projection probe (Stage 2)**: {v2_str}",
            f"- **Pair consistency**: {m['consistency_mean']:.3f}"
            + (f"  ⚠ {m['outlier_pairs_total']} outlier pair(s) ({m['outlier_frac']:.1%} rate)" if m["outlier_pairs_total"] else ""),
            f"- **Selectivity**: {m['selectivity_mean']:.3f}",
            f"- **1-D steering probe acc**: {m['one_d_acc_mean']:.3f}",
            f"- **Permutation significance**: {m['perm_sig_frac']:.1%} of sampled layers",
            f"- **Length confound**: {m['length_severe']} severe, {m['length_moderate']} moderate",
            f"- **Vocab overlap concerns**: {m['vocab_concerns']}",
            f"- **Severe bias layer-pairs**: {m['severe_bias_layers']} ({m['severe_bias_frac']:.1%} of all layer-pairs)",
            f"- **Warnings**: {m['n_warnings']}",
            f"- **Assessment**: {score_to_assessment(r['score'])}",
            f"  *(Stage 1 label: {m['overall_assessment'].split('—')[0].strip()})*",
            "",
        ]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent
    default_results = here.parent / "FINAL_VALIDATION_RESULTS"

    p = argparse.ArgumentParser(description="Rank steering vector variants by validation quality.")
    p.add_argument("--results_dir", default=str(default_results),
                   help="Root of FINAL_VALIDATION_RESULTS (default: auto-detected)")
    p.add_argument("--model", default="qwen2.5-3b",
                   help="Model subdirectory name to summarise (default: qwen2.5-3b)")
    p.add_argument("--output", default=None,
                   help="Write report to this file (default: print to stdout)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results dir not found: {results_dir}")

    rows = []
    for variant_dir in sorted(results_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        model_dir = variant_dir / args.model
        json_paths = list(model_dir.glob("**/validation_results.json"))
        if not json_paths:
            print(f"[skip] {variant_dir.name} — no validation_results.json for {args.model}",
                  file=sys.stderr)
            continue
        json_path = json_paths[0]
        data = _load(json_path)
        probe_v2_paths = list(model_dir.glob("**/probe_v2_results.json"))
        probe_v2_data = _load(probe_v2_paths[0]) if probe_v2_paths else {}
        metrics = extract_metrics(data, probe_v2_data)
        rows.append({
            "variant": variant_dir.name,
            "metrics": metrics,
            "score":   composite_score(metrics),
        })

    if not rows:
        sys.exit(f"No results found under {results_dir} for model '{args.model}'.")

    report = render_report(rows, args.model)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
