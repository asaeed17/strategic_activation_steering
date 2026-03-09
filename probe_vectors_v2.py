#!/usr/bin/env python3
"""
probe_vectors_v2.py

Validates steering vectors by checking whether the *specific extracted direction*
separates the target concept — not just whether the layer is linearly separable.

Two probes are run for each dimension:

  1. Projection Probe (primary)
       Projects activations onto the extracted steering vector (dot product → scalar).
       Trains logistic regression on that single scalar.
       This directly validates the extracted direction. If this is high, your vector
       is doing what you think it's doing.

  2. Full Probe (diagnostic)
       Trains logistic regression on the full activation vector.
       This tells you whether the layer *could* separate the concept in principle.
       If Full >> Projection, your extracted direction is suboptimal — the concept
       is encoded in the layer but your vector isn't aligned with it well.

Both use Leave-One-Out Cross-Validation (LOOCV), which is appropriate for small
datasets (N=20) where k-fold estimates are noisy.

Probe file auto-discovery (default behaviour):
  --probe_dir points to a directory containing per-dimension files:
    probing_{dim_id}_heldout_negotiation.json  → held-out negotiation probe
    probing_{dim_id}_general.json              → non-negotiation generalisation probe
  Files are looked up automatically for each dimension being probed.

  Alternatively, use --probe_file / --nonneg_probe_file to supply a single file
  for all dimensions (legacy behaviour).

--control_file : control dimensions (verbosity, formality, hedging, sentiment)
                 used for bias/confound analysis

By default ALL dimensions from --steering_pairs are probed in one run.

Usage (run all dimensions, auto-discover probe files):
  python probe_vectors_v2.py --model qwen2.5-3b

Usage (specific dimensions):
  python probe_vectors_v2.py --model qwen2.5-3b --dimensions firmness empathy anchoring

Usage (legacy single-file mode):
  python probe_vectors_v2.py --model qwen2.5-3b \\
      --probe_file probing_held_out_contrastive_pairs/probing_firmness_heldout_negotiation.json \\
      --nonneg_probe_file probing_held_out_contrastive_pairs/probing_firmness_general.json \\
      --dimensions firmness --layer 16
"""

import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from extract_vectors import (
    MODELS, HF_TOKEN, format_sample, extract_hidden_states, ModelConfig
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Core probing functions
# ---------------------------------------------------------------------------

def loocv_probe_projection(
    pos_h: np.ndarray,   # (N, H)
    neg_h: np.ndarray,   # (N, H)
    vector: np.ndarray,  # (H,)
) -> Tuple[float, float]:
    """
    PRIMARY PROBE: validates the specific extracted direction.

    Projects each activation onto the steering vector (single dot product scalar),
    then runs LOOCV logistic regression on those scalars.

    This answers: "Does *this specific vector direction* separate the concept?"

    Returns
    -------
    mean_accuracy : float
    std_accuracy  : float
    """
    # normalise vector
    vec = vector / (np.linalg.norm(vector) + 1e-8)

    # project activations onto vector → scalar per example
    pos_proj = pos_h @ vec   # (N,)
    neg_proj = neg_h @ vec   # (N,)

    X = np.concatenate([pos_proj, neg_proj]).reshape(-1, 1)   # (2N, 1)
    y = np.array([1] * len(pos_proj) + [0] * len(neg_proj))

    if len(X) < 4:
        log.warning("Too few samples (%d) for LOOCV — returning train accuracy.", len(X))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        acc = clf.score(X, y)
        return acc, 0.0

    loo = LeaveOneOut()
    scores = []
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))

    return float(np.mean(scores)), float(np.std(scores))


def loocv_probe_full(
    pos_h: np.ndarray,   # (N, H)
    neg_h: np.ndarray,   # (N, H)
) -> Tuple[float, float]:
    """
    DIAGNOSTIC PROBE: tests whether the layer can separate the concept at all.

    Trains logistic regression on the full activation vector.
    Compare against projection probe to diagnose vector quality:

      projection ≈ full  → your vector is well-aligned with the concept direction
      projection << full → the concept is in the layer but your vector missed it

    Returns
    -------
    mean_accuracy : float
    std_accuracy  : float
    """
    X = np.concatenate([pos_h, neg_h], axis=0)   # (2N, H)
    y = np.array([1] * len(pos_h) + [0] * len(neg_h))

    if len(X) < 4:
        log.warning("Too few samples (%d) for LOOCV — returning train accuracy.", len(X))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        acc = clf.score(X, y)
        return acc, 0.0

    loo = LeaveOneOut()
    scores = []
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))

    return float(np.mean(scores)), float(np.std(scores))


def projection_separation(
    pos_h: np.ndarray,   # (N, H)
    neg_h: np.ndarray,   # (N, H)
    vector: np.ndarray,  # (H,)
) -> Tuple[float, float, float]:
    """
    Computes how well the vector separates pos from neg in projection space.

    Returns
    -------
    mean_separation : float   (mean_pos_proj - mean_neg_proj) / pooled_std
                              i.e. Cohen's d — effect size of the separation
    mean_pos        : float   mean projection of positive examples
    mean_neg        : float   mean projection of negative examples
    """
    vec = vector / (np.linalg.norm(vector) + 1e-8)
    pos_proj = pos_h @ vec
    neg_proj = neg_h @ vec

    pooled_std = np.sqrt((pos_proj.std() ** 2 + neg_proj.std() ** 2) / 2 + 1e-8)
    cohens_d = (pos_proj.mean() - neg_proj.mean()) / pooled_std

    return float(cohens_d), float(pos_proj.mean()), float(neg_proj.mean())


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def probe_all_layers(
    pos_h: np.ndarray,        # (N, n_layers, H)
    neg_h: np.ndarray,        # (N, n_layers, H)
    vectors: np.ndarray,      # (n_layers, H)  — the extracted steering vectors
    label: str = "",
) -> Dict:
    """
    Runs both probes across all layers and returns a results dict.
    """
    n_layers = pos_h.shape[1]
    results = {
        "projection_acc":  np.zeros(n_layers),
        "projection_std":  np.zeros(n_layers),
        "full_acc":        np.zeros(n_layers),
        "full_std":        np.zeros(n_layers),
        "cohens_d":        np.zeros(n_layers),
        "mean_pos_proj":   np.zeros(n_layers),
        "mean_neg_proj":   np.zeros(n_layers),
    }

    for l in tqdm(range(n_layers), desc=f"  layers [{label}]", leave=False):
        vec = vectors[l]
        p_h = pos_h[:, l, :]
        n_h = neg_h[:, l, :]

        proj_acc, proj_std = loocv_probe_projection(p_h, n_h, vec)
        full_acc, full_std = loocv_probe_full(p_h, n_h)
        d, mp, mn = projection_separation(p_h, n_h, vec)

        results["projection_acc"][l] = proj_acc
        results["projection_std"][l] = proj_std
        results["full_acc"][l]       = full_acc
        results["full_std"][l]       = full_std
        results["cohens_d"][l]       = d
        results["mean_pos_proj"][l]  = mp
        results["mean_neg_proj"][l]  = mn

    return results


# ---------------------------------------------------------------------------
# Bias analysis
# ---------------------------------------------------------------------------

def bias_analysis(
    steering_vectors: np.ndarray,   # (n_layers, H)
    control_pos_h: np.ndarray,      # (N, n_layers, H)
    control_neg_h: np.ndarray,      # (N, n_layers, H)
) -> Dict:
    """
    For each layer, projects control activations onto the steering vector and
    computes Cohen's d of the separation.

    Near-zero d → steering vector does not encode the control concept (good).
    Large d → steering vector is contaminated by the control concept (bad).
    """
    n_layers = steering_vectors.shape[0]
    cohens_d   = np.zeros(n_layers)
    mean_pos   = np.zeros(n_layers)
    mean_neg   = np.zeros(n_layers)

    for l in range(n_layers):
        d, mp, mn = projection_separation(
            control_pos_h[:, l, :],
            control_neg_h[:, l, :],
            steering_vectors[l],
        )
        cohens_d[l] = d
        mean_pos[l] = mp
        mean_neg[l] = mn

    return {"cohens_d": cohens_d, "mean_pos": mean_pos, "mean_neg": mean_neg}


# ---------------------------------------------------------------------------
# Helpers: activation extraction for a dimension
# ---------------------------------------------------------------------------

def get_activations(
    model,
    tokenizer,
    config: ModelConfig,
    dim: Dict,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pos_h, neg_h) each shape (N, n_layers, H)."""
    pairs = dim["pairs"]
    pos_texts = [format_sample(p["context"], p["positive"], tokenizer, config) for p in pairs]
    neg_texts = [format_sample(p["context"], p["negative"], tokenizer, config) for p in pairs]
    pos_h = extract_hidden_states(model, tokenizer, pos_texts, batch_size)
    neg_h = extract_hidden_states(model, tokenizer, neg_texts, batch_size)
    return pos_h, neg_h


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_results(
    results_by_dataset: Dict[str, Dict],   # dataset_label → probe results
    dim_id: str,
    output_path: Path,
    best_layer: int,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not found, skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Probe Results: {dim_id}  (best layer = {best_layer})", fontsize=13)

    colors = ["steelblue", "darkorange", "green", "red"]

    # Panel 1: Projection accuracy (primary)
    ax = axes[0]
    for i, (label, res) in enumerate(results_by_dataset.items()):
        acc = res["projection_acc"]
        std = res["projection_std"]
        x   = np.arange(len(acc))
        ax.plot(x, acc, label=label, color=colors[i % len(colors)])
        ax.fill_between(x, acc - std, acc + std, alpha=0.15, color=colors[i % len(colors)])
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="chance")
    ax.axvline(best_layer, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("Projection Probe Accuracy\n(validates the specific vector)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("LOOCV Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Full probe accuracy (diagnostic)
    ax = axes[1]
    for i, (label, res) in enumerate(results_by_dataset.items()):
        acc = res["full_acc"]
        std = res["full_std"]
        x   = np.arange(len(acc))
        ax.plot(x, acc, label=label, color=colors[i % len(colors)])
        ax.fill_between(x, acc - std, acc + std, alpha=0.15, color=colors[i % len(colors)])
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.4, label="chance")
    ax.axvline(best_layer, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("Full Probe Accuracy\n(diagnostic: layer separability)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("LOOCV Accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cohen's d effect size
    ax = axes[2]
    for i, (label, res) in enumerate(results_by_dataset.items()):
        ax.plot(res["cohens_d"], label=label, color=colors[i % len(colors)])
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.axvline(best_layer, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("Cohen's d (projection separation)\neffect size of vector direction")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Plot saved → %s", output_path)


def plot_bias(
    bias_results: Dict[str, Dict],   # "dim_vs_control" → {cohens_d, ...}
    dim_id: str,
    output_path: Path,
    best_layer: int,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(10, 5))
    for key, res in bias_results.items():
        plt.plot(res["cohens_d"], label=key)
    plt.axhline(0, color="black", linestyle="--", alpha=0.4)
    plt.axhline(0.2,  color="gray", linestyle=":", alpha=0.4, label="small effect (d=0.2)")
    plt.axhline(-0.2, color="gray", linestyle=":", alpha=0.4)
    plt.axvline(best_layer, color="gray", linestyle=":", alpha=0.6, label=f"best layer ({best_layer})")
    plt.title(f"Bias Analysis: {dim_id} vector vs control dimensions\n"
              f"Near-zero = no confound, large |d| = contamination")
    plt.xlabel("Layer")
    plt.ylabel("Cohen's d (control separation by steering vector)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info("Bias plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    dim_id: str,
    results_by_dataset: Dict[str, Dict],
    bias_results: Dict[str, Dict],
    best_layer: int,
    method: str,
):
    print(f"\n{'=' * 70}")
    print(f"  DIMENSION: {dim_id}   |   METHOD: {method}   |   BEST LAYER: {best_layer}")
    print(f"{'=' * 70}")

    # Probe table
    cohens_d_header = "Cohen's d"
    print(f"\n  {'Dataset':<30} {'Proj Acc':>9} {'±':>4} {'Full Acc':>9} {'±':>4} {cohens_d_header:>10}")
    print(f"  {'-' * 68}")
    for label, res in results_by_dataset.items():
        l = best_layer
        print(
            f"  {label:<30} "
            f"{res['projection_acc'][l]:>9.3f} "
            f"{res['projection_std'][l]:>4.3f} "
            f"{res['full_acc'][l]:>9.3f} "
            f"{res['full_std'][l]:>4.3f} "
            f"{res['cohens_d'][l]:>10.3f}"
        )

    # Interpretation
    print(f"\n  Interpretation at layer {best_layer}:")
    for label, res in results_by_dataset.items():
        proj = res["projection_acc"][best_layer]
        full = res["full_acc"][best_layer]
        gap  = full - proj
        d    = res["cohens_d"][best_layer]
        interp = []
        if proj >= 0.80:
            interp.append("vector strongly encodes concept")
        elif proj >= 0.65:
            interp.append("vector partially encodes concept")
        else:
            interp.append("vector weakly encodes concept")
        if gap > 0.15:
            interp.append(f"vector suboptimal (layer could do {full:.2f} with better direction)")
        if abs(d) >= 0.8:
            interp.append("large effect size")
        elif abs(d) >= 0.5:
            interp.append("medium effect size")
        else:
            interp.append("small effect size")
        print(f"    [{label}] {' | '.join(interp)}")

    # Bias table
    if bias_results:
        print(f"\n  Bias (|d| > 0.5 at best layer = potential confound):")
        cohens_d_header = "Cohen's d"
        print(f"  {'Control':<35} {cohens_d_header:>10}  {'Flag':>6}")
        print(f"  {'-' * 55}")
        for key, res in bias_results.items():
            d = res["cohens_d"][best_layer]
            flag = "⚠️" if abs(d) >= 0.5 else "ok"
            print(f"  {key:<35} {d:>10.3f}  {flag:>6}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _all_dims_from_steering_pairs(steering_pairs_file: str) -> List[str]:
    """Return all dimension IDs from a steering pairs JSON, or empty list on failure."""
    p = Path(steering_pairs_file)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return [d["id"] for d in data.get("dimensions", [])]


def _resolve_probe_file(probe_dir: Optional[str], dim_id: str, suffix: str) -> Optional[Path]:
    """
    Look for probing_{dim_id}_{suffix}.json inside probe_dir.
    Returns the Path if it exists, else None.
    """
    if not probe_dir:
        return None
    candidate = Path(probe_dir) / f"probing_{dim_id}_{suffix}.json"
    return candidate if candidate.exists() else None


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe steering vectors to validate the extracted direction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",        choices=list(MODELS.keys()), default="qwen2.5-3b")
    p.add_argument("--vectors_dir",  default="vectors",
                   help="Root dir containing model subdirs with .npy files")
    p.add_argument("--method",       choices=["mean_diff", "pca"], default="mean_diff",
                   help="Which extraction method's vectors to validate")
    # Per-dimension probe auto-discovery (preferred)
    p.add_argument("--probe_dir",    default="probing_held_out_contrastive_pairs",
                   help="Directory containing per-dimension probe files named "
                        "probing_{dim_id}_heldout_negotiation.json and "
                        "probing_{dim_id}_general.json. "
                        "Used automatically for each dimension unless overridden by "
                        "--probe_file / --nonneg_probe_file.")
    # Legacy single-file overrides (still supported)
    p.add_argument("--probe_file",   default=None,
                   help="Override: single held-out negotiation pairs JSON for all dimensions")
    p.add_argument("--nonneg_probe_file", default=None,
                   help="Override: single non-negotiation pairs JSON for all dimensions")
    p.add_argument("--control_file", default="control_steering_pairs.json",
                   help="Control pairs JSON for bias/confound analysis")
    p.add_argument("--steering_pairs",
                   default="steering_pairs/15dim_20pairs_matched/negotiation_steering_pairs.json",
                   help="Steering pairs JSON used to discover all available dimension IDs "
                        "(only needed when --dimensions is not specified)")
    p.add_argument("--output_dir",   default="probe_results")
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--dimensions",   nargs="+", default=None,
                   help="Dimension IDs to probe. Defaults to all dimensions found in "
                        "--steering_pairs (or all with probe files in --probe_dir).")
    p.add_argument("--layer",        type=int, default=None,
                   help="Layer to use for summary table. Defaults to argmax of projection accuracy on probe_file.")
    return p.parse_args()


def _resolve_dimensions(args) -> List[str]:
    """
    Determine which dimensions to probe.
    Priority:
      1. Explicit --dimensions list
      2. All dims in --steering_pairs JSON
      3. All dims discoverable from --probe_dir (files matching probing_*_heldout_negotiation.json)
    """
    if args.dimensions:
        return args.dimensions

    # Try steering_pairs JSON
    dims = _all_dims_from_steering_pairs(args.steering_pairs)
    if dims:
        log.info("Discovered %d dimensions from %s", len(dims), args.steering_pairs)
        return dims

    # Fallback: scan probe_dir for heldout files
    if args.probe_dir and Path(args.probe_dir).is_dir():
        found = sorted(
            p.stem.replace("probing_", "").replace("_heldout_negotiation", "")
            for p in Path(args.probe_dir).glob("probing_*_heldout_negotiation.json")
        )
        if found:
            log.info("Discovered %d dimensions from probe_dir %s", len(found), args.probe_dir)
            return found

    log.warning("Could not auto-discover dimensions — defaulting to ['firmness'].")
    return ["firmness"]


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    output_dir = Path(args.output_dir) / cfg.alias
    output_dir.mkdir(parents=True, exist_ok=True)

    token = HF_TOKEN if cfg.requires_token else None

    # Resolve dimension list
    dimensions = _resolve_dimensions(args)
    log.info("Probing %d dimension(s): %s", len(dimensions), dimensions)

    # Load model
    log.info("Loading model: %s", cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, token=token, torch_dtype=cfg.torch_dtype, device_map="auto"
    )
    model.eval()

    # Load control pairs
    control_dims = []
    if Path(args.control_file).exists():
        with open(args.control_file) as f:
            control_dims = json.load(f)["dimensions"]
        log.info("Loaded %d control dimensions from %s", len(control_dims), args.control_file)
    else:
        log.warning("Control file not found: %s — bias analysis will be skipped.", args.control_file)

    all_results = {}

    for dim_id in dimensions:
        log.info("=" * 60)
        log.info("DIMENSION: %s", dim_id)

        # Load steering vectors (all layers)
        vec_path = Path(args.vectors_dir) / cfg.alias / args.method / f"{dim_id}_all_layers.npy"
        if not vec_path.exists():
            log.error("Steering vector not found: %s — run extract_vectors.py first.", vec_path)
            continue
        steering_vectors = np.load(vec_path)   # (n_layers, H)
        n_layers = steering_vectors.shape[0]
        log.info("Loaded steering vectors: shape %s", steering_vectors.shape)

        results_by_dataset = {}

        # --- Held-out negotiation probe ---
        # Explicit override takes priority; otherwise auto-discover per-dimension file.
        held_out_path: Optional[Path] = None
        if args.probe_file:
            held_out_path = Path(args.probe_file) if Path(args.probe_file).exists() else None
        else:
            held_out_path = _resolve_probe_file(args.probe_dir, dim_id, "heldout_negotiation")

        if held_out_path:
            log.info("Probing with held-out negotiation pairs: %s", held_out_path)
            with open(held_out_path) as f:
                probe_data = json.load(f)
            probe_dims = {d["id"]: d for d in probe_data["dimensions"]}
            if dim_id in probe_dims:
                pos_h, neg_h = get_activations(model, tokenizer, cfg, probe_dims[dim_id], args.batch_size)
                log.info("  Running projection + full probes across %d layers...", n_layers)
                results_by_dataset["held-out negotiation"] = probe_all_layers(
                    pos_h, neg_h, steering_vectors, label="held-out neg"
                )
            else:
                log.warning("Dimension '%s' not found in held-out negotiation file.", dim_id)
        else:
            log.warning("No held-out negotiation probe file for '%s' — skipping.", dim_id)

        # --- Non-negotiation probe ---
        nonneg_path: Optional[Path] = None
        if args.nonneg_probe_file:
            nonneg_path = Path(args.nonneg_probe_file) if Path(args.nonneg_probe_file).exists() else None
        else:
            nonneg_path = _resolve_probe_file(args.probe_dir, dim_id, "general")

        if nonneg_path:
            log.info("Probing with non-negotiation pairs: %s", nonneg_path)
            with open(nonneg_path) as f:
                nonneg_data = json.load(f)
            nonneg_dims = {d["id"]: d for d in nonneg_data["dimensions"]}
            if dim_id in nonneg_dims:
                pos_h, neg_h = get_activations(model, tokenizer, cfg, nonneg_dims[dim_id], args.batch_size)
                log.info("  Running projection + full probes across %d layers...", n_layers)
                results_by_dataset["non-negotiation general"] = probe_all_layers(
                    pos_h, neg_h, steering_vectors, label="non-neg"
                )
            else:
                log.warning("Dimension '%s' not found in non-negotiation file.", dim_id)
        else:
            log.warning("No non-negotiation probe file for '%s' — skipping.", dim_id)

        if not results_by_dataset:
            log.error("No probe datasets loaded for %s — skipping.", dim_id)
            continue

        # Determine best layer: argmax of projection accuracy on first available dataset
        first_res  = next(iter(results_by_dataset.values()))
        best_layer = args.layer if args.layer is not None else int(np.argmax(first_res["projection_acc"]))
        log.info("Best layer (by projection accuracy): %d", best_layer)

        # --- Bias analysis ---
        bias_results = {}
        if control_dims:
            log.info("Running bias analysis against %d control dimensions...", len(control_dims))
            for c_dim in control_dims:
                c_id = c_dim["id"]
                c_pos_h, c_neg_h = get_activations(model, tokenizer, cfg, c_dim, args.batch_size)
                bias_results[f"{dim_id}_vs_{c_id}"] = bias_analysis(steering_vectors, c_pos_h, c_neg_h)

        # Print summary
        print_summary(dim_id, results_by_dataset, bias_results, best_layer, args.method)

        # Plot
        plot_probe_results(
            results_by_dataset, dim_id,
            output_dir / f"{dim_id}_probe_curves.png",
            best_layer,
        )
        if bias_results:
            plot_bias(
                bias_results, dim_id,
                output_dir / f"{dim_id}_bias.png",
                best_layer,
            )

        # Collect for JSON output
        all_results[dim_id] = {
            "best_layer": best_layer,
            "method": args.method,
            "datasets": {
                label: {k: v.tolist() for k, v in res.items()}
                for label, res in results_by_dataset.items()
            },
            "bias": {
                k: {kk: vv.tolist() for kk, vv in v.items()}
                for k, v in bias_results.items()
            },
        }

    # Save raw results
    out_json = output_dir / "probe_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Raw results saved → %s", out_json)

    log.info("Done.")


if __name__ == "__main__":
    main()