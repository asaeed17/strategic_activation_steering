#!/usr/bin/env python3
"""
validate_vectors.py

Validates that extracted steering vectors encode meaningful behavioural
directions, not noise or surface artefacts. Run AFTER extract_vectors.py
and BEFORE using vectors in negotiation games.

Three validation checks:

  1. PCA Separation — do pos/neg activations separate in the top-2 PCs?
     Metrics: silhouette score, leave-one-out SVM accuracy.

  2. Split-Half Stability — are vectors consistent across independent
     subsets of the contrastive pairs?
     Metric: cosine similarity between first-half and second-half vectors.

  3. Cross-Dimension Similarity — are dimensions actually distinct?
     Metric: pairwise cosine similarity matrix; flag pairs > 0.5.

Usage:
  # Full validation (requires GPU, re-extracts activations)
  python validate_vectors.py --model qwen2.5-3b --pairs_file negotiation_steering_pairs.json \\
      --vectors_dir vectors_gpu --layers 8 12 16 20 24 --output_dir results/validation

  # CPU-only: skip re-extraction, just validate saved vectors
  python validate_vectors.py --model qwen2.5-3b --vectors_dir vectors_gpu \\
      --layers 8 12 16 20 24 --skip_extraction --output_dir results/validation
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Imports from the extraction pipeline
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import extract_vectors
# even when invoked from a subdirectory.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extract_vectors import (
    MODELS,
    compute_mean_diff,
    compute_pca_direction,
    extract_hidden_states,
    format_sample,
)

# ---------------------------------------------------------------------------
# Optional matplotlib — degrade gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Check 1: PCA Separation
# =========================================================================

def pca_separation_check(
    pos_h: np.ndarray,       # (N, n_layers, H)
    neg_h: np.ndarray,       # (N, n_layers, H)
    dim_id: str,
    layers: List[int],
    output_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, float]]:
    """
    For each requested layer, concatenate pos/neg activations, project to
    PC1/PC2, compute silhouette score and leave-one-out SVM accuracy.

    Returns {layer: {"silhouette": float, "svm_acc": float}}.
    """
    results: Dict[int, Dict[str, float]] = {}

    for layer in layers:
        pos_l = pos_h[:, layer, :]   # (N, H)
        neg_l = neg_h[:, layer, :]   # (N, H)

        X = np.concatenate([pos_l, neg_l], axis=0)         # (2N, H)
        y = np.array([1] * pos_l.shape[0] + [0] * neg_l.shape[0])

        # PCA to 2D for visualisation and silhouette
        pca = PCA(n_components=2, svd_solver="full")
        X_2d = pca.fit_transform(X)

        # Silhouette score (-1 to 1; >0.3 = reasonable separation)
        if len(set(y)) > 1:
            sil = float(silhouette_score(X_2d, y))
        else:
            sil = 0.0

        # Leave-one-out SVM accuracy
        loo = LeaveOneOut()
        scaler = StandardScaler()
        correct = 0
        total = 0
        for train_idx, test_idx in loo.split(X):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LinearSVC(max_iter=5000, dual="auto")
            clf.fit(X_train, y_train)
            if clf.predict(X_test)[0] == y_test[0]:
                correct += 1
            total += 1

        svm_acc = correct / total if total > 0 else 0.0

        results[layer] = {"silhouette": sil, "svm_acc": svm_acc}

        # Optional plot
        if HAS_MATPLOTLIB and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            pos_mask = y == 1
            neg_mask = y == 0
            ax.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1],
                       c="tab:blue", label="positive", alpha=0.7, s=40)
            ax.scatter(X_2d[neg_mask, 0], X_2d[neg_mask, 1],
                       c="tab:red", label="negative", alpha=0.7, s=40)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
            ax.set_title(
                f"{dim_id}  layer {layer}\n"
                f"silhouette={sil:.3f}  SVM_LOO={svm_acc:.3f}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_dir / f"pca_separation_{dim_id}_layer{layer:02d}.png",
                dpi=150,
            )
            plt.close(fig)

    return results


# =========================================================================
# Check 2: Split-Half Stability
# =========================================================================

def split_half_stability(
    pos_h: np.ndarray,       # (N, n_layers, H)
    neg_h: np.ndarray,       # (N, n_layers, H)
    layers: List[int],
    methods: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Split pairs into first half / second half, extract vectors from each,
    compute cosine similarity at every requested layer.

    Returns {method: {layer: cosine_similarity}}.
    """
    N = pos_h.shape[0]
    mid = N // 2
    if mid < 2:
        log.warning("Too few pairs (%d) for a meaningful split-half check.", N)
        return {}

    pos_a, pos_b = pos_h[:mid], pos_h[mid:]
    neg_a, neg_b = neg_h[:mid], neg_h[mid:]

    method_fns = {
        "mean_diff": compute_mean_diff,
        "pca": compute_pca_direction,
    }

    results: Dict[str, Dict[int, float]] = {}

    for method in methods:
        fn = method_fns.get(method)
        if fn is None:
            log.warning("Unknown method '%s' — skipping split-half.", method)
            continue

        vec_a = fn(pos_a, neg_a)   # (n_layers, H)
        vec_b = fn(pos_b, neg_b)   # (n_layers, H)

        layer_cos: Dict[int, float] = {}
        for layer in layers:
            a = vec_a[layer]
            b = vec_b[layer]
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            cos = dot / max(na * nb, 1e-12)
            layer_cos[layer] = cos

        results[method] = layer_cos

    return results


def split_half_stability_from_saved(
    vectors_dir: Path,
    dim_id: str,
    layers: List[int],
    methods: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Fallback for --skip_extraction: load the all-layers vector and compute
    cosine similarity between the first-half and second-half of the
    hidden_dim as a rough proxy. But this is NOT the same as split-half on
    activations -- if we only have saved vectors (no activations), we cannot
    do a proper split-half.

    Returns empty dict and logs a warning.
    """
    log.warning(
        "Split-half stability requires re-extracting activations. "
        "Skipping for dimension '%s' because --skip_extraction is set.",
        dim_id,
    )
    return {}


# =========================================================================
# Check 3: Cross-Dimension Similarity
# =========================================================================

def cross_dimension_similarity(
    vectors_dir: Path,
    method: str,
    layer: int,
    dim_ids: List[str],
) -> Optional[Dict]:
    """
    Load vectors for all dimensions at a given method/layer, compute pairwise
    cosine similarity, flag high-similarity pairs.

    Returns a dict with keys: "dim_ids", "matrix" (list of lists),
    "flagged_pairs" (list of {"dim_a", "dim_b", "cosine"} with cosine > 0.5).
    Returns None if insufficient vectors found on disk.
    """
    vec_dir = vectors_dir / method
    vecs = []
    found_ids = []

    for dim_id in dim_ids:
        fpath = vec_dir / f"{dim_id}_layer{layer:02d}.npy"
        if not fpath.exists():
            # try the all_layers file
            all_path = vec_dir / f"{dim_id}_all_layers.npy"
            if all_path.exists():
                all_v = np.load(all_path)
                if layer < all_v.shape[0]:
                    vecs.append(all_v[layer])
                    found_ids.append(dim_id)
                    continue
            log.warning(
                "Vector not found for dim=%s method=%s layer=%d — skipping.",
                dim_id, method, layer,
            )
            continue
        vecs.append(np.load(fpath))
        found_ids.append(dim_id)

    if len(vecs) < 2:
        log.warning(
            "Need at least 2 dimension vectors for cross-dim check, found %d.",
            len(vecs),
        )
        return None

    V = np.stack(vecs)                                       # (D, H)
    norms = np.linalg.norm(V, axis=-1, keepdims=True)
    V_n = V / np.where(norms == 0, 1.0, norms)
    sim_matrix = (V_n @ V_n.T).tolist()                      # (D, D)

    flagged = []
    for i in range(len(found_ids)):
        for j in range(i + 1, len(found_ids)):
            cos = sim_matrix[i][j]
            if abs(cos) > 0.5:
                flagged.append({
                    "dim_a": found_ids[i],
                    "dim_b": found_ids[j],
                    "cosine": round(cos, 4),
                })

    return {
        "dim_ids": found_ids,
        "matrix": [[round(v, 4) for v in row] for row in sim_matrix],
        "flagged_pairs": flagged,
    }


# =========================================================================
# Summary table
# =========================================================================

def _stability_label(cos: float) -> str:
    if cos >= 0.8:
        return "PASS"
    elif cos >= 0.6:
        return "WARN"
    else:
        return "FAIL"


def _separation_label(sil: float, svm: float) -> str:
    # Both must clear thresholds for PASS; either below lower threshold = FAIL
    if sil >= 0.3 and svm >= 0.7:
        return "PASS"
    elif sil < 0.1 or svm < 0.55:
        return "FAIL"
    else:
        return "WARN"


def print_summary(
    pca_results: Dict[str, Dict[int, Dict[str, float]]],
    stability_results: Dict[str, Dict[str, Dict[int, float]]],
    cross_dim_results: Dict[str, Optional[Dict]],
    layers: List[int],
    methods: List[str],
) -> None:
    """Print a human-readable summary table with PASS/WARN/FAIL verdicts."""

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # --- PCA Separation ---
    if pca_results:
        print("\n--- PCA Separation (silhouette > 0.3, SVM LOO acc > 0.7 = PASS) ---")
        header = f"{'dimension':<35s}"
        for layer in layers:
            header += f"  {'L' + str(layer):>8s}"
        print(header)
        print("-" * len(header))

        for dim_id, layer_dict in sorted(pca_results.items()):
            row = f"{dim_id:<35s}"
            for layer in layers:
                if layer in layer_dict:
                    sil = layer_dict[layer]["silhouette"]
                    svm = layer_dict[layer]["svm_acc"]
                    label = _separation_label(sil, svm)
                    row += f"  {label:>8s}"
                else:
                    row += f"  {'--':>8s}"
            print(row)

        # detail lines
        print()
        for dim_id, layer_dict in sorted(pca_results.items()):
            for layer in sorted(layer_dict):
                m = layer_dict[layer]
                label = _separation_label(m["silhouette"], m["svm_acc"])
                print(
                    f"  {dim_id} L{layer:02d}: "
                    f"silhouette={m['silhouette']:.3f}  "
                    f"SVM_LOO={m['svm_acc']:.3f}  "
                    f"[{label}]"
                )

    # --- Split-Half Stability ---
    if stability_results:
        print("\n--- Split-Half Stability (cosine > 0.8 = PASS, 0.6-0.8 = WARN) ---")
        for method in methods:
            print(f"\n  method: {method}")
            header = f"  {'dimension':<33s}"
            for layer in layers:
                header += f"  {'L' + str(layer):>8s}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for dim_id in sorted(stability_results):
                method_dict = stability_results[dim_id].get(method, {})
                row = f"  {dim_id:<33s}"
                for layer in layers:
                    if layer in method_dict:
                        cos = method_dict[layer]
                        label = _stability_label(cos)
                        row += f"  {cos:>5.3f} {label[0]}"
                    else:
                        row += f"  {'--':>8s}"
                print(row)

    # --- Cross-Dimension Similarity ---
    if cross_dim_results:
        print("\n--- Cross-Dimension Similarity (|cosine| > 0.5 flagged) ---")
        for key, result in sorted(cross_dim_results.items()):
            if result is None:
                print(f"  {key}: insufficient vectors")
                continue

            flagged = result.get("flagged_pairs", [])
            if flagged:
                for f in flagged:
                    print(
                        f"  {key}: {f['dim_a']} <-> {f['dim_b']}  "
                        f"cosine={f['cosine']:.4f}  [WARN]"
                    )
            else:
                print(f"  {key}: all pairs below 0.5  [PASS]")

    print("\n" + "=" * 80)
    print("Validation complete.")
    print("=" * 80 + "\n")


# =========================================================================
# Main orchestration
# =========================================================================

def load_model_and_tokenizer(model_key: str, use_quantization: bool = False):
    """Load a model and tokenizer from the MODELS registry."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    config = MODELS[model_key]
    token = None
    if config.requires_token:
        import os
        token = os.environ.get("HF_TOKEN", None)
        if token is None:
            raise RuntimeError(
                f"Model '{config.alias}' is gated and needs HF_TOKEN."
            )

    log.info("Loading tokenizer: %s", config.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        token=token,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model: %s", config.hf_id)
    if use_quantization:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            token=token,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            token=token,
            dtype=config.torch_dtype,
            device_map="auto",
        )
    model.eval()
    return model, tokenizer, config


def run_validation(args: argparse.Namespace) -> None:
    """Run all validation checks according to CLI arguments."""

    model_key = args.model
    if model_key not in MODELS:
        log.error("Unknown model '%s'. Known: %s", model_key, list(MODELS.keys()))
        sys.exit(1)

    model_config = MODELS[model_key]
    vectors_dir = Path(args.vectors_dir) / model_config.alias
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods
    layers = args.layers

    # ------------------------------------------------------------------
    # Discover dimensions from the pairs file or from saved vectors
    # ------------------------------------------------------------------
    dimensions = []
    dim_ids = []

    if not args.skip_extraction:
        pairs_path = Path(args.pairs_file)
        if not pairs_path.exists():
            log.error("Pairs file not found: %s", pairs_path)
            sys.exit(1)

        with open(pairs_path) as f:
            data = json.load(f)

        dimensions = data["dimensions"]

        if args.dimensions:
            requested = set(args.dimensions)
            dimensions = [d for d in dimensions if d["id"] in requested]
            missing = requested - {d["id"] for d in dimensions}
            if missing:
                log.warning("Unknown dimension IDs (skipping): %s", missing)

        dim_ids = [d["id"] for d in dimensions]
        log.info("Dimensions to validate: %s", dim_ids)
    else:
        # Discover dimensions from files on disk
        metadata_path = vectors_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            dim_ids = meta.get("dimensions", [])
        else:
            # Infer from filenames in the first available method directory
            for method in methods:
                method_dir = vectors_dir / method
                if method_dir.exists():
                    for p in method_dir.glob("*_all_layers.npy"):
                        did = p.stem.replace("_all_layers", "")
                        if did not in dim_ids:
                            dim_ids.append(did)
                    break

        if args.dimensions:
            dim_ids = [d for d in dim_ids if d in set(args.dimensions)]

        log.info("Dimensions discovered from disk: %s", dim_ids)

    if not dim_ids:
        log.error("No dimensions found to validate.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve which layers to check.
    # If not specified, read from metadata or use a default set.
    # ------------------------------------------------------------------
    if layers is None:
        metadata_path = vectors_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            n_layers = meta.get("n_layers", 36)
            # Pick ~5 representative layers spread across the model
            step = max(n_layers // 5, 1)
            layers = list(range(0, n_layers, step))
            if (n_layers - 1) not in layers:
                layers.append(n_layers - 1)
        else:
            layers = [8, 12, 16, 20, 24]

    log.info("Layers to validate: %s", layers)

    # ------------------------------------------------------------------
    # Extract activations if needed
    # ------------------------------------------------------------------
    # dim_id -> (pos_h, neg_h)  each (N, n_layers, H)
    activation_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if not args.skip_extraction:
        model, tokenizer, config = load_model_and_tokenizer(
            model_key, use_quantization=args.quantize,
        )

        for dim in dimensions:
            dim_id = dim["id"]
            pairs = dim["pairs"]
            log.info("Extracting activations for '%s' (%d pairs)...", dim_id, len(pairs))

            pos_texts = [
                format_sample(p["context"], p["positive"], tokenizer, config)
                for p in pairs
            ]
            neg_texts = [
                format_sample(p["context"], p["negative"], tokenizer, config)
                for p in pairs
            ]

            pos_h = extract_hidden_states(
                model, tokenizer, pos_texts, batch_size=args.batch_size,
            )
            neg_h = extract_hidden_states(
                model, tokenizer, neg_texts, batch_size=args.batch_size,
            )
            activation_cache[dim_id] = (pos_h, neg_h)

        # Free GPU memory
        import torch
        del model
        torch.cuda.empty_cache()
        log.info("Model unloaded.")

    # ==================================================================
    # Check 1: PCA Separation
    # ==================================================================
    pca_results: Dict[str, Dict[int, Dict[str, float]]] = {}

    if activation_cache:
        log.info("Running PCA separation check...")
        for dim_id, (pos_h, neg_h) in activation_cache.items():
            n_layers = pos_h.shape[1]
            valid_layers = [l for l in layers if 0 <= l < n_layers]
            pca_results[dim_id] = pca_separation_check(
                pos_h, neg_h, dim_id, valid_layers, output_dir,
            )
    else:
        log.info("Skipping PCA separation check (requires activations).")

    # ==================================================================
    # Check 2: Split-Half Stability
    # ==================================================================
    stability_results: Dict[str, Dict[str, Dict[int, float]]] = {}

    if activation_cache:
        log.info("Running split-half stability check...")
        for dim_id, (pos_h, neg_h) in activation_cache.items():
            n_layers = pos_h.shape[1]
            valid_layers = [l for l in layers if 0 <= l < n_layers]
            stability_results[dim_id] = split_half_stability(
                pos_h, neg_h, valid_layers, methods,
            )
    else:
        # Without activations we cannot do a proper split-half
        for dim_id in dim_ids:
            stability_results[dim_id] = split_half_stability_from_saved(
                vectors_dir, dim_id, layers, methods,
            )

    # ==================================================================
    # Check 3: Cross-Dimension Similarity
    # ==================================================================
    cross_dim_results: Dict[str, Optional[Dict]] = {}

    log.info("Running cross-dimension similarity check...")
    for method in methods:
        for layer in layers:
            key = f"{method}/layer{layer:02d}"
            result = cross_dimension_similarity(
                vectors_dir, method, layer, dim_ids,
            )
            cross_dim_results[key] = result

    # ==================================================================
    # Save results to JSON
    # ==================================================================
    output_data = {
        "model": model_key,
        "vectors_dir": str(vectors_dir),
        "layers": layers,
        "methods": methods,
        "pca_separation": {
            dim_id: {
                str(layer): metrics
                for layer, metrics in layer_dict.items()
            }
            for dim_id, layer_dict in pca_results.items()
        },
        "split_half_stability": {
            dim_id: {
                method: {
                    str(layer): cos
                    for layer, cos in layer_dict.items()
                }
                for method, layer_dict in method_dict.items()
            }
            for dim_id, method_dict in stability_results.items()
        },
        "cross_dimension_similarity": {
            key: val for key, val in cross_dim_results.items()
        },
    }

    json_path = output_dir / "validation_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info("Results saved to %s", json_path)

    # Save each cross-dim similarity matrix as its own file
    for key, result in cross_dim_results.items():
        if result is not None:
            safe_key = key.replace("/", "_")
            matrix_path = output_dir / f"cross_dim_{safe_key}.json"
            with open(matrix_path, "w") as f:
                json.dump(result, f, indent=2)

    # ==================================================================
    # Summary
    # ==================================================================
    print_summary(
        pca_results,
        stability_results,
        cross_dim_results,
        layers,
        methods,
    )


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate extracted steering vectors. "
            "Checks separation quality, split-half stability, and "
            "cross-dimension redundancy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model alias (must match a key in extract_vectors.MODELS).",
    )
    p.add_argument(
        "--pairs_file",
        default="negotiation_steering_pairs.json",
        help="Path to the contrastive pairs JSON (default: negotiation_steering_pairs.json).",
    )
    p.add_argument(
        "--vectors_dir",
        default="vectors",
        help="Root vectors directory (default: vectors/). Model alias subdirectory is appended.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Layer indices to validate (default: auto-selected from metadata).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["mean_diff", "pca"],
        help="Which extraction methods to validate (default: mean_diff pca).",
    )
    p.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        metavar="DIM",
        help="Only validate these dimension IDs. Default: all available.",
    )
    p.add_argument(
        "--output_dir",
        default="results/validation",
        help="Directory for output plots and JSON (default: results/validation).",
    )
    p.add_argument(
        "--skip_extraction",
        action="store_true",
        help=(
            "Skip re-extracting activations (CPU mode). "
            "Only cross-dimension similarity will run. "
            "Split-half and PCA separation require activations."
        ),
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Load model in 4-bit NF4 (requires bitsandbytes). Ignored with --skip_extraction.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size for activation extraction (default: 4).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(args)
