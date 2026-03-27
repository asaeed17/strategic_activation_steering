"""
Orthogonal projection experiment: remove control-dimension components from
negotiation steering vectors and measure what remains.

Phase 1 (CPU, default): vector-level analysis — residual norms, cosine changes.
Phase 2 (GPU, --probe): load model, extract hidden states, compare 1-D probe
accuracy before/after projection.

Usage:
    python orthogonal_projection.py --variant neg8dim_12pairs_matched
    python orthogonal_projection.py --variant neg8dim_12pairs_matched --probe
    python orthogonal_projection.py --all-variants
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONTROL_DIMS = ["verbosity", "formality", "hedging", "sentiment", "specificity"]
MODEL_ALIAS = "qwen2.5-3b"  # default; overridden by --model


# ── Vector operations ───────────────────────────────────────────────────

def load_all_layers(vectors_dir: Path, dim_id: str) -> np.ndarray:
    """Load (n_layers, hidden_dim) vector for a dimension."""
    path = vectors_dir / f"{dim_id}_all_layers.npy"
    return np.load(path)


def orthogonalize(v: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """
    Project out control directions from v using Gram-Schmidt.

    v:        (hidden_dim,) — negotiation direction at one layer
    controls: (n_controls, hidden_dim) — control directions at same layer

    Returns cleaned vector (not re-normalized).
    """
    v_clean = v.copy()
    for c in controls:
        c_hat = c / (np.linalg.norm(c) + 1e-10)
        v_clean -= np.dot(v_clean, c_hat) * c_hat
    return v_clean


def run_vector_analysis(neg_dir: Path, ctrl_dir: Path, method: str = "mean_diff"):
    """Phase 1: vector-level analysis (CPU only)."""
    method_dir = neg_dir / MODEL_ALIAS / method
    ctrl_method_dir = ctrl_dir / MODEL_ALIAS / method

    # Discover negotiation dimensions
    neg_dims = sorted(set(
        f.stem.replace("_all_layers", "")
        for f in method_dir.glob("*_all_layers.npy")
    ))
    log.info("Negotiation dims: %s", neg_dims)
    log.info("Control dims: %s", CONTROL_DIMS)

    # Load control vectors
    ctrl_vecs = {}
    for cd in CONTROL_DIMS:
        ctrl_vecs[cd] = load_all_layers(ctrl_method_dir, cd)
    n_layers = ctrl_vecs[CONTROL_DIMS[0]].shape[0]

    results = {}
    for nd in neg_dims:
        v_orig = load_all_layers(method_dir, nd)  # (n_layers, H)

        residual_norms = []
        cosine_before = {cd: [] for cd in CONTROL_DIMS}
        cosine_after = {cd: [] for cd in CONTROL_DIMS}

        v_clean_all = np.zeros_like(v_orig)

        for l in range(n_layers):
            # Stack control directions at this layer
            C = np.stack([ctrl_vecs[cd][l] for cd in CONTROL_DIMS])
            v_l = v_orig[l]

            # Cosines before
            for i, cd in enumerate(CONTROL_DIMS):
                cos = np.dot(v_l, C[i]) / (np.linalg.norm(v_l) * np.linalg.norm(C[i]) + 1e-10)
                cosine_before[cd].append(float(cos))

            # Project out
            v_c = orthogonalize(v_l, C)
            norm_ratio = np.linalg.norm(v_c) / (np.linalg.norm(v_l) + 1e-10)
            residual_norms.append(float(norm_ratio))

            # Re-normalize
            n = np.linalg.norm(v_c)
            v_clean_all[l] = v_c / n if n > 1e-10 else v_c

            # Cosines after
            for i, cd in enumerate(CONTROL_DIMS):
                cos = np.dot(v_clean_all[l], C[i]) / (np.linalg.norm(v_clean_all[l]) * np.linalg.norm(C[i]) + 1e-10)
                cosine_after[cd].append(float(cos))

        results[nd] = {
            "residual_norm_mean": float(np.mean(residual_norms)),
            "residual_norm_std": float(np.std(residual_norms)),
            "residual_norm_per_layer": residual_norms,
            "cosine_before": {cd: float(np.mean(np.abs(cosine_before[cd]))) for cd in CONTROL_DIMS},
            "cosine_after": {cd: float(np.mean(np.abs(cosine_after[cd]))) for cd in CONTROL_DIMS},
            "v_clean_all": v_clean_all,  # kept in memory for optional probe phase
        }

    return results, neg_dims, ctrl_vecs


def run_probe_comparison(variant: str, neg_dims, ctrl_vecs, vector_results, method: str = "mean_diff"):
    """Phase 2: load model, extract hidden states, compare 1-D probe accuracy."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load steering pairs (try both naming conventions)
    pairs_path = Path(f"steering_pairs/{variant}/negotiation_steering_pairs.json")
    if not pairs_path.exists():
        pairs_path = Path(f"steering_pairs/{variant}/ultimatum_steering_pairs.json")
    with open(pairs_path) as f:
        neg_data = json.load(f)

    from extract_vectors import MODELS, HF_TOKEN
    cfg = MODELS[MODEL_ALIAS]
    token = HF_TOKEN if cfg.requires_token else None

    log.info("Loading model: %s", cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, token=token, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers

    def extract_hidden(texts, batch_size=4):
        all_h = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # Last token hidden states, all layers
            for j in range(len(batch)):
                layers_h = []
                for l in range(n_layers):
                    h = out.hidden_states[l + 1][j, -1, :].float().cpu().numpy()
                    layers_h.append(h)
                all_h.append(np.stack(layers_h))
        return np.stack(all_h)  # (n_samples, n_layers, H)

    probe_results = {}
    for dim_data in neg_data["dimensions"]:
        dim_id = dim_data["id"]
        if dim_id not in vector_results:
            continue

        log.info("Probing: %s", dim_id)
        pos_texts = [p["positive"] for p in dim_data["pairs"]]
        neg_texts = [p["negative"] for p in dim_data["pairs"]]

        pos_h = extract_hidden(pos_texts)
        neg_h = extract_hidden(neg_texts)

        # Load original and cleaned vectors
        method_dir = Path(f"vectors/{variant}/negotiation/{MODEL_ALIAS}/{method}")
        v_orig = load_all_layers(method_dir, dim_id)
        v_clean = vector_results[dim_id]["v_clean_all"]

        orig_accs = []
        clean_accs = []

        for l in range(n_layers):
            X_pos = pos_h[:, l, :]
            X_neg = neg_h[:, l, :]
            X = np.concatenate([X_pos, X_neg], axis=0)
            y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

            # 1-D probe: project onto direction, threshold
            proj_orig = (X @ v_orig[l]).reshape(-1, 1)
            proj_clean = (X @ v_clean[l]).reshape(-1, 1)

            cv = StratifiedKFold(n_splits=min(5, len(X_pos)), shuffle=True, random_state=42)

            def cv_acc(proj):
                accs = []
                for train_idx, test_idx in cv.split(proj, y):
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(proj[train_idx], y[train_idx])
                    accs.append(lr.score(proj[test_idx], y[test_idx]))
                return float(np.mean(accs))

            orig_accs.append(cv_acc(proj_orig))
            clean_accs.append(cv_acc(proj_clean))

        probe_results[dim_id] = {
            "orig_1d_mean": float(np.mean(orig_accs)),
            "clean_1d_mean": float(np.mean(clean_accs)),
            "orig_1d_per_layer": orig_accs,
            "clean_1d_per_layer": clean_accs,
            "accuracy_drop": float(np.mean(orig_accs) - np.mean(clean_accs)),
        }
        log.info("  %s: orig=%.3f  clean=%.3f  drop=%.3f",
                 dim_id, probe_results[dim_id]["orig_1d_mean"],
                 probe_results[dim_id]["clean_1d_mean"],
                 probe_results[dim_id]["accuracy_drop"])

    del model
    torch.cuda.empty_cache()
    return probe_results


# ── Reporting ───────────────────────────────────────────────────────────

def print_report(variant, vector_results, probe_results=None):
    print(f"\n{'=' * 70}")
    print(f"  ORTHOGONAL PROJECTION REPORT — {variant}")
    print(f"{'=' * 70}")

    print(f"\n  Controls projected out: {', '.join(CONTROL_DIMS)}")

    # Phase 1: residual norms
    print(f"\n{'=' * 70}")
    print(f"  RESIDUAL NORM (fraction of original vector surviving projection)")
    print(f"{'=' * 70}")
    print(f"  {'Dimension':<30} {'Mean':>8} {'Std':>8} {'Interpretation'}")
    print(f"  {'-' * 66}")

    for nd in sorted(vector_results.keys()):
        r = vector_results[nd]
        mean = r["residual_norm_mean"]
        std = r["residual_norm_std"]
        if mean < 0.3:
            interp = "MOSTLY SURFACE"
        elif mean < 0.6:
            interp = "MIXED"
        elif mean < 0.8:
            interp = "MOSTLY GENUINE"
        else:
            interp = "INDEPENDENT"
        print(f"  {nd:<30} {mean:>8.3f} {std:>8.3f}   {interp}")

    # Cosine before/after
    print(f"\n{'=' * 70}")
    print(f"  MEAN |COSINE| WITH CONTROLS (before -> after projection)")
    print(f"{'=' * 70}")
    header = f"  {'Dimension':<25}"
    for cd in CONTROL_DIMS:
        header += f" {cd[:8]:>10}"
    print(header)
    print(f"  {'-' * (25 + 10 * len(CONTROL_DIMS))}")

    for nd in sorted(vector_results.keys()):
        r = vector_results[nd]
        line_before = f"  {nd:<25}"
        line_after = f"  {'  (after)':<25}"
        for cd in CONTROL_DIMS:
            line_before += f" {r['cosine_before'][cd]:>10.3f}"
            line_after += f" {r['cosine_after'][cd]:>10.3f}"
        print(line_before)
        print(line_after)

    # Phase 2: probe comparison
    if probe_results:
        print(f"\n{'=' * 70}")
        print(f"  1-D PROBE ACCURACY (original vs cleaned direction)")
        print(f"{'=' * 70}")
        print(f"  {'Dimension':<30} {'Original':>10} {'Cleaned':>10} {'Drop':>10} {'Interpretation'}")
        print(f"  {'-' * 76}")

        for nd in sorted(probe_results.keys()):
            p = probe_results[nd]
            drop = p["accuracy_drop"]
            if drop > 0.15:
                interp = "SURFACE-DOMINATED"
            elif drop > 0.05:
                interp = "PARTIAL SURFACE"
            elif drop > -0.02:
                interp = "GENUINE SIGNAL"
            else:
                interp = "IMPROVED (noise removed)"
            print(f"  {nd:<30} {p['orig_1d_mean']:>10.3f} {p['clean_1d_mean']:>10.3f} {drop:>+10.3f}   {interp}")

    print(f"\n{'=' * 70}\n")


def save_results(output_dir, variant, vector_results, probe_results=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable arrays
    serializable = {}
    for nd, r in vector_results.items():
        serializable[nd] = {k: v for k, v in r.items() if k != "v_clean_all"}

    out = {
        "variant": variant,
        "model": MODEL_ALIAS,
        "controls_projected": CONTROL_DIMS,
        "vector_analysis": serializable,
    }
    if probe_results:
        out["probe_comparison"] = probe_results

    with open(output_dir / "orthogonal_projection.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved to %s", output_dir / "orthogonal_projection.json")


# ── Main ────────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    "neg15dim_12pairs_raw", "neg15dim_12pairs_matched",
    "neg15dim_20pairs_matched", "neg15dim_12pairs_matched_V2",
    "neg15dim_80pairs_matched",
    "neg8dim_12pairs_raw", "neg8dim_12pairs_matched",
    "neg8dim_20pairs_matched", "neg8dim_80pairs_matched",
    "neg8dim_12pairs_matched_V2",
]


def process_variant(variant, do_probe=False, method="mean_diff"):
    neg_dir = Path(f"vectors/{variant}/negotiation")
    ctrl_dir = Path(f"vectors/{variant}/control")
    output_dir = Path(f"results/projection/{variant}/{method}")

    if not neg_dir.exists() or not ctrl_dir.exists():
        log.warning("Skipping %s — vectors not found", variant)
        return

    log.info("Processing: %s (method=%s)", variant, method)
    vector_results, neg_dims, ctrl_vecs = run_vector_analysis(neg_dir, ctrl_dir, method=method)

    probe_results = None
    if do_probe:
        probe_results = run_probe_comparison(variant, neg_dims, ctrl_vecs, vector_results, method=method)

    print_report(variant, vector_results, probe_results)
    save_results(output_dir, variant, vector_results, probe_results)


def main():
    parser = argparse.ArgumentParser(description="Orthogonal projection experiment")
    parser.add_argument("--variant", type=str, help="Single variant to process")
    parser.add_argument("--all-variants", action="store_true", help="Process all 8 variants")
    parser.add_argument("--probe", action="store_true",
                        help="Also run 1-D probe comparison (requires GPU)")
    parser.add_argument("--method", type=str, default="mean_diff",
                        choices=["mean_diff", "pca", "logreg"],
                        help="Extraction method to use (default: mean_diff)")
    parser.add_argument("--output-dir", type=str, default="results/projection")
    parser.add_argument("--model", type=str, default=None,
                        help="Model alias (default: qwen2.5-3b). E.g. qwen2.5-7b")
    args = parser.parse_args()

    if args.model:
        global MODEL_ALIAS
        MODEL_ALIAS = args.model

    if not args.variant and not args.all_variants:
        parser.error("Specify --variant or --all-variants")

    variants = ALL_VARIANTS if args.all_variants else [args.variant]

    for v in variants:
        process_variant(v, do_probe=args.probe, method=args.method)


if __name__ == "__main__":
    main()
