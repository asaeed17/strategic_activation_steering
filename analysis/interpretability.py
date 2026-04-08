#!/usr/bin/env python3
"""
Interpretability analyses for activation steering vectors.

The main geometric analyses are CPU-only and operate entirely on saved
steering vectors plus behavioral JSON outputs. An optional `logit_lens`
mode loads the base model to project saved vectors through the final norm
and unembedding matrix.

Outputs:
  - results/interpretability/logit_lens_results.json
  - results/interpretability/cosine_evolution.json
  - results/interpretability/pca_analysis.json
  - results/interpretability/contamination_analysis.json
  - results/figures/fig9_cosine_heatmaps.{png,pdf}
  - results/figures/fig10_cosine_evolution.{png,pdf}
  - results/figures/fig11_pca_steering_space.{png,pdf}
  - results/figures/fig12_contamination_vs_effect.{png,pdf}
  - results/figures/fig13_logit_lens_heatmap.{png,pdf}
  - results/figures/fig14_logit_lens_top_tokens.{png,pdf}
  - mirrored copies under results/interpretability/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VARIANT = "ultimatum_10dim_20pairs_general_matched"
DEFAULT_MODEL = "qwen2.5-7b"
DEFAULT_METHOD = "mean_diff"
DEFAULT_ANALYSIS = "all"

DIMENSIONS = [
    "firmness",
    "empathy",
    "anchoring",
    "fairness_norm",
    "narcissism",
    "spite",
    "greed",
    "flattery",
    "composure",
    "undecidedness",
]
CONTROL_DIMENSIONS = [
    "formality",
    "hedging",
    "sentiment",
    "specificity",
    "verbosity",
]
DIM_LABELS = {
    "firmness": "Firmness",
    "empathy": "Empathy",
    "anchoring": "Anchoring",
    "fairness_norm": "Fairness norm",
    "narcissism": "Narcissism",
    "spite": "Spite",
    "greed": "Greed",
    "flattery": "Flattery",
    "composure": "Composure",
    "undecidedness": "Undecidedness",
}
HEATMAP_LAYERS = [4, 10, 12, 20]
TRACKED_PAIRS = [
    ("firmness", "empathy"),
    ("greed", "narcissism"),
    ("empathy", "flattery"),
    ("firmness", "undecidedness"),
]
BEHAVIOR_LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
BEHAVIOR_ALPHA = 7.0
LOGIT_LENS_LAYERS = [4, 8, 10, 12, 16, 20]
LOGIT_TOP_DIMENSIONS = ["firmness", "greed", "empathy", "fairness_norm"]
PEAK_LAYERS = {
    "firmness": 10,
    "greed": 12,
    "empathy": 10,
    "fairness_norm": 4,
}
LOGIT_MODEL_SPECS = {
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "requires_token": False,
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "requires_token": False,
    },
}
TOKEN_BLOCK_RE = re.compile(r"^[<\x00-\x1f]")
WORDLIKE_RE = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")
NUMERIC_RE = re.compile(r"^\$?\d+(?:[.,]\d+)?%?$")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
BAD_PUNCT_RE = re.compile(r"[\[\]\{\}\(\)\\/@#;:=`|]")
CAMEL_RE = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")

RESULTS_DIR = ROOT / "results" / "interpretability"
FIGURES_DIR = ROOT / "results" / "figures"
VECTORS_ROOT = ROOT / "vectors"
BEHAVIOR_RESULTS_DIR = ROOT / "results" / "ultimatum" / "final_7b_llm_vs_llm"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=["logit_lens", "cosine_evolution", "pca_analysis", "contamination_analysis", "all"],
        default=DEFAULT_ANALYSIS,
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--behavior-results-dir", default=str(BEHAVIOR_RESULTS_DIR))
    return parser.parse_args()


def ensure_dirs(results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)


def get_figure_output_dirs(results_dir: Path, figures_dir: Path) -> List[Path]:
    dirs = [figures_dir, results_dir / "figures"]
    unique_dirs: List[Path] = []
    seen = set()
    for path in dirs:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(path)
    return unique_dirs


def save_figure(fig: plt.Figure, figure_dirs: Sequence[Path], stem: str) -> None:
    for out_dir in figure_dirs:
        for ext in ("png", "pdf"):
            out_path = out_dir / f"{stem}.{ext}"
            fig.savefig(out_path)
            log.info("Saved %s", out_path)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Saved %s", path)


def resolve_local_model_source(model_alias: str) -> Optional[Path]:
    hf_home = os.environ.get("HF_HOME")
    roots = [Path(hf_home)] if hf_home else []
    roots.append(ROOT / ".hf_cache")

    model_dir_name = None
    if model_alias == "qwen2.5-7b":
        model_dir_name = "models--Qwen--Qwen2.5-7B-Instruct"
    elif model_alias == "qwen2.5-3b":
        model_dir_name = "models--Qwen--Qwen2.5-3B-Instruct"

    if model_dir_name is None:
        return None

    for root in roots:
        snapshot_root = root / "hub" / model_dir_name / "snapshots"
        if snapshot_root.exists():
            snapshots = sorted([p for p in snapshot_root.iterdir() if p.is_dir()])
            if snapshots:
                return snapshots[-1]
    return None


def load_model_and_tokenizer(model_alias: str, dtype_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "logit_lens requires the project venv with torch and transformers installed."
        ) from exc

    if model_alias not in LOGIT_MODEL_SPECS:
        raise ValueError(
            f"logit_lens currently supports {sorted(LOGIT_MODEL_SPECS)}; got {model_alias!r}"
        )

    spec = LOGIT_MODEL_SPECS[model_alias]
    hf_token = os.environ.get("HF_TOKEN")
    token = hf_token if spec["requires_token"] else None
    local_source = resolve_local_model_source(model_alias)
    model_source = local_source or spec["hf_id"]
    offline = (
        local_source is not None
        or os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_source),
        token=token,
        padding_side="left",
        local_files_only=offline,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto"}
    if token is not None:
        load_kwargs["token"] = token
    if offline:
        load_kwargs["local_files_only"] = True
    load_kwargs["torch_dtype"] = dtype_map[dtype_name]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_source),
            **load_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_source),
            attn_implementation="eager",
            **load_kwargs,
        )

    model.eval()
    return model, tokenizer


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def get_final_norm(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise AttributeError("Could not find final normalization module on model.")


def get_output_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("Could not find output head on model.")


def token_displayable(token: str) -> bool:
    if not token or token.isspace():
        return False
    if TOKEN_BLOCK_RE.match(token):
        return False
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in token):
        return False
    return True


def is_semantic_token(token: str) -> bool:
    stripped = token.strip()
    if not token_displayable(stripped):
        return False
    if BAD_PUNCT_RE.search(stripped):
        return False
    if stripped.startswith((".", "_", "-", "+")):
        return False
    if stripped.count("$") > 1:
        return False
    if stripped == "$" or (stripped.startswith("$") and NUMERIC_RE.fullmatch(stripped)):
        return True
    if stripped.isupper() and len(stripped) > 4:
        return False
    if CAMEL_RE.search(stripped):
        return False
    if any(ch.isdigit() for ch in stripped) and not (NUMERIC_RE.fullmatch(stripped) or ORDINAL_RE.fullmatch(stripped)):
        return False
    if NUMERIC_RE.fullmatch(stripped) or ORDINAL_RE.fullmatch(stripped):
        return True
    if not WORDLIKE_RE.fullmatch(stripped):
        return False
    lowered = stripped.lower()
    vowels = set("aeiou")
    if len(lowered) < 2:
        return False
    if len(lowered) < 4:
        return any(ch in vowels for ch in lowered)
    return any(ch in vowels for ch in lowered)


def clean_token_for_display(token: str) -> Optional[str]:
    if not token_displayable(token):
        return None
    token = token.strip()
    if not token:
        return None
    return token[:24]


def collect_ranked_tokens(logits, tokenizer, top_k: int, largest: bool) -> List[Dict[str, float]]:
    ordered_ids = logits.argsort(descending=largest).tolist()
    tokens: List[Dict[str, float]] = []
    seen = set()

    def maybe_append(token_id: int, raw_token: str, bucket: List[Dict[str, float]]) -> bool:
        display = clean_token_for_display(raw_token)
        if display is None:
            return False
        key = display.lower()
        if key in seen:
            return False
        score = float(logits[token_id].item())
        if not np.isfinite(score):
            return False
        bucket.append({
            "token_id": int(token_id),
            "token": display,
            "raw_token": raw_token,
            "logit": score,
        })
        seen.add(key)
        return len(bucket) >= top_k

    for token_id in ordered_ids:
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if not is_semantic_token(token):
            continue
        if maybe_append(token_id, token, tokens):
            break

    if len(tokens) < top_k:
        for token_id in ordered_ids:
            token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if maybe_append(token_id, token, tokens):
                break
    return tokens


def cosine_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def safe_correlation(xs: Sequence[float], ys: Sequence[float], fn) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(fn(x, y).statistic)


def load_all_vectors(
    vectors_root: Path,
    variant: str,
    space: str,
    model_alias: str,
    method: str,
    dimensions: Sequence[str],
) -> Dict[str, np.ndarray]:
    base = vectors_root / variant / space / model_alias / method
    vectors: Dict[str, np.ndarray] = {}
    for dim in dimensions:
        path = base / f"{dim}_all_layers.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing vector file: {path}")
        arr = np.load(path)
        if arr.shape != (28, 3584):
            raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
        vectors[dim] = arr.astype(np.float64)
    return vectors


def alpha_to_suffix(alpha: float) -> str:
    return str(float(alpha))


def load_behavioral_results(
    results_dir: Path,
    dimensions: Sequence[str],
    layers: Sequence[int],
    alpha: float,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {dim: {} for dim in dimensions}
    alpha_suffix = alpha_to_suffix(alpha)
    for dim in dimensions:
        for layer in layers:
            path = results_dir / f"{dim}_proposer_L{layer}_a{alpha_suffix}_paired_n50.json"
            if not path.exists():
                log.warning("Missing behavioral result: %s", path)
                out[dim][layer] = {
                    "cohens_d": np.nan,
                    "delta_proposer_pct": np.nan,
                    "steered_accept_rate": np.nan,
                    "baseline_accept_rate": np.nan,
                }
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            out[dim][layer] = {
                "cohens_d": float(summary.get("cohens_d", np.nan)),
                "delta_proposer_pct": float(summary.get("delta_proposer_pct", np.nan)),
                "steered_accept_rate": float(summary.get("steered_accept_rate", np.nan)),
                "baseline_accept_rate": float(summary.get("baseline_accept_rate", np.nan)),
            }
    return out


def run_logit_lens(
    model,
    tokenizer,
    neg_vecs: Dict[str, np.ndarray],
    layers: Sequence[int],
    top_k: int,
    model_alias: str,
    method: str,
) -> Dict:
    import torch

    final_norm = get_final_norm(model)
    lm_head = get_output_head(model)
    device = get_model_device(model)
    dtype = next(model.parameters()).dtype

    results = {
        "metadata": {
            "model": model_alias,
            "method": method,
            "layers": list(layers),
            "top_k": top_k,
            "caveat": (
                "The final normalization is applied to a standalone vector rather than "
                "an in-context residual stream state, so logit-lens results are approximate."
            ),
        },
        "dimensions": {},
    }

    with torch.no_grad():
        for dim in DIMENSIONS:
            dim_payload = {}
            for layer in layers:
                vec = torch.tensor(neg_vecs[dim][layer], dtype=dtype, device=device)
                vec = vec.unsqueeze(0).unsqueeze(0)
                normed = final_norm(vec).squeeze(0).squeeze(0)
                logits = lm_head(normed).float().cpu()
                top_tokens = collect_ranked_tokens(logits, tokenizer, top_k=top_k, largest=True)
                bottom_tokens = collect_ranked_tokens(logits, tokenizer, top_k=top_k, largest=False)
                top1 = top_tokens[0] if top_tokens else None
                dim_payload[str(layer)] = {
                    "top_tokens": top_tokens,
                    "bottom_tokens": bottom_tokens,
                    "top_token": top1["token"] if top1 else None,
                    "top_logit": top1["logit"] if top1 else None,
                }
            results["dimensions"][dim] = dim_payload
    return results


def plot_logit_lens_heatmap(results: Dict, figure_dirs: Sequence[Path]) -> None:
    heat = []
    labels = []
    ann = []
    layers = results["metadata"]["layers"]
    for dim in DIMENSIONS:
        row_vals = []
        row_ann = []
        for layer in layers:
            entry = results["dimensions"][dim][str(layer)]
            row_vals.append(entry["top_logit"] if entry["top_logit"] is not None else np.nan)
            row_ann.append(entry["top_token"] or "")
        heat.append(row_vals)
        ann.append(row_ann)
        labels.append(DIM_LABELS[dim])

    fig, ax = plt.subplots(figsize=(10.5, 7))
    sns.heatmap(
        np.asarray(heat, dtype=float),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=np.asarray(ann, dtype=object),
        fmt="",
        xticklabels=layers,
        yticklabels=labels,
        cbar_kws={"label": "Top-token logit"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("")
    ax.set_title("Logit Lens: Top Promoted Token by Dimension and Layer")
    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig13_logit_lens_heatmap")
    plt.close(fig)


def plot_logit_lens_top_tokens(results: Dict, figure_dirs: Sequence[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dim in zip(axes.flat, LOGIT_TOP_DIMENSIONS):
        layer = PEAK_LAYERS[dim]
        entry = results["dimensions"][dim][str(layer)]
        top_tokens = entry["top_tokens"][:10]
        bottom_tokens = entry["bottom_tokens"][:10]

        tokens = [tok["token"] for tok in bottom_tokens[::-1]] + [tok["token"] for tok in top_tokens]
        values = [tok["logit"] for tok in bottom_tokens[::-1]] + [tok["logit"] for tok in top_tokens]
        colors = ["#4C78A8"] * len(bottom_tokens) + ["#E45756"] * len(top_tokens)
        ypos = np.arange(len(tokens))
        nonzero = [abs(v) for v in values if np.isfinite(v) and abs(v) > 0]
        linthresh = max(1.0, np.percentile(nonzero, 25)) if nonzero else 1.0

        ax.barh(ypos, values, color=colors, alpha=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels(tokens)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_title(f"{DIM_LABELS[dim]}  (Layer {layer})")
        ax.set_xlabel("Logit")
    fig.suptitle("Logit Lens: Top Promoted and Suppressed Tokens", y=1.02)
    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig14_logit_lens_top_tokens")
    plt.close(fig)


def run_cosine_evolution(neg_vecs: Dict[str, np.ndarray]) -> Dict:
    layer_matrices: Dict[str, List[List[float]]] = {}
    tracked_pairs: Dict[str, List[float]] = {}

    for layer in range(28):
        layer_stack = np.stack([neg_vecs[dim][layer] for dim in DIMENSIONS], axis=0)
        layer_matrices[str(layer)] = cosine_similarity(layer_stack).tolist()

    for dim_a, dim_b in TRACKED_PAIRS:
        key = f"{dim_a}__{dim_b}"
        tracked_pairs[key] = []
        a_idx = DIMENSIONS.index(dim_a)
        b_idx = DIMENSIONS.index(dim_b)
        for layer in range(28):
            tracked_pairs[key].append(layer_matrices[str(layer)][a_idx][b_idx])

    return {
        "metadata": {
            "dimensions": DIMENSIONS,
            "heatmap_layers": HEATMAP_LAYERS,
            "tracked_pairs": [list(pair) for pair in TRACKED_PAIRS],
        },
        "layer_matrices": layer_matrices,
        "tracked_pairs": tracked_pairs,
        "sanity_checks": {
            "firmness_empathy_L10": layer_matrices["10"][DIMENSIONS.index("firmness")][DIMENSIONS.index("empathy")],
            "greed_narcissism_L10": layer_matrices["10"][DIMENSIONS.index("greed")][DIMENSIONS.index("narcissism")],
        },
    }


def run_pca_analysis(neg_vecs: Dict[str, np.ndarray]) -> Dict:
    per_layer = []
    layer10_coords = []
    layer10_var_ratio = []

    for layer in range(28):
        layer_stack = np.stack([neg_vecs[dim][layer] for dim in DIMENSIONS], axis=0)
        pca = PCA(n_components=min(len(DIMENSIONS), layer_stack.shape[0]))
        coords = pca.fit_transform(layer_stack)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        per_layer.append({
            "layer": layer,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": cumvar.tolist(),
            "n_components_for_90pct": n_90,
        })

        if layer == 10:
            layer10_var_ratio = pca.explained_variance_ratio_[:2].tolist()
            for idx, dim in enumerate(DIMENSIONS):
                layer10_coords.append({
                    "dimension": dim,
                    "pc1": float(coords[idx, 0]),
                    "pc2": float(coords[idx, 1]),
                })

    return {
        "metadata": {
            "dimensions": DIMENSIONS,
            "scatter_layer": 10,
        },
        "per_layer": per_layer,
        "layer10_scatter": layer10_coords,
        "layer10_pc1_pc2_var_ratio": layer10_var_ratio,
    }


def run_contamination_analysis(
    neg_vecs: Dict[str, np.ndarray],
    ctrl_vecs: Dict[str, np.ndarray],
    behavioral: Dict[str, Dict[int, Dict[str, float]]],
) -> Dict:
    contamination_matrix = []
    records = []

    for dim in DIMENSIONS:
        row = []
        for layer in BEHAVIOR_LAYERS:
            neg_vec = neg_vecs[dim][layer]
            control_scores = {
                ctrl_dim: cosine_between(neg_vec, ctrl_vecs[ctrl_dim][layer])
                for ctrl_dim in CONTROL_DIMENSIONS
            }
            max_ctrl_dim, max_ctrl_cos = max(
                control_scores.items(),
                key=lambda item: abs(item[1]),
            )
            contamination = abs(max_ctrl_cos)
            row.append(contamination)

            behavior = behavioral[dim][layer]
            records.append({
                "dimension": dim,
                "layer": layer,
                "max_control_contamination": contamination,
                "max_control_dimension": max_ctrl_dim,
                "max_control_cosine": float(max_ctrl_cos),
                "control_cosines": control_scores,
                "cohens_d": behavior["cohens_d"],
                "abs_cohens_d": abs(behavior["cohens_d"]) if np.isfinite(behavior["cohens_d"]) else np.nan,
                "delta_proposer_pct": behavior["delta_proposer_pct"],
                "steered_accept_rate": behavior["steered_accept_rate"],
                "baseline_accept_rate": behavior["baseline_accept_rate"],
            })
        contamination_matrix.append(row)

    x = [record["max_control_contamination"] for record in records]
    y = [record["abs_cohens_d"] for record in records]
    pearson = safe_correlation(x, y, pearsonr)
    spearman = safe_correlation(x, y, spearmanr)

    highest = max(records, key=lambda record: record["max_control_contamination"])
    return {
        "metadata": {
            "dimensions": DIMENSIONS,
            "control_dimensions": CONTROL_DIMENSIONS,
            "layers": BEHAVIOR_LAYERS,
            "alpha": BEHAVIOR_ALPHA,
        },
        "contamination_matrix": contamination_matrix,
        "records": records,
        "summary": {
            "pearson_r_contamination_vs_abs_cohens_d": pearson,
            "spearman_r_contamination_vs_abs_cohens_d": spearman,
            "highest_contamination_record": highest,
        },
    }


def plot_cosine_heatmaps(results: Dict, figure_dirs: Sequence[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, layer in zip(axes.flat, HEATMAP_LAYERS):
        mat = np.asarray(results["layer_matrices"][str(layer)], dtype=float)
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            xticklabels=[DIM_LABELS[d] for d in DIMENSIONS],
            yticklabels=[DIM_LABELS[d] for d in DIMENSIONS],
            cbar=ax is axes.flat[0],
            cbar_kws={"shrink": 0.8, "label": "Cosine similarity"},
        )
        ax.set_title(f"Layer {layer}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Inter-Dimension Cosine Similarity Across Layers", y=0.98)
    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig9_cosine_heatmaps")
    plt.close(fig)


def plot_cosine_evolution(results: Dict, figure_dirs: Sequence[Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(28))
    palette = sns.color_palette("tab10", n_colors=len(TRACKED_PAIRS))
    for color, (dim_a, dim_b) in zip(palette, TRACKED_PAIRS):
        key = f"{dim_a}__{dim_b}"
        ax.plot(
            layers,
            results["tracked_pairs"][key],
            linewidth=2.2,
            label=f"{DIM_LABELS[dim_a]} ↔ {DIM_LABELS[dim_b]}",
            color=color,
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine Relationship Evolution Across Network Depth")
    ax.set_xlim(0, 27)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig10_cosine_evolution")
    plt.close(fig)


def plot_pca_steering_space(results: Dict, figure_dirs: Sequence[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n90 = [entry["n_components_for_90pct"] for entry in results["per_layer"]]
    axes[0].plot(range(28), n90, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Components for 90% variance")
    axes[0].set_title("Effective Dimensionality of Steering Space")
    axes[0].set_xlim(0, 27)
    axes[0].set_ylim(1, max(n90) + 0.5)

    palette = sns.color_palette("tab10", n_colors=len(DIMENSIONS))
    var_ratio = results["layer10_pc1_pc2_var_ratio"]
    for color, point in zip(palette, results["layer10_scatter"]):
        axes[1].scatter(point["pc1"], point["pc2"], s=80, color=color)
        axes[1].text(point["pc1"], point["pc2"], f" {DIM_LABELS[point['dimension']]}", va="center")
    xlab = "PC1"
    ylab = "PC2"
    if len(var_ratio) >= 2:
        xlab = f"PC1 ({var_ratio[0] * 100:.1f}% var)"
        ylab = f"PC2 ({var_ratio[1] * 100:.1f}% var)"
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(ylab)
    axes[1].set_title("Layer 10 PCA of Steering Space")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].axvline(0.0, color="black", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig11_pca_steering_space")
    plt.close(fig)


def plot_contamination_vs_effect(results: Dict, figure_dirs: Sequence[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mat = np.asarray(results["contamination_matrix"], dtype=float)
    sns.heatmap(
        mat,
        ax=axes[0],
        cmap="mako",
        vmin=0,
        vmax=max(0.6, float(np.nanmax(mat))),
        xticklabels=BEHAVIOR_LAYERS,
        yticklabels=[DIM_LABELS[d] for d in DIMENSIONS],
        cbar_kws={"label": "Max |cosine| with control vectors"},
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("")
    axes[0].set_title("Control Contamination by Dimension and Layer")

    palette = dict(zip(DIMENSIONS, sns.color_palette("tab10", n_colors=len(DIMENSIONS))))
    for dim in DIMENSIONS:
        dim_records = [record for record in results["records"] if record["dimension"] == dim]
        axes[1].scatter(
            [record["max_control_contamination"] for record in dim_records],
            [record["abs_cohens_d"] for record in dim_records],
            s=55,
            alpha=0.85,
            color=palette[dim],
            label=DIM_LABELS[dim],
        )
    summary = results["summary"]
    axes[1].set_xlabel("Max |cosine| with control vectors")
    axes[1].set_ylabel("|Cohen's d|")
    axes[1].set_title("Control Contamination vs Behavioral Effect Size")
    axes[1].text(
        0.03,
        0.97,
        f"Pearson r = {summary['pearson_r_contamination_vs_abs_cohens_d']:.2f}\n"
        f"Spearman r = {summary['spearman_r_contamination_vs_abs_cohens_d']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axes[1].legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    save_figure(fig, figure_dirs, "fig12_contamination_vs_effect")
    plt.close(fig)


def plot_contamination_vs_effect_by_layer(results: Dict, figure_dirs: Sequence[Path]) -> None:
    palette = dict(zip(DIMENSIONS, sns.color_palette("tab10", n_colors=len(DIMENSIONS))))
    all_records = results["records"]
    x_max = max(record["max_control_contamination"] for record in all_records)
    y_vals = [record["abs_cohens_d"] for record in all_records if np.isfinite(record["abs_cohens_d"])]
    y_max = max(y_vals) if y_vals else 1.0

    for layer in BEHAVIOR_LAYERS:
        layer_records = [record for record in all_records if record["layer"] == layer]
        fig, ax = plt.subplots(figsize=(7, 5))
        for dim in DIMENSIONS:
            dim_records = [record for record in layer_records if record["dimension"] == dim]
            if not dim_records:
                continue
            ax.scatter(
                [record["max_control_contamination"] for record in dim_records],
                [record["abs_cohens_d"] for record in dim_records],
                s=65,
                alpha=0.9,
                color=palette[dim],
                label=DIM_LABELS[dim],
            )

        layer_x = [record["max_control_contamination"] for record in layer_records]
        layer_y = [record["abs_cohens_d"] for record in layer_records]
        pearson = safe_correlation(layer_x, layer_y, pearsonr)
        spearman = safe_correlation(layer_x, layer_y, spearmanr)

        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(0, y_max * 1.08)
        ax.set_xlabel("Max |cosine| with control vectors")
        ax.set_ylabel("|Cohen's d|")
        ax.set_title(f"Control Contamination vs Effect Size (Layer {layer})")
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {pearson:.2f}\nSpearman r = {spearman:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.legend(frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save_figure(fig, figure_dirs, f"fig12_contamination_vs_effect_L{layer}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    behavior_results_dir = Path(args.behavior_results_dir)
    ensure_dirs(results_dir, figures_dir)
    figure_dirs = get_figure_output_dirs(results_dir, figures_dir)

    log.info("Loading negotiation vectors")
    neg_vecs = load_all_vectors(
        vectors_root=VECTORS_ROOT,
        variant=args.variant,
        space="negotiation",
        model_alias=args.model,
        method=args.method,
        dimensions=DIMENSIONS,
    )

    analyses = [args.analysis] if args.analysis != "all" else [
        "logit_lens",
        "cosine_evolution",
        "pca_analysis",
        "contamination_analysis",
    ]

    if "logit_lens" in analyses:
        log.info("Loading model for logit-lens analysis")
        model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
        logit_results = run_logit_lens(
            model=model,
            tokenizer=tokenizer,
            neg_vecs=neg_vecs,
            layers=LOGIT_LENS_LAYERS,
            top_k=args.top_k,
            model_alias=args.model,
            method=args.method,
        )
        save_json(results_dir / "logit_lens_results.json", logit_results)
        plot_logit_lens_heatmap(logit_results, figure_dirs)
        plot_logit_lens_top_tokens(logit_results, figure_dirs)

    if "cosine_evolution" in analyses:
        cosine_results = run_cosine_evolution(neg_vecs)
        save_json(results_dir / "cosine_evolution.json", cosine_results)
        plot_cosine_heatmaps(cosine_results, figure_dirs)
        plot_cosine_evolution(cosine_results, figure_dirs)
        log.info(
            "Sanity check: firmness↔empathy cosine at L10 = %.3f",
            cosine_results["sanity_checks"]["firmness_empathy_L10"],
        )

    if "pca_analysis" in analyses:
        pca_results = run_pca_analysis(neg_vecs)
        save_json(results_dir / "pca_analysis.json", pca_results)
        plot_pca_steering_space(pca_results, figure_dirs)

    if "contamination_analysis" in analyses:
        log.info("Loading control vectors")
        ctrl_vecs = load_all_vectors(
            vectors_root=VECTORS_ROOT,
            variant=args.variant,
            space="control",
            model_alias=args.model,
            method=args.method,
            dimensions=CONTROL_DIMENSIONS,
        )
        log.info("Loading behavioral results")
        behavioral = load_behavioral_results(
            results_dir=behavior_results_dir,
            dimensions=DIMENSIONS,
            layers=BEHAVIOR_LAYERS,
            alpha=BEHAVIOR_ALPHA,
        )
        contamination_results = run_contamination_analysis(neg_vecs, ctrl_vecs, behavioral)
        save_json(results_dir / "contamination_analysis.json", contamination_results)
        plot_contamination_vs_effect(contamination_results, figure_dirs)
        plot_contamination_vs_effect_by_layer(contamination_results, figure_dirs)


if __name__ == "__main__":
    main()
