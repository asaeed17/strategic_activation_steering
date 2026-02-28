#!/usr/bin/env python3
"""
validate_vectors.py

Comprehensive validation suite for steering vectors.

Seven validation checks are implemented, each answering a different question
about whether the extracted steering vectors genuinely encode the intended
negotiation concepts (firmness, empathy, …) or are confounded by surface
features such as text length.

Checks
------
1. **Length Confound**  (no model needed)
   Do positive and negative samples differ systematically in word count?
   If so the model can trivially separate them by length, not concept.

2. **Probe Accuracy + Permutation Test**  (requires model)
   Are per-layer probe accuracies significantly above chance?
   A permutation test shuffles labels to build a null distribution.

3. **Cosine Similarity Matrix**  (requires model or vectors on disk)
   How much do negotiation steering directions overlap with control
   directions (verbosity, formality)?  High overlap = confounding.

4. **Normalised Bias – Cohen's d**  (requires model)
   Project control-dimension hidden states onto the negotiation steering
   vector.  Cohen's d > 0.8 at a layer means the steering vector
   separates verbosity/formality there — that layer is unsafe.

5. **Per-Pair Consistency**  (requires model)
   Does every contrastive pair push activations in the same direction?
   Outlier pairs that point the wrong way dilute the steering vector.

6. **Steering-Direction Probe**  (requires model)
   Project hidden states onto the 1-D steering direction and probe.
   If the full-dimensional probe is 1.0 but this is 0.6, the steering
   vector itself is pointing in the wrong direction.

7. **Selectivity & Layer Recommendation**  (partial without model)
   Combine probe accuracy and bias into a single per-layer score.
   Recommend layers where the target concept is well-encoded AND
   confounding is minimal.

Two modes:
  --full           Load the model, extract hidden states, run all checks.
  --analyze-only   Work with existing probe_results.json + pairs JSON.
                   No GPU needed.  Runs checks 1, 7, and pattern analyses.

Usage examples:
  # Analyse existing results (no GPU)
  python validate_vectors.py --model qwen2.5-3b --analyze-only

  # Full validation (GPU required)
  python validate_vectors.py --model qwen2.5-3b --full

  # Full validation, specific dimensions only
  python validate_vectors.py --model qwen2.5-3b --full --dimensions firmness empathy
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Lazy imports ─────────────────────────────────────────────────────
# extract_vectors.py depends on `transformers`, which may not be installed
# on machines used only for analyse-only mode.  We import it lazily so
# the script can still run --analyze-only without a GPU environment.
_EXTRACT_LOADED = False
MODELS = None
HF_TOKEN = None

def _ensure_extract_vectors():
    """Import extract_vectors on first use (needed for --full mode)."""
    global _EXTRACT_LOADED, MODELS, HF_TOKEN
    if _EXTRACT_LOADED:
        return
    from extract_vectors import (       # noqa: F811
        MODELS as _M, HF_TOKEN as _HF, ModelConfig,
        format_sample, extract_hidden_states, compute_mean_diff,
    )
    MODELS = _M
    HF_TOKEN = _HF
    _EXTRACT_LOADED = True


# Minimal model registry for analyse-only mode (no torch / transformers)
# so we can resolve --model aliases without importing the heavy stack.
_ALIAS_MAP = {
    "qwen2.5-3b": "qwen2.5-3b",
    "qwen2.5-7b": "qwen2.5-7b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "llama3-8b": "llama3-8b",
    "llama3-3b": "llama3-3b",
    "gemma2-9b": "gemma2-9b",
    "gemma2-2b": "gemma2-2b",
    "mistral-7b": "mistral-7b",
}


# ═══════════════════════════════════════════════════════════════════════
# CHECK 1 — Length Confound Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_length_confound(dimensions: List[Dict]) -> Dict[str, Dict]:
    """
    For each dimension, compare the word counts of positive vs negative
    samples.  Returns Cohen's d for the length difference per dimension.

    Why this matters
    ----------------
    If positive samples are systematically *longer* than negative samples
    (common when the "positive" trait requires more explanation), the model
    can learn to separate them just by detecting "how much text is this?"
    rather than understanding the concept.  A Cohen's d > 0.8 is a red
    flag; > 1.5 is severe.
    """
    import re as _re

    def _char_count(text: str) -> int:
        """Count non-whitespace characters (a proxy for token-level length)."""
        return len(_re.sub(r'\s+', '', text))

    results = {}
    for dim in dimensions:
        dim_id = dim["id"]
        pairs = dim["pairs"]

        pos_lens = [len(p["positive"].split()) for p in pairs]
        neg_lens = [len(p["negative"].split()) for p in pairs]

        # Also measure character-level (closer to sub-word token count)
        pos_chars = [_char_count(p["positive"]) for p in pairs]
        neg_chars = [_char_count(p["negative"]) for p in pairs]

        mean_pos = float(np.mean(pos_lens))
        mean_neg = float(np.mean(neg_lens))

        var_pos = float(np.var(pos_lens, ddof=1)) if len(pos_lens) > 1 else 0.0
        var_neg = float(np.var(neg_lens, ddof=1)) if len(neg_lens) > 1 else 0.0
        pooled_std = float(np.sqrt((var_pos + var_neg) / 2)) if (var_pos + var_neg) > 0 else 1e-8
        d = (mean_pos - mean_neg) / pooled_std

        ratio = mean_pos / mean_neg if mean_neg > 0 else float("inf")

        # Character-level Cohen's d
        mc_pos = float(np.mean(pos_chars))
        mc_neg = float(np.mean(neg_chars))
        vc_pos = float(np.var(pos_chars, ddof=1)) if len(pos_chars) > 1 else 0.0
        vc_neg = float(np.var(neg_chars, ddof=1)) if len(neg_chars) > 1 else 0.0
        cp_std = float(np.sqrt((vc_pos + vc_neg) / 2)) if (vc_pos + vc_neg) > 0 else 1e-8
        d_char = (mc_pos - mc_neg) / cp_std

        # Severity based on the WORSE of word and char Cohen's d
        worst_d = max(abs(d), abs(d_char))
        if worst_d > 1.5:
            severity = "SEVERE"
        elif worst_d > 0.8:
            severity = "MODERATE"
        elif worst_d > 0.3:
            severity = "MILD"
        else:
            severity = "NEGLIGIBLE"

        results[dim_id] = {
            "mean_pos_words": round(mean_pos, 1),
            "mean_neg_words": round(mean_neg, 1),
            "length_ratio": round(ratio, 2),
            "cohens_d": round(d, 3),
            "mean_pos_chars": round(mc_pos, 1),
            "mean_neg_chars": round(mc_neg, 1),
            "cohens_d_char": round(d_char, 3),
            "severity": severity,
        }
    return results


def analyze_vocabulary_overlap(dimensions: List[Dict]) -> Dict[str, Dict]:
    """
    Check how distinguishable positive and negative samples are by vocabulary.

    For each dimension, compute:
    - Jaccard similarity of word sets (should be high if pairs are well-matched)
    - Top distinctive words for each side
    - A TF-IDF-like discriminability score

    If vocabulary overlap is LOW, the model can trivially classify samples
    by the presence/absence of specific words rather than conceptual meaning.
    """
    import re as _re
    from collections import Counter

    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in _re.findall(r'\b\w+\b', text)]

    results = {}
    for dim in dimensions:
        dim_id = dim["id"]
        pairs = dim["pairs"]

        pos_words_all = Counter()
        neg_words_all = Counter()
        pos_sets = []
        neg_sets = []

        for p in pairs:
            pt = set(_tokenize(p["positive"]))
            nt = set(_tokenize(p["negative"]))
            pos_sets.append(pt)
            neg_sets.append(nt)
            pos_words_all.update(_tokenize(p["positive"]))
            neg_words_all.update(_tokenize(p["negative"]))

        # Global Jaccard: how much vocab overlap overall?
        all_pos = set().union(*pos_sets) if pos_sets else set()
        all_neg = set().union(*neg_sets) if neg_sets else set()
        union = all_pos | all_neg
        inter = all_pos & all_neg
        jaccard = len(inter) / len(union) if union else 1.0

        # Per-pair Jaccard (how similar is each pair?)
        pair_jaccards = []
        for ps, ns in zip(pos_sets, neg_sets):
            u = ps | ns
            pair_jaccards.append(len(ps & ns) / len(u) if u else 1.0)
        mean_pair_jaccard = float(np.mean(pair_jaccards)) if pair_jaccards else 1.0

        # Top distinctive words (appear much more on one side)
        combined = set(pos_words_all.keys()) | set(neg_words_all.keys())
        discriminative = []
        for w in combined:
            pc = pos_words_all.get(w, 0)
            nc = neg_words_all.get(w, 0)
            total = pc + nc
            if total < 2:
                continue
            ratio = abs(pc - nc) / total  # 0 = balanced, 1 = one-sided
            discriminative.append((w, pc, nc, ratio))
        discriminative.sort(key=lambda x: -x[3])
        top_pos_words = [(w, pc, nc) for w, pc, nc, r in discriminative[:5]
                         if pc > nc]
        top_neg_words = [(w, pc, nc) for w, pc, nc, r in discriminative[:5]
                         if nc > pc]

        # Assessment — lower thresholds are more concerning.
        # Global Jaccard < 0.15 + pair Jaccard < 0.1 = very distinctive pairs.
        # This doesn't mean the pairs are bad, but it means probe accuracy is
        # NOT a reliable validation signal — a bag-of-words classifier could
        # match the probe accuracy without any hidden-state information.
        if jaccard < 0.15 and mean_pair_jaccard < 0.08:
            concern = "CRITICAL_OVERLAP"
            note = ("Extremely low vocabulary overlap — positive and negative "
                    "use almost completely different words.  Probe accuracy is "
                    "NOT a reliable signal; any classifier can separate these "
                    "by surface word features alone.")
        elif jaccard < 0.20:
            concern = "LOW_OVERLAP"
            note = ("Low vocabulary overlap.  High probe accuracy may partly "
                    "reflect word-level features rather than conceptual "
                    "representation.")
        elif mean_pair_jaccard < 0.10:
            concern = "LOW_PAIR_OVERLAP"
            note = ("Within-pair overlap is low — each pair's positive and "
                    "negative differ substantially in wording.")
        else:
            concern = "OK"
            note = None

        results[dim_id] = {
            "global_jaccard":       round(jaccard, 4),
            "mean_pair_jaccard":    round(mean_pair_jaccard, 4),
            "n_unique_pos":         len(all_pos),
            "n_unique_neg":         len(all_neg),
            "top_pos_distinctive":  [(w, pc, nc) for w, pc, nc in top_pos_words[:3]],
            "top_neg_distinctive":  [(w, pc, nc) for w, pc, nc in top_neg_words[:3]],
            "concern":              concern,
            "note":                 note,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 2 — Probe Accuracy with Permutation Test
# ═══════════════════════════════════════════════════════════════════════

def _train_probe(X: np.ndarray, y: np.ndarray, cv: int = 5,
                 seed: int = 42) -> float:
    """Logistic-regression CV accuracy (handles tiny datasets)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n = len(X)
    actual_cv = min(cv, n // 2)
    if actual_cv < 2:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X, y)
        return float(clf.score(X, y))
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=actual_cv).mean())


def permutation_test_per_layer(
    pos_h: np.ndarray,          # (N, n_layers, H)
    neg_h: np.ndarray,          # (N, n_layers, H)
    n_permutations: int = 100,
    cv: int = 5,
    seed: int = 42,
    sample_layers: Optional[List[int]] = None,
) -> Dict[int, Dict]:
    """
    At each layer, compare the real probe accuracy against a null
    distribution built by shuffling labels `n_permutations` times.

    Returns a dict  layer → {real_acc, null_mean, null_std, p_value, significant}

    Why this matters
    ----------------
    With only 24 samples in ~2 000 dimensions, a linear classifier can
    achieve deceptively high accuracy just by finding chance correlations.
    The permutation test tells us whether the real accuracy is genuinely
    above what random labels would produce.
    """
    from tqdm import tqdm

    n_layers = pos_h.shape[1]
    layers = sample_layers if sample_layers else list(range(n_layers))
    rng = np.random.RandomState(seed)
    results = {}

    for l in tqdm(layers, desc="  permutation test", leave=False):
        X = np.concatenate([pos_h[:, l, :], neg_h[:, l, :]], axis=0)
        y = np.concatenate([np.ones(len(pos_h)), np.zeros(len(neg_h))])
        n = len(X)
        actual_cv = min(cv, n // 2)
        if actual_cv < 2:
            results[l] = {"note": "too few samples for CV"}
            continue

        real_acc = _train_probe(X, y, cv=actual_cv, seed=seed)

        null_accs = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            null_accs.append(_train_probe(X, y_perm, cv=actual_cv, seed=seed))
        null_accs = np.array(null_accs)

        p_value = float((np.sum(null_accs >= real_acc) + 1) / (n_permutations + 1))
        results[l] = {
            "real_acc":    round(float(real_acc), 4),
            "null_mean":   round(float(null_accs.mean()), 4),
            "null_std":    round(float(null_accs.std()), 4),
            "p_value":     round(p_value, 4),
            "significant": p_value < 0.05,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3 — Cosine Similarity Matrix
# ═══════════════════════════════════════════════════════════════════════

def compute_cosine_similarity_matrix(
    vectors: Dict[str, np.ndarray],   # dim_id → (n_layers, H)
    layers: Optional[List[int]] = None,
) -> Dict[int, Dict]:
    """
    At each requested layer, compute the pairwise cosine similarity
    between all steering directions (both negotiation and control).

    Returns
    -------
    {layer: {"names": [...], "matrix": [[...], ...], "warnings": [...]}}

    Why this matters
    ----------------
    If cos(firmness, verbosity) is high, the "firmness" vector is largely
    a verbosity vector — any advantage it gives in negotiation may be an
    artefact of making the model more verbose rather than more firm.
    """
    names = sorted(vectors.keys())
    if not names:
        return {}

    first = next(iter(vectors.values()))
    n_layers = first.shape[0]
    test_layers = layers if layers else list(range(n_layers))

    results = {}
    for l in test_layers:
        V = np.stack([vectors[n][l] for n in names])     # (D, H)
        norms = np.linalg.norm(V, axis=-1, keepdims=True)
        V_n = V / np.where(norms < 1e-8, 1.0, norms)
        sim = (V_n @ V_n.T).tolist()

        # Flag pairs with |cosine| > 0.5 (cross-category only)
        warnings = []
        neg_dims = {"firmness", "empathy", "active_listening", "assertiveness",
                    "interest_based_reasoning", "emotional_regulation",
                    "strategic_concession_making", "anchoring", "rapport_building",
                    "batna_awareness", "reframing", "patience", "value_creation",
                    "information_gathering", "clarity_and_directness"}
        con_dims = {"verbosity", "formality"}
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j <= i:
                    continue
                # Cross-category check
                cross = (ni in neg_dims and nj in con_dims) or \
                        (ni in con_dims and nj in neg_dims)
                if cross and abs(sim[i][j]) > 0.5:
                    warnings.append({
                        "dim_a": ni, "dim_b": nj,
                        "cosine": round(sim[i][j], 4),
                    })

        results[l] = {
            "names": names,
            "matrix": [[round(v, 4) for v in row] for row in sim],
            "warnings": warnings,
        }
    return results


def compute_cosine_from_files(
    vectors_dir: Path,
    method: str = "mean_diff",
    layers: Optional[List[int]] = None,
) -> Dict:
    """Load _all_layers.npy files from a model's vectors dir and compute
    pairwise cosine.  Works in analyse-only mode when vectors/ exists."""
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    vectors = {}
    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        vectors[dim_id] = np.load(f)

    if not vectors:
        return {"note": "no _all_layers.npy files found"}

    return compute_cosine_similarity_matrix(vectors, layers)


# ═══════════════════════════════════════════════════════════════════════
# CHECK 4 — Normalised Bias (Cohen's d)
# ═══════════════════════════════════════════════════════════════════════

def compute_cohens_d_bias(
    steering_vecs: np.ndarray,     # (n_layers, H)  — unit-normed direction
    control_pos_h: np.ndarray,     # (N, n_layers, H)
    control_neg_h: np.ndarray,     # (N, n_layers, H)
) -> Dict[str, Any]:
    """
    Project control-dimension hidden states onto the steering vector and
    compute Cohen's d of the separation at each layer.

    Returns per-layer Cohen's d plus summary statistics.

    Why this matters
    ----------------
    Raw projection separations (as in the existing bias analysis) grow
    with hidden-state norms across layers, making it hard to compare
    across layers.  Cohen's d divides by the pooled standard deviation,
    giving an effect-size that is directly comparable:
        |d| < 0.2  → negligible bias
        |d| 0.2–0.8 → moderate
        |d| > 0.8  → SEVERE — steering at this layer will push the
                      control trait (verbosity / formality) alongside
                      the intended negotiation trait.
    """
    n_layers = steering_vecs.shape[0]
    ds = np.zeros(n_layers)

    for l in range(n_layers):
        vec = steering_vecs[l]
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            continue
        vec = vec / norm

        pos_projs = control_pos_h[:, l, :] @ vec
        neg_projs = control_neg_h[:, l, :] @ vec

        mean_diff = float(pos_projs.mean() - neg_projs.mean())
        pooled_var = (float(pos_projs.var(ddof=1)) + float(neg_projs.var(ddof=1))) / 2
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-8
        ds[l] = mean_diff / pooled_std

    max_idx = int(np.argmax(np.abs(ds)))
    severe_layers = [int(l) for l in range(n_layers) if abs(ds[l]) > 0.8]

    return {
        "cohens_d_per_layer": [round(float(v), 4) for v in ds],
        "max_abs_d":          round(float(np.abs(ds).max()), 4),
        "max_abs_d_layer":    max_idx,
        "severe_layers":      severe_layers,
        "n_severe":           len(severe_layers),
    }


# ═══════════════════════════════════════════════════════════════════════
# CHECK 5 — Per-Pair Consistency (Alignment)
# ═══════════════════════════════════════════════════════════════════════

def compute_pair_alignment(
    pos_h: np.ndarray,    # (N, n_layers, H)
    neg_h: np.ndarray,    # (N, n_layers, H)
) -> Dict[str, Any]:
    """
    For each contrastive pair, compute the difference vector and measure
    its cosine similarity with the *mean* difference vector at each layer.

    Low mean alignment → noisy pairs dilute the steering signal.
    Individual pairs with near-zero or negative alignment are outliers
    that should be rewritten or removed.

    Returns per-pair and per-layer alignment statistics.
    """
    diffs = pos_h - neg_h                          # (N, L, H)
    mean_diff = diffs.mean(axis=0)                 # (L, H)

    n_pairs, n_layers, H = diffs.shape
    alignments = np.zeros((n_pairs, n_layers))

    for i in range(n_pairs):
        for l in range(n_layers):
            d_n = np.linalg.norm(diffs[i, l])
            m_n = np.linalg.norm(mean_diff[l])
            if d_n < 1e-8 or m_n < 1e-8:
                alignments[i, l] = 0.0
            else:
                alignments[i, l] = float(
                    np.dot(diffs[i, l], mean_diff[l]) / (d_n * m_n)
                )

    # Summary: per-pair mean alignment (across layers)
    pair_means = alignments.mean(axis=1)           # (N,)
    # Outliers: pairs whose mean alignment is below 0.3
    outlier_idx = [int(i) for i in range(n_pairs) if pair_means[i] < 0.3]

    return {
        "per_layer_mean_alignment": [
            round(float(alignments[:, l].mean()), 4) for l in range(n_layers)
        ],
        "per_pair_mean_alignment": [round(float(v), 4) for v in pair_means],
        "overall_mean":            round(float(pair_means.mean()), 4),
        "outlier_pairs":           outlier_idx,
        "n_outliers":              len(outlier_idx),
    }


# ═══════════════════════════════════════════════════════════════════════
# CHECK 6 — Steering-Direction Probe
# ═══════════════════════════════════════════════════════════════════════

def steering_direction_probe(
    pos_h: np.ndarray,            # (N, n_layers, H)
    neg_h: np.ndarray,            # (N, n_layers, H)
    steering_vecs: np.ndarray,    # (n_layers, H)
    cv: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Project hidden states onto the *steering direction* at each layer
    and train a 1-D logistic regression probe.

    If the full-dimensional probe gives 1.0 but this gives 0.6, the
    steering direction is not capturing the dominant discriminative axis
    — a clear sign the extraction method (mean-diff or PCA) picked up the
    wrong direction.

    Returns per-layer 1-D probe accuracy and a comparison flag.
    """
    n_layers = steering_vecs.shape[0]
    accs = np.zeros(n_layers)

    for l in range(n_layers):
        vec = steering_vecs[l]
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            accs[l] = 0.5
            continue
        vec = vec / norm

        pos_proj = pos_h[:, l, :] @ vec   # (N,)
        neg_proj = neg_h[:, l, :] @ vec   # (N,)

        X = np.concatenate([pos_proj[:, None], neg_proj[:, None]], axis=0)
        y = np.concatenate([np.ones(len(pos_proj)), np.zeros(len(neg_proj))])
        accs[l] = _train_probe(X, y, cv=cv, seed=seed)

    return {
        "per_layer_acc": [round(float(a), 4) for a in accs],
        "best_layer":    int(np.argmax(accs)),
        "best_acc":      round(float(accs.max()), 4),
        "mean_acc":      round(float(accs.mean()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# CHECK 7 — Selectivity & Layer Recommendation
# ═══════════════════════════════════════════════════════════════════════

def compute_selectivity(
    probe_acc: np.ndarray,                      # (n_layers,)
    bias_scores: Dict[str, np.ndarray],         # control_id → (n_layers,) Cohen's d
    bias_weight: float = 0.5,
) -> np.ndarray:
    """
    selectivity[l] = probe_acc[l] − bias_weight * max(|d_c[l]|  for c in controls)

    A high selectivity means the concept is well-encoded AND the steering
    vector does NOT accidentally encode a control trait at that layer.
    """
    n_layers = len(probe_acc)
    max_bias = np.zeros(n_layers)
    for d_arr in bias_scores.values():
        max_bias = np.maximum(max_bias, np.abs(d_arr))

    # Clamp bias contribution to [0, 1] using sigmoid-like mapping
    bias_penalty = np.minimum(max_bias / 2.0, 1.0)
    return probe_acc - bias_weight * bias_penalty


def recommend_layers(
    probe_acc: np.ndarray,
    bias_scores: Dict[str, np.ndarray],
    min_acc: float = 0.85,
    max_bias_d: float = 0.8,
) -> Dict[str, Any]:
    """Return layers that pass both the accuracy and bias thresholds."""
    n_layers = len(probe_acc)
    safe = []
    avoid = []

    for l in range(n_layers):
        acc_ok = probe_acc[l] >= min_acc
        bias_ok = all(abs(d[l]) <= max_bias_d for d in bias_scores.values())
        if acc_ok and bias_ok:
            safe.append(l)
        elif not bias_ok:
            avoid.append(l)

    return {
        "recommended": safe,
        "avoid":       avoid,
        "criteria":    f"accuracy >= {min_acc} AND max |Cohen's d| <= {max_bias_d}",
    }


def compute_approximate_selectivity(
    probe_accs: Dict[str, List[float]],
    raw_bias: Dict[str, List[float]],
) -> Dict[str, Dict]:
    """
    Approximate selectivity from existing probe_results.json data.

    Because we only have *raw* projection separations (not Cohen's d),
    we normalise the bias values to [0, 1] using min-max scaling within
    each bias series.  This gives a *relative* selectivity per layer —
    useful for ranking layers even without full normalisation.

    Selectivity[l] = accuracy[l] − normalised_bias[l]
    """
    results = {}

    for neg_dim, acc_list in probe_accs.items():
        acc = np.array(acc_list)
        # Collect all bias series that involve this dimension
        related_bias = {
            k: np.array(v) for k, v in raw_bias.items()
            if k.startswith(neg_dim + "_vs_")
        }
        if not related_bias:
            results[neg_dim] = {
                "per_layer": [round(float(a), 4) for a in acc],
                "best_layer": int(np.argmax(acc)),
                "note": "no bias data available for this dimension",
            }
            continue

        # Aggregate bias: take max across controls at each layer
        bias_stack = np.stack(list(related_bias.values()))
        max_raw_bias = np.abs(bias_stack).max(axis=0)

        # Min-max normalise to [0, 1]
        b_min = max_raw_bias.min()
        b_max = max_raw_bias.max()
        b_range = b_max - b_min
        if b_range < 1e-8:
            norm_bias = np.zeros_like(max_raw_bias)
        else:
            norm_bias = (max_raw_bias - b_min) / b_range

        selectivity = acc - norm_bias
        best_l = int(np.argmax(selectivity))

        results[neg_dim] = {
            "per_layer": [round(float(s), 4) for s in selectivity],
            "norm_bias": [round(float(b), 4) for b in norm_bias],
            "best_layer": best_l,
            "best_selectivity": round(float(selectivity[best_l]), 4),
        }
    return results


def compute_approx_layer_recommendation(
    probe_accs: Dict[str, List[float]],
    raw_bias: Dict[str, List[float]],
    min_acc: float = 0.90,
    max_norm_bias: float = 0.25,
) -> Dict[str, Dict]:
    """
    Layer recommendations from existing results (no Cohen's d available).

    Uses min-max-normalised bias; `max_norm_bias <= 0.25` keeps us in the
    lowest quartile of the bias range.

    Also includes:
    - `top5`: the 5 layers with best selectivity (acc − norm_bias)
    - `sensitivity`: how many layers survive at stricter thresholds
    """
    results = {}
    for neg_dim, acc_list in probe_accs.items():
        acc = np.array(acc_list)
        n_layers = len(acc)

        related_bias = {
            k: np.array(v) for k, v in raw_bias.items()
            if k.startswith(neg_dim + "_vs_")
        }
        if not related_bias:
            results[neg_dim] = {"recommended": list(range(n_layers)),
                                "avoid": [], "note": "no bias data"}
            continue

        bias_stack = np.stack(list(related_bias.values()))
        max_raw = np.abs(bias_stack).max(axis=0)
        b_min, b_max = max_raw.min(), max_raw.max()
        b_range = b_max - b_min
        norm_bias = (max_raw - b_min) / b_range if b_range > 1e-8 else np.zeros_like(max_raw)

        safe, avoid = [], []
        for l in range(n_layers):
            if acc[l] >= min_acc and norm_bias[l] <= max_norm_bias:
                safe.append(l)
            elif norm_bias[l] > 0.75:
                avoid.append(l)

        # Top-5 by selectivity score
        selectivity = acc - norm_bias
        top_k_idx = np.argsort(-selectivity)[:5].tolist()

        # Sensitivity analysis: how many layers survive at stricter thresholds
        sensitivity = {}
        for thr in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
            n_pass = int(sum(1 for l in range(n_layers)
                             if acc[l] >= min_acc and norm_bias[l] <= thr))
            sensitivity[f"bias<={thr:.2f}"] = n_pass

        results[neg_dim] = {
            "recommended": safe,
            "avoid":       avoid,
            "top5_layers": top_k_idx,
            "top5_scores": [round(float(selectivity[l]), 4) for l in top_k_idx],
            "sensitivity": sensitivity,
            "criteria":    f"accuracy >= {min_acc} AND norm_bias <= {max_norm_bias}",
        }
    return results


def _valid_layer_mask(vecs: np.ndarray) -> np.ndarray:
    """Return boolean mask of layers with no NaN/Inf values.

    Args:
        vecs: array of shape (n_layers, hidden_dim)

    Returns:
        boolean array of shape (n_layers,) — True for usable layers
    """
    return ~(np.isnan(vecs).any(axis=1) | np.isinf(vecs).any(axis=1))


# ═══════════════════════════════════════════════════════════════════════
# CHECK 8 — Vector Norm Profile & Unit Normalization Verification
# ═══════════════════════════════════════════════════════════════════════

def analyze_vector_norms(
    vectors_dir: Path,
    method: str = "mean_diff",
) -> Dict[str, Any]:
    """
    Load per-layer vectors for every available dimension and verify
    they are unit-normed.  Compute norm profiles across layers.

    Why this matters
    ----------------
    Vector extraction should produce unit-normed vectors (per CLAUDE.md:
    "Vectors are unit-normed per layer").  If any vectors deviate from
    unit norm, it indicates an extraction bug or that the normalisation
    step was skipped.  Even if all vectors *are* unit-normed, comparing
    the *pre-normalisation* norms (from the all_layers file) reveals how
    much "raw signal" the model produces at each layer for each concept.
    """
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    results: Dict[str, Any] = {"dimensions": {}, "summary": {}}

    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        all_layers = np.load(f)  # (n_layers, hidden_dim)
        n_layers, hidden_dim = all_layers.shape

        # Identify valid layers (no NaN/Inf)
        valid_mask = _valid_layer_mask(all_layers)
        n_valid = int(valid_mask.sum())
        n_nan_layers = n_layers - n_valid

        # Per-layer norms (NaN for invalid layers)
        norms = np.linalg.norm(all_layers, axis=1)
        valid_norms = norms[valid_mask]
        is_unit_normed = (bool(np.allclose(valid_norms, 1.0, atol=1e-4))
                          if n_valid > 0 else False)

        # Also check individual per-layer files
        per_layer_norms = []
        for l in range(n_layers):
            layer_file = vec_subdir / f"{dim_id}_layer{l:02d}.npy"
            if layer_file.exists():
                v = np.load(layer_file)
                per_layer_norms.append(float(np.linalg.norm(v)))

        per_layer_unit = (bool(np.allclose(per_layer_norms, 1.0, atol=1e-4))
                          if per_layer_norms else None)

        results["dimensions"][dim_id] = {
            "n_layers":          n_layers,
            "n_valid_layers":    n_valid,
            "n_nan_layers":      n_nan_layers,
            "hidden_dim":        hidden_dim,
            "all_layers_norms":  [round(float(n), 6) if not np.isnan(n) else None for n in norms],
            "all_layers_unit_normed": is_unit_normed,
            "per_layer_norms":   [round(n, 6) for n in per_layer_norms] if per_layer_norms else [],
            "per_layer_unit_normed": per_layer_unit,
            "norm_mean":         round(float(valid_norms.mean()), 6) if n_valid > 0 else None,
            "norm_std":          round(float(valid_norms.std()), 6) if n_valid > 0 else None,
            "norm_min":          round(float(valid_norms.min()), 6) if n_valid > 0 else None,
            "norm_max":          round(float(valid_norms.max()), 6) if n_valid > 0 else None,
        }

    # Summary
    n_dims = len(results["dimensions"])
    all_unit = all(d["all_layers_unit_normed"]
                   for d in results["dimensions"].values())
    results["summary"] = {
        "n_dimensions":    n_dims,
        "all_unit_normed": all_unit,
        "note":            ("All vectors are unit-normed as expected."
                            if all_unit else
                            "WARNING: Not all vectors are unit-normed! "
                            "Check extraction pipeline."),
    }
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 9 — Inter-Layer Consistency (Consecutive Layer Cosine)
# ═══════════════════════════════════════════════════════════════════════

def analyze_layer_consistency(
    vectors_dir: Path,
    method: str = "mean_diff",
) -> Dict[str, Any]:
    """
    For each dimension, compute cosine similarity between consecutive layer
    vectors and between each layer and layer 0.

    Why this matters
    ----------------
    If the steering direction changes gradually across layers, the concept
    is "smoothly" represented — typical of genuine semantic features.  A
    sudden drop in cosine between layers L and L+1 suggests a qualitative
    change in what the model encodes, which may indicate the vector is
    picking up a different signal at different depths.

    Also computes "drift from layer 0" — cos(vec[L], vec[0]) — to show
    how far the concept drifts from its initial representation.
    """
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    results = {}
    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        vecs = np.load(f)  # (n_layers, H)
        n_layers = vecs.shape[0]

        valid_mask = _valid_layer_mask(vecs)

        # Normalise for cosine (NaN layers will become zero vectors)
        safe_vecs = np.where(np.isnan(vecs), 0.0, vecs)
        norms = np.linalg.norm(safe_vecs, axis=1, keepdims=True)
        vecs_n = safe_vecs / np.where(norms < 1e-8, 1.0, norms)

        # Consecutive cosine similarity (only between valid layer pairs)
        consecutive_cos = []
        for l in range(n_layers - 1):
            if valid_mask[l] and valid_mask[l + 1]:
                c = float(np.dot(vecs_n[l], vecs_n[l + 1]))
                consecutive_cos.append(round(c, 4))
            else:
                consecutive_cos.append(None)

        # Drift from layer 0
        if valid_mask[0]:
            drift_from_0 = [
                round(float(np.dot(vecs_n[0], vecs_n[l])), 4)
                if valid_mask[l] else None
                for l in range(n_layers)
            ]
        else:
            drift_from_0 = [None] * n_layers

        # Identify "transition points" where consecutive cosine drops below 0.7
        transitions = [l for l, c in enumerate(consecutive_cos)
                       if c is not None and c < 0.7]

        # Overall consistency metric: mean consecutive cosine (valid only)
        valid_cos = [c for c in consecutive_cos if c is not None]
        mean_consecutive = float(np.mean(valid_cos)) if valid_cos else 0.0
        min_consecutive = float(np.min(valid_cos)) if valid_cos else 0.0

        # Interpretation
        if mean_consecutive > 0.95:
            pattern = "highly_consistent"
            note = ("Steering direction is very stable across layers. "
                    "This suggests the concept is uniformly represented.")
        elif mean_consecutive > 0.8:
            pattern = "moderately_consistent"
            note = ("Steering direction evolves gradually across layers. "
                    "Normal for semantic concepts that undergo processing.")
        elif mean_consecutive > 0.5:
            pattern = "variable"
            note = ("Significant variation in steering direction across layers. "
                    "Different layers may encode different aspects of the concept.")
        else:
            pattern = "inconsistent"
            note = ("Steering direction changes dramatically between layers. "
                    "This suggests the vector may be capturing noise or "
                    "qualitatively different features at different depths.")

        results[dim_id] = {
            "consecutive_cosine":   consecutive_cos,
            "drift_from_layer0":    drift_from_0,
            "mean_consecutive_cos": round(mean_consecutive, 4),
            "min_consecutive_cos":  round(min_consecutive, 4),
            "transition_points":    transitions,
            "n_transitions":        len(transitions),
            "pattern":              pattern,
            "note":                 note,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 10 — Cross-Dimension Similarity Across ALL Layers
# ═══════════════════════════════════════════════════════════════════════

def analyze_cross_dimension_similarity(
    vectors_dir: Path,
    method: str = "mean_diff",
) -> Dict[str, Any]:
    """
    Compute pairwise cosine similarity between all dimension vectors at
    every layer.  Produces a full layer×pairs heatmap.

    Why this matters
    ----------------
    If two dimensions (e.g., firmness and anchoring) have high cosine
    similarity at a given layer, steering with one will inadvertently
    steer with the other.  Conversely, layers where all dimensions are
    maximally independent are ideal for targeted steering.

    The analysis also identifies:
    - "Collapse layers" where most dimension pairs exceed |cos|>0.5
    - "Independence sweet spots" where most pairs have |cos|<0.2
    - Which layer has the lowest average cross-dimension similarity
    """
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    # Load all dimensions
    vectors = {}
    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        vectors[dim_id] = np.load(f)

    if len(vectors) < 2:
        return {"note": "Need at least 2 dimensions for cross-dimension analysis",
                "n_dimensions": len(vectors)}

    dim_names = sorted(vectors.keys())
    n_dims = len(dim_names)
    n_layers = next(iter(vectors.values())).shape[0]

    # Compute pairwise cosine at each layer
    pair_names = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            pair_names.append(f"{dim_names[i]}_vs_{dim_names[j]}")

    # Check which layers are valid (no NaN in any dimension)
    global_valid = np.ones(n_layers, dtype=bool)
    for d in dim_names:
        global_valid &= _valid_layer_mask(vectors[d])

    layer_similarities = {}  # layer → {pair → cosine}
    pair_across_layers = {p: [] for p in pair_names}  # pair → [cosine per layer]

    for l in range(n_layers):
        if not global_valid[l]:
            layer_similarities[l] = {p: None for p in pair_names}
            for p in pair_names:
                pair_across_layers[p].append(None)
            continue

        # Stack and normalise
        V = np.stack([vectors[d][l] for d in dim_names])
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_n = V / np.where(norms < 1e-8, 1.0, norms)
        sim_matrix = V_n @ V_n.T

        layer_sims = {}
        pair_idx = 0
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                cos_val = round(float(sim_matrix[i, j]), 4)
                pname = pair_names[pair_idx]
                layer_sims[pname] = cos_val
                pair_across_layers[pname].append(cos_val)
                pair_idx += 1

        layer_similarities[l] = layer_sims

    # Per-layer summary (only valid layers)
    avg_abs_cos_per_layer = []
    for l in range(n_layers):
        vals = [v for v in layer_similarities[l].values() if v is not None]
        if vals:
            avg_abs_cos_per_layer.append(round(float(np.mean(np.abs(vals))), 4))
        else:
            avg_abs_cos_per_layer.append(None)

    # Find best (most independent) and worst (most collapsed) among valid layers
    valid_avgs = [(l, v) for l, v in enumerate(avg_abs_cos_per_layer) if v is not None]
    if valid_avgs:
        best_layer = min(valid_avgs, key=lambda x: x[1])[0]
        worst_layer = max(valid_avgs, key=lambda x: x[1])[0]
    else:
        best_layer, worst_layer = 0, 0

    # Collapse detection: layers where >50% of pairs have |cos|>0.5
    collapse_layers = []
    for l in range(n_layers):
        vals = [v for v in layer_similarities[l].values() if v is not None]
        if not vals:
            continue
        n_high = sum(1 for v in vals if abs(v) > 0.5)
        if n_high > len(vals) / 2:
            collapse_layers.append(l)

    # Independence sweet spots: layers where all pairs |cos|<0.3
    independence_layers = []
    for l in range(n_layers):
        vals = [v for v in layer_similarities[l].values() if v is not None]
        if not vals:
            continue
        if all(abs(v) < 0.3 for v in vals):
            independence_layers.append(l)

    # Per-pair summary across layers
    pair_summaries = {}
    for pname, cos_list in pair_across_layers.items():
        valid_vals = [v for v in cos_list if v is not None]
        if valid_vals:
            arr = np.array(valid_vals)
            pair_summaries[pname] = {
                "mean_cos":     round(float(np.mean(arr)), 4),
                "std_cos":      round(float(np.std(arr)), 4),
                "max_abs_cos":  round(float(np.max(np.abs(arr))), 4),
                "min_abs_cos":  round(float(np.min(np.abs(arr))), 4),
                "per_layer":    [round(float(v), 4) if v is not None else None for v in cos_list],
            }
        else:
            pair_summaries[pname] = {
                "mean_cos": None, "std_cos": None,
                "max_abs_cos": None, "min_abs_cos": None,
                "per_layer": [None] * n_layers,
            }

    return {
        "dimension_names":        dim_names,
        "n_dimensions":           n_dims,
        "n_layers":               n_layers,
        "pair_summaries":         pair_summaries,
        "avg_abs_cos_per_layer":  avg_abs_cos_per_layer,
        "best_independence_layer": best_layer,
        "best_avg_abs_cos":       avg_abs_cos_per_layer[best_layer],
        "worst_collapse_layer":   worst_layer,
        "worst_avg_abs_cos":      avg_abs_cos_per_layer[worst_layer],
        "collapse_layers":        collapse_layers,
        "independence_layers":    independence_layers,
    }


# ═══════════════════════════════════════════════════════════════════════
# CHECK 11 — Vector Concentration / Effective Dimensionality
# ═══════════════════════════════════════════════════════════════════════

def analyze_vector_concentration(
    vectors_dir: Path,
    method: str = "mean_diff",
) -> Dict[str, Any]:
    """
    Measure how concentrated vs distributed each steering vector is
    across the hidden dimensions.

    Metrics:
    - Effective dimensionality: exp(entropy of squared components / sum²)
      A value near hidden_dim means the vector uses all dimensions equally.
      A low value means the vector lives in a narrow subspace.
    - Gini coefficient: inequality of component magnitudes (0=uniform, 1=single spike).
    - Top-k concentration: fraction of L2 norm² explained by the top 10, 50,
      100 components.

    Why this matters
    ----------------
    Surface features (e.g., "is the text long?") tend to be captured by
    a few dominant hidden dimensions.  Genuine semantic concepts (e.g.,
    "is this negotiation firm?") are typically distributed across many
    dimensions.  A steering vector with very low effective dimensionality
    may be capturing a surface-level feature, not a deep concept.

    A random unit vector in d-dimensional space has expected effective
    dimensionality ~0.63d .  Our vectors should be somewhere between
    surface-feature (very low) and random (0.63d).
    """
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    def _gini(arr):
        """Gini coefficient of absolute values."""
        a = np.sort(np.abs(arr))
        n = len(a)
        if n == 0 or a.sum() < 1e-12:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * a) / (n * np.sum(a))) - (n + 1) / n)

    def _effective_dim(vec):
        """Exponential of entropy of squared components (normalised)."""
        sq = vec ** 2
        total = sq.sum()
        if total < 1e-12:
            return 0.0
        p = sq / total
        # Avoid log(0)
        p = p[p > 1e-15]
        entropy = -float(np.sum(p * np.log(p)))
        return float(np.exp(entropy))

    def _topk_concentration(vec, ks=(10, 50, 100)):
        """Fraction of L2² norm in the top-k largest components."""
        sq = vec ** 2
        total = sq.sum()
        if total < 1e-12:
            return {k: 0.0 for k in ks}
        sorted_sq = np.sort(sq)[::-1]
        result = {}
        for k in ks:
            actual_k = min(k, len(sorted_sq))
            result[k] = round(float(sorted_sq[:actual_k].sum() / total), 4)
        return result

    results = {}

    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        vecs = np.load(f)  # (n_layers, H)
        n_layers, hidden_dim = vecs.shape

        eff_dims = []
        ginis = []
        topk_10 = []
        topk_50 = []

        for l in range(n_layers):
            v = vecs[l]
            if np.isnan(v).any() or np.isinf(v).any():
                eff_dims.append(None)
                ginis.append(None)
                topk_10.append(None)
                topk_50.append(None)
                continue
            ed = _effective_dim(v)
            g = _gini(v)
            tk = _topk_concentration(v, ks=(10, 50, 100))
            eff_dims.append(round(ed, 1))
            ginis.append(round(g, 4))
            topk_10.append(tk[10])
            topk_50.append(tk[50])

        valid_eff_dims = [e for e in eff_dims if e is not None]
        valid_ginis = [g for g in ginis if g is not None]
        mean_eff_dim = float(np.mean(valid_eff_dims)) if valid_eff_dims else 0.0
        mean_gini = float(np.mean(valid_ginis)) if valid_ginis else 0.0

        # Random baseline: expected eff dim for d-dim unit vector
        # exp(ln(d) - 1 + 1/(2d)) ≈ d/e ≈ 0.368d for large d
        random_expected = hidden_dim / np.e

        # Interpretation
        ratio = mean_eff_dim / random_expected if random_expected > 0 else 0
        if ratio > 0.8:
            pattern = "near_random"
            note = (f"Effective dimensionality ({mean_eff_dim:.0f}) is close to "
                    f"random baseline ({random_expected:.0f}). The vector is "
                    f"broadly distributed — no evidence of surface-feature "
                    f"concentration.")
        elif ratio > 0.5:
            pattern = "moderate_concentration"
            note = (f"Effective dimensionality ({mean_eff_dim:.0f}) is moderately "
                    f"below random baseline ({random_expected:.0f}). Some "
                    f"concentration, but not extreme.")
        elif ratio > 0.2:
            pattern = "concentrated"
            note = (f"Effective dimensionality ({mean_eff_dim:.0f}) is well below "
                    f"random baseline ({random_expected:.0f}). The vector is "
                    f"concentrated in a subset of dimensions — more likely to "
                    f"capture a specific feature (possibly surface-level).")
        else:
            pattern = "highly_concentrated"
            note = (f"Effective dimensionality ({mean_eff_dim:.0f}) is very low "
                    f"compared to random ({random_expected:.0f}). The steering "
                    f"direction is dominated by a few components — a strong "
                    f"indicator of surface-feature capture.")

        results[dim_id] = {
            "hidden_dim":               hidden_dim,
            "random_expected_eff_dim":  round(random_expected, 1),
            "eff_dim_per_layer":        eff_dims,
            "mean_eff_dim":             round(mean_eff_dim, 1),
            "min_eff_dim":              round(float(min(valid_eff_dims)), 1) if valid_eff_dims else None,
            "max_eff_dim":              round(float(max(valid_eff_dims)), 1) if valid_eff_dims else None,
            "gini_per_layer":           ginis,
            "mean_gini":                round(mean_gini, 4),
            "top10_concentration":      topk_10,
            "top50_concentration":      [round(v, 4) if v is not None else None for v in topk_50],
            "ratio_to_random":          round(ratio, 4),
            "pattern":                  pattern,
            "note":                     note,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 12 — Shared Subspace Analysis (PCA of dimension vectors)
# ═══════════════════════════════════════════════════════════════════════

def analyze_shared_subspace(
    vectors_dir: Path,
    method: str = "mean_diff",
) -> Dict[str, Any]:
    """
    Stack all dimension vectors at each layer, project onto principal
    components, and measure the effective rank.

    Why this matters
    ----------------
    If N different dimension vectors share a common dominant direction,
    they are not truly independent — they all encode the same confound
    (e.g., verbosity or text-length direction).

    Specifically:
    - If PC1 explains >70% of variance across dimensions, there is a
      dominant shared direction.  We identify what it might be (closest
      named dimension to PC1).
    - If effective rank == N (number of dimensions), the vectors are
      roughly independent — good.
    - Computes the "explained variance ratio" at each layer.
    """
    vec_subdir = vectors_dir / method
    if not vec_subdir.exists():
        return {"note": f"{vec_subdir} not found"}

    # Load all dimensions
    vectors = {}
    for f in sorted(vec_subdir.glob("*_all_layers.npy")):
        dim_id = f.stem.replace("_all_layers", "")
        vectors[dim_id] = np.load(f)

    if len(vectors) < 2:
        return {"note": "Need at least 2 dimensions for subspace analysis",
                "n_dimensions": len(vectors)}

    dim_names = sorted(vectors.keys())
    n_dims = len(dim_names)
    n_layers = next(iter(vectors.values())).shape[0]

    layer_results = {}
    pc1_explained_ratios = []
    effective_ranks = []

    for l in range(n_layers):
        V = np.stack([vectors[d][l] for d in dim_names])  # (n_dims, H)

        # Skip layers with NaN/Inf values
        if np.isnan(V).any() or np.isinf(V).any():
            layer_results[l] = {"skipped": True, "reason": "NaN or Inf values"}
            pc1_explained_ratios.append(None)
            effective_ranks.append(None)
            continue

        # Normalise each vector
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_n = V / np.where(norms < 1e-8, 1.0, norms)

        # SVD (with fallback for non-convergence)
        try:
            _, S, Vt = np.linalg.svd(V_n, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition of the Gram matrix
            gram = V_n @ V_n.T
            eigvals = np.linalg.eigvalsh(gram)[::-1]
            S = np.sqrt(np.maximum(eigvals, 0))
            Vt = None  # alignment computation will be skipped

        # Explained variance ratio
        S_sq = S ** 2
        total_var = S_sq.sum()
        if total_var > 1e-12:
            evr = S_sq / total_var
        else:
            evr = np.zeros_like(S_sq)

        # Effective rank: exp(entropy of normalised singular values²)
        p = evr[evr > 1e-15]
        eff_rank = float(np.exp(-np.sum(p * np.log(p)))) if len(p) > 0 else 0.0

        pc1_explained_ratios.append(round(float(evr[0]), 4))
        effective_ranks.append(round(eff_rank, 2))

        # Which dimension is most aligned with PC1?
        alignments = {}
        if Vt is not None:
            pc1_dir = Vt[0]  # first right singular vector
            for i, d in enumerate(dim_names):
                cos_with_pc1 = float(np.dot(V_n[i], pc1_dir))
                alignments[d] = round(cos_with_pc1, 4)

        layer_results[l] = {
            "explained_variance_ratio": [round(float(v), 4) for v in evr],
            "effective_rank":           round(eff_rank, 2),
            "pc1_alignment":            alignments,
        }

    # Summary: across all valid layers
    valid_pc1 = [v for v in pc1_explained_ratios if v is not None]
    valid_ranks = [v for v in effective_ranks if v is not None]
    mean_pc1 = float(np.mean(valid_pc1)) if valid_pc1 else 0.0
    mean_eff_rank = float(np.mean(valid_ranks)) if valid_ranks else 0.0

    # Interpretation
    if mean_pc1 > 0.7:
        concern = "HIGH_SHARED"
        note = (f"PC1 explains {mean_pc1:.0%} of variance on average across "
                f"layers. The {n_dims} dimension vectors share a dominant "
                f"direction — they are NOT independent. Likely a common "
                f"confound (verbosity, text-length) dominates.")
    elif mean_pc1 > 0.5:
        concern = "MODERATE_SHARED"
        note = (f"PC1 explains {mean_pc1:.0%} of variance. Some shared "
                f"structure exists, but dimensions retain partial independence.")
    else:
        concern = "INDEPENDENT"
        note = (f"PC1 explains only {mean_pc1:.0%}. Dimension vectors are "
                f"reasonably independent — good for targeted steering.")

    return {
        "dimension_names":         dim_names,
        "n_dimensions":            n_dims,
        "n_layers":                n_layers,
        "pc1_explained_per_layer": pc1_explained_ratios,
        "effective_rank_per_layer": effective_ranks,
        "mean_pc1_explained":      round(mean_pc1, 4),
        "mean_effective_rank":     round(mean_eff_rank, 2),
        "max_dimensions":          n_dims,
        "layer_details":           layer_results,
        "concern":                 concern,
        "note":                    note,
    }


# ═══════════════════════════════════════════════════════════════════════
# Pattern analysis helpers — analyse-only mode
# ═══════════════════════════════════════════════════════════════════════

def analyze_probe_patterns(probe_data: Dict) -> Dict[str, Dict]:
    """Classify each dimension's accuracy curve by shape."""
    results = {}
    for section in ("negotiation", "control"):
        if section not in probe_data:
            continue
        for dim_id, acc_list in probe_data[section].items():
            accs = np.array(acc_list)
            n = len(accs)
            mean_acc = float(accs.mean())
            std_acc = float(accs.std())
            min_acc = float(accs.min())
            max_acc = float(accs.max())
            best_l = int(np.argmax(accs))

            # Classify the pattern
            if min_acc >= 0.95 and (max_acc - min_acc) < 0.06:
                pattern = "flat_perfect"
                warning = (f"Near-perfect accuracy (>=0.95) at EVERY layer. "
                           f"With only {n} layers × ~24 samples this almost certainly means "
                           f"surface features (text length, punctuation style) "
                           f"are being detected, not a conceptual representation.")
            elif min_acc > 0.85 and (max_acc - min_acc) < 0.15:
                pattern = "flat_high"
                warning = ("Suspiciously uniform high accuracy across all layers. "
                           "Likely detecting surface features (e.g. text length) "
                           "rather than a layer-specific conceptual representation.")
            elif max_acc < 0.60:
                pattern = "flat_low"
                warning = "Concept not well-encoded at any layer."
            elif mean_acc > 0.85 and accs[-n // 4:].mean() < accs[:n // 4].mean() - 0.05:
                pattern = "decreasing_tail"
                warning = "Accuracy drops in later layers — concept fades."
            elif accs[-n // 4:].mean() > accs[:n // 4].mean() + 0.05:
                pattern = "rising"
                warning = ("Accuracy rises toward later layers — concept emerges "
                           "deeper. Earlier layers may be unsuitable for steering.")
            else:
                # Check for a clear peak in the middle
                third = n // 3
                early = accs[:third].mean()
                mid = accs[third:2 * third].mean()
                late = accs[2 * third:].mean()
                if mid > early + 0.05 and mid > late + 0.05:
                    pattern = "peaked"
                    warning = None  # expected healthy pattern
                else:
                    pattern = "other"
                    warning = None

            results[dim_id] = {
                "section":     section,
                "n_layers":    n,
                "mean_acc":    round(mean_acc, 4),
                "std_acc":     round(std_acc, 4),
                "min_acc":     round(min_acc, 4),
                "max_acc":     round(max_acc, 4),
                "best_layer":  best_l,
                "pattern":     pattern,
                "warning":     warning,
                "per_layer":   [round(float(a), 4) for a in accs],
            }
    return results


def analyze_bias_patterns(raw_bias: Dict[str, List[float]]) -> Dict[str, Dict]:
    """Analyse growth patterns in the raw bias (projection separation) data."""
    results = {}
    for key, vals in raw_bias.items():
        arr = np.array(vals)
        n = len(arr)
        if n == 0:
            continue

        # Linear trend (slope)
        x = np.arange(n)
        slope = float(np.polyfit(x, arr, 1)[0])

        # Ratio of late-layer to early-layer bias
        early_mean = float(arr[:n // 4].mean()) if n >= 4 else float(arr[0])
        late_mean = float(arr[-n // 4:].mean()) if n >= 4 else float(arr[-1])
        growth_ratio = late_mean / early_mean if abs(early_mean) > 1e-8 else float("inf")

        # Layer where bias exceeds 2× the early average
        threshold = early_mean * 2
        crossing_layer = None
        for l in range(n):
            if arr[l] > threshold:
                crossing_layer = l
                break

        results[key] = {
            "min":             round(float(arr.min()), 3),
            "max":             round(float(arr.max()), 3),
            "mean":            round(float(arr.mean()), 3),
            "slope":           round(slope, 4),
            "growth_ratio":    round(growth_ratio, 2),
            "early_mean":      round(early_mean, 3),
            "late_mean":       round(late_mean, 3),
            "doubling_layer":  crossing_layer,
        }
    return results


def compute_bias_correlation(raw_bias: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Pairwise Pearson correlation between all bias series.

    If firmness_vs_verbosity and empathy_vs_verbosity are highly correlated,
    both negotiation vectors are being confounded in the same way.
    """
    keys = sorted(raw_bias.keys())
    correlations = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = np.array(raw_bias[keys[i]])
            b = np.array(raw_bias[keys[j]])
            if len(a) == len(b) and len(a) > 1:
                r = float(np.corrcoef(a, b)[0, 1])
                correlations[f"{keys[i]}__vs__{keys[j]}"] = round(r, 4)
    return correlations


# ═══════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════

def generate_report(res: Dict) -> str:
    """Build a human-readable validation report from the results dict."""
    lines = []

    def _sec(title: str):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def _warn(msg: str):
        lines.append(f"  ⚠  {msg}")

    def _ok(msg: str):
        lines.append(f"  ✓  {msg}")

    def _info(msg: str):
        lines.append(f"     {msg}")

    _sec(f"VALIDATION REPORT — {res.get('model', '?')}   [{res.get('mode', '?')}]")
    _info(f"Timestamp: {res.get('timestamp', '?')}")

    # ── Length Confound ──────────────────────────────────────────────
    if "length_confound" in res:
        _sec("CHECK 1: Length Confound Analysis")
        _info("(word-level Cohen's d + character-level Cohen's d; severity = worst of both)")
        lc = res["length_confound"]
        any_severe = False
        for dim_id, stats in sorted(lc.items()):
            sev = stats["severity"]
            d = stats["cohens_d"]
            dc = stats.get("cohens_d_char", d)
            r = stats["length_ratio"]
            mp = stats["mean_pos_words"]
            mn = stats["mean_neg_words"]
            tag = f"[{sev}]"
            line = (f"  {dim_id:<30s}  pos={mp:5.1f}w  neg={mn:5.1f}w  "
                    f"ratio={r:.2f}x  d_word={d:+.2f}  d_char={dc:+.2f}  {tag}")
            lines.append(line)
            if sev in ("SEVERE", "MODERATE"):
                any_severe = True
        if any_severe:
            _warn("Positive samples are systematically longer than "
                  "negatives in several dimensions.")
            _info("ACTION: Rewrite pairs so positive and negative "
                  "samples match in word count,")
            _info("        or add a length-normalisation pre-processing step.")
        else:
            _ok("Length differences are within acceptable range.")

    # ── Vocabulary Overlap ───────────────────────────────────────────
    if "vocabulary_overlap" in res:
        _sec("CHECK 1b: Vocabulary Overlap Analysis")
        _info("Jaccard similarity of word sets (high = good overlap between pos/neg)")
        vo = res["vocabulary_overlap"]
        any_low = False
        for dim_id, stats in sorted(vo.items()):
            gj = stats["global_jaccard"]
            mpj = stats["mean_pair_jaccard"]
            concern = stats["concern"]
            tag = f"[{concern}]"
            line = (f"  {dim_id:<30s}  global_jaccard={gj:.3f}  "
                    f"pair_jaccard={mpj:.3f}  {tag}")
            lines.append(line)
            if stats.get("note"):
                _warn(stats["note"])
                any_low = True
        if not any_low:
            _ok("Vocabulary overlap is acceptable across all dimensions.")

    # ── Probe Patterns ───────────────────────────────────────────────
    if "probe_patterns" in res:
        _sec("CHECK 2: Probe Accuracy Pattern Analysis")
        pp = res["probe_patterns"]
        for dim_id, stats in sorted(pp.items()):
            pat = stats["pattern"]
            mean_a = stats["mean_acc"]
            best_l = stats["best_layer"]
            best_a = stats["max_acc"]
            sec = stats["section"]
            line = (f"  {dim_id:<25s}  [{sec:<11s}]  pattern={pat:<16s}  "
                    f"mean={mean_a:.3f}  best_layer={best_l}  best={best_a:.3f}")
            lines.append(line)
            if stats["warning"]:
                _warn(stats["warning"])

    # ── Permutation Test ─────────────────────────────────────────────
    if "permutation_test" in res:
        _sec("CHECK 2b: Permutation Test (statistical significance)")
        pt = res["permutation_test"]
        for dim_id, layer_results in sorted(pt.items()):
            lines.append(f"  --- {dim_id} ---")
            n_sig = sum(1 for v in layer_results.values()
                        if isinstance(v, dict) and v.get("significant", False))
            n_total = len(layer_results)
            lines.append(f"  Significant layers: {n_sig}/{n_total}")
            # Show a sample of layers
            for l in sorted(layer_results.keys(), key=int):
                v = layer_results[l]
                if isinstance(v, dict) and "real_acc" in v:
                    sig_mark = " *" if v["significant"] else ""
                    lines.append(
                        f"    layer {int(l):>2d}:  real={v['real_acc']:.3f}  "
                        f"null={v['null_mean']:.3f}±{v['null_std']:.3f}  "
                        f"p={v['p_value']:.3f}{sig_mark}"
                    )

    # ── Cosine Similarity ────────────────────────────────────────────
    if "cosine_similarity" in res:
        _sec("CHECK 3: Cosine Similarity Between Steering Directions")
        cs = res["cosine_similarity"]
        if isinstance(cs, dict) and "note" in cs:
            _info(cs["note"])
        else:
            # Show a summary for select layers
            for layer_key in sorted(cs.keys(), key=lambda k: int(k)):
                data = cs[layer_key]
                names = data.get("names", [])
                matrix = data.get("matrix", [])
                warnings = data.get("warnings", [])
                lines.append(f"\n  Layer {layer_key}:")
                if warnings:
                    for w in warnings:
                        _warn(f"cos({w['dim_a']}, {w['dim_b']}) = {w['cosine']:.3f}")
                else:
                    _ok("No concerning cross-category overlap at this layer.")

    # ── Cohen's d Bias ───────────────────────────────────────────────
    if "cohens_d_bias" in res:
        _sec("CHECK 4: Normalised Bias (Cohen's d)")
        cdb = res["cohens_d_bias"]
        for key, data in sorted(cdb.items()):
            max_d = data["max_abs_d"]
            max_l = data["max_abs_d_layer"]
            n_sev = data["n_severe"]
            sev_tag = "SEVERE" if n_sev > 0 else "OK"
            lines.append(f"  {key:<40s}  max|d|={max_d:.2f} @ layer {max_l}  "
                         f"severe_layers={n_sev}  [{sev_tag}]")
            if n_sev > 0:
                _warn(f"{n_sev} layers have |Cohen's d| > 0.8 — "
                      "the steering vector separates this control trait there.")

    # ── Pair Consistency ─────────────────────────────────────────────
    if "pair_consistency" in res:
        _sec("CHECK 5: Per-Pair Consistency (Alignment)")
        pc = res["pair_consistency"]
        for dim_id, data in sorted(pc.items()):
            om = data["overall_mean"]
            no = data["n_outliers"]
            tag = "GOOD" if om > 0.5 and no == 0 else ("WEAK" if om < 0.3 else "OK")
            lines.append(f"  {dim_id:<25s}  mean_alignment={om:.3f}  "
                         f"outlier_pairs={no}  [{tag}]")
            if no > 0:
                _warn(f"Pairs {data['outlier_pairs']} have low alignment — "
                      "consider rewriting them.")

    # ── Steering Direction Probe ─────────────────────────────────────
    if "steering_direction_probe" in res:
        _sec("CHECK 6: Steering-Direction Probe (1-D)")
        sdp = res["steering_direction_probe"]
        for dim_id, data in sorted(sdp.items()):
            ma = data["mean_acc"]
            ba = data["best_acc"]
            bl = data["best_layer"]
            lines.append(f"  {dim_id:<25s}  mean_1d_acc={ma:.3f}  "
                         f"best_1d={ba:.3f} @ layer {bl}")

    # ── Bias Analysis (raw, from existing results) ───────────────────
    if "bias_analysis" in res:
        _sec("Bias Growth Analysis (raw projection separations)")
        ba = res["bias_analysis"]
        for key, data in sorted(ba.items()):
            gr = data["growth_ratio"]
            sl = data["slope"]
            dl = data.get("doubling_layer", "n/a")
            lines.append(f"  {key:<40s}  growth_ratio={gr:.1f}x  "
                         f"slope={sl:.3f}  doubles_at_layer={dl}")
            if isinstance(gr, float) and gr > 5:
                _warn(f"Bias grows {gr:.0f}× from early to late layers — "
                      "later layers disproportionately amplify this confound.")

    # ── Bias Correlation ─────────────────────────────────────────────
    if "bias_correlation" in res and res["bias_correlation"]:
        _sec("Bias Correlation (do different vectors share the same confound?)")
        bc = res["bias_correlation"]
        for key, r in sorted(bc.items()):
            tag = "HIGH" if abs(r) > 0.95 else ("MODERATE" if abs(r) > 0.8 else "LOW")
            lines.append(f"  {key:<60s}  r={r:+.3f}  [{tag}]")
            if abs(r) > 0.95:
                _warn("These two bias series are nearly identical — both "
                      "vectors are confounded in the same way.")

    # ── Selectivity ──────────────────────────────────────────────────
    if "selectivity" in res:
        _sec("CHECK 7: Selectivity & Layer Recommendations")
        sel = res["selectivity"]
        for dim_id, data in sorted(sel.items()):
            bl = data["best_layer"]
            bs = data.get("best_selectivity", "?")
            lines.append(f"  {dim_id:<25s}  best_layer={bl}  selectivity={bs}")

    # ── Layer Recommendations ────────────────────────────────────────
    if "layer_recommendations" in res:
        _sec("Layer Recommendations")
        lr = res["layer_recommendations"]
        for dim_id, data in sorted(lr.items()):
            rec = data.get("recommended", [])
            avoid = data.get("avoid", [])
            crit = data.get("criteria", "")
            top5 = data.get("top5_layers")
            top5s = data.get("top5_scores")
            sens = data.get("sensitivity")
            lines.append(f"  {dim_id}:")
            lines.append(f"    RECOMMEND: {rec if rec else 'none found'}")
            if top5:
                t5_str = ", ".join(f"L{l}({s:.3f})" for l, s in zip(top5, top5s))
                lines.append(f"    TOP-5:     {t5_str}")
            lines.append(f"    AVOID:     {avoid if avoid else 'none'}")
            lines.append(f"    Criteria:  {crit}")
            if sens:
                sens_str = "  ".join(f"{k}→{v}" for k, v in sens.items())
                lines.append(f"    Sensitivity: {sens_str}")

    # ── Vector Norm Profile ──────────────────────────────────────────
    if "vector_norms" in res:
        vn = res["vector_norms"]
        if "note" in vn:
            pass  # skip if no data
        else:
            _sec("CHECK 8: Vector Norm Profile")
            summary = vn.get("summary", {})
            _info(f"Dimensions analysed: {summary.get('n_dimensions', '?')}")
            if summary.get("all_unit_normed"):
                _ok("All vectors are unit-normed as expected.")
            else:
                _warn("Not all vectors are unit-normed — check extraction pipeline.")
            # Report NaN layers
            has_nan = False
            for dim_id, data in sorted(vn.get("dimensions", {}).items()):
                n_nan = data.get("n_nan_layers", 0)
                if n_nan > 0:
                    has_nan = True
            if has_nan:
                _warn("Some layers contain NaN/Inf values — these layers were "
                      "likely not computed during extraction (e.g. layers beyond "
                      "the model's effective depth).")
            for dim_id, data in sorted(vn.get("dimensions", {}).items()):
                unit_tag = "unit-normed" if data["all_layers_unit_normed"] else "NOT NORMED"
                n_nan = data.get("n_nan_layers", 0)
                nan_str = f"  ⚠ {n_nan} NaN layers" if n_nan > 0 else ""
                lines.append(
                    f"  {dim_id:<30s}  [{unit_tag}]  shape=({data['n_layers']}, "
                    f"{data['hidden_dim']})  valid={data.get('n_valid_layers', '?')}/"
                    f"{data['n_layers']}  norm: mean={data['norm_mean']:.4f}  "
                    f"std={data['norm_std']:.4f}  range=[{data['norm_min']:.4f}, "
                    f"{data['norm_max']:.4f}]{nan_str}"
                )

    # ── Inter-Layer Consistency ──────────────────────────────────────
    if "layer_consistency" in res:
        _sec("CHECK 9: Inter-Layer Consistency")
        _info("Cosine similarity between consecutive layer vectors")
        lc = res["layer_consistency"]
        for dim_id, data in sorted(lc.items()):
            if isinstance(data, str):
                continue
            mc = data["mean_consecutive_cos"]
            mn = data["min_consecutive_cos"]
            nt = data["n_transitions"]
            pat = data["pattern"]
            lines.append(
                f"  {dim_id:<30s}  mean_cos={mc:.3f}  min_cos={mn:.3f}  "
                f"transitions={nt}  [{pat}]"
            )
            if data.get("note"):
                _info(f"    → {data['note']}")

    # ── Cross-Dimension Similarity ───────────────────────────────────
    if "cross_dimension_similarity" in res:
        cds = res["cross_dimension_similarity"]
        if "note" not in cds:
            _sec("CHECK 10: Cross-Dimension Similarity (all layers)")
            _info(f"Dimensions: {cds.get('dimension_names', [])}")
            _info(f"Best independence at layer {cds['best_independence_layer']} "
                  f"(avg |cos|={cds['best_avg_abs_cos']:.3f})")
            _info(f"Worst collapse  at layer {cds['worst_collapse_layer']} "
                  f"(avg |cos|={cds['worst_avg_abs_cos']:.3f})")
            if cds.get("collapse_layers"):
                _warn(f"Collapse layers (>50% of pairs |cos|>0.5): "
                      f"{cds['collapse_layers']}")
            if cds.get("independence_layers"):
                _ok(f"Independence layers (all pairs |cos|<0.3): "
                    f"{cds['independence_layers']}")
            # Per-pair summary
            for pname, pdata in sorted(cds.get("pair_summaries", {}).items()):
                lines.append(
                    f"  {pname:<50s}  mean={pdata['mean_cos']:+.3f}  "
                    f"max|cos|={pdata['max_abs_cos']:.3f}"
                )

    # ── Vector Concentration ─────────────────────────────────────────
    if "vector_concentration" in res:
        _sec("CHECK 11: Vector Concentration / Effective Dimensionality")
        vc = res["vector_concentration"]
        for dim_id, data in sorted(vc.items()):
            if not isinstance(data, dict) or "pattern" not in data:
                continue
            ed = data["mean_eff_dim"]
            rnd = data["random_expected_eff_dim"]
            ratio = data["ratio_to_random"]
            gini = data["mean_gini"]
            pat = data["pattern"]
            lines.append(
                f"  {dim_id:<30s}  eff_dim={ed:.0f} / {rnd:.0f} random  "
                f"ratio={ratio:.2f}  gini={gini:.3f}  [{pat}]"
            )
            if data.get("note"):
                _info(f"    → {data['note']}")

    # ── Shared Subspace Analysis ─────────────────────────────────────
    if "shared_subspace" in res:
        ss = res["shared_subspace"]
        if "note" not in ss or "concern" in ss:
            _sec("CHECK 12: Shared Subspace Analysis (PCA)")
            _info(f"Dimensions: {ss.get('dimension_names', '?')}")
            _info(f"Mean PC1 explained: {ss.get('mean_pc1_explained', 0):.1%}")
            _info(f"Mean effective rank: {ss.get('mean_effective_rank', 0):.2f} "
                  f"/ {ss.get('max_dimensions', '?')} dimensions")
            concern = ss.get("concern", "?")
            if concern == "HIGH_SHARED":
                _warn(ss.get("note", ""))
            elif concern == "MODERATE_SHARED":
                _info(f"  Note: {ss.get('note', '')}")
            else:
                _ok(ss.get("note", ""))

            # Show PC1 alignment at a sample layer (middle)
            ld = ss.get("layer_details", {})
            mid_l = ss.get("n_layers", 36) // 2
            if mid_l in ld or str(mid_l) in ld:
                mid_data = ld.get(mid_l, ld.get(str(mid_l), {}))
                if "pc1_alignment" in mid_data:
                    _info(f"  PC1 alignment at layer {mid_l}:")
                    for d, a in sorted(mid_data["pc1_alignment"].items(),
                                       key=lambda x: -abs(x[1])):
                        lines.append(f"    {d:<30s}  cos(PC1)={a:+.3f}")

    # ── CHECK 13: Dose-Response Validation ─────────────────────────────
    dr = res.get("dose_response", {})
    if dr and "dimensions" in dr:
        _sec("CHECK 13: Dose-Response Validation (LLM Judge)")
        dr_md = dr.get("metadata", {})
        _info(f"Alphas tested: {dr.get('alphas', '?')}")
        _info(f"Judges: {', '.join(dr.get('judges_used', ['?']))}")
        lines.append("")
        for dim_name, dim_data in dr["dimensions"].items():
            verdict = dim_data.get("verdict", "?")
            targets = dim_data.get("target_judge_dims", [])
            verdict_icon = {
                "PASS": "PASS", "WEAK_PASS": "WEAK",
                "PARTIAL": "PART", "FAIL": "FAIL",
            }.get(verdict, "?")
            _info(f"  {dim_name} [{verdict_icon}] (targets: {', '.join(targets)})")
            for jn, ja in dim_data.get("judge_analysis", {}).items():
                # Show monotonicity for target dims
                for td in targets:
                    mono = ja.get("monotonicity", {}).get(td, {})
                    rho = mono.get("spearman_rho")
                    p = mono.get("p_value")
                    if rho is not None:
                        sig = "***" if mono.get("significant") else ""
                        lines.append(
                            f"      {td}: rho={rho:+.3f}, p={p:.4f} {sig}"
                        )
                spec = ja.get("specificity", {})
                if spec.get("ratio") is not None:
                    lines.append(
                        f"      Specificity ratio: {spec['ratio']:.2f} "
                        f"({'specific' if spec['specific'] else 'not specific'})"
                    )
                coh = ja.get("coherence_degradation", {})
                if coh.get("degrades"):
                    lines.append(
                        f"      Coherence degrades at alpha={coh['degradation_alpha']}"
                    )
            lines.append("")
        dr_summ = dr.get("summary", {})
        ov = dr_summ.get("overall_verdict", "?")
        if ov == "PASS":
            _ok(f"Overall dose-response: {ov} "
                f"({dr_summ.get('pass_count', 0)}/{dr_summ.get('total_dimensions', 0)} pass)")
        elif ov == "FAIL":
            _warn(f"Overall dose-response: {ov} "
                  f"({dr_summ.get('fail_count', 0)}/{dr_summ.get('total_dimensions', 0)} fail)")
        else:
            _info(f"Overall dose-response: {ov} "
                  f"({dr_summ.get('pass_count', 0)} pass, "
                  f"{dr_summ.get('fail_count', 0)} fail "
                  f"/ {dr_summ.get('total_dimensions', 0)} total)")

    # ── Per-Dimension Traffic Light ────────────────────────────────────
    _sec("PER-DIMENSION SUMMARY (traffic light)")
    _info("  RED = severe problems    AMBER = needs attention    GREEN = acceptable")

    # Gather all negotiation dimension IDs from available data
    all_dims_set = set()
    for source in ("probe_patterns", "length_confound", "selectivity"):
        if source in res:
            all_dims_set.update(res[source].keys())
    # Filter to negotiation-only (exclude control dims from traffic light)
    con_ids = {"verbosity", "formality"}
    neg_dims_set = all_dims_set - con_ids

    for dim_id in sorted(neg_dims_set):
        flags = []

        # Length confound flag
        lc_sev = res.get("length_confound", {}).get(dim_id, {}).get("severity", "?")
        if lc_sev in ("SEVERE", "MODERATE"):
            flags.append("length")

        # Probe pattern flag
        pat = res.get("probe_patterns", {}).get(dim_id, {}).get("pattern", "?")
        if pat in ("flat_perfect", "flat_high"):
            flags.append("flat_probe")
        elif pat == "flat_low":
            flags.append("low_probe")

        # Bias growth flag
        ba = res.get("bias_analysis", {})
        for bk, bv in ba.items():
            if bk.startswith(dim_id + "_vs_"):
                gr = bv.get("growth_ratio", 0)
                if isinstance(gr, (int, float)) and gr > 5:
                    flags.append("bias_growth")
                    break

        # Cohen's d flag (full mode)
        for ck, cv in res.get("cohens_d_bias", {}).items():
            if ck.startswith(dim_id + "_"):
                if cv.get("n_severe", 0) > 0:
                    flags.append("cohens_d")
                    break

        # Vocabulary overlap flag
        vo_concern = res.get("vocabulary_overlap", {}).get(dim_id, {}).get("concern", "OK")
        if vo_concern == "CRITICAL_OVERLAP":
            flags.append("vocab_critical")

        # Vector concentration flag (raw vectors)
        vc_pat = res.get("vector_concentration", {}).get(dim_id, {}).get("pattern", "")
        if vc_pat == "highly_concentrated":
            flags.append("vec_concentrated")

        # Layer consistency flag (raw vectors)
        lc_pat = res.get("layer_consistency", {}).get(dim_id, {}).get("pattern", "")
        if lc_pat in ("variable", "inconsistent"):
            flags.append("unstable_layers")

        # NaN layers flag (raw vectors)
        vn_data = res.get("vector_norms", {}).get("dimensions", {}).get(dim_id, {})
        n_nan = vn_data.get("n_nan_layers", 0)
        if n_nan > 0:
            flags.append("nan_layers")

        # Dose-response flag (functional validation)
        dr_dims = res.get("dose_response", {}).get("dimensions", {})
        dr_verdict = dr_dims.get(dim_id, {}).get("verdict", "")
        if dr_verdict == "FAIL":
            flags.append("dose_fail")
        elif dr_verdict == "WEAK_PASS":
            flags.append("dose_weak")

        # Determine colour
        if len(flags) >= 2:
            colour = "RED"
        elif len(flags) == 1:
            colour = "AMBER"
        else:
            colour = "GREEN"

        icon = {"RED": "🔴", "AMBER": "🟡", "GREEN": "🟢"}[colour]
        flag_str = ", ".join(flags) if flags else "—"
        lines.append(f"  {icon} {dim_id:<30s}  [{colour}]  flags: {flag_str}")

    # ── Quantitative Validity Score ──────────────────────────────────
    _sec("VALIDITY SCORE")
    _info("Scoring: start at 100, deduct points per red flag.")
    score = 100.0
    deductions = []

    # Length confound deductions
    n_severe_lc = sum(1 for v in res.get("length_confound", {}).values()
                      if v.get("severity") in ("SEVERE", "MODERATE"))
    if n_severe_lc > 0:
        pts = min(n_severe_lc * 5, 30)
        score -= pts
        deductions.append(f"{n_severe_lc} dimensions with length confound: −{pts}")

    # Vocabulary overlap deductions
    n_critical_vocab = sum(1 for v in res.get("vocabulary_overlap", {}).values()
                           if v.get("concern") == "CRITICAL_OVERLAP")
    n_low_vocab = sum(1 for v in res.get("vocabulary_overlap", {}).values()
                      if v.get("concern") in ("LOW_OVERLAP", "LOW_PAIR_OVERLAP"))
    if n_critical_vocab > 0:
        pts = min(n_critical_vocab * 3, 15)
        score -= pts
        deductions.append(f"{n_critical_vocab} dims with critical vocab overlap: −{pts}")
    if n_low_vocab > 0:
        pts = min(n_low_vocab * 1, 10)
        score -= pts
        deductions.append(f"{n_low_vocab} dims with low vocab overlap: −{pts}")

    # Flat probe deductions
    n_flat = sum(1 for v in res.get("probe_patterns", {}).values()
                 if v.get("pattern") in ("flat_perfect", "flat_high"))
    if n_flat > 0:
        pts = min(n_flat * 5, 25)
        score -= pts
        deductions.append(f"{n_flat} flat-high/perfect probe patterns: −{pts}")

    # Bias growth deductions
    n_explosive = sum(
        1 for v in res.get("bias_analysis", {}).values()
        if isinstance(v.get("growth_ratio"), (int, float)) and v["growth_ratio"] > 5
    )
    if n_explosive > 0:
        pts = min(n_explosive * 8, 20)
        score -= pts
        deductions.append(f"{n_explosive} explosive bias growth series: −{pts}")

    # High bias correlation
    n_high_corr = sum(1 for v in res.get("bias_correlation", {}).values()
                      if isinstance(v, float) and abs(v) > 0.95)
    if n_high_corr > 0:
        pts = min(n_high_corr * 5, 15)
        score -= pts
        deductions.append(f"{n_high_corr} nearly-identical bias correlations: −{pts}")

    # Sample size penalty
    n_all_dims = len(all_dims_set)
    pp = res.get("probe_patterns", {})
    if pp:
        typical_n = next(iter(pp.values()), {}).get("n_layers", 36)
        n_samples_approx = 24 if typical_n == 36 else 12  # rough
        if n_samples_approx < 30:
            pts = 10
            score -= pts
            deductions.append(f"Small sample size (~{n_samples_approx} per dim): −{pts}")

    # Shared subspace deduction
    ss_concern = res.get("shared_subspace", {}).get("concern", "")
    if ss_concern == "HIGH_SHARED":
        pts = 15
        score -= pts
        deductions.append(f"High shared subspace (PC1 dominates): −{pts}")
    elif ss_concern == "MODERATE_SHARED":
        pts = 5
        score -= pts
        deductions.append(f"Moderate shared subspace: −{pts}")

    # Vector concentration deduction
    vc_data = res.get("vector_concentration", {})
    n_concentrated = sum(1 for v in vc_data.values()
                         if isinstance(v, dict) and
                         v.get("pattern") in ("concentrated", "highly_concentrated"))
    if n_concentrated > 0:
        pts = min(n_concentrated * 5, 15)
        score -= pts
        deductions.append(f"{n_concentrated} concentrated vectors: −{pts}")

    # Layer inconsistency deduction
    lc_data = res.get("layer_consistency", {})
    n_unstable = sum(1 for v in lc_data.values()
                     if isinstance(v, dict) and
                     v.get("pattern") in ("variable", "inconsistent"))
    if n_unstable > 0:
        pts = min(n_unstable * 5, 10)
        score -= pts
        deductions.append(f"{n_unstable} layer-inconsistent dimensions: −{pts}")

    # NaN layer deduction
    vn_dims = res.get("vector_norms", {}).get("dimensions", {})
    n_nan_dims = sum(1 for v in vn_dims.values()
                     if v.get("n_nan_layers", 0) > 0)
    if n_nan_dims > 0:
        max_nan = max(v.get("n_nan_layers", 0) for v in vn_dims.values())
        total_layers = next(iter(vn_dims.values()), {}).get("n_layers", 36)
        nan_frac = max_nan / max(total_layers, 1)
        pts = round(min(nan_frac * 30, 15))  # up to -15 if half layers are NaN
        score -= pts
        deductions.append(
            f"{n_nan_dims} dims with NaN layers (up to {max_nan}/{total_layers}): −{pts}")

    # Dose-response deductions
    dr_data = res.get("dose_response", {}).get("dimensions", {})
    n_dr_fail = sum(1 for v in dr_data.values() if v.get("verdict") == "FAIL")
    n_dr_weak = sum(1 for v in dr_data.values() if v.get("verdict") == "WEAK_PASS")
    if n_dr_fail > 0:
        pts = min(n_dr_fail * 10, 20)
        score -= pts
        deductions.append(f"{n_dr_fail} dose-response FAIL dimensions: −{pts}")
    if n_dr_weak > 0:
        pts = min(n_dr_weak * 3, 9)
        score -= pts
        deductions.append(f"{n_dr_weak} dose-response WEAK_PASS dimensions: −{pts}")

    score = max(score, 0.0)
    for d in deductions:
        _info(f"  {d}")
    lines.append("")
    lines.append(f"  TOTAL SCORE: {score:.0f}/100")
    if score >= 80:
        _info("Interpretation: GOOD — minor issues only. Vectors suitable for experiments.")
    elif score >= 60:
        _info("Interpretation: FAIR — some confounds detected. Interpret results with caution.")
    elif score >= 40:
        _info("Interpretation: NEEDS WORK — significant confounding. Results are not reliable without mitigation.")
    elif score >= 15:
        _info("Interpretation: POOR — major rework of contrastive pairs needed before using vectors.")
    else:
        _info("Interpretation: CRITICAL — vectors are likely encoding surface features, not target concepts. Complete redesign recommended.")

    # ── Warnings & Recommendations ───────────────────────────────────
    if "warnings" in res and res["warnings"]:
        _sec("SUMMARY — Warnings")
        for w in res["warnings"]:
            _warn(w)

    if "recommendations" in res and res["recommendations"]:
        _sec("SUMMARY — Recommended Actions")
        for i, r in enumerate(res["recommendations"], 1):
            _info(f"{i}. {r}")

    # ── Overall Assessment ───────────────────────────────────────────
    _sec("OVERALL ASSESSMENT")
    assessment = res.get("overall_assessment", "See individual checks above.")
    lines.append(f"  {assessment}")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — analyse-only mode
# ═══════════════════════════════════════════════════════════════════════

def run_analyze_only(args) -> Dict:
    """Analyse existing probe_results.json — no GPU, no model loading."""
    alias = _ALIAS_MAP[args.model]

    # ── load existing probe results ──────────────────────────────────
    probe_path = Path(args.probe_dir) / alias / "probe_results.json"
    if not probe_path.exists():
        log.error("Probe results not found at %s", probe_path)
        return {}
    with open(probe_path, encoding="utf-8") as f:
        probe_data = json.load(f)
    log.info("Loaded probe results from %s", probe_path)

    # ── load pairs ───────────────────────────────────────────────────
    with open(args.negotiation_pairs, encoding="utf-8") as f:
        neg_data = json.load(f)
    with open(args.control_pairs, encoding="utf-8") as f:
        con_data = json.load(f)

    all_dims = neg_data["dimensions"] + con_data["dimensions"]
    n_neg_pairs = len(neg_data["dimensions"][0]["pairs"]) if neg_data["dimensions"] else 0
    n_con_pairs = len(con_data["dimensions"][0]["pairs"]) if con_data["dimensions"] else 0

    results: Dict[str, Any] = {
        "model":     alias,
        "mode":      "analyze-only",
        "timestamp": datetime.now().isoformat(),
    }

    # ── Check 1: length confound ─────────────────────────────────────
    log.info("Check 1: Length confound analysis")
    results["length_confound"] = analyze_length_confound(all_dims)

    # ── Check 1b: vocabulary overlap ─────────────────────────────────
    log.info("Check 1b: Vocabulary overlap analysis")
    results["vocabulary_overlap"] = analyze_vocabulary_overlap(all_dims)

    # ── Check 2a: probe accuracy patterns ────────────────────────────
    log.info("Check 2a: Probe accuracy patterns")
    results["probe_patterns"] = analyze_probe_patterns(probe_data)

    # ── Bias growth analysis (raw) ───────────────────────────────────
    log.info("Analysing bias growth patterns")
    raw_bias = probe_data.get("bias", {})
    results["bias_analysis"] = analyze_bias_patterns(raw_bias)

    # ── Bias correlation ─────────────────────────────────────────────
    log.info("Computing bias correlation")
    results["bias_correlation"] = compute_bias_correlation(raw_bias)

    # ── Approximate selectivity ──────────────────────────────────────
    log.info("Computing approximate selectivity")
    neg_accs = probe_data.get("negotiation", {})
    results["selectivity"] = compute_approximate_selectivity(neg_accs, raw_bias)

    # ── Approximate layer recommendations ────────────────────────────
    log.info("Computing layer recommendations")
    results["layer_recommendations"] = compute_approx_layer_recommendation(
        neg_accs, raw_bias
    )

    # ── Raw vector analysis (only if vectors on disk) ───────────────
    vectors_dir = Path(args.vectors_dir) / alias
    # Fallback: try vectors/ if vectors_gpu/ doesn't have this model
    if not vectors_dir.exists():
        fallback = Path("vectors") / alias
        if fallback.exists():
            vectors_dir = fallback
            log.info("Using fallback vectors dir: %s", fallback)
    if vectors_dir.exists():
        log.info("Vectors found on disk — running raw vector analysis")

        # Cosine similarity
        n_layers = len(next(iter(neg_accs.values()))) if neg_accs else 36
        sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
        results["cosine_similarity"] = compute_cosine_from_files(
            vectors_dir, layers=sample_layers
        )

        # CHECK 8: Vector norm profile
        log.info("Check 8: Vector norm profile")
        results["vector_norms"] = analyze_vector_norms(vectors_dir)

        # CHECK 9: Inter-layer consistency
        log.info("Check 9: Inter-layer consistency")
        results["layer_consistency"] = analyze_layer_consistency(vectors_dir)

        # CHECK 10: Cross-dimension similarity
        log.info("Check 10: Cross-dimension similarity (all layers)")
        results["cross_dimension_similarity"] = analyze_cross_dimension_similarity(
            vectors_dir
        )

        # CHECK 11: Vector concentration
        log.info("Check 11: Vector concentration / effective dimensionality")
        results["vector_concentration"] = analyze_vector_concentration(vectors_dir)

        # CHECK 12: Shared subspace analysis
        log.info("Check 12: Shared subspace analysis (PCA)")
        results["shared_subspace"] = analyze_shared_subspace(vectors_dir)
    else:
        results["cosine_similarity"] = {
            "note": ("vectors/ directory not found. Run extract_vectors.py "
                     "on both negotiation and control pairs, then re-run "
                     "this script to get cosine similarity analysis.")
        }

    # ── CHECK 13: Dose-response validation (if available) ────────────
    dose_analysis_path = Path("results/dose_response/dose_response_analysis.json")
    dose_scores_path = Path("results/dose_response_scores.json")
    if dose_analysis_path.exists():
        log.info("Check 13: Loading dose-response analysis results")
        try:
            with open(dose_analysis_path, encoding="utf-8") as f:
                results["dose_response"] = json.load(f)
        except Exception as e:
            log.warning("Failed to load dose-response analysis: %s", e)
    elif dose_scores_path.exists():
        log.info("Check 13: Running dose-response analysis from scores")
        try:
            from dose_response_validation import analyze_dose_response
            results["dose_response"] = analyze_dose_response(
                scores_file=str(dose_scores_path),
                output_dir="results/dose_response",
            )
        except Exception as e:
            log.warning("Failed to run dose-response analysis: %s", e)
    else:
        log.info(
            "Check 13: No dose-response data found. "
            "Run dose_response_validation.py to generate it."
        )

    # ── Compile warnings & recommendations ───────────────────────────
    warnings = []
    recommendations = []

    # Sample size warning
    warnings.append(
        f"Negotiation dimensions have {n_neg_pairs} pairs each "
        f"({2 * n_neg_pairs} samples). Control dimensions have "
        f"{n_con_pairs} pairs each ({2 * n_con_pairs} samples). "
        f"Statistical power is limited."
    )
    recommendations.append(
        "Increase contrastive pairs to 30+ per dimension for robust probing."
    )

    # Length confound
    severe_lc = [k for k, v in results["length_confound"].items()
                 if v["severity"] in ("SEVERE", "MODERATE")]
    if severe_lc:
        warnings.append(
            f"Significant length confound in: {', '.join(severe_lc)}. "
            f"Positive samples are systematically longer."
        )
        recommendations.append(
            "Rewrite contrastive pairs so positive and negative responses "
            "are length-matched (similar word/token counts)."
        )

    # Flat-high probes
    flat_high = [k for k, v in results["probe_patterns"].items()
                 if v["pattern"] in ("flat_high", "flat_perfect")]
    if flat_high:
        warnings.append(
            f"Probe accuracy is uniformly high for: {', '.join(flat_high)}. "
            f"This likely reflects surface features, not layer-specific encoding."
        )
        recommendations.append(
            "Run with --full mode to perform permutation tests and verify "
            "statistical significance at each layer."
        )

    # Bias growth
    explosive_bias = [k for k, v in results["bias_analysis"].items()
                      if isinstance(v.get("growth_ratio"), (int, float)) and v["growth_ratio"] > 5]
    if explosive_bias:
        warnings.append(
            f"Bias grows >5x from early to late layers for: {', '.join(explosive_bias)}. "
            f"Later layers amplify verbosity/formality confounds."
        )
        recommendations.append(
            "Prefer steering at earlier/middle layers where bias is lower."
        )

    # Bias correlation
    high_corr = [k for k, v in results.get("bias_correlation", {}).items()
                 if isinstance(v, float) and abs(v) > 0.95]
    if high_corr:
        warnings.append(
            "Multiple negotiation vectors share nearly identical bias patterns — "
            "they may be capturing the same confound."
        )

    recommendations.append(
        "Extract control steering vectors (run extract_vectors.py on "
        "control_steering_pairs.json) to enable cosine similarity analysis."
    )
    recommendations.append(
        "Run --full validation to get Cohen's d, permutation tests, "
        "pair consistency, and steering-direction probes."
    )

    # NaN layer warnings
    if "vector_norms" in results:
        vn_dims = results["vector_norms"].get("dimensions", {})
        nan_dims = {d: v["n_nan_layers"] for d, v in vn_dims.items()
                    if v.get("n_nan_layers", 0) > 0}
        if nan_dims:
            nan_desc = ", ".join(f"{d} ({n} layers)" for d, n in nan_dims.items())
            warnings.append(
                f"NaN/Inf layers detected: {nan_desc}. These layers were "
                f"likely not computed during extraction. Avoid using them "
                f"for steering."
            )

    # Raw vector warnings
    if "shared_subspace" in results:
        ss = results["shared_subspace"]
        if ss.get("concern") == "HIGH_SHARED":
            warnings.append(
                f"Shared subspace analysis: PC1 explains "
                f"{ss['mean_pc1_explained']:.0%} of variance across dimensions. "
                f"Vectors share a dominant direction (likely a common confound)."
            )
            recommendations.append(
                "Investigate what PC1 represents — compare with verbosity/length "
                "direction. Consider orthogonalising dimension vectors against "
                "the shared component."
            )

    if "vector_concentration" in results:
        vc = results["vector_concentration"]
        concentrated = [d for d, v in vc.items()
                        if isinstance(v, dict) and v.get("pattern") in
                        ("concentrated", "highly_concentrated")]
        if concentrated:
            warnings.append(
                f"Vector concentration: {', '.join(concentrated)} have low "
                f"effective dimensionality — may capture surface features."
            )

    if "layer_consistency" in results:
        lc = results["layer_consistency"]
        unstable = [d for d, v in lc.items()
                    if isinstance(v, dict) and v.get("pattern") in
                    ("variable", "inconsistent")]
        if unstable:
            warnings.append(
                f"Layer consistency: {', '.join(unstable)} have unstable "
                f"steering directions across layers — the concept may not "
                f"have a coherent representation."
            )

    if "cross_dimension_similarity" in results:
        cds = results["cross_dimension_similarity"]
        if cds.get("collapse_layers"):
            warnings.append(
                f"Dimension collapse at layers: {cds['collapse_layers']}. "
                f"Avoid steering at these layers — dimensions are not "
                f"distinguishable."
            )
        if cds.get("independence_layers"):
            recommendations.append(
                f"Best layers for dimension-independent steering: "
                f"{cds['independence_layers']}."
            )

    # Dose-response warnings
    dr = results.get("dose_response", {})
    if dr and "dimensions" in dr:
        dr_fails = [d for d, v in dr["dimensions"].items()
                    if v.get("verdict") == "FAIL"]
        dr_passes = [d for d, v in dr["dimensions"].items()
                     if v.get("verdict") in ("PASS", "WEAK_PASS")]
        if dr_fails:
            warnings.append(
                f"Dose-response FAIL: {', '.join(dr_fails)} — increasing "
                f"alpha does NOT monotonically increase the target trait. "
                f"The vector may not encode what we think."
            )
        if dr_passes:
            recommendations.append(
                f"Dose-response PASS: {', '.join(dr_passes)} show monotonic "
                f"trait increase with alpha. These vectors have functional validity."
            )
        # Check for length confound in dose-response
        for dim_name, dim_data in dr["dimensions"].items():
            for jn, ja in dim_data.get("judge_analysis", {}).items():
                la = ja.get("length_analysis", {})
                if la.get("length_changes_with_alpha"):
                    warnings.append(
                        f"Dose-response length confound: {dim_name} — "
                        f"response length changes with alpha (rho="
                        f"{la['length_spearman_rho']:.3f}). "
                        f"Judge scores may reflect length, not the target trait."
                    )
                    break
    elif not dr:
        recommendations.append(
            "Run dose_response_validation.py to test whether vectors "
            "produce monotonic behavioral changes (gold-standard validation)."
        )

    # Overall assessment
    n_severe = len(severe_lc) + len(flat_high) + len(explosive_bias)
    # Add raw vector severity
    if results.get("shared_subspace", {}).get("concern") == "HIGH_SHARED":
        n_severe += 1
    if results.get("cross_dimension_similarity", {}).get("collapse_layers"):
        n_severe += 1
    # Add dose-response severity
    dr_summary = results.get("dose_response", {}).get("summary", {})
    if dr_summary.get("overall_verdict") == "FAIL":
        n_severe += 2  # functional failure is a major red flag

    if n_severe >= 3:
        overall = ("CONCERNING — Multiple red flags detected. The current "
                   "steering vectors are likely confounded with surface "
                   "features (especially text length / verbosity). "
                   "Results should be interpreted with extreme caution.")
    elif n_severe >= 1:
        overall = ("NEEDS ATTENTION — Some confounding signals detected. "
                   "Layer selection matters: prefer layers in the recommended "
                   "range. Consider improving contrastive pair quality.")
    else:
        overall = ("ACCEPTABLE — No major red flags in the available data. "
                   "Run --full mode for deeper validation.")

    results["warnings"] = warnings
    results["recommendations"] = recommendations
    results["overall_assessment"] = overall

    return results


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — full mode
# ═══════════════════════════════════════════════════════════════════════

def run_full_validation(args) -> Dict:
    """Load model, extract hidden states, run ALL validation checks."""
    _ensure_extract_vectors()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from extract_vectors import format_sample, extract_hidden_states, compute_mean_diff
    from tqdm import tqdm

    cfg = MODELS[args.model]
    token = HF_TOKEN if cfg.requires_token else None

    # ── load pairs ───────────────────────────────────────────────────
    with open(args.negotiation_pairs, encoding="utf-8") as f:
        neg_data = json.load(f)
    with open(args.control_pairs, encoding="utf-8") as f:
        con_data = json.load(f)

    all_neg_dims = neg_data["dimensions"]
    all_con_dims = con_data["dimensions"]
    target_neg = ([d for d in all_neg_dims if d["id"] in args.dimensions]
                  if args.dimensions else all_neg_dims)

    if not target_neg:
        log.error("No matching negotiation dimensions found.")
        return {}

    all_dims_combined = target_neg + all_con_dims

    results: Dict[str, Any] = {
        "model":     cfg.alias,
        "mode":      "full",
        "timestamp": datetime.now().isoformat(),
    }

    # ── Check 1: length confound (no model needed) ───────────────────
    log.info("Check 1: Length confound analysis")
    results["length_confound"] = analyze_length_confound(all_dims_combined)

    # ── Check 1b: vocabulary overlap ─────────────────────────────────
    log.info("Check 1b: Vocabulary overlap analysis")
    results["vocabulary_overlap"] = analyze_vocabulary_overlap(all_dims_combined)

    # ── Load model ───────────────────────────────────────────────────
    log.info("Loading model: %s", cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, token=token, torch_dtype=cfg.torch_dtype, device_map="auto"
    )
    model.eval()

    # ── Extract hidden states for all dimensions ─────────────────────
    log.info("Extracting hidden states for %d dimensions…", len(all_dims_combined))
    hidden_states = {}   # dim_id → {"pos_h": ..., "neg_h": ...}
    steering_vecs = {}   # dim_id → (n_layers, H)

    for dim in tqdm(all_dims_combined, desc="Extracting activations"):
        dim_id = dim["id"]
        pairs = dim["pairs"]
        pos_texts = [format_sample(p["context"], p["positive"], tokenizer, cfg) for p in pairs]
        neg_texts = [format_sample(p["context"], p["negative"], tokenizer, cfg) for p in pairs]

        pos_h = extract_hidden_states(model, tokenizer, pos_texts, args.batch_size)
        neg_h = extract_hidden_states(model, tokenizer, neg_texts, args.batch_size)

        hidden_states[dim_id] = {"pos_h": pos_h, "neg_h": neg_h}
        steering_vecs[dim_id] = compute_mean_diff(pos_h, neg_h)

    n_layers = next(iter(hidden_states.values()))["pos_h"].shape[1]
    log.info("Model has %d transformer layers.", n_layers)

    # ── Check 2: probe accuracy + permutation test ───────────────────
    log.info("Check 2: Probing + permutation test")
    results["probe_patterns"] = {}
    results["permutation_test"] = {}

    for dim in all_dims_combined:
        dim_id = dim["id"]
        hs = hidden_states[dim_id]
        pos_h, neg_h = hs["pos_h"], hs["neg_h"]

        # Probe accuracy per layer
        accs = []
        for l in range(n_layers):
            accs.append(_train_probe(
                np.concatenate([pos_h[:, l, :], neg_h[:, l, :]], axis=0),
                np.concatenate([np.ones(len(pos_h)), np.zeros(len(neg_h))]),
            ))
        accs_arr = np.array(accs)

        section = "control" if dim_id in ("verbosity", "formality") else "negotiation"
        min_acc = float(accs_arr.min())
        max_acc = float(accs_arr.max())
        mean_acc = float(accs_arr.mean())

        # Classify pattern
        if min_acc > 0.85 and (max_acc - min_acc) < 0.15:
            pattern, warning = "flat_high", "Uniformly high — likely surface features."
        elif max_acc < 0.60:
            pattern, warning = "flat_low", "Concept not encoded."
        else:
            pattern, warning = "other", None

        results["probe_patterns"][dim_id] = {
            "section":    section,
            "n_layers":   n_layers,
            "mean_acc":   round(mean_acc, 4),
            "min_acc":    round(min_acc, 4),
            "max_acc":    round(max_acc, 4),
            "best_layer": int(np.argmax(accs_arr)),
            "pattern":    pattern,
            "warning":    warning,
            "per_layer":  [round(float(a), 4) for a in accs],
        }

        # Permutation test (sample every 4th layer to keep runtime reasonable)
        test_layers = list(range(0, n_layers, max(1, n_layers // 9)))
        results["permutation_test"][dim_id] = permutation_test_per_layer(
            pos_h, neg_h,
            n_permutations=args.n_permutations,
            seed=args.seed,
            sample_layers=test_layers,
        )

    # ── Check 3: cosine similarity ───────────────────────────────────
    log.info("Check 3: Cosine similarity matrix")
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    results["cosine_similarity"] = compute_cosine_similarity_matrix(
        steering_vecs, layers=sample_layers
    )

    # ── Check 4: Cohen's d bias ──────────────────────────────────────
    log.info("Check 4: Normalised bias (Cohen's d)")
    neg_dim_ids = [d["id"] for d in target_neg]
    con_dim_ids = [d["id"] for d in all_con_dims]
    results["cohens_d_bias"] = {}

    for nd in neg_dim_ids:
        for cd in con_dim_ids:
            key = f"{nd}_vs_{cd}"
            results["cohens_d_bias"][key] = compute_cohens_d_bias(
                steering_vecs[nd],
                hidden_states[cd]["pos_h"],
                hidden_states[cd]["neg_h"],
            )

    # ── Check 5: per-pair consistency ────────────────────────────────
    log.info("Check 5: Per-pair consistency")
    results["pair_consistency"] = {}
    for dim in all_dims_combined:
        dim_id = dim["id"]
        hs = hidden_states[dim_id]
        results["pair_consistency"][dim_id] = compute_pair_alignment(
            hs["pos_h"], hs["neg_h"]
        )

    # ── Check 6: steering-direction probe ────────────────────────────
    log.info("Check 6: Steering-direction probe (1-D)")
    results["steering_direction_probe"] = {}
    for dim in all_dims_combined:
        dim_id = dim["id"]
        hs = hidden_states[dim_id]
        results["steering_direction_probe"][dim_id] = steering_direction_probe(
            hs["pos_h"], hs["neg_h"], steering_vecs[dim_id],
            seed=args.seed,
        )

    # ── Check 7: selectivity + layer recommendation ──────────────────
    log.info("Check 7: Selectivity & layer recommendations")
    results["selectivity"] = {}
    results["layer_recommendations"] = {}

    for nd in neg_dim_ids:
        probe_acc = np.array(results["probe_patterns"][nd]["per_layer"])
        bias_d = {}
        for cd in con_dim_ids:
            key = f"{nd}_vs_{cd}"
            bias_d[cd] = np.array(
                results["cohens_d_bias"][key]["cohens_d_per_layer"]
            )

        sel = compute_selectivity(probe_acc, bias_d)
        best_l = int(np.argmax(sel))
        results["selectivity"][nd] = {
            "per_layer":        [round(float(s), 4) for s in sel],
            "best_layer":       best_l,
            "best_selectivity": round(float(sel[best_l]), 4),
        }
        results["layer_recommendations"][nd] = recommend_layers(
            probe_acc, bias_d
        )

    # ── Compile warnings & recommendations ───────────────────────────
    warnings, recommendations = _compile_full_warnings(results, target_neg, all_con_dims)
    results["warnings"] = warnings
    results["recommendations"] = recommendations

    # Overall
    n_severe = (sum(1 for v in results["length_confound"].values()
                    if v["severity"] in ("SEVERE", "MODERATE"))
                + sum(1 for v in results["probe_patterns"].values()
                      if v["pattern"] in ("flat_high", "flat_perfect"))
                + sum(1 for v in results["cohens_d_bias"].values()
                      if v["n_severe"] > 0))
    if n_severe >= 3:
        results["overall_assessment"] = (
            "CONCERNING — Multiple red flags. Steering vectors are likely "
            "confounded with surface features. Use recommended layers only.")
    elif n_severe >= 1:
        results["overall_assessment"] = (
            "NEEDS ATTENTION — Some confounding. Layer selection is critical.")
    else:
        results["overall_assessment"] = "ACCEPTABLE — No major red flags."

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def _compile_full_warnings(
    results: Dict, neg_dims: List[Dict], con_dims: List[Dict]
) -> Tuple[List[str], List[str]]:
    """Derive warnings and recommendations from full-mode results."""
    warnings = []
    recommendations = []

    n_neg = len(neg_dims[0]["pairs"]) if neg_dims else 0
    n_con = len(con_dims[0]["pairs"]) if con_dims else 0
    warnings.append(
        f"Sample sizes: {n_neg} neg pairs ({2*n_neg} samples), "
        f"{n_con} control pairs ({2*n_con} samples)."
    )

    # Length confound
    severe_lc = [k for k, v in results["length_confound"].items()
                 if v["severity"] in ("SEVERE", "MODERATE")]
    if severe_lc:
        warnings.append(f"Length confound in: {', '.join(severe_lc)}")
        recommendations.append("Length-match contrastive pairs.")

    # Flat probes
    flat = [k for k, v in results["probe_patterns"].items()
            if v["pattern"] in ("flat_high", "flat_perfect")]
    if flat:
        warnings.append(f"Uniformly high probe accuracy: {', '.join(flat)}")

    # Permutation test failures
    for dim_id, layer_data in results.get("permutation_test", {}).items():
        n_insig = sum(1 for v in layer_data.values()
                      if isinstance(v, dict) and not v.get("significant", True))
        if n_insig > 0:
            warnings.append(
                f"{dim_id}: {n_insig} sampled layers NOT significant (p≥0.05)."
            )

    # Cosine similarity
    for layer_data in results.get("cosine_similarity", {}).values():
        if isinstance(layer_data, dict):
            for w in layer_data.get("warnings", []):
                warnings.append(
                    f"cos({w['dim_a']}, {w['dim_b']}) = {w['cosine']:.3f} "
                    "— vector overlap with control dimension."
                )

    # Cohen's d
    severe_bias = [k for k, v in results.get("cohens_d_bias", {}).items()
                   if v["n_severe"] > 0]
    if severe_bias:
        warnings.append(f"Severe bias (|d|>0.8) in: {', '.join(severe_bias)}")
        recommendations.append("Avoid layers listed as severe; prefer recommended layers.")

    # Pair consistency
    weak_pairs = [k for k, v in results.get("pair_consistency", {}).items()
                  if v["n_outliers"] > 0]
    if weak_pairs:
        warnings.append(f"Outlier pairs detected in: {', '.join(weak_pairs)}")
        recommendations.append("Rewrite or remove outlier contrastive pairs.")

    # Steering direction probe
    for dim_id, v in results.get("steering_direction_probe", {}).items():
        full_best = results["probe_patterns"].get(dim_id, {}).get("max_acc", 1.0)
        dir_best = v.get("best_acc", 0.0)
        if full_best - dir_best > 0.2:
            warnings.append(
                f"{dim_id}: full-dim probe={full_best:.2f} but "
                f"steering-direction probe={dir_best:.2f} — "
                "the steering vector may point in the wrong direction."
            )

    recommendations.append("Increase pairs to 30+ per dimension.")

    return warnings, recommendations


# ═══════════════════════════════════════════════════════════════════════
# Plotting (optional — only if matplotlib available)
# ═══════════════════════════════════════════════════════════════════════

def save_plots(results: Dict, output_dir: Path):
    """Generate and save validation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")
        return

    # 1. Selectivity per layer
    if "selectivity" in results:
        for dim_id, data in results["selectivity"].items():
            vals = data.get("per_layer")
            if vals:
                plt.figure(figsize=(10, 5))
                plt.plot(vals, marker=".", label=dim_id)
                plt.xlabel("Layer")
                plt.ylabel("Selectivity")
                plt.title(f"Selectivity — {dim_id}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f"selectivity_{dim_id}.png")
                plt.close()

    # 2. Probe patterns + permutation (if available)
    if "probe_patterns" in results:
        plt.figure(figsize=(12, 6))
        for dim_id, data in results["probe_patterns"].items():
            per_layer = data.get("per_layer")
            if per_layer:
                plt.plot(per_layer, label=dim_id)
        plt.xlabel("Layer")
        plt.ylabel("Probe Accuracy")
        plt.title("Probe Accuracy Per Layer")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.ylim(0.4, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "probe_accuracy.png")
        plt.close()

    # 3. Cohen's d heatmap
    if "cohens_d_bias" in results:
        keys = sorted(results["cohens_d_bias"].keys())
        if keys:
            fig, ax = plt.subplots(figsize=(12, max(3, len(keys))))
            data_matrix = []
            for k in keys:
                data_matrix.append(results["cohens_d_bias"][k]["cohens_d_per_layer"])
            data_matrix = np.array(data_matrix)
            im = ax.imshow(data_matrix, aspect="auto", cmap="RdBu_r",
                           vmin=-2, vmax=2)
            ax.set_yticks(range(len(keys)))
            ax.set_yticklabels(keys, fontsize=8)
            ax.set_xlabel("Layer")
            ax.set_title("Cohen's d Bias (red = positive separation)")
            fig.colorbar(im, ax=ax, label="Cohen's d")
            fig.tight_layout()
            fig.savefig(output_dir / "cohens_d_bias.png")
            plt.close(fig)

    # 4. Inter-layer consistency (consecutive cosine)
    if "layer_consistency" in results:
        lc = results["layer_consistency"]
        dims_with_data = [(d, v) for d, v in lc.items()
                          if isinstance(v, dict) and "consecutive_cosine" in v]
        if dims_with_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            # 4a: consecutive cosine
            for dim_id, data in dims_with_data:
                cos_vals = [np.nan if v is None else v for v in data["consecutive_cosine"]]
                axes[0].plot(cos_vals, label=dim_id, marker=".")
            axes[0].set_xlabel("Layer L → L+1")
            axes[0].set_ylabel("Cosine Similarity")
            axes[0].set_title("Inter-Layer Consistency")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(-0.1, 1.05)
            # 4b: drift from layer 0
            for dim_id, data in dims_with_data:
                drift_vals = [np.nan if v is None else v for v in data["drift_from_layer0"]]
                axes[1].plot(drift_vals, label=dim_id, marker=".")
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Cosine with Layer 0")
            axes[1].set_title("Drift from Layer 0")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(-0.5, 1.05)
            fig.tight_layout()
            fig.savefig(output_dir / "layer_consistency.png")
            plt.close(fig)

    # 5. Cross-dimension similarity heatmap across layers
    if "cross_dimension_similarity" in results:
        cds = results["cross_dimension_similarity"]
        if "pair_summaries" in cds:
            pair_names = sorted(cds["pair_summaries"].keys())
            if pair_names:
                n_layers = cds.get("n_layers", 36)
                # Replace None with np.nan so imshow gets a float array
                matrix = np.array(
                    [[np.nan if v is None else v
                      for v in cds["pair_summaries"][p]["per_layer"]]
                     for p in pair_names],
                    dtype=float,
                )
                fig, ax = plt.subplots(figsize=(14, max(3, len(pair_names))))
                im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                               vmin=-1, vmax=1)
                ax.set_yticks(range(len(pair_names)))
                ax.set_yticklabels([p.replace("_", " ") for p in pair_names],
                                   fontsize=8)
                ax.set_xlabel("Layer")
                ax.set_title("Cross-Dimension Cosine Similarity Across Layers")
                fig.colorbar(im, ax=ax, label="Cosine similarity")
                fig.tight_layout()
                fig.savefig(output_dir / "cross_dimension_similarity.png")
                plt.close(fig)

    # 6. Vector concentration (effective dimensionality across layers)
    if "vector_concentration" in results:
        vc = results["vector_concentration"]
        dims_with_data = [(d, v) for d, v in vc.items()
                          if isinstance(v, dict) and "eff_dim_per_layer" in v]
        if dims_with_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            random_eff = dims_with_data[0][1].get("random_expected_eff_dim", 750)
            for dim_id, data in dims_with_data:
                eff_vals = [np.nan if v is None else v for v in data["eff_dim_per_layer"]]
                axes[0].plot(eff_vals, label=dim_id, marker=".")
            axes[0].axhline(y=random_eff, color="gray", linestyle="--",
                            label=f"random baseline ({random_eff:.0f})")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Effective Dimensionality")
            axes[0].set_title("Vector Concentration")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            # Gini coefficient
            for dim_id, data in dims_with_data:
                gini_vals = [np.nan if v is None else v for v in data["gini_per_layer"]]
                axes[1].plot(gini_vals, label=dim_id, marker=".")
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Gini Coefficient")
            axes[1].set_title("Component Inequality (0=uniform, 1=single spike)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "vector_concentration.png")
            plt.close(fig)

    # 7. Shared subspace (PC1 explained variance across layers)
    if "shared_subspace" in results:
        ss = results["shared_subspace"]
        if "pc1_explained_per_layer" in ss:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            layers_range = range(ss.get("n_layers", len(ss["pc1_explained_per_layer"])))
            pc1_vals = [np.nan if v is None else v for v in ss["pc1_explained_per_layer"]]
            axes[0].plot(list(layers_range), pc1_vals,
                         color="red", marker=".", label="PC1")
            axes[0].axhline(y=1.0 / ss.get("n_dimensions", 3), color="gray",
                            linestyle="--", label="uniform (1/N)")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Variance Explained")
            axes[0].set_title("PC1 Explained Ratio (higher = more shared)")
            axes[0].set_ylim(0, 1)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            # Effective rank
            rank_vals = [np.nan if v is None else v for v in ss["effective_rank_per_layer"]]
            axes[1].plot(list(layers_range), rank_vals,
                         color="blue", marker=".")
            axes[1].axhline(y=ss.get("n_dimensions", 3), color="gray",
                            linestyle="--", label=f"max ({ss.get('n_dimensions', 3)})")
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Effective Rank")
            axes[1].set_title("Dimension Independence (higher = more independent)")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "shared_subspace.png")
            plt.close(fig)

    log.info("Plots saved to %s", output_dir)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive steering vector validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=list(_ALIAS_MAP.keys()), default="qwen2.5-3b")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true",
                      help="Full validation (requires GPU + model)")
    mode.add_argument("--analyze-only", action="store_true",
                      help="Analyse existing probe_results.json (no GPU)")

    p.add_argument("--dimensions", nargs="+", default=None,
                   help="Negotiation dimensions to validate (default: all probed)")
    p.add_argument("--negotiation_pairs", default="negotiation_steering_pairs.json")
    p.add_argument("--control_pairs", default="control_steering_pairs.json")
    p.add_argument("--probe_dir", default="probe_results",
                   help="Directory with existing probe results")
    p.add_argument("--vectors_dir", default="vectors_gpu",
                   help="Directory with extracted vectors (.npy files). "
                        "Also checked: vectors/ as fallback.")
    p.add_argument("--output_dir", default="validation_results",
                   help="Where to save validation output")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_permutations", type=int, default=100,
                   help="Number of permutations for significance test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    alias = _ALIAS_MAP[args.model]

    output_dir = Path(args.output_dir) / alias
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        _ensure_extract_vectors()
        log.info("Running FULL validation for %s", alias)
        results = run_full_validation(args)
    else:
        log.info("Running ANALYSE-ONLY validation for %s", alias)
        results = run_analyze_only(args)

    if not results:
        log.error("No results produced.")
        return

    # Save JSON
    json_path = output_dir / "validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", json_path)

    # Save report
    report = generate_report(results)
    report_path = output_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    # Print report (handle Windows encoding)
    import sys
    try:
        print(report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")

    # Plots
    save_plots(results, output_dir)


if __name__ == "__main__":
    main()
