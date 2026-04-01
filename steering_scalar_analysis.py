#!/usr/bin/env python3
"""
steering_scalar_analysis.py
============================

Three-part analysis of activation-steering scalars for Qwen 2.5-7B
using the ultimatum_10dim_20pairs_general_matched vectors.

PART 1 — Geometric Analysis
    Empirically measures what each alpha value actually *does* to the hidden
    states, using a calibration forward pass with the real model.
    Reports: norm expansion %, injection ratio %, cosine similarity, and
    effective rotation angle for every alpha tested.

PART 2 — Justification of {-5, 5, 15}
    Plots where the standard sweep falls in the geometric landscape, and
    checks whether any alpha crosses the "danger zone" (>50% norm expansion
    or cosine sim < 0.85).

PART 3 — Three Steering Methods
    Implements and compares:
        (A) Additive         h' = h + α·v              (current / baseline)
        (B) NormPreserving   h' = (h + α·v) / ‖h + α·v‖ · ‖h‖
        (C) Angular          pure rotation in the 2-D plane spanned by h and v̂,
                             preserves ‖h‖ exactly; α maps to angle θ via
                             θ = arctan(α · ‖v‖ / E[‖h‖])

PART 4 — Mini Game Comparison
    Runs N_GAMES proposer-role ultimatum games for three representative
    dimensions (firmness, empathy, flattery) at layer L10, using alphas
    {5, 15}, comparing all three steering methods side-by-side.
    Outputs a tidy results table and a comparison plot.

Run:
    python steering_scalar_analysis.py                   # all parts
    python steering_scalar_analysis.py --skip-games      # skip Part 4 (no games)
    python steering_scalar_analysis.py --n-games 5       # fewer games for quick test
"""

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_vectors import MODELS, HF_TOKEN
from ultimatum_game import (
    build_proposer_system,
    build_responder_system,
    parse_offer,
    parse_response,
    get_transformer_layers,
    POOL_SIZES,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

MODEL_KEY   = "qwen2.5-7b"
VARIANT     = "ultimatum_10dim_20pairs_general_matched"
METHOD      = "mean_diff"            # which extraction method's vectors to load
LAYERS_EVAL = [10, 12, 14]          # layers to analyse geometrically
ALPHAS_SWEEP = list(range(-50, 51, 1))  # full integer sweep for geo analysis
ALPHAS_STANDARD = [-5, 5, 15]       # the values we currently use
DIMS_GAME   = ["firmness", "empathy", "flattery"]  # dims to run games on
LAYER_GAME  = 10                    # layer for game comparison
ALPHAS_GAME = [5, 15]               # alphas to use in game comparison
DEFAULT_N_GAMES = 10                # games per config (override with --n-games)
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 200
CALIB_TEXT = (
    "You have $100 to split between yourself and another player. "
    "Propose a split. The other player will accept or reject it. "
    "If they reject, you both get $0. Make your offer."
)

ROOT = Path(__file__).resolve().parent
VECTORS_DIR = ROOT / "vectors" / VARIANT / "negotiation"

# colour codes for terminal output
GRN = "\033[92m"; YLW = "\033[93m"; RED = "\033[91m"; RST = "\033[0m"; BLD = "\033[1m"


# ══════════════════════════════════════════════════════════════════════════
# Vector loading
# ══════════════════════════════════════════════════════════════════════════

def load_vector(model_alias: str, dimension: str, layer: int) -> Optional[np.ndarray]:
    """Load a single mean-diff steering vector (already unit-normed)."""
    path = VECTORS_DIR / model_alias / METHOD / f"{dimension}_layer{layer:02d}.npy"
    if not path.exists():
        return None
    v = np.load(path).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm          # ensure unit-normed (should already be, but be safe)
    return v


def load_all_vectors(model_alias: str, dimension: str, layers: List[int]) -> Dict[int, np.ndarray]:
    """Load vectors for multiple layers."""
    out = {}
    for l in layers:
        v = load_vector(model_alias, dimension, l)
        if v is not None:
            out[l] = v
    return out


# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Geometric Calibration
# ══════════════════════════════════════════════════════════════════════════

class ActivationCollector:
    """Forward hook that stores the hidden state at a given layer."""

    def __init__(self):
        self.activations: List[torch.Tensor] = []
        self._handle = None

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self.activations.append(h.detach().cpu().float())

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()

    def mean_norm(self) -> float:
        if not self.activations:
            return float("nan")
        all_h = torch.cat(self.activations, dim=0)   # [B*S, d]
        return torch.norm(all_h, p=2, dim=-1).mean().item()

    def collect(self) -> torch.Tensor:
        """Return all collected activations as [N, d_model]."""
        if not self.activations:
            return torch.zeros(0)
        return torch.cat([a.reshape(-1, a.shape[-1]) for a in self.activations], dim=0)


def calibrate_activation_norms(
    model,
    tokenizer,
    layers_to_probe: List[int],
    texts: List[str],
    device: torch.device,
) -> Dict[int, Dict]:
    """
    Run a calibration forward pass, collect hidden-state norms at each layer.
    Returns per-layer stats: {layer: {mean_norm, std_norm, samples}}.
    """
    print(f"\n{BLD}[Calibration] Collecting hidden-state norms at layers {layers_to_probe}...{RST}")
    transformer_layers = get_transformer_layers(model)
    collectors = {}
    for l in layers_to_probe:
        c = ActivationCollector()
        c.register(transformer_layers[l])
        collectors[l] = c

    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").to(device)
            _ = model(**ids)

    for c in collectors.values():
        c.remove()

    stats = {}
    for l, c in collectors.items():
        acts = c.collect()  # [N_tokens, d_model]
        norms = torch.norm(acts, p=2, dim=-1)
        stats[l] = {
            "mean_norm": norms.mean().item(),
            "std_norm":  norms.std().item(),
            "n_tokens":  len(norms),
            "activations": acts,   # keep for later use
        }
        print(f"  Layer {l:2d}: mean_norm = {stats[l]['mean_norm']:.2f} ± {stats[l]['std_norm']:.2f}  (n={len(norms)} tokens)")

    return stats


def compute_geometric_metrics(
    h_batch: torch.Tensor,   # [N, d]
    v: np.ndarray,            # [d], unit-normed
    alpha: float,
) -> Dict:
    """
    Given a batch of hidden states and a unit-normed steering vector, compute
    geometric metrics for h' = h + alpha*v.
    """
    v_t = torch.tensor(v, dtype=torch.float32)

    # original norms
    orig_norms = torch.norm(h_batch, p=2, dim=-1)          # [N]
    mean_orig_norm = orig_norms.mean().item()

    # steered hidden states
    h_steered = h_batch + alpha * v_t.unsqueeze(0)

    # steered norms
    steered_norms = torch.norm(h_steered, p=2, dim=-1)
    mean_steered_norm = steered_norms.mean().item()

    # norm expansion %
    norm_expansion_pct = (mean_steered_norm - mean_orig_norm) / mean_orig_norm * 100

    # injection magnitude = |alpha| * ||v|| = |alpha| * 1 (unit-normed)
    injection_magnitude = abs(alpha) * 1.0
    injection_ratio_pct = injection_magnitude / mean_orig_norm * 100

    # cosine similarity (direction preservation)
    cos_sim = F.cosine_similarity(h_batch, h_steered, dim=-1).mean().item()

    # effective rotation angle (degrees)
    angle_rad = math.acos(max(-1.0, min(1.0, cos_sim)))
    angle_deg = math.degrees(angle_rad)

    # angular steering angle (what theta would give the same injection ratio)
    theta_rad = math.atan2(injection_magnitude, mean_orig_norm)
    theta_deg = math.degrees(theta_rad)

    return {
        "alpha":               alpha,
        "mean_orig_norm":      mean_orig_norm,
        "mean_steered_norm":   mean_steered_norm,
        "norm_expansion_pct":  norm_expansion_pct,
        "injection_ratio_pct": injection_ratio_pct,
        "injection_magnitude": injection_magnitude,
        "cos_sim":             cos_sim,
        "angle_deg":           angle_deg,
        "theta_equivalent_deg":theta_deg,
    }


def run_geometric_analysis(
    calib_stats: Dict[int, Dict],
    model_alias: str,
    dimensions: List[str],
    layers: List[int],
    alphas: List[float],
) -> Dict:
    """
    Full geometric sweep: for each (dimension, layer, alpha), compute metrics.
    Returns nested dict: results[dim][layer][alpha] = metric_dict
    """
    results = {}
    for dim in dimensions:
        results[dim] = {}
        for l in layers:
            if l not in calib_stats:
                continue
            v = load_vector(model_alias, dim, l)
            if v is None:
                print(f"  {YLW}WARNING: no vector for {dim} / L{l}{RST}")
                continue
            acts = calib_stats[l]["activations"]
            results[dim][l] = {}
            for alpha in alphas:
                m = compute_geometric_metrics(acts, v, alpha)
                results[dim][l][alpha] = m
    return results


# ══════════════════════════════════════════════════════════════════════════
# PART 2 — Print justification table
# ══════════════════════════════════════════════════════════════════════════

DANGER_NORM_EXPANSION = 50.0   # % norm expansion considered risky
DANGER_COS_SIM        = 0.85   # cosine sim below this is a significant direction hijack


def classify_alpha(m: Dict) -> str:
    if m["norm_expansion_pct"] > DANGER_NORM_EXPANSION or m["cos_sim"] < DANGER_COS_SIM:
        return f"{RED}RISKY{RST}"
    if m["norm_expansion_pct"] > 25.0 or m["cos_sim"] < 0.93:
        return f"{YLW}MODERATE{RST}"
    return f"{GRN}SAFE{RST}"


def print_scalar_justification(geo_results: Dict, dims: List[str], layers: List[int]):
    print(f"\n{'═'*90}")
    print(f"{BLD}PART 2 — Justification of Standard Scalars {{-5, 5, 15}}{RST}")
    print(f"{'═'*90}")
    print(f"Danger thresholds: norm expansion > {DANGER_NORM_EXPANSION}%  OR  cosine sim < {DANGER_COS_SIM}")
    print()

    h_prime_col = "‖h'‖"
    header = f"{'Dim':12s} {'Layer':>6} {'Alpha':>6} {'||h||':>8} {h_prime_col:>8} "
    header += f"{'Norm Exp%':>10} {'Inj Ratio%':>11} {'CosSim':>8} {'Angle deg':>9} {'Status':>15}"
    print(header)
    print("─" * 90)

    for dim in dims:
        for l in layers:
            if l not in geo_results.get(dim, {}):
                continue
            for alpha in ALPHAS_STANDARD:
                m = geo_results[dim][l].get(alpha)
                if m is None:
                    continue
                status = classify_alpha(m)
                print(
                    f"{dim:12s} {l:>6} {alpha:>+6.1f} "
                    f"{m['mean_orig_norm']:>8.2f} {m['mean_steered_norm']:>8.2f} "
                    f"{m['norm_expansion_pct']:>+10.2f} {m['injection_ratio_pct']:>11.2f} "
                    f"{m['cos_sim']:>8.4f} {m['angle_deg']:>8.2f} {status:>15s}"
                )
    print()


def print_full_sweep_table(geo_results: Dict, dim: str, layer: int, alphas: List[float]):
    """Print the full alpha sweep for one (dim, layer) to find safe boundary."""
    if layer not in geo_results.get(dim, {}):
        return
    print(f"\n{BLD}Full alpha sweep — {dim} / L{layer}{RST}")
    print(f"  {'Alpha':>6} {'Norm Exp%':>10} {'Inj Ratio%':>11} {'CosSim':>8} {'θ_equiv°':>9} Status")
    print(f"  {'─'*6} {'─'*10} {'─'*11} {'─'*8} {'─'*9} {'─'*10}")
    prev_risky = False
    for alpha in alphas:
        m = geo_results[dim][layer].get(alpha)
        if m is None:
            continue
        status = classify_alpha(m)
        risky = "RISKY" in status
        marker = "  ◄─ safe boundary" if (risky and not prev_risky) else ""
        prev_risky = risky
        print(
            f"  {alpha:>+6.0f} {m['norm_expansion_pct']:>+10.2f} "
            f"{m['injection_ratio_pct']:>11.2f} {m['cos_sim']:>8.4f} "
            f"{m['theta_equivalent_deg']:>9.2f} {status}{marker}"
        )


# ══════════════════════════════════════════════════════════════════════════
# PART 3 — Three Hook Implementations
# ══════════════════════════════════════════════════════════════════════════

class AdditiveHook:
    """
    Method A: Standard additive steering (current method).
        h' = h + α · v
    Norm is NOT preserved. Large α can push h out-of-distribution.
    """
    NAME = "additive"

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha
        self._handle = None

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(device=h.device, dtype=h.dtype)
        h_new = h + self.alpha * d
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class NormPreservingHook:
    """
    Method B: Post-addition re-normalisation.
        h' = (h + α · v) / ‖h + α · v‖ · ‖h‖
    Preserves each token's norm exactly, regardless of α.
    Pure direction shift; magnitude is locked.
    """
    NAME = "norm_preserving"

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha
        self._handle = None

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(device=h.device, dtype=h.dtype)
        orig_norms = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_shifted  = h + self.alpha * d
        new_norms  = torch.norm(h_shifted, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_new = h_shifted / new_norms * orig_norms
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


class AngularHook:
    """
    Method C: Angular (rotational) steering — Householder-style rotation.
    Rotates h towards v̂ by angle θ within the 2-D plane spanned by {h, v̂}.
    Norm is preserved exactly. α maps to θ via:

        θ = arctan(|α| · ‖v‖ / E[‖h‖]) · sign(α)

    where E[‖h‖] is the mean calibration norm (provided at construction).

    Decomposition (Gram-Schmidt):
        ĥ = h / ‖h‖
        w = (v̂ - (v̂·ĥ)ĥ) / ‖v̂ - (v̂·ĥ)ĥ‖   ← orthogonal component of v̂ in plane
        h' = ‖h‖ · (cos(θ) · ĥ + sin(θ) · w)

    If v̂ ‖ ĥ (degenerate case), fall back to additive.
    Reference: Turner et al. 2023 (rotation interpretation); Vu et al. 2025.
    """
    NAME = "angular"

    def __init__(self, direction: torch.Tensor, alpha: float, mean_h_norm: float):
        self.direction   = direction                  # unit-normed, [d]
        self.alpha       = alpha
        # convert alpha → rotation angle (radians)
        # injection_ratio = |alpha| * ||v|| / mean_h_norm = |alpha| / mean_h_norm
        self.theta = math.atan2(abs(alpha), mean_h_norm) * math.copysign(1.0, alpha)
        self._handle     = None

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output     # [B, S, d]
        v = self.direction.to(device=h.device, dtype=h.dtype)      # [d]
        theta = self.theta

        orig_norms = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)  # [B,S,1]
        h_unit = h / orig_norms                                     # ĥ

        # project v onto the orthogonal complement of ĥ
        v_parallel_coeff = (h_unit * v).sum(dim=-1, keepdim=True)  # (v̂ · ĥ) [B,S,1]
        v_perp = v - v_parallel_coeff * h_unit                      # [B,S,d]
        v_perp_norm = torch.norm(v_perp, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        w = v_perp / v_perp_norm                                     # unit orthogonal

        # handle degenerate case: v ∥ h (v_perp ≈ 0)
        degenerate = (v_perp_norm.squeeze(-1) < 1e-6)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        h_new = orig_norms * (cos_t * h_unit + sin_t * w)

        # where degenerate, use additive fallback
        if degenerate.any():
            h_fallback = h + self.alpha * v
            mask = degenerate.unsqueeze(-1).expand_as(h)
            h_new = torch.where(mask, h_fallback, h_new)

        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    def register(self, layer_module):
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


def _make_hook(method: str, direction: torch.Tensor, alpha: float, mean_h_norm: float):
    if method == "additive":
        return AdditiveHook(direction, alpha)
    elif method == "norm_preserving":
        return NormPreservingHook(direction, alpha)
    elif method == "angular":
        return AngularHook(direction, alpha, mean_h_norm)
    else:
        raise ValueError(f"Unknown method: {method}")


def measure_method_geometry(
    h_batch: torch.Tensor,   # [N, d]
    v: np.ndarray,            # [d] unit-normed
    alpha: float,
    mean_h_norm: float,
) -> Dict:
    """Compute geometric metrics for all three methods at a given alpha."""
    v_t = torch.tensor(v, dtype=torch.float32)
    results = {}

    # ── Method A: Additive ────────────────────────────────────────────────
    h_a = h_batch + alpha * v_t
    orig_norms    = torch.norm(h_batch, p=2, dim=-1)
    steered_norms_a = torch.norm(h_a, p=2, dim=-1)
    results["additive"] = {
        "mean_orig_norm":    orig_norms.mean().item(),
        "mean_steered_norm": steered_norms_a.mean().item(),
        "norm_expansion_pct": (steered_norms_a.mean() - orig_norms.mean()).item() / orig_norms.mean().item() * 100,
        "cos_sim": F.cosine_similarity(h_batch, h_a, dim=-1).mean().item(),
    }

    # ── Method B: Norm-Preserving ─────────────────────────────────────────
    orig_norms_k = orig_norms.unsqueeze(-1).clamp(min=1e-8)
    h_shifted    = h_batch + alpha * v_t
    new_norms_k  = torch.norm(h_shifted, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    h_b          = h_shifted / new_norms_k * orig_norms_k
    results["norm_preserving"] = {
        "mean_orig_norm":    orig_norms.mean().item(),
        "mean_steered_norm": torch.norm(h_b, p=2, dim=-1).mean().item(),
        "norm_expansion_pct": 0.0,   # by construction
        "cos_sim": F.cosine_similarity(h_batch, h_b, dim=-1).mean().item(),
    }

    # ── Method C: Angular ─────────────────────────────────────────────────
    theta = math.atan2(abs(alpha), mean_h_norm) * math.copysign(1.0, alpha)
    orig_norms_k2 = orig_norms.unsqueeze(-1).clamp(min=1e-8)
    h_unit = h_batch / orig_norms_k2
    v_par  = (h_unit * v_t).sum(dim=-1, keepdim=True) * h_unit
    v_perp = v_t - v_par
    v_perp_norm = torch.norm(v_perp, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    w      = v_perp / v_perp_norm
    h_c    = orig_norms_k2 * (math.cos(theta) * h_unit + math.sin(theta) * w)
    results["angular"] = {
        "mean_orig_norm":    orig_norms.mean().item(),
        "mean_steered_norm": torch.norm(h_c, p=2, dim=-1).mean().item(),
        "norm_expansion_pct": 0.0,   # by construction
        "cos_sim": F.cosine_similarity(h_batch, h_c, dim=-1).mean().item(),
        "theta_deg": math.degrees(theta),
    }

    return results


def print_method_comparison(
    calib_stats: Dict,
    model_alias: str,
    dims: List[str],
    layers: List[int],
    alphas: List[float],
):
    print(f"\n{'═'*100}")
    print(f"{BLD}PART 3 — Three Steering Methods: Geometric Comparison{RST}")
    print(f"{'═'*100}")
    print(f"  A = Additive (current)    h' = h + α·v")
    print(f"  B = NormPreserving        h' = (h + α·v) / ‖h+α·v‖ · ‖h‖")
    print(f"  C = Angular               h' = rotation of h by θ = arctan(|α|/‖h‖) towards v̂")
    print()

    header = (
        f"{'Dim':12s} {'L':>3} {'α':>5} │"
        f" {'‖h′‖_A':>8} {'ΔNorm_A%':>9} {'CosSim_A':>9} │"
        f" {'‖h′‖_B':>8} {'ΔNorm_B%':>9} {'CosSim_B':>9} │"
        f" {'‖h′‖_C':>8} {'ΔNorm_C%':>9} {'CosSim_C':>9} {'θ°':>6}"
    )
    print(header)
    print("─" * 100)

    for dim in dims:
        for l in layers:
            if l not in calib_stats:
                continue
            v = load_vector(model_alias, dim, l)
            if v is None:
                continue
            acts = calib_stats[l]["activations"]
            mean_norm = calib_stats[l]["mean_norm"]
            for alpha in alphas:
                m = measure_method_geometry(acts, v, alpha, mean_norm)
                a = m["additive"]
                b = m["norm_preserving"]
                c = m["angular"]
                theta_s = f"{c.get('theta_deg', 0):>6.2f}"
                print(
                    f"{dim:12s} {l:>3} {alpha:>+5.0f} │"
                    f" {a['mean_steered_norm']:>8.2f} {a['norm_expansion_pct']:>+9.2f} {a['cos_sim']:>9.4f} │"
                    f" {b['mean_steered_norm']:>8.2f} {b['norm_expansion_pct']:>+9.2f} {b['cos_sim']:>9.4f} │"
                    f" {c['mean_steered_norm']:>8.2f} {c['norm_expansion_pct']:>+9.2f} {c['cos_sim']:>9.4f} {theta_s}"
                )


# ══════════════════════════════════════════════════════════════════════════
# PART 4 — Mini Game Comparison
# ══════════════════════════════════════════════════════════════════════════

def _generate_with_hooks(
    model,
    tokenizer,
    messages: List[Dict],
    hooks_spec: List[Tuple],   # [(layer_idx, hook_obj)]
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    transformer_layers = get_transformer_layers(model)

    active_hooks = []
    for layer_idx, hook_obj in hooks_spec:
        if layer_idx < len(transformer_layers):
            hook_obj.register(transformer_layers[layer_idx])
            active_hooks.append(hook_obj)

    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in active_hooks:
            h.remove()

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_ug_game(
    model,
    tokenizer,
    pool: int,
    direction: Optional[np.ndarray],   # None = baseline
    alpha: float,
    layer: int,
    steering_method: str,
    mean_h_norm: float,
    device: torch.device,
) -> Dict:
    """
    Run one paired proposer game:
      1. Generate baseline offer (no hooks)
      2. Generate steered offer (with hooks per steering_method)
      3. Feed both offers to baseline responder
    Returns dict with paired game results.
    """
    # ── Proposer baseline ─────────────────────────────────────────────────
    sys_msg = build_proposer_system(pool, game="ultimatum")
    msgs    = [
        {"role": "system",  "content": sys_msg},
        {"role": "user",    "content": "Make your offer."},
    ]
    bl_text  = _generate_with_hooks(model, tokenizer, msgs, [], MAX_NEW_TOKENS, TEMPERATURE, device)
    bl_offer = parse_offer(bl_text, pool)

    # ── Proposer steered ──────────────────────────────────────────────────
    if direction is not None:
        v_t  = torch.tensor(direction, dtype=torch.float32)
        hook = _make_hook(steering_method, v_t, alpha, mean_h_norm)
        hooks_spec = [(layer, hook)]
    else:
        hooks_spec = []
    st_text  = _generate_with_hooks(model, tokenizer, msgs, hooks_spec, MAX_NEW_TOKENS, TEMPERATURE, device)
    st_offer = parse_offer(st_text, pool)

    def run_responder(offer):
        """Run baseline responder on a fixed offer."""
        if offer is None:
            return None, None, True  # parse error
        ps, rs = offer
        resp_sys = build_responder_system(ps, rs, pool, proposer_text=None)
        resp_msgs = [
            {"role": "system",  "content": resp_sys},
            {"role": "user",    "content": "What is your decision?"},
        ]
        resp_text = _generate_with_hooks(model, tokenizer, resp_msgs, [], MAX_NEW_TOKENS, TEMPERATURE, device)
        decision  = parse_response(resp_text)
        if decision is None:
            return resp_text, None, True
        agreed = (decision == "accept")
        return resp_text, agreed, False

    bl_resp_text, bl_agreed, bl_err = run_responder(bl_offer)
    st_resp_text, st_agreed, st_err = run_responder(st_offer)

    def payoff_pct(offer, agreed):
        if offer is None or agreed is None:
            return None
        ps, rs = offer
        if not agreed:
            return 0.0
        return ps / pool * 100   # proposer's payoff as % of pool

    return {
        "pool":              pool,
        # baseline
        "bl_parse_error":   bl_offer is None,
        "bl_offer_pct":     (bl_offer[0] / pool * 100) if bl_offer else None,
        "bl_agreed":        bl_agreed,
        "bl_payoff_pct":    payoff_pct(bl_offer, bl_agreed),
        "bl_text":          bl_text[:120],
        # steered
        "st_parse_error":   st_offer is None,
        "st_offer_pct":     (st_offer[0] / pool * 100) if st_offer else None,
        "st_agreed":        st_agreed,
        "st_payoff_pct":    payoff_pct(st_offer, st_agreed),
        "st_text":          st_text[:120],
    }


def run_game_comparison(
    model,
    tokenizer,
    device: torch.device,
    calib_stats: Dict,
    model_alias: str,
    dims: List[str],
    layer: int,
    alphas: List[float],
    methods: List[str],
    n_games: int,
    pool_list: List[int],
) -> List[Dict]:
    """
    Run mini-game comparison across (dim, alpha, method).
    Returns flat list of result dicts, each tagged with config info.
    """
    all_results = []
    mean_h_norm = calib_stats.get(layer, {}).get("mean_norm", 50.0)
    game_pools  = pool_list[:n_games]

    total_configs = len(dims) * len(alphas) * len(methods) + len(dims)  # +baseline per dim
    done = 0

    for dim in dims:
        v = load_vector(model_alias, dim, layer)

        # ── Baseline (no steering) ────────────────────────────────────────
        print(f"\n  {BLD}[{dim:12s} | baseline]{RST}")
        bl_games = []
        for pool in game_pools:
            g = run_single_ug_game(
                model, tokenizer, pool,
                direction=None, alpha=0.0, layer=layer,
                steering_method="additive",
                mean_h_norm=mean_h_norm, device=device,
            )
            g.update({"dimension": dim, "alpha": 0.0, "method": "baseline", "layer": layer})
            bl_games.append(g)
            all_results.append(g)
        done += 1
        _print_game_progress(bl_games, "baseline", dim, 0.0, layer)

        # ── Steered configs ───────────────────────────────────────────────
        for alpha in alphas:
            for method in methods:
                print(f"\n  {BLD}[{dim:12s} | α={alpha:+.0f} | {method}]{RST}")
                config_games = []
                for pool in game_pools:
                    g = run_single_ug_game(
                        model, tokenizer, pool,
                        direction=v, alpha=alpha, layer=layer,
                        steering_method=method,
                        mean_h_norm=mean_h_norm, device=device,
                    )
                    g.update({"dimension": dim, "alpha": alpha, "method": method, "layer": layer})
                    config_games.append(g)
                    all_results.append(g)
                done += 1
                _print_game_progress(config_games, method, dim, alpha, layer)

    return all_results


def _print_game_progress(games: List[Dict], method: str, dim: str, alpha: float, layer: int):
    valid_bl = [g for g in games if not g["bl_parse_error"]]
    valid_st = [g for g in games if not g["st_parse_error"]]
    n = len(games)

    def safe_mean(vals):
        v = [x for x in vals if x is not None]
        return sum(v) / len(v) if v else float("nan")

    bl_offer   = safe_mean([g["bl_offer_pct"]  for g in valid_bl])
    st_offer   = safe_mean([g["st_offer_pct"]  for g in valid_st])
    bl_acc     = safe_mean([g["bl_agreed"]      for g in valid_bl if g["bl_agreed"] is not None])
    st_acc     = safe_mean([g["st_agreed"]      for g in valid_st if g["st_agreed"] is not None])
    bl_payoff  = safe_mean([g["bl_payoff_pct"] for g in valid_bl if g["bl_payoff_pct"] is not None])
    st_payoff  = safe_mean([g["st_payoff_pct"] for g in valid_st if g["st_payoff_pct"] is not None])

    delta_offer  = st_offer  - bl_offer  if not math.isnan(st_offer)  else float("nan")
    delta_acc    = st_acc    - bl_acc    if not math.isnan(st_acc)    else float("nan")
    delta_payoff = st_payoff - bl_payoff if not math.isnan(st_payoff) else float("nan")

    print(
        f"    {dim:12s} L{layer} α={alpha:+.0f} [{method:15s}]  "
        f"n={n}  "
        f"offer: {bl_offer:5.1f}→{st_offer:5.1f} (Δ{delta_offer:+.1f}pp)  "
        f"acc: {bl_acc:.2f}→{st_acc:.2f} (Δ{delta_acc:+.2f})  "
        f"cpd: {bl_payoff:5.1f}→{st_payoff:5.1f} (Δ{delta_payoff:+.1f}pp)"
    )


def print_game_summary(results: List[Dict]):
    """Print final cross-method comparison table for game results."""
    print(f"\n{'═'*110}")
    print(f"{BLD}PART 4 — Mini Game Results: Cross-Method Comparison{RST}")
    print(f"{'═'*110}")

    # Group by (dimension, alpha, method)
    from collections import defaultdict
    groups = defaultdict(list)
    for g in results:
        key = (g["dimension"], g["alpha"], g["method"])
        groups[key].append(g)

    def safe_mean(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(v) / len(v) if v else float("nan")

    def fmt(v, digits=1):
        return f"{v:+{digits+5}.{digits}f}" if not math.isnan(v) else "    —  "

    header = (
        f"{'Dimension':12s} {'α':>5} {'Method':>17} │ "
        f"{'N':>3} {'BL Offer':>9} {'ST Offer':>9} {'Δ Offer':>8} │ "
        f"{'BL Acc':>7} {'ST Acc':>7} {'Δ Acc':>7} │ "
        f"{'BL Cpd':>7} {'ST Cpd':>7} {'Δ Cpd':>8}"
    )
    print(header)
    print("─" * 110)

    dims_sorted  = sorted(set(g["dimension"] for g in results))
    alphas_sorted = sorted(set(g["alpha"] for g in results))
    methods_order = ["baseline", "additive", "norm_preserving", "angular"]

    for dim in dims_sorted:
        for alpha in alphas_sorted:
            methods_here = methods_order if alpha != 0 else ["baseline"]
            for method in methods_here:
                key = (dim, alpha, method)
                if key not in groups:
                    continue
                gs = groups[key]
                valid_bl = [g for g in gs if not g["bl_parse_error"]]
                valid_st = [g for g in gs if not g["st_parse_error"]]

                bl_offer  = safe_mean([g["bl_offer_pct"]  for g in valid_bl])
                st_offer  = safe_mean([g["st_offer_pct"]  for g in valid_st]) if method != "baseline" else bl_offer
                bl_acc    = safe_mean([g["bl_agreed"]      for g in valid_bl if g["bl_agreed"] is not None])
                st_acc    = safe_mean([g["st_agreed"]      for g in valid_st if g["st_agreed"] is not None]) if method != "baseline" else bl_acc
                bl_cpd    = safe_mean([g["bl_payoff_pct"] for g in valid_bl if g["bl_payoff_pct"] is not None])
                st_cpd    = safe_mean([g["st_payoff_pct"] for g in valid_st if g["st_payoff_pct"] is not None]) if method != "baseline" else bl_cpd

                d_offer   = st_offer  - bl_offer  if not math.isnan(st_offer)  else float("nan")
                d_acc     = st_acc    - bl_acc    if not math.isnan(st_acc)    else float("nan")
                d_cpd     = st_cpd    - bl_cpd    if not math.isnan(st_cpd)    else float("nan")

                n_err_bl  = sum(g["bl_parse_error"] for g in gs)
                n_err_st  = sum(g["st_parse_error"] for g in gs)
                n_str = f"{len(gs)}" + (f"(err bl={n_err_bl},st={n_err_st})" if n_err_bl+n_err_st else "")

                print(
                    f"{dim:12s} {alpha:>+5.0f} {method:>17s} │ "
                    f"{n_str:>3} {bl_offer:>9.2f} {st_offer:>9.2f} {fmt(d_offer):>8} │ "
                    f"{bl_acc:>7.3f} {st_acc:>7.3f} {fmt(d_acc, 3):>7} │ "
                    f"{bl_cpd:>7.2f} {st_cpd:>7.2f} {fmt(d_cpd):>8}"
                )
        print("─" * 110)

    print()
    print("  Columns: BL = baseline (no steering), ST = steered, Cpd = compound payoff (acc × offer %)")
    layer_used = results[0].get("layer", "?") if results else "?"
    print(f"  Dimensions: {dims_sorted}   Layer: {layer_used}")
    print(f"  Methods: A=additive (current), B=norm_preserving, C=angular")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=MODEL_KEY,
                        help=f"Model key from MODELS registry (default: {MODEL_KEY})")
    parser.add_argument("--eval-layers", nargs="+", type=int, default=None,
                        help="Layers to analyse geometrically (default: [10,12,14] for 7B, auto-scaled for others)")
    parser.add_argument("--alpha-max", type=int, default=50,
                        help="Absolute max for alpha sweep (default: 50; use e.g. 1500 for 32B)")
    parser.add_argument("--alpha-step", type=int, default=1,
                        help="Step size for alpha sweep (default: 1; use e.g. 10 for wide sweeps)")
    parser.add_argument("--skip-games", action="store_true",
                        help="Skip Part 4 (no UG games; only geometric analysis)")
    parser.add_argument("--n-games", type=int, default=DEFAULT_N_GAMES,
                        help=f"Number of games per config in Part 4 (default: {DEFAULT_N_GAMES})")
    parser.add_argument("--dims", nargs="+", default=DIMS_GAME,
                        help="Dimensions to run games on")
    parser.add_argument("--alphas-game", nargs="+", type=float, default=ALPHAS_GAME,
                        help="Alphas to use in Part 4 games")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for Part 4 game experiments (default: LAYER_GAME, auto-scaled for non-7B models)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save Part 4 game results to JSON file")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Load model in 4-bit (default: True)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    args = parser.parse_args()

    t0 = time.time()

    alphas_sweep = list(range(-args.alpha_max, args.alpha_max + 1, args.alpha_step))

    # ── Model alias ────────────────────────────────────────────────────────
    model_info  = MODELS[args.model]
    model_alias = model_info.alias   # e.g. "qwen2.5-7b"

    print(f"\n{'═'*70}")
    print(f"{BLD}  Activation Steering Scalar Analysis{RST}")
    print(f"  Model  : {args.model}  ({model_alias})")
    print(f"  Variant: {VARIANT}")
    print(f"  Method : {METHOD}")
    print(f"{'═'*70}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n{BLD}Loading model...{RST}")
    hf_id   = model_info.hf_id
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_kwargs = dict(trust_remote_code=True)
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)

    model_kwargs = dict(trust_remote_code=True, device_map="auto")
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN
    if args.model.endswith("-gptq"):
        # GPTQ models are already quantized; use auto-gptq loader to avoid
        # optimum version bugs with BitsAndBytesConfig stacking.
        from auto_gptq import AutoGPTQForCausalLM
        _gptq_wrapper = AutoGPTQForCausalLM.from_quantized(
            hf_id, use_safetensors=True, device="cuda:0",
            trust_remote_code=True,
            **({"token": HF_TOKEN} if HF_TOKEN else {}),
        )
        # auto-gptq wraps the model; unwrap to standard HF CausalLM so that
        # get_transformer_layers() and hook registration work identically.
        model = _gptq_wrapper.model
    else:
        if args.quantize and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.eval()
    n_layers = len(get_transformer_layers(model))
    print(f"  Model loaded on {device}. n_layers = {n_layers}")

    # ── Resolve layer lists (auto-scale relative to 7B baseline of 28 layers) ─
    if args.eval_layers is not None:
        layers_eval = args.eval_layers
    else:
        # Scale 7B reference layers [10, 12, 14] (out of 28) proportionally
        ref_layers   = [10, 12, 14]
        ref_n_layers = 28
        layers_eval  = [round(l / ref_n_layers * n_layers) for l in ref_layers]
        if n_layers != ref_n_layers:
            print(f"  Auto-scaled eval layers: {ref_layers} (7B/28) → {layers_eval} ({model_alias}/{n_layers})")

    layer_game = args.layer if args.layer is not None else round(LAYER_GAME / 28 * n_layers) if n_layers != 28 else LAYER_GAME
    if n_layers != 28 and args.layer is None:
        print(f"  Auto-scaled game layer : {LAYER_GAME} (7B/28) → {layer_game} ({model_alias}/{n_layers})")

    # ── Calibration texts ──────────────────────────────────────────────────
    calib_texts = [
        CALIB_TEXT,
        "You are Player A in an Ultimatum Game. You have $100 to split. Make your offer.",
        "Player A proposes: they get $70, you get $30. Do you accept or reject?",
        "Negotiate fairly. Consider both parties' outcomes.",
        "In this game, every dollar counts. Maximize your own payoff.",
    ]

    # ── Part 1: Calibration ────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"{BLD}PART 1 — Geometric Analysis of Activation Norms{RST}")
    print(f"{'═'*70}")
    calib_stats = calibrate_activation_norms(
        model, tokenizer, layers_eval, calib_texts, device
    )

    # Run full geo analysis for a representative subset of dimensions
    DIMS_GEO = ["firmness", "empathy", "flattery"]
    geo_results = run_geometric_analysis(
        calib_stats, model_alias, DIMS_GEO, layers_eval, alphas_sweep
    )

    # Print concise table at ALPHAS_STANDARD
    print(f"\n{BLD}Activation norm statistics per layer:{RST}")
    for l in layers_eval:
        s = calib_stats.get(l, {})
        print(f"  Layer {l:2d}: E[‖h‖] = {s.get('mean_norm',0):.2f}  ±  {s.get('std_norm',0):.2f}")

    # ── Save geo results to JSON for the plotter ──────────────────────────
    geo_export_path = ROOT / "results" / f"scalar_analysis_geo_{model_alias}_a{args.alpha_max}.json"
    geo_export_path.parent.mkdir(parents=True, exist_ok=True)
    geo_export = {}
    for dim, layer_data in geo_results.items():
        geo_export[dim] = {}
        for l, alpha_data in layer_data.items():
            geo_export[dim][str(l)] = {
                str(a): {k: float(v) for k, v in m.items()}
                for a, m in alpha_data.items()
            }
    calib_export = {
        str(l): {"mean_norm": s["mean_norm"], "std_norm": s["std_norm"]}
        for l, s in calib_stats.items()
        if isinstance(s, dict) and "mean_norm" in s
    }
    with open(geo_export_path, "w") as f:
        json.dump({"geo": geo_export, "calib": calib_export}, f, indent=2)
    print(f"\n  [Saved geo results to {geo_export_path}]")

    # ── Part 2: Scalar justification ───────────────────────────────────────
    print_scalar_justification(geo_results, DIMS_GEO, layers_eval)

    # Full sweep for firmness/layer_game to show safe boundary
    print_full_sweep_table(geo_results, "firmness", layer_game, alphas_sweep)
    print_full_sweep_table(geo_results, "empathy",  layer_game, alphas_sweep)

    # ── Part 3: Three methods geometry ─────────────────────────────────────
    print_method_comparison(
        calib_stats, model_alias,
        dims=DIMS_GEO,
        layers=layers_eval,
        alphas=ALPHAS_STANDARD,
    )

    print(f"\n{BLD}Method Interpretation:{RST}")
    print(f"  A (Additive)       : Norm grows with α. At large α, activation drifts off the")
    print(f"                       RMSNorm shell. Risk: gibberish if norm expansion > ~50%.")
    print(f"  B (NormPreserving) : Norm locked; pure direction shift. Equivalent to A at")
    print(f"                       small α (small-angle approx: ‖v‖·α/‖h‖ << 1).")
    print(f"                       At large α, B is strictly weaker (same direction change,")
    print(f"                       no magnitude boost), so needs larger α to match A's effect.")
    print(f"  C (Angular)        : Identical to B in effect but parameterised by rotation angle.")
    print(f"                       θ saturates at 90° — impossible to 'flip' h past v̂.")
    print(f"                       Most conservative; most norm-stable by design.")

    # ── Part 4: Game comparison ────────────────────────────────────────────
    if not args.skip_games:
        print(f"\n{'═'*70}")
        print(f"{BLD}PART 4 — Mini Ultimatum Game Comparison{RST}")
        print(f"  Dimensions: {args.dims}")
        print(f"  Layer     : {layer_game}")
        print(f"  Alphas    : {args.alphas_game}")
        print(f"  Methods   : additive, norm_preserving, angular")
        print(f"  N games   : {args.n_games} per config")
        print(f"{'═'*70}")

        game_results = run_game_comparison(
            model, tokenizer, device,
            calib_stats=calib_stats,
            model_alias=model_alias,
            dims=args.dims,
            layer=layer_game,
            alphas=args.alphas_game,
            methods=["additive", "norm_preserving", "angular"],
            n_games=args.n_games,
            pool_list=POOL_SIZES,
        )

        print_game_summary(game_results)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(game_results, f, indent=2, default=str)
            print(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n{'═'*70}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
