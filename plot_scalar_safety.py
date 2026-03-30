#!/usr/bin/env python3
"""
plot_scalar_safety.py
=====================
Standalone plotting script for steering scalar safety analysis.
Reads results/scalar_analysis_geo.json produced by steering_scalar_analysis.py
(run with --skip-games first to generate the JSON quickly).

Produces 5 publication-quality figures saved to results/scalar_safety_plots/:

  Fig 1 — Safety landscape overview (3-panel: norm expansion, cosine sim,
           injection ratio across α ∈ [-50, +50] for all dims × layers)

  Fig 2 — "Safety envelope" bell-curve style: Gaussian kernel density over
           the cosine-similarity distribution, with coloured safe/moderate/risky
           bands and your standard alphas marked as vertical lines

  Fig 3 — Per-dimension heatmap: cosine similarity at (layer × alpha) for each
           dimension — shows where steering is strong vs degenerate

  Fig 4 — Norm expansion contour: 2-D filled contour over (alpha × layer),
           averaged across dimensions — clean single-page summary

  Fig 5 — Composite "report card" for your standard sweep {-5, 5, 15}:
           radar chart showing all 4 safety metrics per alpha

Usage:
    python plot_scalar_safety.py                          # reads default JSON path
    python plot_scalar_safety.py --json path/to/file.json
    python plot_scalar_safety.py --no-show               # save only, don't display
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        130,
    "savefig.dpi":       220,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# Zone thresholds (matching steering_scalar_analysis.py)
SAFE_NORM_EXP   = 50.0   # % norm expansion below which = SAFE
SAFE_COS_SIM    = 0.85   # cosine sim above which = SAFE
WARN_NORM_EXP   = 25.0
WARN_COS_SIM    = 0.93

# Colour palette
C_SAFE     = "#2ecc71"
C_WARN     = "#f39c12"
C_RISKY    = "#e74c3c"
C_STANDARD = ["#3498db", "#e67e22", "#9b59b6"]   # -5, +5, +15
DIM_COLORS = {
    "firmness":  "#e74c3c",
    "empathy":   "#3498db",
    "flattery":  "#2ecc71",
    "anchoring": "#9b59b6",
    "greed":     "#f39c12",
}

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "scalar_safety_plots"


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_geo(path: Path) -> tuple[dict, dict]:
    with open(path) as f:
        raw = json.load(f)
    geo   = raw["geo"]    # geo[dim][layer_str][alpha_str] = metric_dict
    calib = raw["calib"]  # calib[layer_str] = {mean_norm, std_norm}
    return geo, calib


def flatten_geo(geo: dict) -> list[dict]:
    """Return a flat list of records from the nested geo dict."""
    records = []
    for dim, layer_data in geo.items():
        for layer_str, alpha_data in layer_data.items():
            for alpha_str, m in alpha_data.items():
                records.append({
                    "dim":               dim,
                    "layer":             int(layer_str),
                    "alpha":             float(alpha_str),
                    "norm_expansion_pct": m["norm_expansion_pct"],
                    "injection_ratio_pct":m["injection_ratio_pct"],
                    "cos_sim":           m["cos_sim"],
                    "angle_deg":         m["angle_deg"],
                })
    return records


def classify(norm_exp: float, cos_sim: float) -> str:
    if abs(norm_exp) > SAFE_NORM_EXP or cos_sim < SAFE_COS_SIM:
        return "risky"
    if abs(norm_exp) > WARN_NORM_EXP or cos_sim < WARN_COS_SIM:
        return "moderate"
    return "safe"


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Safety Landscape Overview
# ══════════════════════════════════════════════════════════════════════════

def fig1_landscape(records: list[dict], calib: dict, out_dir: Path, show: bool):
    dims   = sorted(set(r["dim"]   for r in records))
    layers = sorted(set(r["layer"] for r in records))

    fig, axes = plt.subplots(3, len(layers), figsize=(5 * len(layers), 11),
                             sharey="row", sharex="col")
    if len(layers) == 1:
        axes = axes.reshape(-1, 1)

    metrics = ["norm_expansion_pct", "cos_sim", "injection_ratio_pct"]
    ylabels = ["Norm expansion (%)", "Cosine similarity", "Injection ratio (%)"]
    danger_y = [SAFE_NORM_EXP, SAFE_COS_SIM, None]
    warn_y   = [WARN_NORM_EXP, WARN_COS_SIM, None]

    for col, layer in enumerate(layers):
        for row, (met, yl, dy, wy) in enumerate(zip(metrics, ylabels, danger_y, warn_y)):
            ax = axes[row, col]
            sub = [r for r in records if r["layer"] == layer]
            alphas_all = sorted(set(r["alpha"] for r in sub))

            for dim in dims:
                d_sub = [r for r in sub if r["dim"] == dim]
                d_sub.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in d_sub]
                ys = [r[met]     for r in d_sub]
                # smooth slightly for visual clarity
                ys_s = gaussian_filter1d(ys, sigma=0.8)
                col_d = DIM_COLORS.get(dim, "#555")
                ax.plot(xs, ys_s, lw=1.8, color=col_d, label=dim, alpha=0.85)

            # danger / warn bands
            xlim = (min(alphas_all), max(alphas_all))
            if dy is not None:
                ax.axhline(dy, color=C_RISKY, lw=1.2, ls="--", alpha=0.7,
                           label=f"Risky threshold ({dy})")
            if wy is not None:
                ax.axhline(wy, color=C_WARN, lw=1.0, ls=":", alpha=0.6,
                           label=f"Moderate threshold ({wy})")

            # shade zones for cosine sim panel
            if met == "cos_sim":
                ax.axhspan(SAFE_COS_SIM, WARN_COS_SIM, alpha=0.07,
                           color=C_WARN, label="Moderate zone")
                ax.axhspan(0.0, SAFE_COS_SIM, alpha=0.07,
                           color=C_RISKY, label="Risky zone")
                ax.set_ylim(0.84, 1.001)

            # mark standard alphas
            for std_a, sc in zip([-5, 5, 15], C_STANDARD):
                ax.axvline(std_a, color=sc, lw=1.2, ls="-.", alpha=0.6)

            ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.4)
            ax.set_xlim(*xlim)
            ax.set_ylabel(yl if col == 0 else "")
            if row == 0:
                ax.set_title(f"Layer {layer}")
            if row == 2:
                ax.set_xlabel("Steering scalar α")
            if row == 0 and col == len(layers) - 1:
                ax.legend(loc="upper right", fontsize=7.5)

    # legend for standard alphas
    std_patches = [
        mpatches.Patch(color=C_STANDARD[0], label="α = −5  (standard)"),
        mpatches.Patch(color=C_STANDARD[1], label="α = +5  (standard)"),
        mpatches.Patch(color=C_STANDARD[2], label="α = +15 (standard)"),
    ]
    fig.legend(handles=std_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Steering Scalar Safety Landscape  —  Qwen 2.5-7B  |  α ∈ [−50, +50]",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = out_dir / "fig1_safety_landscape.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — "Bell-curve" Safety Envelope
# ══════════════════════════════════════════════════════════════════════════

def fig2_bell_curve(records: list[dict], calib: dict, out_dir: Path, show: bool):
    """
    For each layer, show a smooth curve of cos_sim vs alpha.
    The area under the curve is filled green/amber/red by zone.
    The shape resembles a bell (highest cos_sim at alpha=0, declining symmetrically).
    """
    layers = sorted(set(r["layer"] for r in records))
    fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 5.5),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        sub = [r for r in records if r["layer"] == layer]
        sub.sort(key=lambda r: r["alpha"])

        # mean cos_sim across dims at each alpha
        alpha_vals = sorted(set(r["alpha"] for r in sub))
        means, stds = [], []
        for a in alpha_vals:
            vals = [r["cos_sim"] for r in sub if r["alpha"] == a]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        xs    = np.array(alpha_vals)
        means = np.array(means)
        stds  = np.array(stds)

        # smooth for presentation
        means_s = gaussian_filter1d(means, sigma=1.2)
        std_lo   = gaussian_filter1d(means - stds, sigma=1.2)
        std_hi   = gaussian_filter1d(means + stds, sigma=1.2)

        # zone fills (horizontal bands mapped onto the curve)
        ax.axhspan(WARN_COS_SIM, 1.01, alpha=0.10, color=C_SAFE,    zorder=0)
        ax.axhspan(SAFE_COS_SIM, WARN_COS_SIM, alpha=0.12, color=C_WARN, zorder=0)
        ax.axhspan(0.80, SAFE_COS_SIM, alpha=0.12, color=C_RISKY,   zorder=0)

        # zone threshold lines
        ax.axhline(WARN_COS_SIM,  color=C_WARN,  lw=1.2, ls="--",
                   label=f"Moderate ({WARN_COS_SIM})", zorder=2)
        ax.axhline(SAFE_COS_SIM,  color=C_RISKY, lw=1.2, ls="--",
                   label=f"Risky boundary ({SAFE_COS_SIM})", zorder=2)

        # fill under curve by zone colour
        ax.fill_between(xs, std_lo, std_hi, alpha=0.20, color="#2c3e50", zorder=1,
                        label="±1 std across dims")
        ax.plot(xs, means_s, lw=2.5, color="#2c3e50", zorder=3,
                label="Mean cos sim")

        # gradient fill under mean curve, coloured by zone
        # safe zone (cos_sim > WARN threshold)
        ax.fill_between(xs, 0.80, means_s,
                        where=means_s >= WARN_COS_SIM,
                        alpha=0.35, color=C_SAFE, zorder=1, interpolate=True)
        # moderate zone
        ax.fill_between(xs, 0.80, means_s,
                        where=(means_s >= SAFE_COS_SIM) & (means_s < WARN_COS_SIM),
                        alpha=0.35, color=C_WARN, zorder=1, interpolate=True)
        # risky zone
        ax.fill_between(xs, 0.80, means_s,
                        where=means_s < SAFE_COS_SIM,
                        alpha=0.35, color=C_RISKY, zorder=1, interpolate=True)

        # mark standard alphas with vertical lines + annotations
        for std_a, sc, lbl in zip([-5, 5, 15], C_STANDARD, ["−5", "+5", "+15"]):
            ax.axvline(std_a, color=sc, lw=1.8, ls="-.", zorder=4)
            cos_at = float(np.interp(std_a, xs, means_s))
            ax.annotate(
                f"α={lbl}\n{cos_at:.4f}",
                xy=(std_a, cos_at), xytext=(std_a + 1.5, cos_at - 0.012),
                fontsize=7.5, color=sc,
                arrowprops=dict(arrowstyle="->", color=sc, lw=0.8),
                zorder=5,
            )

        # zone labels on right axis
        ax2 = ax.twinx()
        ax2.set_ylim(0.80, 1.001)
        ax2.set_yticks([0.91, 0.89, 0.83])
        ax2.set_yticklabels(["SAFE", "MODERATE", "RISKY"],
                            fontsize=8, fontweight="bold")
        ax2.tick_params(right=False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # find empirical safe boundary (where mean crosses SAFE threshold)
        crossing_alphas = []
        for i in range(len(xs) - 1):
            if (means_s[i] >= SAFE_COS_SIM) != (means_s[i+1] >= SAFE_COS_SIM):
                crossing_alphas.append(xs[i])
        for cx in crossing_alphas:
            ax.axvline(cx, color=C_RISKY, lw=0.8, ls=":", alpha=0.5)
            ax.text(cx, 0.802, f" α≈{int(cx)}", fontsize=7, color=C_RISKY,
                    va="bottom", ha="left")

        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(0.80, 1.001)
        ax.set_xlabel("Steering scalar α")
        ax.set_ylabel("Cosine similarity  (h, h′)" if layer == layers[0] else "")
        ax.set_title(f"Layer {layer}")
        ax.legend(loc="lower center", fontsize=7.5, ncol=2)

        # zone band legend patches
    zone_patches = [
        mpatches.Patch(color=C_SAFE,  alpha=0.5, label="Safe zone  (cos_sim ≥ 0.93)"),
        mpatches.Patch(color=C_WARN,  alpha=0.5, label="Moderate   (0.85 – 0.93)"),
        mpatches.Patch(color=C_RISKY, alpha=0.5, label="Risky      (cos_sim < 0.85)"),
    ]
    fig.legend(handles=zone_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    calib_mean = np.mean([v["mean_norm"] for v in calib.values()])
    fig.suptitle(
        f"Steering Safety Envelope  —  Cosine Similarity vs α\n"
        f"Qwen 2.5-7B  |  E[‖h‖] ≈ {calib_mean:.0f}  |  Mean ± std across dimensions",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = out_dir / "fig2_safety_envelope.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-dimension heatmap: cos_sim at (alpha × layer)
# ══════════════════════════════════════════════════════════════════════════

def fig3_dim_heatmaps(records: list[dict], out_dir: Path, show: bool):
    dims   = sorted(set(r["dim"]   for r in records))
    layers = sorted(set(r["layer"] for r in records))
    alphas = sorted(set(r["alpha"] for r in records))

    # custom diverging colormap: green (safe) → amber → red (risky)
    cmap = LinearSegmentedColormap.from_list(
        "safety", [C_RISKY, C_WARN, C_SAFE], N=256
    )

    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.5 * n_dims, 4.5), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        # build matrix: rows = layers, cols = alphas
        mat = np.zeros((len(layers), len(alphas)))
        for i, layer in enumerate(layers):
            for j, alpha in enumerate(alphas):
                matches = [r["cos_sim"] for r in records
                           if r["layer"] == layer and r["alpha"] == alpha
                           and r["dim"] == dim]
                mat[i, j] = matches[0] if matches else np.nan

        im = ax.imshow(
            mat, aspect="auto", cmap=cmap,
            vmin=SAFE_COS_SIM - 0.02, vmax=1.0,
            extent=[alphas[0] - 0.5, alphas[-1] + 0.5,
                    len(layers) - 0.5, -0.5],
            origin="upper",
        )

        # overlay contour at threshold values
        X, Y = np.meshgrid(alphas, range(len(layers)))
        try:
            ax.contour(X, Y, mat,
                       levels=[SAFE_COS_SIM, WARN_COS_SIM],
                       colors=[C_RISKY, C_WARN],
                       linewidths=[1.5, 1.0],
                       linestyles=["--", ":"])
        except Exception:
            pass

        # mark standard alphas
        for std_a, sc in zip([-5, 5, 15], C_STANDARD):
            ax.axvline(std_a, color=sc, lw=1.5, ls="-.", alpha=0.85)

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Steering scalar α")
        ax.set_title(dim)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02,
                        label="Cosine similarity  (h, h′)")
    cbar.ax.axhline(SAFE_COS_SIM, color=C_RISKY, lw=1.2, ls="--")
    cbar.ax.axhline(WARN_COS_SIM, color=C_WARN,  lw=1.0, ls=":")

    std_patches = [
        mpatches.Patch(color=c, label=f"α = {a}")
        for a, c in zip(["−5", "+5", "+15"], C_STANDARD)
    ]
    fig.legend(handles=std_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Cosine Similarity Heatmap  (α × Layer per Dimension)\n"
        "Green = safe  |  Amber = moderate  |  Red = risky",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "fig3_dim_heatmaps.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Norm expansion contour (averaged across dims)
# ══════════════════════════════════════════════════════════════════════════

def fig4_norm_contour(records: list[dict], out_dir: Path, show: bool):
    layers = sorted(set(r["layer"] for r in records))
    alphas = sorted(set(r["alpha"] for r in records))

    # average norm_expansion_pct across dims
    mat = np.zeros((len(layers), len(alphas)))
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            vals = [abs(r["norm_expansion_pct"])
                    for r in records if r["layer"] == layer and r["alpha"] == alpha]
            mat[i, j] = np.mean(vals) if vals else np.nan

    # injection ratio matrix
    mat_inj = np.zeros_like(mat)
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            vals = [r["injection_ratio_pct"]
                    for r in records if r["layer"] == layer and r["alpha"] == alpha]
            mat_inj[i, j] = np.mean(vals) if vals else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    X, Y = np.meshgrid(alphas, range(len(layers)))

    for ax, data, title, units in [
        (axes[0], mat,     "|Norm Expansion| (%)", "%"),
        (axes[1], mat_inj, "Injection Ratio (%)",  "%"),
    ]:
        # filled contour
        n_levels = 30
        vmax = np.nanmax(data)
        levels = np.linspace(0, vmax, n_levels)
        cmap_safe = LinearSegmentedColormap.from_list(
            "norm_safe", ["#ffffff", C_WARN, C_RISKY], N=256
        )
        cf = ax.contourf(X, Y, data, levels=levels, cmap=cmap_safe)
        cs = ax.contour( X, Y, data,
                         levels=[1, 5, 10, 25, 50],
                         colors="black", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g%%")

        # danger contour line
        try:
            ax.contour(X, Y, data, levels=[SAFE_NORM_EXP],
                       colors=[C_RISKY], linewidths=2.0, linestyles="--")
        except Exception:
            pass

        # standard alpha lines
        for std_a, sc in zip([-5, 5, 15], C_STANDARD):
            ax.axvline(std_a, color=sc, lw=1.8, ls="-.", alpha=0.9)

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Steering scalar α")
        ax.set_title(title)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)

        cb = fig.colorbar(cf, ax=ax, pad=0.02)
        cb.set_label(units)
        # mark 50% danger level on colorbar
        cb.ax.axhline(SAFE_NORM_EXP if "Norm" in title else 100,
                      color=C_RISKY, lw=1.2, ls="--", alpha=0.8)

    std_patches = [
        mpatches.Patch(color=c, label=f"α = {a}")
        for a, c in zip(["−5", "+5", "+15"], C_STANDARD)
    ]
    axes[1].legend(handles=std_patches, loc="upper right", fontsize=8)

    fig.suptitle(
        "Norm Expansion & Injection Ratio Contours  —  Qwen 2.5-7B\n"
        "(Mean across dimensions, |α| ∈ [0, 50]   —   Dashed red = 50% danger threshold)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "fig4_norm_contour.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 5 — Composite radar report card for {-5, 5, 15}
# ══════════════════════════════════════════════════════════════════════════

def fig5_radar_report(records: list[dict], calib: dict, out_dir: Path, show: bool):
    """Radar chart: for each standard alpha, show 4 normalised safety metrics
    averaged across dims × layers. Larger area = more perturbative (not necessarily bad)."""

    standard_alphas = [-5, 5, 15]
    metric_labels = [
        "Norm\nExpansion %",
        "Injection\nRatio %",
        "Direction\nShift (1−CosSim)",
        "Rotation\nAngle (°)",
    ]
    metric_keys = ["norm_expansion_pct", "injection_ratio_pct",
                   "cos_sim_inv", "angle_deg"]
    # theoretical maxima for normalisation (at α=50)
    max_vals = [50.0, 7.0, 0.15, 90.0]

    N = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, len(standard_alphas),
        figsize=(5 * len(standard_alphas), 5.5),
        subplot_kw=dict(polar=True),
    )
    if len(standard_alphas) == 1:
        axes = [axes]

    for ax, alpha, col in zip(axes, standard_alphas, C_STANDARD):
        sub = [r for r in records if r["alpha"] == alpha]
        if not sub:
            ax.set_visible(False)
            continue

        means = {
            "norm_expansion_pct":  np.mean([abs(r["norm_expansion_pct"])  for r in sub]),
            "injection_ratio_pct": np.mean([r["injection_ratio_pct"]      for r in sub]),
            "cos_sim_inv":         np.mean([1.0 - r["cos_sim"]            for r in sub]),
            "angle_deg":           np.mean([abs(r["angle_deg"])           for r in sub]),
        }

        # normalise to [0, 1]
        vals = [means[k] / mx for k, mx in zip(metric_keys, max_vals)]
        vals += vals[:1]

        # reference circles for 25%, 50%, 75% of max
        for frac, ls, lbl in [(0.25, ":", "25%"), (0.50, "--", "50%"), (0.75, "-.", "75%")]:
            ref = [frac] * N + [frac]
            ax.plot(angles, ref, ls=ls, lw=0.7, color="grey", alpha=0.5)
            ax.text(angles[0], frac + 0.04, lbl, ha="center", va="center",
                    fontsize=6.5, color="grey")

        ax.fill(angles, vals, alpha=0.25, color=col)
        ax.plot(angles, vals, lw=2.0, color=col,
                marker="o", markersize=5)

        # value labels at each spoke
        for angle, val, raw_val, mx, lbl in zip(
            angles[:-1], vals[:-1],
            [means[k] for k in metric_keys], max_vals, metric_labels
        ):
            ax.text(angle, val + 0.10,
                    f"{raw_val:.2f}",
                    ha="center", va="center", fontsize=7.5, color=col,
                    fontweight="bold")

        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.spines["polar"].set_alpha(0.3)

        # colour ring by safety
        status = "SAFE" if alpha in [-5, 5] else "SAFE"   # all are safe per analysis
        if means["cos_sim_inv"] > (1 - SAFE_COS_SIM):
            status = "RISKY"
        elif means["cos_sim_inv"] > (1 - WARN_COS_SIM):
            status = "MODERATE"
        status_col = C_SAFE if status == "SAFE" else C_WARN if status == "MODERATE" else C_RISKY
        ax.set_title(
            f"α = {alpha:+d}",
            fontsize=13, fontweight="bold", color=col, pad=18
        )
        ax.text(0, -0.18, status, transform=ax.transAxes,
                ha="center", fontsize=9, fontweight="bold",
                color=status_col,
                bbox=dict(boxstyle="round,pad=0.3", fc=status_col, alpha=0.12))

    fig.suptitle(
        "Safety Report Card for Standard Sweep  {α = −5, +5, +15}\n"
        "Qwen 2.5-7B  ·  Mean across all dimensions and layers\n"
        "(Values are normalised — larger = more perturbative)",
        fontsize=11, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    out = out_dir / "fig5_radar_report.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Figure 6 — KDE "probability of safety" over alpha
# ══════════════════════════════════════════════════════════════════════════

def fig6_kde_safety(records: list[dict], calib: dict, out_dir: Path, show: bool):
    """
    For each layer, treat each (dim, alpha) cosine-sim value as a sample.
    Plot a kernel-density estimate of where in cos_sim space the mass falls
    as alpha increases — essentially a 2-D KDE over (alpha, cos_sim).
    Then separately show the fraction of (dim) configs rated SAFE/MODERATE/RISKY
    as a stacked area chart across alpha.
    """
    layers = sorted(set(r["layer"] for r in records))
    alphas = sorted(set(r["alpha"] for r in records))

    fig, axes = plt.subplots(2, len(layers), figsize=(5.5 * len(layers), 9))
    if len(layers) == 1:
        axes = axes.reshape(-1, 1)

    for col, layer in enumerate(layers):
        sub = [r for r in records if r["layer"] == layer]

        # ── Top panel: KDE of cos_sim distribution at each alpha ──────────
        ax_top = axes[0, col]

        # Collect all cos_sim values per alpha and build a 2-D grid
        cos_grid = np.zeros((len(alphas), 200))
        cos_axis = np.linspace(0.80, 1.0, 200)
        for j, alpha in enumerate(alphas):
            vals = np.array([r["cos_sim"] for r in sub if r["alpha"] == alpha])
            if len(vals) >= 2:
                try:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    cos_grid[j, :] = kde(cos_axis)
                except Exception:
                    pass

        # normalise each row
        row_max = cos_grid.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        cos_grid_n = cos_grid / row_max

        im = ax_top.imshow(
            cos_grid_n.T, aspect="auto", origin="lower",
            extent=[alphas[0], alphas[-1], 0.80, 1.0],
            cmap="YlGn", vmin=0, vmax=1,
        )
        # threshold lines
        ax_top.axhline(WARN_COS_SIM,  color=C_WARN,  lw=1.2, ls="--",
                       label=f"Moderate ({WARN_COS_SIM})")
        ax_top.axhline(SAFE_COS_SIM,  color=C_RISKY, lw=1.5, ls="--",
                       label=f"Risky boundary ({SAFE_COS_SIM})")
        for std_a, sc in zip([-5, 5, 15], C_STANDARD):
            ax_top.axvline(std_a, color=sc, lw=1.5, ls="-.", alpha=0.85)
        ax_top.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax_top.set_ylabel("Cosine similarity")
        ax_top.set_title(f"Layer {layer}  — KDE density")
        ax_top.legend(loc="lower center", fontsize=7, ncol=2)
        fig.colorbar(im, ax=ax_top, shrink=0.8, label="Normalised density")

        # ── Bottom panel: stacked area — fraction SAFE / MODERATE / RISKY ─
        ax_bot = axes[1, col]
        safe_frac, mod_frac, risky_frac = [], [], []
        for alpha in alphas:
            vals = [r for r in sub if r["alpha"] == alpha]
            n = len(vals) or 1
            classes = [classify(r["norm_expansion_pct"], r["cos_sim"]) for r in vals]
            safe_frac.append(classes.count("safe")     / n)
            mod_frac.append( classes.count("moderate") / n)
            risky_frac.append(classes.count("risky")   / n)

        xs = np.array(alphas)
        ax_bot.stackplot(
            xs,
            [np.array(safe_frac), np.array(mod_frac), np.array(risky_frac)],
            labels=["Safe", "Moderate", "Risky"],
            colors=[C_SAFE, C_WARN, C_RISKY],
            alpha=0.75,
        )
        # mark where safe fraction drops below 1.0
        safe_arr = np.array(safe_frac)
        first_drop = next((xs[i] for i in range(len(xs)) if safe_arr[i] < 1.0), None)
        if first_drop is not None:
            ax_bot.axvline(first_drop, color="black", lw=1.2, ls=":",
                           label=f"First safety drop (α≈{int(first_drop)})")

        for std_a, sc in zip([-5, 5, 15], C_STANDARD):
            ax_bot.axvline(std_a, color=sc, lw=1.5, ls="-.", alpha=0.85)
        ax_bot.axvline(0, color="grey", lw=0.8, alpha=0.4)
        ax_bot.set_xlim(xs.min(), xs.max())
        ax_bot.set_ylim(0, 1.01)
        ax_bot.set_xlabel("Steering scalar α")
        ax_bot.set_ylabel("Fraction of configs")
        ax_bot.set_title(f"Layer {layer}  — Safety classification")
        ax_bot.legend(loc="lower center", fontsize=7.5, ncol=3)

    std_patches = [
        mpatches.Patch(color=c, label=f"α = {a}")
        for a, c in zip(["−5", "+5", "+15"], C_STANDARD)
    ]
    fig.legend(handles=std_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    calib_mean = np.mean([v["mean_norm"] for v in calib.values()])
    fig.suptitle(
        f"Safety Distribution  —  KDE Density & Classification Fraction vs α\n"
        f"Qwen 2.5-7B  |  E[‖h‖] ≈ {calib_mean:.0f}  |  All dimensions",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = out_dir / "fig6_kde_safety.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "results" / "scalar_analysis_geo.json",
        help="Path to scalar_analysis_geo.json",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures only; do not call plt.show()",
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6],
        help="Which figures to produce (default: all 1–6)",
    )
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: JSON file not found: {args.json}")
        print("Run first:  python steering_scalar_analysis.py --skip-games")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    print(f"Loading {args.json} ...")
    geo, calib = load_geo(args.json)
    records    = flatten_geo(geo)
    print(f"  {len(records):,} records  |  "
          f"dims={sorted(geo.keys())}  |  "
          f"layers={sorted(int(l) for l in next(iter(geo.values())).keys())}  |  "
          f"alphas: [{min(r['alpha'] for r in records):.0f}, "
          f"{max(r['alpha'] for r in records):.0f}]")

    calib_mean = np.mean([v["mean_norm"] for v in calib.values()])
    print(f"  Calibration E[‖h‖] ≈ {calib_mean:.1f}")
    print(f"Output directory: {OUT_DIR}\n")

    dispatch = {
        1: (fig1_landscape,  "Fig 1: Safety landscape"),
        2: (fig2_bell_curve, "Fig 2: Safety envelope (bell curve)"),
        3: (fig3_dim_heatmaps,"Fig 3: Dim heatmaps"),
        4: (fig4_norm_contour,"Fig 4: Norm contour"),
        5: (fig5_radar_report,"Fig 5: Radar report card"),
        6: (fig6_kde_safety,  "Fig 6: KDE safety distribution"),
    }

    for fig_id in args.figs:
        fn, label = dispatch[fig_id]
        print(f"Plotting {label} ...")
        try:
            if fig_id in (1, 2):
                fn(records, calib, OUT_DIR, show)
            elif fig_id == 5:
                fn(records, calib, OUT_DIR, show)
            elif fig_id == 6:
                fn(records, calib, OUT_DIR, show)
            else:
                fn(records, OUT_DIR, show)
        except Exception as e:
            print(f"  WARNING: {label} failed — {e}")
            import traceback; traceback.print_exc()

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
