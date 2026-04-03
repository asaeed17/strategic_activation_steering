"""
Generate publication-quality figures from ultimatum game steering experiments.

Reads JSON results from the final 7B LLM-vs-LLM grid
(results/ultimatum/final_7b_llm_vs_llm/) and produces 10 figures
for the COMP0087 SNLP coursework paper.

Usage:
    python analysis/plot_results.py

Output:
    results/figures/fig{1..8}_{name}.{png,pdf}
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_DIR = os.path.join(BASE, "results", "ultimatum", "final_7b_llm_vs_llm")
# Teammate data for cross-design comparison (Fig 8)
TEAMMATE_RB = os.path.join(BASE, "results", "ultimatum", "llm_vs_rulebased")
OUT_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
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
    "font.family": "sans-serif",
})

LAYERS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
ALPHAS = [-7, -5, 5, 7, 15]
DIMENSIONS = [
    "firmness", "empathy", "anchoring", "fairness_norm", "narcissism",
    "spite", "greed", "flattery", "composure", "undecidedness",
]

# Pretty names for display
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

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _try_load(path):
    """Load JSON file, return (summary_dict, n_usable_pairs) or (None, 0)."""
    if not os.path.exists(path):
        return None, 0
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    return s, s.get("n_usable_pairs", s.get("n_valid", 0))


def _final_path(dim, layer, alpha):
    """Return path in the final grid directory."""
    alpha_str = f"{float(alpha):.1f}"
    fname = f"{dim}_proposer_L{layer}_a{alpha_str}_paired_n50.json"
    return os.path.join(FINAL_DIR, fname)


def get_data(dim, layer, alpha):
    """
    Return (cohens_d, p_value, n, summary_dict) from the final grid.
    """
    path = _final_path(dim, layer, alpha)
    s, n = _try_load(path)
    if s is not None:
        return s["cohens_d"], s["paired_ttest_p"], n, s
    return None, None, 0, None


def savefig(fig, name):
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Load entire grid into memory (used by Figs 7, 8, and others)
# ---------------------------------------------------------------------------

def load_full_grid():
    """Load all results from the final grid into a list of dicts."""
    records = []
    for f in glob.glob(os.path.join(FINAL_DIR, "*.json")):
        with open(f) as fh:
            data = json.load(fh)
        s = data["summary"]
        # Parse filename: {dim}_proposer_L{layer}_a{alpha}_paired_n50.json
        basename = os.path.basename(f)
        parts = basename.replace(".json", "").split("_proposer_")
        dim = parts[0]
        rest = parts[1]  # L{layer}_a{alpha}_paired_n50
        tokens = rest.split("_")
        layer = int(tokens[0].replace("L", ""))
        alpha = float(tokens[1].replace("a", ""))
        records.append({
            "dim": dim,
            "layer": layer,
            "alpha": alpha,
            "cohens_d": s["cohens_d"],
            "p_value": s["paired_ttest_p"],
            "n": s.get("n_usable_pairs", s.get("n_valid", 0)),
            "delta_pct": s["delta_proposer_pct"],
            "steered_accept_rate": s.get("steered_accept_rate", np.nan),
            "baseline_accept_rate": s.get("baseline_accept_rate", np.nan),
            "steered_payoff": s.get("steered_mean_payoff_pct", np.nan),
            "baseline_payoff": s.get("baseline_mean_payoff_pct", np.nan),
            "summary": s,
        })
    return records


# ---------------------------------------------------------------------------
# Figure 1, 2, 2b: Dimension x Layer Heat Maps
# ---------------------------------------------------------------------------

def make_heatmap(alpha, fig_num):
    """Create a dimension x layer heat map for a given alpha."""
    # Collect data
    d_matrix = np.full((len(DIMENSIONS), len(LAYERS)), np.nan)
    p_matrix = np.full((len(DIMENSIONS), len(LAYERS)), np.nan)

    for i, dim in enumerate(DIMENSIONS):
        for j, layer in enumerate(LAYERS):
            d, p, n, s = get_data(dim, layer, alpha)
            if d is not None:
                d_matrix[i, j] = d
                p_matrix[i, j] = p

    # Order dimensions by peak layer location (layer index of max |d|)
    peak_layers = []
    for i in range(len(DIMENSIONS)):
        row = d_matrix[i, :]
        if np.all(np.isnan(row)):
            peak_layers.append(len(LAYERS))  # push to end
        else:
            peak_layers.append(np.nanargmax(np.abs(row)))
    order = np.argsort(peak_layers)
    d_matrix = d_matrix[order, :]
    p_matrix = p_matrix[order, :]
    ordered_dims = [DIMENSIONS[i] for i in order]
    ordered_labels = [DIM_LABELS[d] for d in ordered_dims]

    # Determine symmetric color range
    vmax = np.nanmax(np.abs(d_matrix))
    vmax = max(vmax, 0.5)  # minimum range

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(d_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(ordered_dims)):
        for j in range(len(LAYERS)):
            d_val = d_matrix[i, j]
            p_val = p_matrix[i, j]
            if np.isnan(d_val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7,
                        color="gray")
                continue
            sig = p_val is not None and p_val < 0.05
            text = f"{d_val:.2f}"
            if sig:
                text += "*"
            # Choose text color for readability
            text_color = "white" if abs(d_val) > 0.6 * vmax else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, fontweight="bold" if sig else "normal",
                    color=text_color)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_yticks(range(len(ordered_dims)))
    ax.set_yticklabels(ordered_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dimension")

    alpha_label = f"+{alpha}" if alpha > 0 else str(alpha)
    ax.set_title(f"Steering effect size by dimension and layer "
                 f"(\u03b1 = {alpha_label}, n=50)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cohen's d")

    fig.tight_layout()
    savefig(fig, f"fig{fig_num}_heatmap_alpha{'+' if alpha > 0 else ''}{alpha}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Layer Gradient Curves
# ---------------------------------------------------------------------------

def make_layer_gradient():
    """Plot layer gradient curves for top 5 dimensions."""
    # Top 5 configs per spec: firmness(+7), greed(+7), composure(+15),
    # fairness_norm(+7), narcissism(-7)
    curves = [
        ("firmness", 7, "Firmness (+7)"),
        ("greed", 7, "Greed (+7)"),
        ("composure", 15, "Composure (+15)"),
        ("fairness_norm", 7, "Fairness norm (+7)"),
        ("narcissism", -7, "Narcissism (-7)"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("tab10", len(curves))
    markers = ["o", "s", "D", "^", "v"]

    for idx, (dim, alpha, label) in enumerate(curves):
        x_vals = []
        y_vals = []
        for layer in LAYERS:
            d, p, n, s = get_data(dim, layer, alpha)
            if d is not None:
                x_vals.append(layer)
                y_vals.append(d)
        ax.plot(x_vals, y_vals, marker=markers[idx], color=colors[idx],
                label=label, linewidth=1.8, markersize=6, zorder=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Layer gradient: different dimensions peak at different layers")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig3_layer_gradient")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Dose-Response Comparison
# ---------------------------------------------------------------------------

def make_dose_response():
    """Two-panel dose-response: Firmness L10 and Greed L12 across all 5 alphas."""
    all_alphas = [-7, -5, 5, 7, 15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    bar_colors_map = {-7: "#3274A1", -5: "#6BAED6", 5: "#FDAE6B",
                      7: "#E6550D", 15: "#A63603"}

    def _plot_panel(ax, dim, layer, title):
        d_vals = []
        p_vals = []
        labels = []
        colors = []
        for a in all_alphas:
            d, p, n, s = get_data(dim, layer, a)
            d_vals.append(d if d is not None else 0)
            p_vals.append(p)
            a_label = f"+{a}" if a > 0 else str(a)
            labels.append(f"\u03b1={a_label}")
            colors.append(bar_colors_map[a])
            print(f"  Fig4 {dim} L{layer} a={a}: d={d}, p={p}, n={n}")

        x = range(len(all_alphas))
        ax.bar(x, d_vals, color=colors, edgecolor="black", linewidth=0.5,
               width=0.65)
        for i, (dv, pv) in enumerate(zip(d_vals, p_vals)):
            sig = pv is not None and pv < 0.05
            label = f"{dv:.2f}"
            if sig:
                label += "*"
            y_pos = dv + 0.05 if dv >= 0 else dv - 0.05
            va = "bottom" if dv >= 0 else "top"
            ax.text(i, y_pos, label, ha="center", va=va, fontsize=8,
                    fontweight="bold" if sig else "normal")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5)

    _plot_panel(ax1, "firmness", 10, "A: Firmness L10")
    ax1.set_ylabel("Cohen's d")
    _plot_panel(ax2, "greed", 12, "B: Greed L12")

    fig.suptitle("Dose-response across all 5 alphas", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig4_dose_response")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Sign Comparison at Peak Layer
# ---------------------------------------------------------------------------

def make_sign_comparison():
    """Grouped bar chart showing alpha=+7 vs alpha=-7 at best layer per dim."""
    # Find best layer per dim (max |d| across layers at alpha=+7 or -7)
    configs = []
    for dim in DIMENSIONS:
        best_layer = None
        best_abs_d = -1
        for layer in LAYERS:
            for a in [7, -7]:
                d, p, n, s = get_data(dim, layer, a)
                if d is not None and abs(d) > best_abs_d:
                    best_abs_d = abs(d)
                    best_layer = layer
        if best_layer is not None:
            configs.append((dim, best_layer))

    labels = [f"{DIM_LABELS[c[0]]} (L{c[1]})" for c in configs]
    pos_d = []
    neg_d = []

    for dim, layer in configs:
        d_pos, _, _, _ = get_data(dim, layer, 7)
        d_neg, _, _, _ = get_data(dim, layer, -7)
        pos_d.append(d_pos if d_pos is not None else 0)
        neg_d.append(d_neg if d_neg is not None else 0)
        print(f"  Fig5: {dim} L{layer}: +7 d={d_pos}, -7 d={d_neg}")

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, pos_d, width, label="\u03b1 = +7",
           color=sns.color_palette("tab10")[3],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, neg_d, width, label="\u03b1 = -7",
           color=sns.color_palette("tab10")[0],
           edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's d")
    ax.set_title("Sign asymmetry: some dimensions only respond "
                 "to one steering direction")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig5_sign_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Quantization Effect (unchanged — hardcoded values)
# ---------------------------------------------------------------------------

def make_quantization_effect():
    """Show quantization shifts the effective steering dose for firmness L10."""
    # Hardcoded quantized values (4-bit, from the spec)
    quant_d = {3: 1.07, 7: 1.38, 10: 1.54}

    # Unquantized values from final grid (bfloat16)
    unquant_d = {}
    for a in [3, 7, 10]:
        d, p, n, s = get_data("firmness", 10, a)
        # a=3 and a=10 may not exist in the final grid (only -7,-5,5,7,15)
        # Fall back to hardcoded bfloat16 values from prior runs
        if d is not None:
            unquant_d[a] = d
        else:
            # Fallback: these come from layer_gradient_v2 runs
            fallback = {3: 0.55, 7: None, 10: 1.42}
            unquant_d[a] = fallback.get(a, 0)
        print(f"  Fig6: firmness L10 a={a} unquantized: d={unquant_d[a]}")

    # Use alpha=7 from final grid if available
    if unquant_d[7] is None:
        d7, _, _, _ = get_data("firmness", 10, 7)
        unquant_d[7] = d7 if d7 is not None else 1.27

    alphas = [3, 7]
    x = np.arange(len(alphas))
    width = 0.3

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    quant_vals = [quant_d[a] for a in alphas]
    unquant_vals = [unquant_d[a] for a in alphas]

    bars_q = ax.bar(x - width/2, quant_vals, width, label="4-bit quantized",
                    color=sns.color_palette("Set2")[0],
                    edgecolor="black", linewidth=0.5)
    bars_u = ax.bar(x + width/2, unquant_vals, width, label="bfloat16",
                    color=sns.color_palette("Set2")[1],
                    edgecolor="black", linewidth=0.5)

    # Annotate
    for i, a in enumerate(alphas):
        for j, (val, offset) in enumerate([(quant_vals[i], -width/2),
                                            (unquant_vals[i], width/2)]):
            ax.text(x[i] + offset, val + 0.04, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"\u03b1 = {a}" for a in alphas])
    ax.set_ylabel("Cohen's d")
    ax.set_title("Quantization shifts the effective steering dose")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-0.2, 1.8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    savefig(fig, "fig6_quantization")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Pareto Front — |demand shift| vs acceptance rate / payoff
# ---------------------------------------------------------------------------

def make_pareto(grid):
    """Two-panel Pareto: X=|demand shift|, Y=acceptance rate (left), payoff (right).
    Label top configs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    abs_deltas = [abs(r["delta_pct"]) for r in grid]
    accept_rates = [r["steered_accept_rate"] for r in grid]
    payoffs = [r["steered_payoff"] for r in grid]
    labels = [f"{r['dim']} L{r['layer']} a={r['alpha']:+g}" for r in grid]
    sig = [r["p_value"] is not None and r["p_value"] < 0.05 for r in grid]

    # Colors by dimension
    unique_dims = sorted(set(r["dim"] for r in grid))
    dim_colors = {d: c for d, c in zip(unique_dims, sns.color_palette("tab10", len(unique_dims)))}
    colors = [dim_colors[r["dim"]] for r in grid]

    # --- Panel A: |demand shift| vs acceptance rate ---
    for i in range(len(grid)):
        marker = "o" if sig[i] else "x"
        ec = "none" if sig[i] else colors[i]
        ax1.scatter(abs_deltas[i], accept_rates[i], color=colors[i],
                    marker=marker, s=25, alpha=0.7, edgecolors=ec)

    # Label top configs: highest payoff configs
    payoff_order = np.argsort(payoffs)[::-1]
    labeled = set()
    for idx in payoff_order[:8]:
        key = f"{grid[idx]['dim']}_{grid[idx]['layer']}"
        if key not in labeled:
            ax1.annotate(f"{DIM_LABELS.get(grid[idx]['dim'], grid[idx]['dim'])} "
                         f"L{grid[idx]['layer']} a={grid[idx]['alpha']:+g}",
                         (abs_deltas[idx], accept_rates[idx]),
                         fontsize=6, alpha=0.8,
                         textcoords="offset points", xytext=(5, 3))
            labeled.add(key)

    ax1.set_xlabel("|Demand shift| (pp)")
    ax1.set_ylabel("Acceptance rate")
    ax1.set_title("A: Demand shift vs acceptance rate")
    ax1.grid(True, alpha=0.3)

    # --- Panel B: |demand shift| vs payoff ---
    for i in range(len(grid)):
        marker = "o" if sig[i] else "x"
        ec = "none" if sig[i] else colors[i]
        ax2.scatter(abs_deltas[i], payoffs[i], color=colors[i],
                    marker=marker, s=25, alpha=0.7, edgecolors=ec)

    # Baseline payoff line
    baseline_payoffs = [r["baseline_payoff"] for r in grid if not np.isnan(r["baseline_payoff"])]
    if baseline_payoffs:
        ax2.axhline(np.mean(baseline_payoffs), color="gray", linestyle="--",
                     linewidth=0.8, alpha=0.6, label="Baseline avg")

    # Label top payoff configs
    labeled = set()
    for idx in payoff_order[:8]:
        key = f"{grid[idx]['dim']}_{grid[idx]['layer']}"
        if key not in labeled:
            ax2.annotate(f"{DIM_LABELS.get(grid[idx]['dim'], grid[idx]['dim'])} "
                         f"L{grid[idx]['layer']} a={grid[idx]['alpha']:+g}",
                         (abs_deltas[idx], payoffs[idx]),
                         fontsize=6, alpha=0.8,
                         textcoords="offset points", xytext=(5, 3))
            labeled.add(key)

    ax2.set_xlabel("|Demand shift| (pp)")
    ax2.set_ylabel("Payoff (%)")
    ax2.set_title("B: Demand shift vs payoff")
    ax2.legend(loc="best", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Legend for dimensions (shared)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=dim_colors[d], markersize=7,
                              label=DIM_LABELS.get(d, d))
                       for d in unique_dims]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Pareto landscape: demand shift vs outcome", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    savefig(fig, "fig7_pareto_demand_vs_outcome")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Cross-Design Comparison (our data + teammate rule-based data)
# ---------------------------------------------------------------------------

def _load_teammate_configs():
    """Load teammate rule-based results for cross-design comparison.
    Returns list of dicts with dim, layer, alpha, cohens_d, etc."""
    records = []
    # Scan all teammate directories
    if not os.path.exists(TEAMMATE_RB):
        print("  WARNING: Teammate data directory not found, skipping Fig 8")
        return records

    for run_dir in sorted(os.listdir(TEAMMATE_RB)):
        run_path = os.path.join(TEAMMATE_RB, run_dir)
        if not os.path.isdir(run_path):
            continue
        # Traverse subdirectory structure: run_dir/L{x}/proposer/{dim}/final_best.json
        for root, dirs, files in os.walk(run_path):
            if "final_best.json" not in files:
                continue
            with open(os.path.join(root, "final_best.json")) as f:
                data = json.load(f)
            # Extract dim from path
            dim = os.path.basename(root)
            if dim not in DIMENSIONS:
                continue
            # Extract layer
            layers = data.get("layers", [])
            layer = layers[0] if layers else None
            best_alpha = data.get("best_alpha")
            if layer is None or best_alpha is None:
                continue
            # Get the best alpha's offer shift
            alpha_key = str(float(best_alpha))
            alpha_data = data.get("scores_per_alpha", {}).get(alpha_key, {})
            offer_shift = alpha_data.get("offer_shift", {})
            d_val = offer_shift.get("cohens_d")
            if d_val is None:
                continue
            records.append({
                "dim": dim,
                "layer": layer,
                "alpha": best_alpha,
                "cohens_d": d_val,
                "delta_pct": offer_shift.get("mean_delta_pct", 0),
                "source": run_dir,
            })
    return records


def make_cross_design(grid):
    """Compare our LLM-vs-LLM results with teammate's rule-based results."""
    teammate_data = _load_teammate_configs()
    if not teammate_data:
        print("  Skipping Fig 8: no teammate data")
        return

    # For our data: pick best config per dimension (highest |d| at alpha=+7)
    our_best = {}
    for r in grid:
        dim = r["dim"]
        if dim not in our_best or abs(r["cohens_d"]) > abs(our_best[dim]["cohens_d"]):
            our_best[dim] = r

    # For teammate data: pick best per dimension
    tm_best = {}
    for r in teammate_data:
        dim = r["dim"]
        if dim not in tm_best or abs(r["cohens_d"]) > abs(tm_best[dim]["cohens_d"]):
            tm_best[dim] = r

    # Only plot dimensions present in both
    common_dims = sorted(set(our_best.keys()) & set(tm_best.keys()))
    if not common_dims:
        print("  Skipping Fig 8: no common dimensions")
        return

    labels = [DIM_LABELS.get(d, d) for d in common_dims]
    our_d = [our_best[d]["cohens_d"] for d in common_dims]
    tm_d = [tm_best[d]["cohens_d"] for d in common_dims]

    x = np.arange(len(common_dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, our_d, width, label="LLM-vs-LLM (ours)",
           color=sns.color_palette("Set2")[0],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, tm_d, width, label="LLM-vs-RuleBased (teammate)",
           color=sns.color_palette("Set2")[1],
           edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cohen's d (best config)")
    ax.set_title("Cross-design comparison: LLM-vs-LLM vs LLM-vs-RuleBased")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for i in range(len(common_dims)):
        for val, offset in [(our_d[i], -width/2), (tm_d[i], width/2)]:
            y_pos = val + 0.05 if val >= 0 else val - 0.05
            va = "bottom" if val >= 0 else "top"
            ax.text(x[i] + offset, y_pos, f"{val:.2f}",
                    ha="center", va=va, fontsize=7)

    fig.tight_layout()
    savefig(fig, "fig8_cross_design_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Generating publication figures")
    print("=" * 60)

    print("\nFigure 1: Heat map (alpha = +7)")
    make_heatmap(7, 1)

    print("\nFigure 2: Heat map (alpha = -7)")
    make_heatmap(-7, 2)

    print("\nFigure 2b: Heat map (alpha = +15)")
    make_heatmap(15, "2b")

    print("\nFigure 3: Layer gradient curves")
    make_layer_gradient()

    print("\nFigure 4: Dose-response comparison")
    make_dose_response()

    print("\nFigure 5: Sign comparison at peak layer")
    make_sign_comparison()

    print("\nFigure 6: Quantization effect")
    make_quantization_effect()

    # Load full grid for Figs 7-8
    print("\nLoading full grid data...")
    grid = load_full_grid()
    print(f"  Loaded {len(grid)} configs")

    print("\nFigure 7: Pareto (demand shift vs outcome)")
    make_pareto(grid)

    print("\nFigure 8: Cross-design comparison")
    make_cross_design(grid)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
