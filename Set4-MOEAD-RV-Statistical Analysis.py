"""
MOEA/D-RV — Set 4: Statistical Analysis

Following AB-DMOEA (Peng et al., 2024) and ECICPS (Wang et al., 2025).

Tests performed:
    1. Wilcoxon signed-rank test (pairwise comparisons)
    2. Friedman test (multiple-algorithm comparison)
    3. Holm-Bonferroni correction (family-wise error control)
    4. Cohen's d effect size (practical significance)

Inputs are discovered automatically under ./results_set1/, ./results_set2/
and ./results_set4/; the newest .json file in each folder is used.

Outputs are written to ./statistical_analysis/:
    - summary JSON per experiment set
    - Friedman rank bar chart, critical difference diagram, p-value heatmap,
      Wilcoxon improvement bar chart, MIGD distribution boxplot

Usage:
    python "Set5-Statistical_Analysis.py"
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

warnings.filterwarnings("ignore")


# ============================================================================
# CONSTANTS
# ============================================================================

SIGNIFICANCE_LEVEL = 0.05
MIGD_CAP = 1e6  # cap for baseline overflow values

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "statistical_analysis"

STAT_COLORS = {
    "MOEA/D":     "#1B5E20",
    "MOEA/D-KNN": "#0D47A1",
    "MOEA/D-PPS": "#E65100",
    "MOEA/D-AGR": "#4A148C",
    "MOEA/D-HSS": "#00838F",
    "MOEA/D-RV":  "#C62828",
    "Full":           "#C62828",
    "w/o Risk":       "#E65100",
    "w/o HV":         "#0D47A1",
    "w/o Prediction": "#4A148C",
    "w/o Niching":    "#00838F",
    "w/o Scenario":   "#2E7D32",
    "Standard": "#1B5E20",
    "Mild":     "#0D47A1",
    "Rapid":    "#C62828",
}

ALGORITHM_NAMES = {
    "MOEA/D":     "MOEA/D",
    "MOEA/D-KNN": "MOEA/D-KNN",
    "MOEA/D-PPS": "MOEA/D-PPS",
    "MOEA/D-AGR": "MOEA/D-AGR",
    "MOEA/D-HSS": "MOEA/D-HSS",
    "MOEA/D-RV":  "MOEA/D-RV",
    "Full":           "Full",
    "w/o Risk":       "w/o Risk",
    "w/o HV":         "w/o HV",
    "w/o Prediction": "w/o Prediction",
    "w/o Niching":    "w/o Niching",
    "w/o Scenario":   "w/o Scenario",
}

ABLATION_VARIANTS = ["Full", "w/o Risk", "w/o HV", "w/o Prediction", "w/o Niching", "w/o Scenario"]
CONFIG_NAMES = ["Standard", "Mild", "Rapid"]


# ============================================================================
# HELPERS
# ============================================================================

def cap_migd(val):
    """Cap MIGD values (baseline divergence produces overflow)."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    if np.isinf(v):
        return MIGD_CAP
    return min(v, MIGD_CAP)


def discover_results_files():
    """Return the newest .json in each of the three results folders."""
    out = {}
    for set_name, folder in [("set1", "results_set1"),
                              ("set2", "results_set2"),
                              ("set4", "results_set4")]:
        d = BASE_DIR / folder
        if not d.exists():
            print(f"  ! Folder not found for {set_name}: {d}")
            out[set_name] = None
            continue
        jsons = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out[set_name] = jsons[0] if jsons else None
        if out[set_name] is None:
            print(f"  ! No .json found in {d}")
        else:
            print(f"  {set_name}: {out[set_name].name}")
    return out


# ============================================================================
# DATA LOADING
# ============================================================================

def load_set1_results(fp):
    """Load Set 1 (flat: results[problem][algorithm])."""
    if fp is None or not fp.exists():
        return {}
    with open(fp, "r") as f:
        data = json.load(f)
    raw = data.get("results", data)
    out = {}
    for problem, algos in raw.items():
        out[problem] = {}
        for algo, metrics in algos.items():
            if isinstance(metrics, dict):
                m = dict(metrics)
                m["migd_mean"] = cap_migd(m.get("migd_mean"))
                out[problem][algo] = m
    return out


def load_set2_results(fp):
    """Load Set 2 (nested: results[problem][config][algorithm])."""
    if fp is None or not fp.exists():
        return {}
    with open(fp, "r") as f:
        data = json.load(f)
    raw = data.get("results", data)
    flat = {}
    for problem, configs in raw.items():
        flat[problem] = {}
        for config, algos in configs.items():
            for algo, metrics in algos.items():
                key = f"{algo}_{config}"
                if isinstance(metrics, dict):
                    m = dict(metrics)
                    m["migd_mean"] = cap_migd(m.get("migd_mean"))
                    flat[problem][key] = m
    return flat


def load_set4_results(fp):
    """Load Set 4 (flat: results[problem][variant])."""
    if fp is None or not fp.exists():
        return {}
    with open(fp, "r") as f:
        data = json.load(f)
    raw = data.get("results", data)
    out = {}
    for problem, variants in raw.items():
        out[problem] = {}
        for variant, metrics in variants.items():
            if isinstance(metrics, dict):
                m = dict(metrics)
                m["migd_mean"] = cap_migd(m.get("migd_mean"))
                out[problem][variant] = m
    return out


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def holm_bonferroni(p_values, alpha=0.05):
    p = np.array(p_values)
    idx = np.argsort(p)
    sp = p[idx]
    rejected = np.zeros(len(p), dtype=bool)
    for i, pv in enumerate(sp):
        if pv <= alpha / (len(p) - i):
            rejected[idx[i]] = True
        else:
            break
    return rejected.tolist()


def get_common_problems(results, algorithms):
    common = []
    for problem in results.keys():
        ok = True
        for algo in algorithms:
            if algo not in results[problem]:
                ok = False; break
            v = results[problem][algo].get("migd_mean")
            if v is None or np.isnan(v) or v == float("inf"):
                ok = False; break
        if ok:
            common.append(problem)
    return common


def wilcoxon_pairwise_analysis(results, target="MOEA/D-RV"):
    """
    Pairwise Wilcoxon vs `target`. Uses alternative='less' on
    (target, competitor), which tests H1: target < competitor (i.e.,
    target has lower MIGD, which is the better outcome).

    `improvement_pct` = (mean_competitor - mean_target) / mean_competitor * 100,
    so positive values mean the target achieved a lower MIGD than the
    competitor on average.
    """
    comparisons = {}
    algorithms = sorted(results[list(results.keys())[0]].keys())
    if target not in algorithms:
        print(f"  ! Target '{target}' not found")
        return comparisons
    common = get_common_problems(results, algorithms)
    if len(common) < 2:
        return comparisons
    target_vals = [results[p][target]["migd_mean"] for p in common]
    for algo in algorithms:
        if algo == target:
            continue
        algo_vals = [results[p][algo]["migd_mean"] for p in common]
        try:
            stat, pval = wilcoxon(target_vals, algo_vals, alternative="less")
            better = "RV better" if np.mean(target_vals) < np.mean(algo_vals) else "Competitor better"
            comparisons[algo] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "better": better,
                "significant": bool(pval < SIGNIFICANCE_LEVEL),
                "algorithm_name": ALGORITHM_NAMES.get(algo, algo),
                "mean_target": float(np.mean(target_vals)),
                "mean_competitor": float(np.mean(algo_vals)),
                "std_target": float(np.std(target_vals)),
                "std_competitor": float(np.std(algo_vals)),
                "improvement_pct": float(
                    (np.mean(algo_vals) - np.mean(target_vals)) / (np.mean(algo_vals) + 1e-8) * 100
                ),
                "n_pairs": int(len(target_vals)),
            }
        except Exception as e:
            comparisons[algo] = {"error": str(e)}
    return comparisons


def wilcoxon_ablation_analysis(results):
    """
    Pairwise Wilcoxon of each ablation variant against Full.

    `improvement_pct` = (mean_variant - mean_full) / mean_variant * 100,
    so positive values mean the Full configuration achieved a lower MIGD.
    """
    comparisons = {}
    algorithms = sorted(results[list(results.keys())[0]].keys())
    if "Full" not in algorithms:
        return comparisons
    common = get_common_problems(results, algorithms)
    if len(common) < 2:
        return comparisons
    full_vals = [results[p]["Full"]["migd_mean"] for p in common]
    for algo in algorithms:
        if algo == "Full" or algo not in ABLATION_VARIANTS:
            continue
        algo_vals = [results[p][algo]["migd_mean"] for p in common]
        try:
            stat, pval = wilcoxon(full_vals, algo_vals, alternative="less")
            better = "Full better" if np.mean(full_vals) < np.mean(algo_vals) else "Ablation better"
            comparisons[algo] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "better": better,
                "significant": bool(pval < SIGNIFICANCE_LEVEL),
                "algorithm_name": ALGORITHM_NAMES.get(algo, algo),
                "mean_full": float(np.mean(full_vals)),
                "mean_competitor": float(np.mean(algo_vals)),
                "std_full": float(np.std(full_vals)),
                "std_competitor": float(np.std(algo_vals)),
                "improvement_pct": float(
                    (np.mean(algo_vals) - np.mean(full_vals)) / (np.mean(algo_vals) + 1e-8) * 100
                ),
                "n_pairs": int(len(full_vals)),
            }
        except Exception as e:
            comparisons[algo] = {"error": str(e)}
    return comparisons


def friedman_test_analysis(results):
    algorithms = sorted(results[list(results.keys())[0]].keys())
    common = get_common_problems(results, algorithms)
    if len(common) < 2:
        return None, None, algorithms
    data = np.array([[results[p][a]["migd_mean"] for a in algorithms] for p in common])
    try:
        stat, pval = friedmanchisquare(*data.T)
        return float(stat), float(pval), algorithms
    except Exception:
        return None, None, algorithms


def compute_friedman_ranks(results):
    algorithms = sorted(results[list(results.keys())[0]].keys())
    ranks = {a: [] for a in algorithms}
    for problem in results.keys():
        vals, algos = [], []
        for a in algorithms:
            if a in results[problem]:
                v = results[problem][a].get("migd_mean")
                if v is not None and not np.isnan(v) and v != float("inf"):
                    vals.append(v); algos.append(a)
        if len(vals) >= 2:
            order = np.argsort(vals)
            for r, idx in enumerate(order):
                ranks[algos[idx]].append(r + 1)
    return {a: (float(np.mean(ranks[a])) if ranks[a] else None) for a in algorithms}


def compute_win_loss_counts(results, target="MOEA/D-RV"):
    algorithms = sorted(results[list(results.keys())[0]].keys())
    if target not in algorithms:
        return {}
    out = {}
    for algo in algorithms:
        if algo == target:
            continue
        w = l = t = 0
        for p in results.keys():
            if target in results[p] and algo in results[p]:
                tv = results[p][target].get("migd_mean")
                av = results[p][algo].get("migd_mean")
                if tv is not None and av is not None:
                    if not np.isnan(tv) and not np.isnan(av):
                        if tv < av:
                            w += 1
                        elif tv > av:
                            l += 1
                        else:
                            t += 1
        out[algo] = {"wins": w, "losses": l, "ties": t,
                     "algorithm_name": ALGORITHM_NAMES.get(algo, algo)}
    return out


def interpret_effect_size(d):
    a = abs(d)
    if a < 0.2: return "Negligible"
    if a < 0.5: return "Small"
    if a < 0.8: return "Medium"
    return "Large"


def compute_effect_size(comparisons):
    out = {}
    for algo, comp in comparisons.items():
        if "error" in comp or "mean_target" not in comp:
            continue
        st, sc = comp.get("std_target", 0.1), comp.get("std_competitor", 0.1)
        n = comp.get("n_pairs", 10)
        pooled = np.sqrt(((n - 1) * st ** 2 + (n - 1) * sc ** 2) / (2 * n - 2))
        diff = comp["mean_competitor"] - comp["mean_target"]
        d = diff / (pooled + 1e-8)
        out[algo] = {"effect_size": float(d),
                     "interpretation": interpret_effect_size(d),
                     "algorithm_name": comp["algorithm_name"]}
    return out


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_friedman_ranks(avg_ranks, save_path=None, title_suffix=""):
    items = sorted([(a, r) for a, r in avg_ranks.items() if r is not None], key=lambda x: x[1])
    if not items:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    algos = [i[0] for i in items]
    ranks = [i[1] for i in items]
    labels = [ALGORITHM_NAMES.get(a, a) for a in algos]
    colors = [STAT_COLORS.get(a, "#888888") for a in algos]
    bars = ax.bar(labels, ranks, color=colors, edgecolor="#333333", linewidth=1.2, alpha=0.85)
    for bar, r in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{r:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Algorithms", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Friedman Rank (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(f"Friedman Rank Comparison{title_suffix}", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_critical_difference_diagram(avg_ranks, save_path=None):
    items = sorted([(a, r) for a, r in avg_ranks.items() if r is not None], key=lambda x: x[1])
    if len(items) < 2:
        return
    n = len(items)
    q = 2.569
    cd = q * np.sqrt((n * (n + 1)) / (6 * 14))
    fig, ax = plt.subplots(figsize=(12, 4))
    ranks = [i[1] for i in items]
    labels = [ALGORITHM_NAMES.get(i[0], i[0]) for i in items]
    ypos = np.arange(len(items))
    ax.scatter(ranks, ypos, s=100, color="#333333", zorder=3)
    for i in range(len(items)):
        ax.plot([ranks[i], ranks[i]], [ypos[i] - 0.3, ypos[i] + 0.3],
                color="#333333", linewidth=1, zorder=2)
        ax.text(ranks[i] + 0.05, i, f"{labels[i]} ({ranks[i]:.2f})",
                va="center", fontsize=10)
    ax.set_xlabel("Average Rank (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title("Critical Difference Diagram (α = 0.05)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(min(ranks) - 1, max(ranks) + 1)
    ax.set_ylim(-0.5, len(items) - 0.5)
    ax.invert_yaxis()
    ax.annotate(f"CD = {cd:.2f}", xy=(0.95, 0.05), xycoords="axes fraction",
                fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_p_value_heatmap(results, save_path=None):
    algorithms = sorted(results[list(results.keys())[0]].keys())
    n = len(algorithms)
    pmat = np.ones((n, n))
    for i, ai in enumerate(algorithms):
        for j, aj in enumerate(algorithms):
            if i == j:
                continue
            vi, vj = [], []
            for p in results.keys():
                if ai in results[p] and aj in results[p]:
                    a = results[p][ai].get("migd_mean")
                    b = results[p][aj].get("migd_mean")
                    if a is not None and b is not None and not np.isnan(a) and not np.isnan(b):
                        vi.append(a); vj.append(b)
            if len(vi) >= 2:
                try:
                    _, pv = mannwhitneyu(vi, vj)
                    pmat[i, j] = pv
                except Exception:
                    pass
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    labels = [ALGORITHM_NAMES.get(a, a) for a in algorithms]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{pmat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            if i != j and pmat[i, j] < 0.05:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#C62828", linewidth=2))
    ax.set_title("Pairwise P-Value Heatmap (red = significant)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="P-value")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_wilcoxon_improvements(comparisons, save_path=None, title_suffix=""):
    """
    Bar chart of `improvement_pct` vs each competitor.

    When a competitor's mean MIGD exceeds MIGD_CAP (i.e., it did not
    converge), improvement_pct saturates near 100%. Such bars are
    annotated as "capped" so readers do not mistake them for a meaningful
    numerical value.
    """
    valid = {a: c for a, c in comparisons.items() if "error" not in c}
    if not valid:
        return
    algos, imps, colors, sigs, capped = [], [], [], [], []
    for a, c in valid.items():
        algos.append(c["algorithm_name"])
        imps.append(c["improvement_pct"])
        colors.append(STAT_COLORS.get(a, "#888888"))
        sigs.append(c["significant"])
        capped.append(c["mean_competitor"] >= MIGD_CAP * 0.9)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(algos, imps, color=colors, edgecolor="#333333", linewidth=1.2, alpha=0.85)
    for bar, imp, sig, cp in zip(bars, imps, sigs, capped):
        suffix = ""
        if cp:
            suffix += " (capped)"
        if sig:
            suffix += " *"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{imp:.1f}%{suffix}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Competitor Algorithms", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIGD Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"MOEA/D-RV vs Competitors: MIGD Improvement{title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    legend = [Patch(facecolor="none", edgecolor="black",
                    label="* = significant (p < 0.05); capped = competitor did not converge")]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot_comparison(results, save_path=None, title_suffix=""):
    """Boxplot of MIGD distributions across algorithms (Matplotlib ≥ 3.9 compatible)."""
    algorithms = sorted(results[list(results.keys())[0]].keys())
    data, labels, colors = [], [], []
    for a in algorithms:
        vals = []
        for p in results.keys():
            if a in results[p]:
                v = results[p][a].get("migd_mean")
                if v is not None and not np.isnan(v) and v != float("inf"):
                    vals.append(v)
        if vals:
            data.append(vals)
            labels.append(ALGORITHM_NAMES.get(a, a))
            colors.append(STAT_COLORS.get(a, "#888888"))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.7)

    # Matplotlib >= 3.9 replaced the 'labels' kwarg with set_xticks/set_xticklabels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_xlabel("Algorithms", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIGD (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(f"MIGD Distribution Comparison{title_suffix}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_improvements(comparisons, save_path=None):
    valid = {a: c for a, c in comparisons.items() if "error" not in c}
    if not valid:
        return
    items = sorted(valid.items(), key=lambda x: x[1]["improvement_pct"], reverse=True)
    algos, imps, colors, sigs = [], [], [], []
    for a, c in items:
        algos.append(c["algorithm_name"])
        imps.append(c["improvement_pct"])
        colors.append(STAT_COLORS.get(a, "#888888"))
        sigs.append(c["significant"])
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algos, imps, color=colors, edgecolor="#333333", linewidth=1.2, alpha=0.85)
    for bar, imp, sig in zip(bars, imps, sigs):
        label = f"{imp:.1f}%" + (" *" if sig else "")
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.5 if imp >= 0 else -1.5),
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Ablation Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIGD Improvement from Full (%)", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Study: Component Contributions", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    legend = [Patch(facecolor="none", edgecolor="black", label="* = significant (p < 0.05)")]
    ax.legend(handles=legend, loc="upper right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_statistical_report(results, name="Standard Benchmark"):
    print("\n" + "=" * 100)
    print(f"STATISTICAL ANALYSIS REPORT: {name}")
    print("=" * 100)
    algorithms = list(results[list(results.keys())[0]].keys())
    common = get_common_problems(results, algorithms)
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 60)
    print(f"  Problems (total): {len(results)}")
    print(f"  Problems (common valid): {len(common)}")
    print(f"  Algorithms: {len(algorithms)}")
    for a in algorithms:
        print(f"    - {ALGORITHM_NAMES.get(a, a)}")

    print(f"\n2. FRIEDMAN TEST")
    print("-" * 60)
    fs, fp, _ = friedman_test_analysis(results)
    if fs is not None:
        print(f"  chi^2 = {fs:.4f}")
        print(f"  p-value = {fp:.4e}")
        print(f"  Significant (alpha=0.05): {'YES' if fp < SIGNIFICANCE_LEVEL else 'NO'}")
    else:
        print("  Insufficient data")

    print(f"\n3. FRIEDMAN RANKS (lower is better)")
    print("-" * 60)
    ranks = compute_friedman_ranks(results)
    for a, r in sorted(ranks.items(), key=lambda x: x[1] if x[1] is not None else 999):
        if r is not None:
            print(f"  {ALGORITHM_NAMES.get(a, a):<20}: {r:.2f}")

    print(f"\n4. WIN/LOSS/TIE COUNTS (vs MOEA/D-RV)")
    print("-" * 60)
    wlt = compute_win_loss_counts(results)
    for a, c in wlt.items():
        print(f"  {c['algorithm_name']:<20}: {c['wins']}W / {c['losses']}L / {c['ties']}T")

    print(f"\n5. WILCOXON PAIRWISE TESTS")
    print("-" * 60)
    comparisons = wilcoxon_pairwise_analysis(results)
    if comparisons:
        print(f"{'Competitor':<18} {'W-stat':<10} {'p-value':<12} {'Sig':<5} {'Better':<15} {'Improv':<10}")
        print("-" * 80)
        for a, c in comparisons.items():
            if "error" not in c:
                mark = "yes" if c["significant"] else "no"
                print(f"{c['algorithm_name']:<18} {c['statistic']:<10.1f} {c['p_value']:<12.3e} "
                      f"{mark:<5} {c['better']:<15} {c['improvement_pct']:+.1f}%")

    print(f"\n6. EFFECT SIZES (Cohen's d)")
    print("-" * 60)
    es = compute_effect_size(comparisons)
    for a, e in es.items():
        print(f"  {e['algorithm_name']:<20}: d = {e['effect_size']:+.2f} ({e['interpretation']})")

    pvals = [c["p_value"] for c in comparisons.values() if "p_value" in c]
    if pvals:
        print(f"\n7. HOLM-BONFERRONI CORRECTION")
        print("-" * 60)
        rej = holm_bonferroni(pvals)
        for i, (a, c) in enumerate([(a, c) for a, c in comparisons.items() if "p_value" in c]):
            status = "REJECTED" if rej[i] else "not rejected"
            print(f"  {c['algorithm_name']:<20}: p={c['p_value']:.3e} -> {status}")
    print("\n" + "=" * 100)


def print_ablation_report(results, name="Ablation Study"):
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY STATISTICAL REPORT: {name}")
    print("=" * 100)
    algorithms = list(results[list(results.keys())[0]].keys())
    common = get_common_problems(results, algorithms)
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 60)
    print(f"  Problems (common valid): {len(common)}")
    for a in algorithms:
        print(f"    - {ALGORITHM_NAMES.get(a, a)}")

    print(f"\n2. FRIEDMAN TEST")
    print("-" * 60)
    fs, fp, _ = friedman_test_analysis(results)
    if fs is not None:
        print(f"  chi^2 = {fs:.4f}, p = {fp:.4e}")
        print(f"  Significant: {'YES' if fp < SIGNIFICANCE_LEVEL else 'NO'}")

    print(f"\n3. FRIEDMAN RANKS")
    print("-" * 60)
    ranks = compute_friedman_ranks(results)
    for a, r in sorted(ranks.items(), key=lambda x: x[1] if x[1] is not None else 999):
        if r is not None:
            print(f"  {ALGORITHM_NAMES.get(a, a):<20}: {r:.2f}")

    print(f"\n4. WILCOXON ABLATION TESTS (Full vs Variants)")
    print("-" * 60)
    comps = wilcoxon_ablation_analysis(results)
    if comps:
        print(f"{'Variant':<20} {'p-value':<12} {'Sig':<5} {'Better':<15} {'Improv':<10}")
        print("-" * 70)
        for a, c in comps.items():
            if "error" not in c:
                mark = "yes" if c["significant"] else "no"
                print(f"{c['algorithm_name']:<20} {c['p_value']:<12.3e} {mark:<5} "
                      f"{c['better']:<15} {c['improvement_pct']:+.1f}%")
    print("\n" + "=" * 100)


def save_summary(results, out_dir, name, dtype="standard"):
    os.makedirs(out_dir, exist_ok=True)
    summary = {"dataset": name, "type": dtype, "n_problems": len(results),
               "algorithms": list(results[list(results.keys())[0]].keys())}

    fs, fp, _ = friedman_test_analysis(results)
    if fs is not None:
        summary["friedman_test"] = {"statistic": fs, "p_value": fp}

    ranks = compute_friedman_ranks(results)
    summary["friedman_ranks"] = {a: r for a, r in ranks.items() if r is not None}

    comps = wilcoxon_ablation_analysis(results) if dtype == "ablation" else wilcoxon_pairwise_analysis(results)
    summary["wilcoxon_comparisons"] = {}
    for a, c in comps.items():
        if "error" not in c:
            summary["wilcoxon_comparisons"][c["algorithm_name"]] = {
                "p_value": c["p_value"],
                "significant": c["significant"],
                "improvement_pct": c["improvement_pct"],
            }

    wlt = compute_win_loss_counts(results)
    summary["win_loss_counts"] = {c["algorithm_name"]: {"wins": c["wins"],
                                                        "losses": c["losses"],
                                                        "ties": c["ties"]} for c in wlt.values()}

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("MOEA/D-RV — SET 5: STATISTICAL ANALYSIS")
    print("=" * 100)

    print("\nDiscovering input files:")
    files = discover_results_files()
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"MIGD cap: {MIGD_CAP:.0e}")
    print(f"Significance level: {SIGNIFICANCE_LEVEL}")

    print("\n" + "=" * 100)
    print("LOADING RESULTS")
    print("=" * 100)
    s1 = load_set1_results(files["set1"])
    s2 = load_set2_results(files["set2"])
    s4 = load_set4_results(files["set4"])
    print(f"\n  Set 1: {'ok' if s1 else 'missing'} ({len(s1)} problems)")
    print(f"  Set 2: {'ok' if s2 else 'missing'} ({len(s2)} problems)")
    print(f"  Set 4: {'ok' if s4 else 'missing'} ({len(s4)} problems)")

    # ---- SET 1 ----
    if s1:
        print("\n" + "=" * 100)
        print("ANALYZING SET 1: STANDARD BENCHMARK")
        print("=" * 100)
        out = OUTPUT_DIR / "set1"
        plots = out / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        print_statistical_report(s1, "Set 1: Standard Benchmark")
        ranks = compute_friedman_ranks(s1)
        if ranks:
            plot_friedman_ranks(ranks, plots / "friedman_ranks.png")
            plot_critical_difference_diagram(ranks, plots / "critical_difference.png")
        plot_p_value_heatmap(s1, plots / "p_value_heatmap.png")
        plot_wilcoxon_improvements(wilcoxon_pairwise_analysis(s1),
                                    plots / "wilcoxon_improvements.png")
        plot_boxplot_comparison(s1, plots / "boxplot.png")
        save_summary(s1, str(out), "Set 1: Standard Benchmark")
        print(f"\n  Set 1 analysis saved to: {out}")

    # ---- SET 2 ----
    if s2:
        print("\n" + "=" * 100)
        print("ANALYZING SET 2: ENVIRONMENT CONFIGURATIONS")
        print("=" * 100)
        out = OUTPUT_DIR / "set2"
        plots = out / "plots"
        plots.mkdir(parents=True, exist_ok=True)

        for config in CONFIG_NAMES:
            print(f"\n--- Configuration: {config} ---")
            sub = {}
            for p, algos in s2.items():
                sub[p] = {}
                for key, metrics in algos.items():
                    if key.endswith(f"_{config}"):
                        algo = key[:-len(f"_{config}")]
                        sub[p][algo] = metrics
            if sub:
                print_statistical_report(sub, f"Set 2: {config} Configuration")
                ranks = compute_friedman_ranks(sub)
                if ranks:
                    plot_friedman_ranks(ranks,
                                        plots / f"friedman_ranks_{config}.png",
                                        title_suffix=f" ({config})")
                plot_boxplot_comparison(sub,
                                        plots / f"boxplot_{config}.png",
                                        title_suffix=f" ({config})")
                save_summary(sub, str(out / config), f"Set 2: {config}")
        print(f"\n  Set 2 analysis saved to: {out}")

    # ---- SET 4 ----
    if s4:
        print("\n" + "=" * 100)
        print("ANALYZING SET 4: ABLATION STUDY")
        print("=" * 100)
        out = OUTPUT_DIR / "set4"
        plots = out / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        print_ablation_report(s4, "Set 4: Ablation Study")
        ranks = compute_friedman_ranks(s4)
        if ranks:
            plot_friedman_ranks(ranks, plots / "friedman_ranks.png",
                                title_suffix=" (Ablation)")
            plot_critical_difference_diagram(ranks, plots / "critical_difference.png")
        plot_ablation_improvements(wilcoxon_ablation_analysis(s4),
                                    plots / "ablation_improvements.png")
        plot_boxplot_comparison(s4, plots / "boxplot.png",
                                title_suffix=" (Ablation)")
        save_summary(s4, str(out), "Set 4: Ablation Study", "ablation")
        print(f"\n  Set 4 analysis saved to: {out}")

    # ---- Combined ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = {
        "sets": ["set1", "set2", "set4"],
        "migd_cap": MIGD_CAP,
        "significance_level": SIGNIFICANCE_LEVEL,
        "input_files": {k: str(v) if v is not None else None for k, v in files.items()},
    }
    with open(OUTPUT_DIR / "combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"All outputs: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
