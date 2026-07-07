#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting script for Experiment 3 (paper-ready main figures)

This script reads exp3 summary files produced by the diagnostics pipeline and
generates three publication-oriented figures:

Figure 1: Scale-dependent gating behavior
Figure 2: Relative expert contribution across scales
Figure 3: Rotational contribution and dissipation consistency across scales

Expected input files:
    RESULT_ROOT/
        exp3_all_scales_summary.json
    or
        RESULT_ROOT/n3/exp3_summary_n3.json
        RESULT_ROOT/n5/exp3_summary_n5.json
        RESULT_ROOT/n10/exp3_summary_n10.json

Recommended:
- keep this plotting script separate from the exp3 diagnostics script
- run diagnostics first, then run this script only for figure production
"""

import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================
RESULT_ROOT = "/home/Data/zhufuhua/MyData/para2/data/global/theresult_exp3"
OUT_DIR = os.path.join(RESULT_ROOT, "paper_figures")

SCALES = [3, 5, 10]

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# IO
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(fig, path: str):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved:", path)


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_summary() -> Dict[int, Dict]:
    merged_path = os.path.join(RESULT_ROOT, "exp3_all_scales_summary.json")
    if os.path.exists(merged_path):
        with open(merged_path, "r") as f:
            raw = json.load(f)
        out = {}
        for k, v in raw.items():
            out[int(k)] = v
        return out

    out = {}
    for n in SCALES:
        p = os.path.join(RESULT_ROOT, f"n{n}", f"exp3_summary_n{n}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing summary file: {p}")
        with open(p, "r") as f:
            out[n] = json.load(f)
    return out


# ============================================================
# Summary completion helpers
# ============================================================
def add_branch_fraction_if_missing(summary: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    If prior/structured/residual fractions are already present, keep them.
    Otherwise estimate them from mean gates as a fallback.

    Recommended: compute these directly in the diagnostics script later.
    This fallback is only for plotting when exact fractions are absent.
    """
    for n, d in summary.items():
        has_all = all(k in d for k in [
            "prior_frac_mean",
            "struct_frac_mean",
            "res_frac_mean",
        ])
        if has_all:
            continue

        alphaT = _safe_float(d.get("alphaT_mean"))
        alphaS = _safe_float(d.get("alphaS_mean"))
        betaT  = _safe_float(d.get("betaT_mean"))
        betaS  = _safe_float(d.get("betaS_mean"))

        prior_proxy = np.nanmean([alphaT, alphaS])
        struct_proxy = np.nanmean([betaT, betaS])

        # residual does not have explicit gate, so keep a conservative residual proxy.
        # This is only a visualization fallback, not a final physical statistic.
        residual_proxy = max(0.0, 1.0 - 0.5 * (prior_proxy + struct_proxy))

        s = prior_proxy + struct_proxy + residual_proxy + 1e-12
        d["prior_frac_mean"] = float(prior_proxy / s)
        d["struct_frac_mean"] = float(struct_proxy / s)
        d["res_frac_mean"] = float(residual_proxy / s)

    return summary


def add_rot_ratio_if_missing(summary: Dict[int, Dict]) -> Dict[int, Dict]:
    for n, d in summary.items():
        if "rot_ratio_T_mean" not in d:
            d["rot_ratio_T_mean"] = np.nan
        if "rot_ratio_S_mean" not in d:
            d["rot_ratio_S_mean"] = np.nan
    return summary


def add_dissipation_fraction_if_missing(summary: Dict[int, Dict]) -> Dict[int, Dict]:
    for n, d in summary.items():
        if "PiT_diss_positive_frac" not in d:
            d["PiT_diss_positive_frac"] = np.nan
        if "PiS_diss_positive_frac" not in d:
            d["PiS_diss_positive_frac"] = np.nan
    return summary


# ============================================================
# Plot helpers
# ============================================================
def panel_style(ax):
    ax.set_xticks(SCALES)
    ax.set_xlim(min(SCALES) - 0.4, max(SCALES) + 0.4)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.grid(True, axis="both")


def annotate_points(ax, x, y, dy=0.008, fmt="{:.2f}", fontsize=9):
    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            ax.text(xi, yi + dy, fmt.format(yi), ha="center", va="bottom", fontsize=fontsize)


# ============================================================
# Figure 1
# Scale-dependent gating behavior
# ============================================================
def plot_gate_behavior(summary: Dict[int, Dict], out_dir: str):
    x = SCALES
    alphaT = [_safe_float(summary[n].get("alphaT_mean")) for n in x]
    alphaS = [_safe_float(summary[n].get("alphaS_mean")) for n in x]
    betaT  = [_safe_float(summary[n].get("betaT_mean"))  for n in x]
    betaS  = [_safe_float(summary[n].get("betaS_mean"))  for n in x]

    fig, ax = plt.subplots(figsize=(7.2, 4.7))

    ax.plot(x, alphaT, marker="o", label=r"$\alpha_T$ (prior, temperature)")
    ax.plot(x, alphaS, marker="o", label=r"$\alpha_S$ (prior, salinity)")
    ax.plot(x, betaT,  marker="s", label=r"$\beta_T$ (structured, temperature)")
    ax.plot(x, betaS,  marker="s", label=r"$\beta_S$ (structured, salinity)")

    panel_style(ax)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("Coarse-graining factor $n$")
    ax.set_ylabel("Mean gate value")
    ax.set_title("Scale-dependent gating behavior")
    ax.legend(frameon=True, ncol=2, loc="upper left")

    annotate_points(ax, x, alphaT, dy=0.012)
    annotate_points(ax, x, alphaS, dy=0.012)
    annotate_points(ax, x, betaT, dy=0.012)
    annotate_points(ax, x, betaS, dy=0.012)

    savefig(fig, os.path.join(out_dir, "exp3_gate_vs_scale_main.png"))


# ============================================================
# Figure 2
# Relative expert contribution across scales
# ============================================================
def plot_branch_contribution(summary: Dict[int, Dict], out_dir: str):
    x = np.arange(len(SCALES))
    prior = np.array([_safe_float(summary[n].get("prior_frac_mean")) for n in SCALES], dtype=float)
    struct = np.array([_safe_float(summary[n].get("struct_frac_mean")) for n in SCALES], dtype=float)
    res = np.array([_safe_float(summary[n].get("res_frac_mean")) for n in SCALES], dtype=float)

    # normalize again in case fallback or external stats do not sum to one exactly
    total = prior + struct + res + 1e-12
    prior = prior / total
    struct = struct / total
    res = res / total

    fig, ax = plt.subplots(figsize=(7.2, 4.9))

    width = 0.58
    ax.bar(x, prior, width=width, label="Prior contribution")
    ax.bar(x, struct, width=width, bottom=prior, label="Structured contribution")
    ax.bar(x, res, width=width, bottom=prior + struct, label="Residual contribution")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in SCALES])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Coarse-graining factor $n$")
    ax.set_ylabel("Relative contribution")
    ax.set_title("Relative expert contribution across scales")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.22)

    for i in range(len(SCALES)):
        vals = [prior[i], struct[i], res[i]]
        bottoms = [0.0, prior[i], prior[i] + struct[i]]
        for v, b in zip(vals, bottoms):
            if v > 0.07:
                ax.text(i, b + 0.5 * v, f"{v:.2f}", ha="center", va="center", fontsize=9)

    savefig(fig, os.path.join(out_dir, "exp3_branch_contribution_main.png"))


# ============================================================
# Figure 3
# Rotational contribution and dissipation consistency
# ============================================================
def plot_rotation_and_dissipation(summary: Dict[int, Dict], out_dir: str):
    x = SCALES
    rotT = [_safe_float(summary[n].get("rot_ratio_T_mean")) for n in x]
    rotS = [_safe_float(summary[n].get("rot_ratio_S_mean")) for n in x]

    posT = [_safe_float(summary[n].get("PiT_diss_positive_frac")) for n in x]
    posS = [_safe_float(summary[n].get("PiS_diss_positive_frac")) for n in x]

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    # left panel: rotational ratio
    axs[0].plot(x, rotT, marker="o", label="Temperature")
    axs[0].plot(x, rotS, marker="o", label="Salinity")
    panel_style(axs[0])
    ymin = np.nanmin(rotT + rotS) if np.any(np.isfinite(rotT + rotS)) else 0.0
    ymax = np.nanmax(rotT + rotS) if np.any(np.isfinite(rotT + rotS)) else 1.0
    pad = max(0.015, 0.25 * (ymax - ymin))
    axs[0].set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))
    axs[0].set_xlabel("Coarse-graining factor $n$")
    axs[0].set_ylabel(r"$|Q_{\mathrm{rot}}|/(|Q_{\mathrm{diss}}|+|Q_{\mathrm{rot}}|)$")
    axs[0].set_title("Rotational contribution across scales")
    axs[0].legend(frameon=True, loc="best")
    annotate_points(axs[0], x, rotT, dy=0.004)
    annotate_points(axs[0], x, rotS, dy=0.004)

    # right panel: dissipative consistency
    axs[1].plot(x, posT, marker="s", label="Temperature")
    axs[1].plot(x, posS, marker="s", label="Salinity")
    panel_style(axs[1])
    axs[1].set_ylim(0.0, 1.02)
    axs[1].set_xlabel("Coarse-graining factor $n$")
    axs[1].set_ylabel(r"Fraction with $-Q_{\mathrm{diss}}\!\cdot\nabla C > 0$")
    axs[1].set_title("Dissipative consistency across scales")
    axs[1].legend(frameon=True, loc="best")
    annotate_points(axs[1], x, posT, dy=0.015)
    annotate_points(axs[1], x, posS, dy=0.015)

    savefig(fig, os.path.join(out_dir, "exp3_rotation_dissipation_main.png"))


# ============================================================
# Optional text summary for manuscript drafting
# ============================================================
def write_brief_text_summary(summary: Dict[int, Dict], out_dir: str):
    lines: List[str] = []
    lines.append("Experiment 3 plotting summary")
    lines.append("")

    for n in SCALES:
        d = summary[n]
        lines.append(f"n={n}")
        lines.append(
            "  gates: "
            f"alphaT={_safe_float(d.get('alphaT_mean')):.4f}, "
            f"alphaS={_safe_float(d.get('alphaS_mean')):.4f}, "
            f"betaT={_safe_float(d.get('betaT_mean')):.4f}, "
            f"betaS={_safe_float(d.get('betaS_mean')):.4f}"
        )
        lines.append(
            "  rotation: "
            f"rotT={_safe_float(d.get('rot_ratio_T_mean')):.4f}, "
            f"rotS={_safe_float(d.get('rot_ratio_S_mean')):.4f}"
        )
        lines.append(
            "  dissipation: "
            f"posT={_safe_float(d.get('PiT_diss_positive_frac')):.4f}, "
            f"posS={_safe_float(d.get('PiS_diss_positive_frac')):.4f}"
        )
        lines.append(
            "  branch fraction: "
            f"prior={_safe_float(d.get('prior_frac_mean')):.4f}, "
            f"struct={_safe_float(d.get('struct_frac_mean')):.4f}, "
            f"res={_safe_float(d.get('res_frac_mean')):.4f}"
        )
        lines.append("")

    txt_path = os.path.join(out_dir, "exp3_plotting_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved:", txt_path)


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    summary = load_summary()
    summary = add_branch_fraction_if_missing(summary)
    summary = add_rot_ratio_if_missing(summary)
    summary = add_dissipation_fraction_if_missing(summary)

    plot_gate_behavior(summary, OUT_DIR)
    plot_branch_contribution(summary, OUT_DIR)
    plot_rotation_and_dissipation(summary, OUT_DIR)
    write_brief_text_summary(summary, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
