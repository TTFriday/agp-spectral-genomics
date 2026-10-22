# -*- coding: utf-8 -*-
"""
16_make_honest_figures.py

根据 12/13/14 号脚本的输出绘制两张图:
    fig_smoothness_nulls.png  真实 GO 超图 / 大小分布匹配的随机超图 / 基因标签置换零假设
    fig_prediction_skill.png  各特征表示在尺度无关 skill 指标下的表型预测结果

skill = 1 - MAE / MAE_mean_predictor, 0 表示与"恒预测训练折均值"持平。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(BASE, "results")

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

PH_ORDER = ["solidity", "inflor_height", "FvFm", "qL", "leaf_temp_diff"]

REP_ORDER = [
    ("HSG_38d_bandstats", "HSG 38-d spectral features"),
    ("HSG_bands_only", "Band energies only (10d)"),
    ("HGsmooth_coef_K5", "Smoothest 5 coefficients"),
    ("HGsmooth_coef_K10", "Smoothest 10 coefficients"),
    ("HGsmooth_coef_K20", "Smoothest 20 coefficients"),
    ("HGsmooth_coef_K50", "Smoothest 50 coefficients"),
    ("HGlowpass_full_pow1", "Full signed spectrum (× λ)"),
    ("HGlowpass_full_pow2", "Full signed spectrum (× λ²)"),
    ("HGhighfreq_coef_K10", "Most oscillatory 10 (control)"),
    ("HGhighfreq_coef_K50", "Most oscillatory 50 (control)"),
    ("raw_expression", "Raw expression (control)"),
    ("PCA_5d", "PCA 5 components (control)"),
    ("PCA_10d", "PCA 10 components (control)"),
    ("PCA_20d", "PCA 20 components (control)"),
    ("random_38d", "Random features (control)"),
]


def fig_smoothness():
    s = pd.read_csv(os.path.join(RESULTS, "smoothness_test.csv"))
    ab = pd.read_csv(os.path.join(RESULTS, "ablation_honest.csv"))
    sm = ab.groupby("hypergraph")["smoothness"].first()
    real_sm = sm[sm.index == "real"].iloc[0]
    rand_sm = sm[sm.index != "real"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    ax.axhspan(s["perm_mean"].min() * 0.98, s["perm_max"].max() * 1.02,
               color="0.92", zorder=0, label="gene-permutation range")
    x = np.arange(len(s))
    ax.plot(x, s["perm_mean"], "o", ms=3.5, color="tab:orange", label="permutation mean")
    ax.plot(x, s["perm_max"], "v", ms=3.5, color="tab:red", label="permutation max")
    ax.plot(x, s["real"], "o", ms=5, color="tab:blue", label="real signal")
    ax.set_xticks(x)
    ax.set_xticklabels(s["sample"], rotation=90, fontsize=6)
    ax.set_ylabel("spectral smoothness")
    ax.set_xlabel("sample")
    ax.set_ylim(0.135, 0.49)
    ax.legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.5, 0.54),
              ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax.set_title(f"All {int(s['exceed_max'].sum())}/{len(s)} samples above\n"
                 f"their own permutation maximum", fontsize=9)

    ax = axes[1]
    ax.bar([0], [real_sm], width=0.55, color="tab:blue", label="real GO hypergraph")
    ax.bar(np.arange(1, len(rand_sm) + 1), rand_sm.values, width=0.55,
           color="tab:gray", label="size-matched random")
    ax.axhline(s["perm_mean"].mean(), ls="--", lw=1, color="tab:orange",
               label="gene-permutation mean")
    ax.set_xticks(np.arange(len(rand_sm) + 1))
    ax.set_xticklabels(["real"] + [f"R{i}" for i in range(1, len(rand_sm) + 1)],
                       fontsize=7)
    ax.set_ylabel("mean spectral smoothness")
    ax.set_ylim(0, 0.56)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("Real GO exceeds size-matched random", fontsize=9)

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_smoothness_nulls.png")
    fig.savefig(out, bbox_inches="tight", dpi=400)
    plt.close(fig)
    return out


def fig_skill():
    d = pd.read_csv(os.path.join(RESULTS, "spectral_representations.csv"))
    d = d[d.model != "mean_predictor"]
    piv = d.pivot_table(index="model", columns="phenotype", values="skill")
    means = d.groupby("model")["skill"].mean()

    labels = [lbl for key, lbl in REP_ORDER if key in piv.index]
    rows = [key for key, lbl in REP_ORDER if key in piv.index]
    M = piv.loc[rows, [c for c in PH_ORDER if c in piv.columns]].values

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([c.replace("_", "\n") for c in PH_ORDER if c in piv.columns],
                       fontsize=8.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7.0,
                        color="k" if abs(M[i, j]) < vmax * 0.55 else "w")
    ax.set_title("skill vs. mean predictor\n(>0 beats constant prediction)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax = axes[1]
    m = means.loc[rows].values
    order = np.argsort(m)
    ax.barh(np.arange(len(rows)), m[order],
            color=["tab:green" if v > 0 else "tab:red" for v in m[order]], height=0.65)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=8.0)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean skill across phenotypes")
    ax.set_title("Representation ranking", fontsize=9)

    fig.tight_layout()
    out = os.path.join(RESULTS, "fig_prediction_skill.png")
    fig.savefig(out, bbox_inches="tight", dpi=400)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_smoothness())
    print(fig_skill())
