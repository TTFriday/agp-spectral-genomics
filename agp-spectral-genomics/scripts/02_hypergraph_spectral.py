# -*- coding: utf-8 -*-
"""
02_hypergraph_spectral.py
构建 AGP 超图 Laplacian, 对 24 个样本的中心化基因表达做谱分解,
绘制能量谱并验证稀疏性/平滑性假设。

映射结论 (由 03-07 号脚本确证):
    超图基因节点 id == gene_sequences idx 值, 经 node2row.npz 映射到表达矩阵行
    超边 = GO 词条 (gene2go 全证据注释, GO 索引按 GO ID 排序)
    注释来源版本与作者(2024/2025)存在漂移, 但超图结构本身直接取自原始数据

方法 (Zhou 2006 / Feng 2019 对称归一化):
    H ∈ {0,1}^(28679 x 7079)   基因-超边关联
    A = Dv^{-1/2} H De^{-1/2}, B = A A^T (平滑算子), Δ = I - B
    对偶分解: B 的非零特征值 = A^T A (7079x7079) 的特征值 λ
    u_i = A v_i / sqrt(λ_i),  样本投影 x̂_i = v_i^T (A^T x) / sqrt(λ_i)
    信号先在超图覆盖基因上中心化, 消除直流分量 (log1p FPKM 非负均值大,
    不中心化则能量全部集中在最平滑分量, 无法反映结构)

输入:
    processed/aligned_data.npz, processed/node2row.npz
    AGP/data/hypergraph/hyperedge_list.npy
输出:
    processed/hypergraph_spectrum.npz (λ, V, 频带定义)
    results/fig_hypergraph_stats.png, fig_eigenvalues.png, fig_energy_spectrum.png
    results/spectral_summary.csv
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
N_PERM = 10
SEED = 2026
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- 超图
def build_H():
    """hyperedge_list + node2row → 表达行空间 (38286 x 7079) 的关联矩阵。"""
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    map_ = np.load(os.path.join(PROCESSED, "node2row.npz"))
    node2row = dict(zip(map_["node_ids"].tolist(), map_["row_of_node"].tolist()))
    n_genes, n_go = 38286, int(pairs[1].max()) + 1

    rows = np.array([node2row[g] for g in pairs[0].tolist()])
    cols = pairs[1]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(n_genes, n_go)).tocsr()
    H.sum_duplicates()
    H.data[:] = 1.0
    print(f"H: {H.shape}, 非零元 {H.nnz}")
    return H


def hypergraph_stats(H):
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    annotated = d_v > 0
    print(f"超图覆盖基因: {annotated.sum()} / {len(d_v)} ({100*annotated.mean():.1f}%)")
    print(f"基因度 d_v: 中位 {np.median(d_v[annotated]):.0f}, 最大 {d_v.max()}")
    print(f"超边大小 d_e: 中位 {np.median(d_e):.0f}, 最大 {d_e.max()}")
    return d_v, d_e, annotated


# ---------------------------------------------------------------- 谱分解
def dual_eigendecomposition(H):
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    inv_dv2 = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    inv_de2 = 1.0 / np.sqrt(np.maximum(d_e, 1))
    A = (sp.diags(inv_dv2) @ H @ sp.diags(inv_de2)).tocsr()
    S = (sp.diags(inv_de2) @ (H.T @ sp.diags(inv_dv2 ** 2)) @ H
         @ sp.diags(inv_de2)).tocsc()
    S = ((S + S.T) * 0.5).toarray()
    print(f"对偶矩阵 {S.shape}, eigh (evd) 分解中...")
    lam, V = la.eigh(S, driver="evd")
    order = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[order], 0.0), V[:, order]
    print(f"λ 范围 [{lam.min():.6f}, {lam.max():.6f}], "
          f"有效秩(λ>1e-10) = {(lam > 1e-10).sum()}")
    return A, lam, V


# ---------------------------------------------------------------- 投影
def project_sample(A, lam, V, x):
    """x: 已中心化的全基因向量 → (谱系数², 谱外能量占比)"""
    z = A.T @ x
    ok = lam > 1e-10
    xhat = np.zeros_like(lam)
    xhat[ok] = (V[:, ok].T @ z) / np.sqrt(lam[ok])
    e = xhat ** 2
    tot = float(x @ x)
    outside = max(tot - float(e.sum()), 0.0)
    return e, outside / tot if tot > 0 else np.nan


def band_stats(e, lam, ok):
    tot = e.sum()
    if tot <= 0:
        return dict(smoothness=np.nan, pr=np.nan, gini=np.nan,
                    top10=np.nan, top50=np.nan, top100=np.nan,
                    top1pct=np.nan)
    smoothness = float((e * lam).sum() / tot / lam.max())     # ∈[0,1] 越大越平滑
    pr = float(tot ** 2 / (e ** 2).sum())                     # 有效分量数
    srt = np.sort(e[ok])
    n = len(srt)
    cum = np.cumsum(srt)
    gini = float(1.0 - 2.0 * (cum.sum() - 0.5 * tot) / (n * tot))
    desc = np.cumsum(np.sort(e[ok])[::-1]) / tot
    return dict(smoothness=smoothness, pr=pr, gini=gini,
                top10=float(desc[9]), top50=float(desc[49]),
                top100=float(desc[99]),
                top1pct=float(desc[int(0.01 * n) - 1]))


def main():
    os.makedirs(RESULTS, exist_ok=True)

    H = build_H()
    d_v, d_e, annotated = hypergraph_stats(H)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    bins_v = np.arange(0, d_v[annotated].max() + 1.5, 1.0)
    ax.hist(d_v[annotated], bins=bins_v, color="#2b6cb0", alpha=0.85)
    ax.set_xlabel("GO terms per gene ($d_v$)")
    ax.set_ylabel("# genes"); ax.set_yscale("log")
    ax.set_xlim(0, d_v[annotated].max() + 1)
    ax.set_title("Gene degree distribution")
    ax = axes[1]
    # 小尺寸逐整数分箱, 大尺寸对数分箱: 避免等宽 log 分箱在小尺寸处留下空白
    bins_e = np.unique(np.concatenate([
        np.arange(1, 16),
        np.logspace(np.log10(16), np.log10(d_e.max() * 1.05), 40)]))
    ax.hist(d_e, bins=bins_e, color="#c05621", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Genes per GO term ($d_e$, log scale)")
    ax.set_ylabel("# GO terms"); ax.set_yscale("log")
    ax.set_title("Hyperedge size distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_hypergraph_stats.png"), dpi=400)
    plt.close(fig)

    A, lam, V = dual_eigendecomposition(H)
    ok = lam > 1e-10

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(np.arange(1, len(lam) + 1), lam, lw=0.8, color="#2b6cb0")
    ax.axvline(ok.sum(), color="gray", lw=0.8, ls=":")
    ax.text(ok.sum() * 0.97, 0.55, f"rank {int(ok.sum()):,}",
            ha="right", va="center", fontsize=8, color="gray", rotation=90)
    ax.set_xlabel("component index (smooth → oscillatory)")
    ax.set_ylabel(r"$\lambda$ of smoothing operator $AA^T$")
    ax.set_title("Hypergraph spectrum")
    ax = axes[1]
    ax.hist(lam, bins=80, color="#2b6cb0", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("# components")
    ax.set_title("Eigenvalue histogram")
    ax.annotate(f"{int((~ok).sum()):,} null modes", xy=(0.006, 1000),
                xytext=(0.12, 700), fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_eigenvalues.png"), dpi=400)
    plt.close(fig)

    np.savez(os.path.join(PROCESSED, "hypergraph_spectrum.npz"),
             lam=lam, V=V.astype(np.float32), annotated=annotated)

    # ---------- 样本投影 (中心化!) ----------
    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X = data["X_log1p"]
    samples = data["samples"]
    n_samp = X.shape[0]

    ann_pos = np.where(annotated)[0]
    spec_real, frac_out_real = np.zeros((n_samp, len(lam))), np.zeros(n_samp)
    perm_e_stats, perm_smooth = [], []
    for i in range(n_samp):
        x = X[i].copy()
        x[ann_pos] -= x[ann_pos].mean()          # 在超图覆盖基因上中心化
        e, fo = project_sample(A, lam, V, x)
        spec_real[i], frac_out_real[i] = e, fo
        st = band_stats(e, lam, ok)
        for _ in range(N_PERM):
            xp = x.copy()
            xp[ann_pos] = rng.permutation(x[ann_pos])   # 打乱基因-功能对应
            ep, _ = project_sample(A, lam, V, xp)
            perm_e_stats.append(band_stats(ep, lam, ok))
            perm_smooth.append(perm_e_stats[-1]["smoothness"])
        print(f"  {samples[i]}: 平滑度={st['smoothness']:.4f}, "
              f"top50={st['top50']:.3f}, 谱外={100*fo:.1f}%")

    spec_mean, spec_std = spec_real.mean(0), spec_real.std(0)
    perm_df = pd.DataFrame(perm_e_stats)

    # ---------- 汇总 ----------
    rows = []
    for i in range(n_samp):
        st = band_stats(spec_real[i], lam, ok)
        st.update(sample=samples[i], genotype=data["genotypes"][i],
                  frac_outside=frac_out_real[i])
        rows.append(st)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS, "spectral_summary.csv"), index=False)
    print("\n真实样本谱统计:")
    print(summary[["sample", "genotype", "smoothness", "pr", "gini",
                   "top10", "top50", "top100", "frac_outside"]]
          .describe().loc[["mean", "std"]].to_string())

    print("\n置换对照 (均值±std):")
    print(perm_df[["smoothness", "pr", "gini", "top10", "top50"]]
          .agg(["mean", "std"]).to_string())
    print(f"\n平滑度对比: 真实 {summary['smoothness'].mean():.4f} vs "
          f"置换 {perm_df['smoothness'].mean():.4f} "
          f"(效应 = {summary['smoothness'].mean() / perm_df['smoothness'].mean():.2f}x)")
    print(f"谱外能量: 真实 {100*summary['frac_outside'].mean():.1f}% "
          f"vs 置换 {100*perm_df['frac_outside'].mean() if 'frac_outside' in perm_df else float('nan'):.1f}%")

    # ---------- 能量谱图 ----------
    freq = 1.0 - lam
    theta_peak = float(freq[int(np.argmax(spec_mean))])

    # 置换的平均能量谱 (重新投影)
    perm_spec = np.zeros((len(perm_e_stats), len(lam)))
    k = 0
    for i in range(n_samp):
        x = X[i].copy(); x[ann_pos] -= x[ann_pos].mean()
        for _ in range(N_PERM):
            xp = x.copy(); xp[ann_pos] = rng.permutation(xp[ann_pos])
            ep, _ = project_sample(A, lam, V, xp)
            perm_spec[k] = ep; k += 1
    perm_spec_mean = perm_spec.mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    ax.scatter(freq, spec_mean, s=3, color="#2b6cb0", alpha=0.45,
               linewidths=0, label="real (mean over samples)")
    ax.scatter(freq, perm_spec_mean, s=2, color="#c05621", alpha=0.55,
               linewidths=0, label=f"gene-shuffled (n={N_PERM}/sample)")
    ax.axvline(theta_peak, color="gray", lw=0.8, ls=":")
    ax.annotate(f"peak at $\\theta$ ≈ {theta_peak:.2f}",
                xy=(theta_peak, spec_mean.max()),
                xytext=(theta_peak + 0.12, spec_mean.max() * 0.82),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                fontsize=8)
    ax.set_xlabel(r"hypergraph frequency $\theta = 1-\lambda$")
    ax.set_ylabel("spectral energy")
    ax.set_title("Centered expression energy on GO hypergraph")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    csum = np.cumsum(spec_mean) / spec_mean.sum()      # 平滑→振荡 累积
    csum_perm = np.cumsum(perm_spec_mean) / perm_spec_mean.sum()
    kk = np.arange(1, len(lam) + 1)
    ax.plot(kk, csum, color="#2b6cb0", lw=1.2, label="real")
    ax.plot(kk, csum_perm, color="#c05621", lw=1.2, ls="--", label="shuffled")
    ax.axhline(0.9, color="gray", lw=0.8, ls=":")
    k90 = int(np.searchsorted(csum, 0.9)) + 1
    ax.axvline(k90, color="gray", lw=0.8, ls=":")
    ax.annotate(f"90% energy: {k90:,} components",
                xy=(k90, 0.9), xytext=(k90 * 1.45, 0.55),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                fontsize=8)
    ax.axvline(ok.sum(), color="0.8", lw=0.8, ls=":")
    ax.text(ok.sum() * 0.96, 0.25, f"rank {int(ok.sum()):,}",
            ha="right", va="center", fontsize=7, color="gray", rotation=90)
    ax.set_xlabel("components ordered from smoothest to most oscillatory")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Cumulative energy, smooth first")
    ax.set_xlim(1, len(lam))
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_energy_spectrum.png"), dpi=400)
    plt.close(fig)
    print(f"\n90% 能量所需最平滑分量数: {k90} / {len(lam)}")


if __name__ == "__main__":
    main()
