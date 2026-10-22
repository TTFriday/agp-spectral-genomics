# -*- coding: utf-8 -*-
"""
08_predict_band_energy.py
从超图谱系数中提取频带能量特征, 用 Ridge / LASSO + LOOCV 预测 5 个表型,
对比论文基线 (RF MAE=0.5768, HGNN+VAE=0.5530)。

特征提取:
    1. 频带能量 (B=10): 将 7079 个谱分量按 λ 等分为 B 段, 每段能量之和
    2. 谱系数 top-K (K=20): 平均能量最高的 K 个分量的谱系数
    3. 谱统计量: smoothness, PR, gini, top10/50/100, frac_outside (8 维)
    → 拼接为 38 维特征向量

预测:
    Ridge (α 由 LOOCV 在 logspace 选) + LOOCV, 对每个表型独立预测
    指标: MAE, 对比论文基线

消融:
    (a) 真实超图 vs 随机超边 (保持超边大小分布, 打乱基因归属)
    (b) 学习曲线: n_train = 5, 8, 12, 16, 20 → 50 次随机划分

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz,
    processed/node2row.npz, AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/prediction_results.csv, results/prediction_features.csv
    results/fig_prediction.png (MAE 对比柱状图)
    results/fig_learning_curve.png
    results/fig_ablation_random.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026
rng = np.random.default_rng(SEED)

N_BANDS = 10       # 频带数
TOP_K = 20         # top-K 谱系数
N_PERM_HYPER = 10  # 随机超图消融次数


# ================================================================ 超图 & 谱分解
def build_H():
    """构建基因-超边关联矩阵 H (38286 x 7079)"""
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    map_ = np.load(os.path.join(PROCESSED, "node2row.npz"))
    node2row = dict(zip(map_["node_ids"].tolist(), map_["row_of_node"].tolist()))
    n_genes = 38286
    rows = np.array([node2row[g] for g in pairs[0].tolist()])
    cols = pairs[1]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(n_genes, int(cols.max()) + 1)).tocsr()
    H.sum_duplicates()
    H.data[:] = 1.0
    return H


def build_random_H(H_real, seed=0):
    """保持超边大小分布不变, 随机打乱每条超边内的基因归属"""
    rng_local = np.random.default_rng(seed)
    H_coo = H_real.tocoo()
    d_e = np.asarray(H_real.sum(axis=0)).ravel().astype(int)
    n_genes = H_real.shape[0]
    rows_list, cols_list = [], []
    for e_idx in range(len(d_e)):
        size = d_e[e_idx]
        genes = rng_local.choice(n_genes, size=size, replace=False)
        rows_list.append(genes)
        cols_list.append(np.full(size, e_idx))
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    H_rand = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=H_real.shape).tocsr()
    H_rand.sum_duplicates()
    H_rand.data[:] = 1.0
    return H_rand


def spectral_decompose(H):
    """对偶空间谱分解 → A, λ, V"""
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    inv_dv2 = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    inv_de2 = 1.0 / np.sqrt(np.maximum(d_e, 1))
    A = (sp.diags(inv_dv2) @ H @ sp.diags(inv_de2)).tocsr()
    S = (sp.diags(inv_de2) @ (H.T @ sp.diags(inv_dv2 ** 2)) @ H
         @ sp.diags(inv_de2)).tocsc()
    S = ((S + S.T) * 0.5).toarray()
    lam, V = la.eigh(S, driver="evd")
    order = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[order], 0.0), V[:, order]
    return A, lam, V


def project_all(A, lam, V, X, annotated):
    """投影所有样本 → 谱系数矩阵 (n_samples x 7079)"""
    ann_pos = np.where(annotated)[0]
    ok = lam > 1e-10
    n_samp = X.shape[0]
    n_comp = len(lam)
    X_hat = np.zeros((n_samp, n_comp))
    for i in range(n_samp):
        x = X[i].copy()
        x[ann_pos] -= x[ann_pos].mean()
        z = A.T @ x
        X_hat[i, ok] = (V[:, ok].T @ z) / np.sqrt(lam[ok])
    return X_hat


# ================================================================ 特征提取
def extract_features(X_hat, lam):
    """
    从谱系数矩阵提取特征:
        1. 频带能量 (N_BANDS 维)
        2. top-K 谱系数 (TOP_K 维)
        3. 谱统计量 (8 维)
    → (n_samples, N_BANDS + TOP_K + 8)
    """
    n_samp, n_comp = X_hat.shape
    ok = lam > 1e-10
    energies = X_hat ** 2  # (n_samp, n_comp)

    # --- 1. 频带能量 ---
    # 按 λ 从高到低 (平滑→振荡) 等分
    band_size = n_comp // N_BANDS
    band_edges = [i * band_size for i in range(N_BANDS + 1)]
    band_edges[-1] = n_comp
    band_feats = np.zeros((n_samp, N_BANDS))
    for b in range(N_BANDS):
        s, e = band_edges[b], band_edges[b + 1]
        band_feats[:, b] = energies[:, s:e].sum(axis=1)

    # --- 2. top-K 谱系数 (按平均能量选) ---
    mean_energy = energies.mean(axis=0)
    top_idx = np.argsort(mean_energy)[::-1][:TOP_K]
    top_feats = X_hat[:, top_idx]  # 保留符号信息

    # --- 3. 谱统计量 ---
    stats_feats = np.zeros((n_samp, 8))
    for i in range(n_samp):
        e = energies[i]
        tot = e.sum()
        if tot <= 0:
            continue
        # smoothness
        stats_feats[i, 0] = float((e * lam).sum() / tot / lam.max())
        # PR (有效分量数)
        stats_feats[i, 1] = float(tot ** 2 / (e ** 2).sum())
        # gini
        srt = np.sort(e[ok])
        n = len(srt)
        cum = np.cumsum(srt)
        stats_feats[i, 2] = float(1.0 - 2.0 * (cum.sum() - 0.5 * tot) / (n * tot))
        # top-10/50/100
        desc = np.cumsum(np.sort(e[ok])[::-1]) / tot
        stats_feats[i, 3] = float(desc[9]) if n > 9 else float(desc[-1])
        stats_feats[i, 4] = float(desc[49]) if n > 49 else float(desc[-1])
        stats_feats[i, 5] = float(desc[99]) if n > 99 else float(desc[-1])
        # frac_outside (谱外能量)
        stats_feats[i, 6] = max(tot - float(e[ok].sum()), 0.0) / tot
        # effective dim (90% energy)
        c90 = np.cumsum(np.sort(e[ok])[::-1]) / tot
        stats_feats[i, 7] = float(np.searchsorted(c90, 0.9) + 1)

    # 拼接
    feats = np.hstack([band_feats, top_feats, stats_feats])
    feat_names = ([f"band_{b}" for b in range(N_BANDS)]
                  + [f"coef_{k}" for k in range(TOP_K)]
                  + ["smoothness", "PR", "gini", "top10", "top50",
                     "top100", "frac_out", "eff_dim"])
    return feats, np.array(feat_names), top_idx


# ================================================================ 预测
def loocv_ridge(feats, y, alphas=None):
    """LOOCV + Ridge, 返回 (MAE, 最优α, 预测值)"""
    if alphas is None:
        alphas = np.logspace(-3, 4, 30)
    n = len(y)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan, np.nan, np.full(n, np.nan)

    X_v = feats[valid]
    y_v = y[valid]
    n_v = len(y_v)

    # LOOCV 选 α
    best_alpha, best_mae = alphas[0], np.inf
    for alpha in alphas:
        preds = np.zeros(n_v)
        for i in range(n_v):
            mask = np.ones(n_v, dtype=bool)
            mask[i] = False
            model = Ridge(alpha=alpha)
            model.fit(X_v[mask], y_v[mask])
            preds[i] = model.predict(X_v[i:i + 1])
        mae = mean_absolute_error(y_v, preds)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    # 用最优 α 做 LOOCV 得最终预测
    preds_full = np.full(n, np.nan)
    for i in range(n_v):
        mask = np.ones(n_v, dtype=bool)
        mask[i] = False
        model = Ridge(alpha=best_alpha)
        model.fit(X_v[mask], y_v[mask])
        preds_full[np.where(valid)[0][i]] = model.predict(X_v[i:i + 1])

    return best_mae, best_alpha, preds_full


def loocv_lasso(feats, y, alphas=None):
    """LOOCV + LASSO"""
    if alphas is None:
        alphas = np.logspace(-4, 1, 20)
    n = len(y)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan, np.nan, np.full(n, np.nan)

    X_v = feats[valid]
    y_v = y[valid]
    n_v = len(y_v)

    best_alpha, best_mae = alphas[0], np.inf
    for alpha in alphas:
        preds = np.zeros(n_v)
        for i in range(n_v):
            mask = np.ones(n_v, dtype=bool)
            mask[i] = False
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_v[mask])
            model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_tr, y_v[mask])
            preds[i] = model.predict(scaler.transform(X_v[i:i + 1]))
        mae = mean_absolute_error(y_v, preds)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    preds_full = np.full(n, np.nan)
    for i in range(n_v):
        mask = np.ones(n_v, dtype=bool)
        mask[i] = False
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_v[mask])
        model = Lasso(alpha=best_alpha, max_iter=5000)
        model.fit(X_tr, y_v[mask])
        preds_full[np.where(valid)[0][i]] = model.predict(
            scaler.transform(X_v[i:i + 1]))

    return best_mae, best_alpha, preds_full


# ================================================================ 消融
def learning_curve(feats, y, train_sizes, n_repeats=50):
    """学习曲线: 不同训练集大小下的 MAE (50 次随机划分)"""
    valid = ~np.isnan(y)
    X_v, y_v = feats[valid], y[valid]
    n_v = len(y_v)
    results = {}
    for n_train in train_sizes:
        if n_train >= n_v:
            # 用 LOOCV
            mae, _, _ = loocv_ridge(X_v, y_v)
            results[n_train] = [mae]
        else:
            maes = []
            for rep in range(n_repeats):
                perm = rng.permutation(n_v)
                tr, te = perm[:n_train], perm[n_train:]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_v[tr])
                X_te = scaler.transform(X_v[te])
                model = Ridge(alpha=1.0)
                model.fit(X_tr, y_v[tr])
                preds = model.predict(X_te)
                maes.append(mean_absolute_error(y_v[te], preds))
            results[n_train] = maes
    return results


# ================================================================ 主流程
def main():
    os.makedirs(RESULTS, exist_ok=True)

    # ---------- 加载数据 ----------
    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X = data["X_log1p"]
    Y = data["Y"]
    Y_short = data["Y_short"]
    Y_cols = data["Y_cols"]
    samples = data["samples"]
    genotypes = data["genotypes"]
    n_samp, n_genes = X.shape

    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    lam, V = spec["lam"], spec["V"]
    annotated = spec["annotated"]

    # ---------- 重建 A 并投影 ----------
    H = build_H()
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    inv_dv2 = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    inv_de2 = 1.0 / np.sqrt(np.maximum(d_e, 1))
    A = (sp.diags(inv_dv2) @ H @ sp.diags(inv_de2)).tocsr()

    print("投影 24 个样本到超图谱域...")
    X_hat = project_all(A, lam, V, X, annotated)
    print(f"谱系数矩阵: {X_hat.shape}")

    # ---------- 特征提取 ----------
    feats, feat_names, top_idx = extract_features(X_hat, lam)
    print(f"特征矩阵: {feats.shape} ({N_BANDS} 频带 + {TOP_K} 系数 + 8 统计)")

    # 保存特征
    feat_df = pd.DataFrame(feats, columns=feat_names)
    feat_df["sample"] = samples
    feat_df["genotype"] = genotypes
    for i, name in enumerate(Y_short):
        feat_df[name] = Y[:, i]
    feat_df.to_csv(os.path.join(RESULTS, "prediction_features.csv"), index=False)

    # ---------- Ridge LOOCV 预测 ----------
    print("\n" + "=" * 60)
    print("Ridge LOOCV 预测")
    print("=" * 60)
    results = []
    preds_ridge = np.full_like(Y, np.nan)
    for i, name in enumerate(Y_short):
        y = Y[:, i]
        mae, alpha, preds = loocv_ridge(feats, y)
        preds_ridge[:, i] = preds
        valid = ~np.isnan(y)
        mae_full = mean_absolute_error(y[valid], preds[valid])
        results.append(dict(phenotype=name, full_name=Y_cols[i],
                            model="Ridge", MAE=mae_full, best_alpha=alpha,
                            n_valid=int(valid.sum())))
        print(f"  {name:18s}: MAE={mae_full:.4f}  (α={alpha:.2e}, n={valid.sum()})")

    # ---------- LASSO LOOCV 预测 ----------
    print("\n" + "=" * 60)
    print("LASSO LOOCV 预测")
    print("=" * 60)
    preds_lasso = np.full_like(Y, np.nan)
    for i, name in enumerate(Y_short):
        y = Y[:, i]
        mae, alpha, preds = loocv_lasso(feats, y)
        preds_lasso[:, i] = preds
        valid = ~np.isnan(y)
        mae_full = mean_absolute_error(y[valid], preds[valid])
        results.append(dict(phenotype=name, full_name=Y_cols[i],
                            model="LASSO", MAE=mae_full, best_alpha=alpha,
                            n_valid=int(valid.sum())))
        print(f"  {name:18s}: MAE={mae_full:.4f}  (α={alpha:.2e}, n={valid.sum()})")

    # ---------- 汇总 ----------
    res_df = pd.DataFrame(results)
    # 论文基线
    baseline_rf = dict(phenotype="average", model="RF (paper)",
                       MAE=0.5768, best_alpha=np.nan, n_valid=24,
                       full_name="Random Forest baseline")
    baseline_hgnn = dict(phenotype="average", model="HGNN+VAE (paper)",
                         MAE=0.5530, best_alpha=np.nan, n_valid=24,
                         full_name="HGNN+VAE baseline")

    avg_ridge = res_df[res_df["model"] == "Ridge"]["MAE"].mean()
    avg_lasso = res_df[res_df["model"] == "LASSO"]["MAE"].mean()
    avg_row = dict(phenotype="average", model="Ridge (ours)",
                   MAE=avg_ridge, best_alpha=np.nan, n_valid=24,
                   full_name="Average")
    avg_row_l = dict(phenotype="average", model="LASSO (ours)",
                     MAE=avg_lasso, best_alpha=np.nan, n_valid=24,
                     full_name="Average")

    res_all = pd.concat([res_df,
                         pd.DataFrame([avg_row, avg_row_l,
                                       baseline_rf, baseline_hgnn])])
    res_all.to_csv(os.path.join(RESULTS, "prediction_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"  Ridge (ours):     avg MAE = {avg_ridge:.4f}")
    print(f"  LASSO (ours):     avg MAE = {avg_lasso:.4f}")
    print(f"  RF (paper):       avg MAE = 0.5768")
    print(f"  HGNN+VAE (paper): avg MAE = 0.5530")

    # ---------- 柱状图对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 各表型 MAE
    ax = axes[0]
    x_pos = np.arange(len(Y_short))
    w = 0.2
    ridge_maes = res_df[res_df["model"] == "Ridge"]["MAE"].values
    lasso_maes = res_df[res_df["model"] == "LASSO"]["MAE"].values
    ax.bar(x_pos - w / 2, ridge_maes, w, label="Ridge (ours)", color="#2b6cb0")
    ax.bar(x_pos + w / 2, lasso_maes, w, label="LASSO (ours)", color="#c05621")
    ax.axhline(0.5768, color="#38a169", ls="--", lw=1.2, label="RF (paper)")
    ax.axhline(0.5530, color="#d69e2e", ls="--", lw=1.2, label="HGNN+VAE (paper)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(Y_short, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("MAE (LOOCV)")
    ax.set_title("Per-phenotype prediction error")
    ax.legend(frameon=False, fontsize=8)

    # 右: 平均 MAE 对比
    ax = axes[1]
    models = ["Ridge\n(ours)", "LASSO\n(ours)", "RF\n(paper)", "HGNN+VAE\n(paper)"]
    maes = [avg_ridge, avg_lasso, 0.5768, 0.5530]
    colors = ["#2b6cb0", "#c05621", "#38a169", "#d69e2e"]
    bars = ax.bar(models, maes, color=colors, alpha=0.85)
    for bar, m in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average MAE")
    ax.set_title("Overall prediction performance")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_prediction.png"), dpi=160)
    plt.close(fig)

    # ================================================================ 消融
    # ---------- (a) 随机超图 ----------
    print("\n" + "=" * 60)
    print("消融 (a): 真实超图 vs 随机超边")
    print("=" * 60)
    random_maes = []
    for perm_i in range(N_PERM_HYPER):
        H_rand = build_random_H(H, seed=SEED + perm_i)
        A_r, lam_r, V_r = spectral_decompose(H_rand)
        X_hat_r = project_all(A_r, lam_r, V_r, X, annotated)
        feats_r, _, _ = extract_features(X_hat_r, lam_r)
        maes_i = []
        for j in range(len(Y_short)):
            mae_r, _, _ = loocv_ridge(feats_r, Y[:, j])
            maes_i.append(mae_r)
        avg_mae = np.nanmean(maes_i)
        random_maes.append(avg_mae)
        print(f"  随机超图 #{perm_i + 1}: avg MAE = {avg_mae:.4f}")

    real_avg = avg_ridge
    rand_mean = np.mean(random_maes)
    rand_std = np.std(random_maes)
    print(f"\n  真实: {real_avg:.4f}")
    print(f"  随机: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"  提升: {(rand_mean - real_avg) / rand_mean * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Real\nhypergraph", "Random\nhyperedges"],
           [real_avg, rand_mean],
           yerr=[0, rand_std],
           color=["#2b6cb0", "#c05621"], alpha=0.85,
           capsize=5, error_kw=dict(lw=1.5))
    ax.set_ylabel("Average MAE (Ridge LOOCV)")
    ax.set_title("Ablation: real vs random hypergraph")
    for i, (label, val) in enumerate(zip(["Real", "Random"],
                                          [real_avg, rand_mean])):
        ax.text(i, val + 0.01, f"{val:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_ablation_random.png"), dpi=160)
    plt.close(fig)

    # ---------- (b) 学习曲线 ----------
    print("\n" + "=" * 60)
    print("消融 (b): 学习曲线")
    print("=" * 60)
    train_sizes = [5, 8, 12, 16, 20, 24]
    lc_results = {}
    for j, name in enumerate(Y_short):
        lc = learning_curve(feats, Y[:, j], train_sizes)
        lc_results[name] = lc
        for n_tr in train_sizes:
            vals = lc[n_tr]
            print(f"  {name:18s} n={n_tr:2d}: "
                  f"MAE={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors_lc = ["#2b6cb0", "#c05621", "#38a169", "#d69e2e", "#805ad5"]
    for j, name in enumerate(Y_short):
        means, stds = [], []
        for n_tr in train_sizes:
            vals = lc_results[name][n_tr]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        ax.plot(train_sizes, means, "-o", color=colors_lc[j], label=name,
                markersize=4, lw=1.2)
        ax.fill_between(train_sizes, means - stds, means + stds,
                        color=colors_lc[j], alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("MAE (mean ± std)")
    ax.set_title("Learning curve (Ridge)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(train_sizes)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_learning_curve.png"), dpi=160)
    plt.close(fig)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
