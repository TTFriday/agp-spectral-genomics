# -*- coding: utf-8 -*-
"""
09_optimize_prediction.py
优化预测：核方法、表型归一化、特征选择、异常值处理

改进点:
    1. 核 Ridge (RBF / 多项式) 处理非线性
    2. 表型 z-score 归一化后预测
    3. 特征子集对比 (频带 / 系数 / 统计 / 组合)
    4. inflor_height 异常值检测 (IQR / 3σ)

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz
    AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/optimization_results.csv
    results/fig_optimization.png
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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026
rng = np.random.default_rng(SEED)

N_BANDS = 10
TOP_K = 20


# ================================================================ 超图 & 谱分解
def build_H():
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


def spectral_decompose(H):
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
def extract_features(X_hat, lam, feature_set="all"):
    n_samp, n_comp = X_hat.shape
    ok = lam > 1e-10
    energies = X_hat ** 2

    band_size = n_comp // N_BANDS
    band_edges = [i * band_size for i in range(N_BANDS + 1)]
    band_edges[-1] = n_comp
    band_feats = np.zeros((n_samp, N_BANDS))
    for b in range(N_BANDS):
        s, e = band_edges[b], band_edges[b + 1]
        band_feats[:, b] = energies[:, s:e].sum(axis=1)

    mean_energy = energies.mean(axis=0)
    top_idx = np.argsort(mean_energy)[::-1][:TOP_K]
    top_feats = X_hat[:, top_idx]

    stats_feats = np.zeros((n_samp, 8))
    for i in range(n_samp):
        e = energies[i]
        tot = e.sum()
        if tot <= 0:
            continue
        stats_feats[i, 0] = float((e * lam).sum() / tot / lam.max())
        stats_feats[i, 1] = float(tot ** 2 / (e ** 2).sum())
        srt = np.sort(e[ok])
        n = len(srt)
        cum = np.cumsum(srt)
        stats_feats[i, 2] = float(1.0 - 2.0 * (cum.sum() - 0.5 * tot) / (n * tot))
        desc = np.cumsum(np.sort(e[ok])[::-1]) / tot
        stats_feats[i, 3] = float(desc[9]) if n > 9 else float(desc[-1])
        stats_feats[i, 4] = float(desc[49]) if n > 49 else float(desc[-1])
        stats_feats[i, 5] = float(desc[99]) if n > 99 else float(desc[-1])
        stats_feats[i, 6] = max(tot - float(e[ok].sum()), 0.0) / tot
        c90 = np.cumsum(np.sort(e[ok])[::-1]) / tot
        stats_feats[i, 7] = float(np.searchsorted(c90, 0.9) + 1)

    if feature_set == "band":
        return band_feats, [f"band_{b}" for b in range(N_BANDS)]
    elif feature_set == "coef":
        return top_feats, [f"coef_{k}" for k in range(TOP_K)]
    elif feature_set == "stats":
        return stats_feats, ["smoothness", "PR", "gini", "top10", "top50",
                             "top100", "frac_out", "eff_dim"]
    else:  # all
        feats = np.hstack([band_feats, top_feats, stats_feats])
        names = ([f"band_{b}" for b in range(N_BANDS)]
                 + [f"coef_{k}" for k in range(TOP_K)]
                 + ["smoothness", "PR", "gini", "top10", "top50",
                    "top100", "frac_out", "eff_dim"])
        return feats, names


# ================================================================ 预测
def loocv_predict(feats, y, model_type="ridge", alphas=None, kernel_params=None):
    """LOOCV 预测，支持 Ridge / KernelRidge / SVR"""
    n = len(y)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan, np.full(n, np.nan)

    X_v = feats[valid]
    y_v = y[valid]
    n_v = len(y_v)

    if alphas is None:
        alphas = np.logspace(-3, 4, 20)

    best_alpha, best_mae = alphas[0], np.inf
    loo = LeaveOneOut()

    for alpha in alphas:
        preds = np.zeros(n_v)
        for train_idx, test_idx in loo.split(X_v):
            X_tr, X_te = X_v[train_idx], X_v[test_idx]
            y_tr = y_v[train_idx]

            if model_type == "ridge":
                model = Ridge(alpha=alpha)
            elif model_type == "krb_rbf":
                model = KernelRidge(alpha=alpha, kernel="rbf",
                                    gamma=kernel_params.get("gamma", 0.1))
            elif model_type == "krb_poly":
                model = KernelRidge(alpha=alpha, kernel="poly",
                                    degree=kernel_params.get("degree", 3),
                                    gamma=kernel_params.get("gamma", 1.0))
            elif model_type == "svr_rbf":
                model = SVR(kernel="rbf", C=1.0 / alpha,
                            gamma=kernel_params.get("gamma", 0.1))
            else:
                model = Ridge(alpha=alpha)

            model.fit(X_tr, y_tr)
            preds[test_idx[0]] = model.predict(X_te)[0]

        mae = mean_absolute_error(y_v, preds)
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    # 最终 LOOCV
    preds_full = np.full(n, np.nan)
    for train_idx, test_idx in loo.split(X_v):
        X_tr, X_te = X_v[train_idx], X_v[test_idx]
        y_tr = y_v[train_idx]

        if model_type == "ridge":
            model = Ridge(alpha=best_alpha)
        elif model_type == "krb_rbf":
            model = KernelRidge(alpha=best_alpha, kernel="rbf",
                                gamma=kernel_params.get("gamma", 0.1))
        elif model_type == "krb_poly":
            model = KernelRidge(alpha=best_alpha, kernel="poly",
                                degree=kernel_params.get("degree", 3),
                                gamma=kernel_params.get("gamma", 1.0))
        elif model_type == "svr_rbf":
            model = SVR(kernel="rbf", C=1.0 / best_alpha,
                        gamma=kernel_params.get("gamma", 0.1))
        else:
            model = Ridge(alpha=best_alpha)

        model.fit(X_tr, y_tr)
        preds_full[np.where(valid)[0][test_idx[0]]] = model.predict(X_te)[0]

    return best_mae, preds_full


# ================================================================ 异常值处理
def detect_outliers(y, method="iqr", threshold=1.5):
    """检测异常值，返回布尔掩码 (True=正常)"""
    valid = ~np.isnan(y)
    y_v = y[valid]

    if method == "iqr":
        q1, q3 = np.percentile(y_v, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        mask = (y >= lower) & (y <= upper)
        mask[np.isnan(y)] = False
    elif method == "zscore":
        mean, std = np.nanmean(y), np.nanstd(y)
        mask = np.abs(y - mean) <= threshold * std
        mask[np.isnan(y)] = False
    else:
        mask = valid

    return mask


# ================================================================ 主流程
def main():
    os.makedirs(RESULTS, exist_ok=True)

    # 加载数据
    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X = data["X_log1p"]
    Y = data["Y"]
    Y_short = data["Y_short"]
    Y_cols = data["Y_cols"]
    samples = data["samples"]
    n_samp = X.shape[0]

    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    lam, V = spec["lam"], spec["V"]
    annotated = spec["annotated"]

    # 重建超图 & 投影
    H = build_H()
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    inv_dv2 = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    inv_de2 = 1.0 / np.sqrt(np.maximum(d_e, 1))
    A = (sp.diags(inv_dv2) @ H @ sp.diags(inv_de2)).tocsr()

    print("投影样本...")
    X_hat = project_all(A, lam, V, X, annotated)

    # ================================================================ 实验矩阵
    results = []
    feature_sets = ["all", "band", "coef", "stats"]
    model_types = ["ridge", "krb_rbf", "krb_poly", "svr_rbf"]
    normalization = [False, True]  # 是否 z-score 归一化表型
    outlier_methods = [None, "iqr", "zscore"]

    print("\n" + "=" * 70)
    print("优化实验矩阵")
    print("=" * 70)

    for feat_set in feature_sets:
        feats, feat_names = extract_features(X_hat, lam, feat_set)
        print(f"\n特征集: {feat_set} ({feats.shape[1]} 维)")

        for norm in normalization:
            for outlier_method in outlier_methods:
                for model_type in model_types:
                    maes = []
                    for i, name in enumerate(Y_short):
                        y = Y[:, i].copy()

                        # 异常值处理
                        if outlier_method:
                            mask = detect_outliers(y, outlier_method)
                            y[~mask] = np.nan

                        # 表型归一化
                        if norm:
                            scaler = RobustScaler()
                            y_valid = y[~np.isnan(y)].reshape(-1, 1)
                            scaler.fit(y_valid)
                            y = scaler.transform(y.reshape(-1, 1)).ravel()
                            y[np.isnan(Y[:, i])] = np.nan

                        # 核方法参数
                        kernel_params = {}
                        if model_type in ["krb_rbf", "svr_rbf"]:
                            kernel_params["gamma"] = 0.01
                        elif model_type == "krb_poly":
                            kernel_params["gamma"] = 1.0
                            kernel_params["degree"] = 3

                        # LOOCV
                        alphas = np.logspace(-3, 4, 15)
                        mae, preds = loocv_predict(feats, y, model_type,
                                                    alphas, kernel_params)

                        # 如果归一化了，还原 MAE 到原始尺度
                        if norm and not np.isnan(mae):
                            y_orig = Y[:, i]
                            y_valid_orig = y_orig[~np.isnan(y_orig)]
                            mae = mae * np.nanstd(y_orig)

                        maes.append(mae)

                    avg_mae = np.nanmean(maes)
                    results.append(dict(
                        feature_set=feat_set,
                        normalization=norm,
                        outlier_method=outlier_method or "none",
                        model=model_type,
                        avg_mae=avg_mae,
                        n_features=feats.shape[1]
                    ))

                    print(f"  {model_type:10s} norm={norm} outlier={outlier_method or 'none':5s}: "
                          f"MAE={avg_mae:.4f}")

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RESULTS, "optimization_results.csv"), index=False)

    # 找最佳配置
    best_idx = res_df["avg_mae"].idxmin()
    best = res_df.loc[best_idx]
    print("\n" + "=" * 70)
    print(f"最佳配置: {best['model']} + {best['feature_set']} "
          f"(norm={best['normalization']}, outlier={best['outlier_method']})")
    print(f"最佳 MAE: {best['avg_mae']:.4f}")
    print("=" * 70)

    # 对比基线
    print("\n对比:")
    print(f"  最佳 (ours): {best['avg_mae']:.4f}")
    print(f"  RF (paper):  0.5768")
    print(f"  HGNN+VAE:    0.5530")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 不同特征集对比 (Ridge, no norm, no outlier)
    ax = axes[0, 0]
    subset = res_df[(res_df["model"] == "ridge") &
                    (res_df["normalization"] == False) &
                    (res_df["outlier_method"] == "none")]
    ax.bar(subset["feature_set"], subset["avg_mae"], color="#2b6cb0", alpha=0.85)
    ax.axhline(0.5768, color="#38a169", ls="--", label="RF (paper)")
    ax.set_ylabel("MAE")
    ax.set_title("Feature set comparison (Ridge)")
    ax.legend()

    # 2. 不同模型对比 (all features, no norm, no outlier)
    ax = axes[0, 1]
    subset = res_df[(res_df["feature_set"] == "all") &
                    (res_df["normalization"] == False) &
                    (res_df["outlier_method"] == "none")]
    ax.bar(subset["model"], subset["avg_mae"], color="#c05621", alpha=0.85)
    ax.axhline(0.5768, color="#38a169", ls="--", label="RF (paper)")
    ax.set_ylabel("MAE")
    ax.set_title("Model comparison (all features)")
    ax.legend()

    # 3. 归一化影响 (all features, Ridge)
    ax = axes[1, 0]
    subset = res_df[(res_df["feature_set"] == "all") &
                    (res_df["model"] == "ridge") &
                    (res_df["outlier_method"] == "none")]
    ax.bar(["No norm", "Z-score"], subset["avg_mae"], color="#805ad5", alpha=0.85)
    ax.set_ylabel("MAE")
    ax.set_title("Normalization effect")

    # 4. 异常值处理影响 (all features, Ridge, no norm)
    ax = axes[1, 1]
    subset = res_df[(res_df["feature_set"] == "all") &
                    (res_df["model"] == "ridge") &
                    (res_df["normalization"] == False)]
    ax.bar(subset["outlier_method"], subset["avg_mae"], color="#d69e2e", alpha=0.85)
    ax.set_ylabel("MAE")
    ax.set_title("Outlier handling")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_optimization.png"), dpi=160)
    plt.close(fig)

    print(f"\n优化结果已保存: results/optimization_results.csv")
    print(f"可视化: results/fig_optimization.png")


if __name__ == "__main__":
    main()
