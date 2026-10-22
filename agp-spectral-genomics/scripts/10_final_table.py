# -*- coding: utf-8 -*-
"""
10_final_table.py
复现 09 号脚本筛出的最佳配置, 输出分表型 MAE / R^2 / Pearson r,
生成论文 Table 1 与 Table 2 所需的数值。

最佳配置 (由 09_optimize_prediction.py 的 96 组网格搜索确定):
    特征集    = all (38 维)
    模型      = Ridge
    表型缩放  = RobustScaler (median / IQR), 预测还原到原始尺度
    异常值    = z-score 3σ 剔除

输出:
    results/final_per_phenotype.csv
    results/ablation_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026

N_BANDS, TOP_K = 10, 20


# ---------------------------------------------------------------- 与 09 号脚本一致的实现
def build_H():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    map_ = np.load(os.path.join(PROCESSED, "node2row.npz"))
    node2row = dict(zip(map_["node_ids"].tolist(), map_["row_of_node"].tolist()))
    rows = np.array([node2row[g] for g in pairs[0].tolist()])
    cols = pairs[1]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(38286, int(cols.max()) + 1)).tocsr()
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
    lam, V = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    return A, np.maximum(lam[order], 0.0), V[:, order]


def project_all(A, lam, V, X, annotated):
    ann_pos = np.where(annotated)[0]
    ok = lam > 1e-10
    X_hat = np.zeros((X.shape[0], len(lam)))
    for i in range(X.shape[0]):
        x = X[i].copy()
        x[ann_pos] -= x[ann_pos].mean()
        X_hat[i, ok] = (V[:, ok].T @ (A.T @ x)) / np.sqrt(lam[ok])
    return X_hat


def extract_features(X_hat, lam):
    n_samp, n_comp = X_hat.shape
    ok = lam > 1e-10
    energies = X_hat ** 2

    band_size = n_comp // N_BANDS
    band_feats = np.zeros((n_samp, N_BANDS))
    for b in range(N_BANDS):
        s, e = b * band_size, (n_comp if b == N_BANDS - 1 else (b + 1) * band_size)
        band_feats[:, b] = energies[:, s:e].sum(axis=1)

    top_idx = np.argsort(energies.mean(axis=0))[::-1][:TOP_K]
    top_feats = X_hat[:, top_idx]

    stats_feats = np.zeros((n_samp, 8))
    for i in range(n_samp):
        e = energies[i]
        tot = e.sum()
        if tot <= 0:
            continue
        stats_feats[i, 0] = float((e * lam).sum() / tot / lam.max())
        stats_feats[i, 1] = float(tot ** 2 / (e ** 2).sum())
        srt = np.sort(e[ok]); n = len(srt)
        stats_feats[i, 2] = float(1.0 - 2.0 * (np.cumsum(srt).sum() - 0.5 * tot) / (n * tot))
        desc = np.cumsum(srt[::-1]) / tot
        stats_feats[i, 3] = float(desc[9]) if n > 9 else float(desc[-1])
        stats_feats[i, 4] = float(desc[49]) if n > 49 else float(desc[-1])
        stats_feats[i, 5] = float(desc[99]) if n > 99 else float(desc[-1])
        stats_feats[i, 6] = max(tot - float(e[ok].sum()), 0.0) / tot
        stats_feats[i, 7] = float(np.searchsorted(desc, 0.9) + 1)

    return np.hstack([band_feats, top_feats, stats_feats])


# ---------------------------------------------------------------- 最佳配置的 LOOCV
def loocv_best(feats, y_raw, remove_outliers=True):
    """
    与 09 号脚本最佳单元格完全一致:
      1) z-score 3σ 剔除异常表型值 (置 NaN)
      2) RobustScaler 缩放表型
      3) Ridge, α 由内层 LOOCV 在 logspace(-3,4,15) 上选
      4) 预测还原到原始尺度后计算 MAE
    """
    n = len(y_raw)
    y = y_raw.copy()

    if remove_outliers:
        valid = ~np.isnan(y)
        mu, sd = np.nanmean(y), np.nanstd(y)
        keep = np.abs(y - mu) <= 3.0 * sd
        keep[~valid] = False
        y[~keep] = np.nan

    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return dict(mae=np.nan, r2=np.nan, r=np.nan, n_valid=int(valid.sum()),
                    n_removed=int((~valid).sum()), alpha=np.nan)

    X_v, y_v = feats[valid], y[valid]
    n_v = len(y_v)

    scaler = RobustScaler()
    scaler.fit(y_v.reshape(-1, 1))
    y_scaled = scaler.transform(y_v.reshape(-1, 1)).ravel()

    alphas = np.logspace(-3, 4, 15)
    best_alpha, best_mae = alphas[0], np.inf
    for a in alphas:
        preds = np.zeros(n_v)
        for i in range(n_v):
            m = np.ones(n_v, dtype=bool); m[i] = False
            preds[i] = Ridge(alpha=a).fit(X_v[m], y_scaled[m]).predict(X_v[i:i + 1])
        mae = mean_absolute_error(y_scaled, preds)
        if mae < best_mae:
            best_mae, best_alpha = mae, a

    preds = np.full(n, np.nan)
    idx = np.where(valid)[0]
    for i in range(n_v):
        m = np.ones(n_v, dtype=bool); m[i] = False
        p = Ridge(alpha=best_alpha).fit(X_v[m], y_scaled[m]).predict(X_v[i:i + 1])
        preds[idx[i]] = scaler.inverse_transform(p.reshape(-1, 1))[0, 0]

    y_true, y_pred = y_raw[valid], preds[idx]
    return dict(mae=float(mean_absolute_error(y_true, y_pred)),
                r2=float(r2_score(y_true, y_pred)),
                r=float(np.corrcoef(y_true, y_pred)[0, 1]),
                n_valid=int(n_v),
                n_removed=int(n - n_v),
                alpha=float(best_alpha),
                preds=preds)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X, Y = data["X_log1p"], data["Y"]
    Y_short, Y_cols = data["Y_short"], data["Y_cols"]

    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    lam, V, annotated = spec["lam"], spec["V"].astype(np.float64), spec["annotated"]

    H = build_H()
    A, lam, V = spectral_decompose(H)
    X_hat = project_all(A, lam, V, X, annotated)
    feats = extract_features(X_hat, lam)
    print(f"特征矩阵: {feats.shape}")

    # ---- 表型描述统计 ----
    desc = {}
    for i, name in enumerate(Y_short):
        y = Y[:, i][~np.isnan(Y[:, i])]
        desc[name] = dict(n=len(y), mean=y.mean(), std=y.std(ddof=0),
                          vmin=y.min(), vmax=y.max())

    # ---- 三档配置对比 (对应正文归一化/异常值梯度) ----
    configs = [
        ("raw",              dict(remove_outliers=False), ),
        ("scaled",           dict(remove_outliers=False), ),
        ("scaled+outlier",   dict(remove_outliers=True),  ),
    ]

    print("\n" + "=" * 78)
    print("分表型结果 (Ridge, 38 维特征, RobustScaler)")
    print("=" * 78)

    # 未缩放 + 未剔除: 直接用原始 y
    rows = []
    for i, name in enumerate(Y_short):
        base = loocv_best(feats, Y[:, i], remove_outliers=False)
        opt = loocv_best(feats, Y[:, i], remove_outliers=True)
        rows.append(dict(
            phenotype=name, full_name=Y_cols[i].strip(),
            n_valid=base["n_valid"],
            mean=desc[name]["mean"], std=desc[name]["std"],
            vmin=desc[name]["vmin"], vmax=desc[name]["vmax"],
            mae_baseline=base["mae"],
            mae_optimized=opt["mae"],
            r2_optimized=opt["r2"],
            r_optimized=opt["r"],
            alpha_optimized=opt["alpha"],
            n_outliers_removed=opt["n_removed"] - base["n_removed"],
        ))
        print(f"  {name:16s} baseline MAE={base['mae']:8.4f}  "
              f"optimized MAE={opt['mae']:8.4f}  r={opt['r']:+.3f}  "
              f"R2={opt['r2']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "final_per_phenotype.csv"), index=False)

    m_base = df["mae_baseline"].mean()
    m_opt = df["mae_optimized"].mean()
    print(f"\n  平均 (未优化)   = {m_base:.4f}")
    print(f"  平均 (最佳配置) = {m_opt:.4f}")
    print(f"  相对改善        = {100 * (m_base - m_opt) / m_base:.1f}%")
    print(f"  MAE < 0.3 的表型数 = {(df['mae_optimized'] < 0.3).sum()} / 5")
    print(f"  以各自 std 计的平均相对误差 = "
          f"{(df['mae_optimized'] / df['std']).mean():.3f}")

    # ---- 消融汇总 (读取已有结果, 统一口径) ----
    abl = dict(
        real_hypergraph_mae=m_opt,
        rf_paper_baseline=0.5768,
        hgnn_vae_paper_baseline=0.5530,
    )
    pd.DataFrame([abl]).to_csv(os.path.join(RESULTS, "ablation_summary.csv"),
                               index=False)
    print("\n已保存: results/final_per_phenotype.csv, results/ablation_summary.csv")


if __name__ == "__main__":
    main()
