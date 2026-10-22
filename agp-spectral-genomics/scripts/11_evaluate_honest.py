# -*- coding: utf-8 -*-
"""
11_evaluate_honest.py
尺度无关的表型预测评估, 用于校正 08/09 号脚本中 MAE 口径带来的误导。

问题背景:
    各表型量纲差异极大 (solidity std=0.05, inflor_height std=12.65)。
    绝对 MAE 小不等于预测得好: 只预测样本均值即可得到约 0.8 x std 的 MAE。
    因此必须与均值基线比较, 并使用尺度无关指标。

指标:
    MAE            平均绝对误差 (原始尺度)
    MAE_rel        MAE / std(y)
    skill          1 - MAE / MAE_mean_predictor   (>0 才表示优于预测均值)
    R2_loocv       留出法聚合 R^2 (可为负, 负值=不如均值基线)
    r_pearson      预测-实测 Pearson 相关 (聚合)
    p_value        相关系数双侧 t 检验

对照:
    mean_baseline  恒预测训练折均值 (下界基线)
    ridge_raw      直接用 38286 维表达 (无超图谱特征) —— 检验谱特征是否有增益
    ridge_pca      38 维随机投影 (PCA 前 38 主成分) —— 按维度数对齐的对照
    ridge_hsg      超图谱 38 维特征 (本文方法)

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz
    AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/honest_evaluation.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026

N_BANDS, TOP_K = 10, 20
ALPHAS = np.logspace(-3, 4, 15)


# ---------------------------------------------------------------- 超图
def build_H():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    map_ = np.load(os.path.join(PROCESSED, "node2row.npz"))
    node2row = dict(zip(map_["node_ids"].tolist(), map_["row_of_node"].tolist()))
    rows = np.array([node2row[g] for g in pairs[0].tolist()])
    cols = pairs[1]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(38286, int(cols.max()) + 1)).tocsr()
    H.sum_duplicates(); H.data[:] = 1.0
    return H


def dual_matrix(H):
    """返回归一化关联算子 A 与对偶空间 Gram S = A^T A (稠密, 7079^2)。"""
    d_v = np.asarray(H.sum(axis=1)).ravel()
    d_e = np.asarray(H.sum(axis=0)).ravel()
    inv_dv2 = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    inv_de2 = 1.0 / np.sqrt(np.maximum(d_e, 1))
    A = (sp.diags(inv_dv2) @ H @ sp.diags(inv_de2)).tocsr()
    S = (sp.diags(inv_de2) @ (H.T @ sp.diags(inv_dv2 ** 2)) @ H
         @ sp.diags(inv_de2)).tocsc()
    S = ((S + S.T) * 0.5).toarray()
    return A, S


def spectral_decompose(H):
    A, S = dual_matrix(H)
    lam, V = la.eigh(S, driver="evd")
    o = np.argsort(lam)[::-1]
    return A, np.maximum(lam[o], 0.0), V[:, o]


def spectral_cached(H, check_n=200, tol=1e-4, verbose=True):
    """
    优先复用 02 号脚本缓存的谱 (lam, V), 免去重复的 7079 阶稠密特征分解 (实测 >10 分钟)。
    缓存是否对应当前超图, 用特征方程残差抽样校验:
        ||S V[:, k] - lam_k V[:, k]|| / ||S V[:, k]||
    并与 02 号脚本一致地按 lam 降序排列。校验不通过则回退到重新分解。
    """
    path = os.path.join(PROCESSED, "hypergraph_spectrum.npz")
    A, S = dual_matrix(H)
    if os.path.exists(path):
        z = np.load(path)
        lam, V = np.maximum(z["lam"].astype(np.float64), 0.0), z["V"].astype(np.float64)
        o = np.argsort(lam)[::-1]
        lam, V = lam[o], V[:, o]
        SV = S @ V[:, :check_n]
        res = np.linalg.norm(SV - V[:, :check_n] * lam[None, :check_n], axis=0)
        rel = res / np.maximum(np.linalg.norm(SV, axis=0), 1e-30)
        orth = np.abs(V[:, :check_n].T @ V[:, :check_n] - np.eye(check_n)).max()
        if verbose:
            print(f"  缓存谱校验: 残差 max={rel.max():.2e}  正交偏差={orth:.2e}  "
                  f"lam[0]={lam[0]:.6f} lam[-1]={lam[-1]:.6f}")
        if rel.max() < tol:
            return A, lam, V
        if verbose:
            print("  缓存与当前超图不符, 重新分解")
    return spectral_decompose(H)


def project_all(A, lam, V, X, annotated):
    ann_pos = np.where(annotated)[0]
    ok = lam > 1e-10
    X_hat = np.zeros((X.shape[0], len(lam)))
    for i in range(X.shape[0]):
        x = X[i].copy(); x[ann_pos] -= x[ann_pos].mean()
        X_hat[i, ok] = (V[:, ok].T @ (A.T @ x)) / np.sqrt(lam[ok])
    return X_hat


def extract_features(X_hat, lam):
    n_samp, n_comp = X_hat.shape
    ok = lam > 1e-10
    energies = X_hat ** 2
    band_size = n_comp // N_BANDS
    band = np.zeros((n_samp, N_BANDS))
    for b in range(N_BANDS):
        s = b * band_size
        e = n_comp if b == N_BANDS - 1 else (b + 1) * band_size
        band[:, b] = energies[:, s:e].sum(axis=1)
    top = np.argsort(energies.mean(axis=0))[::-1][:TOP_K]
    coef = X_hat[:, top]
    st = np.zeros((n_samp, 8))
    for i in range(n_samp):
        e = energies[i]; tot = e.sum()
        if tot <= 0:
            continue
        st[i, 0] = float((e * lam).sum() / tot / lam.max())
        st[i, 1] = float(tot ** 2 / (e ** 2).sum())
        srt = np.sort(e[ok]); n = len(srt)
        st[i, 2] = float(1 - 2 * (np.cumsum(srt).sum() - .5 * tot) / (n * tot))
        desc = np.cumsum(srt[::-1]) / tot
        st[i, 3] = float(desc[9]) if n > 9 else 1.0
        st[i, 4] = float(desc[49]) if n > 49 else 1.0
        st[i, 5] = float(desc[99]) if n > 99 else 1.0
        st[i, 6] = max(tot - float(e[ok].sum()), 0.) / tot
        st[i, 7] = float(np.searchsorted(desc, .9) + 1)
    return np.hstack([band, coef, st])


# ---------------------------------------------------------------- 评估核心
# Ridge 在 p >> n 时不能用原始空间的 Cholesky (会构造 p x p 矩阵, 38286 维约 12GB)。
# 这里全部运算改写为 Gram 形式: 只需 n x n 矩阵, 且折内居中可由 Gram 子矩阵本身推出。
#
# 关键恒等式: 对子集 S, 居中核
#   K_c[i,j] = G[i,j] - mean_{l in S} G[i,l] - mean_{l in S} G[l,j] + mean_{l,m in S} G[l,m]
# 故任意折的 Gram 都能从全局 Gram 的 24x24 子块算出, 无需再碰 p 维数据。
def gram_centered(G_sub):
    k = G_sub.shape[0]
    r = G_sub.mean(1, keepdims=True)
    c = G_sub.mean(0, keepdims=True)
    m = G_sub.mean()
    return G_sub - r - c + m


def ridge_dual_predict(G_tr, y_tr, g_te, alpha):
    """
    折内重拟合的 Ridge 预测 (与 sklearn Ridge(fit_intercept=True) 逐折拟合等价):
        G_tr : k x k 训练样本的全局 Gram 子块
        y_tr : k     训练目标
        g_te : 1 x k 测试行与训练行的全局 Gram
    居中与截距均在折内完成。
    """
    k = len(y_tr)
    Kc = gram_centered(G_tr)
    yc = y_tr - y_tr.mean()
    coef = np.linalg.solve(Kc + alpha * np.eye(k), yc)
    g_te_c = g_te - G_tr.mean(0, keepdims=True)      # 测试行的居中核
    return float(g_te_c @ coef) + y_tr.mean(), coef, Kc, yc


def loo_via_dual(G_tr, y_tr, alpha):
    """
    在训练折内部再做一次留一 (对偶形式), 用于内层选 α。
    返回每个内层留出点的预测。
    """
    k = len(y_tr)
    idx = np.arange(k)
    preds = np.zeros(k)
    for j in range(k):
        S = np.setdiff1d(idx, [j])
        preds[j], _, _, _ = ridge_dual_predict(G_tr[np.ix_(S, S)], y_tr[S],
                                               G_tr[j, S][None, :], alpha)
    return preds


def nested_loocv(G, y, alphas=ALPHAS):
    """
    严格嵌套 LOOCV (全程 Gram 形式):
        外层留出 1 个样本 → 内层在剩余 k 个样本上做 k 次留一选 α → 用该 α 预测。
    G: n x n 全局 Gram (特征已做无监督全局标准化)
    """
    n = len(y)
    idx = np.arange(n)
    preds = np.zeros(n)
    for i in range(n):
        S = np.setdiff1d(idx, [i])
        G_tr, y_tr = G[np.ix_(S, S)], y[S]

        best_a, best_err = alphas[0], np.inf
        for a in alphas:
            err = mean_absolute_error(y_tr, loo_via_dual(G_tr, y_tr, a))
            if err < best_err:
                best_err, best_a = err, a

        preds[i], _, _, _ = ridge_dual_predict(G_tr, y_tr, G[i, S][None, :], best_a)
    return preds


def fixed_alpha_loocv(G, y, alpha):
    """固定 α、不做任何选择的 LOOCV —— 彻底排除调参乐观偏差"""
    n = len(y)
    idx = np.arange(n)
    preds = np.zeros(n)
    for i in range(n):
        S = np.setdiff1d(idx, [i])
        preds[i], _, _, _ = ridge_dual_predict(G[np.ix_(S, S)], y[S],
                                               G[i, S][None, :], alpha)
    return preds


def loo_closed_form_identity(G, y, alpha):
    """留一恒等式 y_i - r_i/(1-h_ii), 仅作自检用"""
    k = len(y)
    Kc = gram_centered(G)
    yc = y - y.mean()
    Ainv = np.linalg.inv(Kc + alpha * np.eye(k))
    coef = Ainv @ yc
    fitted = Kc @ coef + y.mean()
    h = np.clip(np.einsum('ij,ji->i', Kc, Ainv) * (1 - 1/k) + 1/k, 0, 1 - 1e-12)
    return y - (y - fitted) / (1 - h)


def summarize(y, preds, tag):
    """与均值基线对比的尺度无关汇总"""
    n = len(y)
    m_pred = np.full(n, y.mean())        # 均值预测 (用全集均值, 对所有方法一致)
    mae = mean_absolute_error(y, preds)
    mae_base = mean_absolute_error(y, m_pred)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    if np.std(preds) < 1e-12:
        r, p = 0.0, 1.0
    else:
        r, p = sps.pearsonr(y, preds)
    # 塌缩诊断: 当模型被过度收缩时, 留一预测趋于"其余样本的均值"
    #   pred_i ~= (n*ȳ - y_i)/(n-1), 与 y  mechanically 负相关 (r -> -1)。
    # 因此 r 接近 -1 且 std(preds) << std(y) 表示模型塌缩到均值, 而非学到反向信号。
    collapse = float(np.std(preds) / np.std(y, ddof=0)) if np.std(y, ddof=0) > 0 else 0.0
    return dict(model=tag, n=n,
                mae=mae, mae_mean_baseline=mae_base,
                mae_over_std=mae / np.std(y, ddof=0),
                skill=1 - mae / mae_base,
                r2=r2, r=float(r), p=float(p),
                pred_std_ratio=collapse,
                std_y=float(np.std(y, ddof=0)))


def gram_of(feats, tol=1e-8):
    """特征做无监督全局标准化后返回 Gram 矩阵。
    标准化只用特征本身的分布, 不涉及表型标签, 故不构成标签泄漏。

    数值守卫: 方差相对极小的列属于数值噪声 (如 PCA 末位退化主成分、能量近零的
    谱系数)。对其做 z-score 会把噪声放大到 O(1), 在 n=24 上足以制造虚假的
    "完美相关" —— FvFm 上曾出现 PCA_5d 的 r=-0.9999 而 skill 仍为负。
    因此仅保留 sd > tol * max(sd) 的列。
    """
    mu = feats.mean(0)
    sd = feats.std(0)
    smax = float(sd.max()) if sd.size else 0.0
    keep = sd > tol * smax if smax > 0 else np.zeros(feats.shape[1], bool)
    Z = (feats[:, keep] - mu[keep]) / sd[keep]
    dropped = int((~keep).sum())
    if dropped:
        print(f"      [数值守卫] 丢弃 {dropped}/{feats.shape[1]} 个近零方差列")
    return Z @ Z.T


def verify_against_sklearn():
    """Gram 对偶折内重拟合 vs sklearn Ridge 逐样本拟合 一致性自检"""
    rng = np.random.default_rng(0)
    worst = 0.0
    ident_worst = 0.0
    for (n, p) in [(20, 38), (20, 500), (24, 4000)]:
        X = rng.standard_normal((n, p))
        X = (X - X.mean(0)) / X.std(0)
        y = rng.standard_normal(n) * 2 + 5
        G = X @ X.T
        for alpha in [1e-2, 1.0, 1e3]:
            sk = np.zeros(n)
            dual = np.zeros(n)
            for i in range(n):
                S = np.setdiff1d(np.arange(n), [i])
                sk[i] = Ridge(alpha=alpha).fit(X[S], y[S]).predict(X[i:i + 1])[0]
                dual[i] = ridge_dual_predict(G[np.ix_(S, S)], y[S],
                                             G[i, S][None, :], alpha)[0]
            # 内层留一: 对偶逐折重拟合 vs 留一恒等式 (仅作偏差量级参考)
            loo_dual = loo_via_dual(G, y, alpha)
            loo_ident = loo_closed_form_identity(G, y, alpha)
            e1 = np.abs(sk - dual).max()
            e2 = np.abs(loo_dual - loo_ident).max()
            worst = max(worst, e1)
            ident_worst = max(ident_worst, e2)
            print(f"  n={n:3d} p={p:5d} a={alpha:8.2e}  "
                  f"|dual-sklearn|={e1:.2e}  |LOO对偶-恒等式|={e2:.2e}")
    assert worst < 1e-8, f"自检未通过, 与 sklearn 最大偏差 {worst:.3e}"
    print(f"  对偶重拟合与 sklearn 最大偏差 {worst:.2e} -> 通过")
    print(f"  注: 留一恒等式 y_i-r_i/(1-h_ii) 对带截距的岭回归不精确成立"
          f" (最大偏差 {ident_worst:.2e}), 故内层选 α 一律用逐折重拟合。")
    print()
    return worst


def main():
    os.makedirs(RESULTS, exist_ok=True)

    print("自检: Gram 对偶重拟合 vs sklearn / 留一恒等式")
    verify_against_sklearn()

    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X, Y, Y_short = data["X_log1p"], data["Y"], data["Y_short"]
    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    annotated = spec["annotated"]

    H = build_H()
    A, lam, V = spectral_cached(H)
    X_hat = project_all(A, lam, V, X, annotated)
    f_hsg = extract_features(X_hat, lam)

    # 对照组特征
    ann_pos = np.where(annotated)[0]
    Xs = StandardScalerLike(X[:, ann_pos])
    # 24 个样本的秩上限为 23, 且末尾主成分数值退化 (方向不稳定), 取前 5 维作为稳健对照。
    f_pca = PCA(n_components=5, random_state=SEED).fit_transform(Xs)
    rng = np.random.default_rng(SEED)
    f_rand = rng.standard_normal((X.shape[0], 38))
    print(f"特征维度: HSG={f_hsg.shape[1]}, PCA(前5)={f_pca.shape[1]}, "
          f"随机={f_rand.shape[1]}, 原始表达={Xs.shape[1]}\n")

    # 预先算好各特征表示的全局 Gram (24x24), 之后全部评估在 Gram 上进行
    grams = {tag: gram_of(F) for tag, F in
             [("HSG_38d", f_hsg), ("PCA_5d", f_pca),
              ("raw_expression", Xs), ("random_38d", f_rand)]}

    print("=" * 92)
    print("评估: 嵌套 LOOCV (α 逐折选) 与 固定 α (不调参)")
    print("=" * 92)

    rows = []
    for i, name in enumerate(Y_short):
        y = Y[:, i]
        ok = ~np.isnan(y)
        y_v = y[ok]
        idx = np.where(ok)[0]
        for tag, G in grams.items():
            sub = G[np.ix_(idx, idx)]
            # (1) 严格嵌套: α 在每个外折内部由内层 LOOCV 选出
            pr = nested_loocv(sub, y_v)
            s = summarize(y_v, pr, tag); s["phenotype"] = name
            s["protocol"] = "nested_LOOCV"; rows.append(s)
            # (2)(3) 固定 α, 完全不调参, 排除选择乐观偏差
            for a in (1.0, 100.0):
                pr = fixed_alpha_loocv(sub, y_v, a)
                s = summarize(y_v, pr, tag); s["phenotype"] = name
                s["protocol"] = f"fixed_alpha={a:g}"; rows.append(s)
        s = summarize(y_v, np.full(len(y_v), y_v.mean()), "mean_predictor")
        s["phenotype"] = name; s["protocol"] = "baseline"
        rows.append(s)
        print(f"  {name} 完成")

    df = pd.DataFrame(rows)
    cols = ["phenotype", "model", "protocol", "n", "mae", "mae_mean_baseline",
            "mae_over_std", "skill", "r2", "r", "p", "std_y"]
    df = df[cols]
    df.to_csv(os.path.join(RESULTS, "honest_evaluation.csv"), index=False)

    print("\n" + "=" * 92)
    print("分表型 x 方法 (skill>0 或 R2>0 才算优于均值基线)")
    print("=" * 92)
    for name in Y_short:
        sub = df[df.phenotype == name]
        print(f"\n【{name}】 std={sub['std_y'].iloc[0]:.4f}, "
              f"均值基线 MAE={sub.loc[sub.model == 'mean_predictor', 'mae'].iloc[0]:.4f}")
        print(sub[["model", "protocol", "mae", "mae_over_std", "skill", "r2", "r", "p"]]
              .to_string(index=False))

    print("\n" + "=" * 92)
    print("按 方法 x 评估协议 平均 (排除均值基线)")
    print("=" * 92)
    mm = df[df.model != "mean_predictor"].groupby(["protocol", "model"]).agg(
        mean_mae_over_std=("mae_over_std", "mean"),
        mean_skill=("skill", "mean"),
        mean_r2=("r2", "mean"),
        mean_r=("r", "mean"),
        n_positive_r2=("r2", lambda s: int((s > 0).sum())),
    )
    print(mm.to_string())

    print("\n" + "=" * 92)
    print("结论判据: 是否存在优于恒预测均值的表型")
    print("=" * 92)
    for proto, grp in df[df.model != "mean_predictor"].groupby("protocol"):
        pos = grp[grp.skill > 0].groupby("phenotype").size()
        print(f"  {proto:22s}: 优于均值基线的表型数 = {len(pos)} / 5  "
              f"平均 skill={grp['skill'].mean():+.3f}  "
              f"平均 R2={grp['r2'].mean():+.3f}")
        for ph in pos.index:
            r = grp[(grp.phenotype == ph) & (grp.skill > 0)]["r"].mean()
            print(f"      + {ph} (skill={r:.3f} 为该表型最优方法的 r)")
    print("\n注: skill = 1 - MAE/MAE_mean_predictor; <=0 表示不优于恒预测均值。")


def StandardScalerLike(M):
    mu, sd = M.mean(0), M.std(0)
    return (M - mu) / np.where(sd > 0, sd, 1.0)


if __name__ == "__main__":
    main()
