# -*- coding: utf-8 -*-
"""
12_ablation_honest.py
在严格嵌套 LOOCV 协议下重测"真实超图 vs 随机超图"消融。

动机:
    08 号脚本的消融是在"用同一份 LOOCV 既选 α 又报性能"的协议下得到的
    (真实 1.66 vs 随机 1.89, 提升 12.5%)。该协议存在选择乐观偏差,
    且各表型量纲悬殊使平均 MAE 被 inflor_height 主导, 结论不可靠。
    此处用尺度无关指标 (skill / R2) 与嵌套协议重测, 判断结论是否成立。

随机超图: 保持超边大小分布不变, 重新随机分配成员基因 (同 08 号脚本)。

输入:
    processed/aligned_data.npz, processed/node2row.npz
    AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/ablation_honest.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
from scipy import stats as sps
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026
N_PERM = int(os.environ.get("N_PERM", "8"))
N_BANDS, TOP_K = 10, 20
ALPHAS = np.logspace(-3, 4, 12)


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


def build_random_H(H_real, seed):
    rng = np.random.default_rng(seed)
    d_e = np.asarray(H_real.sum(axis=0)).ravel().astype(int)
    n_genes = H_real.shape[0]
    rl, cl = [], []
    for e in range(len(d_e)):
        g = rng.choice(n_genes, size=d_e[e], replace=False)
        rl.append(g); cl.append(np.full(d_e[e], e))
    Hr = sp.coo_matrix((np.ones(sum(len(x) for x in rl)),
                        (np.concatenate(rl), np.concatenate(cl))),
                       shape=H_real.shape).tocsr()
    Hr.sum_duplicates(); Hr.data[:] = 1.0
    return Hr


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
    o = np.argsort(lam)[::-1]
    return A, np.maximum(lam[o], 0.0), V[:, o]


def features_from(H, X, annotated):
    A, lam, V = spectral_decompose(H)
    ann_pos = np.where(annotated)[0]
    ok = lam > 1e-10
    n_comp = len(lam)
    n_samp = X.shape[0]
    X_hat = np.zeros((n_samp, n_comp))
    for i in range(n_samp):
        x = X[i].copy(); x[ann_pos] -= x[ann_pos].mean()
        X_hat[i, ok] = (V[:, ok].T @ (A.T @ x)) / np.sqrt(lam[ok])

    energies = X_hat ** 2
    bs = n_comp // N_BANDS
    band = np.zeros((n_samp, N_BANDS))
    for b in range(N_BANDS):
        s = b * bs
        e = n_comp if b == N_BANDS - 1 else (b + 1) * bs
        band[:, b] = energies[:, s:e].sum(axis=1)
    top = np.argsort(energies.mean(0))[::-1][:TOP_K]
    st = np.zeros((n_samp, 8))
    for i in range(n_samp):
        en = energies[i]; tot = en.sum()
        if tot <= 0:
            continue
        st[i, 0] = float((en * lam).sum() / tot / lam.max())
        st[i, 1] = float(tot ** 2 / (en ** 2).sum())
        srt = np.sort(en[ok]); n = len(srt)
        st[i, 2] = float(1 - 2 * (np.cumsum(srt).sum() - .5 * tot) / (n * tot))
        desc = np.cumsum(srt[::-1]) / tot
        st[i, 3] = float(desc[9]); st[i, 4] = float(desc[49]); st[i, 5] = float(desc[99])
        st[i, 6] = max(tot - float(en[ok].sum()), 0.) / tot
        st[i, 7] = float(np.searchsorted(desc, .9) + 1)
    F = np.hstack([band, X_hat[:, top], st])
    smooth = st[:, 0].mean()
    mu, sd = F.mean(0), F.std(0)
    Z = (F - mu) / np.where(sd > 0, sd, 1.0)
    return Z @ Z.T, smooth


def gram_centered(Gs):
    r = Gs.mean(1, keepdims=True); c = Gs.mean(0, keepdims=True)
    return Gs - r - c + Gs.mean()


def ridge_pred(G_tr, y_tr, g_te, alpha):
    k = len(y_tr)
    Kc = gram_centered(G_tr)
    coef = np.linalg.solve(Kc + alpha * np.eye(k), y_tr - y_tr.mean())
    return float((g_te - G_tr.mean(0, keepdims=True)) @ coef) + y_tr.mean()


def inner_loo(G_tr, y_tr, alpha):
    """在训练折内部再做留一 (全部使用 0..k-1 局部索引, 不与外层索引混用)"""
    k = len(y_tr)
    out = np.zeros(k)
    for j in range(k):
        T = np.setdiff1d(np.arange(k), [j])
        out[j] = ridge_pred(G_tr[np.ix_(T, T)], y_tr[T],
                            G_tr[j, T][None, :], alpha)
    return out


def nested_loocv(G, y):
    """
    严格嵌套 LOOCV: 外层留出 i, 内层在剩余样本上逐 α 做留一选 α, 再预测 i。
    Gt 始终以局部索引 0..k-1 访问, 避免索引空间混用。
    """
    n = len(y)
    idx = np.arange(n)
    preds = np.zeros(n)
    for i in range(n):
        S = np.setdiff1d(idx, [i])
        Gt, yt = G[np.ix_(S, S)], y[S]
        best_a, best_e = ALPHAS[0], np.inf
        for a in ALPHAS:
            e = mean_absolute_error(yt, inner_loo(Gt, yt, a))
            if e < best_e:
                best_e, best_a = e, a
        preds[i] = ridge_pred(Gt, yt, G[i, S][None, :], best_a)
    return preds


def score(y, preds, smooth, tag):
    mae = mean_absolute_error(y, preds)
    mae0 = mean_absolute_error(y, np.full(len(y), y.mean()))
    r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r = float(np.corrcoef(y, preds)[0, 1]) if np.std(preds) > 1e-12 else 0.0
    return dict(hypergraph=tag, smoothness=smooth, phenotype=None,
                mae=mae, skill=1 - mae / mae0, r2=r2, r=r, n=len(y))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X, Y, Y_short = data["X_log1p"], data["Y"], data["Y_short"]
    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    annotated = spec["annotated"]

    H = build_H()
    rows = []

    print(">>> 真实超图")
    G_real, sm_real = features_from(H, X, annotated)
    print(f"    平均平滑度 = {sm_real:.4f}")
    for j, name in enumerate(Y_short):
        ok = ~np.isnan(Y[:, j]); yv = Y[:, j][ok]
        ii = np.where(ok)[0]
        pr = nested_loocv(G_real[np.ix_(ii, ii)], yv)
        s = score(yv, pr, sm_real, "real"); s["phenotype"] = name
        rows.append(s)
        print(f"    {name:16s} skill={s['skill']:+.3f} R2={s['r2']:+.3f} r={s['r']:+.3f}")

    for k in range(N_PERM):
        seed = SEED + k
        print(f">>> 随机超图 #{k + 1}/{N_PERM} (seed={seed})")
        Hr = build_random_H(H, seed)
        Gr, smr = features_from(Hr, X, annotated)
        print(f"    平均平滑度 = {smr:.4f}")
        for j, name in enumerate(Y_short):
            ok = ~np.isnan(Y[:, j]); yv = Y[:, j][ok]
            ii = np.where(ok)[0]
            pr = nested_loocv(Gr[np.ix_(ii, ii)], yv)
            s = score(yv, pr, smr, f"random_{seed}"); s["phenotype"] = name
            rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ablation_honest.csv"), index=False)

    print("\n" + "=" * 78)
    print("诚实协议下的超图结构消融")
    print("=" * 78)
    real = df[df.hypergraph == "real"]
    rand = df[df.hypergraph != "real"]
    for m in ("skill", "r2", "r"):
        print(f"  {m:6s}: 真实 {real[m].mean():+.4f}   "
              f"随机 {rand[m].mean():+.4f} ± {rand[m].std():.4f}")
    print(f"  平滑度: 真实 {real['smoothness'].mean():.4f}  "
          f"随机 {rand['smoothness'].mean():.4f}")

    # 逐表型配对检验: 真实 vs 每个随机超图
    print("\n  逐表型 真实 vs 随机(均值) 及配对符号检验:")
    for name in Y_short:
        rv = real.loc[real.phenotype == name, "skill"].iloc[0]
        dv = rand.loc[rand.phenotype == name, "skill"].values
        better = int((dv < rv).sum())
        print(f"    {name:16s} 真实 skill={rv:+.3f}  随机均值={dv.mean():+.3f}  "
              f"真实优于 {better}/{len(dv)} 个随机超图")

    # 平滑度差异检验 (超图结构本身的性质, 与预测无关)
    rsm = real["smoothness"].iloc[0]
    dsm = rand.groupby("hypergraph")["smoothness"].first().values
    t, p = sps.ttest_1samp(dsm, rsm)
    print(f"\n  平滑度 真实={rsm:.4f} vs 随机={dsm.mean():.4f}  "
          f"配对 t 检验 p={p:.2e}")
    print("\n判读: 若随机超图平滑度显著低于真实, 则'结构捕获生物学关系'成立;")
    print("      若预测 skill 无差异, 则'结构提升预测'不成立。两者需分开陈述。")


if __name__ == "__main__":
    main()
