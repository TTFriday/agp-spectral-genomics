# -*- coding: utf-8 -*-
"""
13_smoothness_test.py
严格检验"基因表达信号在 GO 超图上平滑"这一核心结论 (替代 02 号脚本的口头描述)。

对照设计:
    对每个样本, 保持其表达值多重集不变, 仅在超图覆盖的基因之间随机重排
    (gene-label permutation)。这打断了"哪个基因表达多少"与"该基因属于哪些 GO 项"
    的对应关系, 但保留表达分布与超图结构, 因而是平滑性的恰当零假设。

统计:
    1. 逐样本配对: 每个样本的真实平滑度 vs 它自己的 N_PERM 个置换 (Wilcoxon 符号秩)
    2. 合并:      Mann-Whitney U 与 Welch t
    3. 经验 p 值:  置换平滑度 >= 真实均值 的比例 (Monte-Carlo)

效率:
    朴素实现需对每个置换做一次 7079x7079 稠密 matvec (24 x 200 次), 过慢。
    这里把一批置换的目标信号拼成矩阵, 谱投影合并为一次 BLAS gemm。

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz,
    processed/node2row.npz, AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/smoothness_test.csv
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as sps

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")
SEED = 2026
N_PERM = 200          # 每样本置换次数
CHUNK = 100           # 每次 gemm 处理的信号条数


def build_A():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    m = np.load(os.path.join(PROCESSED, "node2row.npz"))
    n2r = dict(zip(m["node_ids"].tolist(), m["row_of_node"].tolist()))
    rows = np.array([n2r[g] for g in pairs[0].tolist()])
    cols = pairs[1]
    H = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(38286, int(cols.max()) + 1)).tocsr()
    H.sum_duplicates(); H.data[:] = 1.0
    d_v = np.asarray(H.sum(1)).ravel()
    d_e = np.asarray(H.sum(0)).ravel()
    idv = np.where(d_v > 0, 1.0 / np.sqrt(np.maximum(d_v, 1)), 0.0)
    ide = 1.0 / np.sqrt(np.maximum(d_e, 1))
    return (sp.diags(idv) @ H @ sp.diags(ide)).tocsr(), d_v


def smoothness_from_coeff(q, lam):
    """q = 谱系数 (已含 1/sqrt(lam) 归一), 返回平滑度 ∈[0,1]"""
    e = q ** 2
    tot = e.sum()
    if tot <= 0:
        return np.nan
    return float((e * lam).sum() / tot / lam.max())


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = np.load(os.path.join(PROCESSED, "aligned_data.npz"))
    X, samples = data["X_log1p"], data["samples"]

    spec = np.load(os.path.join(PROCESSED, "hypergraph_spectrum.npz"))
    lam = spec["lam"]
    V = spec["V"].astype(np.float64)
    annotated = spec["annotated"]

    A, _ = build_A()
    ap = np.where(annotated)[0]
    ok = lam > 1e-10
    Vk = V[:, ok]
    lamk = lam[ok]

    def project(S):
        """S: (38286 x B) 信号矩阵 -> (B,) 平滑度。一次稀疏乘 + 一次 gemm。"""
        Z = A.T @ S                                   # (n_go x B)
        Q = (Vk.T @ Z) / np.sqrt(lamk)[:, None]       # (n_comp x B)
        e = Q ** 2
        tot = e.sum(0)
        return (e * lamk[:, None]).sum(0) / tot / lam.max()

    # ---- 真实平滑度 ----
    real = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i].copy(); x[ap] -= x[ap].mean()
        real[i] = float(project(x[:, None])[0])

    # ---- 逐样本配对置换 ----
    perm = np.zeros((X.shape[0], N_PERM))
    for i in range(X.shape[0]):
        x = X[i].copy(); x[ap] -= x[ap].mean()
        xa = x[ap]
        done = 0
        while done < N_PERM:
            b = min(CHUNK, N_PERM - done)
            S = np.zeros((38286, b))
            for j in range(b):
                S[ap, j] = rng.permutation(xa)
            perm[i, done:done + b] = project(S)
            done += b
        print(f"  {samples[i]}: 真实={real[i]:.4f} 置换均值={perm[i].mean():.4f}")

    # ---- 统计检验 ----
    rows = []
    ratio = real.mean() / perm.mean()
    # (1) 逐样本配对 Wilcoxon (真实 vs 该样本的置换分布中位数)
    w = sps.wilcoxon(real, np.median(perm, axis=1), alternative="greater")
    # (2) 合并 Mann-Whitney
    u = sps.mannwhitneyu(real, perm.ravel(), alternative="greater")
    # (3) Welch t
    t = sps.ttest_ind(real, perm.ravel(), equal_var=False, alternative="greater")
    # (4) 经验 p 值: 随机抽一个样本的随机置换 >= 真实均值的概率
    emp = float((perm.ravel() >= real.mean()).mean())

    print("\n" + "=" * 70)
    print(f"真实   平滑度 = {real.mean():.4f} ± {real.std(ddof=0):.4f}  (n={len(real)} 样本)")
    print(f"置换   平滑度 = {perm.mean():.4f} ± {perm.std(ddof=0):.4f}  "
          f"(n={perm.size} = {len(real)}×{N_PERM})")
    print(f"效应量       = {ratio:.2f}x")
    print("-" * 70)
    print(f"逐样本配对 Wilcoxon  W={w.statistic:.1f}  p={w.pvalue:.3e}")
    print(f"合并 Mann-Whitney U  U={u.statistic:.1f}  p={u.pvalue:.3e}")
    print(f"合并 Welch t         t={t.statistic:.1f}  p={t.pvalue:.3e}")
    print(f"经验 p 值 (MC)       p={emp:.4f}  (上限 1/{perm.size})")
    print(f"全部 24 个样本的真实值均超过其置换分布最大值: "
          f"{int((real > perm.max(axis=1)).sum())}/{len(real)}")

    for i in range(len(real)):
        rows.append(dict(sample=samples[i], real=real[i],
                         perm_mean=perm[i].mean(), perm_max=perm[i].max(),
                         exceed_max=bool(real[i] > perm[i].max()),
                         ratio=real[i] / perm[i].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "smoothness_test.csv"),
                              index=False)

    summary = dict(ratio=ratio, real_mean=real.mean(), perm_mean=perm.mean(),
                   wilcoxon_p=w.pvalue, mwu_p=u.pvalue, welch_p=t.pvalue,
                   empirical_p=emp,
                   n_samples_exceed_permmax=int((real > perm.max(axis=1)).sum()))
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS, "smoothness_test_summary.csv"), index=False)
    print("\n已保存: results/smoothness_test.csv, smoothness_test_summary.csv")


if __name__ == "__main__":
    main()
