# -*- coding: utf-8 -*-
"""
14_spectral_representations.py

预注册式对比: 在同一个严格嵌套 LOOCV 协议下, 检验"是否存在任何一种超图谱谱表示
能够保留原始表达 / PCA 所保留的预测信号"。

动机 (来自 11 号脚本的结论):
    08/09 号脚本所报告的 38 维频带能量特征在严格协议下不优于恒预测均值;
    而 PCA-5d 与原始表达在 inflor_height / leaf_temp_diff 上确有显著相关。
    频带特征是"能量统计量"(丢掉符号与方向), 因此无法判断失败来自
    (a) 超图谱基本身不含信号, 还是
    (b) 特征构造方式把信号压掉了。
    本脚本加入保留符号的"最平滑 K 个谱系数"(等价于以 GO 超图谱为基的 PCA),
    以及谱低通/加权变体, 直接区分 (a) 与 (b)。

协议纪律:
    - 特征标准化仅使用无监督全局统计量 (不看标签), 不构成标签泄漏。
    - α 一律在外折内部用内层 LOOCV 选择 (嵌套), 不做全局择优。
    - 所有表示、所有 K、所有表型全部输出, 不挑选。判据先写死:
      skill > 0 且 R2 > 0 才算优于均值基线。
    - 因存在多种表示 x 多个 K 的多重比较, 结论仅以"家族层面是否一致"为准,
      单个 p 值不作为宣称依据。

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz
    AGP/data/hypergraph/hyperedge_list.npy
输出:
    results/spectral_representations.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
from sklearn.decomposition import PCA

import importlib.util as ilu

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
RESULTS = os.path.join(BASE, "results")
SEED = 2026


def _load_h11():
    """复用 11 号脚本中已通过与 sklearn 逐折一致性自检的 Gram 机器。"""
    path = os.path.join(HERE, "11_evaluate_honest.py")
    spec = ilu.spec_from_file_location("honest_eval", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H11 = _load_h11()


def main():
    data = np.load(os.path.join(BASE, "processed", "aligned_data.npz"))
    X, Y, Y_short = data["X_log1p"], data["Y"], data["Y_short"]
    spec = np.load(os.path.join(BASE, "processed", "hypergraph_spectrum.npz"))
    annotated = spec["annotated"]

    H = H11.build_H()
    A, lam, V = H11.spectral_cached(H)
    X_hat = H11.project_all(A, lam, V, X, annotated)   # 24 x 7079 谱系数(有符号)
    Xs = H11.StandardScalerLike(X[:, np.where(annotated)[0]])

    # 谱系数按 λ 从最平滑到最振荡排序 (spectral_decompose 已降序)
    reps = {}

    # --- (1) 最平滑 K 个谱系数: 以 GO 超图谱为基的降维, 保留符号 ---
    for K in (5, 10, 20, 50):
        reps[f"HGsmooth_coef_K{K}"] = X_hat[:, :K]

    # --- (2) 谱低通加权: 系数乘以 λ^s, 相当于对信号施加平滑滤波器 ---
    for s in (1.0, 2.0):
        reps[f"HGlowpass_full_pow{s:g}"] = X_hat * (lam ** s)[None, :]

    # --- (3) 最振荡 K 个系数 (作为方向性对照: 若高频才含信号, 应在此体现) ---
    nz = int((lam > 1e-10).sum())
    for K in (10, 50):
        reps[f"HGhighfreq_coef_K{K}"] = X_hat[:, nz - K:nz]

    # --- (4) 原 38 维频带能量特征 (纯统计量, 无符号) ---
    reps["HSG_38d_bandstats"] = H11.extract_features(X_hat, lam)
    reps["HSG_bands_only"] = H11.extract_features(X_hat, lam)[:, :H11.N_BANDS]

    # --- (5) 参照组: 数据驱动 PCA / 原始表达 / 纯噪声 ---
    for K in (5, 10, 20):
        reps[f"PCA_{K}d"] = PCA(n_components=K, random_state=SEED).fit_transform(Xs)
    reps["raw_expression"] = Xs
    reps["random_38d"] = np.random.default_rng(SEED).standard_normal((X.shape[0], 38))

    grams = {t: H11.gram_of(F) for t, F in reps.items()}

    print("=" * 100)
    print(f"表示数 = {len(grams)}, 表型数 = {len(Y_short)}, 协议 = 严格嵌套 LOOCV")
    print("=" * 100)

    rows = []
    for i, ph in enumerate(Y_short):
        y = Y[:, i]
        ok = ~np.isnan(y)
        y_v = y[ok]
        idx = np.where(ok)[0]
        base = H11.summarize(y_v, np.full(len(y_v), y_v.mean()), "mean_predictor")
        base["phenotype"] = ph
        rows.append(base)
        for tag, G in grams.items():
            sub = G[np.ix_(idx, idx)]
            pr = H11.nested_loocv(sub, y_v)
            s = H11.summarize(y_v, pr, tag)
            s["phenotype"] = ph
            s["n_feat"] = reps[tag].shape[1]
            rows.append(s)
        print(f"  {ph} 完成")

    df = pd.DataFrame(rows)
    cols = ["phenotype", "model", "n_feat", "n", "mae", "mae_mean_baseline",
            "mae_over_std", "skill", "r2", "r", "p", "pred_std_ratio", "std_y"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(RESULTS, "spectral_representations.csv"), index=False)

    print("\n" + "=" * 100)
    print("跨表型平均 (按表示; 排除均值基线)")
    print("=" * 100)
    mm = df[df.model != "mean_predictor"].groupby("model").agg(
        n_feat=("n_feat", "first"),
        mean_mae_over_std=("mae_over_std", "mean"),
        mean_skill=("skill", "mean"),
        mean_r2=("r2", "mean"),
        n_skill_pos=("skill", lambda s: int((s > 0).sum())),
        n_r2_pos=("r2", lambda s: int((s > 0).sum())),
        best_r=("r", "max"),
    ).sort_values("mean_skill", ascending=False)
    print(mm.to_string())

    print("\n" + "=" * 100)
    print("逐表型: 每个表型上 skill 最好的 5 个表示")
    print("=" * 100)
    for ph in Y_short:
        sub = df[(df.phenotype == ph) & (df.model != "mean_predictor")]
        print(f"\n【{ph}】 std={sub['std_y'].iloc[0]:.4f}  "
              f"均值基线 MAE={sub['mae_mean_baseline'].iloc[0]:.4f}")
        print(sub.nlargest(5, "skill")[["model", "n_feat", "mae", "skill",
                                        "r2", "r", "p"]].to_string(index=False))

    print("\n注: 本表含多重比较, 单个 p 值不足以支撑结论; 以家族层面一致性为准。")

    # --- 塌缩诊断: |r|>0.99 的单元格是否确为"近似恒预测" ---
    print("\n" + "=" * 100)
    print("塌缩诊断: |r| > 0.99 的单元格的 std(pred)/std(y)")
    print("=" * 100)
    nz = df[(df.model != "mean_predictor") & (df.r.abs() > 0.99)]
    if len(nz):
        print(nz[["phenotype", "model", "r", "skill", "pred_std_ratio"]]
              .to_string(index=False))
        print(f"\n最大 std(pred)/std(y) = {nz.pred_std_ratio.max():.6f} "
              f"(共 {len(nz)} 个单元格)")
    else:
        print("无 |r|>0.99 的单元格")


if __name__ == "__main__":
    main()
