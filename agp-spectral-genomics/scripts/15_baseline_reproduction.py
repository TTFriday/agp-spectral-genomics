# -*- coding: utf-8 -*-
"""
15_baseline_reproduction.py

在完全相同的严格协议下检验非线性模型, 回答一个决定论文定位的问题:
    在 n=24 的 AGP 数据集上, 是否有任何方法能够优于"恒预测样本均值"?

背景:
    AGP 数据集论文报告 RF MAE=0.577, HGNN+VAE MAE=0.553, 但其表型是否归一化、
    划分方式与截距口径均未公开, 绝对 MAE 无法直接比较。
    skill = 1 - MAE/MAE_mean_predictor 对表型的任意线性缩放不变,
    因此可以把所有方法放到同一把尺子上。

做法:
    - 岭回归(嵌套选 α)的结果直接取自 14 号脚本, 不重复计算。
    - 这里补充三个不调参的非线性模型: 随机森林、梯度提升、3 近邻。
      逐折重新拟合, 超参数固定, 因此不存在选择乐观偏差。
    - 原始表达维度过高(2.4 万), 树模型按"总方差前 5000 的基因"截断。
      该筛选是无监督的(不看表型), 不构成标签泄漏。

输入:
    processed/aligned_data.npz, processed/hypergraph_spectrum.npz
    results/spectral_representations.csv (岭回归结果)
输出:
    results/baseline_reproduction.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

import importlib.util as ilu

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
RESULTS = os.path.join(BASE, "results")
SEED = 2026
TOP_GENES = 5000


def _load_h11():
    path = os.path.join(HERE, "11_evaluate_honest.py")
    spec = ilu.spec_from_file_location("honest_eval", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H11 = _load_h11()


def plain_loocv(make_model, F, y):
    """逐折重新拟合的 LOOCV (不调参, 无标签泄漏)"""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        S = np.setdiff1d(np.arange(n), [i])
        mdl = make_model()
        mdl.fit(F[S], y[S])
        preds[i] = mdl.predict(F[i:i + 1])[0]
    return preds


def drop_lowvar(F, tol=1e-8):
    sd = F.std(0)
    return F[:, sd > tol * float(sd.max())]


def main():
    data = np.load(os.path.join(BASE, "processed", "aligned_data.npz"))
    X, Y, Y_short = data["X_log1p"], data["Y"], data["Y_short"]
    spec = np.load(os.path.join(BASE, "processed", "hypergraph_spectrum.npz"))
    annotated = spec["annotated"]
    ann_pos = np.where(annotated)[0]

    H = H11.build_H()
    A, lam, V = H11.spectral_cached(H)
    X_hat = H11.project_all(A, lam, V, X, annotated)

    Xs = H11.StandardScalerLike(X[:, ann_pos])
    # 无监督地按方差排序取前 TOP_GENES 个基因, 供树模型使用
    order = np.argsort(Xs.var(0))[::-1][:TOP_GENES]
    feats = {
        "raw_top5000": Xs[:, order],
        "HSG_38d": H11.extract_features(X_hat, lam),
        "PCA_5d": PCA(n_components=5, random_state=SEED).fit_transform(Xs),
    }
    models = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.1, random_state=SEED),
        "kNN_k3": lambda: KNeighborsRegressor(n_neighbors=3),
    }

    rows = []
    for tag, F in feats.items():
        Fc = drop_lowvar(F)
        for i, ph in enumerate(Y_short):
            y = Y[:, i]
            ok = ~np.isnan(y)
            yv = y[ok]
            idx = np.where(ok)[0]
            sub_F = Fc[idx]
            r = H11.summarize(yv, np.full(len(yv), yv.mean()), "mean_predictor")
            r.update(phenotype=ph, feat="-", n_feat=0)
            rows.append(r)
            for mtag, mk in models.items():
                pr = plain_loocv(mk, sub_F, yv)
                r = H11.summarize(yv, pr, mtag)
                r.update(phenotype=ph, feat=tag, n_feat=Fc.shape[1])
                rows.append(r)
                print(f"  {tag} x {ph} x {mtag}  skill={r['skill']:+.3f}")

    df = pd.DataFrame(rows)
    df["protocol"] = "plain_LOOCV_fixed_hyperparams"

    # 岭回归(嵌套选 α)的结果在 spectral_representations.csv 中, 两表在论文里并列呈现,
    # 此处不合并: 二者的特征集合不完全一致 (树模型用了方差前 5000 基因)。

    cols = ["feat", "phenotype", "model", "protocol", "n_feat", "n", "mae",
            "mae_mean_baseline", "mae_over_std", "skill", "r2", "r", "p",
            "pred_std_ratio", "std_y"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(RESULTS, "baseline_reproduction.csv"), index=False)

    print("\n" + "=" * 100)
    print("按 特征 x 模型 平均 (排除均值基线)")
    print("=" * 100)
    mm = df[df.model != "mean_predictor"].groupby(["feat", "model"]).agg(
        mean_skill=("skill", "mean"), mean_r2=("r2", "mean"),
        n_skill_pos=("skill", lambda s: int((s > 0).sum())),
        best_r=("r", "max")).sort_values("mean_skill", ascending=False)
    print(mm.round(3).to_string())

    print("\n" + "=" * 100)
    print("逐表型 skill")
    print("=" * 100)
    print(df[df.model != "mean_predictor"].pivot_table(
        index=["feat", "model"], columns="phenotype", values="skill").round(3).to_string())

    pos = df[(df.model != "mean_predictor") & (df.skill > 0)]
    print("\n总体: 优于恒预测均值的 (特征 x 模型 x 表型) 组合数 = %d / %d"
          % (len(pos), len(df[df.model != "mean_predictor"])))
    if len(pos):
        print(pos.sort_values("skill", ascending=False)
              [["feat", "model", "phenotype", "skill", "r", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
