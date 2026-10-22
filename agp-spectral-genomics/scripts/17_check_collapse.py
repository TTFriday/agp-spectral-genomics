# -*- coding: utf-8 -*-
"""
17_check_collapse.py -- 校验稿件中"塌缩单元格"的论断:
|r| > 0.99 的 (表示, 表型) 组合, 其 std(pred)/std(y) 是否确实 < 5%。
只跑被点名的组合, 不重跑全表。
"""
import os, importlib.util as ilu
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
SEED = 2026

sp = ilu.spec_from_file_location("h11", os.path.join(HERE, "11_evaluate_honest.py"))
H11 = ilu.module_from_spec(sp); sp.loader.exec_module(H11)

d = pd.read_csv(os.path.join(BASE, "results", "spectral_representations.csv"))
d = d[(d.model != "mean_predictor") & (d.r.abs() > 0.99)]
print("待校验单元格数 =", len(d))

data = np.load(os.path.join(BASE, "processed", "aligned_data.npz"))
X, Y, Y_short = data["X_log1p"], data["Y"], data["Y_short"]
spec = np.load(os.path.join(BASE, "processed", "hypergraph_spectrum.npz"))
annotated = spec["annotated"]
H = H11.build_H()
A, lam, V = H11.spectral_cached(H)
X_hat = H11.project_all(A, lam, V, X, annotated)
Xs = H11.StandardScalerLike(X[:, np.where(annotated)[0]])
nz = int((lam > 1e-10).sum())

reps = {}
for K in (5, 10, 20, 50):
    reps[f"HGsmooth_coef_K{K}"] = X_hat[:, :K]
for K in (10, 50):
    reps[f"HGhighfreq_coef_K{K}"] = X_hat[:, nz - K:nz]
for K in (5, 10, 20):
    reps[f"PCA_{K}d"] = PCA(n_components=K, random_state=SEED).fit_transform(Xs)
reps["random_38d"] = np.random.default_rng(SEED).standard_normal((X.shape[0], 38))

grams = {t: H11.gram_of(F) for t, F in reps.items()}
rows = []
for _, rec in d.iterrows():
    ph, tag = rec.phenotype, rec.model
    i = list(Y_short).index(ph)
    y = Y[:, i]; ok = ~np.isnan(y); y_v = y[ok]
    idx = np.where(ok)[0]
    preds = H11.nested_loocv(grams[tag][np.ix_(idx, idx)], y_v)
    ratio = float(np.std(preds) / np.std(y_v, ddof=0))
    rows.append((ph, tag, rec.r, rec.skill, ratio, np.corrcoef(preds, y_v)[0, 1]))
    print(f"  {ph:16s} {tag:20s} r={rec.r:+.4f} ratio={ratio:.5f}")

t = pd.DataFrame(rows, columns=["ph", "model", "r_csv", "skill", "ratio", "r_re"])
print("\n最大 std(pred)/std(y) = %.5f  (阈值 0.05)" % t.ratio.max())
print("全部低于 5%%:", bool((t.ratio < 0.05).all()))
