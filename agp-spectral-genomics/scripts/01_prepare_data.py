# -*- coding: utf-8 -*-
"""
01_prepare_data.py
对齐 AGP 数据集中的基因表达矩阵与表型数据。

输入:
    AGP/data/arabidopsis_dataset_naive/gene_expression_fpkm.csv  (38286 基因 x 24 样本)
    AGP/data/arabidopsis_dataset_naive/gene_expression_tpm.csv   (38286 基因 x 24 样本)
    AGP/data/arabidopsis_dataset_naive/phenotype_data.csv        (77 植株 x 42 表型)

输出 (processed/):
    aligned_data.npz : X_fpkm, X_tpm, X_log1p (24 x 38286), Y (24 x n_targets),
                       样本名、基因索引、表型列名、基因型标签
    sample_alignment.csv : 24 个配对样本的元数据
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("AGP_DATA",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
NAIVE_DIR = os.path.join(DATA_DIR, "arabidopsis_dataset_naive")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "processed")

# 论文使用的 5 个目标表型 (Serna-Aguilera et al., 2025)
TARGET_PHENOTYPES = [
    "solidity measured by Suxing",
    "Inflorescence height (mm) on 6/24/2024",
    "Light AVG FvP_over_FmP ",
    "Light AVG qL",
    "Light AVG Leaf Temp Differential",
]
# 简短名称, 用于图表和结果文件
TARGET_SHORT = ["solidity", "inflor_height", "FvFm", "qL", "leaf_temp_diff"]


def genotype_of(sample_name: str) -> str:
    """样本名形如 Col_1 / fad7ex1ex2_24, 下划线前即基因型。"""
    return sample_name.rsplit("_", 1)[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 读取表达矩阵 ----------
    fpkm = pd.read_csv(os.path.join(NAIVE_DIR, "gene_expression_fpkm.csv"), index_col=0)
    tpm = pd.read_csv(os.path.join(NAIVE_DIR, "gene_expression_tpm.csv"), index_col=0)
    print(f"FPKM 矩阵: {fpkm.shape}, TPM 矩阵: {tpm.shape}")

    # FPKM 文件除样本列外还带一列冗余的 idx, 去掉
    if "idx" in fpkm.columns:
        fpkm = fpkm.drop(columns=["idx"])

    # TPM 文件列名带 _fpkm 后缀但实为 TPM, 统一去掉后缀
    tpm.columns = [c.replace("_fpkm", "") for c in tpm.columns]

    # 列顺序对齐检查
    assert list(fpkm.columns) == list(tpm.columns), "FPKM 与 TPM 样本列不一致"
    samples = list(fpkm.columns)
    n_genes = fpkm.shape[0]
    print(f"样本数: {len(samples)}, 基因数: {n_genes}")

    # ---------- 读取表型 ----------
    pheno = pd.read_csv(os.path.join(NAIVE_DIR, "phenotype_data.csv"))
    pheno = pheno.rename(columns={"filename for image": "sample"})
    print(f"表型表: {pheno.shape}, 样本标识列 sample")

    # ---------- 按样本名对齐 ----------
    expr_samples = set(samples)
    pheno_samples = set(pheno["sample"])
    paired = [s for s in samples if s in pheno_samples]
    print(f"表达样本 {len(expr_samples)} 个, 表型样本 {len(pheno_samples)} 个, "
          f"配对样本 {len(paired)} 个")
    missing = expr_samples - pheno_samples
    if missing:
        print("警告: 以下表达样本在表型表中找不到:", sorted(missing))

    pheno_idx = pheno.set_index("sample").loc[paired]

    # 检查目标表型列是否存在
    for col in TARGET_PHENOTYPES:
        if col not in pheno_idx.columns:
            raise KeyError(f"目标表型列不存在: {col!r}")

    Y = pheno_idx[TARGET_PHENOTYPES].to_numpy(dtype=np.float64)
    n_missing_per_target = np.isnan(Y).sum(axis=0)
    print("每个目标表型的缺失样本数:",
          dict(zip(TARGET_SHORT, n_missing_per_target.tolist())))

    # ---------- 整理 X 矩阵 (样本 x 基因) ----------
    X_fpkm = fpkm[paired].to_numpy(dtype=np.float64).T          # (24, 38286)
    X_tpm = tpm[paired].to_numpy(dtype=np.float64).T            # (24, 38286)
    X_log1p = np.log1p(X_fpkm)

    genotypes = [genotype_of(s) for s in paired]
    print("配对样本基因型分布:", pd.Series(genotypes).value_counts().to_dict())

    # ---------- 保存 ----------
    gene_idx = fpkm.index.to_numpy()  # 基因行索引 (0..38285), 与超图节点索引同源
    np.savez_compressed(
        os.path.join(OUT_DIR, "aligned_data.npz"),
        X_fpkm=X_fpkm, X_tpm=X_tpm, X_log1p=X_log1p,
        Y=Y, Y_short=np.array(TARGET_SHORT),
        Y_cols=np.array(TARGET_PHENOTYPES),
        samples=np.array(paired), genotypes=np.array(genotypes),
        gene_idx=gene_idx,
    )

    meta = pd.DataFrame({
        "sample": paired, "genotype": genotypes,
        "pheno_row": [list(pheno["sample"]).index(s) for s in paired],
    })
    for j, short in enumerate(TARGET_SHORT):
        meta[short] = Y[:, j]
    meta.to_csv(os.path.join(OUT_DIR, "sample_alignment.csv"), index=False)

    print(f"\n已保存: {os.path.join(OUT_DIR, 'aligned_data.npz')}")
    print(f"X 矩阵形状: {X_fpkm.shape}, Y 矩阵形状: {Y.shape}")


if __name__ == "__main__":
    main()
