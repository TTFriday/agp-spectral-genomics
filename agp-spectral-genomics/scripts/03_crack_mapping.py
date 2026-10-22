# -*- coding: utf-8 -*-
"""
03_crack_mapping.py  (gene2go 版)
破译 hyperedge_list.npy 中 (基因节点, GO超边) 索引的真实含义。

结构线索:
    - 基因节点索引: 28,679 个, 分布在 0..38,026 (非紧致) → 完整基因列表空间的子集
    - GO 超边索引: 0..7,078 紧致完整 → 独立的 7,079 词条列表
    - gene_sequences_added_v5.csv 按 NCBI GeneID 升序 (idx 有缺口, 首个缺口 25,756)

方法: 用 NCBI gene2go (taxon 3702) 重建 GeneID→GO 关联, 枚举候选索引假设,
与实际配对集合计算 Jaccard / recall / precision。

输入:
    AGP/data/hypergraph/hyperedge_list.npy
    AGP/data/dataframe/gene_sequences_added_v5.csv
    AGP/data/dataframe/go_data.csv
    external/gene2go.gz

输出: results/mapping_crack_report.csv, 控制台报告
"""

import os
import gzip
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
G2G_PATH = os.path.join(BASE, "external", "gene2go.gz")
RESULTS = os.path.join(BASE, "results")


def parse_gene2go():
    """taxon 3702: GeneID → set(GO:ID)。返回 (ann, 首次出现顺序)。
    文件按 tax_id 升序, 3702 之后的内容可以忽略; 文件可能不完整(截断),
    需容忍 EOF。"""
    ann = {}
    term_order, seen = [], set()
    try:
        with gzip.open(G2G_PATH, "rt", errors="ignore") as f:
            f.readline()
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 8:
                    continue
                tax = int(cols[0])
                if tax > 3702:
                    break
                if tax != 3702:
                    continue
                gene_id = int(cols[1])
                go_id = cols[2]
                qualifier = cols[4]
                if "NOT" in qualifier.upper():
                    continue
                ann.setdefault(gene_id, set()).add(go_id)
                if go_id not in seen:
                    seen.add(go_id)
                    term_order.append(go_id)
    except (EOFError, OSError):
        pass                    # 截断文件: 3702 块已完整读入
    return ann, term_order


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    hg_gene, hg_go = pairs[0], pairs[1]
    seq = pd.read_csv(os.path.join(AGP_DATA, "dataframe",
                                   "gene_sequences_added_v5.csv"),
                      sep="\t", usecols=["idx", "ncbi_id", "name"])

    # ---- gene2go 注释, 限制到我们的基因 ----
    ann_all, g2g_term_order = parse_gene2go()
    print(f"gene2go (3702): {len(ann_all)} 个基因有注释")

    ncbi_ids = seq["ncbi_id"].astype(int)
    idx_vals = seq["idx"].to_numpy()
    row_pos = np.arange(len(seq))

    gene_ann_by_idx, gene_ann_by_row = {}, {}
    for i, (iv, gid) in enumerate(zip(idx_vals, ncbi_ids)):
        ts = ann_all.get(gid)
        if ts:
            gene_ann_by_idx[iv] = ts
            gene_ann_by_row[i] = ts
    print(f"我们的 {len(seq)} 个基因中 gene2go 注释了 {len(gene_ann_by_idx)} 个 "
          f"(超图基因节点 {len(np.unique(hg_gene))})")

    terms_used = sorted({t for ts in gene_ann_by_idx.values() for t in ts})
    print(f"注释到的 GO 词条: {len(terms_used)} (超图 GO 节点 7,079)")

    go_data = pd.read_csv(os.path.join(AGP_DATA, "dataframe", "go_data.csv"), sep="\t")
    g2g_term_name = {}
    try:
        with gzip.open(G2G_PATH, "rt", errors="ignore") as f:
            f.readline()
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 8:
                    continue
                tax = int(cols[0])
                if tax > 3702:
                    break
                if tax == 3702:
                    g2g_term_name.setdefault(cols[2], cols[5])
    except (EOFError, OSError):
        pass
    go_data_name = go_data.set_index("read_id")["name"].to_dict()
    go_data_order = {gid: i for i, gid in enumerate(go_data["read_id"])}

    candidate_orders = {
        "G1_sorted_by_GO_ID": sorted(terms_used),
        "G2_first_appearance_g2g": [t for t in g2g_term_order if t in set(terms_used)],
        "G3_name_from_gene2go": sorted(terms_used,
                                       key=lambda t: g2g_term_name.get(t, t).lower()),
        "G4_name_from_go_data": sorted(terms_used,
                                       key=lambda t: str(go_data_name.get(t, t)).lower()),
        "G5_go_data_row_order": sorted(terms_used,
                                       key=lambda t: go_data_order.get(t, 10 ** 9)),
    }

    actual = set(zip(hg_gene.tolist(), hg_go.tolist()))
    print(f"实际配对数: {len(actual)}\n")

    rows = []
    for go_label, go_list in candidate_orders.items():
        term2idx = {t: i for i, t in enumerate(go_list)}
        for gene_label, gene_ann in (("A_idx_value", gene_ann_by_idx),
                                     ("B_row_position", gene_ann_by_row)):
            pred = set()
            for g, ts in gene_ann.items():
                for t in ts:
                    if t in term2idx:
                        pred.add((g, term2idx[t]))
            inter = len(actual & pred)
            jac = inter / len(actual | pred)
            rec = inter / len(actual)
            prec = inter / max(len(pred), 1)
            rows.append({"go_hypothesis": go_label, "gene_hypothesis": gene_label,
                         "n_go_terms": len(go_list), "n_pred_pairs": len(pred),
                         "jaccard": jac, "recall": rec, "precision": prec})
            print(f"{go_label:26s} + {gene_label:14s}: J={jac:.3f} "
                  f"R={rec:.3f} P={prec:.3f}")

    report = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    report.to_csv(os.path.join(RESULTS, "mapping_crack_report.csv"), index=False)

    # ---- 最佳假设的差异分析 ----
    best = report.iloc[0]
    print(f"\n最佳: {best['go_hypothesis']} + {best['gene_hypothesis']} "
          f"(Jaccard {best['jaccard']:.3f})")
    go_list = candidate_orders[best["go_hypothesis"]]
    term2idx = {t: i for i, t in enumerate(go_list)}
    gene_ann = (gene_ann_by_idx if best["gene_hypothesis"] == "A_idx_value"
                else gene_ann_by_row)
    pred = set()
    for g, ts in gene_ann.items():
        for t in ts:
            if t in term2idx:
                pred.add((g, term2idx[t]))
    inv_term = {i: t for i, t in enumerate(go_list)}
    only_hg = sorted(actual - pred)
    only_pred = sorted(pred - actual)
    print(f"仅在超图中: {len(only_hg)}, 仅在重建中: {len(only_pred)}")
    idx2name = seq.set_index("idx")["name"]
    for g, t in only_hg[:8]:
        term = inv_term.get(t, "?")
        print(f"  超图独有: 基因 {g} ({idx2name.get(g, '?')}) GO {term} "
              f"({g2g_term_name.get(term, '?')})")
    for g, t in only_pred[:8]:
        term = inv_term.get(t, "?")
        print(f"  重建独有: 基因 {g} ({idx2name.get(g, '?')}) GO {term} "
              f"({g2g_term_name.get(term, '?')})")


if __name__ == "__main__":
    main()
