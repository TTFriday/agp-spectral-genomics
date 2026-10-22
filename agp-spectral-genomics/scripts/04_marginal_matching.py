# -*- coding: utf-8 -*-
"""
04_marginal_matching.py
用与索引排列无关的统计量 (边际分布) 逐步锁定超图索引映射。

Step 1  注释来源验证:
    超图 per-基因度数多重集  vs  gene2go per-基因 GO 计数多重集
    超图 per-超边大小多重集  vs  gene2go per-词条基因数多重集
Step 2  GO 排序锁定 (只用大小, 与基因排序解耦):
    对每种 GO 排序假设, 比较 "该排序下第 i 个词条的基因数" 与
    "超图第 i 条超边的大小" 的吻合率。
Step 3  基因排序锁定 (GO 排序已定时, 与 GO 排序解耦):
    对每种基因排序假设, 比较 "该假设下第 g 个基因的 GO 数" 与
    "超图节点 g 的度数" 的吻合率。

输入: external/gene2go.gz, AGP 数据, results/mapping_crack_report.csv 无关
输出: results/marginal_matching.csv, 控制台报告
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
    """taxon 3702 全量: GeneID → set(GO), term → set(GeneID), term 首现顺序。"""
    gene_ann, term_genes = {}, {}
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
                gid, go = int(cols[1]), cols[2]
                if "NOT" in cols[4].upper():
                    continue
                gene_ann.setdefault(gid, set()).add(go)
                term_genes.setdefault(go, set()).add(gid)
                if go not in seen:
                    seen.add(go)
                    term_order.append(go)
    except (EOFError, OSError):
        pass
    return gene_ann, term_genes, term_order


def name_from_go_data():
    go = pd.read_csv(os.path.join(AGP_DATA, "dataframe", "go_data.csv"), sep="\t")
    return go


def qmatch(a, b):
    """两个大小向量的匹配率: 按同一索引逐一比较的相对误差 <5% 的比例。"""
    if len(a) != len(b):
        return 0.0
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a - b) / np.maximum(b, 1) < 0.05))


def main():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    hg_gene, hg_go = pairs[0], pairs[1]

    seq = pd.read_csv(os.path.join(AGP_DATA, "dataframe",
                                   "gene_sequences_added_v5.csv"),
                      sep="\t", usecols=["idx", "ncbi_id", "name"])
    go_data = name_from_go_data()

    gene_ann, term_genes, term_order = parse_gene2go()
    print(f"gene2go (3702): 基因 {len(gene_ann)}, 词条 {len(term_genes)}")

    ncbi_ids = seq["ncbi_id"].astype(int).to_numpy()
    # 限制到我们的基因宇宙
    our = set(ncbi_ids.tolist())
    gene_ann = {g: ts for g, ts in gene_ann.items() if g in our}
    term_genes = {t: gs & our for t, gs in term_genes.items()}
    term_genes = {t: gs for t, gs in term_genes.items() if gs}
    print(f"限制到 38286 基因宇宙: 基因 {len(gene_ann)}, 词条 {len(term_genes)}")

    # ---------- Step 1a: per-基因度数多重集 ----------
    d_v_hg = np.bincount(hg_gene, minlength=38311)
    d_v_hg_nz = np.sort(d_v_hg[d_v_hg > 0])
    d_v_ann = np.sort(np.array([len(ts) for ts in gene_ann.values()]))
    print(f"\n[Step1a] 基因度数: 超图 {len(d_v_hg_nz)} 个 "
          f"(中位 {np.median(d_v_hg_nz):.0f}, 最大 {d_v_hg_nz.max()}) vs "
          f"gene2go {len(d_v_ann)} 个 (中位 {np.median(d_v_ann):.0f}, "
          f"最大 {d_v_ann.max()})")
    n = min(len(d_v_hg_nz), len(d_v_ann))
    if n:
        rel = np.abs(d_v_hg_nz[-n:] - d_v_ann[-n:]) / np.maximum(d_v_ann[-n:], 1)
        print(f"        尾部对齐相对误差中位: {np.median(rel):.3f}")

    # ---------- Step 1b: per-超边大小多重集 ----------
    d_e_hg = np.sort(np.bincount(hg_go, minlength=int(hg_go.max()) + 1))
    d_e_hg = d_e_hg[d_e_hg > 0]
    d_e_ann = np.sort(np.array([len(gs) for gs in term_genes.values()]))
    print(f"[Step1b] 超边大小: 超图 {len(d_e_hg)} 条 "
          f"(中位 {np.median(d_e_hg):.0f}, 最大 {d_e_hg.max()}) vs "
          f"gene2go {len(d_e_ann)} 条 (中位 {np.median(d_e_ann):.0f}, "
          f"最大 {d_e_ann.max()})")
    n = min(len(d_e_hg), len(d_e_ann))
    rel = np.abs(d_e_hg[-n:] - d_e_ann[-n:]) / np.maximum(d_e_ann[-n:], 1)
    print(f"        尾部对齐相对误差中位: {np.median(rel):.3f}")

    # ---------- Step 2: GO 排序假设 × 超边大小吻合率 ----------
    terms_used = sorted(term_genes)
    go_data_order = {gid: i for i, gid in enumerate(go_data["read_id"])}
    go_data_name = go_data.set_index("read_id")["name"].to_dict()
    g2g_name = {}
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
                    g2g_name.setdefault(cols[2], cols[5])
    except (EOFError, OSError):
        pass

    hg_de = np.bincount(hg_go, minlength=int(hg_go.max()) + 1)
    hg_dv = np.bincount(hg_gene, minlength=38311)

    go_candidates = {
        "G1_sorted_by_GO_ID": sorted(terms_used),
        "G2_first_appearance_g2g": [t for t in term_order if t in set(terms_used)],
        "G3_name_from_gene2go": sorted(terms_used,
                                       key=lambda t: g2g_name.get(t, t).lower()),
        "G4_name_from_go_data": sorted(terms_used,
                                       key=lambda t: str(go_data_name.get(t, t)).lower()),
        "G5_go_data_row_order": sorted(terms_used,
                                       key=lambda t: go_data_order.get(t, 10 ** 9)),
        "G6_go_data_idx_direct": None,   # 词条 = go_data 第 0..7078 行, 只统计有注释的
    }

    rows = []
    print("\n[Step2] GO 排序假设 × 超边大小吻合率 (同索引比较):")
    for label, go_list in go_candidates.items():
        if label == "G6_go_data_idx_direct":
            sub = go_data.iloc[:7079]
            sizes = np.array([len(term_genes.get(t, set())) for t in sub["read_id"]])
            m = qmatch(hg_de[:7079], sizes)
            # 也比较整体大小多重集
            multi = qmatch(np.sort(hg_de[:7079]), np.sort(sizes))
            rows.append({"stage": "GO", "hypothesis": label,
                         "index_match": m, "multiset_match": multi})
            print(f"  {label:26s}: 同索引吻合率={m:.3f}, 多重集吻合率={multi:.3f}")
            continue
        sizes = np.array([len(term_genes[t]) for t in go_list])
        m = qmatch(hg_de[:len(sizes)], sizes)
        multi = qmatch(np.sort(hg_de[:len(sizes)]), np.sort(sizes))
        rows.append({"stage": "GO", "hypothesis": label,
                     "index_match": m, "multiset_match": multi})
        print(f"  {label:26s}: 同索引吻合率={m:.3f}, 多重集吻合率={multi:.3f}")

    # ---------- Step 3: 基因排序假设 × 节点度数吻合率 ----------
    print("\n[Step3] 基因排序假设 × 节点度数吻合率:")
    idx2ncbi = dict(zip(seq["idx"].to_numpy(), ncbi_ids))
    pos2ncbi = dict(enumerate(ncbi_ids))
    names_sorted = seq.sort_values("name", key=lambda s: s.astype(str))
    name2ncbi = dict(zip(names_sorted["name"].astype(str), names_sorted["ncbi_id"].astype(int)))

    def gene_match(mapping):
        """mapping: 节点id → ncbi_id (只对存在的节点)。比较度数。"""
        nodes = np.array(sorted(mapping))
        got = np.array([len(gene_ann.get(mapping[g], set())) for g in nodes])
        ref = hg_dv[nodes]
        ok = (got == ref).mean()
        rel5 = (np.abs(got - ref) / np.maximum(ref, 1) < 0.05).mean()
        return ok, rel5

    gene_candidates = {
        "A_idx_value": idx2ncbi,
        "B_row_position": pos2ncbi,
        "C_alpha_by_name": name2ncbi,
    }
    for label, mapping in gene_candidates.items():
        ok, rel5 = gene_match(mapping)
        rows.append({"stage": "gene", "hypothesis": label,
                     "index_match": ok, "multiset_match": rel5})
        print(f"  {label:26s}: 完全吻合率={ok:.3f}, 5%容差吻合率={rel5:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "marginal_matching.csv"),
                              index=False)


if __name__ == "__main__":
    main()
