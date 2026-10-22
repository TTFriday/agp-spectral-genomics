# -*- coding: utf-8 -*-
"""
07_assign_go_terms.py
用位集 Jaccard 将 7,079 条超边全量指派到 gene2go GO 词条,
恢复 GO 索引排序, 并锁定 超图节点 → 表达矩阵行 的映射。

输出:
    processed/node2row.npz          节点id → 表达行 的映射
    results/hyperedge_assignment.csv  超边 → GO 词条指派 (含 Jaccard)
    results/go_order_analysis.csv    GO 排序假设检验
    控制台: 锚点检验 + 排序模式分析
"""

import os
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
G2G_PATH = os.path.join(BASE, "external", "gene2go.gz")
PROCESSED = os.path.join(BASE, "processed")
RESULTS = os.path.join(BASE, "results")


def main():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    hg_gene, hg_go = pairs[0], pairs[1]

    seq = pd.read_csv(os.path.join(AGP_DATA, "dataframe",
                                   "gene_sequences_added_v5.csv"),
                      sep="\t", usecols=["idx", "ncbi_id", "name"])

    # ---------- 1. 节点 → 表达行 ----------
    idx_vals = seq["idx"].to_numpy()
    node_ids = np.unique(hg_gene)
    pos_of_idx = np.searchsorted(idx_vals, node_ids)
    found = (pos_of_idx < len(idx_vals)) & (idx_vals[np.minimum(pos_of_idx, len(idx_vals) - 1)] == node_ids)
    n_missing = int((~found).sum())
    print(f"超图节点 {len(node_ids)} 个, 在表达 idx 空间中缺失 {n_missing} 个")
    if n_missing:
        print("  缺失节点(落在 idx 缺口):", node_ids[~found][:30])
    node2row = dict(zip(node_ids[found].tolist(),
                        pos_of_idx[found].tolist()))
    np.savez_compressed(os.path.join(PROCESSED, "node2row.npz"),
                        node_ids=np.array(sorted(node2row)),
                        row_of_node=np.array([node2row[n] for n in sorted(node2row)]))
    print(f"已保存 node2row: {len(node2row)} 个节点 → 表达行")

    # ---------- 2. gene2go 词条 → 基因位集 ----------
    n_total = len(seq)
    ncbi2pos = dict(zip(seq["ncbi_id"].astype(int), range(n_total)))
    idx2name = dict(zip(seq["idx"], seq["name"].astype(str)))

    term_mask, term_ids = {}, []
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
                p = ncbi2pos.get(gid)
                if p is None:
                    continue
                if go not in term_mask:
                    term_mask[go] = 0
                    term_ids.append(go)
                term_mask[go] |= (1 << p)
    except (EOFError, OSError):
        pass
    print(f"gene2go: {len(term_ids)} 个词条覆盖我们的基因宇宙")

    term_sizes = {t: int(m.bit_count()) for t, m in term_mask.items()}
    term_first_seen = {t: i for i, t in enumerate(term_ids)}   # 基因升序首现顺序

    # ---------- 3. 超边位集 + 全量指派 ----------
    edge_mask, edge_sizes = {}, defaultdict(int)
    for g, t in zip(hg_gene.tolist(), hg_go.tolist()):
        p = node2row.get(g)
        if p is None:
            continue
        edge_mask[t] = edge_mask.get(t, 0) | (1 << p)
        edge_sizes[t] += 1
    edges = sorted(edge_mask)
    print(f"超边 {len(edges)} 条 (可映射节点部分)")

    rows = []
    for e in edges:
        em = edge_mask[e]
        es = em.bit_count()
        best_j, best_t = 0.0, None
        for t, tm in term_mask.items():
            ts = tm.bit_count()
            # 大小预过滤: 版本漂移下大小变化一般 <30%
            if ts < es * 0.65 or ts > es * 1.45:
                continue
            inter = (em & tm).bit_count()
            if inter == 0:
                continue
            jac = inter / (es + ts - inter)
            if jac > best_j:
                best_j, best_t = jac, t
        rows.append({"hyperedge": e, "size": es,
                     "assigned_go": best_t, "jaccard": best_j,
                     "term_size": term_sizes.get(best_t),
                     "term_first_seen": term_first_seen.get(best_t)})
    asg = pd.DataFrame(rows)
    asg.to_csv(os.path.join(RESULTS, "hyperedge_assignment.csv"), index=False)
    print(f"指派完成: 中位 Jaccard = {asg['jaccard'].median():.3f}, "
          f"J>0.9 占比 {(asg['jaccard'] > 0.9).mean():.1%}, "
          f"未匹配 {(asg['assigned_go'].isna()).sum()} 条")

    # ---------- 4. GO 排序假设检验 ----------
    hi = asg[asg["jaccard"] > 0.9].copy()
    print(f"\n高置信指派 (J>0.9): {len(hi)} 条, 用于排序分析")
    go_data = pd.read_csv(os.path.join(AGP_DATA, "dataframe", "go_data.csv"), sep="\t")
    go_row = {g: i for i, g in enumerate(go_data["read_id"])}

    hi["rank_by_go_id"] = hi["assigned_go"].rank(method="first")
    hi["rank_by_first_seen"] = hi["term_first_seen"].rank(method="first")
    hi["rank_by_go_data"] = hi["assigned_go"].map(
        lambda g: go_row.get(g, np.nan)).rank(method="first")
    hi["rank_by_name"] = hi["assigned_go"].map(
        lambda g: str(go_data.set_index('read_id')['name'].get(g, g))).rank(method="first")

    from scipy.stats import spearmanr
    res = []
    for col in ("rank_by_go_id", "rank_by_first_seen",
                "rank_by_go_data", "rank_by_name"):
        sub = hi[["hyperedge", col]].dropna()
        rho, p = spearmanr(sub["hyperedge"], sub[col])
        res.append({"ordering": col, "spearman": rho, "n": len(sub)})
        print(f"  超边id vs {col:22s}: Spearman ρ = {rho:.4f} (n={len(sub)})")
    pd.DataFrame(res).to_csv(os.path.join(RESULTS, "go_order_analysis.csv"),
                             index=False)

    # ---------- 5. 锚点检验 ----------
    print("\n锚点检验 (MIPS 家族):")
    name2idx = {v: k for k, v in idx2name.items()}
    mips_nodes = [name2idx[n] for n in ("MIPS1", "MIPS2", "MIPS3", "IPS1")
                  if n in name2idx]
    mips_pos = [node2row[n] for n in mips_nodes]
    mips_mask = 0
    for p in mips_pos:
        mips_mask |= (1 << p)
    shared = [t for t, m in term_mask.items()
              if (m & mips_mask).bit_count() == len(mips_pos)]
    print(f"  gene2go 中 4 个 MIPS 基因共享词条: {shared}")
    for e in (734, 1095, 1210):
        r = asg[asg["hyperedge"] == e]
        if len(r):
            print(f"  超边 {e} → {r.iloc[0]['assigned_go']} (J={r.iloc[0]['jaccard']:.3f})")

    # ---------- 6. 光合作用生物学核查 ----------
    print("\n生物学核查: 'photosynthesis, light harvesting' 词条匹配哪些超边:")
    for t, m in term_mask.items():
        if t in ("GO:0009768", "GO:0009769", "GO:0019684", "GO:0019685"):
            es = m.bit_count()
            best_j, best_e = 0.0, None
            for e, em in edge_mask.items():
                ee = em.bit_count()
                if ee < es * 0.65 or ee > es * 1.45:
                    continue
                inter = (em & m).bit_count()
                if inter == 0:
                    continue
                jac = inter / (es + ee - inter)
                if jac > best_j:
                    best_j, best_e = jac, e
            if best_e is not None:
                em = edge_mask[best_e]
                genes = [idx2name[g] for g, r in node2row.items()
                         if em >> r & 1][:10]
                print(f"  {t} → 超边 {best_e} (J={best_j:.3f}), "
                      f"示例基因: {', '.join(genes)}")


if __name__ == "__main__":
    main()
