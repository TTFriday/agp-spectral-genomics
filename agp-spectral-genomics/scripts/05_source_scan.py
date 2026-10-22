# -*- coding: utf-8 -*-
"""
05_source_scan.py
系统扫描注释来源与 GO 排序假设, 找到与 hyperedge_list.npy 最吻合的组合。

维度:
    证据代码过滤: all / no_IEA / no_IEA_ISS / experimental_only
    GO 排序:      G6_direct (go_data idx 直接用)
                  G6_compacted (go_data 行序, 压缩掉无注释词条)
                  G5_go_data_row_order (词条级排序, 同 G6 但以 read_id 排)
                  G1_sorted_by_GO_ID
    基因排序:     A_idx_value (固定, 已由边际匹配确认为最优)

指标: 配对集 Jaccard / recall / precision, 基因度数吻合率, 超边大小吻合率
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

EVIDENCE_ALL = None
EVIDENCE_NO_IEA = {"IEA"}
EVIDENCE_NO_IEA_ISS = {"IEA", "ISS"}
EVIDENCE_EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
                         "HTP", "HDA", "HEP", "HGI", "IC", "BA"}
FILTERS = {
    "all": EVIDENCE_ALL,
    "no_IEA": EVIDENCE_NO_IEA,
    "no_IEA_ISS": EVIDENCE_NO_IEA_ISS,
    "experimental_only": EVIDENCE_EXPERIMENTAL,
}


def parse_gene2go_kept(drop):
    """GeneID → set(GO), 词条→基因集合, 按 drop 集合剔除证据代码。"""
    gene_ann, term_genes = {}, {}
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
                if drop and cols[3] in drop:
                    continue
                gid, go = int(cols[1]), cols[2]
                gene_ann.setdefault(gid, set()).add(go)
                term_genes.setdefault(go, set()).add(gid)
    except (EOFError, OSError):
        pass
    return gene_ann, term_genes


def main():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    hg_gene, hg_go = pairs[0], pairs[1]
    actual = set(zip(hg_gene.tolist(), hg_go.tolist()))
    hg_dv = np.bincount(hg_gene, minlength=38311)
    hg_de = np.bincount(hg_go, minlength=int(hg_go.max()) + 1)

    seq = pd.read_csv(os.path.join(AGP_DATA, "dataframe",
                                   "gene_sequences_added_v5.csv"),
                      sep="\t", usecols=["idx", "ncbi_id", "name"])
    idx2ncbi = dict(zip(seq["idx"].to_numpy(), seq["ncbi_id"].astype(int)))
    our_ids = set(idx2ncbi.values())
    go_data = pd.read_csv(os.path.join(AGP_DATA, "dataframe", "go_data.csv"), sep="\t")
    go_rows = go_data["read_id"].tolist()

    rows = []
    for f_label, drop in FILTERS.items():
        gene_ann, term_genes = parse_gene2go_kept(drop)
        # 限制到我们的基因宇宙
        gene_ann = {g: ts for g, ts in gene_ann.items() if g in our_ids}
        term_genes = {t: {g for g in gs if g in our_ids}
                      for t, gs in term_genes.items()}
        term_genes = {t: gs for t, gs in term_genes.items() if gs}
        n_genes_ann = len(gene_ann)

        # 基因度数吻合率 (假设 A)
        nodes = np.array(sorted(idx2ncbi))
        got = np.array([len(gene_ann.get(idx2ncbi[g], set())) for g in nodes])
        deg_match = float((got == hg_dv[nodes]).mean())

        for g_label in ("G6_direct", "G6_compacted", "G1_sorted_by_GO_ID"):
            if g_label == "G6_direct":
                go_list = go_rows
            elif g_label == "G6_compacted":
                have = [t for t in go_rows if t in term_genes]
                go_list = have
            else:
                go_list = sorted(term_genes)
            term2idx = {t: i for i, t in enumerate(go_list)}

            pred = set()
            for g, ts in gene_ann.items():
                for t in ts:
                    j = term2idx.get(t)
                    if j is not None and j <= 38285:
                        pred.add((g, j))
            inter = len(actual & pred)
            jac = inter / len(actual | pred)
            rec = inter / len(actual)
            prec = inter / max(len(pred), 1)
            # 超边大小同索引吻合率 (仅前 7079; 假设词条不足时截短)
            m = min(len(go_list), 7079)
            sizes = np.array([len(term_genes.get(t, set())) for t in go_list[:m]])
            size_match = float((np.abs(sizes - hg_de[:m]) /
                                np.maximum(hg_de[:m], 1) < 0.05).mean())
            rows.append({"filter": f_label, "go_order": g_label,
                         "n_genes_ann": n_genes_ann, "n_terms": len(go_list),
                         "deg_match": deg_match, "size_match": size_match,
                         "jaccard": jac, "recall": rec, "precision": prec})
            print(f"{f_label:20s} + {g_label:14s}: 基因{n_genes_ann} 词条{len(go_list):5d} "
                  f"deg={deg_match:.3f} size={size_match:.3f} "
                  f"J={jac:.3f} R={rec:.3f} P={prec:.3f}")

    rep = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    rep.to_csv(os.path.join(RESULTS, "source_scan.csv"), index=False)
    print("\nTop 5:")
    print(rep.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
