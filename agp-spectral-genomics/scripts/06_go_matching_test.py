# -*- coding: utf-8 -*-
"""
06_go_matching_test.py
在候选注释源中为超图超边寻找最优 GO 词条匹配 (Jaccard),
判定作者真正使用的注释来源。

前提假设 (已由 MIPS 家族锚点检验支持):
    基因节点 id == gene_sequences_added_v5.csv 的 idx 值 (NCBI GeneID 序)

来源:
    gene2go_all / gene2go_no_IEA     (NCBI, 本地)
    tair.gaf_current                 (PANTHER 重绘版, 本地)
输出: results/go_matching_test.csv
"""

import os
import gzip
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AGP_DATA = os.environ.get("AGP_DATA",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
G2G_PATH = os.path.join(BASE, "external", "gene2go.gz")
GAF_PATH = os.path.join(BASE, "external", "tair.gaf.gz")
RESULTS = os.path.join(BASE, "results")
SEED = 7


def load_sources(seq):
    """返回 {源名: (term→AGI集合, gene→词条集合)}"""
    idx2name = dict(zip(seq["idx"], seq["name"].astype(str)))
    our_names = set(idx2name.values()) - {n for n in idx2name.values()
                                          if n.startswith("38") and len(n) > 11}
    our_names = {n for n in our_names if n.startswith("AT")}
    sources = {}

    # ---- gene2go ----
    for label, drop in (("gene2go_all", set()), ("gene2go_noIEA", {"IEA"})):
        term2genes, gene2terms = {}, {}
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
                    if tax != 3702 or cols[3] in drop:
                        continue
                    gid, go = int(cols[1]), cols[2]
                    term2genes.setdefault(go, set()).add(gid)
                    gene2terms.setdefault(gid, set()).add(go)
        except (EOFError, OSError):
            pass
        our_ids = set(seq["ncbi_id"].astype(int))
        term2genes = {t: gs for t, gs in term2genes.items() if gs & our_ids}
        term2genes = {t: gs & our_ids for t, gs in term2genes.items()}
        sources[label] = (term2genes, gene2terms)

    # ---- tair.gaf (AGI 名) ----
    term2genes, gene2terms = {}, {}
    with gzip.open(GAF_PATH, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 15:
                continue
            if "NOT" in cols[3].split("|"):
                continue
            agi, go = cols[1], cols[4]
            if not agi.startswith("AT") or agi not in our_names:
                continue
            term2genes.setdefault(go, set()).add(agi)
            gene2terms.setdefault(agi, set()).add(go)
    sources["tair_gaf_current"] = (term2genes, gene2terms)
    return sources, idx2name


def main():
    pairs = np.load(os.path.join(AGP_DATA, "hypergraph", "hyperedge_list.npy"))
    hg_gene, hg_go = pairs[0], pairs[1]

    seq = pd.read_csv(os.path.join(AGP_DATA, "dataframe",
                                   "gene_sequences_added_v5.csv"),
                      sep="\t", usecols=["idx", "ncbi_id", "name"])
    sources, idx2name = load_sources(seq)

    # 每条超边的基因节点集合
    from collections import defaultdict
    edge_genes = defaultdict(set)
    for g, t in zip(hg_gene.tolist(), hg_go.tolist()):
        edge_genes[t].add(g)

    rng = np.random.default_rng(SEED)
    cand_edges = [e for e, gs in edge_genes.items() if 2 <= len(gs) <= 40]
    sample = rng.choice(cand_edges, size=min(300, len(cand_edges)), replace=False)

    rows = []
    for src_label, (term2genes, gene2terms) in sources.items():
        # gene 维度统一: gene2go 用 ncbi_id, gaf 用 AGI 名 → 都转 AGI 名
        if src_label.startswith("gene2go"):
            id2name = dict(zip(seq["ncbi_id"].astype(int), seq["name"].astype(str)))
            term2agi = {t: {id2name.get(g, "?") for g in gs} for t, gs in term2genes.items()}
        else:
            term2agi = term2genes
        # 反向索引: AGI → 词条
        agi2terms = defaultdict(set)
        for t, gs in term2agi.items():
            for a in gs:
                agi2terms[a].add(t)

        best_js = []
        for e in sample:
            agi_set = {idx2name.get(g, "?") for g in edge_genes[e]}
            if "?" in agi_set or not agi_set:
                continue
            cands = set()
            for a in agi_set:
                cands |= agi2terms.get(a, set())
            best_j, best_t = 0.0, None
            for t in cands:
                ts = term2agi.get(t, set())
                if not ts:
                    continue
                inter = len(agi_set & ts)
                jac = inter / len(agi_set | ts)
                if jac > best_j:
                    best_j, best_t = jac, t
            best_js.append((best_j, best_t, e, tuple(sorted(agi_set))))
        arr = np.array([j for j, *_ in best_js])
        rows.append({"source": src_label,
                     "median_best_jaccard": float(np.median(arr)),
                     "frac_j_gt_0.5": float((arr > 0.5).mean()),
                     "frac_j_gt_0.9": float((arr > 0.9).mean()),
                     "n_sample": len(arr)})
        print(f"{src_label:16s}: 中位最优 Jaccard = {np.median(arr):.3f}, "
              f">0.5 占比 {(arr > 0.5).mean():.2%}, >0.9 占比 {(arr > 0.9).mean():.2%}")
        for j, t, e, agis in sorted(best_js, reverse=True)[:3]:
            print(f"    例: 超边 {e} (大小 {len(edge_genes[e])}) → {t} (J={j:.2f})")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "go_matching_test.csv"),
                              index=False)


if __name__ == "__main__":
    main()
