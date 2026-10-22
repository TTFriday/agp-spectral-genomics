# AGP Spectral Genomics

Hypergraph spectral analysis of gene expression in Arabidopsis, with a scale-free evaluation of what those features can and cannot predict.

## Overview

This project implements Hypergraph Spectral Genomics (HSG): gene expression is treated as a signal on a Gene Ontology (GO)-derived hypergraph, and the normalized hypergraph Laplacian is diagonalized to obtain a frequency basis for that signal. The repository contains both the structural analysis (spectral smoothness of expression against permutation and random-hypergraph nulls) and the phenotype-prediction evaluation that accompanies it.

Scripts 08 and 09 contain the prediction pipeline as it was first built, scored by mean absolute error averaged over phenotypes. Scripts 10 to 17 are the audit of that pipeline and the analysis that replaces it; the audit changed the conclusions, so both are kept and run in order. See "Results" below.

## Requirements

- Python 3.10+
- NumPy >= 2.0
- SciPy >= 1.15
- scikit-learn >= 1.6
- pandas >= 2.0
- matplotlib >= 3.9

Install dependencies:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

## Data

This project uses the AGP (*Arabidopsis* Gene Expression and Phenotype) dataset:

- `arabidopsis_dataset_naive/gene_expression_fpkm.csv`, `gene_expression_tpm.csv`, `phenotype_data.csv`
- `dataframe/gene_sequences_added_v5.csv`, `dataframe/go_data.csv`
- `hypergraph/hyperedge_list.npy`

GO annotations are read from `external/gene2go.gz` (NCBI gene2go, taxon 3702; not redistributed here). Script 06 additionally compares against a TAIR GAF file, `external/tair.gaf.gz`.

The expected layout is:

```
data/
├── arabidopsis_dataset_naive/
│   ├── gene_expression_fpkm.csv
│   ├── gene_expression_tpm.csv
│   └── phenotype_data.csv
├── dataframe/
│   ├── gene_sequences_added_v5.csv
│   └── go_data.csv
└── hypergraph/
    └── hyperedge_list.npy
external/
├── gene2go.gz
└── tair.gaf.gz     # only needed by script 06
```

Place `data/` at the repository root, or point the `AGP_DATA` environment variable at the directory that holds it. `external/gene2go.gz` is always read from the repository root.

## Usage

Run the scripts in order.

### 1. Data preparation

```bash
python scripts/01_prepare_data.py
```

Aligns expression and phenotype data and writes `processed/aligned_data.npz`.

### 2. Hypergraph index mapping (once)

```bash
python scripts/03_crack_mapping.py
python scripts/04_marginal_matching.py
python scripts/05_source_scan.py
python scripts/06_go_matching_test.py
python scripts/07_assign_go_terms.py
```

These scripts document how the node and hyperedge indices in `hyperedge_list.npy` were identified. The last one writes `processed/node2row.npz`, the mapping from hypergraph node ids to expression rows that every later script reads, so this step is required before script 02.

### 3. Hypergraph spectral analysis

```bash
python scripts/02_hypergraph_spectral.py
```

Constructs the normalized hypergraph operator, performs the dual spectral decomposition, and analyzes the spectral properties of expression. Outputs:

- `processed/hypergraph_spectrum.npz`
- `results/fig_hypergraph_stats.png`
- `results/fig_eigenvalues.png`
- `results/fig_energy_spectrum.png`
- `results/spectral_summary.csv`

### 4. Phenotype prediction (absolute-MAE pipeline)

```bash
python scripts/08_predict_band_energy.py
```

Extracts band-energy features and performs Ridge/LASSO LOOCV prediction. Outputs:

- `results/prediction_results.csv`
- `results/prediction_features.csv`
- `results/fig_prediction.png`
- `results/fig_learning_curve.png`
- `results/fig_ablation_random.png`

### 5. Configuration search (absolute-MAE pipeline)

```bash
python scripts/09_optimize_prediction.py
```

Sweeps feature sets, models, phenotype normalization and outlier handling. Outputs:

- `results/optimization_results.csv`
- `results/fig_optimization.png`

### 6. Audit of the MAE-based numbers

```bash
python scripts/10_final_table.py
```

Reproduces the best configuration of script 09 and reports each phenotype's MAE next to
its own standard deviation and mean-predictor error. This is what exposes the scale
confound: a small MAE on a small-scale phenotype is not evidence of prediction.

### 7. Scale-free evaluation

```bash
python scripts/11_evaluate_honest.py      # nested LOOCV, skill/R^2, self-check vs scikit-learn
python scripts/12_ablation_honest.py      # real vs 8 size-matched random hypergraphs (N_PERM env)
python scripts/13_smoothness_test.py      # 200 gene-label permutations per sample
python scripts/14_spectral_representations.py  # 15 feature representations, same protocol
python scripts/15_baseline_reproduction.py     # random forest / gradient boosting / kNN
python scripts/16_make_honest_figures.py       # figures for the corrected analysis
python scripts/17_check_collapse.py            # checks the collapsed-cell diagnostic directly
```

Script 11 exposes the Gram-based ridge machinery used by 12, 14 and 15, and verifies it
against per-fold scikit-learn fits before anything is computed. Outputs land in
`results/` as `honest_evaluation.csv`, `ablation_honest.csv`, `smoothness_test.csv`,
`spectral_representations.csv` and `baseline_reproduction.csv`. Script 17 re-runs only the cells whose
leave-one-out correlation is near -1 and prints the standard deviation of their predictions relative to
the phenotype standard deviation; the values sit at 1/(n-1), confirming that those cells are constant
predictors rather than inverse ones.

## Output Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/                    # Analysis scripts, run in numeric order
├── data/                       # AGP input data (not distributed)
├── external/                   # gene2go.gz (not distributed)
├── processed/
│   ├── aligned_data.npz        # Expression matrix, phenotypes, sample ids
│   ├── hypergraph_spectrum.npz # Eigenvalues and eigenvectors
│   └── node2row.npz            # Gene node to expression row mapping
└── results/
    ├── fig_hypergraph_stats.png      # Degree and hyperedge size distributions
    ├── fig_eigenvalues.png           # Eigenvalue spectrum of the smoothing operator
    ├── fig_energy_spectrum.png       # Spectral energy, real vs shuffled
    ├── fig_smoothness_nulls.png      # Real vs permutation and random-hypergraph nulls
    ├── fig_prediction_skill.png      # Scale-free skill for every representation
    ├── fig_prediction.png            # Absolute-MAE pipeline (superseded)
    ├── fig_learning_curve.png        # Learning curve (superseded)
    ├── fig_ablation_random.png       # Real vs random hypergraph (superseded)
    ├── fig_optimization.png          # Optimization sweep (superseded)
    ├── spectral_summary.csv          # Per-sample spectral statistics
    ├── smoothness_test.csv           # Permutation null for spectral smoothness
    ├── honest_evaluation.csv         # Nested LOOCV, skill and R^2
    ├── ablation_honest.csv           # Real vs size-matched random hypergraphs
    ├── spectral_representations.csv  # 15 feature representations, same protocol
    ├── baseline_reproduction.csv     # Random forest / gradient boosting / kNN
    ├── prediction_results.csv        # Absolute-MAE results (superseded)
    ├── prediction_features.csv       # Extracted features
    └── optimization_results.csv      # All optimization configurations (superseded)
```

Outputs marked superseded come from scripts 08-09 and are kept because the audit in
scripts 10-17 compares against them.

## Method

1. **Hypergraph construction**: gene-GO incidence from NCBI gene2go (taxon 3702, all evidence codes); genes are nodes, GO terms are hyperedges
2. **Spectral decomposition**: symmetric normalization $A = D_v^{-1/2} H D_e^{-1/2}$; the spectrum of $AA^T$ is obtained from the smaller dual matrix $A^T A$
3. **Signal projection**: each sample's log1p FPKM vector, centered on annotated genes, is projected onto the spectral basis
4. **Feature extraction**: 38-dimensional summary (10 band energies + 20 signed top-energy coefficients + 8 spectral statistics)
5. **Evaluation**: ridge with strictly nested leave-one-out $\alpha$ selection, scored as skill $= 1 - \text{MAE}/\text{MAE}_{\text{mean predictor}}$, alongside random forest, gradient boosting and $k$NN controls

## Results

**Structural finding (holds up).** Expression is markedly smoother on the GO hypergraph than under gene-label permutation: mean smoothness 0.458 against 0.159 (2.88x), Wilcoxon signed-rank $p = 6 \times 10^{-8}$, and all 24 samples above their own permutation maximum over 200 permutations. The smoothing operator has rank 5,879 with 1,200 null modes. Energy is concentrated in the smooth end of the spectrum: the 50 most energetic components carry 49.6% of the total, and the 50 smoothest carry 39.0% against 0.8% after permutation. Replacing the annotated gene sets by size-matched random hyperedges drops smoothness to 0.251 (range 0.247-0.256 over eight random hypergraphs), so the effect is attributable to the content of the annotations rather than to their degree profile. Roughly 40% of signal energy (39.8%) lies outside the spectral span of the hypergraph, which bounds what any feature set built on it can describe.

**Prediction finding (does not hold up).** The band-energy pipeline of scripts 08-09 reported mean MAE 0.917 with four of five phenotypes below 0.3, and a 12.5% advantage of the real hypergraph over random ones. Both statements were artefacts: absolute MAE averaged over phenotypes whose scales differ by a factor of 250, $\alpha$ chosen on the evaluation data, and outlier deletion that changed which samples were evaluated. Under nested evaluation against each phenotype's own mean predictor, the 38-dimensional spectral features have mean skill $-0.22$ (mean $R^2 = -0.50$) and are beaten by 38 random Gaussian features ($-0.10$); the real hypergraph ($-0.22$) predicts worse than size-matched random hypergraphs (range $-0.13$ to $-0.04$, mean $-0.08$). Across nine feature-model pairs in the nonlinear baseline sweep the best mean skill is $-0.004$ (random forest on the 5,000 most variable genes); the only representation with a positive mean is ridge on five principal components ($+0.04$), which does not use the hypergraph. The full set of signed spectral coefficients recovers the weak inflorescence-height signal that raw expression carries ($+0.17$ for both), whereas the 38-d band-energy summary does not ($-0.02$), so the loss is in energy-domain summarization rather than in the GO basis.


