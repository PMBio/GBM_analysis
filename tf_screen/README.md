# TF screen analysis

Analysis pipeline for the 55-TF gain-of-function screen in patient-derived GB cell lines, accompanying the manuscript

> *(paper citation — to be updated on publication)*

This subdirectory of the [`PMBio/GBM_analysis`](https://github.com/PMBio/GBM_analysis) repository contains all code used to go from cleaned, guide-assigned single-cell AnnData to the per-TF effect tables and differential-expression results that feed the paper's TF-screen figures (Fig. 3, Extended Data Fig. 5). Final figure layout was done in Affinity Designer from these tables — the scripts here produce the underlying data, not the publication-ready panels.

For the corresponding experimental description, see the manuscript supplement, section 3.4.

---

## Pipeline overview

```
SCREEN_ANNDATA  +  ATLAS_RNA_ANNDATA  +  atlas annotations  +  scDoRI topic genes
                              │
                              ▼
      ┌────────────────────────────────────────────────┐
      │ 1. preprocessing/                              │
      │    build_joint_anndata.py                      │
      │    run_harmony.py                              │
      │    qc_figures.py                               │
      └────────────────────────────────────────────────┘
                              │
                              ▼
      ┌────────────────────────────────────────────────┐
      │ 2. classifiers/                                │
      │    train_state_classifier.py  (5 coarse,       │
      │                                9 fine states)  │
      │    train_topic_classifier.py  (19 topics)      │
      └────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ 3. tf_       │  │ 4. state_    │  │ 5. dose_     │
     │   expression/│  │    testing/  │  │    response/ │
     │              │  │              │  │              │
     │ tf_self_     │  │ state_       │  │ build_       │
     │  expression  │  │  testing     │  │  metacells   │
     │ control_p10_ │  │ topic_       │  │ correlation_ │
     │  validation  │  │  testing     │  │  analysis    │
     │              │  │ proportion_  │  │              │
     │              │  │  testing     │  │              │
     └──────────────┘  └──────────────┘  └──────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                  ┌────────────────────────────┐
                  │ 6. differential_expression/│
                  │    wilcoxon_de.py          │
                  └────────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────────┐
                  │ 7. downstream/             │
                  │    gsea_state_signatures   │
                  │    high_confidence_drivers │
                  │    mechanism_decomposition │
                  │                  (stubs)   │
                  └────────────────────────────┘
```

Steps 1–6 are fully open and run as a stand-alone pipeline. Step 7 is a placeholder for analyses currently maintained separately; see `downstream/README.md`.

---

## Repository layout

```
tf_screen/
├── README.md                         ← this file
├── config.template.py                ← copy to config.py, edit input paths
├── __init__.py
├── utils/
│   ├── __init__.py
│   └── io.py                         ← shared helpers (norm, FC/p10 selection)
├── preprocessing/                    ← step 1
├── classifiers/                      ← step 2
├── tf_expression/                    ← step 3
├── state_testing/                    ← step 4
├── dose_response/                    ← step 5
├── differential_expression/          ← step 6
├── downstream/                       ← step 7 (stubs)
├── scripts/                          ← LSF cluster submission examples
└── docs/
    ├── data_availability.md
    └── pipeline_overview.md
```

Every numbered step has its own `README.md` describing what it produces, what it expects as input, and how to run it. Read those first.

---

## What this repository produces

Quick map of where each script's output lives. Paths below are relative to the `OUTPUT_DIR` you set in `config.py`.

### Step 1 — `preprocessing/`

| Script | Produces |
|---|---|
| `build_joint_anndata.py` | `joint_anndata/joint_gbm_oe_anndata.h5ad` — the joint AnnData of atlas (exclusive-topic cells) + screen cells on a shared feature space. Used by every downstream step. |
| `run_harmony.py` | `harmony/harmony_embeddings.npy` — 200-PC Harmony-corrected embedding, row-aligned to the joint AnnData. |
| `qc_figures.py` | `qc_figures/*.{pdf,png}` — per-cell RNA QC panels and guide-assignment diagnostics (data behind ED Fig. 5a–d). |

### Step 2 — `classifiers/`

| Script | Produces |
|---|---|
| `train_state_classifier.py` | `models/clf_celltype_{fine,coarse}.pkl` (+ scaler + label encoder), plus per-cell `prob_ct_*` and `prob_coarse_*` columns added to the joint AnnData. Also `models/state_classifier_metrics.csv`. |
| `train_topic_classifier.py` | `models/clf_topic.pkl` (+ scaler + label encoder), per-cell `prob_topic_*` columns added to the joint AnnData. Also `models/topic_classifier_metrics.csv`. |

### Step 3 — `tf_expression/`

| Script | Produces |
|---|---|
| `tf_self_expression.py` | `tf_expression/tf_expression_results.csv` — per (TF, cell line) and pooled: arithmetic-mean and geometric-mean log2FC of the TF's own expression, Sarle's bimodality coefficient, Mann–Whitney U with BH-FDR. Source of the OE-vs-Ctrl induction summary in the paper. |
| `control_p10_validation.py` | `control_state_analysis/control_state_shifts.csv` — per (TF, state) coarse-state composition shift between all controls and the p10-filtered subset. Confirms p10 doesn't bias state composition (data behind ED Fig. 5c). |

### Step 4 — `state_testing/`

| Script | Produces |
|---|---|
| `state_testing.py` | `state_testing/<strategy>/{fine_grained,coarse}_effects.csv` for each of 16 FC × control-percentile filtering strategies. Each row is one (TF, state, cell line) with t-test, Wilcoxon, permutation p-values + FDRs, Cohen's d, and a `sig_consensus` flag. **Headline strategy is `FC0.5_p10`.** Source of the TF × state heatmap in Fig. 3c. |
| `topic_testing.py` | `topic_testing/<strategy>/topic_effects.csv` — same as above but for the 16 tested topics. Source of the TF × topic scatter in Fig. 3d. |
| `proportion_testing.py` | `proportion_testing/<strategy>/proportion_lrt.csv` (per-TF multinomial LRT) + `proportion_fisher.csv` (per-TF-per-state Fisher's exact). Source of the OR > 1.5 "black-box" overlay annotation in Fig. 3c. |

### Step 5 — `dose_response/`

| Script | Produces |
|---|---|
| `build_metacells.py` | `metacells/metacells.h5ad` — ~2,700 Leiden metacells with aggregated expression and state/topic probabilities. |
| `correlation_analysis.py` | `correlation/correlation_{combined,oe_only,control_only}.csv` — per (TF, state) Spearman/Pearson + permutation p-value + BH-FDR across the three populations. The within-OE numbers are the dose-response data in the paper text. |

### Step 6 — `differential_expression/`

| Script | Produces |
|---|---|
| `wilcoxon_de.py` | `differential_expression/<strategy>/<cell_line>__<TF>__DE.csv` (per-(TF, line) Wilcoxon tables with `log2FC`, `pvals_adj`, `signed_fdr`) + `intersection/*.csv` (genes significant in ≥2/3 lines with consistent direction). The `signed_fdr` column is the input to the GSEA stub in step 7. **The headline strategy is `all_all`** (all OE vs all Ctrl, no FC/p10 filter on the DE itself). |

### Step 7 — `downstream/` (stubs)

These are placeholders for the GSEA + high-confidence-driver + activator/repressor decomposition analyses. The production code is maintained separately and will be added in the next release. See `downstream/README.md`.

---

## Getting started

### 1. Clone

```bash
git clone git@github.com:PMBio/GBM_analysis.git
cd GBM_analysis
```

### 2. Set up a Python environment

The pipeline expects Python 3.10 and the package versions used to produce the published figures:

```
scanpy==1.9.5       anndata==0.9.2       harmonypy==0.2.0
scikit-learn==1.3.0   statsmodels==0.14.1   scipy==1.11.3
numpy==1.24.3      pandas==2.0.3
```

Plus `matplotlib`, `seaborn`, `joblib` (any recent versions).

```bash
conda create -n tf_screen python=3.10
conda activate tf_screen
pip install scanpy==1.9.5 anndata==0.9.2 harmonypy==0.2.0 \
            scikit-learn==1.3.0 statsmodels==0.14.1 scipy==1.11.3 \
            numpy==1.24.3 pandas==2.0.3 \
            matplotlib seaborn joblib
```

No installation of this repo itself is needed — each script adds the package root to `sys.path` so imports work regardless of where you launch from.

### 3. Get the data

```
Zenodo:  [TO FILL: DOI on publication]
GEO:     [TO FILL: GSE accession for the raw FASTQs]
```

The Zenodo deposit contains the cleaned screen `.h5ad`, the GB patient atlas RNA `.h5ad`, the precomputed Harmony embedding, the trained classifier `.pkl` files, and the per-strategy effect tables. See `docs/data_availability.md` for the file-by-file inventory.

### 4. Configure paths

```bash
cd tf_screen
cp config.template.py config.py
$EDITOR config.py   # set SCREEN_ANNDATA, ATLAS_RNA_ANNDATA, OUTPUT_DIR, …
```

`config.py` is gitignored. All scripts read paths and parameters from it.

### 5. Run

From the `tf_screen/` directory:

```bash
python preprocessing/build_joint_anndata.py
python preprocessing/run_harmony.py
python preprocessing/qc_figures.py

python classifiers/train_state_classifier.py
python classifiers/train_topic_classifier.py

python tf_expression/tf_self_expression.py
python tf_expression/control_p10_validation.py

python state_testing/state_testing.py
python state_testing/topic_testing.py
python state_testing/proportion_testing.py

python dose_response/build_metacells.py
python dose_response/correlation_analysis.py

python differential_expression/wilcoxon_de.py
```

Each script is independent — pick up at any step provided the preceding ones have run successfully. Each per-step `README.md` lists exact inputs and runnable commands.

LSF cluster submission templates are in `scripts/`.

---

## Numerical reproducibility

Every stochastic step uses `config.RANDOM_SEED` (default 42), passed explicitly to PCA, Harmony, classifier train/test splits, `LogisticRegression` fits, Leiden clustering and its KNN graph, and the permutation tests.

Per-comparison permutation seeds (one per (TF, state, cell line) cell in the state-testing grid) are derived from a deterministic hash of the comparison labels, so the null distribution for any single cell is reproducible even if other parts of the grid are skipped.

The dependency stack itself is the only source of run-to-run variation; the versions listed above match the conda environment used to produce the published numbers.

---

## License

MIT, matching the parent `PMBio/GBM_analysis` repository.

## Contact

For questions about the code, open an issue on the GitHub repository or contact `manu.saraswat AT dkfz.de`.
