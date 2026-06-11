# Data availability

This document lists every input the pipeline needs, where to obtain it, and where it ends up.

## What ships with the code

Nothing — this repository contains only the analysis scripts. All input data and the deposited intermediates live in external archives.

## Where to get the data

| Source | Contents |
|---|---|
| **GEO** `[TO FILL: GSE accession]` | Raw FASTQs for the 55-TF screen + matched RNA library + raw amplicon CRISPR library. |
| **Zenodo** `[TO FILL: DOI]` | Processed `.h5ad` AnnData objects, the 200-PC Harmony embedding, trained classifier `.pkl` files, the full effect-size table across the 16 testing strategies, and the per-TF DE intersection tables. |
| **GB patient atlas** | Linked from the [`PMBio/GBM_analysis`](https://github.com/PMBio/GBM_analysis) parent repository (atlas RNA AnnData, per-cell annotation table, scDoRI topic gene rankings, top-5,000-cells-per-topic table, GBM/TME cell-state lookup tables). |

## Files needed by each pipeline step

For each input below, set the matching variable in your local `config.py` to the downloaded path.

### Step 1 — `preprocessing/`

| File (Zenodo or atlas repo) | `config.py` variable |
|---|---|
| `gbm_tf_screen_clean.h5ad` (raw counts, guide-assigned cells) | `SCREEN_ANNDATA` |
| `15_05_23_rna.h5ad` (atlas RNA) | `ATLAS_RNA_ANNDATA` |
| `annotation_obs.csv` | `ATLAS_ANNOTATION_OBS` |
| `top5k_cells_per_topic.tsv` | `ATLAS_TOPIC_TOP_CELLS` |
| `top_500_genes_per_topic.csv` | `ATLAS_TOPIC_GENES` |
| `annot_gbm.txt` | `ANNOT_GBM_TSV` |
| `TME_cell_state_map.tsv` | `ANNOT_TME_TSV` |

### Step 2 — `classifiers/`

Step 2 consumes the joint AnnData and the Harmony `.npy` that step 1 writes. No additional input.

If you want to skip steps 1–2 entirely and start from the trained models, download these Zenodo files and place them in `MODELS_DIR`:

```
clf_celltype_fine.pkl     scaler_celltype_fine.pkl     le_celltype_fine.pkl
clf_celltype_coarse.pkl   scaler_celltype_coarse.pkl   le_celltype_coarse.pkl
clf_topic.pkl             scaler_topic.pkl             le_topic.pkl
state_classifier_metrics.csv
topic_classifier_metrics.csv
```

You will also need the joint AnnData with the `prob_*` columns already populated (this is also on Zenodo as `joint_gbm_oe_anndata_with_predictions.h5ad`).

### Steps 3–6

All inputs are produced by earlier steps. No external data needed beyond step 1's inputs.

### Step 7

Currently external (see `downstream/README.md`).

## Output sizes

For one full pipeline run on the published dataset:

| Path | Approx. size |
|---|---|
| `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | ~3 GB |
| `HARMONY_DIR/harmony_embeddings.npy` | ~600 MB |
| `MODELS_DIR/` | ~50 MB |
| `STATE_TESTING_DIR/` (16 strategies × {fine, coarse}) | ~1.5 GB |
| `TOPIC_TESTING_DIR/` (16 strategies) | ~1 GB |
| `PROPORTION_DIR/` | ~500 MB |
| `METACELL_DIR/metacells.h5ad` | ~50 MB |
| `CORRELATION_DIR/` | ~200 MB |
| `DE_DIR/all_all/` (per-(TF, line) DE tables) | ~12 GB |
| `DE_DIR/` (all four strategies) | ~50 GB |

If disk space is tight, restrict `wilcoxon_de.py` to the primary `all_all` strategy and delete the per-(TF, line) CSVs after the combined `all_DE_results.csv` is written (the combined file contains the same rows).
