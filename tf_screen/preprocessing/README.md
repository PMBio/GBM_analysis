# Preprocessing (step 1)

Three scripts that prepare the data for classification and statistical testing:

1. **`build_joint_anndata.py`** — concatenates the GB patient atlas (subset to topic-exclusive cells) and the guide-assigned screen cells into a shared AnnData object on a common gene set.
2. **`run_harmony.py`** — runs PCA on the joint object and applies Harmony to remove batch effects, producing the 200-PC embedding shared by all downstream classifiers.
3. **`qc_figures.py`** — generates per-cell RNA QC panels and guide-assignment diagnostics (the source of panels in Extended Data Fig. 5a–d).

Run them in this order. `qc_figures.py` is independent — it reads only the cleaned screen AnnData, so it can be run before or after the other two.

---

## Inputs (set in `config.py`)

| Variable | Purpose |
|---|---|
| `SCREEN_ANNDATA` | Cleaned screen AnnData (raw counts, one row per guide-assigned cell). |
| `ATLAS_RNA_ANNDATA` | GB patient atlas RNA AnnData. |
| `ATLAS_ANNOTATION_OBS` | Atlas per-cell annotation CSV. |
| `ATLAS_TOPIC_TOP_CELLS` | Top-5,000 cells per topic, wide format. |
| `ATLAS_TOPIC_GENES` | Per-topic top-500 gene rankings. |
| `ANNOT_GBM_TSV`, `ANNOT_TME_TSV` | Coarse-state name lookup tables. |

## Outputs

| Step | Path | Content |
|---|---|---|
| `build_joint_anndata` | `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | Joint atlas+screen AnnData. |
| `build_joint_anndata` | `JOINT_ANNDATA_DIR/feature_genes.csv` | Shared gene list. |
| `run_harmony` | `HARMONY_DIR/harmony_embeddings.npy` | 200-PC Harmony embedding, row-aligned to the joint AnnData. |
| `run_harmony` | `HARMONY_DIR/obs_with_batch.csv` | Cell metadata snapshot with the batch covariate. |
| `qc_figures` | `OUTPUT_DIR/qc_figures/` | PDFs and PNGs of QC panels + summary CSV. |

---

## Run

```bash
python preprocessing/build_joint_anndata.py
python preprocessing/run_harmony.py
python preprocessing/qc_figures.py
```


