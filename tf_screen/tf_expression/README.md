# TF expression diagnostics (step 3)

Two scripts that characterise (i) how strongly each TF is induced relative to its non-targeting control, and (ii) whether the bottom-percentile control filter used in downstream tests distorts the control reference population.

1. **`tf_self_expression.py`** — per-TF log2 fold-change and bimodality coefficient, per cell line and pooled. Source for the dot/colour heatmap of TF induction, the OE-vs-Ctrl violins, and the cell-line concordance scatter in Extended Data Fig. 5.
2. **`control_p10_validation.py`** — per-TF coarse-state composition of control cells before and after filtering to the bottom 10% (and 25%) of TF expressors. Source for the entropy / state-shift panel in Extended Data Fig. 5c (right).

The two scripts are independent: `tf_self_expression.py` reads the raw screen AnnData and does not need the classifier outputs; `control_p10_validation.py` reads the joint AnnData and the state classifier.

---

## Inputs

| Script | Path |
|---|---|
| `tf_self_expression.py` | `SCREEN_ANNDATA` |
| `control_p10_validation.py` | `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad`, `HARMONY_DIR/harmony_embeddings.npy`, `MODELS_DIR/*` |

## Outputs

| Script | Path | Content |
|---|---|---|
| `tf_self_expression.py` | `TF_EXPRESSION_DIR/tf_expression_results.csv` | Per (TF, scope) row: means, std, both log2FC flavours, detection rates, Cohen's d, MWU p-value, BH padj. |
| `tf_self_expression.py` | `TF_EXPRESSION_DIR/fig_*.{pdf,png}` | Heatmap, volcano, top-TF violins, OE-vs-Ctrl scatter, cell-line concordance. |
| `control_p10_validation.py` | `OUTPUT_DIR/control_state_analysis/control_states_by_cell_line.csv` | Coarse-state composition of controls per cell line. |
| `control_p10_validation.py` | `.../control_states_tf_filtering.csv` | Per (TF, filter ∈ {all, bottom_10, bottom_25}) row: mean prob + argmax prop per coarse state. |
| `control_p10_validation.py` | `.../control_state_shifts.csv` | Per (TF, filter, state) row: baseline vs filtered values + their difference. |
| `control_p10_validation.py` | `.../fig_*.{pdf,png}` | By-cell-line and shift visualisations. |

---

## Run

```bash
python tf_expression/tf_self_expression.py
python tf_expression/control_p.py10_validation
```

---

## Why the p10 control filter doesn't bias the analysis

Control cells in GB are heterogeneous: any given TF is intrinsically high in some cell states (e.g. SOX10 in OPC-like cells) and intrinsically low in others. Comparing OE cells against the full control pool therefore lets some control cells "leak" toward the perturbation's target state. Restricting the control pool to the bottom 10% of TF expressors (the *p10* strategy) suppresses this leakage.

`control_p10_validation.py` checks that this restriction does not itself bias the cell-state composition of the control pool. For each TF and each cell line it reports the change in coarse-state proportions (and entropy) between the full and p10-filtered controls. The pooled median shift across all 55 TFs is close to zero (Extended Data Fig. 5c, right), confirming that p10 enriches for low TF-expressing controls without systematically skewing state composition.

