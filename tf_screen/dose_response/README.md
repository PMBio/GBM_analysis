# Dose-response analysis (step 5)

Two scripts that test whether the population-level state shifts measured in step 4 are driven by **dose-dependent TF reprogramming** — i.e. cells that express more of the TF really do shift further toward the target state — rather than by guide-RNA delivery noise or by any single-cell-level artefact.

1. **`build_metacells.py`** — Leiden-cluster the screen cells into metacells (separately per cell line for controls, and per TF × cell line for OE), then aggregate gene expression and state/topic probabilities by mean within each cluster.
2. **`correlation_analysis.py`** — for every (TF, coarse state) pair, correlate per-metacell mean TF expression with per-metacell mean state probability, across three populations (combined / OE-only / Ctrl-only) and two metrics (log-normalised / control-z-scored).

The within-OE correlations are the strictest version of the dose-response claim and are the source of the per-TF dose-response annotation row in Fig. 3f.

---

## Inputs

| Script | Path | Source |
|---|---|---|
| `build_metacells.py` | `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | step 1 |
| `build_metacells.py` | `HARMONY_DIR/harmony_embeddings.npy` | step 1 |
| `correlation_analysis.py` | `METACELL_DIR/metacells.h5ad` | step 5a |
| `correlation_analysis.py` | `STATE_TESTING_DIR/FC0.5_p10/coarse_effects.csv` | step 4 (used for overlay plots) |

## Outputs

| Script | Path | Content |
|---|---|---|
| `build_metacells.py` | `METACELL_DIR/metacells.h5ad` | Metacell AnnData. `.X` = mean log1p expression per cluster; `.obs` carries `prob_coarse_<state>` and `prob_topic_<Topic_N>` aggregated by mean over the cluster's cells. |
| `build_metacells.py` | `METACELL_DIR/metacell_summary.csv` | Cell-counts per metacell (no expression). |
| `correlation_analysis.py` | `CORRELATION_DIR/correlation_combined.csv` | TF × state correlations on OE + Ctrl metacells. |
| `correlation_analysis.py` | `CORRELATION_DIR/correlation_oe_only.csv` | … on OE metacells only (strictest dose-response). |
| `correlation_analysis.py` | `CORRELATION_DIR/correlation_control_only.csv` | … on Ctrl metacells only (endogenous covariation). |
| `correlation_analysis.py` | `CORRELATION_DIR/correlation_per_cell_line.csv` | Combined-population analysis per cell line (BG5, P3, S24). |
| `correlation_analysis.py` | `CORRELATION_DIR/heatmaps/`, `scatter_plots/`, etc. | Figures. |

---

## Run

```bash
python dose_response/build_metacells.py
python dose_response/correlation_analysis.py
```

---

## Metacell construction details

| Parameter | Value | Source |
|---|---|---|
| Clustering | Leiden | `sc.tl.leiden` |
| Resolution | 2.0 | `config.LEIDEN_RESOLUTION` |
| Input space | 200 Harmony PCs | `config.N_PCS` |
| Neighbor graph | `sc.pp.neighbors` with `random_state=42` | |
| Minimum cells per condition | 10 | `config.METACELL_MIN_CELLS` |
| Random seed | 42 | `config.RANDOM_SEED` |

**Two clustering schemes:**

* **Control metacells** are clustered once per cell line (3 schemes total) and the resulting metacells are **shared** across every TF analysis within that line. This gives a stable, well-sampled control reference.
* **OE metacells** are clustered once per (TF, cell line) condition (165 schemes total) so each TF's metacells reflect that TF's own expression heterogeneity.

The total ends up being ~2,700 metacells across all 55 TFs and 3 cell lines (70 control + ~2,632 OE in the paper's run); the exact number is deterministic given the same input data and seed.

## Why three populations?

| Population | Cells | What it tests |
|---|---|---|
| `combined` | OE + Ctrl metacells | Primary discovery analysis. Maximum dynamic range in TF expression. Mixes biological between-condition variation with within-OE dose response. |
| `oe_only` | OE metacells alone | The strictest dose-response claim: *among cells that received the guide*, does **how much** TF is induced predict the state shift? This is the within-OE correlation reported in the paper text. |
| `control_only` | Ctrl metacells alone | Endogenous covariation between TF expression and state in unperturbed GB cells. Used to control for natural co-variation that would inflate the combined-population signal. |

## Why two metrics?

| Metric | Definition | When to prefer |
|---|---|---|
| Log-normalised | Per-metacell mean log1p expression of the TF gene. | Biological interpretation -- you're looking at the actual TF level. |
| Control-z-scored | `(tf_expr_log - ctrl_mean) / ctrl_std`, where the control statistics come from the matched cell line. | Cross-TF comparison -- removes per-TF baseline differences in mean and variance so correlation coefficients are comparable across TFs. The paper uses this as the primary metric in the per-TF dose-response annotation row of Fig. 3f. |

Both are reported in the output CSVs (`r_*_log` and `r_*_zscore`); switching primary metric is a single-line config change.

---

## Statistical test

For each (TF, state, population) triple, the script runs:

1. **Pearson correlation** on the raw vectors (parametric).
2. **Spearman correlation** on the raw vectors (rank-based, robust to outliers).
3. **Permutation test**: 1,000 shuffles of the TF expression vector; the null is the Spearman correlation distribution under independence. Reported as `perm_pval_{log,zscore}`.

BH FDR correction is applied per (population, metric) combination, with `fdr_log` and `fdr_zscore` columns. Significance default: FDR < 0.05.

---

## Notes

* **Column-name adapter.** The metacell AnnData stores aggregated state probabilities as `prob_coarse_<state>` to match the convention used everywhere else in the repo. `correlation_analysis.py` includes a small `_rename_probability_columns` helper at load time so the existing plotting body's `prob_<state>` references keep working.
* **No Proliferative.** Coarse states are taken from `config.COARSE_GROUPS`, which excludes Proliferative. Both the metacell builder and the correlation analyser therefore work over the five non-Proliferative states only.
* **Topic-level dose-response.** Topic probabilities are also aggregated into the metacell `.obs` (`prob_topic_<Topic_N>` columns) by `build_metacells.py`. A topic-level correlation analysis can be obtained by editing `correlation_analysis.py`'s `Config.COARSE_STATES` to point at the topic columns
