# Differential expression (step 6)

Single-cell Wilcoxon rank-sum differential expression of OE versus control cells, per cell line, for every TF. The output is (i) per-(TF, cell line) DE tables, (ii) a cross-line intersection of genes significant in ≥2/3 lines with consistent direction, and (iii) a per-gene **signed-FDR** score that is the input to the pre-ranked GSEA in `downstream/gsea_state_signatures.py`.

The script supports four cell-selection strategies (all OE vs all Ctrl, FC-filtered OE vs all Ctrl, all OE vs p10 Ctrl, FC-filtered OE vs p10 Ctrl). **The primary analysis used in Fig. 3 of the paper is `all_all`** (all cells, both sides); the other three are sensitivity strategies reported in the supplementary table.

---

## Inputs

| Path | Source |
|---|---|
| `SCREEN_ANNDATA` (cleaned, raw counts) | upstream |

## Outputs

For each strategy `S` ∈ {`all_all`, `fc_all`, `all_p10`, `fc_p10`}:

| Path | Content |
|---|---|
| `DE_DIR/S/<cell_line>__<TF>__DE.csv` | Wilcoxon table for one (TF, line); columns: `names`, `logfoldchanges`, `pvals`, `pvals_adj`, `log2FC`, `signed_fdr`, `TF`, `cell_line`, `n_oe`, `n_ctrl`. |
| `DE_DIR/S/all_DE_results.csv` | Concatenation across all (TF, line). |
| `DE_DIR/S/intersection/all_intersection_results.csv` | Genes significant in ≥2/3 cell lines, direction-consistency flagged. |
| `DE_DIR/S/intersection/intersection_summary.csv` | Per-TF up / down / total counts. |
| `DE_DIR/strategy_comparison.csv` | Hit counts across all strategies. |

---

## Run

```bash
# All four strategies:
python differential_expression/wilcoxon_de.py

# Only the primary:
python differential_expression/wilcoxon_de.py all_all

# Pick any subset:
python differential_expression/wilcoxon_de.py all_all fc_p10
```

---

## The signed-FDR ranking

For every gene in every (TF, cell line) Wilcoxon output, the script computes

```
log2FC     = scanpy.logfoldchanges / ln(2)      # convert from ln to log2
signed_fdr = sign(log2FC) * (-log10(pvals_adj))
```

This collapses direction and significance into a single ranking variable suitable for pre-ranked GSEA. Positive values are strongly up-regulated; negative values are strongly down-regulated; zero is non-significant.

Tiny p-values are clipped at `1e-300` before the `-log10` to avoid `inf`; the relative order is preserved.

## The four strategies

| Strategy | OE filter | Ctrl filter | Use |
|---|---|---|---|
| **`all_all`** | none | none | **Primary.** All assigned OE cells vs all control cells. Preserves population-level transcriptional signal, including low-induction OE cells that are still transcriptionally affected. |
| `fc_all` | FC0.5 (≥1.5× ctrl mean) | none | Sensitivity: focuses on high-expressing OE cells. |
| `all_p10` | none | bottom 10% TF expressors | Sensitivity: contrasts against TF-low controls. |
| `fc_p10` | FC0.5 | bottom 10% | Most restrictive: high-OE vs TF-low Ctrl. |

Both FC and p10 thresholds operate on **10,000-counts-per-cell-normalised** TF expression (removes library-size bias in cell selection). The Wilcoxon test itself uses raw counts that scanpy normalises and log1ps internally before ranking.

## Cross-cell-line intersection

For every (TF, gene), the intersection step keeps the gene if it is significant (FDR < 0.05) in **at least 2 of 3 cell lines** *and* the log2FCs in the significant lines agree in direction.

The intersection file's `direction_consistent` flag captures the direction-agreement requirement; the per-TF `intersection_summary.csv` only counts the direction-consistent genes.

---

## Notes

* **Wilcoxon is the primary DE in the paper.** The pseudobulk DESeq2 analysis (referenced in the supplementary methods as a sensitivity check) is not included in this repo; it was run with a paired-drop, downsampled-control design but is no longer the headline analysis.
* **Minimum cells per group**: 20 (`config.MIN_CELLS_PER_GROUP`). Comparisons below this threshold are skipped and not written to disk.
* **Random seed**: scanpy's `rank_genes_groups` is deterministic given the same input; no seed is required.
* **Output volume**: at 55 TFs × 3 cell lines × 4 strategies × ≈20k genes, the per-(TF, line) CSVs total ~50 GB if you keep them all. Consider running only the primary strategy first, then dropping the per-(TF, line) files and keeping only the combined ``all_DE_results.csv`` and the intersection outputs.
