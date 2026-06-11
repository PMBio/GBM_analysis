# Pipeline overview

This document walks through the analysis pipeline step by step, mirroring the structure of the supplementary methods (manuscript §3.4.1–3.4.19). Reading this together with the supplementary methods gives the full story; reading the per-folder READMEs gives the operational detail for running each step.

## Step 1 — Preprocessing

Two intertwined goals here:

1. **Build a joint reference–screen object.** The classifier in step 2 needs to learn cell-state and topic identity from atlas cells whose ground-truth labels exist, then apply that learning to screen cells whose labels do not. Both populations must live in the same expression space (same genes, same coordinates) for this transfer to work. `build_joint_anndata.py` concatenates the screen (~261k cells) with the atlas (~96k exclusive-topic cells) on a shared gene set of ~4,300 features: the union of the top-500 marker genes per topic, plus the 55 TF genes force-added so we can always measure overexpression.

2. **Remove batch structure without removing biology.** The screen and the atlas were generated on different platforms (ScaleBio vs 10x Chromium) and the screen alone has 156 biological/technical batches. `run_harmony.py` runs PCA on the normalised joint feature matrix (200 components), then applies Harmony with `sample` as the batch covariate. The result is a 200-dimensional coordinate system in which a screen cell and a reference atlas cell with the same biological identity sit on top of each other; this is what the classifier learns from.

A separate script (`qc_figures.py`) generates the per-cell RNA QC panels and guide-assignment diagnostics. These figures (ED Fig. 5a–d) only summarise the data — no further filtering is applied beyond the upstream ScaleBio cell call and the confident guide-assignment step.

## Step 2 — Classifiers

Two multinomial logistic-regression models, each fit on the atlas portion of the joint Harmony embedding and applied to every cell (atlas + screen):

* **State classifier.** Two parallel fits — one over 9 fine GB cell states (`predicted_celltype`, `prob_ct_*`), one over 5 coarse states (`predicted_coarse`, `prob_coarse_*`). The five coarse states are defined by `config.COARSE_GROUPS` and explicitly exclude *Proliferative* — proliferation is a transient cell-cycle programme, not a stable cell-state identity, so atlas cells with that annotation are excluded from training.

* **Topic classifier.** Single fit over the 19 GB-intrinsic topics in `config.TOPICS_TRAINED`. The multinomial (softmax) formulation means per-cell topic probabilities **sum to one** — so a TF-induced increase in one topic's probability necessarily comes from decreases in others, giving the perturbation a direct compositional interpretation.

For both models, the trained classifier, its standard scaler, and its label encoder are saved as `.pkl` files in `MODELS_DIR`. Predictions are also written back into the joint AnnData's `.obs` columns so downstream steps don't need to re-load and re-predict.

## Step 3 — TF self-expression diagnostics

Two things to characterise before any state-effect testing:

1. **How strongly is each TF actually induced?** `tf_self_expression.py` computes the log2 fold-change of every TF's own gene in OE vs control cells, per cell line and pooled, in both arithmetic-mean and geometric-mean (log-space mean) flavours. The geometric-mean version is the headline log2FC reported in the paper. The same script also computes Sarle's bimodality coefficient on the OE distribution per TF — every one of the 55 TFs has BC > 0.555, confirming that OE populations are heterogeneous mixtures of induced and basal cells. This bimodality is why downstream tests filter OE cells to FC0.5 (1.5× linear above the control mean).

2. **Does the p10 control filter distort the control reference?** `control_p10_validation.py` checks that restricting controls to the bottom 10% of TF expressors does not systematically skew their cell-state composition. The median shift across all 55 TFs is near zero (ED Fig. 5c, right) — the filter enriches for low TF-expressing controls without changing what kind of cells they are.

## Step 4 — State, topic, and proportion testing

The core of the screen analysis.

For every (TF, state) — and analogously (TF, topic) — and for every cell line plus pooled, three statistical tests run in parallel under each of 16 filtering strategies (4 OE thresholds × 4 control percentiles):

* **Permutation test with matched subsampling.** Primary test. For each comparison, controls are randomly subsampled to the size of the OE arm without replacement, the combined OE+Ctrl labels are shuffled 1,000 times, and the permuted mean difference is compared with the observed. The matched-subsampling step prevents the very large control pool from dominating the null.
* **Welch's t-test.** Confirmatory; reported in the supplementary table.
* **Mann–Whitney U.** Confirmatory; reported in the supplementary table.

All three p-value sets get BH-adjusted across the full testing axis. A consensus hit requires permutation FDR < 0.05 *and* at least one of (t-test FDR or Wilcoxon FDR) < 0.05 *and* |Cohen's d| > 0.2 (small-to-medium effect-size floor).

A separate `proportion_testing.py` script does the same TF × state question through a **compositional** lens: each cell is hard-assigned to its argmax coarse state (Proliferative-dominant cells dropped), and the per-TF effect is tested with (i) a global multinomial likelihood-ratio test ("does this TF change the state distribution at all?") and (ii) a per-TF, per-state Fisher's exact test. The Fisher odds-ratio threshold OR > 1.5 is the "black-box" overlay annotation in Fig. 3c.

The 16-strategy grid is a sensitivity analysis. The headline strategy in Fig. 3 is `FC0.5_p10` (1.5× linear OE, bottom-10% controls).

## Step 5 — Dose-response

Population-level state shifts (step 4) tell you whether a TF moves cells on average. They don't tell you whether the shift is *dose-dependent* — i.e. whether cells that express more of the TF really do sit further toward the target state than cells that express less. That's the dose-response question.

To answer it, single-cell expression is too noisy to correlate directly, so `build_metacells.py` aggregates cells into Leiden clusters on the Harmony embedding (resolution 2.0; ~70 control + ~2,632 OE metacells across all 55 TFs and 3 cell lines), separately per (cell line) for controls and per (TF, cell line) for OE. Within each cluster, expression and state/topic probabilities are aggregated by mean.

`correlation_analysis.py` then correlates per-metacell mean TF expression with per-metacell mean state probability for every (TF, state) pair. Three populations are tested independently — combined (OE + Ctrl), OE-only (strictest dose-response), and Ctrl-only (endogenous covariation control) — and significance is by 1,000-permutation test on the Spearman correlation, BH-FDR'd per population.

## Step 6 — Differential expression

Per-cell-line Wilcoxon rank-sum DE on the screen AnnData, OE vs Ctrl, all cells on both sides (`all_all` strategy). The output gives per-(TF, cell line) DE tables with the standard scanpy fields plus a **signed-FDR ranking** score:

```
signed_fdr = sign(log2FC) * (-log10(q))
```

This is what downstream pre-ranked GSEA consumes. A cross-line intersection step also identifies genes significant in ≥2/3 cell lines with consistent direction — the reproducible per-TF DEG set.

Three sensitivity strategies (FC-filtered OE, p10 Ctrl, both) are also runnable from the same script for the supplementary table.

## Step 7 — Downstream integration

Currently external; stubs in this repo.

The three downstream analyses combine outputs from steps 4–6 to produce Fig. 3f and ED Fig. 5h–i:

* **GSEA against state signatures** — pre-ranked GSEA of each TF's signed-FDR ranking against the GB state marker sets. Source of the NES annotation in Fig. 3f.
* **High-confidence drivers** — a TF is a high-confidence driver of a state when three independent lines of evidence agree: significant state-probability shift (step 4) **and** significant GSEA NES (step 7a) **and** significant within-OE dose-response correlation (step 5). 31 such drivers across 4 core states in the paper.
* **Activator/repressor decomposition** — splits each driver's transcriptional response into "activation of target-state markers" and "repression of alternative-state markers". Source of the bivariate annotation in Fig. 3f.
