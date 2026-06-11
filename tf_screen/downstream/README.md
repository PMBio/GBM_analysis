# Downstream analyses (step 7)

The three scripts in this folder produce the integrated outputs shown in **Fig. 3f** (waterfall + driver annotation) and **Extended Data Fig. 5h–i**. They take the per-TF state effects (step 4), per-TF DE rankings (step 6), and per-TF dose-response correlations (step 5) and combine them into a driver-classification and mechanism-decomposition pipeline.

## Status

All three scripts in this folder are **placeholder stubs**. The production code is maintained by Laura and will be merged into this repository in the next release. Each stub raises `NotImplementedError` when imported and documents the planned interface in its module docstring.

| File | What it will do |
|---|---|
| `gsea_state_signatures.py` | Pre-ranked GSEA of per-TF Wilcoxon DE rankings against reference GB state marker sets. Source of the NES annotation in Fig. 3f. |
| `high_confidence_drivers.py` | Classifies a TF as a high-confidence driver of a state if (i) it passes the state-testing consensus criterion with positive Cohen's d, (ii) its DE ranking shows positive NES against the state marker set, and (iii) its OE-metacell dose-response correlation is positive and significant. Yielded 31 drivers across four core states in the paper. |
| `mechanism_decomposition.py` | Splits each driver's effect into an "activation" component (up-regulation of target-state markers) and a "repression" component (down-regulation of alternative-state markers). Source of the bivariate annotation in Fig. 3f. |

## Inputs (when implemented)

| File | From step |
|---|---|
| `STATE_TESTING_DIR/FC0.5_p10/coarse_effects.csv` | 4 |
| `DE_DIR/all_all/all_DE_results.csv` | 6 |
| `CORRELATION_DIR/correlation_oe_only.csv` | 5 |
| State marker gene sets | external — atlas-derived |

## Outputs (planned)

```
OUTPUT_DIR/downstream/
├── gsea/<TF>/<cell_line>.csv
├── gsea/gsea_combined.csv
├── high_confidence_drivers.csv
├── driver_ranking.csv
└── mechanism_scores.csv
```

## Notes for the next release

* The signed-FDR ranking column (`signed_fdr`) produced by `differential_expression/wilcoxon_de.py` is already in the format expected by `gseapy.prerank`. No additional ranking conversion should be needed.
* Coarse state names should be taken from `config.COARSE_GROUPS.keys()` (five non-Proliferative states), matching what step 4 outputs.
* The "four core states" used by `high_confidence_drivers.py` in the paper are AC-states, Neuronal-states, OPC-states, and Stress-states (OPCs excluded). When merging the production code, this choice should either be made explicit via a constant at the top of the file or read from `config`.
