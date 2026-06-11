"""High-confidence GB state driver classification.

This module is currently a STUB. The production script is maintained by
L. and will be added to this repository in the next release.

Expected interface
------------------

For each of the four core GB states, the script classifies a TF as a
high-confidence driver of that state if **all three** of the following
hold (per the supplementary methods):

1. Strength of state change: the TF passes the consensus significance
   criterion (perm FDR < 0.05, t-test or Wilcoxon FDR < 0.05, |d| > 0.2)
   for that state in the pooled FC0.5_p10 analysis from
   ``state_testing/state_testing.py``, with positive Cohen's d.

2. Gene-signature enrichment: the corresponding state marker gene set
   is significantly enriched in the TF's per-cell-line Wilcoxon DE
   ranking (positive NES, q_GSEA < 0.05) from
   ``downstream/gsea_state_signatures.py``.

3. Dose-dependent state conversion: the within-OE Pearson correlation
   between TF expression and state probability across OE metacells is
   positive and significant (q < 0.05, r > 0) from
   ``dose_response/correlation_analysis.py``.

Output (planned)
----------------

``OUTPUT_DIR/downstream/high_confidence_drivers.csv``
    One row per (TF, state) flagged as a driver, with the three
    criteria's individual values for transparency.
``OUTPUT_DIR/downstream/driver_ranking.csv``
    Drivers ranked by per-TF Delta-state-score normalised by log2FC of
    TF expression (the Fig. 3f waterfall ordering).

Usage (planned)
---------------
``python downstream/high_confidence_drivers.py``
"""

raise NotImplementedError(
    "high_confidence_drivers is a placeholder; the implementation is "
    "maintained separately and will be added to this repository in the "
    "next release. See downstream/README.md for details."
)
