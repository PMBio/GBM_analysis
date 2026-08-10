"""Pre-ranked gene-set enrichment analysis of OE-induced transcriptional
responses against reference GB state marker gene sets.

This module is currently a STUB. The corresponding production script is
maintained by L. and is to be added here as part of the next repository
release. See ``downstream/README.md`` for context.

Expected interface (subject to L.'s confirmation):

* Input: per-TF, per-cell-line Wilcoxon DE results from
  ``differential_expression/wilcoxon_de.py`` (i.e.
  ``DE_DIR/all_all/all_DE_results.csv``).
* Ranking variable: the ``signed_fdr`` column computed by
  ``wilcoxon_de.py``.
* Gene-set source: state marker gene sets defined from the GB patient
  atlas.
* Implementation: pre-ranked GSEA (e.g. ``gseapy.prerank``) with 1,000
  permutations and the classic enrichment statistic.

Output (planned):

* ``OUTPUT_DIR/downstream/gsea/<TF>/<cell_line>.csv``
    Per-TF, per-cell-line NES, p-value, and BH-adjusted q-value for each
    state.
* ``OUTPUT_DIR/downstream/gsea/gsea_combined.csv``
    Aggregated across cell lines for input to
    ``high_confidence_drivers.py``.

Usage (planned)
---------------
``python downstream/gsea_state_signatures.py``
"""

raise NotImplementedError(
    "gsea_state_signatures is a placeholder; the implementation is "
    "maintained separately and will be added to this repository in the "
    "next release. See downstream/README.md for details."
)
