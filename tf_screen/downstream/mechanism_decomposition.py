"""Activator vs repressor decomposition of high-confidence state drivers.

This module is currently a STUB. The production script is maintained by
L. and will be added to this repository in the next release.

Expected interface
------------------

For each high-confidence driver from ``high_confidence_drivers.py``, two
scores are computed against the relevant state marker sets:

* **Activation score** -- contribution of *up*-regulated state-marker
  genes to the TF's transcriptional response.
* **Repression score** -- contribution of *down*-regulated markers of
  the *alternative* (opposing) GB states.

The bivariate (activation, repression) pair is the source of the
heatmap annotation in Fig. 3f, separating TFs that primarily activate
state markers (e.g. MEOX2, PRRX1, JUN) from those that primarily repress
alternative-state markers (e.g. MYT1L, ST18, SOX10) from those that
combine both modes (e.g. SOX11, SOX4, FOXO1, CEBPD).

Output (planned)
----------------

``OUTPUT_DIR/downstream/mechanism_scores.csv``
    One row per high-confidence (TF, state) with the activation and
    repression scores plus the per-gene contributions used to compute
    them.

Usage (planned)
---------------
``python downstream/mechanism_decomposition.py``
"""

raise NotImplementedError(
    "mechanism_decomposition is a placeholder; the implementation is "
    "maintained separately and will be added to this repository in the "
    "next release. See downstream/README.md for details."
)
