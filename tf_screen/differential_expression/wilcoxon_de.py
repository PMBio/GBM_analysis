"""Per-cell-line Wilcoxon differential expression analysis (primary DE).

For every (TF, cell line), this script runs the Wilcoxon rank-sum
``rank_genes_groups`` test on the screen AnnData, restricted to one of
four cell-selection strategies:

    ``all_all``       all OE cells vs all Ctrl cells       (primary)
    ``fc_all``        FC0.5-filtered OE vs all Ctrl
    ``all_p10``       all OE vs bottom-10% Ctrl
    ``fc_p10``        FC0.5-filtered OE vs bottom-10% Ctrl

The primary analysis used in Fig. 3 of the paper is ``all_all`` (all
cells, both sides). The other three are sensitivity strategies used in
the supplementary table.

Both filters operate on **10,000-counts-per-cell-normalised** expression
of the TF gene to remove library-size bias in cell selection; the
Wilcoxon test itself uses raw counts that scanpy normalises and log1ps
internally before ranking.

Per-gene outputs include scanpy's ``logfoldchanges`` and ``pvals_adj``,
plus a **signed-FDR** ranking score::

    signed_fdr = sign(log2FC) * (-log10(q))

which combines direction and significance into a single number suitable
as input to pre-ranked GSEA (see ``downstream/gsea_state_signatures.py``).

A cross-line **intersection** pass identifies genes significant in
>= 2/3 of cell lines with consistent direction; the resulting table is
the reproducible per-TF DEG set used in the supplementary tables.

Inputs
------
``SCREEN_ANNDATA``
    Cleaned screen AnnData, raw counts.

Outputs
-------
``DE_DIR/<strategy>/<cell_line>__<TF>__DE.csv``
    Per (TF, cell line) Wilcoxon table (all genes), with ``signed_fdr``.
``DE_DIR/<strategy>/all_DE_results.csv``
    Concatenated per-line results across all TFs.
``DE_DIR/<strategy>/intersection/<TF>__intersection.csv``
    Per-TF genes significant in >= 2/3 lines with consistent direction.
``DE_DIR/<strategy>/intersection/intersection_summary.csv``
    Per-TF up/down gene counts.
``DE_DIR/strategy_comparison.csv``
    Hit counts across all four strategies.

Usage
-----
``python differential_expression/wilcoxon_de.py [strategy ...]``

If no strategy is given, all four are run. Examples::

    python differential_expression/wilcoxon_de.py
    python differential_expression/wilcoxon_de.py all_all
    python differential_expression/wilcoxon_de.py all_all fc_p10
"""

from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import (compute_norm_factors, list_cell_lines, list_tfs,
                             select_oe_and_ctrl_indices)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
# Each strategy is (fc_threshold, control_percentile). ``None`` skips the
# filter entirely; ``config.FC_THRESHOLD`` and ``config.CONTROL_PERCENTILE``
# pick up the headline FC0.5 and p10 values.

STRATEGIES: dict[str, tuple[float | None, float | None]] = {
    "all_all": (None,                  None),
    "fc_all":  (config.FC_THRESHOLD,   None),
    "all_p10": (None,                  config.CONTROL_PERCENTILE),
    "fc_p10":  (config.FC_THRESHOLD,   config.CONTROL_PERCENTILE),
}

PRIMARY_STRATEGY = "all_all"


# ---------------------------------------------------------------------------
# Wilcoxon DE for one comparison
# ---------------------------------------------------------------------------

def run_wilcoxon(adata: sc.AnnData, oe_idx: np.ndarray, ctrl_idx: np.ndarray,
                 tf: str, cell_line: str,
                 min_cells: int = config.MIN_CELLS_PER_GROUP
                 ) -> pd.DataFrame | None:
    """Run ``scanpy.tl.rank_genes_groups`` (Wilcoxon) on the given indices.

    Returns the per-gene results dataframe annotated with the TF and cell
    line, augmented with a ``signed_fdr`` column for GSEA ranking, or
    ``None`` if either group is below ``min_cells``.
    """
    n_oe, n_ctrl = int(oe_idx.size), int(ctrl_idx.size)
    if n_oe < min_cells or n_ctrl < min_cells:
        return None

    all_idx = np.concatenate([oe_idx, ctrl_idx])
    sub = adata[all_idx].copy()
    sub.obs["condition"] = ["OE"] * n_oe + ["Ctrl"] * n_ctrl

    # Normalise to 10k + log1p in-place if the matrix still looks like counts.
    X = sub.X
    x_max = X.max() if not hasattr(X, "toarray") else X.toarray().max()
    if x_max > 20:
        sc.pp.normalize_total(sub, target_sum=1e4, inplace=True)
        sc.pp.log1p(sub)

    sc.tl.rank_genes_groups(
        sub, groupby="condition", groups=["OE"], reference="Ctrl",
        method="wilcoxon", use_raw=False, key_added="de",
    )
    df = sc.get.rank_genes_groups_df(sub, group="OE", key="de")
    df["TF"] = tf
    df["cell_line"] = cell_line
    df["n_oe"] = n_oe
    df["n_ctrl"] = n_ctrl

    # Signed-FDR ranking for GSEA. The +1e-300 floor keeps -log10 finite
    # at machine-zero p-values without changing the ranking.
    q = df["pvals_adj"].astype(float).clip(lower=1e-300)
    log2fc = df["logfoldchanges"].astype(float) / np.log(2)
    df["log2FC"] = log2fc
    df["signed_fdr"] = np.sign(log2fc) * (-np.log10(q))

    del sub
    gc.collect()
    return df


# ---------------------------------------------------------------------------
# Per-strategy driver
# ---------------------------------------------------------------------------

def run_strategy(adata: sc.AnnData, tfs: list[str], cell_lines: list[str],
                 strategy: str, norm_factors: np.ndarray) -> pd.DataFrame:
    """Run Wilcoxon DE for every (TF, cell line) under one strategy."""
    fc, ctrl_pct = STRATEGIES[strategy]
    out_dir = config.DE_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Strategy: {strategy}  (FC={fc}, Ctrl_p={ctrl_pct}) ===")
    rows: list[pd.DataFrame] = []
    for ti, tf in enumerate(tfs, 1):
        print(f"  [{ti}/{len(tfs)}] {tf}")
        for cl in cell_lines:
            oe_idx, ctrl_idx = select_oe_and_ctrl_indices(
                adata, tf, cl,
                fc_threshold=fc, control_percentile=ctrl_pct,
                norm_factors=norm_factors,
            )
            df = run_wilcoxon(adata, oe_idx, ctrl_idx, tf, cl)
            if df is None:
                print(f"     {cl}: {len(oe_idx)} OE, {len(ctrl_idx)} Ctrl -> SKIP")
                continue
            n_sig = int((df["pvals_adj"] < config.FDR_THRESHOLD).sum())
            print(f"     {cl}: {len(oe_idx)} OE, {len(ctrl_idx)} Ctrl"
                  f" -> {n_sig} sig genes")
            df.to_csv(out_dir / f"{cl}__{tf}__DE.csv", index=False)
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_dir / "all_DE_results.csv", index=False)
    print(f"  combined: {len(combined):,} rows -> {out_dir/'all_DE_results.csv'}")
    return combined


# ---------------------------------------------------------------------------
# Cross-cell-line intersection
# ---------------------------------------------------------------------------

def intersect_cell_lines(de_combined: pd.DataFrame, strategy: str,
                         min_lines: int = 2,
                         fdr_threshold: float = config.FDR_THRESHOLD
                         ) -> pd.DataFrame:
    """For every (TF, gene), keep rows where the gene is significant in
    ``>= min_lines`` cell lines with consistent log2FC direction.
    """
    if de_combined.empty:
        return de_combined

    out_dir = config.DE_DIR / strategy / "intersection"
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_lines = sorted(c for c in de_combined["cell_line"].unique() if c != "pooled")
    rows = []
    for tf in sorted(de_combined["TF"].unique()):
        tf_df = de_combined[de_combined["TF"] == tf]
        for gene in tf_df["names"].unique():
            gene_df = tf_df[tf_df["names"] == gene]
            sig_lines: list[str] = []
            lfcs: dict[str, float] = {}
            for cl in cell_lines:
                cl_rows = gene_df[gene_df["cell_line"] == cl]
                if cl_rows.empty:
                    continue
                row = cl_rows.iloc[0]
                lfcs[cl] = float(row["logfoldchanges"])
                if row["pvals_adj"] < fdr_threshold:
                    sig_lines.append(cl)
            if len(sig_lines) < min_lines:
                continue
            sig_lfcs = [lfcs[cl] for cl in sig_lines]
            direction_consistent = (all(x > 0 for x in sig_lfcs)
                                    or all(x < 0 for x in sig_lfcs))
            mean_lfc = float(np.mean(list(lfcs.values())))
            rec = {
                "TF": tf, "gene": gene,
                "n_sig_lines": len(sig_lines),
                "sig_in": ",".join(sig_lines),
                "direction_consistent": direction_consistent,
                "mean_lfc": mean_lfc,
            }
            rec.update({f"lfc_{cl}": lfcs.get(cl, np.nan) for cl in cell_lines})
            rows.append(rec)

    inter = pd.DataFrame(rows)
    if inter.empty:
        return inter

    inter = inter.sort_values("mean_lfc", key=lambda s: s.abs(), ascending=False)
    inter.to_csv(out_dir / "all_intersection_results.csv", index=False)

    consistent = inter[inter["direction_consistent"]]
    summary = (
        consistent.groupby("TF")
        .agg(n_up=("mean_lfc", lambda s: int((s > 0).sum())),
             n_down=("mean_lfc", lambda s: int((s < 0).sum())),
             n_total=("gene", "count"))
        .reset_index()
        .sort_values("n_total", ascending=False)
    )
    summary.to_csv(out_dir / "intersection_summary.csv", index=False)
    print(f"  intersection: {len(inter):,} gene-rows from {summary.shape[0]} TFs"
          f" -> {out_dir}")
    return inter


# ---------------------------------------------------------------------------
# Cross-strategy comparison
# ---------------------------------------------------------------------------

def compare_strategies(all_intersections: dict[str, pd.DataFrame]) -> None:
    """Roll up significant-gene counts across the four strategies."""
    rows = []
    for strategy, df in all_intersections.items():
        if df.empty:
            rows.append({"strategy": strategy, "n_tfs_with_hits": 0,
                         "total_de_genes": 0,
                         "primary": strategy == PRIMARY_STRATEGY})
            continue
        consistent = df[df["direction_consistent"]]
        per_tf = consistent.groupby("TF").size()
        rows.append({
            "strategy":         strategy,
            "n_tfs_with_hits":  int(per_tf.size),
            "total_de_genes":   int(per_tf.sum()),
            "mean_de_per_tf":   round(float(per_tf.mean()), 1) if per_tf.size else 0,
            "median_de_per_tf": int(per_tf.median()) if per_tf.size else 0,
            "max_de_per_tf":    int(per_tf.max()) if per_tf.size else 0,
            "total_up":         int(consistent[consistent["mean_lfc"] > 0].shape[0]),
            "total_down":       int(consistent[consistent["mean_lfc"] < 0].shape[0]),
            "primary":          strategy == PRIMARY_STRATEGY,
        })
    pd.DataFrame(rows).to_csv(config.DE_DIR / "strategy_comparison.csv", index=False)
    print(f"\nWrote {config.DE_DIR / 'strategy_comparison.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    strategies = argv if argv else list(STRATEGIES.keys())
    invalid = [s for s in strategies if s not in STRATEGIES]
    if invalid:
        raise SystemExit(
            f"Unknown strategy/strategies {invalid}. "
            f"Known: {list(STRATEGIES.keys())}"
        )

    print(f"Loading screen AnnData from {config.SCREEN_ANNDATA}...")
    adata = sc.read_h5ad(config.SCREEN_ANNDATA)
    print(f"   Cells: {adata.n_obs:,}   Genes: {adata.n_vars:,}")

    print("Computing per-cell normalisation factors (used for FC/p10 cell"
          " selection only)...")
    norm_factors = compute_norm_factors(adata.X)

    tfs = list_tfs(adata)
    cell_lines = list_cell_lines(adata)
    print(f"   TFs: {len(tfs)}   Cell lines: {cell_lines}")

    config.DE_DIR.mkdir(parents=True, exist_ok=True)
    intersections: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        combined = run_strategy(adata, tfs, cell_lines, strategy, norm_factors)
        intersections[strategy] = intersect_cell_lines(combined, strategy)

    compare_strategies(intersections)
    print("Done.")


if __name__ == "__main__":
    main()
