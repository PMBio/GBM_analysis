"""Build metacells by Leiden clustering on the Harmony embedding.

Metacells are aggregated cells used downstream by ``correlation_analysis``
to test whether per-cell TF expression *level* (not just guide
assignment) predicts per-cell state probability. Aggregation suppresses
the zero-inflated noise of single-cell TF measurements without smoothing
across TF identities or cell lines.

Two metacell schemes are built:

* **Control metacells.** All control-guide cells of one cell line are
  clustered with Leiden at ``LEIDEN_RESOLUTION``; the resulting
  clusters become control metacells shared across every TF comparison
  for that line.
* **OE metacells.** For each (TF, cell line), the OE cells of that
  condition are clustered independently with the same parameters. A
  condition with fewer than ``METACELL_MIN_CELLS`` cells is skipped.

Aggregation within a Leiden cluster:

* Mean of log1p-normalised expression for every gene.
* Mean of per-cell coarse-state probability for every state.
* Mean of per-cell topic probability for every trained topic.
* Cell count and the cluster's (guide, cell line) labels.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Must contain ``prob_coarse_<state>`` columns from
    ``classifiers/train_state_classifier.py``. Topic probability columns
    (``prob_topic_<Topic_N>``), if present, are also aggregated.
``HARMONY_DIR/harmony_embeddings.npy``
    Row-aligned to the joint AnnData on disk.

Outputs
-------
``METACELL_DIR/metacells.h5ad``
    Metacell AnnData. ``X`` is a dense mean-expression matrix of shape
    ``(n_metacells, n_genes)``; ``obs`` contains metacell metadata plus
    the aggregated state and topic probability columns.
``METACELL_DIR/metacell_summary.csv``
    Cell-counts per metacell (without expression).

Usage
-----
``python dose_response/build_metacells.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import list_cell_lines, list_tfs, normalise_lognorm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Leiden helper
# ---------------------------------------------------------------------------

def leiden_on_subset(adata: sc.AnnData, indices: np.ndarray,
                     resolution: float = config.LEIDEN_RESOLUTION,
                     n_pcs: int = config.N_PCS) -> np.ndarray:
    """Leiden cluster a subset of cells on the first ``n_pcs`` Harmony PCs."""
    sub = adata[indices].copy()
    sub.obsm["X_pca"] = sub.obsm["X_harmony"][:, :n_pcs]
    sc.pp.neighbors(sub, n_pcs=n_pcs, use_rep="X_pca",
                    random_state=config.RANDOM_SEED)
    sc.tl.leiden(sub, resolution=resolution, key_added="leiden",
                 random_state=config.RANDOM_SEED)
    return sub.obs["leiden"].values


# ---------------------------------------------------------------------------
# Per-cluster aggregation
# ---------------------------------------------------------------------------

def _prob_columns(adata: sc.AnnData) -> tuple[list[str], list[str]]:
    """Return the lists of (coarse-state, topic) probability columns
    present on ``adata.obs``.
    """
    coarse = [c for c in adata.obs.columns if c.startswith("prob_coarse_")]
    topic = [c for c in adata.obs.columns if c.startswith("prob_topic_")]
    return coarse, topic


def aggregate_cluster(adata: sc.AnnData, cluster_labels: np.ndarray,
                      label_prefix: str,
                      coarse_cols: list[str], topic_cols: list[str]
                      ) -> list[dict]:
    """Aggregate cells of one (cell line, guide) stratum to metacells.

    Returns a list of dicts (one per Leiden cluster). Each dict contains
    metadata, the cluster's mean expression vector (under key
    ``"mean_expr"``), and the cluster's mean of every probability
    column.
    """
    metacells = []
    for cluster in np.unique(cluster_labels):
        keep = cluster_labels == cluster
        sub = adata[keep]
        if sub.n_obs == 0:
            continue

        X = sub.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        mean_expr = np.mean(X, axis=0)

        record = {
            "metacell_id": f"{label_prefix}_mc{cluster}",
            "n_cells":          sub.n_obs,
            "cell_line":        sub.obs["cell_line"].iloc[0],
            "guide_assignment": sub.obs["guide_assignment"].iloc[0],
            "cluster":          str(cluster),
            "mean_expr":        mean_expr,
        }
        for col in coarse_cols + topic_cols:
            record[col] = float(sub.obs[col].mean())
        metacells.append(record)
    return metacells


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def build_all_metacells(adata: sc.AnnData) -> list[dict]:
    """Run the per-cell-line control clustering and the per-(TF, cell line)
    OE clustering, and return the combined list of metacells.
    """
    coarse_cols, topic_cols = _prob_columns(adata)
    print(f"   Aggregating {len(coarse_cols)} coarse-state and "
          f"{len(topic_cols)} topic probability columns.")

    all_metacells: list[dict] = []
    tfs = list_tfs(adata)
    cell_lines = list_cell_lines(adata)

    print("\nControl metacells (per cell line):")
    ctrl_mask = adata.obs["guide_assignment"] == "Ctrl"
    for cl in cell_lines:
        idx = np.where(ctrl_mask.values
                       & (adata.obs["cell_line"] == cl).values)[0]
        if idx.size < config.METACELL_MIN_CELLS:
            print(f"   {cl}: too few ctrl cells ({idx.size}); skipping")
            continue
        labels = leiden_on_subset(adata, idx)
        new = aggregate_cluster(adata[idx], labels, f"{cl}_Ctrl",
                                coarse_cols, topic_cols)
        all_metacells.extend(new)
        print(f"   {cl}: {idx.size:,} ctrl cells -> {len(new)} metacells "
              f"(mean {idx.size / max(1, len(new)):.1f} cells/metacell)")

    print("\nOE metacells (per TF, per cell line):")
    for ti, tf in enumerate(tfs, 1):
        tf_mask = (adata.obs["guide_assignment"] == tf).values
        for cl in cell_lines:
            idx = np.where(tf_mask & (adata.obs["cell_line"] == cl).values)[0]
            if idx.size < config.METACELL_MIN_CELLS:
                continue
            labels = leiden_on_subset(adata, idx)
            new = aggregate_cluster(adata[idx], labels, f"{cl}_{tf}",
                                    coarse_cols, topic_cols)
            all_metacells.extend(new)
        if ti % 10 == 0:
            print(f"   [{ti}/{len(tfs)}] processed; total so far: "
                  f"{len(all_metacells):,} metacells")
    print(f"   total metacells: {len(all_metacells):,}")
    return all_metacells


def metacells_to_anndata(metacells: list[dict], var_names: list[str]
                         ) -> sc.AnnData:
    """Pack the per-cluster dicts into an AnnData where each row is a
    metacell and ``X`` is the mean-expression matrix.
    """
    X = np.vstack([mc.pop("mean_expr") for mc in metacells])
    obs = pd.DataFrame(metacells)
    obs.index = obs["metacell_id"]
    var = pd.DataFrame(index=var_names)
    return sc.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = config.METACELL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_path = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    harmony_path = config.HARMONY_DIR / "harmony_embeddings.npy"

    print(f"Loading joint AnnData from {joint_path}...")
    adata = sc.read_h5ad(joint_path)
    normalise_lognorm(adata)

    print(f"Loading Harmony embedding from {harmony_path}...")
    adata.obsm["X_harmony"] = np.load(harmony_path)

    # Restrict to the screen cells (atlas cells should never be metacells).
    is_atlas = adata.obs["dataset"].astype(str).str.lower().str.contains("gbm", na=False)
    adata_oe = adata[~is_atlas.values].copy()
    print(f"   Screen cells: {adata_oe.n_obs:,}")

    metacells = build_all_metacells(adata_oe)
    meta_adata = metacells_to_anndata(metacells, list(adata_oe.var_names))
    print(f"\nMetacell AnnData: {meta_adata.n_obs:,} metacells, "
          f"{meta_adata.n_vars:,} genes")

    out_path = out_dir / "metacells.h5ad"
    meta_adata.write_h5ad(out_path)
    meta_adata.obs.to_csv(out_dir / "metacell_summary.csv")
    print(f"Wrote {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
