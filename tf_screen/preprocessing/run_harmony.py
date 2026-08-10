"""Run PCA + Harmony on the joint atlas+screen AnnData.

The 200-PC Harmony-corrected embedding produced here is the input to all
downstream classifiers (state and topic) and to the metacell construction.
It is computed on the joint set of atlas and screen cells so that screen
cells share a coordinate system with the atlas.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Joint AnnData produced by ``build_joint_anndata.py``.

Outputs
-------
``HARMONY_DIR/harmony_embeddings.npy``
    Float32 array of shape ``(n_cells, N_PCS)``. Row order matches the
    joint AnnData on disk.
``HARMONY_DIR/obs_with_batch.csv``
    Cell metadata snapshot with the batch covariate used for correction.

Notes
-----
* Batch covariate is taken from ``obs['batch']``. For the screen cells
  this is the literal string "OE"; for the atlas it is ``obs['sample']``
  cast to string.
* The matrix passed to PCA is library-size-normalised to 10,000 counts
  per cell and log1p-transformed. ``utils.normalise_lognorm`` is
  idempotent, so re-running on an already-normalised joint object is
  safe.

Usage
-----
``python preprocessing/run_harmony.py``
"""

from __future__ import annotations

import warnings
from pathlib import Path

import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import normalise_lognorm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Harmony
# ---------------------------------------------------------------------------

def run_pca_and_harmony(adata: sc.AnnData) -> np.ndarray:
    """Run ``sc.tl.pca`` with ``N_PCS`` components followed by Harmony.

    Returns the corrected embedding as a 2-D ``np.ndarray`` of shape
    ``(n_cells, N_PCS)``.
    """
    normalise_lognorm(adata)

    print(f"   PCA: {config.N_PCS} components...")
    sc.tl.pca(adata, n_comps=config.N_PCS, svd_solver="arpack",
              random_state=config.RANDOM_SEED)

    print(f"   Harmony: max_iter={config.HARMONY_MAX_ITER}, batch={config.HARMONY_BATCH_KEY!r}")
    ho = hm.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        config.HARMONY_BATCH_KEY,
        max_iter_harmony=config.HARMONY_MAX_ITER,
        verbose=False,
    )

    # harmonypy returns the corrected matrix as ``Z_corr``. Older releases
    # may return it transposed; harmonise the shape here.
    Z_corr = ho.Z_corr
    if hasattr(Z_corr, "cpu"):
        Z_corr = Z_corr.cpu().numpy()
    elif hasattr(Z_corr, "numpy"):
        Z_corr = Z_corr.numpy()
    if Z_corr.shape[0] == config.N_PCS and Z_corr.shape[1] != config.N_PCS:
        Z_corr = Z_corr.T
    return np.asarray(Z_corr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = config.HARMONY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_path = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    print(f"Loading joint AnnData from {joint_path}...")
    adata = sc.read_h5ad(joint_path)
    print(f"   Cells: {adata.n_obs:,}   Genes: {adata.n_vars:,}")
    print(f"   Batches: {adata.obs[config.HARMONY_BATCH_KEY].nunique()}")

    Z_corr = run_pca_and_harmony(adata)
    print(f"   Harmony embedding: {Z_corr.shape}")

    np.save(out_dir / "harmony_embeddings.npy", Z_corr)
    adata.obs.to_csv(out_dir / "obs_with_batch.csv")
    print(f"Wrote {out_dir / 'harmony_embeddings.npy'}")
    print("Done.")


if __name__ == "__main__":
    main()
