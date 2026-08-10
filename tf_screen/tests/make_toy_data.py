"""Generate a synthetic toy dataset that exercises the full pipeline end to end.

The toy dataset mimics the structure of the real inputs:

* A joint AnnData of atlas + screen cells with all the obs columns expected
  by downstream scripts (`dataset`, `cell_line`, `sample`, `guide_assignment`,
  `celltype` (fine), `celltype_coarse`, `topic`, `batch`).
* A row-aligned Harmony embedding (`.npy`).
* A separate screen AnnData containing only the screen cells, with the
  extra obs columns required by `tf_self_expression.py` and the DE script
  (`guide_has_assignment`).

To keep classifiers / statistical tests non-degenerate:

* All fine cell states get at least 200 atlas cells.
* Atlas cells of the same fine state cluster in the Harmony embedding,
  so the LogisticRegression classifiers can actually learn signal.
* OE cells of TF X have their X-gene expression boosted ~8x over Ctrl,
  so the FC0.5 filter selects something non-empty.

Deterministic given seed 42.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util as _ilu

# Don't depend on tf_screen/config.py existing — it's gitignored and the
# pipeline only needs it after the toy config has been written. Load the
# template directly to get the canonical COARSE_GROUPS / TOPICS_TRAINED /
# N_PCS values.
_TEMPLATE_PATH = _REPO_ROOT / "tf_screen" / "config.template.py"
_spec = _ilu.spec_from_file_location("_tf_screen_config_template", _TEMPLATE_PATH)
default_config = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(default_config)


SEED = 42

# Use a deliberate 55-TF list (matching the real screen's size) so the toy
# dataset behaves like the real one. The names are placeholders; what
# matters for the pipeline is that they appear both as var_names and as
# guide labels.
TFS = [
    "ASCL1", "OLIG2", "SOX2",  "SOX10", "POU3F2", "NEUROG2", "DLX2",
    "LHX2",  "FOXG1", "NFIA",  "NFIB",  "NR2E1",  "PAX6",   "EMX1",
    "EMX2",  "MEIS2", "BCL11B","TBR1",  "EOMES",  "GLI1",   "GLI2",
    "GLI3",  "MEF2C", "TFAP2A","TFAP2C","RUNX1",  "RUNX2",  "MYC",
    "MYCN",  "STAT3", "STAT1", "IRF1",  "IRF8",   "JUN",    "FOS",
    "ATF3",  "ATF4",  "CEBPB", "CEBPD", "BHLHE40","BHLHE41","ID1",
    "ID2",   "ID3",   "HES1",  "HES5",  "ELF1",   "ETV1",   "ETV4",
    "ETV5",  "KLF4",  "KLF6",  "EGR1",  "ZEB1",   "ZEB2",
]
assert len(TFS) == 55

CELL_LINES = ["BG5", "P3", "S24"]
SAMPLES_PER_LINE = {cl: [f"{cl}-90k-R1", f"{cl}-135k-R2"] for cl in CELL_LINES}

N_ATLAS = 2_000
N_SCREEN = 3_000
N_GENES = 500


def _fine_to_coarse_map() -> dict[str, str]:
    return {fine: coarse for coarse, fines in default_config.COARSE_GROUPS.items()
            for fine in fines}


def _gene_names() -> list[str]:
    """500 genes total. First 55 are the TFs; remaining 445 are placeholders."""
    extra = [f"GENE_{i:04d}" for i in range(N_GENES - len(TFS))]
    return TFS + extra


def _build_atlas(rng: np.random.Generator,
                 gene_names: list[str]) -> tuple[ad.AnnData, np.ndarray]:
    """Atlas portion: N_ATLAS cells, one fine cell type per cell, drawn from
    the union of COARSE_GROUPS values. Returns AnnData and the matching
    Harmony embedding rows (atlas cells form fine-state clusters in PC
    space so classifiers learn signal).
    """
    fine_to_coarse = _fine_to_coarse_map()
    fine_states = sorted(fine_to_coarse.keys())
    topics = [f"Topic_{t}" for t in default_config.TOPICS_TRAINED]

    # Distribute atlas cells roughly evenly across fine states.
    n_per_state = N_ATLAS // len(fine_states)
    fine_labels: list[str] = []
    for s in fine_states:
        fine_labels.extend([s] * n_per_state)
    while len(fine_labels) < N_ATLAS:
        fine_labels.append(fine_states[len(fine_labels) % len(fine_states)])
    rng.shuffle(fine_labels)

    fine_arr = np.array(fine_labels)
    coarse_arr = np.array([fine_to_coarse[f] for f in fine_arr])
    topic_arr = np.array(rng.choice(topics, size=N_ATLAS))

    # Expression: Poisson counts, baseline lambda=2.
    X = rng.poisson(lam=2.0, size=(N_ATLAS, len(gene_names))).astype(np.float32)

    # Donor samples for atlas (used as batch covariate in Harmony).
    donors = [f"atlas_donor_{i:02d}" for i in range(1, 6)]
    samples = np.array([donors[i % len(donors)] for i in range(N_ATLAS)])

    obs = pd.DataFrame({
        "dataset":          "GBM_atlas",
        "cell_line":        "atlas",
        "sample":           samples,
        "guide_assignment": "atlas",
        "celltype":         fine_arr,
        "celltype_coarse":  coarse_arr,
        "topic":            topic_arr,
        "batch":            samples,
    }, index=[f"atlas_cell_{i:06d}" for i in range(N_ATLAS)])

    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)

    # Build Harmony embedding rows so atlas cells of the same fine state
    # cluster (state-specific mean vector + small noise).
    fine_to_idx = {s: i for i, s in enumerate(fine_states)}
    n_pcs = default_config.N_PCS
    state_means = rng.normal(0, 4.0, size=(len(fine_states), n_pcs))
    Z_atlas = state_means[[fine_to_idx[f] for f in fine_arr]]
    Z_atlas = Z_atlas + rng.normal(0, 0.5, size=Z_atlas.shape)
    return adata, Z_atlas.astype(np.float32)


def _build_screen(rng: np.random.Generator,
                  gene_names: list[str]) -> tuple[ad.AnnData, np.ndarray]:
    """Screen portion: N_SCREEN cells, each assigned to one of CELL_LINES and
    a guide (Ctrl ~10% or a TF). TF-targeted cells get a strong boost in
    the TF gene's expression so FC filters select something non-empty.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    cell_lines_arr = rng.choice(CELL_LINES, size=N_SCREEN)
    samples_arr = np.array([rng.choice(SAMPLES_PER_LINE[cl])
                            for cl in cell_lines_arr])

    # ~10% Ctrl, the rest distributed across the 55 TFs.
    n_ctrl = int(round(0.10 * N_SCREEN))
    n_oe = N_SCREEN - n_ctrl
    guides = ["Ctrl"] * n_ctrl + list(rng.choice(TFS, size=n_oe))
    rng.shuffle(guides)
    guides_arr = np.array(guides)

    # Baseline Poisson counts.
    X = rng.poisson(lam=2.0, size=(N_SCREEN, len(gene_names))).astype(np.float32)
    # Boost each TF's own expression in its OE cells by ~8x.
    for tf in TFS:
        gi = gene_to_idx[tf]
        oe_mask = guides_arr == tf
        if oe_mask.sum() == 0:
            continue
        boosted = rng.poisson(lam=16.0, size=oe_mask.sum()).astype(np.float32)
        X[oe_mask, gi] = boosted

    obs = pd.DataFrame({
        "dataset":              "TF_screen",
        "cell_line":            cell_lines_arr,
        "sample":               samples_arr,
        "guide_assignment":     guides_arr,
        "guide_has_assignment": True,
        "celltype":             pd.Categorical([None] * N_SCREEN,
                                               categories=sorted(_fine_to_coarse_map().keys())),
        "celltype_coarse":      pd.Categorical(
            [None] * N_SCREEN,
            categories=list(default_config.COARSE_GROUPS.keys())),
        "topic":                pd.Categorical([None] * N_SCREEN),
        "batch":                "OE",
    }, index=[f"screen_cell_{i:06d}" for i in range(N_SCREEN)])

    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)

    # Screen cells: noise around the centroid of one of the fine states
    # (random per cell). The classifier should still be able to predict
    # something coherent. Per-line shifts give batch correction a real
    # job to do.
    fine_states = sorted(_fine_to_coarse_map().keys())
    n_pcs = default_config.N_PCS
    state_means = rng.normal(0, 4.0, size=(len(fine_states), n_pcs))
    assign = rng.integers(0, len(fine_states), size=N_SCREEN)
    Z = state_means[assign] + rng.normal(0, 1.0, size=(N_SCREEN, n_pcs))
    return adata, Z.astype(np.float32)


def make_toy_data(out_dir: Path | None = None) -> dict[str, Path]:
    """Build the toy dataset on disk. Returns a dict of file paths written."""
    out_dir = out_dir or (_HERE / "toy_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    gene_names = _gene_names()

    print(f"Building atlas portion ({N_ATLAS} cells)...")
    atlas, Z_atlas = _build_atlas(rng, gene_names)
    print(f"Building screen portion ({N_SCREEN} cells)...")
    screen, Z_screen = _build_screen(rng, gene_names)

    # Joint AnnData: atlas concat screen, row order = atlas then screen.
    print("Concatenating atlas + screen into joint AnnData...")
    joint = ad.concat([atlas, screen], join="outer", merge="unique")
    # Ensure category dtypes stay consistent across the join.
    for col in ("dataset", "cell_line", "sample", "guide_assignment",
                "celltype", "celltype_coarse", "topic", "batch"):
        if col in joint.obs.columns:
            joint.obs[col] = joint.obs[col].astype(str)

    Z = np.vstack([Z_atlas, Z_screen]).astype(np.float32)
    assert Z.shape == (N_ATLAS + N_SCREEN, default_config.N_PCS)
    assert joint.n_obs == Z.shape[0]

    # Guide counts obsm for screen cells (random Poisson).
    guide_names = [f"sgRNA_{tf}" for tf in TFS] + ["sgRNA_Ctrl"]
    guide_counts_screen = rng.poisson(
        lam=1.0, size=(N_SCREEN, len(guide_names))).astype(np.int32)
    joint.uns["guide_names"] = guide_names

    # Pad guide_counts so .obsm rows align with the joint AnnData.
    guide_counts = np.zeros((joint.n_obs, len(guide_names)), dtype=np.int32)
    guide_counts[N_ATLAS:] = guide_counts_screen
    joint.obsm["guide_counts"] = guide_counts

    joint_path = out_dir / "joint_gbm_oe_anndata.h5ad"
    joint.write_h5ad(joint_path)
    print(f"  wrote {joint_path}  ({joint.n_obs} cells x {joint.n_vars} genes)")

    harmony_path = out_dir / "harmony_embeddings.npy"
    np.save(harmony_path, Z)
    print(f"  wrote {harmony_path}  shape={Z.shape}")

    # Standalone screen anndata for tf_self_expression.py and wilcoxon_de.py.
    screen_path = out_dir / "gbm_tf_screen_clean.h5ad"
    screen.write_h5ad(screen_path)
    print(f"  wrote {screen_path}  ({screen.n_obs} cells x {screen.n_vars} genes)")

    return {"joint": joint_path, "harmony": harmony_path, "screen": screen_path}


if __name__ == "__main__":
    paths = make_toy_data()
    print("\nDone. Wrote:")
    for k, p in paths.items():
        print(f"  {k:10s} -> {p}")
