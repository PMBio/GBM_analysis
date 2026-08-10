"""Build the joint AnnData object used for batch correction and classification.

The joint object concatenates the GB patient multi-ome atlas (subset to
the union of cells appearing in only a single topic's top-5,000) with the
guide-assigned cells of the TF screen, restricted to a shared feature
space defined as the union of the top-``N_GENES_PER_TOPIC`` genes per
topic from the atlas plus the 55 TF guide-target genes (force-added so
that TF overexpression is always measurable).

Outputs
-------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Joint AnnData object (raw counts, obs columns: batch, topic, celltype,
    dataset, sample, cell_line, guide_assignment).
``JOINT_ANNDATA_DIR/feature_genes.csv``
    The shared gene list.
``JOINT_ANNDATA_DIR/joint_obs.csv``
    Cell metadata table (for QC and downstream lookup).

Usage
-----
``python preprocessing/build_joint_anndata.py``
"""

from __future__ import annotations

import warnings
from collections import Counter
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

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature gene set
# ---------------------------------------------------------------------------

def load_feature_genes(topic_genes_path: Path, topics: list[int], n_per_topic: int,
                      tf_symbols: list[str]) -> list[str]:
    """Return the union of top-``n_per_topic`` genes across ``topics``,
    plus any ``tf_symbols`` not already present.
    """
    rankings = pd.read_csv(topic_genes_path)
    genes: set[str] = set()
    for t in topics:
        top = rankings[rankings["topic"] == t].head(n_per_topic)["names"].tolist()
        genes.update(top)
    missing_tfs = [tf for tf in tf_symbols if tf not in genes]
    if missing_tfs:
        print(f"   TFs not in topic-feature set, force-adding: {missing_tfs}")
    genes.update(missing_tfs)
    return sorted(genes)


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

def load_atlas_with_annotations() -> sc.AnnData:
    """Load atlas RNA AnnData and attach the merged coarse-state label
    (``celltype`` column derived from ``annotation_coarse``).
    """
    adata = sc.read_h5ad(config.ATLAS_RNA_ANNDATA)
    obs_df = pd.read_csv(config.ATLAS_ANNOTATION_OBS)
    adata.obs = adata.obs.merge(obs_df, how="left", on="cell_id")
    adata.obs["cell_id_new"] = adata.obs["cell_id"].values
    adata.obs = adata.obs.set_index("cell_id")

    # Merge GBM and TME state-name tables into one annotation map.
    annot_gbm = pd.read_csv(config.ANNOT_GBM_TSV, sep="\t")
    annot_gbm["malignant"] = "GBM"
    annot_tme = pd.read_csv(config.ANNOT_TME_TSV, sep="\t")
    annot_tme["malignant"] = "TME"
    annot_tme = annot_tme[["TME_GBM_granular", "annotation_coarse", "malignant"]]
    annot_gbm.columns = annot_tme.columns
    annot_map = pd.concat([annot_tme, annot_gbm])

    adata.obs = adata.obs.merge(annot_map, on="TME_GBM_granular", how="left")
    adata.obs["celltype"] = adata.obs["annotation_coarse"].fillna("unknown")
    adata.obs = adata.obs.set_index("cell_id_new")
    adata.obs.index = adata.obs.index.astype(str)
    return adata


def filter_atlas_to_exclusive_topic_cells(adata: sc.AnnData,
                                          top_cells_path: Path,
                                          topics: list[int]) -> sc.AnnData:
    """Restrict the atlas to cells that appear in exactly one topic's
    top-5,000 list (exclusive cells), and attach a ``topic`` column.
    """
    df_topic = pd.read_csv(top_cells_path, sep="\t")
    cell_to_topic: dict[str, str] = {}
    cell_counts: Counter[str] = Counter()
    for t in topics:
        col = f"Topic_{t}"
        if col not in df_topic.columns:
            continue
        for cell in df_topic[col].dropna():
            cell_counts[cell] += 1
            cell_to_topic[cell] = col
    exclusive = {c: cell_to_topic[c] for c in cell_to_topic if cell_counts[c] == 1}
    keep = [c for c in exclusive if c in adata.obs.index]
    adata = adata[keep].copy()
    adata.obs["topic"] = [exclusive[c] for c in keep]
    adata.obs["batch"] = adata.obs["sample"].astype(str) if "sample" in adata.obs.columns else "atlas"
    adata.obs["dataset"] = "GBM_atlas"
    return adata


# ---------------------------------------------------------------------------
# OE screen
# ---------------------------------------------------------------------------

def load_screen_assigned_cells() -> sc.AnnData:
    """Load the cleaned screen AnnData and restrict to guide-assigned cells."""
    adata = sc.read_h5ad(config.SCREEN_ANNDATA)

    if "cell_line" not in adata.obs.columns:
        adata.obs["cell_line"] = adata.obs["sample"].str.split("-").str[0]

    if "guide_assignment" in adata.obs.columns:
        mask = ~adata.obs["guide_assignment"].astype(str).str.startswith("Unassign")
        adata = adata[mask].copy()

    adata.obs["batch"] = "OE"
    adata.obs["topic"] = "unknown"
    adata.obs["celltype"] = "unknown"
    adata.obs["dataset"] = "TF_screen"
    return adata


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------

def concat_to_shared_features(atlas: sc.AnnData, screen: sc.AnnData,
                              feature_genes: list[str]) -> sc.AnnData:
    """Subset atlas and screen to the shared feature space, harmonise obs
    columns, and concatenate.
    """
    common = sorted(set(feature_genes) & set(atlas.var_names) & set(screen.var_names))
    print(f"   Shared features after intersection: {len(common):,}")
    atlas = atlas[:, common].copy()
    screen = screen[:, common].copy()

    keep_cols = ["batch", "topic", "celltype", "dataset"]
    if "sample" in atlas.obs.columns:
        keep_cols.append("sample")
    if "cell_line" in screen.obs.columns:
        keep_cols.append("cell_line")
    if "guide_assignment" in screen.obs.columns:
        keep_cols.append("guide_assignment")

    for col in keep_cols:
        if col not in atlas.obs.columns:
            atlas.obs[col] = "unknown"
        if col not in screen.obs.columns:
            screen.obs[col] = "unknown"

    atlas.obs = atlas.obs[keep_cols].copy()
    screen.obs = screen.obs[keep_cols].copy()
    atlas.var = atlas.var[[]]
    screen.var = screen.var[[]]

    joint = sc.concat([atlas, screen], join="outer", label="source",
                      keys=["GBM", "OE"])
    if joint.obs.index.duplicated().any():
        joint.obs_names_make_unique()
    return joint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = config.JOINT_ANNDATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading screen and identifying TF symbols...")
    screen = load_screen_assigned_cells()
    tf_symbols = sorted(
        g for g in screen.obs["guide_assignment"].unique()
        if g != "Ctrl" and not str(g).startswith("Unassign")
    )
    print(f"   Screen cells (assigned): {screen.n_obs:,}")
    print(f"   TFs: {len(tf_symbols)}")

    print("\n2. Loading atlas with annotations...")
    atlas = load_atlas_with_annotations()
    print(f"   Atlas cells (raw): {atlas.n_obs:,}")

    print("\n3. Filtering atlas to exclusive topic cells...")
    atlas = filter_atlas_to_exclusive_topic_cells(
        atlas, config.ATLAS_TOPIC_TOP_CELLS, config.TOPICS_TRAINED,
    )
    print(f"   Atlas cells (exclusive): {atlas.n_obs:,}")

    print("\n4. Building shared feature gene set...")
    feature_genes = load_feature_genes(
        config.ATLAS_TOPIC_GENES,
        config.TOPICS_TRAINED,
        config.N_GENES_PER_TOPIC,
        tf_symbols,
    )
    print(f"   Feature genes (union of top-{config.N_GENES_PER_TOPIC} per topic + TFs):"
          f" {len(feature_genes):,}")

    print("\n5. Concatenating atlas + screen to shared feature space...")
    joint = concat_to_shared_features(atlas, screen, feature_genes)
    print(f"   Joint object: {joint.n_obs:,} cells x {joint.n_vars:,} genes")
    print(f"   Atlas:  {(joint.obs['dataset'] == 'GBM_atlas').sum():,}")
    print(f"   Screen: {(joint.obs['dataset'] == 'TF_screen').sum():,}")
    print(f"   Batches: {joint.obs['batch'].nunique()}")

    print("\n6. Saving...")
    joint.write_h5ad(out_dir / "joint_gbm_oe_anndata.h5ad")
    pd.DataFrame({"gene": joint.var_names}).to_csv(out_dir / "feature_genes.csv", index=False)
    joint.obs.to_csv(out_dir / "joint_obs.csv")
    print(f"   Wrote {out_dir / 'joint_gbm_oe_anndata.h5ad'}")
    print("Done.")


if __name__ == "__main__":
    main()
