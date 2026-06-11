"""Train the topic-level classifier on the 200-PC Harmony embedding.

A single multinomial L2-regularised logistic-regression model is fit on
the atlas cells of the joint AnnData. The training cells are atlas cells
that were marked as belonging to exactly one of the GB-intrinsic topics
in ``config.TOPICS_TRAINED`` (the ``topic`` column was populated by
``build_joint_anndata.py`` and contains values like ``"Topic_24"``).

The classifier produces per-cell probabilities over the trained topics
that sum to one per cell, giving TF-induced shifts in topic mass a
direct compositional interpretation.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
``HARMONY_DIR/harmony_embeddings.npy``

Outputs
-------
``MODELS_DIR/clf_topic.pkl``, ``scaler_topic.pkl``, ``le_topic.pkl``
``MODELS_DIR/topic_classifier_metrics.csv``
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Updated in-place with per-cell topic probability columns
    (``prob_topic_<Topic_N>``).

Notes
-----
* The classifier is trained on all topics in ``TOPICS_TRAINED``. The
  smaller set ``TOPICS_TESTED`` used for the perturbation analysis is
  enforced downstream (in ``state_testing/topic_testing.py``), not
  here -- excluded topics still receive a probability and still
  contribute to the softmax denominator.

Usage
-----
``python classifiers/train_topic_classifier.py``
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Training-cell selection
# ---------------------------------------------------------------------------

def _trained_topic_labels() -> set[str]:
    """Allowed values for ``obs['topic']`` (e.g. ``"Topic_24"``)."""
    return {f"Topic_{t}" for t in config.TOPICS_TRAINED}


def select_training_cells(adata: sc.AnnData) -> np.ndarray:
    """Atlas cells whose ``topic`` is one of the trained topics."""
    is_atlas = adata.obs["dataset"].astype(str).str.lower().str.contains("gbm", na=False).values
    allowed = _trained_topic_labels()
    has_valid_topic = adata.obs["topic"].isin(allowed).values
    return is_atlas & has_valid_topic


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_topic_classifier(adata: sc.AnnData
                           ) -> tuple[LogisticRegression, StandardScaler, LabelEncoder, dict]:
    """Fit a multinomial L2 logistic regression on a stratified 80/20
    split of atlas cells with a trained-topic label.
    """
    train_mask = select_training_cells(adata)
    n_train = int(train_mask.sum())
    print(f"   Training cells: {n_train:,}")

    X = adata.obsm["X_harmony"][train_mask]
    y = adata.obs.loc[train_mask, "topic"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"   Topics in training data: {len(le.classes_)} "
          f"({', '.join(le.classes_)})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=y_enc,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # NOTE: scikit-learn >=1.7 removed the multi_class kwarg; multinomial
    # (soft-max) is the default for the lbfgs solver, so omitting it
    # preserves the original behaviour while staying forward-compatible.
    model = LogisticRegression(
        solver=config.LR_SOLVER,
        max_iter=config.LR_MAX_ITER_TOPIC,
        C=config.LR_C,
        class_weight=config.LR_CLASS_WEIGHT,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    bal = balanced_accuracy_score(y_te, y_pred)
    print(f"   accuracy={acc:.4f}  balanced={bal:.4f}")

    metrics = {
        "n_train_cells": n_train,
        "n_topics_trained": int(len(le.classes_)),
        "topics_trained": ",".join(le.classes_),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "n_pcs": config.N_PCS,
    }
    return model, scaler, le, metrics


# ---------------------------------------------------------------------------
# Prediction on the full joint object
# ---------------------------------------------------------------------------

def predict_for_all_cells(adata: sc.AnnData,
                          model: LogisticRegression,
                          scaler: StandardScaler,
                          le: LabelEncoder) -> sc.AnnData:
    """Predict topic probabilities for every cell and attach as
    ``prob_topic_<Topic_N>`` columns. Probabilities sum to one per cell.
    """
    X = scaler.transform(adata.obsm["X_harmony"])
    P = model.predict_proba(X)
    adata.obs["predicted_topic"] = le.inverse_transform(model.predict(X))
    for i, name in enumerate(le.classes_):
        adata.obs[f"prob_topic_{name}"] = P[:, i]
    return adata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(model: LogisticRegression,
                scaler: StandardScaler,
                le: LabelEncoder,
                metrics: dict,
                out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  out_dir / "clf_topic.pkl")
    joblib.dump(scaler, out_dir / "scaler_topic.pkl")
    joblib.dump(le,     out_dir / "le_topic.pkl")
    pd.DataFrame([metrics]).to_csv(out_dir / "topic_classifier_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    joint_path = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    harmony_path = config.HARMONY_DIR / "harmony_embeddings.npy"

    print(f"Loading joint AnnData from {joint_path}...")
    adata = sc.read_h5ad(joint_path)
    print(f"Loading Harmony embedding from {harmony_path}...")
    adata.obsm["X_harmony"] = np.load(harmony_path)
    assert adata.obsm["X_harmony"].shape[0] == adata.n_obs, \
        f"Harmony embedding rows ({adata.obsm['X_harmony'].shape[0]}) " \
        f"do not match adata.n_obs ({adata.n_obs})"

    print("\nTraining topic classifier (multinomial)...")
    model, scaler, le, metrics = train_topic_classifier(adata)

    print("\nPredicting topic probabilities for all cells...")
    adata = predict_for_all_cells(adata, model, scaler, le)

    print("\nSaving model, metrics, and updated AnnData...")
    save_models(model, scaler, le, metrics, config.MODELS_DIR)
    adata.write_h5ad(joint_path)
    print(f"   Wrote models to {config.MODELS_DIR}")
    print(f"   Updated         {joint_path}")
    print("Done.")


if __name__ == "__main__":
    main()
