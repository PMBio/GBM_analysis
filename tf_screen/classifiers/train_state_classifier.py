"""Train the two state classifiers (fine, 9 classes; coarse, 5 classes).

Both classifiers are multinomial L2-regularised logistic-regression models
fit on the 200-PC Harmony embedding of atlas cells. Training cells are
the atlas (``dataset == "GBM_atlas"``) subset of the joint AnnData,
restricted to cells with a valid topic and a malignant cell-state label.

Coarse states are defined by ``config.COARSE_GROUPS``; the *Proliferative*
state is excluded from both training and testing (it is a transient
cell-cycle programme, not a stable cell-state identity), via
``EXCLUDE_CELLTYPES``.

After training, predictions (per-cell probabilities for every fine and
coarse state) are written back into the joint AnnData ``.obs`` table for
all cells (atlas + screen), and the joint object is re-saved.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
``HARMONY_DIR/harmony_embeddings.npy``

Outputs
-------
``MODELS_DIR/clf_celltype_fine.pkl``, ``scaler_celltype_fine.pkl``, ``le_celltype_fine.pkl``
``MODELS_DIR/clf_celltype_coarse.pkl``, ``scaler_celltype_coarse.pkl``, ``le_celltype_coarse.pkl``
``MODELS_DIR/state_classifier_metrics.csv``
    Train/test accuracy and balanced accuracy for both classifiers.
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Updated in-place with per-cell state probability columns
    (``prob_ct_<fine>`` and ``prob_coarse_<coarse>``).

Usage
-----
``python classifiers/train_state_classifier.py``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

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
# Helpers
# ---------------------------------------------------------------------------

def _is_atlas(adata: sc.AnnData) -> np.ndarray:
    return adata.obs["dataset"].astype(str).str.lower().str.contains("gbm", na=False).values


def _coarse_map() -> Mapping[str, str]:
    """Build a fine -> coarse lookup dict from ``config.COARSE_GROUPS``."""
    return {fine: coarse for coarse, fines in config.COARSE_GROUPS.items() for fine in fines}


def _excluded_cell_types() -> set[str]:
    """Fine cell types that should be excluded from training entirely.

    The coarse map covers every state we want to model. Anything outside
    its keys is excluded from training; this naturally drops TME / atlas
    fine states (lymphocytes, oligodendrocytes etc.) and ``Proliferative``.
    """
    keep = {fine for fines in config.COARSE_GROUPS.values() for fine in fines}
    return {"unknown", "Unknown"} | {"Proliferative"} | _atlas_extras() - keep


def _atlas_extras() -> set[str]:
    """Other atlas labels the screen project explicitly drops from training."""
    return {
        "Macrophages", "Lymphocytes", "Vascular-associated",
        "Oligodendrocytes", "Astrocytes", "Undefined",
        "Neurons (Inh)", "Neurons (Exc)",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def select_training_cells(adata: sc.AnnData) -> np.ndarray:
    """Boolean mask of atlas cells with a valid (kept) coarse-mappable
    fine-state label and a non-missing topic.
    """
    is_atlas = _is_atlas(adata)
    has_topic = (adata.obs["topic"].notna() & (adata.obs["topic"] != "unknown")).values
    keep_fine = set(_coarse_map().keys())  # only fine states that map to a coarse state
    valid_celltype = adata.obs["celltype"].isin(keep_fine).values
    return is_atlas & has_topic & valid_celltype


def _fit_logreg(X: np.ndarray, y: np.ndarray,
                random_state: int = config.RANDOM_SEED
                ) -> tuple[LogisticRegression, StandardScaler, LabelEncoder, float, float]:
    """Fit a multinomial L2 logistic regression on a stratified 80/20 split.

    Returns the fitted ``(model, scaler, label_encoder)`` plus the test-set
    accuracy and balanced accuracy.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=random_state,
        stratify=y_enc,
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(
        multi_class="multinomial",
        solver=config.LR_SOLVER,
        max_iter=config.LR_MAX_ITER_STATE,
        C=config.LR_C,
        class_weight=config.LR_CLASS_WEIGHT,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)
    return model, scaler, le, accuracy_score(y_te, y_pred), balanced_accuracy_score(y_te, y_pred)


def train_state_classifiers(adata: sc.AnnData
                           ) -> tuple[dict, dict, dict]:
    """Train the fine and coarse state classifiers and return both plus
    a metrics summary.
    """
    train_mask = select_training_cells(adata)
    print(f"   Training cells: {train_mask.sum():,}")

    X_train = adata.obsm["X_harmony"][train_mask]
    y_fine = adata.obs.loc[train_mask, "celltype"].values

    # --- Fine (9 classes) ---
    print("\n   Fitting fine classifier (multinomial, 9 classes)...")
    clf_f, sc_f, le_f, acc_f, bal_f = _fit_logreg(X_train, y_fine)
    print(f"     classes={len(le_f.classes_)}  accuracy={acc_f:.4f}  balanced={bal_f:.4f}")

    # --- Coarse (5 classes) ---
    fine_to_coarse = _coarse_map()
    y_coarse = np.array([fine_to_coarse.get(c, "Unknown") for c in y_fine])
    mask_c = y_coarse != "Unknown"
    X_coarse = X_train[mask_c]
    y_coarse = y_coarse[mask_c]

    print("\n   Fitting coarse classifier (multinomial, 5 classes)...")
    clf_c, sc_c, le_c, acc_c, bal_c = _fit_logreg(X_coarse, y_coarse)
    print(f"     classes={len(le_c.classes_)}  accuracy={acc_c:.4f}  balanced={bal_c:.4f}")

    fine = {"clf": clf_f, "scaler": sc_f, "le": le_f,
            "accuracy": acc_f, "balanced_accuracy": bal_f}
    coarse = {"clf": clf_c, "scaler": sc_c, "le": le_c,
              "accuracy": acc_c, "balanced_accuracy": bal_c}
    metrics = {
        "n_train_cells_fine": int(train_mask.sum()),
        "n_train_cells_coarse": int(mask_c.sum()),
        "fine_accuracy": acc_f, "fine_balanced_accuracy": bal_f,
        "coarse_accuracy": acc_c, "coarse_balanced_accuracy": bal_c,
        "n_pcs": config.N_PCS,
    }
    return fine, coarse, metrics


# ---------------------------------------------------------------------------
# Prediction on the full joint object
# ---------------------------------------------------------------------------

def predict_for_all_cells(adata: sc.AnnData, fine: dict, coarse: dict) -> sc.AnnData:
    """Predict fine and coarse state probabilities for every cell in
    ``adata`` and attach them as ``obs`` columns:

    * ``prob_ct_<fine_state>`` for each fine class
    * ``prob_coarse_<coarse_state>`` for each coarse class
    * ``predicted_celltype`` and ``predicted_coarse`` (argmax labels)
    """
    X = adata.obsm["X_harmony"]

    Xf = fine["scaler"].transform(X)
    Pf = fine["clf"].predict_proba(Xf)
    adata.obs["predicted_celltype"] = fine["le"].inverse_transform(fine["clf"].predict(Xf))
    for i, name in enumerate(fine["le"].classes_):
        adata.obs[f"prob_ct_{name}"] = Pf[:, i]

    Xc = coarse["scaler"].transform(X)
    Pc = coarse["clf"].predict_proba(Xc)
    adata.obs["predicted_coarse"] = coarse["le"].inverse_transform(coarse["clf"].predict(Xc))
    for i, name in enumerate(coarse["le"].classes_):
        adata.obs[f"prob_coarse_{name}"] = Pc[:, i]
    return adata


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(fine: dict, coarse: dict, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fine["clf"],    out_dir / "clf_celltype_fine.pkl")
    joblib.dump(fine["scaler"], out_dir / "scaler_celltype_fine.pkl")
    joblib.dump(fine["le"],     out_dir / "le_celltype_fine.pkl")
    joblib.dump(coarse["clf"],    out_dir / "clf_celltype_coarse.pkl")
    joblib.dump(coarse["scaler"], out_dir / "scaler_celltype_coarse.pkl")
    joblib.dump(coarse["le"],     out_dir / "le_celltype_coarse.pkl")
    pd.DataFrame([metrics]).to_csv(out_dir / "state_classifier_metrics.csv", index=False)


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

    fine, coarse, metrics = train_state_classifiers(adata)
    print("\nPredicting state probabilities for all cells...")
    adata = predict_for_all_cells(adata, fine, coarse)

    print("\nSaving models, metrics, and updated AnnData...")
    save_models(fine, coarse, metrics, config.MODELS_DIR)
    adata.write_h5ad(joint_path)
    print(f"   Wrote models to {config.MODELS_DIR}")
    print(f"   Updated         {joint_path}")
    print("Done.")


if __name__ == "__main__":
    main()
