"""Shared helpers used across the pipeline."""

from __future__ import annotations

import numpy as np
import scanpy as sc


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def detect_data_type(X, n_sample: int = 500, rng: np.random.Generator | None = None) -> str:
    """Decide whether ``X`` contains raw counts, library-size-normalised counts,
    or log1p-normalised counts.

    Returns one of ``"raw_counts"``, ``"normalised"`` or ``"lognorm"``.

    Decision is based on the coefficient of variation of per-cell library
    sizes (small means the matrix has already been normalised to a fixed
    target) and on the median non-zero value and integer fraction (which
    distinguish log1p-transformed data from linear normalised data).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    sample = X[idx].toarray() if hasattr(X, "toarray") else np.asarray(X[idx])

    row_sums = sample.sum(axis=1)
    rs_mean = float(row_sums.mean())
    rs_cv = float(row_sums.std() / rs_mean) if rs_mean > 0 else float("inf")

    non_zero = sample[sample > 0]
    frac_int = float(np.mean(np.abs(non_zero - np.round(non_zero)) < 1e-5)) if non_zero.size else 0.0
    med_nonzero = float(np.median(non_zero)) if non_zero.size else 0.0

    if rs_cv > 0.05:
        return "raw_counts"
    if med_nonzero < 5 and frac_int < 0.3:
        return "lognorm"
    return "normalised"


def normalise_lognorm(adata, target_sum: float = 1e4, copy: bool = False):
    """Normalise to ``target_sum`` counts per cell and log1p-transform.

    Idempotent: if ``adata.X`` is already log1p-normalised, returns it
    unchanged. Modifies in place unless ``copy=True``.
    """
    if copy:
        adata = adata.copy()
    if detect_data_type(adata.X) != "lognorm":
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    return adata


def compute_norm_factors(X) -> np.ndarray:
    """Per-cell scaling factors that bring each row sum to 10,000.

    ``normalised_count = raw_count * factor[cell_i]``.
    Use to apply FC / percentile filters on library-size-normalised
    expression without materialising a normalised copy of the matrix.
    """
    lib_sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    factors = np.zeros_like(lib_sizes)
    nonzero = lib_sizes > 0
    factors[nonzero] = 1e4 / lib_sizes[nonzero]
    return factors


# ---------------------------------------------------------------------------
# TF expression extraction
# ---------------------------------------------------------------------------

def get_tf_expression(X, cell_indices: np.ndarray, gene_idx: int,
                     norm_factors: np.ndarray | None = None) -> np.ndarray:
    """Return TF expression for ``cell_indices`` as a 1-D array.

    If ``norm_factors`` is provided, expression is returned in linear
    10k-normalised space (used for FC / percentile threshold checks).
    Otherwise the values from ``X`` are returned as-is.
    """
    expr = X[cell_indices, gene_idx]
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    expr = np.asarray(expr, dtype=np.float64).ravel()
    if norm_factors is not None:
        expr = expr * norm_factors[cell_indices]
    return expr


def select_oe_and_ctrl_indices(adata, tf: str, cell_line: str,
                               fc_threshold: float | None = 0.5,
                               control_percentile: float | None = 10,
                               norm_factors: np.ndarray | None = None
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Pick OE and Ctrl cell indices for one (TF, cell line) comparison.

    Both filters operate in 10k-normalised expression space if
    ``norm_factors`` is provided, which removes library-size bias from
    cell selection. ``fc_threshold=None`` skips the OE filter;
    ``control_percentile=None`` skips the Ctrl filter.

    Returns ``(oe_idx, ctrl_idx)`` as 1-D arrays of integer indices into
    ``adata.X``.
    """
    obs = adata.obs
    cl_mask = (obs["cell_line"] == cell_line).values
    oe_mask = cl_mask & (obs["guide_assignment"] == tf).values
    ctrl_mask = cl_mask & (obs["guide_assignment"] == "Ctrl").values

    oe_idx = np.where(oe_mask)[0]
    ctrl_idx = np.where(ctrl_mask)[0]
    if oe_idx.size == 0 or ctrl_idx.size == 0:
        return oe_idx, ctrl_idx

    if fc_threshold is None and control_percentile is None:
        return oe_idx, ctrl_idx

    if tf not in list(adata.var_names):
        return oe_idx, ctrl_idx
    gene_idx = list(adata.var_names).index(tf)

    factors = norm_factors if norm_factors is not None else compute_norm_factors(adata.X)
    ctrl_expr = get_tf_expression(adata.X, ctrl_idx, gene_idx, norm_factors=factors)
    oe_expr = get_tf_expression(adata.X, oe_idx, gene_idx, norm_factors=factors)

    if fc_threshold is not None:
        ctrl_mean = float(np.mean(ctrl_expr))
        if ctrl_mean > 0:
            keep_oe = (oe_expr / ctrl_mean) >= (1.0 + fc_threshold)
        else:
            keep_oe = oe_expr > 0
        oe_idx = oe_idx[keep_oe]

    if control_percentile is not None:
        cutoff = float(np.percentile(ctrl_expr, control_percentile))
        ctrl_idx = ctrl_idx[ctrl_expr <= cutoff]

    return oe_idx, ctrl_idx


# ---------------------------------------------------------------------------
# AnnData helpers
# ---------------------------------------------------------------------------

def list_tfs(adata, guide_col: str = "guide_assignment") -> list[str]:
    """Return the list of TF guide labels in ``adata``, sorted alphabetically.

    Excludes the non-targeting control and any unassigned categories.
    """
    return sorted(
        g for g in adata.obs[guide_col].unique()
        if g != "Ctrl" and not str(g).startswith("Unassign")
    )


def list_cell_lines(adata, col: str = "cell_line") -> list[str]:
    """Return the list of cell lines in ``adata``, sorted alphabetically."""
    return sorted(adata.obs[col].unique())
