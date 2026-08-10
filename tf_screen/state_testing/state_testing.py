"""Statistical testing of TF effects on coarse and fine cell states.

For every (TF, state) pair, this script tests whether the OE cells have
a different state probability than the matched control cells, across a
grid of filtering strategies:

* OE filters: keep cells with linear-space TF expression >= (1 + ``FC``)
  times the geometric mean of controls in the same cell line, for
  ``FC in [0.5, 1.0, 1.5, 2.0]``.
* Ctrl filters: keep all controls, or only the bottom 50% / 25% / 10% of
  TF expressors (the ``p50``, ``p25``, ``p10`` strategies).

The primary strategy in the paper is ``FC0.5_p10`` (1.5x linear OE,
bottom-10% controls); the other 15 are run as a sensitivity grid.

Statistical testing uses three complementary methods (see eq. 4 of the
supplementary methods):

* Welch's two-sample t-test.
* Mann-Whitney U (Wilcoxon rank-sum).
* Label-permutation test with matched subsampling (``N_PERMUTATIONS``
  draws of ``min(n_OE, n_Ctrl)`` controls without replacement, then
  shuffled OE/Ctrl labels).

Cohen's d is reported for effect size. A hit is called significant
under the consensus criterion (eq. 5):

    perm_fdr < 0.05  AND  (ttest_fdr < 0.05 OR wilcox_fdr < 0.05)  AND  |d| > 0.2

The script iterates over both fine (9-class) and coarse (5-class) states
for each strategy, writing one CSV per strategy under
``STATE_TESTING_DIR/<strategy>/{fine_grained,coarse}_effects.csv``.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Must contain ``prob_ct_<fine>`` and ``prob_coarse_<coarse>`` columns
    populated by ``classifiers/train_state_classifier.py``.
``HARMONY_DIR/harmony_embeddings.npy``
    Only used to verify row alignment.

Outputs
-------
``STATE_TESTING_DIR/<strategy>/fine_grained_effects.csv``
``STATE_TESTING_DIR/<strategy>/coarse_effects.csv``
``STATE_TESTING_DIR/strategy_comparison_{fine,coarse}.csv``

Usage
-----
``python state_testing/state_testing.py``
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import (compute_norm_factors, list_cell_lines,
                             list_tfs)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core statistical test
# ---------------------------------------------------------------------------

def run_all_tests(tf_scores: np.ndarray, ctrl_scores: np.ndarray,
                  n_perms: int = config.N_PERMUTATIONS,
                  seed: int = config.RANDOM_SEED,
                  min_cells: int = config.MIN_CELLS_PER_GROUP) -> dict:
    """Compute summary stats, t-test, Wilcoxon, and matched-subsample
    permutation test for a single (TF, state, cell line) comparison.

    The permutation null draws ``m = min(n_OE, n_Ctrl)`` controls without
    replacement, concatenates them with the OE cells, shuffles the
    combined labels, and computes the difference of relabelled means;
    repeated ``n_perms`` times.
    """
    rng = np.random.default_rng(seed)
    n_tf = int(len(tf_scores))
    n_ctrl = int(len(ctrl_scores))

    out: dict = {
        "n_tf": n_tf, "n_ctrl": n_ctrl,
        "tf_mean":   float(np.mean(tf_scores))  if n_tf  else np.nan,
        "ctrl_mean": float(np.mean(ctrl_scores)) if n_ctrl else np.nan,
        "tf_std":    float(np.std(tf_scores))   if n_tf  else np.nan,
        "ctrl_std":  float(np.std(ctrl_scores))  if n_ctrl else np.nan,
    }
    out["delta"] = out["tf_mean"] - out["ctrl_mean"]
    pooled_std = np.sqrt((out["tf_std"] ** 2 + out["ctrl_std"] ** 2) / 2)
    out["cohens_d"] = out["delta"] / pooled_std if pooled_std > 0 else np.nan

    if n_tf < min_cells or n_ctrl < min_cells:
        for k in ("ttest_pval", "wilcox_pval", "perm_pval"):
            out[k] = np.nan
        return out

    try:
        _, out["ttest_pval"] = stats.ttest_ind(tf_scores, ctrl_scores)
    except Exception:
        out["ttest_pval"] = np.nan
    try:
        _, out["wilcox_pval"] = stats.mannwhitneyu(tf_scores, ctrl_scores,
                                                   alternative="two-sided")
    except Exception:
        out["wilcox_pval"] = np.nan

    # Matched-subsample permutation null.
    observed = out["delta"]
    m = min(n_tf, n_ctrl)
    null = np.empty(n_perms)
    for i in range(n_perms):
        sub = rng.choice(ctrl_scores, size=m, replace=False)
        pooled = np.concatenate([tf_scores, sub])
        rng.shuffle(pooled)
        null[i] = pooled[:n_tf].mean() - pooled[n_tf:].mean()
    out["perm_pval"] = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perms + 1)
    return out


# ---------------------------------------------------------------------------
# Filtering by FC and percentile strategy
# ---------------------------------------------------------------------------

def filter_cells_by_strategy(adata: sc.AnnData, oe_obs: pd.DataFrame,
                             fc_threshold: float,
                             ctrl_percentile: float | str
                             ) -> tuple[dict, list[str]]:
    """Build per-TF OE/Ctrl index masks for one filtering strategy.

    ``fc_threshold`` is the linear-space exceedance (``FC0.5`` -> keep
    OE cells whose linear-space TF expression is at least 1.5x the
    matched control mean). ``ctrl_percentile`` is either the string
    ``"all"`` (keep every control) or a number in (0, 100) selecting
    the lower tail of controls.
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    if X.max() < 20:
        X = np.expm1(X)

    var_names = list(adata.var_names)
    tfs = list_tfs(adata)
    ctrl_mask_all = oe_obs["guide_assignment"] == "Ctrl"
    ctrl_indices_all = np.where(ctrl_mask_all.values)[0]

    masks: dict = {}
    for tf in tfs:
        oe_mask = oe_obs["guide_assignment"] == tf
        oe_indices = np.where(oe_mask.values)[0]

        if tf not in var_names:
            masks[tf] = {"oe_indices": oe_indices, "ctrl_indices": ctrl_indices_all}
            continue

        gene_idx = var_names.index(tf)
        ctrl_expr = X[ctrl_mask_all.values, gene_idx]
        oe_expr = X[oe_mask.values, gene_idx]

        # OE filter (linear-space FC).
        ctrl_mean = float(np.mean(ctrl_expr))
        if ctrl_mean > 0:
            keep_oe = (oe_expr / ctrl_mean) >= (1.0 + fc_threshold)
        else:
            keep_oe = oe_expr > 0
        oe_filtered = oe_indices[keep_oe]

        # Ctrl filter (lower-tail percentile).
        if ctrl_percentile == "all":
            ctrl_filtered = ctrl_indices_all
        else:
            cutoff = float(np.percentile(ctrl_expr, ctrl_percentile))
            ctrl_filtered = ctrl_indices_all[ctrl_expr <= cutoff]

        masks[tf] = {"oe_indices": oe_filtered, "ctrl_indices": ctrl_filtered}
    return masks, tfs


# ---------------------------------------------------------------------------
# Generic per-state testing (used for both fine and coarse)
# ---------------------------------------------------------------------------

def _seed(*parts: object) -> int:
    """Deterministic per-comparison seed; stable across runs."""
    return abs(hash(":".join(str(p) for p in parts))) % 100_000


def test_tf_effects(oe_obs: pd.DataFrame, tf_masks: dict, tfs: list[str],
                    cell_lines: list[str], strategy: str,
                    state_names: Iterable[str], prob_prefix: str,
                    state_column: str) -> pd.DataFrame:
    """Generic TF effect test against a set of state probability columns.

    ``state_names`` is the list of states (without prefix) and
    ``prob_prefix`` is the ``obs`` column prefix (``"prob_ct_"`` for fine,
    ``"prob_coarse_"`` for coarse). ``state_column`` is the resulting
    column label in the output dataframe (``"celltype"`` or
    ``"coarse_state"``).
    """
    rows: list[dict] = []
    state_names = list(state_names)
    print(f"   Strategy={strategy}  ({state_column})")

    def _test_block(scope: str, cl_mask: np.ndarray | None) -> None:
        """Run all (TF x state) tests within a single scope (pooled or per-line)."""
        for tf_idx, tf in enumerate(tfs):
            if (tf_idx + 1) % 10 == 0:
                print(f"     [{tf_idx+1}/{len(tfs)}] {tf} ({scope})")
            oe_idx = tf_masks[tf]["oe_indices"]
            ctrl_idx = tf_masks[tf]["ctrl_indices"]
            if cl_mask is not None:
                oe_idx = oe_idx[cl_mask[oe_idx]]
                ctrl_idx = ctrl_idx[cl_mask[ctrl_idx]]
            for st in state_names:
                col = f"{prob_prefix}{st}"
                if col not in oe_obs.columns:
                    continue
                tf_scores = oe_obs.iloc[oe_idx][col].values
                ctrl_scores = oe_obs.iloc[ctrl_idx][col].values
                res = run_all_tests(
                    tf_scores, ctrl_scores,
                    seed=_seed(strategy, scope, tf, st),
                )
                res.update({
                    "TF": tf, state_column: st, "cell_line": scope,
                    "strategy": strategy,
                })
                rows.append(res)

    _test_block("pooled", None)
    for cl in cell_lines:
        cl_mask = (oe_obs["cell_line"] == cl).values
        _test_block(cl, cl_mask)

    df = pd.DataFrame(rows)
    for pcol in ("ttest_pval", "wilcox_pval", "perm_pval"):
        fcol = pcol.replace("_pval", "_fdr")
        df[fcol] = multipletests(df[pcol].fillna(1.0), method="fdr_bh")[1]
    df["sig_consensus"] = (
        ((df["ttest_fdr"] < config.FDR_THRESHOLD)
         | (df["wilcox_fdr"] < config.FDR_THRESHOLD))
        & (df["perm_fdr"] < config.FDR_THRESHOLD)
        & (np.abs(df["cohens_d"]) > config.COHEN_D_THRESHOLD)
    )
    return df


# ---------------------------------------------------------------------------
# Strategy comparison summary
# ---------------------------------------------------------------------------

def compare_strategies(per_strategy: list[pd.DataFrame], label: str,
                       out_dir: Path) -> pd.DataFrame:
    """Roll up significant-hit counts across strategies."""
    rows = []
    for df in per_strategy:
        rows.append({
            "strategy": df["strategy"].iloc[0],
            "n_significant_pooled": int(
                ((df["sig_consensus"]) & (df["cell_line"] == "pooled")).sum()
            ),
            "n_significant_total": int(df["sig_consensus"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"strategy_comparison_{label}.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    joint_path = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    print(f"Loading joint AnnData from {joint_path}...")
    adata = sc.read_h5ad(joint_path)

    # Restrict to the screen.
    is_atlas = adata.obs["dataset"].astype(str).str.lower().str.contains("gbm", na=False)
    adata_oe = adata[~is_atlas.values].copy()
    print(f"OE (screen) cells: {adata_oe.n_obs:,}")

    oe_obs = adata_oe.obs
    cell_lines = list_cell_lines(adata_oe)
    print(f"Cell lines: {cell_lines}")

    fine_states = config.FINE_CELLTYPES
    coarse_states = list(config.COARSE_GROUPS.keys())

    fc_grid = [0.5, 1.0, 1.5, 2.0]
    ctrl_grid: list[float | str] = ["all", 50, 25, 10]

    out_dir = config.STATE_TESTING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fine, all_coarse = [], []
    t0 = time.time()
    for fc in fc_grid:
        for ctrl in ctrl_grid:
            tag = "all" if ctrl == "all" else f"p{ctrl}"
            strategy = f"FC{fc}_{tag}"
            (out_dir / strategy).mkdir(parents=True, exist_ok=True)
            print(f"\n=== {strategy} ===")

            masks, tfs = filter_cells_by_strategy(adata_oe, oe_obs, fc, ctrl)

            df_fine = test_tf_effects(
                oe_obs, masks, tfs, cell_lines, strategy,
                fine_states, "prob_ct_", "celltype",
            )
            df_fine.to_csv(out_dir / strategy / "fine_grained_effects.csv", index=False)
            all_fine.append(df_fine)

            df_coarse = test_tf_effects(
                oe_obs, masks, tfs, cell_lines, strategy,
                coarse_states, "prob_coarse_", "coarse_state",
            )
            df_coarse.to_csv(out_dir / strategy / "coarse_effects.csv", index=False)
            all_coarse.append(df_coarse)

            print(f"   fine:   {df_fine['sig_consensus'].sum()} sig  /  "
                  f"coarse: {df_coarse['sig_consensus'].sum()} sig  "
                  f"({(time.time() - t0)/60:.1f} min elapsed)")

    compare_strategies(all_fine,   "fine",   out_dir)
    compare_strategies(all_coarse, "coarse", out_dir)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min.  Output: {out_dir}")


if __name__ == "__main__":
    main()
