"""Compositional / proportion testing as an orthogonal confirmation of
the probability-based state effects.

Each OE cell is hard-assigned to its single most-probable coarse state
(argmax over the five non-Proliferative coarse states). Cells whose
argmax over **all six** states is Proliferative are dropped rather than
re-assigned to a runner-up, because Proliferative is a transient cell-
cycle programme; mixing it into a stable-state composition test would
contaminate the comparison.

For every (TF, cell line) and pooled, two complementary compositional
tests are run:

* **Multinomial likelihood-ratio test** (per TF, one global p-value):
  fits ``state ~ is_OE + cell_line`` against ``state ~ cell_line``
  (pooled) or ``state ~ is_OE`` against the intercept-only null
  (per cell line). Asks "does this TF change the overall composition?"
* **Fisher's exact test** (per TF, per state, post-hoc): 2x2 contingency
  of (OE/Ctrl) x (in state / not in state). Reports odds ratio and the
  ``delta_prop = p_OE - p_Ctrl`` for interpretability. The ``OR > 1.5``
  threshold is the "black-box" overlay annotation in Fig. 3c.

Both tests are applied across the same 16-strategy FC x ctrl grid as in
``state_testing.py``. The headline analysis uses ``FC0.5_p10``.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Must contain ``prob_coarse_<state>`` columns from
    ``classifiers/train_state_classifier.py``.

Outputs
-------
``PROPORTION_DIR/<strategy>/proportion_lrt.csv``
    One row per (TF, cell_line) with LRT statistic, p, FDR, and the
    per-state log-OR coefficients from the full multinomial model.
``PROPORTION_DIR/<strategy>/proportion_fisher.csv``
    One row per (TF, state, cell_line) with Fisher's odds ratio,
    p, FDR, and ``delta_prop``.
``PROPORTION_DIR/all_strategies_{lrt,fisher}.csv``
    Concatenation across strategies.
``PROPORTION_DIR/strategy_comparison/`` summary plots.

Usage
-----
``python state_testing/proportion_testing.py``
"""

import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Local config alias
# ---------------------------------------------------------------------------
# The original code's body references ``Config.<KEY>``; keep a local class
# pointing at the canonical config values so the body can stay verbatim.


class Config:
    ANNDATA_PATH = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    BASE_DIR     = config.OUTPUT_DIR
    OUT_DIR      = config.PROPORTION_DIR

    FC_THRESHOLDS       = [0.5, 1.0, 1.5, 2.0]
    CONTROL_PERCENTILES = ["all", 50, 25, 10]

    MIN_CELLS = config.MIN_CELLS_PER_GROUP // 2   # 10 in original
    COARSE_STATES = list(config.COARSE_GROUPS.keys())   # 5 non-Proliferative
    PROLIFERATIVE = "Proliferative"
    FDR_THRESHOLD = config.FDR_THRESHOLD


Config.OUT_DIR.mkdir(exist_ok=True, parents=True)
(Config.OUT_DIR / "strategy_comparison").mkdir(exist_ok=True)
# =====================================================================
# FILTER CELLS BY STRATEGY
# (identical logic to statistical_testing_multiStrategy.py)
# =====================================================================

def filter_cells_by_strategy(adata_oe, oe_obs, fc_threshold, ctrl_percentile):
    """
    Filter OE cells by TF expression FC threshold, and control cells by
    expression percentile. Logic is IDENTICAL to the existing multi-strategy
    statistical testing script.

    OE filtering
    ------------
    For each TF gene found in the expression matrix:
        fc = oe_expr_linear / ctrl_mean_linear
        keep OE cells where fc >= (1 + fc_threshold)
    If TF gene not in expression matrix: keep all OE cells for that TF.

    Control filtering
    -----------------
    ctrl_percentile == 'all'  → keep all control cells
    ctrl_percentile == N      → keep the BOTTOM N% of controls by TF expression
                                (i.e. controls with low baseline TF expression)

    Parameters
    ----------
    adata_oe       : AnnData subset to OE cells (with X_harmony already loaded)
    oe_obs         : adata_oe.obs (DataFrame)
    fc_threshold   : float, linear-space threshold (e.g. 1.0 → fc >= 2.0×)
    ctrl_percentile: 'all' or int (10 / 25 / 50)

    Returns
    -------
    tf_filtered_masks : dict  {tf_name: {'oe_indices': array, 'ctrl_indices': array}}
                        Indices are INTEGER POSITIONS into oe_obs.
    tfs               : list of TF names (sorted)
    """
    X = adata_oe.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Undo log normalization to get linear-space expression
    if X.max() < 20:      # data is log-normalised (ln scale)
        X = np.expm1(X)

    var_names = list(adata_oe.var_names)

    tfs = sorted([
        g for g in oe_obs['guide_assignment'].unique()
        if g != 'Ctrl' and not str(g).startswith('Unassign')
    ])

    tf_filtered_masks = {}
    ctrl_mask_all = oe_obs['guide_assignment'] == 'Ctrl'

    for tf in tfs:

        if tf not in var_names:
            # TF not in expression matrix — no expression-based filtering
            tf_filtered_masks[tf] = {
                'oe_indices':   np.where(oe_obs['guide_assignment'] == tf)[0],
                'ctrl_indices': np.where(ctrl_mask_all)[0]
            }
            continue

        gene_idx = var_names.index(tf)

        # Control expression for this TF gene
        ctrl_expr = X[ctrl_mask_all.values, gene_idx]
        ctrl_mean = np.mean(ctrl_expr)

        # OE expression for this TF
        oe_mask  = oe_obs['guide_assignment'] == tf
        oe_expr  = X[oe_mask.values, gene_idx]

        # FC filter (linear space)
        if ctrl_mean > 0:
            fc = oe_expr / ctrl_mean
            high_expr = fc >= (1 + fc_threshold)
        else:
            high_expr = oe_expr > 0   # fallback if ctrl mean is 0

        oe_indices  = np.where(oe_mask)[0]
        oe_filtered = oe_indices[high_expr]

        # Control strategy
        if ctrl_percentile == 'all':
            ctrl_filtered = np.where(ctrl_mask_all)[0]
        else:
            # Keep BOTTOM ctrl_percentile% (low-expressing controls)
            pct_cutoff   = np.percentile(ctrl_expr, ctrl_percentile)
            ctrl_low     = ctrl_expr <= pct_cutoff
            ctrl_indices = np.where(ctrl_mask_all)[0]
            ctrl_filtered = ctrl_indices[ctrl_low]

        tf_filtered_masks[tf] = {
            'oe_indices':   oe_filtered,
            'ctrl_indices': ctrl_filtered
        }

    return tf_filtered_masks, tfs


# =====================================================================
# MAX STATE ASSIGNMENT (Proliferative cells excluded)
# =====================================================================

def assign_max_state(oe_obs):
    """
    Assign each cell to its highest-probability coarse state, considering
    ONLY the 5 non-Proliferative states.

    Cells for which Proliferative has higher probability than ALL 5 states
    are flagged as Proliferative-dominant and excluded (keep_mask = False).
    They are NOT re-assigned to their second-best state.

    Parameters
    ----------
    oe_obs : DataFrame containing prob_coarse_{state} columns for all 6 states

    Returns
    -------
    state_assignments : pd.Series (same index as oe_obs)
                        str state name for non-Proliferative cells
                        NaN for Proliferative-dominant cells
    keep_mask         : numpy bool array (length = len(oe_obs)), True = non-Proliferative-dominant
    """
    # Columns for the 5 non-Proliferative states
    state_cols = {s: f'prob_coarse_{s}' for s in Config.COARSE_STATES}
    prolif_col = f'prob_coarse_{Config.PROLIFERATIVE}'

    missing = [c for c in list(state_cols.values()) + [prolif_col]
               if c not in oe_obs.columns]
    # if missing:
    #     raise ValueError(f"Missing probability columns in oe_obs: {missing}")

    # Probability matrix over 5 non-Proliferative states
    prob_matrix = oe_obs[[state_cols[s] for s in Config.COARSE_STATES]].values   # (n, 5)
    prolif_prob = oe_obs[prolif_col].values                                        # (n,)
    #prolif_prob = np.zeros((prob_matrix.shape[0],1))

    max_nonprolif_prob  = prob_matrix.max(axis=1)
    max_nonprolif_state = np.array(Config.COARSE_STATES)[prob_matrix.argmax(axis=1)]

    # Keep cell if at least one non-Proliferative state beats Proliferative
    keep_mask = prolif_prob <= max_nonprolif_prob

    # state_assignments = pd.Series(
    #     np.where(keep_mask, max_nonprolif_state, np.nan),
    #     index=oe_obs.index,
    #     dtype=object,
    #     name='max_state'
    # )
    state_assignments = pd.Series(
        max_nonprolif_state,
        index=oe_obs.index,
        dtype=object,
        name='max_state'
    )

    return state_assignments, keep_mask


# =====================================================================
# MULTINOMIAL LIKELIHOOD RATIO TEST
# =====================================================================

def multinomial_lrt(state_labels, is_oe, cell_line=None):
    """
    Test whether TF overexpression shifts cell state composition.

    Model comparison
    ----------------
    Pooled (cell_line provided):
        Full:  state ~ is_OE + cell_line   (sklearn, multinomial LR)
        Null:  state ~ cell_line            (sklearn, multinomial LR)

    Per cell line (cell_line=None):
        Full:  state ~ is_OE               (sklearn, multinomial LR)
        Null:  state ~ intercept only       (analytical: class base rates)

    Degrees of freedom
    ------------------
    Adding is_OE adds (n_observed_classes - 1) parameters to the model
    (one coefficient per non-reference class in the multinomial parameterisation).

    Parameters
    ----------
    state_labels : array-like of str  (state name for each cell)
    is_oe        : array-like of 0/1
    cell_line    : array-like of str or None

    Returns
    -------
    dict with keys: lrt_stat, df, pval, log_OR_multinomial (per state), n_classes
    None if fitting failed or only 1 state observed.
    """
    state_labels = np.array(state_labels)
    is_oe        = np.array(is_oe, dtype=float)
    n            = len(state_labels)

    # Encode states as integers
    le = LabelEncoder()
    y  = le.fit_transform(state_labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return None   # can't model composition with only 1 state present

    # ---- Build feature matrices ----
    if cell_line is not None:
        # Pooled: covariate = cell line (one-hot, drop first to avoid collinearity)
        cl_dummies = pd.get_dummies(
            pd.Series(cell_line), drop_first=True
        ).values.astype(float)

        X_full = np.column_stack([is_oe, cl_dummies])   # [is_OE | cell_line dummies]
        X_null = cl_dummies if cl_dummies.shape[1] > 0 else np.zeros((n, 0))
    else:
        # Per-line: no covariates
        X_full = is_oe.reshape(-1, 1)
        X_null = None   # handled analytically below

    # ---- Fit full model ----
    try:
        clf_full = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1e6,              # essentially no regularisation
            max_iter=500,
            fit_intercept=True,
            tol=1e-4
        )
        clf_full.fit(X_full, y)
    except Exception as e:
        return None

    proba_full = np.clip(clf_full.predict_proba(X_full), 1e-15, 1.0)
    ll_full    = float(np.sum(np.log(proba_full[np.arange(n), y])))

    # ---- Compute null log-likelihood ----
    if X_null is not None and X_null.shape[1] > 0:
        # Null has cell_line covariates → fit with sklearn
        try:
            clf_null = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                C=1e6,
                max_iter=500,
                fit_intercept=True,
                tol=1e-4
            )
            clf_null.fit(X_null, y)
            proba_null = np.clip(clf_null.predict_proba(X_null), 1e-15, 1.0)
            ll_null    = float(np.sum(np.log(proba_null[np.arange(n), y])))
        except Exception:
            return None
    else:
        # Null = intercept only → analytical (base rates)
        counts  = np.bincount(y, minlength=n_classes).astype(float)
        p_null  = counts / n
        ll_null = float(np.sum(np.log(np.maximum(p_null[y], 1e-15))))

    # ---- LRT statistic ----
    lrt_stat = max(0.0, 2.0 * (ll_full - ll_null))   # clamp numerical noise at 0
    df_lrt   = n_classes - 1                           # is_OE adds (K-1) parameters
    pval     = float(chi2.sf(lrt_stat, df=df_lrt))

    # ---- Per-state log-OR from multinomial coefficients ----
    # clf_full.coef_ shape: (n_classes, n_features)
    # coef_[k, 0] = effect of is_OE on log-probability of class k
    # (relative to the soft-max normalisation; interpretable as direction + magnitude)
    log_OR_multinomial = {}
    for k in range(n_classes):
        state_name = le.inverse_transform([k])[0]   # original string
        log_OR_multinomial[state_name] = float(clf_full.coef_[k, 0])

    return {
        'lrt_stat':            lrt_stat,
        'df':                  df_lrt,
        'pval':                pval,
        'log_OR_multinomial':  log_OR_multinomial,
        'n_classes':           n_classes
    }


# =====================================================================
# FISHER'S EXACT TEST (per state, post-hoc)
# =====================================================================

def fisher_posthoc(state_labels, is_oe_bool):
    """
    Per-state two-sided Fisher's exact test.

    2×2 contingency table for each state:
        [ OE in state     |  OE not in state  ]
        [ ctrl in state   |  ctrl not in state ]

    Run for ALL TFs regardless of LRT significance; FDR correction handles it.

    Parameters
    ----------
    state_labels : array-like of str (only non-Proliferative cells)
    is_oe_bool   : boolean array (True = OE)

    Returns
    -------
    dict: {state: {'prop_oe', 'prop_ctrl', 'delta_prop', 'fisher_OR', 'fisher_pval',
                   'n_oe_in_state', 'n_ctrl_in_state'}}
    """
    state_labels = np.array(state_labels)
    is_oe_bool   = np.array(is_oe_bool, dtype=bool)

    n_oe   = is_oe_bool.sum()
    n_ctrl = (~is_oe_bool).sum()

    results = {}
    for state in Config.COARSE_STATES:
        in_state = state_labels == state

        n_oe_in    = int((is_oe_bool  & in_state).sum())
        n_oe_out   = int(n_oe   - n_oe_in)
        n_ctrl_in  = int((~is_oe_bool & in_state).sum())
        n_ctrl_out = int(n_ctrl - n_ctrl_in)

        try:
            fisher_OR, pval = fisher_exact(
                [[n_oe_in, n_oe_out],
                 [n_ctrl_in, n_ctrl_out]],
                alternative='two-sided'
            )
        except Exception:
            fisher_OR, pval = np.nan, np.nan

        prop_oe   = n_oe_in   / n_oe   if n_oe   > 0 else np.nan
        prop_ctrl = n_ctrl_in / n_ctrl if n_ctrl > 0 else np.nan

        if not (np.isnan(prop_oe) or np.isnan(prop_ctrl)):
            delta_prop = prop_oe - prop_ctrl
        else:
            delta_prop = np.nan

        results[state] = {
            'n_oe_in_state':  n_oe_in,
            'n_ctrl_in_state': n_ctrl_in,
            'prop_oe':         prop_oe,
            'prop_ctrl':       prop_ctrl,
            'delta_prop':      delta_prop,
            'fisher_OR':       fisher_OR,
            'fisher_pval':     pval
        }

    return results


# =====================================================================
# RUN ALL TESTS FOR ONE STRATEGY
# =====================================================================

def run_proportion_tests(oe_obs, state_assignments, keep_mask_arr,
                         tf_masks, tfs, cell_lines, strategy_name):
    """
    Run Multinomial LRT + Fisher's exact for all TFs under one strategy.

    Parameters
    ----------
    oe_obs            : DataFrame (full OE obs, all cells)
    state_assignments : pd.Series aligned to oe_obs, NaN for Proliferative cells
    keep_mask_arr     : numpy bool array (integer-position mask into oe_obs)
                        True  = non-Proliferative-max cell
                        False = Proliferative-max cell (excluded)
    tf_masks          : dict {tf: {'oe_indices': int-pos array, 'ctrl_indices': int-pos array}}
    tfs               : list of TF names
    cell_lines        : list of cell line names
    strategy_name     : str

    Returns
    -------
    df_lrt    : DataFrame, one row per TF × cell_line
    df_fisher : DataFrame, one row per TF × state × cell_line
    """
    print(f"\n   Running proportion tests ({strategy_name})...")

    lrt_rows    = []
    fisher_rows = []

    def _make_nan_lrt_row(tf, cl, n_oe, n_ctrl, n_oe_pd, n_ctrl_pd):
        row = {
            'TF': tf, 'cell_line': cl, 'strategy': strategy_name,
            'n_oe': n_oe, 'n_ctrl': n_ctrl,
            'n_oe_prolif_dropped':   n_oe_pd,
            'n_ctrl_prolif_dropped': n_ctrl_pd,
            'lrt_stat': np.nan, 'lrt_df': np.nan, 'lrt_pval': np.nan,
            'n_states_observed': np.nan,
        }
        for s in Config.COARSE_STATES:
            row[f'log_OR_multi_{s}'] = np.nan
        return row

    def _make_nan_fisher_rows(tf, cl, n_oe, n_ctrl):
        rows = []
        for s in Config.COARSE_STATES:
            rows.append({
                'TF': tf, 'state': s, 'cell_line': cl, 'strategy': strategy_name,
                'n_oe': n_oe, 'n_ctrl': n_ctrl,
                'n_oe_in_state': np.nan, 'n_ctrl_in_state': np.nan,
                'prop_oe': np.nan, 'prop_ctrl': np.nan,
                'delta_prop': np.nan, 'fisher_OR': np.nan, 'fisher_pval': np.nan
            })
        return rows

    # ------------------------------------------------------------------
    # POOLED (cell line as covariate in LRT)
    # ------------------------------------------------------------------
    print("     Pooled (cell_line covariate in LRT)...")

    for tf_i, tf in enumerate(tfs):
        if (tf_i + 1) % 10 == 0:
            print(f"       [{tf_i+1}/{len(tfs)}] {tf}")

        oe_idx   = tf_masks[tf]['oe_indices']
        ctrl_idx = tf_masks[tf]['ctrl_indices']

        # Apply Proliferative-exclusion mask (integer-position indexing)
        oe_keep   = oe_idx[keep_mask_arr[oe_idx]]
        ctrl_keep = ctrl_idx[keep_mask_arr[ctrl_idx]]

        n_oe   = len(oe_keep)
        n_ctrl = len(ctrl_keep)
        n_oe_pd   = len(oe_idx)   - n_oe    # Proliferative cells dropped
        n_ctrl_pd = len(ctrl_idx) - n_ctrl

        if n_oe < Config.MIN_CELLS or n_ctrl < Config.MIN_CELLS:
            lrt_rows.append(_make_nan_lrt_row(tf, 'pooled', n_oe, n_ctrl, n_oe_pd, n_ctrl_pd))
            fisher_rows.extend(_make_nan_fisher_rows(tf, 'pooled', n_oe, n_ctrl))
            continue

        # Build combined arrays for this TF
        all_idx     = np.concatenate([oe_keep, ctrl_keep])
        state_arr   = state_assignments.iloc[all_idx].values          # str
        is_oe_arr   = np.concatenate([np.ones(n_oe), np.zeros(n_ctrl)])
        cl_arr      = oe_obs['cell_line'].iloc[all_idx].values

        # ---- Multinomial LRT ----
        lrt = multinomial_lrt(state_arr, is_oe_arr, cell_line=cl_arr)

        lrt_row = {
            'TF': tf, 'cell_line': 'pooled', 'strategy': strategy_name,
            'n_oe': n_oe, 'n_ctrl': n_ctrl,
            'n_oe_prolif_dropped':   n_oe_pd,
            'n_ctrl_prolif_dropped': n_ctrl_pd,
        }
        if lrt is not None:
            lrt_row.update({
                'lrt_stat':          lrt['lrt_stat'],
                'lrt_df':            lrt['df'],
                'lrt_pval':          lrt['pval'],
                'n_states_observed': lrt['n_classes'],
            })
            for s in Config.COARSE_STATES:
                lrt_row[f'log_OR_multi_{s}'] = lrt['log_OR_multinomial'].get(s, np.nan)
        else:
            lrt_row.update({'lrt_stat': np.nan, 'lrt_df': np.nan,
                            'lrt_pval': np.nan, 'n_states_observed': np.nan})
            for s in Config.COARSE_STATES:
                lrt_row[f'log_OR_multi_{s}'] = np.nan

        lrt_rows.append(lrt_row)

        # ---- Fisher's exact (per state, post-hoc) ----
        fisher = fisher_posthoc(state_arr, is_oe_arr.astype(bool))
        for state, fr in fisher.items():
            fisher_rows.append({
                'TF': tf, 'state': state, 'cell_line': 'pooled',
                'strategy': strategy_name,
                'n_oe': n_oe, 'n_ctrl': n_ctrl,
                **fr
            })

    # ------------------------------------------------------------------
    # PER CELL LINE (no cell-line covariate)
    # ------------------------------------------------------------------
    for cl in cell_lines:
        print(f"     {cl}...")

        # Boolean mask for this cell line (integer-position safe)
        cl_bool = (oe_obs['cell_line'] == cl).values    # numpy bool, length = len(oe_obs)

        for tf in tfs:
            oe_idx   = tf_masks[tf]['oe_indices']
            ctrl_idx = tf_masks[tf]['ctrl_indices']

            # Filter: cell line AND non-Proliferative
            oe_in_cl   = oe_idx[cl_bool[oe_idx]]
            ctrl_in_cl = ctrl_idx[cl_bool[ctrl_idx]]

            oe_keep_cl   = oe_in_cl[keep_mask_arr[oe_in_cl]]
            ctrl_keep_cl = ctrl_in_cl[keep_mask_arr[ctrl_in_cl]]

            n_oe   = len(oe_keep_cl)
            n_ctrl = len(ctrl_keep_cl)
            n_oe_pd   = len(oe_in_cl)   - n_oe
            n_ctrl_pd = len(ctrl_in_cl) - n_ctrl

            if n_oe < Config.MIN_CELLS or n_ctrl < Config.MIN_CELLS:
                lrt_rows.append(_make_nan_lrt_row(tf, cl, n_oe, n_ctrl, n_oe_pd, n_ctrl_pd))
                fisher_rows.extend(_make_nan_fisher_rows(tf, cl, n_oe, n_ctrl))
                continue

            all_idx   = np.concatenate([oe_keep_cl, ctrl_keep_cl])
            state_arr = state_assignments.iloc[all_idx].values
            is_oe_arr = np.concatenate([np.ones(n_oe), np.zeros(n_ctrl)])

            # No cell-line covariate for per-line
            lrt = multinomial_lrt(state_arr, is_oe_arr, cell_line=None)

            lrt_row = {
                'TF': tf, 'cell_line': cl, 'strategy': strategy_name,
                'n_oe': n_oe, 'n_ctrl': n_ctrl,
                'n_oe_prolif_dropped':   n_oe_pd,
                'n_ctrl_prolif_dropped': n_ctrl_pd,
            }
            if lrt is not None:
                lrt_row.update({
                    'lrt_stat':          lrt['lrt_stat'],
                    'lrt_df':            lrt['df'],
                    'lrt_pval':          lrt['pval'],
                    'n_states_observed': lrt['n_classes'],
                })
                for s in Config.COARSE_STATES:
                    lrt_row[f'log_OR_multi_{s}'] = lrt['log_OR_multinomial'].get(s, np.nan)
            else:
                lrt_row.update({'lrt_stat': np.nan, 'lrt_df': np.nan,
                                'lrt_pval': np.nan, 'n_states_observed': np.nan})
                for s in Config.COARSE_STATES:
                    lrt_row[f'log_OR_multi_{s}'] = np.nan

            lrt_rows.append(lrt_row)

            fisher = fisher_posthoc(state_arr, is_oe_arr.astype(bool))
            for state, fr in fisher.items():
                fisher_rows.append({
                    'TF': tf, 'state': state, 'cell_line': cl,
                    'strategy': strategy_name,
                    'n_oe': n_oe, 'n_ctrl': n_ctrl,
                    **fr
                })

    # ------------------------------------------------------------------
    # Build DataFrames and apply FDR correction
    # ------------------------------------------------------------------
    df_lrt    = pd.DataFrame(lrt_rows)
    df_fisher = pd.DataFrame(fisher_rows)

    all_cls = ['pooled'] + list(cell_lines)

    # LRT FDR: correct across all TFs within each cell_line separately
    df_lrt['lrt_fdr'] = np.nan
    for cl in all_cls:
        mask = df_lrt['cell_line'] == cl
        pvals = df_lrt.loc[mask, 'lrt_pval'].fillna(1.0).values
        df_lrt.loc[mask, 'lrt_fdr'] = multipletests(pvals, method='fdr_bh')[1]

    df_lrt['lrt_sig'] = df_lrt['lrt_fdr'] < Config.FDR_THRESHOLD

    # Fisher FDR: correct across all TF × state pairs within each cell_line
    df_fisher['fisher_fdr'] = np.nan
    for cl in all_cls:
        mask = df_fisher['cell_line'] == cl
        pvals = df_fisher.loc[mask, 'fisher_pval'].fillna(1.0).values
        df_fisher.loc[mask, 'fisher_fdr'] = multipletests(pvals, method='fdr_bh')[1]

    df_fisher['fisher_sig'] = df_fisher['fisher_fdr'] < Config.FDR_THRESHOLD

    return df_lrt, df_fisher


# =====================================================================
# STRATEGY COMPARISON PLOTS
# =====================================================================

def compare_strategies(all_lrt, all_fisher, cell_lines):
    """
    Summary comparison across all 24 strategies.
    Saves plots and CSV summaries to strategy_comparison/.
    """
    print("\n   Comparing strategies across all runs...")

    df_lrt_all    = pd.concat(all_lrt,    ignore_index=True)
    df_fisher_all = pd.concat(all_fisher, ignore_index=True)

    # Parse FC and ctrl from strategy string (e.g. "FC1.0_p25")
    def _parse_fc(s):
        return float(s.split('_')[0].replace('FC', ''))
    def _parse_ctrl(s):
        return s.split('_')[1]

    for df in [df_lrt_all, df_fisher_all]:
        df['fc_threshold']  = df['strategy'].apply(_parse_fc)
        df['ctrl_strategy'] = df['strategy'].apply(_parse_ctrl)

    # ---- Summaries ----
    pooled_lrt = df_lrt_all[df_lrt_all['cell_line'] == 'pooled'].copy()
    summary_lrt = (
        pooled_lrt.groupby('strategy')
        .agg(n_sig_lrt=('lrt_sig', 'sum'),
             mean_lrt_stat=('lrt_stat', lambda x: x.mean()))
        .reset_index()
    )
    summary_lrt['fc_threshold']  = summary_lrt['strategy'].apply(_parse_fc)
    summary_lrt['ctrl_strategy'] = summary_lrt['strategy'].apply(_parse_ctrl)
    summary_lrt.to_csv(
        Config.OUT_DIR / 'strategy_comparison' / 'lrt_strategy_summary.csv',
        index=False
    )

    pooled_fisher = df_fisher_all[df_fisher_all['cell_line'] == 'pooled'].copy()
    summary_fisher = (
        pooled_fisher.groupby(['strategy', 'state'])
        .agg(n_sig_fisher=('fisher_sig', 'sum'),
             mean_abs_delta=('delta_prop', lambda x: np.abs(x).mean()))
        .reset_index()
    )
    summary_fisher.to_csv(
        Config.OUT_DIR / 'strategy_comparison' / 'fisher_strategy_summary.csv',
        index=False
    )

    # ---- Plot 1: LRT strategy comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    srt = summary_lrt.sort_values('n_sig_lrt', ascending=True)
    ax.barh(range(len(srt)), srt['n_sig_lrt'], color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(srt)))
    ax.set_yticklabels(srt['strategy'], fontsize=8)
    ax.set_xlabel('# TFs with significant LRT (FDR < 0.05)', fontweight='bold')
    ax.set_title('Multinomial LRT: significant TFs per strategy (pooled)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    pivot = (
        pooled_lrt.groupby(['fc_threshold', 'ctrl_strategy'])['lrt_sig']
        .sum().unstack(fill_value=0)
    )
    im2 = ax.imshow(pivot.values.astype(float), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(pivot.values[i, j])),
                    ha='center', va='center', fontsize=9, fontweight='bold')
    plt.colorbar(im2, ax=ax, shrink=0.7)
    ax.set_title('LRT: significant TFs — FC × control strategy', fontweight='bold')
    ax.set_xlabel('Control strategy')
    ax.set_ylabel('FC threshold (linear addition)')

    plt.tight_layout()
    plt.savefig(
        Config.OUT_DIR / 'strategy_comparison' / 'lrt_strategy_comparison.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("   ✓ Saved: lrt_strategy_comparison.pdf")

    # ---- Plot 2: Fisher strategy comparison ----
    total_fisher_per_strategy = (
        pooled_fisher.groupby('strategy')['fisher_sig']
        .sum().reset_index()
        .rename(columns={'fisher_sig': 'n_sig_total'})
    )
    total_fisher_per_strategy['fc_threshold']  = total_fisher_per_strategy['strategy'].apply(_parse_fc)
    total_fisher_per_strategy['ctrl_strategy'] = total_fisher_per_strategy['strategy'].apply(_parse_ctrl)

    best_strategy = summary_lrt.sort_values('n_sig_lrt', ascending=False).iloc[0]['strategy']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    srt_f = total_fisher_per_strategy.sort_values('n_sig_total', ascending=True)
    ax.barh(range(len(srt_f)), srt_f['n_sig_total'], color='darkorange', edgecolor='white')
    ax.set_yticks(range(len(srt_f)))
    ax.set_yticklabels(srt_f['strategy'], fontsize=8)
    ax.set_xlabel("# significant Fisher's TF-state pairs (FDR < 0.05)", fontweight='bold')
    ax.set_title("Fisher's exact: total significant effects per strategy (pooled)",
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    best_state_counts = (
        summary_fisher[summary_fisher['strategy'] == best_strategy]
        .set_index('state')['n_sig_fisher']
    )
    # Sort by count for readability
    best_state_counts = best_state_counts.sort_values(ascending=False)
    ax.bar(range(len(best_state_counts)), best_state_counts.values,
           color='darkgreen', edgecolor='white')
    ax.set_xticks(range(len(best_state_counts)))
    ax.set_xticklabels(best_state_counts.index, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('# significant TF-state pairs', fontweight='bold')
    ax.set_title(f"Fisher: significant effects by state\n({best_strategy})",
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        Config.OUT_DIR / 'strategy_comparison' / 'fisher_strategy_comparison.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("   ✓ Saved: fisher_strategy_comparison.pdf")

    # ---- Plot 3: Best strategy heatmap (TF × state, delta_prop) ----
    _plot_best_strategy_heatmap(
        df_lrt_all, df_fisher_all, best_strategy
    )

    # ---- Plot 4: Cell-line consistency for best strategy ----
    _plot_cell_line_consistency(df_fisher_all, best_strategy, cell_lines)

    print(f"   ✓ Best strategy: {best_strategy} "
          f"({summary_lrt.sort_values('n_sig_lrt', ascending=False).iloc[0]['n_sig_lrt']}"
          f" LRT-significant TFs, pooled)")

    return best_strategy


def _plot_best_strategy_heatmap(df_lrt_all, df_fisher_all, best_strategy):
    """Heatmap of delta_prop (TF × state) for best strategy, pooled."""

    best_lrt = df_lrt_all[
        (df_lrt_all['strategy'] == best_strategy) &
        (df_lrt_all['cell_line'] == 'pooled')
    ].copy()

    best_fisher = df_fisher_all[
        (df_fisher_all['strategy'] == best_strategy) &
        (df_fisher_all['cell_line'] == 'pooled')
    ].copy()

    # Pivot: TF × state → delta_prop
    pivot_delta = best_fisher.pivot(index='TF', columns='state', values='delta_prop')
    pivot_sig   = best_fisher.pivot(index='TF', columns='state', values='fisher_sig')

    # Sort TFs: LRT-significant first (sorted by FDR), then rest (sorted by lrt_stat)
    lrt_idx = best_lrt.set_index('TF')
    sig_tfs  = lrt_idx[lrt_idx['lrt_sig']].sort_values('lrt_fdr').index.tolist()
    nsig_tfs = lrt_idx[~lrt_idx['lrt_sig']].sort_values('lrt_stat', ascending=False).index.tolist()
    tf_order = [t for t in sig_tfs + nsig_tfs if t in pivot_delta.index]

    pivot_delta = pivot_delta.reindex(tf_order)
    pivot_sig   = pivot_sig.reindex(tf_order)

    # Reorder columns to match Config.COARSE_STATES
    col_order = [c for c in Config.COARSE_STATES if c in pivot_delta.columns]
    pivot_delta = pivot_delta[col_order]
    pivot_sig   = pivot_sig[col_order]

    n_tfs = len(tf_order)
    fig_h = max(8, n_tfs * 0.32 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    # Symmetric colour scale capped at ±0.3
    vmax = min(0.3, np.nanpercentile(np.abs(pivot_delta.values), 95))
    im = ax.imshow(
        pivot_delta.values.astype(float),
        cmap='RdBu_r', aspect='auto',
        vmin=-vmax, vmax=vmax
    )

    # Mark Fisher-significant cells
    for i in range(pivot_sig.shape[0]):
        for j in range(pivot_sig.shape[1]):
            val = pivot_sig.iloc[i, j]
            if val is not None and val is not np.nan and val:
                ax.text(j, i, '●', ha='center', va='center',
                        fontsize=7, color='black', alpha=0.7)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(n_tfs))

    # Annotate LRT-significant TFs on y-axis
    ytick_labels = []
    for tf in tf_order:
        prefix = '★ ' if tf in sig_tfs else '   '
        ytick_labels.append(f'{prefix}{tf}')
    ax.set_yticklabels(ytick_labels, fontsize=8)

    # Separator line between LRT-sig and non-sig TFs
    if sig_tfs:
        ax.axhline(len(sig_tfs) - 0.5, color='black', linewidth=1.5, linestyle='--')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.5)
    cbar.set_label('Δ proportion (OE − ctrl)', fontsize=9)

    ax.set_title(
        f'State proportion changes: {best_strategy} (pooled)\n'
        f'★ = LRT FDR < 0.05   ● = Fisher FDR < 0.05',
        fontweight='bold', fontsize=11
    )

    plt.tight_layout()
    plt.savefig(
        Config.OUT_DIR / 'strategy_comparison' / f'best_strategy_{best_strategy}_heatmap.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"   ✓ Saved: best_strategy_{best_strategy}_heatmap.pdf")


def _plot_cell_line_consistency(df_fisher_all, best_strategy, cell_lines):
    """
    For each significant TF-state pair (pooled, best strategy),
    show delta_prop in each cell line as a dot plot.
    """
    fisher_best = df_fisher_all[df_fisher_all['strategy'] == best_strategy].copy()

    # Identify sig TF-state pairs (pooled)
    sig_pairs = fisher_best[
        (fisher_best['cell_line'] == 'pooled') &
        fisher_best['fisher_sig']
    ][['TF', 'state']].drop_duplicates()

    if len(sig_pairs) == 0:
        print("   (No significant Fisher pairs in best strategy — skipping consistency plot)")
        return

    # Cap at 40 pairs for readability
    if len(sig_pairs) > 40:
        # Take the 40 with smallest fisher p-value in pooled
        pooled_pvals = fisher_best[fisher_best['cell_line'] == 'pooled'].set_index(['TF', 'state'])['fisher_pval']
        sig_pairs = sig_pairs.copy()
        sig_pairs['pval'] = sig_pairs.apply(
            lambda r: pooled_pvals.get((r['TF'], r['state']), 1.0), axis=1
        )
        sig_pairs = sig_pairs.nsmallest(40, 'pval')[['TF', 'state']]

    all_cls = ['pooled'] + list(cell_lines)
    colors  = {'pooled': 'black', 'BG5': '#1f77b4', 'P3': '#ff7f0e', 'S24': '#2ca02c'}
    default_colors = ['#9467bd', '#8c564b', '#e377c2']

    fig, ax = plt.subplots(figsize=(10, max(6, len(sig_pairs) * 0.45)))

    y_pos = range(len(sig_pairs))
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    for i, (_, row) in enumerate(sig_pairs.iterrows()):
        tf, state = row['TF'], row['state']
        for j, cl in enumerate(all_cls):
            subset = fisher_best[
                (fisher_best['TF'] == tf) &
                (fisher_best['state'] == state) &
                (fisher_best['cell_line'] == cl)
            ]
            if subset.empty:
                continue
            delta = subset.iloc[0]['delta_prop']
            is_sig = subset.iloc[0]['fisher_sig']
            if np.isnan(delta):
                continue
            color = colors.get(cl, default_colors[j % len(default_colors)])
            marker = 'D' if is_sig else 'o'
            ms = 8 if cl == 'pooled' else 6
            ax.plot(delta, i + j * 0.2 - len(all_cls) * 0.1,
                    marker=marker, color=color, markersize=ms,
                    alpha=0.9 if is_sig else 0.5,
                    label=cl if i == 0 else '_')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{r['TF']} | {r['state']}" for _, r in sig_pairs.iterrows()],
                       fontsize=8)
    ax.set_xlabel('Δ proportion (OE − ctrl)', fontweight='bold')
    ax.set_title(
        f'Cell-line consistency: {best_strategy}\n'
        f'Significant TF-state pairs (pooled Fisher FDR < 0.05)\n'
        f'Diamond = FDR < 0.05 in that line, circle = not significant',
        fontweight='bold', fontsize=10
    )
    ax.legend(title='Cell line', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(
        Config.OUT_DIR / 'strategy_comparison' / f'cell_line_consistency_{best_strategy}.pdf',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"   ✓ Saved: cell_line_consistency_{best_strategy}.pdf")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':

    print("=" * 80)
    print("PROPORTION-BASED TESTING - MULTIPLE STRATEGIES")
    print("=" * 80)
    print("Test:   Multinomial LRT (global) + Fisher's exact (per state)")
    print("Cells:  Max-state assignment; Proliferative-max cells EXCLUDED")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load data (identical pattern to statistical_testing_multiStrategy.py)
    # ------------------------------------------------------------------
    print("\nLoading data...")
    import scanpy as sc
    adata = sc.read_h5ad(Config.ANNDATA_PATH)

    if adata.X.max() > 20:
        print("Normalizing (log1p to 10k)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    print(f"Total cells: {len(adata):,}")

    # Load Harmony into FULL adata BEFORE any subsetting
    # (prevents shape mismatch when subsetting obsm)
    print("\nLoading Harmony embeddings...")
    harmony_path = Config.BASE_DIR / 'harmony_200pc' / 'harmony_embeddings.npy'
    if not harmony_path.exists():
        raise FileNotFoundError(f"Harmony not found: {harmony_path}")
    X_harmony = np.load(harmony_path)
    adata.obsm['X_harmony'] = X_harmony
    print(f"  ✓ Harmony shape: {X_harmony.shape}")

    # Subset to OE cells
    gbm_mask  = adata.obs['dataset'].astype(str).str.lower().str.contains('gbm', na=False)
    adata_oe  = adata[~gbm_mask].copy()
    print(f"OE cells: {len(adata_oe):,} | Harmony shape: {adata_oe.obsm['X_harmony'].shape}")

    # ------------------------------------------------------------------
    # Load models and predict coarse state probabilities
    # ------------------------------------------------------------------
    print("\nLoading coarse state models...")
    model_dir   = Config.BASE_DIR / 'models'
    clf_coarse  = joblib.load(model_dir / 'clf_celltype_coarse.pkl')
    scaler_coarse = joblib.load(model_dir / 'scaler_celltype_coarse.pkl')
    le_coarse   = joblib.load(model_dir / 'le_celltype_coarse.pkl')
    print(f"  ✓ Coarse classes: {list(le_coarse.classes_)}")

    print("\nPredicting coarse state probabilities on OE cells...")
    X_oe_scaled   = scaler_coarse.transform(adata_oe.obsm['X_harmony'])
    coarse_proba  = clf_coarse.predict_proba(X_oe_scaled)

    for i, cls_name in enumerate(le_coarse.classes_):
        adata_oe.obs[f'prob_coarse_{cls_name}'] = coarse_proba[:, i]
    print("  ✓ Probabilities stored in adata_oe.obs")
    adata_oe.obs[f'prob_coarse_{Config.PROLIFERATIVE}']= 0

    # Verify all required probability columns are present
    required_cols = [f'prob_coarse_{s}' for s in Config.COARSE_STATES + [Config.PROLIFERATIVE]]
    missing = [c for c in required_cols if c not in adata_oe.obs.columns]
    if missing:
        raise ValueError(
            f"Missing probability columns: {missing}\n"
            f"Check that le_coarse.classes_ contains all states. "
            f"Found: {list(le_coarse.classes_)}"
        )

    oe_obs = adata_oe.obs.copy()

    # ------------------------------------------------------------------
    # Assign max state, exclude Proliferative-dominant cells
    # ------------------------------------------------------------------
    print("\nAssigning max states (Proliferative cells excluded)...")
    state_assignments, keep_mask = assign_max_state(oe_obs)
    keep_mask_arr = np.asarray(keep_mask, dtype=bool)   # ensure numpy bool array

    n_total  = len(oe_obs)
    n_kept   = int(keep_mask_arr.sum())
    n_prolif = n_total - n_kept

    print(f"  Total OE cells:               {n_total:>8,}")
    print(f"  Proliferative-max (excluded): {n_prolif:>8,}  ({100*n_prolif/n_total:.1f}%)")
    print(f"  Retained cells:               {n_kept:>8,}  ({100*n_kept/n_total:.1f}%)")
    print(f"\n  State composition (retained cells):")
    for s in Config.COARSE_STATES:
        n = int((state_assignments == s).sum())
        print(f"    {s:<22} {n:>7,}  ({100*n/n_kept:.1f}%)")

    cell_lines = sorted(oe_obs['cell_line'].unique())
    print(f"\n  Cell lines: {cell_lines}")

    # ------------------------------------------------------------------
    # Run all 24 strategies
    # ------------------------------------------------------------------
    all_lrt_results    = []
    all_fisher_results = []

    total_strategies = len(Config.FC_THRESHOLDS) * len(Config.CONTROL_PERCENTILES)
    strategy_idx     = 0
    start_time       = time.time()

    for fc_thresh in Config.FC_THRESHOLDS:
        for ctrl_pct in Config.CONTROL_PERCENTILES:
            strategy_idx += 1
            ctrl_str      = 'all' if ctrl_pct == 'all' else f'p{ctrl_pct}'
            strategy_name = f"FC{fc_thresh}_{ctrl_str}"

            print(f"\n{'='*80}")
            print(f"STRATEGY {strategy_idx}/{total_strategies}: {strategy_name}")
            print(f"  OE filter:      fc >= {1 + fc_thresh:.1f}×  (linear space)")
            print(f"  Control filter: {ctrl_pct}")
            print(f"{'='*80}")

            strategy_dir = Config.OUT_DIR / strategy_name
            strategy_dir.mkdir(exist_ok=True)

            # Filter cells by expression (same logic as existing script)
            print("   Filtering cells by expression...")
            tf_masks, tfs = filter_cells_by_strategy(
                adata_oe, oe_obs, fc_thresh, ctrl_pct
            )
            print(f"   TFs found: {len(tfs)}")

            # Run LRT + Fisher
            df_lrt, df_fisher = run_proportion_tests(
                oe_obs, state_assignments, keep_mask_arr,
                tf_masks, tfs, cell_lines, strategy_name
            )

            # Save per-strategy
            df_lrt.to_csv(strategy_dir / 'proportion_lrt.csv', index=False)
            df_fisher.to_csv(strategy_dir / 'proportion_fisher.csv', index=False)

            # Print summary
            n_sig_lrt    = int(df_lrt[(df_lrt['cell_line'] == 'pooled') & df_lrt['lrt_sig']].shape[0])
            n_sig_fisher = int(df_fisher[(df_fisher['cell_line'] == 'pooled') & df_fisher['fisher_sig']].shape[0])
            print(f"   ✓ LRT significant TFs (pooled):            {n_sig_lrt}")
            print(f"   ✓ Fisher significant TF-state pairs (pooled): {n_sig_fisher}")
            print(f"   ✓ Saved to: {strategy_dir}")
            print(f"   Elapsed: {(time.time() - start_time)/60:.1f} min")

            all_lrt_results.append(df_lrt)
            all_fisher_results.append(df_fisher)

    # ------------------------------------------------------------------
    # Compare all strategies + produce summary plots
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("COMPARING ALL STRATEGIES")
    print(f"{'='*80}")

    best_strategy = compare_strategies(all_lrt_results, all_fisher_results, cell_lines)

    # ------------------------------------------------------------------
    # Save combined outputs
    # ------------------------------------------------------------------
    print("\nSaving combined results...")
    df_lrt_combined    = pd.concat(all_lrt_results,    ignore_index=True)
    df_fisher_combined = pd.concat(all_fisher_results, ignore_index=True)

    df_lrt_combined.to_csv(Config.OUT_DIR / 'all_strategies_lrt.csv',    index=False)
    df_fisher_combined.to_csv(Config.OUT_DIR / 'all_strategies_fisher.csv', index=False)

    print("  ✓ all_strategies_lrt.csv")
    print("  ✓ all_strategies_fisher.csv")

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time:       {total_time/60:.1f} minutes")
    print(f"Strategies run:   {total_strategies}")
    print(f"Best strategy:    {best_strategy}")
    print(f"Output:           {Config.OUT_DIR}")
    print(f"\nOutput files per strategy:")
    print(f"  proportion_lrt.csv     — 1 row per TF × cell_line (global LRT)")
    print(f"  proportion_fisher.csv  — 1 row per TF × state × cell_line (Fisher post-hoc)")
    print(f"\nCombined files:")
    print(f"  all_strategies_lrt.csv")
    print(f"  all_strategies_fisher.csv")
    print(f"\nComparison plots:")
    print(f"  strategy_comparison/lrt_strategy_comparison.pdf")
    print(f"  strategy_comparison/fisher_strategy_comparison.pdf")
    print(f"  strategy_comparison/best_strategy_{best_strategy}_heatmap.pdf")
    print(f"  strategy_comparison/cell_line_consistency_{best_strategy}.pdf")
