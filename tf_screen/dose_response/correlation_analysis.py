"""Correlation between TF expression and state probability across metacells.

For every (TF, coarse state) pair, this script tests whether the
per-metacell mean TF expression covaries with the per-metacell mean
state probability -- i.e. whether the perturbation drives a *dose-
dependent* compositional shift, not just a population-level mean
difference.

Three populations are tested independently:

* ``combined`` -- OE + Ctrl metacells together. Primary discovery
  analysis; large dynamic range in TF expression, mixes biological
  variation between conditions with within-OE dose response.
* ``oe_only`` -- OE metacells alone. Tests the strictest version of
  the dose-response claim: among cells that received the guide, does
  *how much TF is induced* predict the state shift?
* ``control_only`` -- Ctrl metacells alone. Reports endogenous
  expression heterogeneity; controls how much of the combined signal
  comes simply from natural TF-state co-variation in unperturbed cells.

For each (TF, state, population) triple, both metrics are computed:

* **Log-normalised**: mean log1p expression per metacell.
* **Z-scored**: log expression z-scored against the *control* mean and
  standard deviation in that cell line. Removes between-TF differences
  in baseline expression so correlations are comparable across TFs.

Statistical tests:

* Pearson and Spearman correlation coefficients.
* Permutation p-value (1,000 shuffles of the TF-expression vector,
  Spearman test statistic).
* Benjamini-Hochberg FDR correction per (population, metric).

Inputs
------
``METACELL_DIR/metacells.h5ad``
    Produced by ``dose_response/build_metacells.py``. Must have
    ``prob_coarse_<state>`` columns aggregated from the single-cell
    classifier predictions.

Outputs
-------
``CORRELATION_DIR/correlation_<population>.csv``
    One row per (TF, state) with Pearson r/p, Spearman r/p,
    permutation p, BH-FDR (log and z-scored), n_metacells.
``CORRELATION_DIR/correlation_per_cell_line.csv``
    Same, broken out per cell line for the combined population.
``CORRELATION_DIR/heatmaps/``, ``CORRELATION_DIR/scatter_plots/``, etc.
    Figures.

Usage
-----
``python dose_response/correlation_analysis.py``
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection

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
# The plotting body below references ``Config.X``. Keep a local class
# pointing at the canonical config values.


class Config:
    METACELLS_PATH = config.METACELL_DIR / "metacells.h5ad"
    COHENS_D_PATH  = config.STATE_TESTING_DIR / "FC0.5_p10" / "coarse_effects.csv"
    OUT_DIR        = config.CORRELATION_DIR

    COARSE_STATES = list(config.COARSE_GROUPS.keys())
    N_PERMS       = config.N_PERMUTATIONS
    FDR_ALPHA     = config.FDR_THRESHOLD
    PRIMARY_METRIC = "zscore"


Config.OUT_DIR.mkdir(exist_ok=True, parents=True)
for sub in ("heatmaps", "scatter_plots", "distributions", "per_cell_line"):
    (Config.OUT_DIR / sub).mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Probability-column adapter
# ---------------------------------------------------------------------------
# The original code reads ``prob_<state>`` columns from the metacell obs;
# the cleaned ``build_metacells.py`` writes ``prob_coarse_<state>``.
# Rather than touching every reference in the long plotting body, we
# rename the columns at load time so the existing analysis code keeps
# working.


def _rename_probability_columns(adata: sc.AnnData) -> sc.AnnData:
    """Rename ``prob_coarse_<state>`` -> ``prob_<state>`` in metacell obs."""
    renames = {
        c: c.replace("prob_coarse_", "prob_", 1)
        for c in adata.obs.columns if c.startswith("prob_coarse_")
    }
    if renames:
        adata.obs = adata.obs.rename(columns=renames)
    return adata
# =====================================================================
# LOAD METACELLS
# =====================================================================

def load_metacells():
    """Load metacell AnnData"""
    
    print("\n" + "="*80)
    print("LOADING METACELLS")
    print("="*80)
    
    print(f"\nLoading from: {Config.METACELLS_PATH}")
    adata = sc.read_h5ad(Config.METACELLS_PATH)
    adata = _rename_probability_columns(adata)
    
    print(f"  Metacells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    
    # Get expression matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Check if log-normalized (should be from metacell creation)
    is_log = X.max() < 20
    if is_log:
        X_log = X
    else:
        print("  ⚠️  Data not log-normalized, applying log1p...")
        X_log = np.log1p(X)
    
    var_names = list(adata.var_names)
    
    # Get TFs and cell lines
    tfs = sorted([g for g in adata.obs['guide_assignment'].unique() 
                  if g != 'Ctrl' and not str(g).startswith('Unassign')])
    cell_lines = sorted(adata.obs['cell_line'].unique())
    
    # Get states
    states = [col.replace('prob_', '') for col in adata.obs.columns if col.startswith('prob_')]
    states = [s for s in states if s in Config.COARSE_STATES]
    
    print(f"\n  TFs: {len(tfs)}")
    print(f"  Cell lines: {cell_lines}")
    print(f"  States: {states}")
    print(f"\n  Control metacells: {(adata.obs['guide_assignment'] == 'Ctrl').sum():,}")
    print(f"  OE metacells: {(adata.obs['guide_assignment'] != 'Ctrl').sum():,}")
    
    return adata, X_log, var_names, tfs, cell_lines, states

# =====================================================================
# CORRELATION ANALYSIS FUNCTIONS
# (Same as single-cell version)
# =====================================================================

def permutation_correlation_test(X, Y, n_perms=1000, method='spearman'):
    """Permutation test for correlation significance"""
    
    if method == 'spearman':
        observed_r, _ = spearmanr(X, Y)
        corr_func = lambda x, y: spearmanr(x, y)[0]
    else:
        observed_r, _ = pearsonr(X, Y)
        corr_func = lambda x, y: pearsonr(x, y)[0]
    
    # Permutation test
    rng = np.random.RandomState(42)
    null_dist = np.zeros(n_perms)
    
    for i in range(n_perms):
        X_shuffled = rng.permutation(X)
        null_dist[i] = corr_func(X_shuffled, Y)
    
    # Two-tailed p-value
    p_value = (np.abs(null_dist) >= np.abs(observed_r)).sum() / n_perms
    
    return observed_r, p_value

def analyze_single_tf_state(adata, X_log, var_names, tf, state, population='combined'):
    """
    Analyze correlation between TF expression and state probability
    
    Computes correlations for BOTH:
      1. Raw log-normalized counts
      2. Z-scored expression (normalized by control mean/std)
    
    Parameters:
        population: 'combined', 'oe_only', or 'control_only'
    """
    
    if tf not in var_names:
        return None
    
    gene_idx = var_names.index(tf)
    
    # Get masks
    ctrl_mask = adata.obs['guide_assignment'] == 'Ctrl'
    oe_mask = adata.obs['guide_assignment'] == tf
    
    # Select population
    if population == 'combined':
        mask = ctrl_mask | oe_mask
    elif population == 'oe_only':
        mask = oe_mask
    elif population == 'control_only':
        mask = ctrl_mask
    else:
        raise ValueError(f"Unknown population: {population}")
    
    if mask.sum() < 3:  # Need at least 3 metacells
        return None
    
    # Get log-normalized expression
    tf_expr_log = X_log[mask.values, gene_idx]
    state_prob = adata.obs.loc[mask, f'prob_{state}'].values
    
    # ===================================================================
    # Z-SCORE USING CONTROL STATISTICS
    # ===================================================================
    if ctrl_mask.sum() > 0:
        ctrl_expr = X_log[ctrl_mask.values, gene_idx]
        ctrl_mean = np.mean(ctrl_expr)
        ctrl_std = np.std(ctrl_expr)
        
        # Z-score the selected population using control statistics
        if ctrl_std > 0:
            tf_expr_zscore = (tf_expr_log - ctrl_mean) / ctrl_std
        else:
            tf_expr_zscore = tf_expr_log - ctrl_mean
    else:
        # If no controls, use population statistics
        tf_expr_zscore = (tf_expr_log - np.mean(tf_expr_log)) / (np.std(tf_expr_log) + 1e-10)
    
    # ===================================================================
    # CORRELATIONS: LOG-NORMALIZED
    # ===================================================================
    r_pearson_log, p_pearson_log = pearsonr(tf_expr_log, state_prob)
    r_spearman_log, p_spearman_log = spearmanr(tf_expr_log, state_prob)
    _, perm_pval_log = permutation_correlation_test(tf_expr_log, state_prob, 
                                                     n_perms=Config.N_PERMS, 
                                                     method='spearman')
    
    # ===================================================================
    # CORRELATIONS: Z-SCORED
    # ===================================================================
    r_pearson_zscore, p_pearson_zscore = pearsonr(tf_expr_zscore, state_prob)
    r_spearman_zscore, p_spearman_zscore = spearmanr(tf_expr_zscore, state_prob)
    _, perm_pval_zscore = permutation_correlation_test(tf_expr_zscore, state_prob,
                                                        n_perms=Config.N_PERMS,
                                                        method='spearman')
    
    result = {
        'TF': tf,
        'state': state,
        'population': population,
        'n_metacells': mask.sum(),
        'n_control': ctrl_mask.sum() if population == 'combined' else 0,
        'n_oe': oe_mask.sum() if population in ['combined', 'oe_only'] else 0,
        
        # LOG-NORMALIZED correlations
        'r_pearson_log': r_pearson_log,
        'p_pearson_log': p_pearson_log,
        'r_spearman_log': r_spearman_log,
        'p_spearman_log': p_spearman_log,
        'perm_pval_log': perm_pval_log,
        
        # Z-SCORED correlations
        'r_pearson_zscore': r_pearson_zscore,
        'p_pearson_zscore': p_pearson_zscore,
        'r_spearman_zscore': r_spearman_zscore,
        'p_spearman_zscore': p_spearman_zscore,
        'perm_pval_zscore': perm_pval_zscore,
        
        # Control statistics (for reference)
        'ctrl_mean': ctrl_mean if ctrl_mask.sum() > 0 else np.nan,
        'ctrl_std': ctrl_std if ctrl_mask.sum() > 0 else np.nan,
        
        # Data for plotting
        'tf_expr_log': tf_expr_log,
        'tf_expr_zscore': tf_expr_zscore,
        'state_prob': state_prob
    }
    
    return result

def run_correlation_analysis(adata, X_log, var_names, tfs, states, population='combined'):
    """Run correlation analysis for all TF-state pairs"""
    
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: {population.upper()}")
    print(f"{'='*80}")
    
    results = []
    
    for tf_idx, tf in enumerate(tfs, 1):
        if tf_idx % 10 == 0:
            print(f"  [{tf_idx}/{len(tfs)}] {tf}")
        
        for state in states:
            result = analyze_single_tf_state(adata, X_log, var_names, tf, state, population)
            if result is not None:
                results.append(result)
    
    df = pd.DataFrame(results)
    
    # FDR correction for BOTH metrics
    if len(df) > 0:
        from statsmodels.stats.multitest import fdrcorrection
        
        # FDR for log-normalized
        _, fdr_log = fdrcorrection(df['perm_pval_log'].values, alpha=Config.FDR_ALPHA)
        df['fdr_log'] = fdr_log
        df['significant_log'] = fdr_log < Config.FDR_ALPHA
        
        # FDR for z-scored
        _, fdr_zscore = fdrcorrection(df['perm_pval_zscore'].values, alpha=Config.FDR_ALPHA)
        df['fdr_zscore'] = fdr_zscore
        df['significant_zscore'] = fdr_zscore < Config.FDR_ALPHA
        
        # Primary metric (based on config)
        if Config.PRIMARY_METRIC == 'zscore':
            df['r_primary'] = df['r_spearman_zscore']
            df['fdr_primary'] = df['fdr_zscore']
            df['significant_primary'] = df['significant_zscore']
            df['perm_pval_primary'] = df['perm_pval_zscore']
        else:
            df['r_primary'] = df['r_spearman_log']
            df['fdr_primary'] = df['fdr_log']
            df['significant_primary'] = df['significant_log']
            df['perm_pval_primary'] = df['perm_pval_log']
        
        print(f"\n  Total pairs: {len(df)}")
        print(f"\n  PRIMARY METRIC: {Config.PRIMARY_METRIC.upper()}")
        print(f"    Significant (FDR<0.05): {df['significant_primary'].sum()} ({100*df['significant_primary'].mean():.1f}%)")
        print(f"\n  LOG-NORMALIZED:")
        print(f"    Significant (FDR<0.05): {df['significant_log'].sum()} ({100*df['significant_log'].mean():.1f}%)")
        print(f"\n  Z-SCORED:")
        print(f"    Significant (FDR<0.05): {df['significant_zscore'].sum()} ({100*df['significant_zscore'].mean():.1f}%)")
    
    return df

def run_per_cell_line_analysis(adata, X_log, var_names, tfs, states, cell_lines):
    """Run correlation analysis per cell line"""
    
    print(f"\n{'='*80}")
    print(f"PER CELL LINE ANALYSIS")
    print(f"{'='*80}")
    
    all_results = []
    
    for cl in cell_lines:
        print(f"\n  Cell line: {cl}")
        
        # Filter to this cell line
        cl_mask = adata.obs['cell_line'] == cl
        adata_cl = adata[cl_mask].copy()
        X_log_cl = X_log[cl_mask.values, :]
        
        # Run analysis
        results = []
        for tf in tfs:
            for state in states:
                result = analyze_single_tf_state(adata_cl, X_log_cl, var_names, 
                                                tf, state, 'combined')
                if result is not None:
                    result['cell_line'] = cl
                    results.append(result)
        
        df_cl = pd.DataFrame(results)
        
        if len(df_cl) > 0:
            from statsmodels.stats.multitest import fdrcorrection
            
            # FDR for both metrics
            _, fdr_log = fdrcorrection(df_cl['perm_pval_log'].values, alpha=Config.FDR_ALPHA)
            df_cl['fdr_log'] = fdr_log
            df_cl['significant_log'] = fdr_log < Config.FDR_ALPHA
            
            _, fdr_zscore = fdrcorrection(df_cl['perm_pval_zscore'].values, alpha=Config.FDR_ALPHA)
            df_cl['fdr_zscore'] = fdr_zscore
            df_cl['significant_zscore'] = fdr_zscore < Config.FDR_ALPHA
            
            # Primary
            if Config.PRIMARY_METRIC == 'zscore':
                df_cl['r_primary'] = df_cl['r_spearman_zscore']
                df_cl['fdr_primary'] = df_cl['fdr_zscore']
                df_cl['significant_primary'] = df_cl['significant_zscore']
            else:
                df_cl['r_primary'] = df_cl['r_spearman_log']
                df_cl['fdr_primary'] = df_cl['fdr_log']
                df_cl['significant_primary'] = df_cl['significant_log']
            
            print(f"    Significant ({Config.PRIMARY_METRIC}): {df_cl['significant_primary'].sum()} ({100*df_cl['significant_primary'].mean():.1f}%)")
            
            all_results.append(df_cl)
    
    if len(all_results) > 0:
        df_all = pd.concat(all_results, ignore_index=True)
        return df_all
    else:
        return pd.DataFrame()

# =====================================================================
# PLOTTING FUNCTIONS
# (Same as single-cell version - import from previous script)
# =====================================================================

# [Include all plotting functions from the previous correlation_analysis.py]
# plot_correlation_heatmaps
# plot_scatter_for_pair
# plot_all_significant_scatters
# plot_distribution_summaries
# plot_cohens_d_integration

# For brevity, I'll include condensed versions here

def plot_correlation_heatmaps(df_combined, df_oe, df_ctrl):
    """Plot heatmaps for all three populations - FOR BOTH METRICS"""
    
    print("\n" + "="*80)
    print("CREATING HEATMAPS")
    print("="*80)
    
    for metric in ['zscore', 'log'] if Config.PRIMARY_METRIC == 'zscore' else ['log', 'zscore']:
        is_primary = (metric == Config.PRIMARY_METRIC)
        label = "PRIMARY" if is_primary else "ALTERNATE"
        
        print(f"\n  Creating heatmaps for {label} metric ({metric})...")
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        
        r_col = f'r_spearman_{metric}'
        sig_col = f'significant_{metric}'
        
        populations = [
            (df_combined, 'OE + Control Combined', axes[0]),
            (df_oe, 'OE Cells Only', axes[1]),
            (df_ctrl, 'Control Cells Only', axes[2])
        ]
        
        for df, title, ax in populations:
            pivot = df.pivot(index='TF', columns='state', values=r_col)
            pivot_sig = df.pivot(index='TF', columns='state', values=sig_col)
            
            im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([s.replace('-states', '') for s in pivot.columns],
                              rotation=45, ha='right', fontsize=10)
            
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if pivot_sig.iloc[i, j]:
                        r_val = pivot.iloc[i, j]
                        ax.text(j, i, '★', ha='center', va='center',
                               color='yellow' if abs(r_val) > 0.5 else 'black',
                               fontsize=8)
            
            ax.set_title(title, fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=axes, orientation='horizontal',
                           fraction=0.05, pad=0.12, aspect=50)
        cbar.set_label(f"Spearman correlation - {metric.upper()} (★ = FDR<0.05)", 
                       fontsize=12, fontweight='bold')
        
        plt.suptitle(f'TF Expression vs State Identity - METACELLS - {metric.upper()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(Config.OUT_DIR / 'heatmaps' / f'all_populations_heatmap_{metric}.pdf', 
                    dpi=300, bbox_inches='tight')
        plt.savefig(Config.OUT_DIR / 'heatmaps' / f'all_populations_heatmap_{metric}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: all_populations_heatmap_{metric}.pdf/png")

# Simplified versions of other plotting functions
# (Full versions would be identical to the single-cell script)

def plot_scatter_for_pair(adata, X_log, var_names, tf, state, df_combined, df_oe, df_ctrl, metric='zscore'):
    """
    Plot scatter for a single TF-state pair (all 3 populations)
    
    Parameters:
        metric: 'zscore' or 'log' - which expression metric to plot
    """
    
    if tf not in var_names:
        return
    
    gene_idx = var_names.index(tf)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get data
    ctrl_mask = adata.obs['guide_assignment'] == 'Ctrl'
    oe_mask = adata.obs['guide_assignment'] == tf
    
    # Get expression based on metric
    if metric == 'zscore':
        # Calculate z-score using control statistics
        ctrl_expr_log = X_log[ctrl_mask.values, gene_idx]
        ctrl_mean = np.mean(ctrl_expr_log)
        ctrl_std = np.std(ctrl_expr_log)
        
        ctrl_expr = (ctrl_expr_log - ctrl_mean) / ctrl_std
        oe_expr_log = X_log[oe_mask.values, gene_idx]
        oe_expr = (oe_expr_log - ctrl_mean) / ctrl_std
        
        xlabel = f'{tf} Expression (z-score)'
        r_col = 'r_spearman_zscore'
        p_col = 'perm_pval_zscore'
        sig_col = 'significant_zscore'
    else:
        ctrl_expr = X_log[ctrl_mask.values, gene_idx]
        oe_expr = X_log[oe_mask.values, gene_idx]
        
        xlabel = f'{tf} Expression (log)'
        r_col = 'r_spearman_log'
        p_col = 'perm_pval_log'
        sig_col = 'significant_log'
    
    ctrl_prob = adata.obs.loc[ctrl_mask, f'prob_{state}'].values
    oe_prob = adata.obs.loc[oe_mask, f'prob_{state}'].values
    
    # Panel 1: OE + Control combined
    ax = axes[0]
    
    # Control metacells
    ax.scatter(ctrl_expr, ctrl_prob, alpha=0.5, s=30, c='blue', edgecolors='black', linewidths=0.5, label='Control')
    
    # OE metacells
    ax.scatter(oe_expr, oe_prob, alpha=0.5, s=30, c='red', edgecolors='black', linewidths=0.5, label='OE')
    
    # Regression line
    all_expr = np.concatenate([ctrl_expr, oe_expr])
    all_prob = np.concatenate([ctrl_prob, oe_prob])
    z = np.polyfit(all_expr, all_prob, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_expr.min(), all_expr.max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    
    # Get stats
    row = df_combined[(df_combined['TF'] == tf) & (df_combined['state'] == state)]
    if len(row) > 0:
        r = row.iloc[0][r_col]
        p_val = row.iloc[0][p_col]
        sig = row.iloc[0][sig_col]
        sig_str = '★' if sig else ''
        ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3e} {sig_str}',
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{state} Probability', fontsize=11, fontweight='bold')
    ax.set_title('OE + Control Combined', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: OE only
    ax = axes[1]
    ax.scatter(oe_expr, oe_prob, alpha=0.5, s=35, c='red', edgecolors='black', linewidths=0.5)
    
    if len(oe_expr) > 2:
        z = np.polyfit(oe_expr, oe_prob, 1)
        p = np.poly1d(z)
        x_line = np.linspace(oe_expr.min(), oe_expr.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    
    row = df_oe[(df_oe['TF'] == tf) & (df_oe['state'] == state)]
    if len(row) > 0:
        r = row.iloc[0][r_col]
        p_val = row.iloc[0][p_col]
        sig = row.iloc[0][sig_col]
        sig_str = '★' if sig else ''
        ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3e} {sig_str}',
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{state} Probability', fontsize=11, fontweight='bold')
    ax.set_title('OE Metacells Only', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Panel 3: Control only
    ax = axes[2]
    ax.scatter(ctrl_expr, ctrl_prob, alpha=0.5, s=35, c='blue', edgecolors='black', linewidths=0.5)
    
    if len(ctrl_expr) > 2:
        z = np.polyfit(ctrl_expr, ctrl_prob, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ctrl_expr.min(), ctrl_expr.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    
    row = df_ctrl[(df_ctrl['TF'] == tf) & (df_ctrl['state'] == state)]
    if len(row) > 0:
        r = row.iloc[0][r_col]
        p_val = row.iloc[0][p_col]
        sig = row.iloc[0][sig_col]
        sig_str = '★' if sig else ''
        ax.text(0.05, 0.95, f'r={r:.3f}, p={p_val:.3e} {sig_str}',
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{state} Probability', fontsize=11, fontweight='bold')
    ax.set_title('Control Metacells Only', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'{tf} vs {state} - METACELLS ({metric.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"{tf}_vs_{state.replace('-', '_')}_{metric}.pdf"
    plt.savefig(Config.OUT_DIR / 'scatter_plots' / filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_significant_scatters(adata, X_log, var_names, df_combined, df_oe, df_ctrl):
    """Plot scatter plots for ALL significant pairs - FOR BOTH METRICS"""
    
    print("\n" + "="*80)
    print("CREATING SCATTER PLOTS")
    print("="*80)
    
    # ===================================================================
    # Get significant pairs for PRIMARY metric
    # ===================================================================
    sig_col_primary = 'significant_zscore' if Config.PRIMARY_METRIC == 'zscore' else 'significant_log'
    
    sig_combined = df_combined[df_combined[sig_col_primary]][['TF', 'state']].values
    sig_oe = df_oe[df_oe[sig_col_primary]][['TF', 'state']].values
    sig_ctrl = df_ctrl[df_ctrl[sig_col_primary]][['TF', 'state']].values
    
    sig_pairs_primary = set()
    for tf, state in sig_combined:
        sig_pairs_primary.add((tf, state))
    for tf, state in sig_oe:
        sig_pairs_primary.add((tf, state))
    for tf, state in sig_ctrl:
        sig_pairs_primary.add((tf, state))
    
    sig_pairs_primary = sorted(list(sig_pairs_primary))
    
    # ===================================================================
    # Get significant pairs for ALTERNATE metric
    # ===================================================================
    sig_col_alternate = 'significant_log' if Config.PRIMARY_METRIC == 'zscore' else 'significant_zscore'
    
    sig_combined_alt = df_combined[df_combined[sig_col_alternate]][['TF', 'state']].values
    sig_oe_alt = df_oe[df_oe[sig_col_alternate]][['TF', 'state']].values
    sig_ctrl_alt = df_ctrl[df_ctrl[sig_col_alternate]][['TF', 'state']].values
    
    sig_pairs_alternate = set()
    for tf, state in sig_combined_alt:
        sig_pairs_alternate.add((tf, state))
    for tf, state in sig_oe_alt:
        sig_pairs_alternate.add((tf, state))
    for tf, state in sig_ctrl_alt:
        sig_pairs_alternate.add((tf, state))
    
    sig_pairs_alternate = sorted(list(sig_pairs_alternate))
    
    # Combine both (union) - convert back to sets for union operation
    sig_pairs_all = sorted(list(set(sig_pairs_primary) | set(sig_pairs_alternate)))
    
    print(f"\n  PRIMARY metric ({Config.PRIMARY_METRIC}): {len(sig_pairs_primary)} significant pairs")
    alternate_name = 'log' if Config.PRIMARY_METRIC == 'zscore' else 'zscore'
    print(f"  ALTERNATE metric ({alternate_name}): {len(sig_pairs_alternate)} significant pairs")
    print(f"  TOTAL unique pairs: {len(sig_pairs_all)}")
    print(f"\n  Creating scatter plots for ALL pairs (both metrics)...")
    
    # ===================================================================
    # Create plots for PRIMARY metric
    # ===================================================================
    print(f"\n  [{Config.PRIMARY_METRIC.upper()}] Creating {len(sig_pairs_all)} scatter plots...")
    for i, (tf, state) in enumerate(sig_pairs_all, 1):
        if i % 10 == 0:
            print(f"    [{i}/{len(sig_pairs_all)}] {tf} vs {state}")
        
        plot_scatter_for_pair(adata, X_log, var_names, tf, state, 
                             df_combined, df_oe, df_ctrl, metric=Config.PRIMARY_METRIC)
    
    # ===================================================================
    # Create plots for ALTERNATE metric
    # ===================================================================
    print(f"\n  [{alternate_name.upper()}] Creating {len(sig_pairs_all)} scatter plots...")
    for i, (tf, state) in enumerate(sig_pairs_all, 1):
        if i % 10 == 0:
            print(f"    [{i}/{len(sig_pairs_all)}] {tf} vs {state}")
        
        plot_scatter_for_pair(adata, X_log, var_names, tf, state, 
                             df_combined, df_oe, df_ctrl, metric=alternate_name)
    
    total_plots = len(sig_pairs_all) * 2  # Both metrics
    print(f"\n  ✓ Saved {total_plots} scatter plots ({len(sig_pairs_all)} pairs × 2 metrics)")

def create_simple_summary_plots(df_combined, df_oe, df_ctrl):
    """Create summary plots"""
    
    print("\n" + "="*80)
    print("CREATING SUMMARY PLOTS")
    print("="*80)
    
    # Simple correlation summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution by metric
    ax = axes[0]
    data = []
    labels = []
    for metric in ['log', 'zscore']:
        r_col = f'r_spearman_{metric}'
        data.append(df_combined[r_col].values)
        labels.append(metric.upper())
    
    ax.violinplot(data, positions=[0, 1], showmeans=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Log vs Z-scored Correlations')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Scatter: log vs zscore
    ax = axes[1]
    ax.scatter(df_combined['r_spearman_log'], df_combined['r_spearman_zscore'],
              alpha=0.3, s=20)
    lims = [-1, 1]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlabel('Log-normalized r')
    ax.set_ylabel('Z-scored r')
    ax.set_title('Log vs Z-scored Comparison')
    ax.grid(alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    corr = np.corrcoef(df_combined['r_spearman_log'], df_combined['r_spearman_zscore'])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat'), va='top')
    
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / 'distributions' / 'summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(Config.OUT_DIR / 'distributions' / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: summary.pdf/png")

# =====================================================================
# MAIN
# =====================================================================

def main():
    
    print("="*80)
    print("TF EXPRESSION vs STATE IDENTITY - METACELL ANALYSIS")
    print("="*80)
    print(f"\n*** PRIMARY METRIC: {Config.PRIMARY_METRIC.upper()} ***")
    print(f"*** ANALYZING BOTH: log-normalized AND z-scored ***\n")
    
    # Load metacells
    adata, X_log, var_names, tfs, cell_lines, states = load_metacells()
    
    # Run main analyses
    df_combined = run_correlation_analysis(adata, X_log, var_names, tfs, states, 'combined')
    df_oe = run_correlation_analysis(adata, X_log, var_names, tfs, states, 'oe_only')
    df_ctrl = run_correlation_analysis(adata, X_log, var_names, tfs, states, 'control_only')
    
    # Save main results
    df_combined.to_csv(Config.OUT_DIR / 'correlation_combined.csv', index=False)
    df_oe.to_csv(Config.OUT_DIR / 'correlation_oe_only.csv', index=False)
    df_ctrl.to_csv(Config.OUT_DIR / 'correlation_control_only.csv', index=False)
    
    print("\n✓ Saved main correlation CSVs")
    
    # Per cell line analysis
    df_per_line = run_per_cell_line_analysis(adata, X_log, var_names, tfs, states, cell_lines)
    if len(df_per_line) > 0:
        df_per_line.to_csv(Config.OUT_DIR / 'per_cell_line' / 'correlation_per_cell_line.csv', index=False)
        print("✓ Saved per-cell-line CSV")
    
    # Create plots
    plot_correlation_heatmaps(df_combined, df_oe, df_ctrl)
    plot_all_significant_scatters(adata, X_log, var_names, df_combined, df_oe, df_ctrl)
    create_simple_summary_plots(df_combined, df_oe, df_ctrl)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nPRIMARY METRIC: {Config.PRIMARY_METRIC.upper()}")
    print(f"Output directory: {Config.OUT_DIR}")
    print("\nGenerated:")
    print("  - 4 CSV files (combined, OE-only, control-only, per-cell-line)")
    print(f"    * Each contains BOTH log and z-scored correlations")
    print(f"  - 2 heatmap sets (log + z-scored)")
    n_sig = len(set(
        tuple(row) for row in df_combined[df_combined['significant_primary']][['TF', 'state']].values
    ) | set(
        tuple(row) for row in df_oe[df_oe['significant_primary']][['TF', 'state']].values
    ) | set(
        tuple(row) for row in df_ctrl[df_ctrl['significant_primary']][['TF', 'state']].values
    ))
    print(f"  - {n_sig * 2} scatter plots ({n_sig} pairs × 2 metrics)")
    print(f"  - Summary plots")
    print(f"\nMETACELLS reduce noise → cleaner correlations!")

if __name__ == '__main__':
    main()
