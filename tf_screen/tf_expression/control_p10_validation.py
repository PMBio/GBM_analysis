"""Control-cell state composition with and without p10 TF-expression filtering.

For every TF, this script asks: does the bottom-10th-percentile filter
(``p10``) systematically distort the cell-state composition of the
control population that is then compared with OE cells?

It computes the coarse-state distribution of controls under three
conditions:

* All control cells (baseline).
* Bottom 10% of control cells by TF expression (the ``p10`` filter).
* Bottom 25% of control cells by TF expression (the ``p25`` filter).

Two distribution metrics are computed for each condition:

* **Mean probabilities** -- soft assignments, sum to one. Use the
  classifier output as a continuous distribution over coarse states.
* **Argmax proportions** -- hard assignments. Cells are bucketed by
  their argmax state and the proportion in each state is reported.

The per-TF shift between baseline and filtered conditions is the
quantity reported in Extended Data Fig. 5c (right panel): if the median
shift is near zero, the p10 filter does not bias the control reference
population.

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Must contain ``prob_coarse_<state>`` columns produced by
    ``classifiers/train_state_classifier.py``.

Outputs
-------
``OUTPUT_DIR/control_state_analysis/control_states_by_cell_line.csv``
``OUTPUT_DIR/control_state_analysis/control_states_tf_filtering.csv``
``OUTPUT_DIR/control_state_analysis/control_state_shifts.csv``
``OUTPUT_DIR/control_state_analysis/fig_*.{pdf,png}``

Usage
-----
``python tf_expression/control_p.py10_validation``
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import normalise_lognorm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class Config:
    """Local config alias. Original script's plotting code references
    ``Config.OUT_DIR``, ``Config.BOTTOM_PERCENTILES``, and
    ``Config.COARSE_STATES``; we keep that interface to avoid touching
    the (already-validated) plotting body.
    """
    ANNDATA_PATH = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    HARMONY_PATH = config.HARMONY_DIR / "harmony_embeddings.npy"
    MODEL_DIR    = config.MODELS_DIR
    OUT_DIR      = config.OUTPUT_DIR / "control_state_analysis"

    BOTTOM_PERCENTILES = [10, 25]
    COARSE_STATES = list(config.COARSE_GROUPS.keys())


Config.OUT_DIR.mkdir(exist_ok=True, parents=True)
# LOAD AND PREDICT
# =====================================================================

def load_data_and_predict():
    """Load data and predict on all OE cells"""
    
    print("\n" + "="*80)
    print("LOADING DATA AND PREDICTING")
    print("="*80)
    
    print("\nLoading AnnData...")
    adata = sc.read_h5ad(Config.ANNDATA_PATH)
    
    if adata.X.max() > 20:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    print("\nLoading Harmony embeddings...")
    X_harmony = np.load(Config.HARMONY_PATH)
    adata.obsm['X_harmony'] = X_harmony
    
    gbm_mask = adata.obs['dataset'].astype(str).str.lower().str.contains('gbm', na=False)
    adata_oe = adata[~gbm_mask].copy()
    
    print(f"OE cells: {adata_oe.n_obs:,}")
    
    print("\nLoading models...")
    clf = joblib.load(Config.MODEL_DIR / 'clf_celltype_coarse.pkl')
    scaler = joblib.load(Config.MODEL_DIR / 'scaler_celltype_coarse.pkl')
    le = joblib.load(Config.MODEL_DIR / 'le_celltype_coarse.pkl')
    
    print("Predicting on ALL OE cells...")
    X_scaled = scaler.transform(adata_oe.obsm['X_harmony'])
    probs = clf.predict_proba(X_scaled)
    
    for i, state in enumerate(le.classes_):
        adata_oe.obs[f'prob_{state}'] = probs[:, i]
    
    # Argmax assignment
    argmax_indices = np.argmax(probs, axis=1)
    adata_oe.obs['argmax_state'] = [le.classes_[i] for i in argmax_indices]
    
    print("✓ Predictions complete")
    
    ctrl_mask = adata_oe.obs['guide_assignment'] == 'Ctrl'
    adata_ctrl = adata_oe[ctrl_mask].copy()
    
    print(f"Control cells: {adata_ctrl.n_obs:,}")
    
    return adata_oe, adata_ctrl, le.classes_

# =====================================================================
# ANALYSIS FUNCTIONS
# =====================================================================

def get_state_metrics(adata, indices, states):
    """Get both mean probabilities and argmax proportions"""
    
    # Mean probabilities
    mean_probs = {}
    for state in states:
        prob_col = f'prob_{state}'
        mean_probs[f'mean_prob_{state}'] = adata.obs.iloc[indices][prob_col].mean()
    
    # Argmax proportions
    argmax_states = adata.obs.iloc[indices]['argmax_state']
    proportions = {}
    for state in states:
        count = (argmax_states == state).sum()
        proportions[f'prop_{state}'] = count / len(indices) if len(indices) > 0 else 0
    
    return {**mean_probs, **proportions}

# =====================================================================
# ANALYSIS 1: BY CELL LINE
# =====================================================================

def analyze_by_cell_line(adata_ctrl, states):
    """Analyze control states by cell line"""
    
    print("\n" + "="*80)
    print("ANALYSIS 1: BY CELL LINE")
    print("="*80)
    
    cell_lines = sorted(adata_ctrl.obs['cell_line'].unique())
    
    results = []
    
    for cl in cell_lines:
        cl_mask = adata_ctrl.obs['cell_line'] == cl
        cl_indices = np.where(cl_mask)[0]
        n_cells = len(cl_indices)
        
        print(f"\n{cl}: {n_cells} cells")
        
        metrics = get_state_metrics(adata_ctrl, cl_indices, states)
        
        results.append({
            'cell_line': cl,
            'n_cells': n_cells,
            **metrics
        })
        
        # Print
        print("  Mean probabilities:")
        for state in states:
            print(f"    {state:20s}: {metrics[f'mean_prob_{state}']:.3f}")
        
        print("  Argmax proportions:")
        for state in states:
            print(f"    {state:20s}: {100*metrics[f'prop_{state}']:5.1f}%")
    
    df = pd.DataFrame(results)
    
    path = Config.OUT_DIR / 'control_states_by_cell_line.csv'
    df.to_csv(path, index=False)
    print(f"\n✓ Saved: {path}")
    
    return df

# =====================================================================
# ANALYSIS 2: TF FILTERING
# =====================================================================

def analyze_tf_filtering(adata_oe, states):
    """Analyze effect of removing high TF expressors"""
    
    print("\n" + "="*80)
    print("ANALYSIS 2: TF FILTERING")
    print("="*80)
    
    X = adata_oe.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    if X.max() < 20:
        X = np.expm1(X)
    
    var_names = list(adata_oe.var_names)
    
    tfs = sorted([g for g in adata_oe.obs['guide_assignment'].unique() 
                  if g != 'Ctrl' and not str(g).startswith('Unassign')])
    
    print(f"\nTFs: {len(tfs)}")
    
    ctrl_mask = adata_oe.obs['guide_assignment'] == 'Ctrl'
    ctrl_indices = np.where(ctrl_mask)[0]
    
    results = []
    
    for tf_idx, tf in enumerate(tfs, 1):
        if tf_idx % 10 == 0:
            print(f"  [{tf_idx}/{len(tfs)}] {tf}")
        
        if tf not in var_names:
            continue
        
        gene_idx = var_names.index(tf)
        ctrl_expr = X[ctrl_indices, gene_idx]
        
        # All controls
        metrics_all = get_state_metrics(adata_oe, ctrl_indices, states)
        results.append({
            'TF': tf,
            'filter': 'all',
            'n_cells': len(ctrl_indices),
            **metrics_all
        })
        
        # Bottom percentiles
        for pct in Config.BOTTOM_PERCENTILES:
            cutoff = np.percentile(ctrl_expr, pct)
            bottom_mask = ctrl_expr <= cutoff
            bottom_indices = ctrl_indices[bottom_mask]
            
            if len(bottom_indices) < 10:
                continue
            
            metrics_bottom = get_state_metrics(adata_oe, bottom_indices, states)
            results.append({
                'TF': tf,
                'filter': f'bottom_{pct}',
                'n_cells': len(bottom_indices),
                **metrics_bottom
            })
    
    df = pd.DataFrame(results)
    
    path = Config.OUT_DIR / 'control_states_tf_filtering.csv'
    df.to_csv(path, index=False)
    print(f"\n✓ Saved: {path}")
    
    return df

# =====================================================================
# CALCULATE SHIFTS
# =====================================================================

def calculate_shifts(df):
    """Calculate shifts for both mean probs and argmax proportions"""
    
    print("\n" + "="*80)
    print("CALCULATING SHIFTS")
    print("="*80)
    
    shifts = []
    
    for tf in df['TF'].unique():
        df_tf = df[df['TF'] == tf]
        baseline = df_tf[df_tf['filter'] == 'all']
        
        if len(baseline) == 0:
            continue
        
        baseline = baseline.iloc[0]
        
        for pct in Config.BOTTOM_PERCENTILES:
            filtered = df_tf[df_tf['filter'] == f'bottom_{pct}']
            
            if len(filtered) == 0:
                continue
            
            filtered = filtered.iloc[0]
            
            for state in Config.COARSE_STATES:
                # Mean prob shifts
                base_prob = baseline[f'mean_prob_{state}']
                filt_prob = filtered[f'mean_prob_{state}']
                prob_shift = filt_prob - base_prob
                
                # Argmax prop shifts
                base_prop = baseline[f'prop_{state}']
                filt_prop = filtered[f'prop_{state}']
                prop_shift = filt_prop - base_prop
                
                shifts.append({
                    'TF': tf,
                    'filter': f'bottom_{pct}',
                    'state': state,
                    'baseline_mean_prob': base_prob,
                    'filtered_mean_prob': filt_prob,
                    'mean_prob_shift': prob_shift,
                    'baseline_prop': base_prop,
                    'filtered_prop': filt_prop,
                    'prop_shift': prop_shift
                })
    
    df_shifts = pd.DataFrame(shifts)
    
    path = Config.OUT_DIR / 'control_state_shifts.csv'
    df_shifts.to_csv(path, index=False)
    print(f"\n✓ Saved: {path}")
    
    return df_shifts

# =====================================================================
# PLOTTING
# =====================================================================

def create_plots(df_by_cl, df_tf, df_shifts):
    """Create comprehensive plots"""
    
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    plot_by_cell_line(df_by_cl)
    
    if len(df_shifts) > 0:
        plot_shifts_comparison(df_shifts)
        plot_example_tfs(df_tf)
        plot_shift_heatmaps(df_shifts)

def plot_by_cell_line(df):
    """Plot both mean probs and argmax proportions by cell line"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cell_lines = df['cell_line'].unique()
    states = Config.COARSE_STATES
    
    # Panel 1: Mean probabilities (sum to 1)
    ax = axes[0]
    bottom = np.zeros(len(cell_lines))
    
    for state in states:
        probs = [df[df['cell_line'] == cl][f'mean_prob_{state}'].values[0] for cl in cell_lines]
        ax.bar(cell_lines, probs, bottom=bottom, label=state, alpha=0.8, edgecolor='black')
        bottom += probs
    
    ax.set_ylabel('Mean Probability', fontsize=12, fontweight='bold')
    ax.set_title('Mean State Probabilities (Soft Assignment)\nSum = 1.0', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 2: Argmax proportions (sum to 100%)
    ax = axes[1]
    bottom = np.zeros(len(cell_lines))
    
    for state in states:
        props = [100 * df[df['cell_line'] == cl][f'prop_{state}'].values[0] for cl in cell_lines]
        ax.bar(cell_lines, props, bottom=bottom, label=state, alpha=0.8, edgecolor='black')
        bottom += props
    
    ax.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
    ax.set_title('Argmax State Proportions (Hard Assignment)\nSum = 100%', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / 'by_cell_line.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(Config.OUT_DIR / 'by_cell_line.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: by_cell_line.pdf/png")

def plot_shifts_comparison(df_shifts):
    """Compare mean prob shifts vs argmax prop shifts"""
    
    # Get bottom_10 only
    df_10 = df_shifts[df_shifts['filter'] == 'bottom_10']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, state in enumerate(Config.COARSE_STATES):
        ax = axes[i]
        
        df_state = df_10[df_10['state'] == state].copy()
        
        # Sort by prop_shift
        df_state = df_state.sort_values('prop_shift', ascending=True)
        
        # Take top and bottom
        top_n = 15
        df_plot = pd.concat([df_state.head(top_n), df_state.tail(top_n)])
        
        y_pos = np.arange(len(df_plot))
        
        # Plot both shifts
        width = 0.4
        ax.barh(y_pos - width/2, 100 * df_plot['prop_shift'], width, 
               label='Argmax Prop Shift', alpha=0.7, edgecolor='black', color='steelblue')
        ax.barh(y_pos + width/2, 100 * df_plot['mean_prob_shift'], width,
               label='Mean Prob Shift', alpha=0.7, edgecolor='black', color='coral')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['TF'], fontsize=7)
        ax.set_xlabel('Shift (%)', fontsize=11, fontweight='bold')
        ax.set_title(state, fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    
    axes[5].axis('off')
    
    plt.suptitle('State Shifts: Mean Probability vs Argmax Proportion\n(Bottom 10% vs All Controls)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / 'shifts_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(Config.OUT_DIR / 'shifts_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: shifts_comparison.pdf/png")

def plot_example_tfs(df_tf):
    """Show before/after for example TFs"""
    
    example_tfs = df_tf['TF'].unique()[:12]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, tf in enumerate(example_tfs):
        if i >= 12:
            break
        
        ax = axes[i]
        
        df_tf_data = df_tf[df_tf['TF'] == tf]
        
        filters = ['all', 'bottom_10', 'bottom_25']
        x = np.arange(len(Config.COARSE_STATES))
        width = 0.25
        
        for j, filt in enumerate(filters):
            df_filt = df_tf_data[df_tf_data['filter'] == filt]
            if len(df_filt) == 0:
                continue
            
            # Use argmax proportions
            props = [100 * df_filt.iloc[0][f'prop_{state}'] for state in Config.COARSE_STATES]
            ax.bar(x + j*width, props, width, label=filt, alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace('-states', '') for s in Config.COARSE_STATES],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Proportion (%)', fontsize=10)
        ax.set_title(tf, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Example TFs: Argmax Proportions Before/After Filtering', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / 'example_tfs.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(Config.OUT_DIR / 'example_tfs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: example_tfs.pdf/png")

def plot_shift_heatmaps(df_shifts):
    """Heatmaps of shifts"""
    
    df_10 = df_shifts[df_shifts['filter'] == 'bottom_10']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    
    # Mean prob shifts
    ax = axes[0]
    pivot_prob = df_10.pivot(index='TF', columns='state', values='mean_prob_shift')
    total_shift = pivot_prob.abs().sum(axis=1)
    pivot_prob = pivot_prob.loc[total_shift.sort_values(ascending=False).index].head(30)
    
    im = ax.imshow(pivot_prob.values, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax.set_yticks(np.arange(len(pivot_prob)))
    ax.set_yticklabels(pivot_prob.index, fontsize=9)
    ax.set_xticks(np.arange(len(Config.COARSE_STATES)))
    ax.set_xticklabels(Config.COARSE_STATES, rotation=45, ha='right', fontsize=10)
    ax.set_title('Mean Probability Shifts', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability Shift')
    
    # Argmax prop shifts
    ax = axes[1]
    pivot_prop = df_10.pivot(index='TF', columns='state', values='prop_shift')
    total_shift = pivot_prop.abs().sum(axis=1)
    pivot_prop = pivot_prop.loc[total_shift.sort_values(ascending=False).index].head(30)
    
    im = ax.imshow(pivot_prop.values, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
    ax.set_yticks(np.arange(len(pivot_prop)))
    ax.set_yticklabels(pivot_prop.index, fontsize=9)
    ax.set_xticks(np.arange(len(Config.COARSE_STATES)))
    ax.set_xticklabels(Config.COARSE_STATES, rotation=45, ha='right', fontsize=10)
    ax.set_title('Argmax Proportion Shifts', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Proportion Shift')
    
    plt.suptitle('Top 30 TFs: State Shifts (Bottom 10% vs All)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / 'shift_heatmaps.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(Config.OUT_DIR / 'shift_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved: shift_heatmaps.pdf/png")

# =====================================================================
# MAIN
# =====================================================================

def main():
    
    print("="*80)
    print("CONTROL CELL STATE ANALYSIS - REVISED")
    print("="*80)
    
    adata_oe, adata_ctrl, states = load_data_and_predict()
    
    df_by_cl = analyze_by_cell_line(adata_ctrl, states)
    df_tf = analyze_tf_filtering(adata_oe, states)
    df_shifts = calculate_shifts(df_tf)
    
    create_plots(df_by_cl, df_tf, df_shifts)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nCell line differences (Argmax):")
    for cl in df_by_cl['cell_line'].unique():
        row = df_by_cl[df_by_cl['cell_line'] == cl].iloc[0]
        top_state = max(Config.COARSE_STATES, key=lambda s: row[f'prop_{s}'])
        top_prop = row[f'prop_{top_state}']
        print(f"  {cl}: {100*top_prop:.1f}% {top_state}")
    
    if len(df_shifts) > 0:
        print("\nMean systematic shifts (Bottom 10%, Argmax):")
        for state in Config.COARSE_STATES:
            df_state = df_shifts[(df_shifts['state'] == state) & (df_shifts['filter'] == 'bottom_10')]
            if len(df_state) > 0:
                mean_shift = df_state['prop_shift'].mean()
                print(f"  {state:20s}: {100*mean_shift:+.2f}%")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput: {Config.OUT_DIR}")

if __name__ == '__main__':
    main()
