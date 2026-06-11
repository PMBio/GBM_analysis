"""TF self-expression analysis: log2FC and bimodality of OE versus control cells.

For every TF in the screen, this script quantifies how strongly that TF
is induced in cells assigned to its guide, compared with the cells
assigned to the non-targeting control guide, both per cell line and
pooled across all three.

Two complementary fold-change definitions are computed:

* ``log2fc_counts``  -- log2 of the ratio of arithmetic means of
  10k-normalised counts. Susceptible to extreme-value bias.
* ``log2fc_lognorm`` -- difference of mean log1p-normalised expression,
  divided by ln(2). Equivalent to the log2 ratio of geometric means.
  **This is the headline log2FC reported throughout the paper.**

Sarle's sample-corrected bimodality coefficient is computed on the OE
distribution per TF (and separately on the control distribution) to
justify the FC0.5 + p10 filtering strategy in downstream tests.

Significance (Mann--Whitney U on raw OE vs Ctrl) is reported with BH
multiple-testing correction, *per scope* (pooled, BG5, P3, S24).

Inputs
------
``SCREEN_ANNDATA`` (the cleaned screen object; ``cell_line`` and
``guide_assignment`` columns required, ``guide_has_assignment`` optional).

Outputs
-------
``TF_EXPRESSION_DIR/tf_expression_results.csv``
    One row per (TF, scope) with means/medians/std, both log2FC
    flavours, detection rates, Cohen's d, MWU p-value, BH-adjusted padj.
``TF_EXPRESSION_DIR/fig_*.{pdf,png}``
    Heatmap, volcano, top-TF violins, expression scatter, cell-line
    concordance.

Usage
-----
``python tf_expression/tf_self_expression.py``
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import detect_data_type

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths and plotting style
# ---------------------------------------------------------------------------


class Config:
    """Local config alias -- kept as a class so the plotting code below
    (which references ``Config.OUT_DIR``) does not need to change.
    """
    ANNDATA_PATH = config.SCREEN_ANNDATA
    OUT_DIR = config.TF_EXPRESSION_DIR


Config.OUT_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

CELL_LINE_COLORS = {"P3": "#E64B35", "S24": "#4DBBD5", "BG5": "#00A087"}
# =====================================================================
# DATA LOADING
# =====================================================================

def load_and_prepare():
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    adata = sc.read_h5ad(Config.ANNDATA_PATH)
    print(f"  Total cells loaded: {adata.n_obs:,}")

    # Keep OE cells only (not GBM baseline)
    # oe_mask = ~adata.obs['dataset'].astype(str).str.lower().str.contains('gbm', na=False)
    # adata   = adata[oe_mask].copy()
    print(f"  OE cells: {adata.n_obs:,}")

    # ── CRITICAL: filter to guide-assigned cells ──────────────────────
    if 'guide_has_assignment' in adata.obs.columns:
        n_before = adata.n_obs
        adata    = adata[adata.obs['guide_has_assignment'] == True].copy()
        print(f"  After guide_has_assignment filter: {adata.n_obs:,} "
              f"(removed {n_before - adata.n_obs:,} unassigned)")
    else:
        print("  WARNING: guide_has_assignment column not found — "
              "unassigned cells may be included in control group.")

    # ── Data type ─────────────────────────────────────────────────────
    print("\n  Checking data type …")
    X_raw_dense = adata.X
    if hasattr(X_raw_dense, 'toarray'):
        X_raw_dense = X_raw_dense.toarray()

    dtype = detect_data_type(adata.X)

    if dtype == 'lognorm':
        print("  ✓ Data is log1p-normalised (base e). Using as-is.")
        X_lognorm = X_raw_dense
        X_counts  = np.expm1(X_raw_dense)   # back to library-size-normalised counts
    elif dtype == 'raw_counts':
        print("  ✓ Raw counts detected. Applying normalize_total(1e4) + log1p ...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        X_lognorm = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
        X_counts  = np.expm1(X_lognorm)
    else:
        # dtype == 'normalised': row sums are equal but not log-transformed
        print("  ✓ Library-size-normalised (not log). Applying log1p ...")
        X_lognorm = np.log1p(X_raw_dense)
        X_counts  = X_raw_dense

    # Sanity check: after all paths X_lognorm max should be < 15
    lognorm_max = float(X_lognorm.max())
    lognorm_med = float(np.median(X_lognorm[X_lognorm > 0]))
    print(f"  Post-processing check → X_lognorm max={lognorm_max:.3f}, "
          f"median non-zero={lognorm_med:.4f}")
    if lognorm_max > 20:
        print("  WARNING: X_lognorm max > 20 — normalisation may not have worked correctly.")
    else:
        print("  ✓ X_lognorm looks correct.")

    var_names  = list(adata.var_names)

    # TFs = guide labels that are not control or unassigned
    tfs = sorted([g for g in adata.obs['guide_assignment'].unique()
                  if g not in ('Ctrl',) and not str(g).startswith('Unassign')])
    cell_lines = sorted(adata.obs['cell_line'].unique())

    print(f"\n  TFs to analyse: {len(tfs)}")
    print(f"  Cell lines    : {cell_lines}")

    return adata, X_counts, X_lognorm, var_names, tfs, cell_lines, dtype


# =====================================================================
# ANALYSIS
# =====================================================================

def analyze_single_tf(X_counts, X_lognorm, gene_idx, tf, obs, scope_label, scope_mask=None):
    """
    Returns summary stats dict for one TF in one scope (cell line or pooled).

    FIX vs v1: log2fc is now correctly computed.
      - From log1p (base-e) data: log2fc = (mean_oe_log - mean_ctrl_log) / ln(2)
      - From raw data: log2fc = log2(mean_oe / mean_ctrl)
      Both are computed; use log2fc_from_lognorm for lognorm data.
    """
    if scope_mask is not None:
        obs_sub      = obs[scope_mask]
        X_c_sub      = X_counts[scope_mask.values]
        X_l_sub      = X_lognorm[scope_mask.values]
    else:
        obs_sub = obs
        X_c_sub = X_counts
        X_l_sub = X_lognorm

    ctrl_idx = (obs_sub['guide_assignment'] == 'Ctrl').values
    oe_idx   = (obs_sub['guide_assignment'] == tf).values

    ctrl_c   = X_c_sub[ctrl_idx, gene_idx]
    oe_c     = X_c_sub[oe_idx,   gene_idx]
    ctrl_l   = X_l_sub[ctrl_idx, gene_idx]
    oe_l     = X_l_sub[oe_idx,   gene_idx]

    n_ctrl, n_oe = len(ctrl_c), len(oe_c)

    def safe_div(a, b):
        return a / b if b > 0 else (np.inf if a > 0 else np.nan)

    ctrl_mean_c = np.mean(ctrl_c)
    oe_mean_c   = np.mean(oe_c)
    ctrl_mean_l = np.mean(ctrl_l)
    oe_mean_l   = np.mean(oe_l)

    # ── fold changes ──────────────────────────────────────────────────
    # From normalised counts (can be biased by dropouts, but interpretable)
    fc_counts       = safe_div(oe_mean_c, ctrl_mean_c)
    log2fc_counts   = np.log2(fc_counts) if np.isfinite(fc_counts) and fc_counts > 0 else np.nan

    # From log1p space (v1 called this "log2fc_log" but it was loge FC, not log2!)
    # Correct: difference in log1p = ln(oe+1) - ln(ctrl+1) ≈ ln(FC) → divide by ln(2) for log2
    log2fc_lognorm  = (oe_mean_l - ctrl_mean_l) / np.log(2)   # ← FIXED

    # Detection
    ctrl_det = float(np.mean(ctrl_c > 0))
    oe_det   = float(np.mean(oe_c   > 0))

    # Statistics (only if enough cells)
    if n_ctrl >= 5 and n_oe >= 5:
        _, pval_mwu = stats.mannwhitneyu(oe_c, ctrl_c, alternative='two-sided')
        pooled_std  = np.sqrt((np.std(ctrl_c)**2 + np.std(oe_c)**2) / 2)
        cohens_d    = safe_div(oe_mean_c - ctrl_mean_c, pooled_std)
    else:
        pval_mwu = cohens_d = np.nan

    return {
        'TF': tf, 'scope': scope_label,
        'n_ctrl': n_ctrl, 'n_oe': n_oe,
        # counts space
        'ctrl_mean': ctrl_mean_c, 'oe_mean': oe_mean_c,
        'ctrl_median': np.median(ctrl_c), 'oe_median': np.median(oe_c),
        'ctrl_std': np.std(ctrl_c), 'oe_std': np.std(oe_c),
        'ctrl_cv': safe_div(np.std(ctrl_c), ctrl_mean_c),
        'ctrl_var': np.var(ctrl_c),
        'log2fc_counts': log2fc_counts,
        'fc_counts': fc_counts,
        # lognorm space
        'ctrl_mean_log': ctrl_mean_l, 'oe_mean_log': oe_mean_l,
        'ctrl_var_log': np.var(ctrl_l), 'oe_var_log': np.var(oe_l),
        'log2fc_lognorm': log2fc_lognorm,   # CORRECT log2FC from log1p data
        # detection
        'ctrl_detection': ctrl_det, 'oe_detection': oe_det,
        # stats
        'pval_mwu': pval_mwu, 'cohens_d': cohens_d,
    }


def run_analysis(adata, X_counts, X_lognorm, var_names, tfs, cell_lines):
    print("\n" + "="*70)
    print("ANALYSING TF SELF-EXPRESSION")
    print("="*70)

    records = []
    missing = []

    for i, tf in enumerate(tfs, 1):
        if i % 10 == 0:
            print(f"  [{i}/{len(tfs)}] {tf}")
        if tf not in var_names:
            missing.append(tf)
            continue
        gene_idx = var_names.index(tf)

        # pooled
        records.append(analyze_single_tf(X_counts, X_lognorm, gene_idx, tf,
                                          adata.obs, 'pooled', None))
        # per cell line
        for cl in cell_lines:
            mask = adata.obs['cell_line'] == cl
            records.append(analyze_single_tf(X_counts, X_lognorm, gene_idx, tf,
                                              adata.obs, cl, mask))

    if missing:
        print(f"\n  ⚠  {len(missing)} TF(s) not in var_names: {missing}")

    df = pd.DataFrame(records)

    # ── BH FDR correction (per scope) ────────────────────────────────
    for scope in df['scope'].unique():
        mask_scope = df['scope'] == scope
        pvals = df.loc[mask_scope, 'pval_mwu'].values
        valid = np.isfinite(pvals)
        padj  = np.full(len(pvals), np.nan)
        if valid.sum() > 0:
            _, padj_valid, _, _ = multipletests(pvals[valid], method='fdr_bh')
            padj[valid] = padj_valid
        df.loc[mask_scope, 'padj_mwu_bh'] = padj

    df.to_csv(Config.OUT_DIR / 'tf_expression_results.csv', index=False)
    print(f"\n  ✓ Results saved to: {Config.OUT_DIR / 'tf_expression_results.csv'}")
    return df


# =====================================================================
# PLOTTING
# =====================================================================

def plot_summary_heatmap(df, cell_lines):
    """
    Heatmap: TFs (rows) × cell lines (cols), colour = log2FC.
    Dot size encodes −log10(padj) so significance is visible.
    Nature-standard combined dot/colour heatmap.
    """
    print("\n  Plotting: summary heatmap …")

    df_cl = df[df['scope'].isin(cell_lines)].copy()

    # pivot
    fc_wide   = df_cl.pivot(index='TF', columns='scope', values='log2fc_lognorm')
    padj_wide = df_cl.pivot(index='TF', columns='scope', values='padj_mwu_bh')

    # order by pooled FC
    df_pool   = df[df['scope'] == 'pooled'].set_index('TF')
    tf_order  = df_pool['log2fc_lognorm'].dropna().sort_values(ascending=False).index
    tf_order  = [t for t in tf_order if t in fc_wide.index]

    fc_wide   = fc_wide.loc[tf_order]
    padj_wide = padj_wide.loc[tf_order]
    sig_wide  = (-np.log10(padj_wide + 1e-300)).fillna(0).clip(upper=6)

    n_tf = len(tf_order)
    fig_h = max(8, n_tf * 0.28)
    fig, ax = plt.subplots(figsize=(5 + len(cell_lines) * 1.5, fig_h))

    # colour mesh
    vmax = np.nanpercentile(np.abs(fc_wide.values), 95)
    vmax = max(vmax, 0.5)
    im   = ax.imshow(fc_wide.values, aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)

    # dot overlay (size ∝ significance)
    nrow, ncol = fc_wide.shape
    for r in range(nrow):
        for c in range(ncol):
            s = sig_wide.iloc[r, c]
            if np.isfinite(s) and s > 0:
                ax.scatter(c, r, s=s * 12, color='black', alpha=0.6, zorder=3)

    # axes labels
    ax.set_xticks(range(len(cell_lines)))
    ax.set_xticklabels(cell_lines, fontsize=11)
    ax.set_yticks(range(n_tf))
    ax.set_yticklabels(tf_order, fontsize=7)
    ax.set_xlabel('Cell line', fontsize=12)
    ax.set_title('TF self-expression: log2FC (OE vs Ctrl)\n'
                 'Dot size = −log10(padj), colour = log2FC',
                 fontsize=13, fontweight='bold')

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('log2FC', rotation=270, labelpad=15, fontsize=11)

    # legend for dot sizes
    for sig_val, label in [(2, 'p<0.01'), (4, 'p<0.0001')]:
        ax.scatter([], [], s=sig_val * 12, color='black', alpha=0.6,
                   label=label)
    ax.legend(title='padj', loc='lower right', fontsize=9, title_fontsize=9)

    plt.tight_layout()
    fig.savefig(Config.OUT_DIR / 'fig_summary_heatmap.pdf', bbox_inches='tight')
    fig.savefig(Config.OUT_DIR / 'fig_summary_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: fig_summary_heatmap.pdf/png")


def plot_volcano(df):
    """Volcano plot (log2FC vs −log10(padj)) for pooled analysis."""
    print("  Plotting: volcano …")

    df_p = df[df['scope'] == 'pooled'].dropna(subset=['log2fc_lognorm', 'padj_mwu_bh']).copy()
    df_p['neg_log10_padj'] = -np.log10(df_p['padj_mwu_bh'].clip(lower=1e-300))

    sig = (df_p['padj_mwu_bh'] < 0.05) & (df_p['log2fc_lognorm'].abs() > np.log2(1.5))
    colors = np.where(sig, '#E64B35', '#AAAAAA')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_p['log2fc_lognorm'], df_p['neg_log10_padj'],
               c=colors, s=40, alpha=0.8, edgecolors='none')

    # label top hits
    top = df_p.nlargest(15, 'neg_log10_padj')
    for _, row in top.iterrows():
        ax.annotate(row['TF'],
                    (row['log2fc_lognorm'], row['neg_log10_padj']),
                    textcoords='offset points', xytext=(4, 2),
                    fontsize=7, ha='left', va='bottom')

    ax.axhline(-np.log10(0.05), color='grey', linestyle='--', lw=1.2, label='padj=0.05')
    ax.axvline(np.log2(1.5),    color='grey', linestyle=':',  lw=1.2, label='FC=1.5×')
    ax.axvline(-np.log2(1.5),   color='grey', linestyle=':',  lw=1.2)
    ax.set_xlabel('log2FC (OE vs Ctrl, pooled)', fontsize=12)
    ax.set_ylabel('−log10(padj, BH)', fontsize=12)
    ax.set_title('TF self-expression: significance vs effect size\n'
                 'Red = padj<0.05 & |log2FC|>0.58 (FC>1.5×)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # summary annotation
    n_up   = (sig & (df_p['log2fc_lognorm'] > 0)).sum()
    n_down = (sig & (df_p['log2fc_lognorm'] < 0)).sum()
    ax.text(0.02, 0.97, f'Up: {n_up}\nDown: {n_down}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(Config.OUT_DIR / 'fig_volcano_pooled.pdf', bbox_inches='tight')
    fig.savefig(Config.OUT_DIR / 'fig_volcano_pooled.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: fig_volcano_pooled.pdf/png")


def plot_top_tf_violins(adata, var_names, df, n_top=20):
    """
    Publication-quality violin plots for the top n_top TFs by log2FC.
    Shows actual single-cell expression distributions (OE vs Ctrl), per cell line.
    This is the key 'did it work' figure for Nature.
    """
    print(f"  Plotting: top-{n_top} TF violin panels …")

    df_pool   = df[df['scope'] == 'pooled'].dropna(subset=['log2fc_lognorm'])
    top_tfs   = df_pool.nlargest(n_top, 'log2fc_lognorm')['TF'].tolist()
    top_tfs   = [t for t in top_tfs if t in var_names]

    cell_lines = sorted(adata.obs['cell_line'].unique())
    ncols = 4
    nrows = int(np.ceil(len(top_tfs) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             squeeze=False)
    fig.suptitle(f'Top {n_top} TFs by log2FC: expression in OE vs Ctrl (per cell line)',
                 fontsize=14, fontweight='bold', y=1.01)

    for i, tf in enumerate(top_tfs):
        ax = axes[i // ncols][i % ncols]
        gene_idx = var_names.index(tf)

        # collect per-cell-line expression
        plot_data = []
        plot_labels = []
        plot_colors = []

        for cl in cell_lines:
            ctrl_mask = (adata.obs['cell_line'] == cl) & (adata.obs['guide_assignment'] == 'Ctrl')
            oe_mask   = (adata.obs['cell_line'] == cl) & (adata.obs['guide_assignment'] == tf)

            X = adata.X
            if hasattr(X, 'toarray'):
                ctrl_expr = X[ctrl_mask.values].toarray()[:, gene_idx].ravel()
                oe_expr   = X[oe_mask.values].toarray()[:, gene_idx].ravel()
            else:
                ctrl_expr = np.array(X[ctrl_mask.values])[:, gene_idx].ravel()
                oe_expr   = np.array(X[oe_mask.values])[:, gene_idx].ravel()

            if len(ctrl_expr) > 3:
                plot_data.append(ctrl_expr)
                plot_labels.append(f'{cl}\nCtrl')
                plot_colors.append((*matplotlib.colors.to_rgb(CELL_LINE_COLORS[cl]), 0.4))

            if len(oe_expr) > 3:
                plot_data.append(oe_expr)
                plot_labels.append(f'{cl}\nOE')
                plot_colors.append((*matplotlib.colors.to_rgb(CELL_LINE_COLORS[cl]), 1.0))

        if not plot_data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(plot_data, showmedians=True, showmeans=False, widths=0.8)
        for j, (body, color) in enumerate(zip(parts['bodies'], plot_colors)):
            body.set_facecolor(color[:3])
            body.set_alpha(color[3])
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)

        # fetch pooled stats for annotation
        row_pool = df[(df['TF'] == tf) & (df['scope'] == 'pooled')]
        if len(row_pool):
            lfc  = row_pool.iloc[0]['log2fc_lognorm']
            padj = row_pool.iloc[0]['padj_mwu_bh']
            sig_label = '***' if padj < 0.001 else ('**' if padj < 0.01 else ('*' if padj < 0.05 else 'ns'))
            ax.set_title(f'{tf}\nlog2FC={lfc:.2f}, {sig_label}', fontsize=9)
        else:
            ax.set_title(tf, fontsize=9)

        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('log1p(expr)', fontsize=8)

    # hide unused axes
    for j in range(len(top_tfs), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(Config.OUT_DIR / f'fig_top{n_top}_tf_violins.pdf', bbox_inches='tight')
    fig.savefig(Config.OUT_DIR / f'fig_top{n_top}_tf_violins.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: fig_top{n_top}_tf_violins.pdf/png")


def plot_expression_scatter(df):
    """Scatter: ctrl mean vs OE mean, all TFs, all cell lines."""
    print("  Plotting: expression scatter …")

    df_pool = df[df['scope'] == 'pooled'].dropna(subset=['ctrl_mean', 'oe_mean'])
    cell_lines = sorted(df[df['scope'] != 'pooled']['scope'].unique())

    fig, axes = plt.subplots(1, len(cell_lines) + 1,
                             figsize=((len(cell_lines) + 1) * 4.5, 4.5))

    for ax, scope in zip(axes, ['pooled'] + cell_lines):
        sub = df[df['scope'] == scope].dropna(subset=['ctrl_mean', 'oe_mean'])
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        color = CELL_LINE_COLORS.get(scope, '#555555')

        sig = (sub['padj_mwu_bh'] < 0.05) & (sub['log2fc_lognorm'].abs() > np.log2(1.5))
        ax.scatter(sub.loc[~sig, 'ctrl_mean'], sub.loc[~sig, 'oe_mean'],
                   s=25, alpha=0.6, color='#AAAAAA', edgecolors='none', label='ns')
        ax.scatter(sub.loc[sig,  'ctrl_mean'], sub.loc[sig,  'oe_mean'],
                   s=35, alpha=0.9, color=color, edgecolors='black', lw=0.5,
                   label='sig (padj<0.05)')

        # identity and 1.5× lines
        m = max(sub['ctrl_mean'].max(), sub['oe_mean'].max())
        ax.plot([0, m], [0, m],     'k--',    lw=1.2, alpha=0.5)
        ax.plot([0, m], [0, 1.5*m], 'r--',    lw=1.0, alpha=0.5, label='1.5×')
        ax.plot([0, m], [0, 2*m],   'orange', lw=1.0, alpha=0.5, linestyle=':', label='2×')

        ax.set_xlabel('Ctrl mean (norm. counts)', fontsize=10)
        ax.set_ylabel('OE mean (norm. counts)',   fontsize=10)
        ax.set_title(scope, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('TF self-expression: OE vs Ctrl mean expression',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(Config.OUT_DIR / 'fig_expression_scatter.pdf', bbox_inches='tight')
    fig.savefig(Config.OUT_DIR / 'fig_expression_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: fig_expression_scatter.pdf/png")


def plot_cell_line_concordance(df, cell_lines):
    """
    Pairwise scatter of log2FC across cell lines (reproducibility check).
    Key for Nature: shows effect is consistent across genetic backgrounds.
    """
    print("  Plotting: cell line concordance …")

    from itertools import combinations
    pairs = list(combinations(cell_lines, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (cl1, cl2) in zip(axes, pairs):
        d1 = df[df['scope'] == cl1].set_index('TF')['log2fc_lognorm']
        d2 = df[df['scope'] == cl2].set_index('TF')['log2fc_lognorm']
        merged = pd.concat([d1.rename('x'), d2.rename('y')], axis=1).dropna()

        ax.scatter(merged['x'], merged['y'], s=40, alpha=0.7,
                   edgecolors='black', lw=0.5, color='#2E86AB')

        # label top TFs
        pool = df[df['scope'] == 'pooled'].set_index('TF')['log2fc_lognorm']
        for tf in pool.nlargest(8).index:
            if tf in merged.index:
                ax.annotate(tf, (merged.loc[tf, 'x'], merged.loc[tf, 'y']),
                            fontsize=7, xytext=(4, 2),
                            textcoords='offset points')

        lim = max(merged.abs().max().max() * 1.1, 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color='grey', lw=0.8, linestyle=':')
        ax.axvline(0, color='grey', lw=0.8, linestyle=':')
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.2, alpha=0.5)

        r, p = stats.pearsonr(merged['x'], merged['y'])
        ax.set_xlabel(f'{cl1} log2FC', fontsize=11)
        ax.set_ylabel(f'{cl2} log2FC', fontsize=11)
        ax.set_title(f'{cl1} vs {cl2}\nr = {r:.3f}, p = {p:.2e}',
                     fontsize=11, fontweight='bold')

    plt.suptitle('log2FC concordance across cell lines\n'
                 '(each dot = one TF; r = Pearson)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(Config.OUT_DIR / 'fig_cell_line_concordance.pdf', bbox_inches='tight')
    fig.savefig(Config.OUT_DIR / 'fig_cell_line_concordance.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: fig_cell_line_concordance.pdf/png")


def print_summary(df):
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    df_p = df[df['scope'] == 'pooled'].dropna(subset=['log2fc_lognorm', 'padj_mwu_bh'])
    print(f"  TFs analysed              : {len(df_p)}")
    print(f"  Median log2FC             : {df_p['log2fc_lognorm'].median():.3f}")
    print(f"  TFs with log2FC > 0.58    : {(df_p['log2fc_lognorm'] > np.log2(1.5)).sum()} "
          f"({100*(df_p['log2fc_lognorm'] > np.log2(1.5)).mean():.1f}%)")
    print(f"  TFs with padj < 0.05      : {(df_p['padj_mwu_bh'] < 0.05).sum()} "
          f"({100*(df_p['padj_mwu_bh'] < 0.05).mean():.1f}%)")
    print(f"  TFs sig & FC > 1.5×       : "
          f"{((df_p['padj_mwu_bh'] < 0.05) & (df_p['log2fc_lognorm'] > np.log2(1.5))).sum()}")
    print("\n  Top 10 TFs by log2FC:")
    top = df_p.nlargest(10, 'log2fc_lognorm')[['TF', 'log2fc_lognorm', 'fc_counts', 'padj_mwu_bh', 'cohens_d']]
    top.columns = ['TF', 'log2FC', 'FC (counts)', 'padj (BH)', "Cohen's d"]
    print(top.to_string(index=False))


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*70)
    print("TF SELF-EXPRESSION ANALYSIS v2")
    print("="*70)

    adata, X_counts, X_lognorm, var_names, tfs, cell_lines, data_type = load_and_prepare()
    df = run_analysis(adata, X_counts, X_lognorm, var_names, tfs, cell_lines)

    print("\n" + "="*70)
    print("CREATING FIGURES")
    print("="*70)
    plot_summary_heatmap(df, cell_lines)
    plot_volcano(df)
    plot_top_tf_violins(adata, var_names, df, n_top=20)
    plot_expression_scatter(df)
    plot_cell_line_concordance(df, cell_lines)

    print_summary(df)

    print(f"\n  ✓ All outputs saved to: {Config.OUT_DIR}")


if __name__ == '__main__':
    main()
