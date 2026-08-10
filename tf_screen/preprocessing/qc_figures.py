"""Quality-control figures for the TF screen.

Produces the panels shown in Extended Data Fig. 5a-d:

* Fig 1 -- per-cell RNA QC metrics (all cells, pre-filter overview).
* Fig 2 -- guide-assignment outcome distribution.
* Fig 3 -- RNA QC metrics restricted to guide-assigned cells.
* Fig 4 -- QC stratified by seeding density.
* Fig 5 -- QC stratified by replicate / Ligation / RT batch variables.
* Fig 6 -- TF representation across cell lines.
* Fig 7 -- library / technical comparison.
* Fig 8 -- guide-call quality: UMI threshold sensitivity, dominance ratio.

All input filtering is upstream; this script does not filter, it only
visualises and reports summary statistics.

Inputs
------
``SCREEN_ANNDATA`` (raw-count screen object with guide assignments).

Outputs
-------
``OUTPUT_DIR/qc_figures/`` -- PDFs + PNGs of all panels, plus a summary
CSV table.

Usage
-----
``python preprocessing/qc_figures.py``
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

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and plotting style
# ---------------------------------------------------------------------------

ADATA_PATH = config.SCREEN_ANNDATA
FIG_DIR = config.OUTPUT_DIR / "qc_figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Nature-compatible style.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,  # TrueType fonts in PDF (required by Nature)
    "ps.fonttype": 42,
})

CELL_LINE_COLORS = {"P3": "#E64B35", "S24": "#4DBBD5", "BG5": "#00A087"}
ASSIGNMENT_COLORS = {
    "Assigned":               "#2E86AB",
    "Unassigned_NoGuide":     "#A23B72",
    "Unassigned_LowUMI":      "#F18F01",
    "Unassigned_Multiplet":   "#C73E1D",
    "No_CRISPR_Data":         "#95A3A4",
}

PALETTE_DENSITY   = sns.color_palette("rocket",  8)
PALETTE_REPLICATE = sns.color_palette("muted",   8)
PALETTE_LIGATION  = sns.color_palette("Set2",    8)
PALETTE_RT        = sns.color_palette("tab10",  10)


# =============================================================================
# HELPERS
# =============================================================================

def save_fig(fig, stem, fig_dir=FIG_DIR):
    fig.savefig(fig_dir / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(fig_dir / f'{stem}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {stem}.pdf/png")


def violin_by_group(ax, adata_sub, col, group_col, palette, ylabel,
                    title, order=None, showfliers=False, yscale=None):
    """Seaborn violin with individual group colours."""
    if order is None:
        order = sorted(adata_sub.obs[group_col].unique())
    df = adata_sub.obs[[col, group_col]].copy()
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(order)}
    sns.violinplot(data=df, x=group_col, y=col, order=order,
                   palette=color_map, ax=ax, inner='quartile',
                   linewidth=0.8, cut=0)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    if yscale:
        ax.set_yscale(yscale)


def gini(arr):
    """Gini coefficient of an array (0 = perfect equality, 1 = maximum inequality)."""
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return np.nan
    cumulative = np.cumsum(arr)
    return (2 * np.sum((np.arange(1, n + 1)) * arr) / (n * cumulative[-1])) - (n + 1) / n


# =============================================================================
# DATA TYPE CHECK
# =============================================================================

def check_data_type(adata, n_sample=500):
    """
    Inspect adata.X and return one of: 'raw_counts', 'normalised', 'lognorm'

    Heuristics:
      1. If values are all non-negative integers and max > 100  → raw counts
      2. If max < 20 and values are floats                     → log-normalised (log1p)
      3. Otherwise                                              → normalised (not log)

    This is printed as a clear diagnostic for the record.
    """
    print("\n" + "="*60)
    print("DATA TYPE CHECK")
    print("="*60)

    X = adata.X
    if hasattr(X, 'toarray'):
        sample_idx = np.random.choice(adata.n_obs, min(n_sample, adata.n_obs), replace=False)
        sample = X[sample_idx].toarray()
    else:
        sample_idx = np.random.choice(adata.n_obs, min(n_sample, adata.n_obs), replace=False)
        sample = np.array(X[sample_idx])

    max_val   = float(sample.max())
    min_val   = float(sample.min())
    frac_int  = np.mean(np.abs(sample - np.round(sample)) < 1e-6)
    mean_val  = float(sample.mean())
    sparsity  = float((sample == 0).mean())

    print(f"  Shape           : {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  Sample max      : {max_val:.3f}")
    print(f"  Sample min      : {min_val:.3f}")
    print(f"  Sample mean     : {mean_val:.4f}")
    print(f"  Fraction integer: {frac_int:.3f}")
    print(f"  Sparsity (=0)   : {sparsity:.3f}")

    if min_val < 0:
        dtype = 'scaled'
        print("\n  ⚠  SCALED / CENTRED DATA detected (negative values).")
        print("     Downstream QC metrics should use a raw-count layer.")
    elif frac_int > 0.98 and max_val > 100:
        dtype = 'raw_counts'
        print("\n  ✓  RAW COUNT data detected.")
        print("     For QC plots: using adata.X directly.")
        print("     For TF expression: normalise with sc.pp.normalize_total + sc.pp.log1p.")
    elif max_val < 20 and frac_int < 0.5:
        dtype = 'lognorm'
        print("\n  ✓  LOG-NORMALISED data detected (log1p, base-e).")
        print("     For fold-change: use mean(OE) − mean(Ctrl) in log space,")
        print("     then divide by ln(2) to convert to log2-FC.")
        print("     Raw counts recoverable via np.expm1(X).")
    elif max_val < 1000 and frac_int < 0.5:
        dtype = 'normalised'
        print("\n  ✓  NORMALISED (not log) data detected.")
        print("     Apply np.log1p before fold-change calculations.")
    else:
        dtype = 'unknown'
        print("\n  ?  Data type ambiguous — inspect manually.")

    print("="*60)
    return dtype


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading AnnData...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# --- data type ---
data_type = check_data_type(adata)

# --- assignment categories ---
def categorize_assignment(row):
    if not row['guide_has_assignment']:
        # further split by reason
        ga = str(row['guide_assignment'])
        if 'NoGuide' in ga:    return 'Unassigned_NoGuide'
        if 'LowUMI' in ga:     return 'Unassigned_LowUMI'
        if 'Multiplet' in ga:  return 'Unassigned_Multiplet'
        return 'No_CRISPR_Data'
    return 'Assigned'

adata.obs['assignment_category'] = adata.obs.apply(categorize_assignment, axis=1)

# guide-assigned subset — used for most QC plots
adata_assigned = adata[adata.obs['guide_has_assignment'] == True].copy()
print(f"\nGuide-assigned cells: {adata_assigned.n_obs:,} / {adata.n_obs:,} "
      f"({100*adata_assigned.n_obs/adata.n_obs:.1f}%)")

# convenience
CELL_LINES = sorted(adata.obs['cell_line'].unique())
SAMPLES    = sorted(adata.obs['sample'].unique())
SAMPLE_COLORS = [CELL_LINE_COLORS[s.split('-')[0]] for s in SAMPLES]


# =============================================================================
# FIGURE 1 — RNA QC METRICS: ALL CELLS
# (reviewers need to see the full, unfiltered QC to judge data quality)
# =============================================================================
print("\n[Fig 1] RNA QC — all cells")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('RNA QC metrics — all cells (pre-assignment filter)', y=1.01,
             fontsize=14, fontweight='bold')

# 1a. UMIs violin by sample
violin_by_group(axes[0, 0], adata, 'total_counts', 'sample',
                SAMPLE_COLORS, 'UMIs per cell',
                'A. UMIs per cell (by sample)', order=SAMPLES)

# 1b. Genes violin by sample
violin_by_group(axes[0, 1], adata, 'n_genes', 'sample',
                SAMPLE_COLORS, 'Genes per cell',
                'B. Genes per cell (by sample)', order=SAMPLES)

# 1c. Mito fraction violin by sample
adata.obs['mito_pct'] = adata.obs['mito_fraction'] * 100
violin_by_group(axes[0, 2], adata, 'mito_pct', 'sample',
                SAMPLE_COLORS, 'Mitochondrial (%)',
                'C. Mito fraction (by sample)', order=SAMPLES)
axes[0, 2].axhline(y=5,  color='red',    linestyle='--', alpha=0.6, lw=1.5, label='5%')
axes[0, 2].axhline(y=20, color='orange', linestyle='--', alpha=0.6, lw=1.5, label='20%')
axes[0, 2].legend(fontsize=9)

# 1d. UMIs by cell line (density)
violin_by_group(axes[1, 0], adata, 'total_counts', 'cell_line',
                [CELL_LINE_COLORS[cl] for cl in CELL_LINES],
                'UMIs per cell', 'D. UMIs by cell line',
                order=CELL_LINES)

# 1e. Genes vs UMIs scatter (coloured by mito)
ax = axes[1, 1]
sc_idx = np.random.choice(adata.n_obs, min(5000, adata.n_obs), replace=False)
sc_ = ax.scatter(adata.obs['total_counts'].iloc[sc_idx],
                 adata.obs['n_genes'].iloc[sc_idx],
                 c=adata.obs['mito_pct'].iloc[sc_idx],
                 cmap='viridis', s=2, alpha=0.5, vmin=0, vmax=20)
plt.colorbar(sc_, ax=ax, label='Mito (%)')
ax.set_xlabel('UMIs per cell')
ax.set_ylabel('Genes per cell')
ax.set_title('E. Genes vs UMIs (colour = mito%)')

# 1f. Summary table by cell line
ax = axes[1, 2]
ax.axis('off')
summary = adata.obs.groupby('cell_line').agg(
    Cells=('total_counts', 'count'),
    UMI_median=('total_counts', 'median'),
    Genes_median=('n_genes', 'median'),
    Mito_median_pct=('mito_pct', 'median'),
    Saturation_median=('saturation', 'median'),
).round(1)
summary.columns = ['Cells', 'UMI (med)', 'Genes (med)', 'Mito% (med)', 'Sat (med)']
tbl = ax.table(cellText=summary.values, colLabels=summary.columns,
               rowLabels=summary.index, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.6)
ax.set_title('F. Summary by cell line', fontsize=12, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'fig1_rna_qc_all_cells')


# =============================================================================
# FIGURE 2 — GUIDE ASSIGNMENT OVERVIEW  (all cells)
# =============================================================================
print("[Fig 2] Guide assignment overview")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Guide assignment overview — all cells', y=1.01,
             fontsize=14, fontweight='bold')

# 2a. Pie
ax = axes[0, 0]
ac = adata.obs['assignment_category'].value_counts()
colors = [ASSIGNMENT_COLORS.get(k, '#888') for k in ac.index]
wedges, texts, autotexts = ax.pie(ac.values, labels=ac.index, autopct='%1.1f%%',
                                   colors=colors, startangle=90,
                                   textprops={'fontsize': 9})
ax.set_title('A. Overall assignment')

# 2b. Assignment rate by sample
ax = axes[0, 1]
asgn_rate = adata.obs.groupby('sample')['guide_has_assignment'].mean() * 100
asgn_rate = asgn_rate.reindex(SAMPLES)
bars = ax.bar(range(len(SAMPLES)), asgn_rate.values, color=SAMPLE_COLORS, alpha=0.85, edgecolor='white')
ax.axhline(asgn_rate.mean(), color='black', linestyle='--', lw=1.5,
           label=f'Mean: {asgn_rate.mean():.1f}%')
ax.set_xticks(range(len(SAMPLES)))
ax.set_xticklabels(SAMPLES, rotation=45, ha='right')
ax.set_ylabel('Assignment rate (%)')
ax.set_title('B. Assignment rate by sample')
ax.legend(fontsize=9)
ax.set_ylim(0, 100)

# 2c. Stacked bar: assignment categories by sample
ax = axes[0, 2]
cats = ['Assigned', 'Unassigned_NoGuide', 'Unassigned_LowUMI', 'Unassigned_Multiplet', 'No_CRISPR_Data']
pct = (adata.obs.groupby('sample')['assignment_category']
       .value_counts(normalize=True).unstack(fill_value=0) * 100)
pct = pct.reindex(SAMPLES)
bottom = np.zeros(len(SAMPLES))
for cat in cats:
    if cat in pct.columns:
        vals = pct[cat].values
        ax.bar(range(len(SAMPLES)), vals, bottom=bottom,
               label=cat, color=ASSIGNMENT_COLORS.get(cat, '#888'), alpha=0.9)
        bottom += vals
ax.set_xticks(range(len(SAMPLES)))
ax.set_xticklabels(SAMPLES, rotation=45, ha='right')
ax.set_ylabel('% cells')
ax.set_title('C. Assignment breakdown by sample')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0, 100)

# 2d. Guide UMI distribution (assigned)
ax = axes[1, 0]
for cl in CELL_LINES:
    sub = adata_assigned.obs[adata_assigned.obs['cell_line'] == cl]
    ax.hist(sub['guide_total_umis'].clip(upper=150), bins=60,
            alpha=0.55, label=cl, density=True, color=CELL_LINE_COLORS[cl])
ax.axvline(adata_assigned.obs['guide_total_umis'].median(), color='black',
           linestyle='--', lw=1.5,
           label=f"Median: {adata_assigned.obs['guide_total_umis'].median():.0f}")
ax.set_xlabel('Guide UMIs per cell')
ax.set_ylabel('Density')
ax.set_title('D. Guide UMI distribution (assigned)')
ax.legend(fontsize=9)

# 2e. Guide dominance ratio
ax = axes[1, 1]
for cl in CELL_LINES:
    sub = adata_assigned.obs[adata_assigned.obs['cell_line'] == cl]
    ax.hist(sub['guide_dominance_ratio'].dropna(), bins=50,
            alpha=0.55, label=cl, density=True, color=CELL_LINE_COLORS[cl])
ax.axvline(0.5,  color='red',    linestyle='--', lw=1.5, label='0.5 threshold')
ax.axvline(0.75, color='orange', linestyle='--', lw=1.5, label='0.75 threshold')
ax.set_xlabel('Guide dominance ratio')
ax.set_ylabel('Density')
ax.set_title('E. Guide dominance ratio (assigned)')
ax.legend(fontsize=9)

# 2f. n_guides detected per cell
ax = axes[1, 2]
max_guides = min(int(adata.obs['guide_n_detected'].max()), 8)
for cl in CELL_LINES:
    sub = adata.obs[adata.obs['cell_line'] == cl]
    counts = sub['guide_n_detected'].clip(upper=max_guides).value_counts(normalize=True).sort_index() * 100
    ax.plot(counts.index, counts.values, 'o-', label=cl,
            color=CELL_LINE_COLORS[cl], lw=2, ms=6)
ax.set_xlabel('Number of guides detected')
ax.set_ylabel('% of cells')
ax.set_title('F. Guides detected per cell (multiplet check)')
ax.legend(fontsize=9)
ax.set_xticks(range(0, max_guides + 1))

plt.tight_layout()
save_fig(fig, 'fig2_guide_assignment')


# =============================================================================
# FIGURE 3 — RNA QC METRICS: GUIDE-ASSIGNED CELLS ONLY
# =============================================================================
print("[Fig 3] RNA QC — guide-assigned cells only")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle(
    f'RNA QC metrics — guide-assigned cells only '
    f'(n={adata_assigned.n_obs:,}, {100*adata_assigned.n_obs/adata.n_obs:.1f}% of total)',
    y=1.01, fontsize=14, fontweight='bold')

adata_assigned.obs['mito_pct'] = adata_assigned.obs['mito_fraction'] * 100

# 3a. UMIs by cell line — assigned only
violin_by_group(axes[0, 0], adata_assigned, 'total_counts', 'cell_line',
                [CELL_LINE_COLORS[cl] for cl in CELL_LINES],
                'UMIs per cell', 'A. UMIs by cell line (assigned)', order=CELL_LINES)

# 3b. Genes by cell line — assigned only
violin_by_group(axes[0, 1], adata_assigned, 'n_genes', 'cell_line',
                [CELL_LINE_COLORS[cl] for cl in CELL_LINES],
                'Genes per cell', 'B. Genes by cell line (assigned)', order=CELL_LINES)

# 3c. Mito by cell line — assigned only
violin_by_group(axes[0, 2], adata_assigned, 'mito_pct', 'cell_line',
                [CELL_LINE_COLORS[cl] for cl in CELL_LINES],
                'Mitochondrial (%)', 'C. Mito fraction (assigned)', order=CELL_LINES)
axes[0, 2].axhline(5, color='red', linestyle='--', alpha=0.6, lw=1.5)

# 3d. UMIs by sample — assigned only
violin_by_group(axes[1, 0], adata_assigned, 'total_counts', 'sample',
                SAMPLE_COLORS, 'UMIs per cell',
                'D. UMIs by sample (assigned)', order=SAMPLES)

# 3e. Genes vs UMIs, coloured by guide UMIs (assigned)
ax = axes[1, 1]
sc_idx = np.random.choice(adata_assigned.n_obs, min(5000, adata_assigned.n_obs), replace=False)
sc_ = ax.scatter(adata_assigned.obs['total_counts'].iloc[sc_idx],
                 adata_assigned.obs['n_genes'].iloc[sc_idx],
                 c=np.log1p(adata_assigned.obs['guide_total_umis'].iloc[sc_idx]),
                 cmap='plasma', s=2, alpha=0.5)
plt.colorbar(sc_, ax=ax, label='log(guide UMIs + 1)')
ax.set_xlabel('RNA UMIs per cell')
ax.set_ylabel('Genes per cell')
ax.set_title('E. Genes vs UMIs (colour = guide UMIs)')

# 3f. Comparison table: all vs assigned
ax = axes[1, 2]
ax.axis('off')
rows = []
for cl in CELL_LINES:
    all_sub  = adata.obs[adata.obs['cell_line'] == cl]
    asgn_sub = adata_assigned.obs[adata_assigned.obs['cell_line'] == cl]
    rows.append([
        cl,
        f"{len(all_sub):,}",
        f"{len(asgn_sub):,}",
        f"{all_sub['total_counts'].median():.0f}",
        f"{asgn_sub['total_counts'].median():.0f}",
    ])
tbl = ax.table(
    cellText=rows,
    colLabels=['Cell line', 'N (all)', 'N (assigned)',
               'UMI med (all)', 'UMI med (asgn)'],
    cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.7)
ax.set_title('F. All vs assigned cells comparison', fontsize=12, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'fig3_rna_qc_assigned_only')


# =============================================================================
# FIGURE 4 — QC STRATIFIED BY CELL DENSITY
# (pooled seedings at different densities can affect UMI/gene yield)
# =============================================================================
print("[Fig 4] QC by cell density")

DENSITIES = sorted(adata_assigned.obs['cell_density'].unique())
density_colors = {d: PALETTE_DENSITY[i % len(PALETTE_DENSITY)] for i, d in enumerate(DENSITIES)}

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('RNA QC stratified by cell seeding density (assigned cells)',
             y=1.01, fontsize=14, fontweight='bold')

for col_i, (metric, ylabel, title_sfx) in enumerate([
        ('total_counts', 'UMIs per cell', 'A. UMIs'),
        ('n_genes',      'Genes per cell','B. Genes'),
        ('mito_pct',     'Mito (%)',      'C. Mito%'),
]):
    ax = axes[0, col_i]
    violin_by_group(ax, adata_assigned, metric, 'cell_density',
                    [density_colors[d] for d in DENSITIES],
                    ylabel, f'{title_sfx} by density', order=DENSITIES)

# 4d. Assignment rate by density
ax = axes[1, 0]
asgn_by_dens = adata.obs.groupby('cell_density')['guide_has_assignment'].mean() * 100
asgn_by_dens = asgn_by_dens.reindex(DENSITIES)
ax.bar(range(len(DENSITIES)),
       asgn_by_dens.values,
       color=[density_colors[d] for d in DENSITIES], alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(DENSITIES)))
ax.set_xticklabels(DENSITIES, rotation=45, ha='right')
ax.set_ylabel('Assignment rate (%)')
ax.set_title('D. Guide assignment rate by density')
ax.set_ylim(0, 100)

# 4e. Guide UMIs by density
violin_by_group(axes[1, 1], adata_assigned, 'guide_total_umis', 'cell_density',
                [density_colors[d] for d in DENSITIES],
                'Guide UMIs', 'E. Guide UMIs by density', order=DENSITIES)

# 4f. Saturation by density
violin_by_group(axes[1, 2], adata_assigned, 'saturation', 'cell_density',
                [density_colors[d] for d in DENSITIES],
                'Sequencing saturation', 'F. Saturation by density', order=DENSITIES)

plt.tight_layout()
save_fig(fig, 'fig4_qc_by_cell_density')


# =============================================================================
# FIGURE 5 — QC STRATIFIED BY REPLICATE & BATCH VARIABLES
# (replicate, Ligation, RT, i5 — key for reviewers asking about batch effects)
# =============================================================================
print("[Fig 5] QC by replicate / batch variables")

batch_vars = {
    'replicate': (sorted(adata_assigned.obs['replicate'].unique()), PALETTE_REPLICATE),
    'Ligation':  (sorted(adata_assigned.obs['Ligation'].unique()),  PALETTE_LIGATION),
    'RT':        (sorted(adata_assigned.obs['RT'].unique()),         PALETTE_RT),
}

for var_name, (var_order, palette) in batch_vars.items():
    n_groups = len(var_order)
    color_list = [palette[i % len(palette)] for i in range(n_groups)]

    fig, axes = plt.subplots(2, 3, figsize=(max(14, 3 * n_groups), 11))
    fig.suptitle(f'RNA QC stratified by {var_name} (assigned cells)',
                 y=1.01, fontsize=14, fontweight='bold')

    for col_i, (metric, ylabel, letter) in enumerate([
            ('total_counts',      'UMIs per cell',     'A'),
            ('n_genes',           'Genes per cell',     'B'),
            ('mito_pct',          'Mito (%)',           'C'),
    ]):
        violin_by_group(axes[0, col_i], adata_assigned, metric, var_name,
                        color_list, ylabel,
                        f'{letter}. {ylabel} by {var_name}',
                        order=var_order)

    violin_by_group(axes[1, 0], adata_assigned, 'guide_total_umis', var_name,
                    color_list, 'Guide UMIs',
                    f'D. Guide UMIs by {var_name}', order=var_order)

    # Assignment rate
    ax = axes[1, 1]
    asgn = adata.obs.groupby(var_name)['guide_has_assignment'].mean() * 100
    asgn = asgn.reindex(var_order)
    ax.bar(range(len(var_order)), asgn.values,
           color=color_list, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(var_order)))
    ax.set_xticklabels(var_order, rotation=45, ha='right')
    ax.set_ylabel('Assignment rate (%)')
    ax.set_title(f'E. Assignment rate by {var_name}')
    ax.set_ylim(0, 100)

    violin_by_group(axes[1, 2], adata_assigned, 'saturation', var_name,
                    color_list, 'Sequencing saturation',
                    f'F. Saturation by {var_name}', order=var_order)

    plt.tight_layout()
    save_fig(fig, f'fig5_qc_by_{var_name.lower()}')


# =============================================================================
# FIGURE 6 — TF REPRESENTATION (assigned cells)
# =============================================================================
print("[Fig 6] TF representation")

tf_counts = adata_assigned.obs['guide_assignment'].value_counts()
tf_order  = tf_counts.index.tolist()
tf_colors = ['#E64B35' if 'Ctrl' in t else '#2E86AB' for t in tf_order]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('TF representation (guide-assigned cells only)',
             y=1.01, fontsize=14, fontweight='bold')

# 6a. Cells per TF
ax = axes[0, 0]
ax.barh(range(len(tf_order)), tf_counts.values, color=tf_colors, alpha=0.85)
ax.set_yticks(range(len(tf_order)))
ax.set_yticklabels(tf_order, fontsize=7)
ax.set_xlabel('Number of cells')
ax.set_title(f'A. Cells per TF (n={len(tf_order)})')
ax.invert_yaxis()
ax.axvline(tf_counts.median(), color='red', linestyle='--', lw=1.5,
           label=f'Median: {tf_counts.median():.0f}')
ax.legend(fontsize=9)

# 6b. Gini coefficient per cell line (guide uniformity)
ax = axes[0, 1]
gini_vals = {}
for cl in CELL_LINES:
    sub = adata_assigned.obs[adata_assigned.obs['cell_line'] == cl]
    tf_c = sub['guide_assignment'].value_counts()
    gini_vals[cl] = gini(tf_c.values)
ax.bar(gini_vals.keys(), gini_vals.values(),
       color=[CELL_LINE_COLORS[cl] for cl in gini_vals], alpha=0.85, edgecolor='white')
ax.set_ylabel('Gini coefficient\n(0 = uniform, 1 = concentrated)')
ax.set_title('B. Guide library uniformity (Gini)')
ax.set_ylim(0, 1)
for i, (cl, g) in enumerate(gini_vals.items()):
    ax.text(i, g + 0.01, f'{g:.3f}', ha='center', va='bottom', fontsize=10)

# 6c. TF cells across replicates (mean ± std), top 25
ax = axes[1, 0]
top25 = tf_counts.head(25).index.tolist()
tf_by_rep = (adata_assigned.obs[adata_assigned.obs['guide_assignment'].isin(top25)]
             .groupby(['sample', 'guide_assignment']).size().unstack(fill_value=0))
means = tf_by_rep[top25].mean()
stds  = tf_by_rep[top25].std()
x = range(len(top25))
ax.bar(x, means.values, yerr=stds.values, capsize=3,
       color='#2E86AB', alpha=0.8, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(top25, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Cells per sample (mean ± std)')
ax.set_title('C. Top 25 TFs: cells across samples')

# 6d. Median guide UMIs per TF
ax = axes[1, 1]
umi_by_tf = (adata_assigned.obs.groupby('guide_assignment')['guide_total_umis']
             .median().sort_values(ascending=False))
colors_umi = ['#E64B35' if 'Ctrl' in t else '#00A087' for t in umi_by_tf.index]
ax.barh(range(len(umi_by_tf)), umi_by_tf.values, color=colors_umi, alpha=0.85)
ax.set_yticks(range(len(umi_by_tf)))
ax.set_yticklabels(umi_by_tf.index, fontsize=7)
ax.set_xlabel('Median guide UMIs')
ax.set_title('D. Median guide UMIs per TF')
ax.invert_yaxis()
ax.axvline(umi_by_tf.median(), color='red', linestyle='--', lw=1.5)

plt.tight_layout()
save_fig(fig, 'fig6_tf_representation')


# =============================================================================
# FIGURE 7 — GUIDE QUALITY: THRESHOLD SENSITIVITY
# (key for Nature: show results are robust to threshold choice)
# =============================================================================
print("[Fig 7] Guide quality / threshold sensitivity")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Guide quality metrics & threshold sensitivity', y=1.01,
             fontsize=14, fontweight='bold')

# 7a. Assignment rate vs guide UMI threshold
ax = axes[0, 0]
thresholds = list(range(1, 31))
rates_all = []
rates_singlet = []
for thr in thresholds:
    total = adata.n_obs
    above = (adata.obs['guide_max_umis'] >= thr).sum()
    above_sing = ((adata.obs['guide_max_umis'] >= thr) & adata.obs['guide_is_singlet']).sum()
    rates_all.append(100 * above / total)
    rates_singlet.append(100 * above_sing / total)
ax.plot(thresholds, rates_all,     'o-', color='#2E86AB', lw=2, ms=5, label='≥ threshold (any)')
ax.plot(thresholds, rates_singlet, 's-', color='#E64B35', lw=2, ms=5, label='singlet only')
ax.axvline(adata_assigned.obs['guide_max_umis'].median(), color='grey',
           linestyle=':', lw=1.5, label='median guide UMI')
ax.set_xlabel('Guide UMI threshold')
ax.set_ylabel('% cells assigned')
ax.set_title('A. Assignment rate vs UMI threshold')
ax.legend(fontsize=9)

# 7b. Dominance ratio vs guide UMI
ax = axes[0, 1]
sc_idx = np.random.choice(adata_assigned.n_obs, min(5000, adata_assigned.n_obs), replace=False)
ax.scatter(adata_assigned.obs['guide_total_umis'].iloc[sc_idx].clip(upper=200),
           adata_assigned.obs['guide_dominance_ratio'].iloc[sc_idx],
           s=2, alpha=0.3, color='#4DBBD5')
ax.axhline(0.5,  color='red',    linestyle='--', lw=1.5, label='0.5')
ax.axhline(0.75, color='orange', linestyle='--', lw=1.5, label='0.75')
ax.set_xlabel('Guide total UMIs')
ax.set_ylabel('Guide dominance ratio')
ax.set_title('B. Guide UMIs vs dominance ratio')
ax.legend(fontsize=9)

# 7c. Singlet vs multiplet UMI distributions
ax = axes[0, 2]
has_crispr = adata.obs[adata.obs['assignment_category'] != 'No_CRISPR_Data']
for cat, color in [('guide_is_singlet', '#2E86AB'), ('guide_is_multiplet', '#C73E1D')]:
    sub = has_crispr[has_crispr[cat] == True]
    if len(sub) > 0:
        ax.hist(sub['guide_total_umis'].clip(upper=150), bins=60, alpha=0.6,
                label=cat.replace('guide_is_', ''), density=True, color=color)
ax.set_xlabel('Guide UMIs per cell')
ax.set_ylabel('Density')
ax.set_title('C. Singlet vs multiplet guide UMIs')
ax.legend(fontsize=9)

# 7d. Sequencing saturation by sample
ax = axes[1, 0]
violin_by_group(ax, adata_assigned, 'saturation', 'sample',
                SAMPLE_COLORS, 'Saturation',
                'D. Sequencing saturation by sample', order=SAMPLES)
ax.axhline(0.5, color='red', linestyle='--', lw=1.5, alpha=0.7, label='50%')
ax.legend(fontsize=9)

# 7e. RNA QC: assigned vs unassigned (same cells, side by side)
ax = axes[1, 1]
unassigned = adata[adata.obs['guide_has_assignment'] == False]
data_compare = {
    f'Assigned\n(n={adata_assigned.n_obs:,})': adata_assigned.obs['total_counts'].values,
    f'Unassigned\n(n={len(unassigned):,})':    unassigned.obs['total_counts'].values,
}
parts = ax.violinplot(list(data_compare.values()), showmedians=True, showmeans=False)
for i, (body, color) in enumerate(zip(parts['bodies'], ['#2E86AB', '#A23B72'])):
    body.set_facecolor(color)
    body.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(list(data_compare.keys()), fontsize=9)
ax.set_ylabel('UMIs per cell')
ax.set_title('E. UMIs: assigned vs unassigned cells')
# Mann-Whitney U
u_stat, p_val = stats.mannwhitneyu(
    adata_assigned.obs['total_counts'].values,
    unassigned.obs['total_counts'].values,
    alternative='two-sided')
ax.text(0.5, 0.95, f'MWU p = {p_val:.2e}', transform=ax.transAxes,
        ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 7f. n_genes per cell: assigned vs unassigned
ax = axes[1, 2]
data_compare2 = {
    'Assigned':   adata_assigned.obs['n_genes'].values,
    'Unassigned': unassigned.obs['n_genes'].values,
}
parts2 = ax.violinplot(list(data_compare2.values()), showmedians=True)
for i, (body, color) in enumerate(zip(parts2['bodies'], ['#2E86AB', '#A23B72'])):
    body.set_facecolor(color)
    body.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(list(data_compare2.keys()))
ax.set_ylabel('Genes per cell')
ax.set_title('F. Genes: assigned vs unassigned cells')

plt.tight_layout()
save_fig(fig, 'fig7_guide_quality_threshold_sensitivity')


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary_lines = {
    'Total cells':            f"{adata.n_obs:,}",
    'Total genes':            f"{adata.n_vars:,}",
    'Data type':              data_type,
    'Median UMIs/cell (all)': f"{adata.obs['total_counts'].median():,.0f}",
    'Median genes/cell (all)':f"{adata.obs['n_genes'].median():,.0f}",
    'Median mito% (all)':     f"{adata.obs['mito_pct'].median():.2f}%",
    '': '',
    'Guide-assigned cells':   f"{adata_assigned.n_obs:,} ({100*adata_assigned.n_obs/adata.n_obs:.1f}%)",
    'Unique TFs':             f"{adata_assigned.obs['guide_assignment'].nunique()}",
    'Median guide UMIs':      f"{adata_assigned.obs['guide_total_umis'].median():.0f}",
    'Singlet rate':           f"{100*adata_assigned.obs['guide_is_singlet'].mean():.1f}%",
    'Multiplet rate':         f"{100*adata_assigned.obs['guide_is_multiplet'].mean():.1f}%",
}
for k, v in summary_lines.items():
    if k:
        print(f"  {k:<35}: {v}")
    else:
        print()

# Per cell line
print("\n--- By Cell Line (assigned cells) ---")
for cl in CELL_LINES:
    s = adata_assigned.obs[adata_assigned.obs['cell_line'] == cl]
    a = adata.obs[adata.obs['cell_line'] == cl]
    print(f"  {cl}: {len(s):,} assigned / {len(a):,} total "
          f"({100*len(s)/len(a):.1f}%)  |  "
          f"UMI med={s['total_counts'].median():.0f}  |  "
          f"guide UMI med={s['guide_total_umis'].median():.0f}")

# Save CSV
sample_csv = adata_assigned.obs.groupby('sample').agg(
    n_assigned=('total_counts', 'count'),
    umi_median=('total_counts', 'median'),
    genes_median=('n_genes', 'median'),
    mito_pct_median=('mito_pct', 'median'),
    guide_umi_median=('guide_total_umis', 'median'),
    saturation_median=('saturation', 'median'),
).round(2)
sample_csv.to_csv(FIG_DIR / 'sample_summary_assigned_cells.csv')

print(f"\nSummary table saved: {FIG_DIR / 'sample_summary_assigned_cells.csv'}")

print("\n" + "="*70)
print(f"All figures saved to: {FIG_DIR}")
print("="*70)
print("""
NATURE REVIEWER CHECKLIST — additional analyses to consider:
  □ Doublet detection (Scrublet / DoubletFinder) — run before guide assignment?
  □ Ambient RNA decontamination (SoupX / CellBender) — applied?
  □ PCA / UMAP coloured by batch variables to visualise batch effects
  □ Pseudobulk correlation across replicates (show reproducibility)
  □ CRISPR screen power calculation (cells per TF vs statistical power)
  □ Normalisation method: which scran / normalize_total + log1p?
  □ HVG selection parameters (n_top_genes, min_mean, max_mean)
""")
