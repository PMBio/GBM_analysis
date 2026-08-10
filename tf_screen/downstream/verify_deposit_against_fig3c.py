#!/usr/bin/env python3
"""Verify that the GEO deposit AnnData faithfully reproduces Fig. 3c.

What this checks
----------------
The deposit ``gbm_tf_screen_clean_annotated.h5ad`` contains the same
``prob_coarse_*`` columns and the same raw counts that were used to
produce Fig. 3c. So if we re-run the FC0.5_p10 strategy of the
statistical-testing pipeline using ONLY the deposit (raw counts for
the FC filter, prob_coarse_* for the test), we should reproduce the
reference table at:

    validation_analysis_200pc/statistical_testing/FC0.5_p10/
        coarse_effects.csv

That reference table is what Fig. 3c is drawn from. Bit-exact (within
floating-point precision) recovery means anyone with the deposit can
recreate Fig. 3c's data.

What we compare
---------------
Joined on (TF, coarse_state, cell_line):

    cohens_d         exact agreement expected (deterministic)
    ttest_pval       exact (deterministic)
    wilcox_pval      exact (deterministic)
    perm_pval        ~3-4 decimal places (1,000 random permutations
                     with the same seed)
    sig_consensus    should be identical for >99% of rows; any
                     disagreement is dumped for inspection.

Pass criteria
-------------
    PASS: max |cohens_d_diff| < 1e-9  AND  sig_consensus agreement >= 99.5%
    WARN: max |cohens_d_diff| < 1e-6  AND  sig_consensus agreement >= 99.0%
    FAIL: anything else.

Usage
-----
``python verify_deposit_against_fig3c.py``
"""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    # The deposit we want to verify.
    DEPOSIT = Path(
        "/omics/groups/OE0540/internal/users/msaraswat/gbm_tf_oe_screen/"
        "crispr_reseq/anndata/gbm_tf_screen_clean_annotated.h5ad"
    )

    # The reference table the deposit should be able to reproduce.
    REFERENCE = Path(
        "/omics/groups/OE0540/internal/users/msaraswat/gbm_tf_oe_screen/"
        "crispr_reseq/figures/validation_analysis_200pc/statistical_testing/"
        "FC0.5_p10/coarse_effects.csv"
    )

    # Output: regenerated table and the diff report.
    OUT_DIR = Path("./verify_deposit_output")

    # Strategy: FC0.5_p10 -- 1.5x linear OE, bottom-10% controls.
    FC_THRESHOLD = 0.5          # OE >= (1 + 0.5) x ctrl mean
    CTRL_PERCENTILE = 10        # keep ctrl cells in bottom 10%

    # Statistical testing parameters.
    MIN_CELLS = 10
    N_PERMS = 1000
    SEED = 42

    # Coarse-state column prefix in the deposit.
    PROB_PREFIX = "prob_coarse_"


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Core statistical test (lifted verbatim from
# statistical_testing_multiStrategy.py so behaviour matches exactly).
# ---------------------------------------------------------------------------

def run_all_tests(tf_scores: np.ndarray, ctrl_scores: np.ndarray,
                  n_perms: int = Config.N_PERMS,
                  seed: int = Config.SEED) -> dict:
    """T-test, Mann-Whitney, matched-subsample permutation test."""
    rng = np.random.default_rng(seed)
    n_tf, n_ctrl = len(tf_scores), len(ctrl_scores)

    out = {
        "n_tf": n_tf, "n_ctrl": n_ctrl,
        "tf_mean":   float(np.mean(tf_scores))  if n_tf else np.nan,
        "ctrl_mean": float(np.mean(ctrl_scores)) if n_ctrl else np.nan,
        "tf_std":    float(np.std(tf_scores))   if n_tf else np.nan,
        "ctrl_std":  float(np.std(ctrl_scores))  if n_ctrl else np.nan,
    }
    out["delta"] = out["tf_mean"] - out["ctrl_mean"]
    pooled = np.sqrt((out["tf_std"] ** 2 + out["ctrl_std"] ** 2) / 2)
    out["cohens_d"] = out["delta"] / pooled if pooled > 0 else np.nan

    if n_tf < Config.MIN_CELLS or n_ctrl < Config.MIN_CELLS:
        out["ttest_pval"] = out["wilcox_pval"] = out["perm_pval"] = np.nan
        return out

    try:
        _, out["ttest_pval"] = stats.ttest_ind(tf_scores, ctrl_scores)
    except Exception:
        out["ttest_pval"] = np.nan
    try:
        _, out["wilcox_pval"] = stats.mannwhitneyu(
            tf_scores, ctrl_scores, alternative="two-sided")
    except Exception:
        out["wilcox_pval"] = np.nan

    # Matched-subsample permutation.
    observed = out["delta"]
    m = min(n_tf, n_ctrl)
    null = np.empty(n_perms)
    for i in range(n_perms):
        sub = rng.choice(ctrl_scores, size=m, replace=False)
        pooled_arr = np.concatenate([tf_scores, sub])
        rng.shuffle(pooled_arr)
        null[i] = pooled_arr[:n_tf].mean() - pooled_arr[n_tf:].mean()
    out["perm_pval"] = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perms + 1)
    return out


# ---------------------------------------------------------------------------
# Cell filtering (FC0.5 OE + p10 Ctrl)
# ---------------------------------------------------------------------------

def filter_for_strategy(adata: ad.AnnData) -> dict[str, dict]:
    """For each TF, build (oe_indices, ctrl_indices) under FC0.5_p10.

    Operates on 10k-normalised TF expression for the cell selection only.
    The downstream test reads ``prob_coarse_*`` columns from ``obs``,
    not the expression matrix.
    """
    banner("2. Filtering cells per TF (FC0.5 OE + p10 Ctrl)")

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    # If the deposit ever ships log-transformed, undo it for the filter:
    if X.max() < 20:
        X = np.expm1(X)
    # 10k normalisation.
    lib_sizes = X.sum(axis=1).astype(np.float64)
    nz = lib_sizes > 0
    factors = np.zeros_like(lib_sizes)
    factors[nz] = 1e4 / lib_sizes[nz]
    # We won't materialise the whole normalised matrix; instead we'll
    # normalise per TF gene below to save memory.

    var_names = list(adata.var_names)
    guide = adata.obs["guide_assignment"].astype(str)
    cell_lines = adata.obs["cell_line"].astype(str)

    tfs = sorted(g for g in guide.unique()
                 if g not in ("Ctrl",) and not g.startswith("Unassign"))
    print(f"   TFs: {len(tfs)}")
    print(f"   Cell lines: {sorted(cell_lines.unique())}")

    masks: dict[str, dict] = {}
    for ti, tf in enumerate(tfs, 1):
        if tf not in var_names:
            masks[tf] = None
            continue
        gene_idx = var_names.index(tf)

        # 10k-normalised expression of this TF in all cells.
        expr_raw = X[:, gene_idx]
        expr_10k = expr_raw * factors

        oe_global = np.where(guide.values == tf)[0]
        ctrl_global = np.where(guide.values == "Ctrl")[0]

        oe_keep: list[int] = []
        ctrl_keep: list[int] = []
        for cl in sorted(cell_lines.unique()):
            cl_mask = (cell_lines.values == cl)
            oe_cl = oe_global[cl_mask[oe_global]]
            ctrl_cl = ctrl_global[cl_mask[ctrl_global]]
            if len(oe_cl) == 0 or len(ctrl_cl) == 0:
                continue

            ctrl_expr = expr_10k[ctrl_cl]
            ctrl_mean = float(np.mean(ctrl_expr))

            # FC0.5 filter on OE
            oe_expr = expr_10k[oe_cl]
            if ctrl_mean > 0:
                keep_oe_mask = (oe_expr / ctrl_mean) >= (1.0 + Config.FC_THRESHOLD)
            else:
                keep_oe_mask = oe_expr > 0
            oe_keep.extend(oe_cl[keep_oe_mask].tolist())

            # p10 filter on Ctrl
            cutoff = float(np.percentile(ctrl_expr, Config.CTRL_PERCENTILE))
            ctrl_keep.extend(ctrl_cl[ctrl_expr <= cutoff].tolist())

        masks[tf] = {
            "oe_indices":   np.array(oe_keep, dtype=int),
            "ctrl_indices": np.array(ctrl_keep, dtype=int),
        }
        if ti % 10 == 0:
            print(f"   [{ti}/{len(tfs)}] processed")
    return masks


# ---------------------------------------------------------------------------
# Run all (TF, state, cell_line) tests
# ---------------------------------------------------------------------------

def _seed(*parts: object) -> int:
    return abs(hash(":".join(str(p) for p in parts))) % 100_000


def regenerate_table(adata: ad.AnnData, tf_masks: dict) -> pd.DataFrame:
    """Reproduce FC0.5_p10 coarse_effects.csv from the deposit."""
    banner("3. Re-running statistical tests")

    obs = adata.obs
    cell_lines = sorted(obs["cell_line"].astype(str).unique())
    coarse_cols = [c for c in obs.columns if c.startswith(Config.PROB_PREFIX)]
    coarse_states = [c.replace(Config.PROB_PREFIX, "") for c in coarse_cols]
    print(f"   coarse states ({len(coarse_states)}): {coarse_states}")
    print(f"   cell lines ({len(cell_lines)}): {cell_lines}")

    rows = []
    tfs = [t for t in tf_masks if tf_masks[t] is not None]
    print(f"   TFs to test: {len(tfs)}")

    def _block(scope: str, cl_mask):
        for ti, tf in enumerate(tfs, 1):
            m = tf_masks[tf]
            oe_idx = m["oe_indices"]
            ctrl_idx = m["ctrl_indices"]
            if cl_mask is not None:
                oe_idx = oe_idx[cl_mask[oe_idx]]
                ctrl_idx = ctrl_idx[cl_mask[ctrl_idx]]
            for st, col in zip(coarse_states, coarse_cols):
                tf_scores = obs.iloc[oe_idx][col].values
                ctrl_scores = obs.iloc[ctrl_idx][col].values
                res = run_all_tests(
                    tf_scores, ctrl_scores,
                    seed=_seed("FC0.5_p10", scope, tf, st),
                )
                res.update({
                    "TF": tf, "coarse_state": st,
                    "cell_line": scope, "strategy": "FC0.5_p10",
                })
                rows.append(res)
            if ti % 10 == 0:
                print(f"     [{ti}/{len(tfs)}] {tf} ({scope})")

    print("   pooled...")
    _block("pooled", None)
    cl_values = obs["cell_line"].astype(str).values
    for cl in cell_lines:
        print(f"   {cl}...")
        _block(cl, cl_values == cl)

    df = pd.DataFrame(rows)
    # BH-FDR per test.
    for pcol in ("ttest_pval", "wilcox_pval", "perm_pval"):
        fcol = pcol.replace("_pval", "_fdr")
        df[fcol] = multipletests(df[pcol].fillna(1.0), method="fdr_bh")[1]
    df["sig_consensus"] = (
        ((df["ttest_fdr"] < 0.05) | (df["wilcox_fdr"] < 0.05))
        & (df["perm_fdr"] < 0.05)
        & (df["cohens_d"].abs() > 0.2)
    )
    print(f"   regenerated table: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def compare(regen: pd.DataFrame, reference: pd.DataFrame, out_dir: Path) -> dict:
    banner("4. Comparing regenerated vs reference")
    print(f"   regenerated: {len(regen):,} rows")
    print(f"   reference:   {len(reference):,} rows")

    keys = ["TF", "coarse_state", "cell_line"]
    merged = regen.merge(reference, on=keys, suffixes=("_new", "_ref"),
                         how="outer", indicator=True)
    only_new = (merged["_merge"] == "left_only").sum()
    only_ref = (merged["_merge"] == "right_only").sum()
    in_both = (merged["_merge"] == "both").sum()
    print(f"   rows only in regenerated: {only_new}")
    print(f"   rows only in reference:   {only_ref}")
    print(f"   rows in both:             {in_both}")

    both = merged[merged["_merge"] == "both"].copy()

    # Numerical diffs.
    diffs = {}
    for col in ("cohens_d", "delta", "ttest_pval", "wilcox_pval",
                "tf_mean", "ctrl_mean"):
        a = both[f"{col}_new"].values
        b = both[f"{col}_ref"].values
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() == 0:
            diffs[col] = np.nan
        else:
            diffs[col] = float(np.max(np.abs(a[valid] - b[valid])))

    # Permutation p-values: only check that they're close, not exact.
    pa = both["perm_pval_new"].values
    pb = both["perm_pval_ref"].values
    valid = ~(np.isnan(pa) | np.isnan(pb))
    perm_pval_max_diff = float(np.max(np.abs(pa[valid] - pb[valid]))) if valid.any() else np.nan

    print(f"\n   max |diff| per column (deterministic columns):")
    for col in ("cohens_d", "delta", "tf_mean", "ctrl_mean",
                "ttest_pval", "wilcox_pval"):
        v = diffs[col]
        flag = " <- WORRY" if (not np.isnan(v) and v > 1e-6) else ""
        print(f"     {col:14s}  {v:.3e}{flag}")
    print(f"\n   max |diff| perm_pval (stochastic): {perm_pval_max_diff:.3e}")

    # sig_consensus agreement.
    agree = (both["sig_consensus_new"] == both["sig_consensus_ref"])
    n_agree = int(agree.sum())
    n_disagree = int((~agree).sum())
    pct_agree = 100.0 * n_agree / len(both) if len(both) else 0
    print(f"\n   sig_consensus agreement: {n_agree}/{len(both)} ({pct_agree:.2f}%)")

    disagreements = both.loc[~agree, keys + [
        "cohens_d_new", "cohens_d_ref",
        "perm_fdr_new", "perm_fdr_ref",
        "sig_consensus_new", "sig_consensus_ref",
    ]]
    if not disagreements.empty:
        print("\n   Rows where sig_consensus disagrees:")
        for _, row in disagreements.head(20).iterrows():
            print(f"     {row['TF']:8s} {row['coarse_state']:18s} {row['cell_line']:8s}  "
                  f"d={row['cohens_d_new']:+.4f}/{row['cohens_d_ref']:+.4f}  "
                  f"q={row['perm_fdr_new']:.3g}/{row['perm_fdr_ref']:.3g}  "
                  f"sig={row['sig_consensus_new']}/{row['sig_consensus_ref']}")
        disagreements.to_csv(out_dir / "sig_consensus_disagreements.csv", index=False)

    # Verdict.
    cohens_d_ok = diffs["cohens_d"] < 1e-9
    cohens_d_warn = diffs["cohens_d"] < 1e-6
    sig_ok = pct_agree >= 99.5
    sig_warn = pct_agree >= 99.0

    if cohens_d_ok and sig_ok:
        verdict = "PASS"
    elif cohens_d_warn and sig_warn:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "verdict": verdict,
        "rows_only_new": int(only_new),
        "rows_only_ref": int(only_ref),
        "rows_both":     int(in_both),
        "max_diff_cohens_d":   diffs["cohens_d"],
        "max_diff_ttest_pval": diffs["ttest_pval"],
        "max_diff_wilcox_pval": diffs["wilcox_pval"],
        "max_diff_perm_pval":  perm_pval_max_diff,
        "sig_consensus_agreement_pct": pct_agree,
        "sig_consensus_disagreements": n_disagree,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1. Loading deposit + reference")
    print(f"Deposit:   {Config.DEPOSIT}")
    adata = ad.read_h5ad(Config.DEPOSIT)
    print(f"   cells: {adata.n_obs:,}   genes: {adata.n_vars:,}")
    print(f"\nReference: {Config.REFERENCE}")
    ref = pd.read_csv(Config.REFERENCE)
    print(f"   rows: {len(ref):,}")

    tf_masks = filter_for_strategy(adata)
    regen = regenerate_table(adata, tf_masks)
    regen.to_csv(Config.OUT_DIR / "coarse_effects_from_deposit.csv", index=False)
    print(f"   wrote {Config.OUT_DIR / 'coarse_effects_from_deposit.csv'}")

    summary = compare(regen, ref, Config.OUT_DIR)

    banner("VERDICT")
    print(f"   {summary['verdict']}")
    for k, v in summary.items():
        if k == "verdict":
            continue
        print(f"     {k:35s} {v}")

    pd.Series(summary).to_csv(Config.OUT_DIR / "summary.csv")
    print(f"\nFull outputs in: {Config.OUT_DIR}")


if __name__ == "__main__":
    main()
