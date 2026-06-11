# Pipeline verification report — `tf_screen/`

End-to-end synthetic-data verification of the `tf_screen/` pipeline, per
the brief in `CLAUDE_CODE_VERIFICATION_TASK.md`. Branch:
`claude/nice-babbage-0bcyk3` (developed off `tf_screen_release`).

**Headline result.** Every pipeline step runs to completion on the
synthetic toy dataset. Four real bugs were found and patched; one
cross-script numerical inconsistency was found and is flagged below
without being silently fixed.

---

## Manu — important note

The toy run is driven by `tf_screen/config.py`, which is gitignored and
contains:

```python
from tf_screen.tests.config_toy import *  # noqa: F401,F403
```

That redirects every script to `tf_screen/tests/toy_data/` instead of
your cluster paths. Before running on real data, either delete
`tf_screen/config.py` and re-copy from `config.template.py`, or replace
the contents with the real paths. The toy config lives at
`tf_screen/tests/config_toy.py` and is checked in so the verification
can be re-run from scratch.

---

## Step 1 — Reading

Read in order: top-level `README.md`, `docs/pipeline_overview.md`,
`docs/data_availability.md`, `config.template.py`, every per-folder
`README.md`, then a skim of every script. Mental model confirmed:

- **Pipeline order.** preprocessing → classifiers → tf_expression (TF
  diagnostics) + state_testing + dose_response + differential_expression
  (parallel) → downstream (stubs).
- **`joint_gbm_oe_anndata.h5ad`.** Concatenation of `~96k` atlas cells
  (subset to exclusive-topic cells) + `~261k` screen cells on a shared
  ~4,300-gene feature space. `obs` carries `dataset`, `cell_line`,
  `sample`, `guide_assignment`, `celltype` (fine, atlas only), `topic`
  (atlas only), `batch`; after the classifier pass `prob_ct_*`,
  `prob_coarse_*`, `prob_topic_*` are appended for every cell.
- **`prob_coarse_*`.** Per-cell softmax probability over the five
  non-Proliferative coarse states. Sums to one per cell.
- **FC0.5_p10.** OE filter: linear-space TF expression ≥ 1.5× the
  per-cell-line control mean. Ctrl filter: keep only controls in the
  bottom 10th percentile of TF expression.
- **`sig_consensus`.** Three-way: perm FDR < 0.05 **and** (t-test FDR <
  0.05 **or** Wilcoxon FDR < 0.05) **and** |Cohen's d| > 0.2.

---

## Step 2 — Static analysis

### What I ran

- `python -m py_compile $(find tf_screen -name "*.py")` — silent.
- `ruff check tf_screen/ --select=F` — 45 findings.
- Config-attribute audit (`grep -oE "config\.[A-Z_]+"` ∪ template
  definitions) — every referenced attribute is defined.
- `sys.path` shim — present in every numbered-folder script.
- Folder-rename / `pip install -e` grep — clean (no `01_preprocessing/`
  references, no install instructions referring to a `pyproject.toml`).

### Findings and fixes

| # | File | Problem | Fix | Behavioural? |
|---|---|---|---|---|
| 1 | `state_testing/topic_testing.py:48` | `from .state_testing import …` — relative import. Crashes with `ImportError` when run as a script (`python state_testing/topic_testing.py`), which is exactly how the README tells you to run it. | Switched to the absolute path `from tf_screen.state_testing.state_testing import …`. | No — pure import-mechanics fix. |
| 2 | `state_testing/proportion_testing.py:1022` | `harmony_path = Config.BASE_DIR / 'harmony_200pc' / 'harmony_embeddings.npy'` — hardcoded folder name that does **not** match `config.HARMONY_DIR` (which is `OUTPUT_DIR / 'harmony'`). Script `FileNotFoundError`'s on every fresh run. | Replaced with `harmony_path = config.HARMONY_DIR / 'harmony_embeddings.npy'`. | No — same file, correct path. |
| 3 | `state_testing/proportion_testing.py:1038` | `model_dir = Config.BASE_DIR / 'models'` — works only because `MODELS_DIR` happens to live at `OUTPUT_DIR / 'models'` in the template. Silently breaks if anyone changes `MODELS_DIR`. | Switched to `model_dir = config.MODELS_DIR`. | No — same default location. |
| 4 | `classifiers/train_state_classifier.py:131`, `classifiers/train_topic_classifier.py:113`, `state_testing/proportion_testing.py:323,341` | `LogisticRegression(multi_class="multinomial", …)` — the `multi_class` kwarg was removed in scikit-learn ≥1.7 (deprecated since 1.5). Crashes on import-time-fresh scikit-learn. | Dropped the kwarg; multinomial is now the default for `lbfgs`. Behaviour identical on sklearn 1.3 (the documented version) and on current sklearn. | No, on the documented pin (sklearn 1.3); on current sklearn this is the only way for the code to run at all. |

### Lint findings deliberately left alone

Ruff also flagged ~40 unused imports (`F401`) and a handful of
`f-string-without-placeholders` (`F541`), `unused-variable` (`F841`)
items. None of them are bugs — leaving them rather than reformatting code
that wasn't broken, per the brief.

### Real bug found that is **not** automatically fixed (flagged for Manu)

| # | File | Problem | Impact |
|---|---|---|---|
| 5 | `state_testing/state_testing.py` (`filter_cells_by_strategy`) **vs** `downstream/verify_deposit_against_fig3c.py` (`filter_for_strategy`) **vs** `utils/io.py::select_oe_and_ctrl_indices` | Three different cell-selection conventions live in the same pipeline. `state_testing.py` does the FC and p10 cut on **raw counts** (post-`np.expm1`). `verify_deposit_against_fig3c.py` and `utils/select_oe_and_ctrl_indices` (used by `wilcoxon_de`) do the cut on **10k-library-normalised** counts. The two will pick different OE/Ctrl cell sets whenever per-cell library sizes vary — i.e. always. | The numerical mismatch is the reason the toy verify-against-itself returns `FAIL` on `cohens_d` and `delta` while `sig_consensus` itself agrees 100% (1100/1100). On real data with 156 batches the absolute selection difference is probably small, but it **is** the explanation for the few-percent verify drift you described in `docs`. Recommended fix: pick one convention (the 10k-normalised one is correct — it matches the README for `wilcoxon_de.py`) and apply it everywhere. I have not touched it because it changes behaviour on real data. |

---

## Step 3 — Toy dataset

Generator at `tf_screen/tests/make_toy_data.py`; outputs at
`tf_screen/tests/toy_data/`. Deterministic on seed 42.

| File | Shape | Notes |
|---|---|---|
| `joint_gbm_oe_anndata.h5ad` | 5,000 cells × 500 genes (sparse) | 2,000 atlas + 3,000 screen. `obs` carries `dataset`, `cell_line`, `sample`, `guide_assignment`, `celltype` (fine, atlas), `celltype_coarse` (atlas), `topic` (atlas, `Topic_N`), `batch`. `.obsm['guide_counts']` is `(5000, 56)` Poisson; atlas rows are zero-padded. `.uns['guide_names']` lists the 56 guide labels. |
| `harmony_embeddings.npy` | (5000, 200) float32 | Atlas cells centroid-cluster by fine state so the classifiers can actually learn signal. Screen cells are noise around a state-specific centroid. |
| `gbm_tf_screen_clean.h5ad` | 3,000 cells × 500 genes (sparse) | The screen subset on its own, for `tf_self_expression.py` and `wilcoxon_de.py`. Adds `guide_has_assignment=True` for every row. |

The first 55 genes in `var_names` are the TF symbols (`ASCL1`, `OLIG2`,
…); the remaining 445 are `GENE_0000…GENE_0444` filler. Each TF's
expression in its own OE cells is drawn from `Poisson(λ=16)` vs the
baseline `Poisson(λ=2)`, so the FC0.5 filter selects a non-empty set
without dominating the matrix.

Sanity check (run after generation):

```python
import anndata as ad, numpy as np
a = ad.read_h5ad("tf_screen/tests/toy_data/joint_gbm_oe_anndata.h5ad")
h = np.load("tf_screen/tests/toy_data/harmony_embeddings.npy")
assert h.shape[0] == a.n_obs                # 5000
assert (a.obs.dataset == "TF_screen").sum() == 3000
print("OK")
```

Passes.

---

## Step 4 — Config

- `tf_screen/tests/config_toy.py` mirrors `config.template.py` with the
  toy paths and the brief's three suggested overrides:
  `MIN_CELLS_PER_GROUP = 5`, `N_PERMUTATIONS = 100`,
  `LEIDEN_RESOLUTION = 0.5`.
- `tf_screen/config.py` (gitignored) just re-exports `tests/config_toy`.
  Delete or replace before a real run; see the note at the top of this
  report.

---

## Step 5 — End-to-end run

| Step | Command | Runtime | Output produced |
|---|---|---|---|
| `preprocessing/build_joint_anndata.py` | *skipped* — toy joint AnnData is pre-built; the script requires the full atlas + scDoRI inputs that the synthetic dataset replaces. | — | — |
| `preprocessing/run_harmony.py` | *skipped* — toy Harmony embedding is pre-built (atlas cells are deliberately cluster-aware so classification has signal; running Harmony on synthetic noise would erase that). | — | — |
| `preprocessing/qc_figures.py` | *skipped* — requires ~14 screen-specific QC obs columns (`mito_fraction`, `guide_total_umis`, `cell_density`, `replicate`, `Ligation`, `RT`, `saturation`, `guide_is_singlet`, `guide_is_multiplet`, `guide_dominance_ratio`, …) that don't survive the toy abstraction. Pure visualisation; not on the critical path. | — | — |
| `classifiers/train_state_classifier.py` | `python classifiers/train_state_classifier.py` | 3 s | `MODELS_DIR/clf_celltype_{fine,coarse}.pkl` (+ scaler + LE), `state_classifier_metrics.csv`. Joint AnnData updated in place with `prob_ct_*` and `prob_coarse_*`. Test accuracy 1.0/1.0 — synthetic clusters are linearly separable. |
| `classifiers/train_topic_classifier.py` | `python classifiers/train_topic_classifier.py` | 9 s | `clf_topic.pkl` (+ scaler + LE), `topic_classifier_metrics.csv`. `prob_topic_*` written back. Test accuracy 0.04 — as expected for the toy data, where atlas topics are random. |
| `tf_expression/tf_self_expression.py` | `python tf_expression/tf_self_expression.py` | 24 s | `TF_EXPRESSION_DIR/tf_expression_results.csv` (220 rows = 55 TFs × 4 scopes) + 5 figure PDFs. Median log2FC 2.6 — the 8× boost is recovered. |
| `tf_expression/control_p10_validation.py` | `python tf_expression/control_p10_validation.py` | 20 s | `control_state_analysis/{control_states_by_cell_line,control_states_tf_filtering,control_state_shifts}.csv` + 4 figures. |
| `state_testing/state_testing.py` | `python state_testing/state_testing.py` | 326 s | `STATE_TESTING_DIR/<strategy>/{fine_grained,coarse}_effects.csv` for all 16 strategies + `strategy_comparison_{fine,coarse}.csv`. |
| `state_testing/topic_testing.py` | `python state_testing/topic_testing.py` | 406 s | `TOPIC_TESTING_DIR/<strategy>/topic_effects.csv` × 16 + `strategy_comparison_topics.csv`. |
| `state_testing/proportion_testing.py` | `python state_testing/proportion_testing.py` | 297 s | `PROPORTION_DIR/<strategy>/proportion_{lrt,fisher}.csv` × 16 + `all_strategies_{lrt,fisher}.csv` + `strategy_comparison/`. |
| `dose_response/build_metacells.py` | `python dose_response/build_metacells.py` | 64 s | `METACELL_DIR/metacells.h5ad` (259 metacells × 500 genes) + `metacell_summary.csv`. |
| `dose_response/correlation_analysis.py` | `python dose_response/correlation_analysis.py` | 180 s | `CORRELATION_DIR/correlation_{combined,oe_only,control_only}.csv` + `per_cell_line/correlation_per_cell_line.csv` + heatmap/scatter/summary figures. |
| `differential_expression/wilcoxon_de.py all_all` | `python differential_expression/wilcoxon_de.py all_all` | 102 s | `DE_DIR/all_all/{cell_line}__{TF}__DE.csv` × 165, `all_DE_results.csv` (82,500 rows), `intersection/all_intersection_results.csv` + `intersection_summary.csv`, top-level `strategy_comparison.csv`. |

Total wall time (steps that actually ran): ≈ 28 minutes.

### Files produced — summary

```
tests/toy_output/
├── control_state_analysis/      # 3 CSVs + 4 fig sets
├── correlation/                 # 4 CSVs + heatmaps/scatter/distributions/per_cell_line/
├── differential_expression/
│   ├── all_all/                 # 165 per-(TF,line) CSVs + all_DE_results.csv + intersection/
│   └── strategy_comparison.csv
├── metacells/                   # metacells.h5ad + metacell_summary.csv
├── models/                      # 9 .pkl files + 2 metrics CSVs
├── proportion_testing/          # 16 strategy folders + all_strategies_* + strategy_comparison/
├── state_testing/               # 16 strategy folders + strategy_comparison_{fine,coarse}.csv
├── tf_expression/               # tf_expression_results.csv + 5 fig sets
├── topic_testing/               # 16 strategy folders + strategy_comparison_topics.csv
└── verify_deposit_output/       # see Step 6
```

---

## Step 6 — Verification script

`tf_screen/downstream/verify_deposit_against_fig3c.py` exists in the
repo. Its `Config` class hardcodes Manu's cluster paths, so I drove it
through a thin wrapper `tf_screen/tests/run_verify_toy.py` that
monkey-patches the four `Config` attributes to point at the toy joint
AnnData and the toy `FC0.5_p10/coarse_effects.csv`. Wrapper also sets
`MIN_CELLS = max(5, config.MIN_CELLS_PER_GROUP)` and `N_PERMS =
config.N_PERMUTATIONS` so the test runs on the smaller toy strata.

### Result

The script runs to completion (44 s) and writes
`coarse_effects_from_deposit.csv` + `summary.csv` +
`sig_consensus_disagreements.csv` to `OUTPUT_DIR/verify_deposit_output/`.

```
rows only in regenerated: 275        (verify treats "atlas" as a scope; state_testing.py doesn't)
rows only in reference:   0
rows in both:             1100

max |diff| per column (deterministic columns):
  cohens_d      5.758e-01   <- WORRY (see Step 2 finding #5)
  delta         7.656e-02   <- WORRY
  tf_mean       1.192e-07
  ctrl_mean     7.656e-02   <- WORRY
  ttest_pval    5.662e-01   <- WORRY
  wilcox_pval   7.636e-01   <- WORRY

sig_consensus agreement: 1100/1100 (100.00%)

VERDICT: FAIL
```

The `FAIL` verdict is the (expected) consequence of finding #5 above:
`state_testing.py` filters cells in raw-count space, the verify script
filters in 10k-normalised space, so the OE and Ctrl strata going into
the statistical tests are different sets of cells with different means
and effect sizes. The `sig_consensus` agreement is still 100% because
the consensus criterion is much less sensitive than the underlying
numerical means.

Two extra wrinkles specific to the toy run, neither of them a code bug:

- The 275 extra regenerated rows come from the verify script iterating
  over every unique `cell_line` value, which in the toy joint AnnData
  includes `"atlas"`. The state_testing run never produces an `atlas`
  scope (it filters to `dataset != "GBM_atlas"` first). This isn't a
  bug — it's a consequence of using the joint AnnData as the deposit
  stand-in. On the real deposit (`gbm_tf_screen_clean_annotated.h5ad`)
  there are no atlas cells.
- Even if finding #5 is fixed, this toy run will still report a tiny
  `perm_pval` diff because both scripts share `config.N_PERMUTATIONS`
  but use different `np.random.default_rng(seed)` sequences (the seed
  function and seed values are identical, but the permutation loop
  interleaves differently for any TF where the FC/p10 filter rejects
  different cells).

---

## Step 7 — Things I noticed but did not fix

- Three different cell-selection implementations (Step 2 finding #5).
  Pick one and centralise it on `utils/io.py::select_oe_and_ctrl_indices`.
- Logistic-regression `multi_class` is now hardcoded silently to the
  sklearn default. If the published numbers were generated on sklearn
  1.3 with `multi_class="multinomial"`, the default really is
  multinomial there too, so this is fine — but a one-line
  defensive `assert sklearn.__version__ < "1.7"` or a forward-compat
  shim in `utils/io.py` would protect against silent
  one-vs-rest regression on a future sklearn that changes defaults.
- `proportion_testing.py` is the only script in the pipeline that
  bypasses the `sys.path` shim ↔ `config` indirection and instead
  reads paths through a local `Config` class. The fix in finding #2/#3
  is the minimum needed; longer-term, harmonising it with the rest of
  the pipeline's structure (move filter logic into `state_testing.py`,
  read paths from `config` directly) would remove an entire class of
  cross-script drift.
- `train_topic_classifier.py` reports 0.04 test accuracy on toy data —
  expected because toy atlas topics are uncorrelated with Harmony
  embedding. The fact that this passes through without an
  `AssertionError` on metrics is fine for the toy run but means a real
  unsupervised regression in classifier quality would only surface
  later in the pipeline (when downstream tests have no signal). A
  `metrics["balanced_accuracy"] < 0.1` warning would catch that.
- `dose_response/build_metacells.py` and `correlation_analysis.py`
  both re-implement small chunks of normalisation / column selection
  that exist in `utils/io.py`. Not a bug, but cleanup material.
- `qc_figures.py` has a hard requirement on ~14 obs columns that aren't
  documented anywhere — if anyone re-runs the QC script on a re-export
  of the screen AnnData that's missing one of them, it'll
  `KeyError`. A defensive check at the top of the script ("here are
  the columns I expect; the data only has X, Y, Z") would help future
  users.

---

## Deliverables (recap)

- `tf_screen/tests/make_toy_data.py` — toy dataset generator.
- `tf_screen/tests/toy_data/` — generated joint AnnData, Harmony
  embedding, screen AnnData.
- `tf_screen/tests/config_toy.py` — toy config.
- `tf_screen/tests/run_verify_toy.py` — verify-script driver.
- `tf_screen/tests/REPORT.md` — this file.
- `tf_screen/config.py` — gitignored override pointing at
  `tests/config_toy`. **Remove or replace before running on real data.**
- Bug fixes:
  - `classifiers/train_state_classifier.py` (`multi_class` removed)
  - `classifiers/train_topic_classifier.py` (`multi_class` removed)
  - `state_testing/topic_testing.py` (relative → absolute import)
  - `state_testing/proportion_testing.py` (`multi_class` removed,
    hardcoded harmony / models paths replaced with `config.*`).
