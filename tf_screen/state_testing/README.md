# State testing (step 4)

Three scripts implementing the statistical pipeline for the TF screen's primary effect-size analysis (Fig. 3c–d) and its orthogonal compositional confirmation (Fig. 3c "black-box" overlay).

1. **`state_testing.py`** — TF effects on coarse (5-class) and fine (9-class) cell-state probabilities, across a 16-strategy filtering grid.
2. **`topic_testing.py`** — TF effects on the 16 tested topic probabilities, same grid, same tests.
3. **`proportion_testing.py`** — orthogonal compositional test: hard-assign each OE cell to its argmax coarse state, then test TF × state with multinomial LRT and Fisher's exact.

All three scripts share the same FC × control-percentile filtering grid and use the same `run_all_tests` permutation framework (`state_testing.py` is the source of truth; `topic_testing.py` imports its testing functions).

---

## Inputs

| Path | Source |
|---|---|
| `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | populated by `classifiers/train_state_classifier.py` (adds `prob_coarse_*`, `prob_ct_*`) and `classifiers/train_topic_classifier.py` (adds `prob_topic_*`) |

## Outputs

| Script | Path | Content |
|---|---|---|
| `state_testing.py` | `STATE_TESTING_DIR/<strategy>/fine_grained_effects.csv` | One row per (TF, fine state, scope) with t/Wilcoxon/perm p-values, FDRs, Cohen's d, `sig_consensus`. |
| `state_testing.py` | `STATE_TESTING_DIR/<strategy>/coarse_effects.csv` | Same, for the 5 coarse states. |
| `state_testing.py` | `STATE_TESTING_DIR/strategy_comparison_{fine,coarse}.csv` | Summary across the 16 strategies. |
| `topic_testing.py` | `TOPIC_TESTING_DIR/<strategy>/topic_effects.csv` | One row per (TF, topic, scope). |
| `topic_testing.py` | `TOPIC_TESTING_DIR/strategy_comparison_topics.csv` | Summary across strategies. |
| `proportion_testing.py` | `PROPORTION_DIR/<strategy>/proportion_lrt.csv` | One row per (TF, scope): LRT statistic, p, FDR, per-state log-OR coefs. |
| `proportion_testing.py` | `PROPORTION_DIR/<strategy>/proportion_fisher.csv` | One row per (TF, state, scope): Fisher OR, p, FDR, `delta_prop`. |
| `proportion_testing.py` | `PROPORTION_DIR/all_strategies_{lrt,fisher}.csv` | Concatenations + strategy_comparison plots. |

---

## Run

```bash
python state_testing/state_testing.py
python state_testing/topic_testing.py
python state_testing/proportion_testing.py
```

`topic_testing.py` re-uses the filter and test code from `state_testing.py`, so run them in either order — the imports are at the module level.

---

## The consensus significance criterion

A TF × state (or TF × topic) effect is flagged `sig_consensus = True` when **all three** of the following hold:

| Condition | Threshold |
|---|---|
| Permutation FDR | `< 0.05` (primary) |
| t-test FDR **or** Wilcoxon FDR | `< 0.05` (confirmatory) |
| Cohen's d | `> 0.2` (effect-size floor) |

This is what's reported as "significant" in Fig. 3c. The permutation test is the primary; t/Wilcoxon are there to confirm and to populate the per-test columns in the supplementary tables.

## The compositional ("black-box" overlay) criterion

`proportion_testing.py`'s Fisher's exact odds ratio gives a separate compositional view. The Fig. 3c "black-box" annotation marks effects with `fisher_OR > 1.5` *and* `fisher_fdr < 0.05` from the FC0.5_p10 strategy. This is independent of the probability-based consensus criterion above.

---

## The 16-strategy filtering grid

| Dimension | Values | Meaning |
|---|---|---|
| `fc` (OE filter) | 0.5, 1.0, 1.5, 2.0 | Keep OE cells with linear-space TF expression ≥ (1 + fc) × control mean. |
| `ctrl` (Ctrl filter) | "all", 50, 25, 10 | Keep all controls, or only the bottom 50% / 25% / 10% of TF expressors. |

Headline strategy in the paper: **`FC0.5_p10`** (= 1.5× linear OE, bottom-10% controls).

## Notes

* **Permutation seed.** Each (strategy × scope × TF × state) comparison gets a deterministic seed derived from a hash of the comparison labels, so the permutation null is reproducible per cell across runs. The grand-overall random seed is `config.RANDOM_SEED`.
* **Proliferative exclusion** in proportion testing happens at the cell level: any cell whose argmax over **all six** coarse states (including Proliferative) is Proliferative is dropped from the proportion analysis. The remaining cells are then re-bucketed into the five non-Proliferative states by argmax-over-five.
* **Topic exclusion.** `topic_testing.py` filters down to the 16 topics in `config.TOPICS_TESTED` (Topics 7, 23, 29 dropped) before testing. The other three topics are still present as probability columns on the AnnData (so the softmax remains valid) but are not subjected to TF tests.
