# Classifiers (step 2)

Two scripts that train the supervised classifiers used downstream to assign per-cell state and topic probabilities to every screen cell.

1. **`train_state_classifier.py`** — fits two multinomial logistic regressions on the Harmony embedding: a *fine* classifier (9 GB cell states) and a *coarse* classifier (5 GB cell states defined by `config.COARSE_GROUPS`). The *Proliferative* state is excluded from both because it is a cell-cycle programme rather than a stable cell-state identity.
2. **`train_topic_classifier.py`** — fits a single multinomial logistic regression over the 19 GB-intrinsic topics listed in `config.TOPICS_TRAINED`. Topic probabilities sum to one per cell.

Both scripts (i) train on the atlas portion of the joint object, (ii) apply the trained model to **every** cell (atlas + screen) and write the resulting probabilities back into the joint AnnData's `.obs` columns, and (iii) persist the model, the standard scaler, and the label encoder as `.pkl` files for later use without re-training.

---

## Inputs

| Path | Source |
|---|---|
| `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | `preprocessing/build_joint_anndata.py` |
| `HARMONY_DIR/harmony_embeddings.npy` | `preprocessing/run_harmony.py` |

## Outputs

| Path | Content |
|---|---|
| `MODELS_DIR/clf_celltype_fine.pkl` + scaler + le | Fine 9-class state classifier |
| `MODELS_DIR/clf_celltype_coarse.pkl` + scaler + le | Coarse 5-class state classifier |
| `MODELS_DIR/clf_topic.pkl` + scaler + le | 19-class topic classifier |
| `MODELS_DIR/state_classifier_metrics.csv` | Test-set accuracy + balanced accuracy |
| `MODELS_DIR/topic_classifier_metrics.csv` | Test-set accuracy + balanced accuracy |
| `JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad` | Updated with `prob_ct_*`, `prob_coarse_*`, `prob_topic_*` columns |

---

## Run

```bash
python classifiers/train_state_classifier.py
python classifiers/train_topic_classifier.py
```

Either order is fine — the two scripts don't depend on each other and both append non-overlapping columns to the joint AnnData.

---

## What's in the `obs` table after this step

For every cell in the joint object:

* `predicted_celltype` (argmax label) and `prob_ct_<fine>` for each of the 9 fine states
* `predicted_coarse` (argmax label) and `prob_coarse_<coarse>` for each of the 5 coarse states
* `predicted_topic` (argmax label) and `prob_topic_<Topic_N>` for each of the 19 trained topics

Downstream scripts read these `.obs` columns directly; the `.pkl` files exist so that the models can be reused (e.g. for a new batch of screen cells) without re-training.

---

## Notes

* **Multinomial, not one-vs-rest.** All three classifiers use `multi_class="multinomial"` (soft-max). The 5 coarse-state probabilities sum to one per cell, and the 19 trained-topic probabilities also sum to one. The earlier classifier in the original codebase was fit with `multi_class="ovr"`; switching to multinomial gives the per-cell probabilities a direct compositional interpretation (a TF-induced *increase* in one state's probability necessarily comes from *decreases* in others), which is what the paper relies on.

* **Topic exclusion happens downstream.** All 19 topics in `config.TOPICS_TRAINED` are used to fit the topic classifier. The smaller list `config.TOPICS_TESTED` (16 topics — Topics 7, 23, and 29 dropped due to unstable cross-validation performance) is applied later in `state_testing/topic_testing.py`. Keeping all 19 in the classifier preserves the soft-max normalisation.

* **Random seed.** `config.RANDOM_SEED` (default 42) is passed to both the 80/20 stratified split and the logistic-regression fit, so results are deterministic given the same input matrix.

* **What's *not* re-runnable bit-identically.** The original models in the Zenodo deposit were trained with `multi_class="ovr"` for the topic classifier; re-training with the cleaned multinomial code will produce different probability values (though qualitatively similar predictions). The classifier *.pkl* files in the deposit are the authoritative versions used to produce the published figures.
