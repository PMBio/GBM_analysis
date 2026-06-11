"""Statistical testing of TF effects on topic probabilities.

Identical statistical framework to ``state_testing.py``, applied to topic
probabilities instead of cell-state probabilities. The 19 trained topics
(``config.TOPICS_TRAINED``) are filtered down to the 16 tested topics
(``config.TOPICS_TESTED``); the excluded topics still contribute to the
softmax denominator (set during classifier training) but their per-cell
probabilities are not subjected to TF-effect testing.

For consistency, this script also exposes the full FC x ctrl filtering
grid used by ``state_testing.py``. The headline result in Fig. 3d comes
from the FC0.5_p10 strategy (``config.FC_THRESHOLD`` = 0.5,
``config.CONTROL_PERCENTILE`` = 10).

Inputs
------
``JOINT_ANNDATA_DIR/joint_gbm_oe_anndata.h5ad``
    Must contain ``prob_topic_<Topic_N>`` columns from
    ``classifiers/train_topic_classifier.py``.

Outputs
-------
``TOPIC_TESTING_DIR/<strategy>/topic_effects.csv``
``TOPIC_TESTING_DIR/strategy_comparison_topics.csv``

Usage
-----
``python state_testing/topic_testing.py``
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import scanpy as sc

# Make `from tf_screen import ...` work when this script is run as a file.
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tf_screen import config
from tf_screen.utils import list_cell_lines

from .state_testing import (compare_strategies, filter_cells_by_strategy,
                            test_tf_effects)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    joint_path = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    print(f"Loading joint AnnData from {joint_path}...")
    adata = sc.read_h5ad(joint_path)

    is_atlas = adata.obs["dataset"].astype(str).str.lower().str.contains("gbm", na=False)
    adata_oe = adata[~is_atlas.values].copy()
    print(f"OE (screen) cells: {adata_oe.n_obs:,}")

    oe_obs = adata_oe.obs
    cell_lines = list_cell_lines(adata_oe)

    # Use only the topics on the analysis whitelist. Probability columns
    # for the excluded topics may still be present in obs but are not tested.
    topic_labels = [f"Topic_{t}" for t in config.TOPICS_TESTED]
    print(f"Testing {len(topic_labels)} topics: {', '.join(topic_labels)}")

    fc_grid = [0.5, 1.0, 1.5, 2.0]
    ctrl_grid = ["all", 50, 25, 10]

    out_dir = config.TOPIC_TESTING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t0 = time.time()
    for fc in fc_grid:
        for ctrl in ctrl_grid:
            tag = "all" if ctrl == "all" else f"p{ctrl}"
            strategy = f"FC{fc}_{tag}"
            (out_dir / strategy).mkdir(parents=True, exist_ok=True)
            print(f"\n=== {strategy} ===")

            masks, tfs = filter_cells_by_strategy(adata_oe, oe_obs, fc, ctrl)

            df = test_tf_effects(
                oe_obs, masks, tfs, cell_lines, strategy,
                topic_labels, "prob_topic_", "topic",
            )
            df.to_csv(out_dir / strategy / "topic_effects.csv", index=False)
            all_results.append(df)

            print(f"   topics: {df['sig_consensus'].sum()} sig  "
                  f"({(time.time() - t0)/60:.1f} min elapsed)")

    compare_strategies(all_results, "topics", out_dir)
    print(f"\nDone in {(time.time() - t0)/60:.1f} min.  Output: {out_dir}")


if __name__ == "__main__":
    main()
