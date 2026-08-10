"""Drive ``downstream/verify_deposit_against_fig3c.py`` on the toy dataset.

The verify script hardcodes the deposit + reference paths on Manu's
cluster. For the toy run we just monkey-patch ``Config`` to point at the
toy joint AnnData (which already has the ``prob_coarse_*`` columns from
the toy ``train_state_classifier.py`` run) and at the FC0.5_p10
``coarse_effects.csv`` that the toy ``state_testing.py`` run produced.

Comparing the script against its own output should give ~100% agreement —
the goal here is to confirm the script runs without errors, not to test
the numerical claim about the deposit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tf_screen import config
from tf_screen.downstream import verify_deposit_against_fig3c as vd


def main() -> None:
    vd.Config.DEPOSIT = config.JOINT_ANNDATA_DIR / "joint_gbm_oe_anndata.h5ad"
    vd.Config.REFERENCE = (config.STATE_TESTING_DIR / "FC0.5_p10"
                           / "coarse_effects.csv")
    vd.Config.OUT_DIR = config.OUTPUT_DIR / "verify_deposit_output"
    # Toy data has small per-(TF, line) counts; keep the threshold modest so
    # the test isn't entirely NaN. (The real-data threshold is 10.)
    vd.Config.MIN_CELLS = max(5, config.MIN_CELLS_PER_GROUP)
    vd.Config.N_PERMS   = config.N_PERMUTATIONS

    print(f"   DEPOSIT   -> {vd.Config.DEPOSIT}")
    print(f"   REFERENCE -> {vd.Config.REFERENCE}")
    print(f"   OUT_DIR   -> {vd.Config.OUT_DIR}")

    vd.main()


if __name__ == "__main__":
    main()
