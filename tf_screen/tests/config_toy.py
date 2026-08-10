"""Config override that points the TF screen pipeline at the synthetic
toy dataset under ``tf_screen/tests/toy_data/``.

Activated by writing ``tf_screen/config.py`` with::

    from tf_screen.tests.config_toy import *  # noqa: F401,F403

so the rest of the pipeline picks it up without code changes.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Inputs (toy data)
# ---------------------------------------------------------------------------

SCREEN_ANNDATA        = _HERE / "toy_data" / "gbm_tf_screen_clean.h5ad"
ATLAS_RNA_ANNDATA     = _HERE / "toy_data" / "joint_gbm_oe_anndata.h5ad"
ATLAS_ANNOTATION_OBS  = _HERE / "toy_data" / "atlas_annotation_obs.csv"
ATLAS_TOPIC_TOP_CELLS = _HERE / "toy_data" / "top5k_cells_per_topic.tsv"
ATLAS_TOPIC_GENES     = _HERE / "toy_data" / "top_500_genes_per_topic.csv"
ANNOT_GBM_TSV         = _HERE / "toy_data" / "annot_gbm.txt"
ANNOT_TME_TSV         = _HERE / "toy_data" / "TME_cell_state_map.tsv"
CELLTYPE_MARKERS      = _HERE / "toy_data" / "celltype_markers_merged.csv"


# ---------------------------------------------------------------------------
# Output root (toy)
# ---------------------------------------------------------------------------

OUTPUT_DIR        = _HERE / "toy_output"
JOINT_ANNDATA_DIR = _HERE / "toy_data"             # pre-built joint AnnData lives here
HARMONY_DIR       = _HERE / "toy_data"             # pre-built Harmony embedding
MODELS_DIR        = OUTPUT_DIR / "models"
TF_EXPRESSION_DIR = OUTPUT_DIR / "tf_expression"
STATE_TESTING_DIR = OUTPUT_DIR / "state_testing"
TOPIC_TESTING_DIR = OUTPUT_DIR / "topic_testing"
PROPORTION_DIR    = OUTPUT_DIR / "proportion_testing"
METACELL_DIR      = OUTPUT_DIR / "metacells"
CORRELATION_DIR   = OUTPUT_DIR / "correlation"
DE_DIR            = OUTPUT_DIR / "differential_expression"


# ---------------------------------------------------------------------------
# Cell-state grouping (identical to template)
# ---------------------------------------------------------------------------

COARSE_GROUPS = {
    "AC-states":       ["AC-gliosis-like", "AC-progenitor-like"],
    "Neuronal-states": ["NPC-neuronal-like", "OPC-neuronal-like"],
    "OPC-states":      ["OPC-like", "OPC-NPC-like"],
    "OPCs":            ["OPCs"],
    "Stress-states":   ["Hypoxic", "Gliosis-like"],
}
FINE_CELLTYPES = sorted({s for v in COARSE_GROUPS.values() for s in v})


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

TOPICS_TRAINED = [37, 24, 41, 49, 29, 36, 34, 2, 25, 7, 44, 20,
                  9, 33, 19, 12, 11, 23, 40, 3]
TOPICS_TESTED  = [37, 24, 41, 49, 36, 34, 2, 25, 44, 20,
                  9, 33, 19, 12, 11, 40, 3]


# ---------------------------------------------------------------------------
# Feature gene selection / Harmony / classifier hyper-params
# ---------------------------------------------------------------------------

N_GENES_PER_TOPIC = 500

# Toy Harmony embedding was saved with 200 PCs; keep the same value so
# downstream scripts can use the embedding as-is. (We bypass run_harmony.)
N_PCS               = 200
HARMONY_MAX_ITER    = 30
HARMONY_BATCH_KEY   = "batch"

LR_SOLVER           = "lbfgs"
LR_C                = 1.0
LR_CLASS_WEIGHT     = "balanced"
LR_MAX_ITER_STATE   = 4000
LR_MAX_ITER_TOPIC   = 2000
TRAIN_TEST_SPLIT    = 0.2
MAX_CELLS_PER_CLASS = 5000


# ---------------------------------------------------------------------------
# Filtering / statistical testing — tuned smaller for toy run
# ---------------------------------------------------------------------------

FC_THRESHOLD       = 0.5
CONTROL_PERCENTILE = 10

# Toy data has small per-TF counts; lower the floors / perms accordingly.
MIN_CELLS_PER_GROUP = 5
N_PERMUTATIONS      = 100
FDR_THRESHOLD       = 0.05
COHEN_D_THRESHOLD   = 0.2


# ---------------------------------------------------------------------------
# Metacell construction — coarser Leiden on small data
# ---------------------------------------------------------------------------

LEIDEN_RESOLUTION  = 0.5
METACELL_MIN_CELLS = 10


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
