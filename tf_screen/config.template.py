"""
Configuration for the TF screen analysis pipeline.

Copy this file to ``config.py`` and edit the paths below to point at your
local copies of the input data. ``config.py`` is gitignored; only this
template is tracked.

Inputs you need (download instructions: see ``docs/data_availability.md``):

* The cleaned per-cell AnnData of the screen (raw counts).
* The GB patient multi-ome atlas RNA AnnData.
* Atlas annotation tables (cell-level annotation CSV, topic top-cells TSV,
  and the GBM / TME state-name lookup tables).
* Per-topic gene rankings from scDoRI.
* Per-cell-state marker gene list.

All output paths are derived from ``OUTPUT_DIR``.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Cleaned screen AnnData (raw counts, one row per guide-assigned cell).
SCREEN_ANNDATA = Path("/path/to/gbm_tf_screen_clean.h5ad")

# GB patient multi-ome atlas (RNA).
ATLAS_RNA_ANNDATA = Path("/path/to/atlas/15_05_23_rna.h5ad")

# Atlas cell-level annotations.
ATLAS_ANNOTATION_OBS = Path("/path/to/atlas/annotation_obs.csv")

# Top-5,000 cells per topic, wide format (one column per Topic_N).
ATLAS_TOPIC_TOP_CELLS = Path("/path/to/atlas/top5k_cells_per_topic.tsv")

# Per-topic top-500 gene rankings.
ATLAS_TOPIC_GENES = Path("/path/to/atlas/top_500_genes_per_topic.csv")

# Atlas state-name lookup tables (used to build coarse-state labels).
ANNOT_GBM_TSV = Path("/path/to/atlas/annot_gbm.txt")
ANNOT_TME_TSV = Path("/path/to/atlas/TME_cell_state_map.tsv")

# Per-coarse-state marker gene list (used for marker-score QC).
CELLTYPE_MARKERS = Path("/path/to/celltype_markers_merged.csv")


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("/path/to/output/tf_screen_analysis")

# Sub-directories produced by individual pipeline steps. The scripts create
# these on first use, so there is no need to mkdir them manually.
JOINT_ANNDATA_DIR     = OUTPUT_DIR / "joint_anndata"
HARMONY_DIR           = OUTPUT_DIR / "harmony"
MODELS_DIR            = OUTPUT_DIR / "models"
TF_EXPRESSION_DIR     = OUTPUT_DIR / "tf_expression"
STATE_TESTING_DIR     = OUTPUT_DIR / "state_testing"
TOPIC_TESTING_DIR     = OUTPUT_DIR / "topic_testing"
PROPORTION_DIR        = OUTPUT_DIR / "proportion_testing"
METACELL_DIR          = OUTPUT_DIR / "metacells"
CORRELATION_DIR       = OUTPUT_DIR / "correlation"
DE_DIR                = OUTPUT_DIR / "differential_expression"


# ---------------------------------------------------------------------------
# Cell-state grouping
# ---------------------------------------------------------------------------

# Mapping from coarse GB cell state -> set of fine atlas state labels that
# compose it. Proliferative is excluded entirely: it is a transient cell-
# cycle programme rather than a stable cell-state identity and would
# contaminate the five-state target.
COARSE_GROUPS = {
    "AC-states":       ["AC-gliosis-like", "AC-progenitor-like"],
    "Neuronal-states": ["NPC-neuronal-like", "OPC-neuronal-like"],
    "OPC-states":      ["OPC-like", "OPC-NPC-like"],
    "OPCs":            ["OPCs"],
    "Stress-states":   ["Hypoxic", "Gliosis-like"],
}

# Fine cell states actually used for classification (i.e. the union of the
# values in COARSE_GROUPS, plus any singletons treated separately).
FINE_CELLTYPES = sorted(set(s for v in COARSE_GROUPS.values() for s in v))


# ---------------------------------------------------------------------------
# Topics used in the analysis
# ---------------------------------------------------------------------------

# All GB-intrinsic topics for which a classifier is trained. The classifier
# outputs probabilities over all of these, and they sum to one per cell.
TOPICS_TRAINED = [37, 24, 41, 49, 29, 36, 34, 2, 25, 7, 44, 20,
                  9, 33, 19, 12, 11, 23, 40, 3]

# Topics retained for primary statistical testing (a subset of the above:
# topics 7, 23, 29 are excluded due to unstable classifier performance
# under cross-validation).
TOPICS_TESTED = [37, 24, 41, 49, 36, 34, 2, 25, 44, 20,
                 9, 33, 19, 12, 11, 40, 3]


# ---------------------------------------------------------------------------
# Feature gene selection
# ---------------------------------------------------------------------------

# Number of top genes per topic to take from ATLAS_TOPIC_GENES when
# building the joint feature space. The union across topics, plus the
# 55 TF guide-target gene symbols, defines the gene set used for PCA
# and Harmony.
N_GENES_PER_TOPIC = 500


# ---------------------------------------------------------------------------
# Harmony parameters
# ---------------------------------------------------------------------------

N_PCS               = 200
HARMONY_MAX_ITER    = 30
HARMONY_BATCH_KEY   = "batch"


# ---------------------------------------------------------------------------
# Classifier hyperparameters
# ---------------------------------------------------------------------------

LR_SOLVER           = "lbfgs"
LR_C                = 1.0
LR_CLASS_WEIGHT     = "balanced"
LR_MAX_ITER_STATE   = 4000
LR_MAX_ITER_TOPIC   = 2000
TRAIN_TEST_SPLIT    = 0.2
MAX_CELLS_PER_CLASS = 5000


# ---------------------------------------------------------------------------
# Filtering thresholds for the primary analysis
# ---------------------------------------------------------------------------

# OE cells retained for testing if their linear-space TF expression is at
# least (1 + FC_THRESHOLD) times the geometric mean of control TF
# expression in the matched cell line. FC_THRESHOLD = 0.5 corresponds to
# a 1.5-fold induction (the "FC0.5" strategy in the paper).
FC_THRESHOLD = 0.5

# Control cells retained for testing if their TF expression is at or
# below this percentile of all control cells in the matched cell line.
# CONTROL_PERCENTILE = 10 corresponds to the bottom 10% ("p10" strategy).
CONTROL_PERCENTILE = 10


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

MIN_CELLS_PER_GROUP = 20
N_PERMUTATIONS      = 1000
FDR_THRESHOLD       = 0.05
COHEN_D_THRESHOLD   = 0.2


# ---------------------------------------------------------------------------
# Metacell construction
# ---------------------------------------------------------------------------

LEIDEN_RESOLUTION  = 2.0
METACELL_MIN_CELLS = 10


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
