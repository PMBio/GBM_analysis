# Cluster submission scripts

LSF job submission templates that the original analyses ran under. They're shipped as reference for anyone reproducing the pipeline on an LSF-based HPC.

## Files

| Script | Step | Job |
|---|---|---|
| `submit_preprocessing.sh` | 1 | Builds joint AnnData, runs Harmony |
| `submit_classifiers.sh` | 2 | Trains state + topic classifiers |
| `submit_state_testing.sh` | 4 | State + topic + proportion testing (the heaviest job, ~24h) |
| `submit_dose_response.sh` | 5 | Builds metacells + runs correlation analysis |
| `submit_de.sh` | 6 | Per-line Wilcoxon DE (all four strategies) |

Each script declares typical memory and time requirements based on what the original runs needed. Adjust for your cluster.

## Notes

* The original cluster used `conda activate LINGER`. Replace with the name of your conda env (or `source ./venv/bin/activate` if using a venv).
* Output paths are read from `config.py`. Set `OUTPUT_DIR` to somewhere on the cluster's scratch space before submitting.
* The DE job for all four strategies writes ~50 GB. If disk is limited, restrict to the primary `all_all` strategy: change `python differential_expression/wilcoxon_de.py` to `python differential_expression/wilcoxon_de.py all_all`.
* For Slurm or other schedulers, adapt the `#BSUB` directives. The shape of the workflow (one job per pipeline step) is unchanged.
