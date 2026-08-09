# GBM analysis

Analysis code for **Decoding Plasticity Regulators and Transition Trajectories in
Glioblastoma with Single-cell Multiomics** — Saraswat M, Rueda-Gensini L,
Heinzelmann E, Gracia T, Memi T *et al.*

![Dataset overview](dataset.png)

Everything here is downstream of **scDoRI** ([bioFAM/scDoRI](https://github.com/bioFAM/scDoRI)).
No data ships with the code — see [Data availability](#data-availability).

---

## Contents

| | | data it reads |
|---|---|---|
| `TAP_TRS_analysis.ipynb` | Topic activation potential (TAP), topic repression score (TRS), net topic transitions, state-level regulation | `data/` + Zenodo GRN tensors |
| `plot_supp_peak_object.ipynb` | Peak-level panels: peak-topic UMAPs, accessibility per state, TF binding, gene–peak links | Zenodo peak AnnData |
| `plasticity_analysis.ipynb` | Epigenetic plasticity — entropy of ATAC-based cell-state classifier probabilities | ArrayExpress metacell objects |
| `python_scripts/topic_regulation.py` | TAP/TRS implementation ([API](#taptrs-api)) | — |
| `tf_screen/` | 55-TF overexpression screen pipeline, Fig. 3 and ED Fig. 5 — [own README](tf_screen/README.md) | GEO |

---

## TAP/TRS inputs

| file | shape | role |
|---|---|---|
| `data/topic_tf_activator_true_exp_05_03_raw.tsv` | 50 topics × 195 TFs | activator TF activity per topic — defines each topic's regulators |
| `data/topic_tf_repressor_true_exp_05_03_raw.tsv` | 50 topics × 195 TFs | repressor TF activity per topic |
| `data/gene_exp_by_selected_topics_scaled_allTFs.csv` | 195 TFs × 20 topics | mean scaled TF expression per topic |
| `data/gene_score_by_selected_topics_scaled_allTFs.csv` | 195 TFs × 20 topics | mean scaled TF gene activity (ATAC) per topic |
| `data/topic_gene_sel.csv` | 3,192 genes × 31 topics | topic-gene loadings; gene axis order for the GRN tensors |
| Zenodo — ATAC-based activator GRN | 50 × 195 × 3,192 | activating TF→gene links per topic |
| Zenodo — fine-tuned repressor GRN | 50 × 195 × 3,192 | repressing TF→gene links per topic |
| Zenodo — topic activity per coarse state | 5 states × 50 topics | rolls topic-level scores up to cell states |

### TAP/TRS API

`python_scripts/topic_regulation.py`

| function | returns |
|---|---|
| `get_activity_score` | TF activity per topic — summed weight of a TF's links to that topic's target genes |
| `scale_gex_acc_topics` | scaled TF expression and gene activity across topics |
| `compute_topic_regulation_potential` | source × target TAP or TRS matrix, plus the contributing TF→TR links. Optional permutation significance |
| `compute_randomized_topic_regulation` | null background from randomised GRNs |
| `get_net_regulatory_interactions` | net directed interactions, combining TAP and TRS |
| `scale_topic_regulation_target_topic` | scores normalised within target topic |
| `get_tf_topic_regulation` | per-TF decomposition of a topic-pair score |
| `compute_state_level_regulation` | topic-level scores aggregated to cell states |

Significance testing is off by default — enabling it runs 1,000 randomisations
per topic and takes hours.

---

## Data availability

| Archive | Accession | Contents |
|---|---|---|
| Web portal | [www.gbmspace.org](https://www.gbmspace.org) | Interactive browser for the processed multiome atlas |
| ArrayExpress | `E-MTAB-17183` | Processed snRNA-seq and snATAC-seq AnnData; metacell-level RNA counts and ATAC gene scores (plasticity analysis) |
| EGA | `EGAD00001015526` | Raw multiome sequencing data (controlled access) |
| GEO | `GSE294518` | All remaining NGS data, including the TF screen |
| Zenodo | `<ZENODO DOI>` | scDoRI regulatory output + SCENIC+ comparison networks |

Reference genome: hg38 (GRCh38).

**On Zenodo**, with its own `00_README.md`:

- Three scDoRI GRN tensors (ATAC-based activator, fine-tuned activator,
  fine-tuned repressor), each 50 topics × 195 TFs × 3,192 genes, as `.npz` — plus
  the same links as one edge-list table
- Topic-level matrices: topic-gene loadings, TF activity per topic, TF expression
  and gene activity per topic, topic activity per coarse state, topic annotation
- SCENIC+ networks (global + nine per cell type) and per-topic AUCell enrichment
- Peak AnnData: 182,677 peaks × 195 TFs — in-silico ChIP binding, peak-topic
  activity, accessibility per state, peak UMAP, gene–peak links

Each `.npz` stores its own `topics`, `tfs` and `genes` labels next to `grn`, so
axes cannot fall out of sync.

---

## Things that will trip you up

**Two activator GRNs, not interchangeable.** The ATAC-based one is not fine-tuned
against TF–gene expression and is what TAP uses; the fine-tuned one is anchored
to observed expression. Different scales, 45,110 shared links, r = 0.09 on that
support — don't mix or compare their weights.

**Repressor weights are negative.** In both the GRN tensors and the peak object.
Percentile colour limits applied naively will highlight the weakest sites.

**Raw TF binding is not comparable between TFs.** Standardise with the per-TF
mean and sd stored in the peak object before comparing factors.

**Gene–peak links are effectively binary.** Median 0.83, top decile above 0.98 —
treat as set membership, not weight. Summed link weight per gene tracks peak
count and therefore gene length (Spearman 0.98), so it is not comparable between
genes; ranking by it returns long genes. Comparing a gene against itself across
topics is fine. Per-peak argmax assignment is safe — under 2 % of peaks link to
more than one gene.

**SCENIC+ was run on metacells**, not single cells, and the per-cell-type
networks are inferred independently — they are not subsets of the global network
and shared links can differ in direction.

**Topic subsets differ by file.** 50 topics trained, 31 with gene loadings, 20 in
the TF-by-topic matrices, **17 GB tumour topics** for TAP/TRS. The order of the 17
is unsorted and significant — it is the row/column order of the published heat
maps. Topic-set membership is in the Zenodo topic annotation table.

---

## Citation

> <FULL PAPER CITATION>

Please also cite scDoRI ([bioFAM/scDoRI](https://github.com/bioFAM/scDoRI)).

## License

MIT — see [LICENSE](LICENSE). Zenodo data is CC BY 4.0.

## Contact

Open an issue, or contact `manu.saraswat AT dkfz.de`.
