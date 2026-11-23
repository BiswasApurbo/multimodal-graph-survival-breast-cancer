# Multimodal Graph-Based Survival Learning for Breast Cancer Prognosis

Code and results for the paper:

> M. Momo, A. Biswas, S. Tasnim, I. Tabassum, T. Sultan, M.F. Mridha, M.R. Islam.
> *Multimodal Graph-Based Survival Learning for Breast Cancer Prognosis Using Clinical, Transcriptomic, and Genomic Data.* Array (Elsevier), under revision (ARRAY-D-26-06140).

The framework builds one inductive k-nearest-neighbour patient graph per modality (clinical, RNA expression, somatic mutation). Each graph is encoded with GraphSAGE, and the modality embeddings are fused with a Transformer encoder under a Cox partial-likelihood loss (**DualGraphSurvival**, **TripleGraphSurvival**). It is evaluated against matched controls: the same network without edges, a shuffled graph and a single unified graph.

**Main finding:** under a leakage-free protocol, the graph adds no survival discrimination beyond the node features, for METABRIC (internal) or TCGA-BRCA (external). The patient graphs largely re-encode ER status and PAM50 subtype.

---

## Repository contents

```
.
├── ARRAY_Revision_Experiments.ipynb   # all experiments (E0–E7) reported in the paper
├── make_figures.py                    # manuscript Figures 3–8 from the result folders
├── results/
│   ├── Revision_R1/                   # overall survival (OS), primary endpoint
│   └── Revision_R1_DSS/               # disease-specific survival (DSS), sensitivity analysis
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Data

The data are not redistributed here. All sources are public:

| Cohort | Source | Files used |
|---|---|---|
| METABRIC (Curtis et al., *Nature* 2012, doi:10.1038/nature10983; EGA EGAS00000000083) | Public export of the cBioPortal METABRIC data: <https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric> | `METABRIC_RNA_Mutation.csv` |
| TCGA-BRCA PanCancer Atlas | cBioPortal study `brca_tcga_pan_can_atlas_2018`: <https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018> | `data_clinical_patient.txt`, `data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt`, `data_mutations.txt` |

Put all four files in one folder and set `DATA_DIR` in the CONFIG cell of the notebook. In the paper, TCGA mutations were retrieved through the cBioPortal web API on 23 November 2026 (protein-altering variants only).

## How to run

The notebook was developed in Google Colab with a GPU runtime. Running it locally works as well: remove the `google.colab` / `drive.mount` lines and set `DATA_DIR` and `OUT_DIR` to local paths.

1. Install the dependencies (`pip install -r requirements.txt`; install PyTorch Geometric for your CUDA version as described at <https://pytorch-geometric.readthedocs.io>).
2. In the CONFIG cell, set `QUICK = True` and run all cells to check the setup (a few minutes).
3. Set `QUICK = False`, `ENDPOINT = "OS"`, `OUT_DIR = ".../Revision_R1"` and run all cells (about 1–2 h on a GPU).
4. For the DSS sensitivity analysis, set `ENDPOINT = "DSS"`, `OUT_DIR = ".../Revision_R1_DSS"`, restart the runtime and run all cells again.
5. Run `make_figures.py` (set `BASE` at the top) to produce Figures 3–8 in `Revision_R1/manuscript_figures/`.

Seeds: 42, 123, 456, 789, 2025, 7, 11, 21, 99, 314. The library versions used for the paper are in `results/Revision_R1/versions.json`, and the full configuration in `config.json`.

## Protocol in brief

- **Endpoint:** overall survival (death from any cause), in months; DSS as a secondary endpoint.
- **Splits:** 10 stratified 70/15/15 train/validation/test splits of METABRIC (1,332 / 286 / 286). The validation set is used only for early stopping and tuning; all reported internal numbers are on the test sets.
- **No leakage:** imputation, scaling and feature selection (top 100 genes and top 40 mutations by univariate Cox score test) are fitted on the training patients of each split only.
- **Inductive graphs:** k-NN (k = 10) among training patients only; held-out and external patients receive edges from their k nearest training patients and send none.
- **External evaluation:** METABRIC-trained models with harmonised features (age, stage, 100 shared genes, 40 mutations) applied to 996 mutation-profiled TCGA patients; TCGA-trained models are compared on the same TCGA test patients.

## Results files → paper

| Paper | Experiment | File(s) in `results/Revision_R1/` (and `Revision_R1_DSS/`) |
|---|---|---|
| Table 2 (cohorts) | Summary | `cohort_summary.csv` |
| Table 3 (features) | Summary | `feature_dimensions_and_splits.csv`, `selected_genes_seed42.csv`, `selected_mutations_seed42.csv` |
| Table 6 (graph statistics) | E5 | `e5_graph_statistics.csv` |
| Table 7 (selected hyper-parameters) | E0 | `e0_hparam_search.csv`, `e0_selected_hparams.json` |
| Table 9 (main comparison) | E1 | `e1_internal_runs.csv`, `e1_summary_mean_sd.csv`, `e1_bootstrap_ci_seed42.csv` |
| Table 10 (paired tests) | E1 | `e1_paired_tests.csv` |
| Table 11, Figs 4–5 (graph vs no-graph ablation) | E2 | `e2_ablation_runs.csv`, `e2_ablation_summary.csv`, `e2_ablation_paired_tests.csv` |
| Table 12 (sensitivity: k, metric, aggregation, fusion, node features) | E3, E4 | `e3_sensitivity_*.csv`, `e4_fusion_aggregation_*.csv` |
| Table 13 (subtype analyses), Fig 8 (calibration, decision curves) | E6 | `e6_*.csv` |
| Table 14, Fig 6 (cross-cohort) | E7 | `e7_cross_cohort_runs.csv`, `e7_cross_cohort_summary.csv`, `e7_mutation_imputation_sensitivity.csv`, `mutation_frequency_by_cohort.csv` |
| Fig 7 (Kaplan–Meier, hazard ratios) | E1, E7 | `km_logrank_hr.csv`, `e1_predictions_seed42.npz` |

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The archived version of this repository has a Zenodo DOI (badge to be added after the first release).

## License

Code: MIT License (see `LICENSE`). The data remain under the terms of their original providers.

## Contact

Md. Rakibul Islam (corresponding author), md.rakibul.islam@mdu.se
