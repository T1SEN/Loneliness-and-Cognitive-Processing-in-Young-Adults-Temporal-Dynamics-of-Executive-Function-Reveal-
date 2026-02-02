# Loneliness Article - OSF Submission

This repository contains code, data, and outputs for the loneliness and executive function study. It includes preprocessing, primary analyses, and supplementary analyses used for the manuscript and OSF submission.

## Repository structure

- `data/raw/` : raw exports (input data)
- `data/complete_overall/` : QC-passed dataset used for analyses
- `static/preprocessing/` : preprocessing and QC code
- `static/analysis/` : core analyses and supplementary generators
- `static/wcst_phase/` : WCST phase analyses
- `static/stroop_lmm/` : Stroop trial-level mixed models
- `static/figures_tables/` : figure and table scripts
- `outputs/` : generated figures and statistics
- `doc/` : methods and supplementary materials

## Requirements

- Python 3.11+ recommended
- Install packages:

```
python -m pip install -r requirements.txt
```

## Quick start (reproduce analyses)

From the repository root:

```
python -m static.run_overall_pipeline
```

If `data/complete_overall/` is already built, you can skip preprocessing:

```
python -m static.run_overall_pipeline --skip-preprocess
```

## Key analysis conventions

- All regressions use OLS (non-robust) standard errors.
- Covariates: DASS Depression, Anxiety, Stress, age, gender.
- WCST phase RTs are computed on all trials (errors included) using valid RTs and excluding timeouts.
- Stroop trial-level LMMs are reported in the Supplementary Materials.
- The PRP task is present in the dataset but is not used in the manuscript analyses.

## Outputs

Primary outputs are under:

- `outputs/stats/core/overall/`
- `outputs/figures/core/`

Supplementary outputs are under:

- `outputs/stats/supplementary/overall/`
- `outputs/figures/supplementary/`

## Documentation

- `doc/methods_detailed.md` : detailed methods (English, OSF-ready)
- `doc/supplementary_materials.md` : supplementary results (English, OSF-ready)

## Reproducibility note (S7 in Supplementary)

S7.1 tests general within-task slowing (segment x UCLA) across conditions. The main hypothesis concerns interference drift, which is tested directly in S7.2 via trial_scaled x cond x UCLA. Therefore, a null S7.1 interaction does not contradict a significant S7.2 interaction.
