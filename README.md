# Loneliness Article - OSF Submission

This repository contains code, data, and outputs for the loneliness and executive function study. It includes preprocessing, primary analyses, and supplementary analyses used for the manuscript and OSF submission.

## Repository structure

- `data/raw/` : raw exports (input data)
- `data/complete_overall/` : QC-passed dataset used for analyses
- `lib/` : data collection code (source app/scripts)
- `static/preprocessing/` : preprocessing and QC code
- `static/analysis/` : core analyses and supplementary generators
- `static/wcst_phase/` : WCST phase analyses
- `static/stroop_lmm/` : Stroop trial-level mixed models
- `static/stroop_supplementary/` : Stroop supplementary analyses
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

## OSF submission (detailed guide)

This section describes how to prepare a **public OSF package** for this project.
It is written to maximize reproducibility while minimizing re-identification risk.

### OSF readiness (current repo)

Status checklist:

- [x] Analysis pipeline and outputs reproducible (`static/run_overall_pipeline.py`)
- [x] Methods and supplementary docs present (`doc/`)
- [ ] `LICENSE` file missing
- [ ] `CITATION.cff` file missing
- [ ] `data/public/` not created yet
- [ ] Data dictionary not created yet

Known sensitive fields in current data (do **not** make public as-is):

- `data/complete_overall/1_participants_info.csv` includes `studentId`, `birthDate`, `courseName`, `professorName`, `classSection`, `createdAt`
- `data/complete_overall/3_cognitive_tests_summary.csv` includes `timestamp`
- `data/complete_overall/4b_wcst_trials.csv` includes `timestamp`

### Public vs. restricted content

**Recommended public content**

- `README.md`
- `requirements.txt`
- `doc/`
- `static/`
- `outputs/` (derived results only)
- `data/public/` (de-identified; see below)

**Recommended restricted/private content**

- `data/raw/`
- `data/complete_overall/1_participants_info.csv`
- Any file containing direct identifiers or exact timestamps

Direct identifiers to remove or mask include:

- `studentId`
- `birthDate`
- `courseName`
- `professorName`
- `classSection`
- `createdAt`

### De-identification checklist (public data)

- Remove all direct identifiers listed above.
- Replace `participantId` with a random, non-reversible ID (one-way mapping stored privately).
- Remove exact timestamps; if necessary, coarsen to date or study week.
- Bin `age` (e.g., 18-19, 20-21, 22-24, 25+), especially for small cells.
- Remove any free-text fields.
- Verify that no table or output exposes very small cells (e.g., n < 5).

### Suggested public data layout

Create a public-only folder (do not include raw data):

```
data/public/
  participants_public.csv
  surveys_public.csv
  features_public.csv
  stroop_trials_public.csv        # optional (trial-level)
  wcst_trials_public.csv          # optional (trial-level)
  metadata.json                   # schema + units + missing codes
```

### Codebook / data dictionary

Add a data dictionary with:

- column name
- data type
- units / scale
- valid ranges
- missingness codes
- derivation notes for computed variables

Recommended location:

- `doc/data_dictionary.md` (or `data/public/codebook.csv`)

### Reproducibility steps (public package)

1. Create a clean environment and install dependencies:

```
python -m pip install -r requirements.txt
```

2. Run the full pipeline from repository root:

```
python -m static.run_overall_pipeline
```

3. Verify that expected outputs exist:

- `outputs/stats/core/overall/`
- `outputs/figures/core/`
- `outputs/stats/supplementary/overall/`
- `outputs/figures/supplementary/`

4. (Optional) Record file hashes for OSF integrity checks:

```
Get-FileHash outputs/stats/core/overall/* | Format-Table -AutoSize
```

### OSF component map (example)

- **Code and documentation**: `README.md`, `requirements.txt`, `static/`, `doc/`
- **Derived outputs**: `outputs/`
- **Public data (de-identified)**: `data/public/`
- **Restricted raw data**: `data/raw/` (private component or not uploaded)

### Citation and license (recommended)

Add the following to the repository for OSF compliance:

- `LICENSE` (e.g., CC-BY 4.0 for documents, MIT for code)
- `CITATION.cff` (preferred) or a citation block in `README.md`

### Data use / ethics statement (recommended)

Include a brief statement indicating:

- data are de-identified
- only de-identified data are publicly shared
- raw data are restricted or not shared

## Reproducibility note (S7 in Supplementary)

S7.1 tests general within-task slowing (segment x UCLA) across conditions. The main hypothesis concerns interference drift, which is tested directly in S7.2 via trial_scaled x cond x UCLA. Therefore, a null S7.1 interaction does not contradict a significant S7.2 interaction.
