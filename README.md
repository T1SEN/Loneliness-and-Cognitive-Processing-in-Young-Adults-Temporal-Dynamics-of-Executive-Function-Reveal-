# Loneliness Article - Public Reproduction

This repository is configured for **public-only reproduction** of the manuscript analyses.

## Runtime data contract

All analyses run from `data/public/` and require exactly these files:

- `data/public/demographics_public.csv`
- `data/public/surveys_public.csv`
- `data/public/features_public.csv`
- `data/public/stroop_trials_public.csv`
- `data/public/wcst_trials_public.csv`

Rules enforced at pipeline start:

- every file must include `public_id`
- `public_id` sets must match exactly across all five files
- `demographics_public.csv` must have columns `public_id, gender, age`
- `gender` must normalize to `male` or `female`
- `age` must be numeric

## Install

```bash
python -m pip install -r requirements.txt
```

## Run full reproduction (main + sensitivity + supplementary)

```bash
python -m static.run_overall_pipeline --expected-n 212
```

Notes:

- default run is locked to manuscript common sample (`N=212`)
- to run intentionally with a different sample size:

```bash
python -m static.run_overall_pipeline --expected-n 212 --allow-n-mismatch
```

## Key outputs

- Tables: `outputs/tables/`
- Stats: `outputs/stats/`
- Figures (PNG/PDF): `outputs/figures/`

## Public data generation (deterministic)

If you need to regenerate `data/public/` from restricted source files:

```bash
python -m static.preprocessing.make_public_data --input-dir <restricted_input_dir> --output-dir data/public
```

Generation rules:

- deterministic `public_id` from `participantId` (SHA-256 based)
- no `metadata.json` output
- demographics `gender` normalized to `male/female`

## Repository structure

- `data/public/` public runtime inputs
- `static/` analysis and preprocessing code
- `outputs/` generated results
- `doc/` manuscript support documents
