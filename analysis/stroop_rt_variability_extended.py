"""
Stroop RT Variability Extended Analysis
=======================================
Extends basic IIV analysis to condition-specific variability.

Analyses:
1. Congruent trial variability
2. Incongruent trial variability
3. Variability ratio (Incong/Cong)
4. UCLA × Gender on each variability metric
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials


OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/rt_variability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STROOP RT VARIABILITY EXTENDED ANALYSIS")
print("=" * 80)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\n[1] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

demo = master[
    ["participant_id", "gender_normalized", "ucla_total", "dass_depression", "dass_anxiety", "dass_stress"]
].copy()
demo["gender"] = demo["gender_normalized"].fillna("").astype(str).str.strip().str.lower()
demo = demo.dropna(subset=["ucla_total"])

trials, trials_summary = load_stroop_trials(use_cache=True)
rt_col = "rt" if "rt" in trials.columns else "rt_ms" if "rt_ms" in trials.columns else None
if not rt_col:
    raise KeyError("Stroop trials missing rt/rt_ms column")
if rt_col != "rt":
    trials["rt"] = trials[rt_col]

if "type" not in trials.columns:
    for cand in ["condition", "cond"]:
        if cand in trials.columns:
            trials = trials.rename(columns={cand: "type"})
            break
if "type" not in trials.columns:
    raise KeyError("Stroop trials missing condition/type column")

if "is_timeout" not in trials.columns:
    if "timeout" in trials.columns:
        trials["is_timeout"] = trials["timeout"]
    else:
        trials["is_timeout"] = False

trials_clean = trials[
    (trials["is_timeout"] == False)
    & (trials["rt"].notna())
    & (trials["rt"] > 0)
    & (trials["rt"] < 10000)
    & (trials["type"].isin(["congruent", "incongruent"]))
].copy()

print(
    f"  Valid trials: {len(trials_clean):,} "
    f"(n_participants={trials_summary.get('n_participants', trials_clean['participant_id'].nunique())})"
)

# ---------------------------------------------------------------------------
# Condition-specific variability
# ---------------------------------------------------------------------------
print("\n[2] Computing condition-specific variability...")


def compute_variability(group: pd.DataFrame) -> pd.Series:
    metrics = {}
    for cond in ["congruent", "incongruent"]:
        data = group[group["type"] == cond]["rt"]
        if len(data) >= 10:
            metrics[f"{cond}_rt_sd"] = data.std()
            metrics[f"{cond}_rt_cv"] = data.std() / data.mean() if data.mean() > 0 else np.nan
            metrics[f"{cond}_rt_iqr"] = data.quantile(0.75) - data.quantile(0.25)
        else:
            metrics[f"{cond}_rt_sd"] = np.nan
            metrics[f"{cond}_rt_cv"] = np.nan
            metrics[f"{cond}_rt_iqr"] = np.nan
    return pd.Series(metrics)


variability_df = (
    trials_clean.groupby("participant_id")
    .apply(compute_variability)
    .reset_index()
    .merge(
        demo[["participant_id", "gender", "ucla_total", "dass_depression", "dass_anxiety", "dass_stress"]],
        on="participant_id",
        how="inner",
    )
)

variability_df["rt_cv_ratio"] = variability_df["incongruent_rt_cv"] / variability_df["congruent_rt_cv"]

print(f"  Metrics computed for N={len(variability_df)}")
variability_df.to_csv(OUTPUT_DIR / "rt_variability_metrics.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Gender correlations
# ---------------------------------------------------------------------------
print("\n[3] Gender-stratified correlations...")

results = []
for gender in ["male", "female"]:
    data = variability_df[variability_df["gender"] == gender]
    for metric in ["congruent_rt_cv", "incongruent_rt_cv", "rt_cv_ratio"]:
        valid = data.dropna(subset=["ucla_total", metric])
        if len(valid) >= 10:
            r, p = pearsonr(valid["ucla_total"], valid[metric])
            results.append({"gender": gender, "metric": metric, "n": len(valid), "r": r, "p": p})

gender_corr_df = pd.DataFrame(results)
print(gender_corr_df)
gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Simple regression examples (optional)
# ---------------------------------------------------------------------------
if len(variability_df) >= 20:
    variability_df = variability_df.dropna(subset=["rt_cv_ratio", "ucla_total"])
    variability_df["gender_male"] = (variability_df["gender"] == "male").astype(int)
    model = smf.ols("rt_cv_ratio ~ ucla_total + gender_male + ucla_total:gender_male", data=variability_df).fit()
    with open(OUTPUT_DIR / "rt_cv_ratio_regression.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STROOP RT VARIABILITY EXTENDED - SUMMARY")
print("=" * 80)

print(
    f"""
Metrics saved to: {OUTPUT_DIR / 'rt_variability_metrics.csv'}
Gender correlations saved to: {OUTPUT_DIR / 'gender_stratified_correlations.csv'}
Regression (if run) saved to: {OUTPUT_DIR / 'rt_cv_ratio_regression.txt'}
"""
)

print("\n" + "=" * 80)
print("Stroop RT variability extended complete!")
print("=" * 80)
