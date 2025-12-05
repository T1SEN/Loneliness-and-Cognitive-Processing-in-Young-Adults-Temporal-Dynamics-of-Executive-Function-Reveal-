"""
RT Percentile Group Comparison Analysis
========================================
Tests UCLA × Gender effects on participant-level RT percentiles (10th, 25th, 50th, 75th, 90th).

Method:
1) Compute participant RT percentiles from trial data (PRP/Stroop/WCST)
2) OLS: percentile ~ UCLA × Gender + DASS + Age (per task/condition)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials, load_stroop_trials, load_wcst_trials

warnings.filterwarnings("ignore")
np.random.seed(42)

MIN_N_REGRESSION = 30

OUTPUT_DIR = Path("results/analysis_outputs/rt_percentile_group_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("RT PERCENTILE GROUP COMPARISON ANALYSIS")
print("=" * 80)
print("\nPurpose: Test UCLA × Gender effects on participant RT percentiles")
print("Method: Compute percentiles → OLS group comparison")
print("Note: This is NOT conditional quantile regression\n")

# ---------------------------------------------------------------------------
# Load covariates from master
# ---------------------------------------------------------------------------
print("Loading participant covariates...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

covariates = master[
    ["participant_id", "ucla_total", "dass_depression", "dass_anxiety", "dass_stress", "age", "gender_normalized"]
].rename(columns={"gender_normalized": "gender"})
covariates["gender"] = covariates["gender"].fillna("").astype(str).str.strip().str.lower()
covariates["gender_male"] = (covariates["gender"] == "male").astype(int)

scaler = StandardScaler()
covariates["z_age"] = scaler.fit_transform(covariates[["age"]])
covariates["z_ucla"] = scaler.fit_transform(covariates[["ucla_total"]])
covariates["z_dass_dep"] = scaler.fit_transform(covariates[["dass_depression"]])
covariates["z_dass_anx"] = scaler.fit_transform(covariates[["dass_anxiety"]])
covariates["z_dass_str"] = scaler.fit_transform(covariates[["dass_stress"]])

print(f"  Loaded covariates for {len(covariates)} participants")

quantiles_to_compute = [0.10, 0.25, 0.50, 0.75, 0.90]
all_quantile_data = []

# ---------------------------------------------------------------------------
# PRP quantiles (T2 RT by SOA bin)
# ---------------------------------------------------------------------------
print("\nComputing PRP T2_RT quantiles...")
prp_trials, prp_summary = load_prp_trials(use_cache=True, rt_min=200, rt_max=5000, require_t1_correct=False, enforce_short_long_only=False)

def categorize_soa(soa):
    if pd.isna(soa):
        return "other"
    if soa <= 150:
        return "short"
    if 300 <= soa <= 600:
        return "medium"
    if soa >= 1200:
        return "long"
    return "other"

prp_trials["soa_cat"] = prp_trials["soa"].apply(categorize_soa)
for soa_condition in ["short", "long"]:
    prp_soa = prp_trials[
        (prp_trials["soa_cat"] == soa_condition)
        & prp_trials["t2_rt"].notna()
        & prp_trials["t2_rt"].between(200, 5000)
    ]
    if len(prp_soa) < 100:
        print(f"  Skipping PRP {soa_condition} SOA (insufficient trials)")
        continue

    quantile_df = prp_soa.groupby("participant_id")["t2_rt"].quantile(quantiles_to_compute).unstack().reset_index()
    quantile_df.columns = ["participant_id"] + [f"q_{int(q*100)}" for q in quantiles_to_compute]
    quantile_df = quantile_df.merge(covariates, on="participant_id", how="inner")
    quantile_df["task"] = "PRP"
    quantile_df["condition"] = soa_condition
    all_quantile_data.append(quantile_df)
    print(f"  PRP {soa_condition} SOA: {len(quantile_df)} participants")

# ---------------------------------------------------------------------------
# WCST quantiles (overall RT)
# ---------------------------------------------------------------------------
print("\nComputing WCST RT quantiles...")
wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)
rt_col_wcst = "reactionTimeMs" if "reactionTimeMs" in wcst_trials.columns else "rt_ms" if "rt_ms" in wcst_trials.columns else None
if not rt_col_wcst:
    raise KeyError("WCST trials missing reaction time column")

wcst_clean = wcst_trials[wcst_trials[rt_col_wcst].between(200, 5000)].copy()
if len(wcst_clean) >= 100:
    quantile_df = wcst_clean.groupby("participant_id")[rt_col_wcst].quantile(quantiles_to_compute).unstack().reset_index()
    quantile_df.columns = ["participant_id"] + [f"q_{int(q*100)}" for q in quantiles_to_compute]
    quantile_df = quantile_df.merge(covariates, on="participant_id", how="inner")
    quantile_df["task"] = "WCST"
    quantile_df["condition"] = "overall"
    all_quantile_data.append(quantile_df)
    print(f"  WCST: {len(quantile_df)} participants")

# ---------------------------------------------------------------------------
# Stroop quantiles (incongruent trials)
# ---------------------------------------------------------------------------
print("\nComputing Stroop RT quantiles (incongruent trials)...")
stroop_trials, stroop_summary = load_stroop_trials(use_cache=True)
rt_col_stroop = "rt" if "rt" in stroop_trials.columns else "rt_ms" if "rt_ms" in stroop_trials.columns else None
if not rt_col_stroop:
    raise KeyError("Stroop trials missing rt/rt_ms column")
if rt_col_stroop != "rt":
    stroop_trials["rt"] = stroop_trials[rt_col_stroop]

if "type" not in stroop_trials.columns:
    for cand in ["condition", "cond"]:
        if cand in stroop_trials.columns:
            stroop_trials = stroop_trials.rename(columns={cand: "type"})
            break
stroop_clean = stroop_trials[
    (stroop_trials["type"].str.lower() == "incongruent")
    & stroop_trials["rt"].notna()
    & stroop_trials["rt"].between(200, 5000)
].copy()

if len(stroop_clean) >= 100:
    quantile_df = stroop_clean.groupby("participant_id")["rt"].quantile(quantiles_to_compute).unstack().reset_index()
    quantile_df.columns = ["participant_id"] + [f"q_{int(q*100)}" for q in quantiles_to_compute]
    quantile_df = quantile_df.merge(covariates, on="participant_id", how="inner")
    quantile_df["task"] = "Stroop"
    quantile_df["condition"] = "incongruent"
    all_quantile_data.append(quantile_df)
    print(f"  Stroop incongruent: {len(quantile_df)} participants")

if not all_quantile_data:
    print("\nERROR: No quantile data computed. Check trial data.")
    sys.exit(1)

combined_df = pd.concat(all_quantile_data, ignore_index=True)
print(f"\nTotal datasets: {len(all_quantile_data)}")
print(f"Combined: {len(combined_df)} participant×task rows")

# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Quantile Regression Models (OLS)")
print("=" * 80)

quantile_results = []
for task_name in combined_df["task"].unique():
    for condition in combined_df[combined_df["task"] == task_name]["condition"].unique():
        subset = combined_df[(combined_df["task"] == task_name) & (combined_df["condition"] == condition)].copy()
        if len(subset) < MIN_N_REGRESSION:
            continue

        print(f"\n{task_name} - {condition}:")
        print("-" * 60)
        print(f"  N = {len(subset)}")

        for q in quantiles_to_compute:
            q_col = f"q_{int(q*100)}"
            df_q = subset.dropna(subset=[q_col, "z_ucla", "gender_male", "z_dass_dep", "z_dass_anx", "z_dass_str", "z_age"])
            if len(df_q) < MIN_N_REGRESSION:
                continue

            df_q["z_rt"] = scaler.fit_transform(df_q[[q_col]])
            formula = "z_rt ~ z_ucla * gender_male + z_dass_dep + z_dass_anx + z_dass_str + z_age"
            model = smf.ols(formula, data=df_q).fit()

            quantile_results.append(
                {
                    "task": task_name,
                    "condition": condition,
                    "quantile": q,
                    "n": len(df_q),
                    "beta_ucla": model.params.get("z_ucla", np.nan),
                    "p_ucla": model.pvalues.get("z_ucla", np.nan),
                    "beta_gender": model.params.get("gender_male", np.nan),
                    "p_gender": model.pvalues.get("gender_male", np.nan),
                    "beta_interaction": model.params.get("z_ucla:gender_male", np.nan),
                    "p_interaction": model.pvalues.get("z_ucla:gender_male", np.nan),
                    "adj_r2": model.rsquared_adj,
                }
            )

quantile_df = pd.DataFrame(quantile_results)
quantile_df.to_csv(OUTPUT_DIR / "rt_percentile_models.csv", index=False, encoding="utf-8-sig")
combined_df.to_csv(OUTPUT_DIR / "rt_percentile_data.csv", index=False, encoding="utf-8-sig")

print("\nModel summaries saved:")
print(f"  - {OUTPUT_DIR / 'rt_percentile_models.csv'}")
print(f"  - {OUTPUT_DIR / 'rt_percentile_data.csv'}")

print("\n" + "=" * 80)
print("RT percentile analysis complete!")
print("=" * 80)
