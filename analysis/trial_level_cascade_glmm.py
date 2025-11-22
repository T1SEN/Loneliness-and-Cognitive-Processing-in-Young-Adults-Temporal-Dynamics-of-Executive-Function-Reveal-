"""
trial_level_cascade_glmm.py

Trial-level mixed-effects models for error cascade and post-error slowing (PES).
Tests whether cascades/PES are moderated by UCLA × Gender with DASS controls.

Outputs:
- cascade_glmm_results.csv
- pes_glmm_results.csv
- cascade_by_previous_error.png
"""

import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analysis.utils.data_loader_utils import load_master_dataset, PRP_RT_MAX
from analysis.utils.trial_data_loader import load_prp_trials

warnings.filterwarnings("ignore")

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive/trial_cascade_glmm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TRIAL-LEVEL ERROR CASCADE & POST-ERROR SLOWING (GLMM)")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# 1. LOAD TRIAL-LEVEL DATA
# ---------------------------------------------------------------------------
print("Loading trial data...")

prp_trials, prp_summary = load_prp_trials(
    use_cache=True,
    rt_min=200,
    rt_max=PRP_RT_MAX,
    require_t1_correct=False,          # keep T1 errors for cascade computation
    enforce_short_long_only=False,     # retain medium/other SOA bins
    require_t2_correct_for_rt=False,   # keep T2 errors for cascade/PES flags
    drop_timeouts=True,                # remove timeouts (no behavioral response)
)

if "trial_index" not in prp_trials.columns:
    for cand in ["trialIndex", "trial"]:
        if cand in prp_trials.columns:
            prp_trials = prp_trials.rename(columns={cand: "trial_index"})
            break
if "trial_index" not in prp_trials.columns:
    raise KeyError("PRP trials missing trial index column")

if "t2_correct" not in prp_trials.columns:
    raise KeyError("PRP trials missing t2_correct column")

print(
    f"  PRP trials loaded: {len(prp_trials):,} rows "
    f"(n_participants={prp_summary.get('n_participants', prp_trials['participant_id'].nunique())})"
)

# ---------------------------------------------------------------------------
# 2. LOAD PARTICIPANT-LEVEL PREDICTORS
# ---------------------------------------------------------------------------
print("Loading participant predictors...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

master = master.rename(columns={"gender_normalized": "gender"})
master["gender"] = master["gender"].fillna("").astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()

master["z_ucla"] = zscore(master["ucla_total"])
master["z_dass_dep"] = zscore(master["dass_depression"])
master["z_dass_anx"] = zscore(master["dass_anxiety"])
master["z_dass_str"] = zscore(master["dass_stress"])
if "age" in master.columns:
    master["z_age"] = zscore(master["age"])

print(f"  Participant predictors: {len(master)} participants")
print()

# ---------------------------------------------------------------------------
# 3. PREPARE TRIAL DATA
# ---------------------------------------------------------------------------
print("Preparing trial-level data...")

prp_clean = prp_trials[
    (prp_trials["t1_correct"].notna())
    & (prp_trials["t2_correct"].notna())
    & (prp_trials["t2_rt"].notna())
    & (prp_trials["t2_rt"] > 200)
    & (prp_trials["t2_rt"] < PRP_RT_MAX)
    & (prp_trials["soa"].notna())
].copy()

if prp_clean["t1_correct"].dtype != bool:
    prp_clean["t1_correct"] = prp_clean["t1_correct"].astype(bool)
if prp_clean["t2_correct"].dtype != bool:
    prp_clean["t2_correct"] = prp_clean["t2_correct"].astype(bool)

prp_clean["error_cascade"] = (~prp_clean["t1_correct"]) & (~prp_clean["t2_correct"])
prp_clean["error_cascade_int"] = prp_clean["error_cascade"].astype(int)

prp_clean = prp_clean.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)

prp_clean["t2_error_prev"] = prp_clean.groupby("participant_id")["t2_correct"].shift(1).fillna(True)
prp_clean["t2_error_prev"] = (~prp_clean["t2_error_prev"]).astype(int)

def categorize_soa(soa_val: float) -> str:
    if soa_val <= 150:
        return "short"
    if 300 <= soa_val <= 600:
        return "medium"
    if soa_val >= 1200:
        return "long"
    return "other"

prp_clean["soa_category"] = prp_clean["soa"].apply(categorize_soa)
prp_clean = prp_clean[prp_clean["soa_category"].isin(["short", "medium", "long"])].copy()

prp_clean["soa_scaled"] = zscore(prp_clean["soa"])

merge_cols = ["participant_id", "z_ucla", "z_dass_dep", "z_dass_anx", "z_dass_str", "gender_male"]
if "z_age" in master.columns:
    merge_cols.append("z_age")
prp_clean = prp_clean.merge(master[merge_cols], on="participant_id", how="left")

prp_clean = prp_clean.dropna(subset=["z_ucla", "gender_male", "z_dass_dep", "z_dass_anx", "z_dass_str"])

print(f"  Clean trial data: {len(prp_clean):,} trials")
print(f"  Participants: {prp_clean['participant_id'].nunique()}")
print(f"  Error cascade rate: {prp_clean['error_cascade_int'].mean():.3f}")
print()

# ---------------------------------------------------------------------------
# 4. MODEL 1: ERROR CASCADE GLMM (Logistic Mixed Model)
# ---------------------------------------------------------------------------
print("=" * 80)
print("MODEL 1: ERROR CASCADE (Logistic GLMM)")
print("=" * 80)
print()

print("Fitting logistic mixed model...")
print("Formula: error_cascade ~ t2_error_prev + z_ucla + gender_male + soa_scaled + DASS + (1|participant)")
print()

cascade_results = None
try:
    formula_cascade = (
        "error_cascade_int ~ t2_error_prev + z_ucla + C(gender_male) + "
        "soa_scaled + z_dass_dep + z_dass_anx + z_dass_str"
    )

    if len(prp_clean) > 3000:
        prp_sample = prp_clean.sample(n=3000, random_state=42)
        print(f"  Subsampling to {len(prp_sample)} trials for computational efficiency")
    else:
        prp_sample = prp_clean

    model_cascade = smf.mixedlm(
        formula_cascade,
        data=prp_sample,
        groups=prp_sample["participant_id"],
        re_formula="~1",
    )

    result_cascade = model_cascade.fit(method="lbfgs", maxiter=100)

    print(result_cascade.summary())
    print()

    cascade_results = pd.DataFrame(
        {
            "parameter": result_cascade.params.index,
            "coefficient": result_cascade.params.values,
            "std_err": result_cascade.bse.values,
            "z_value": result_cascade.tvalues.values,
            "p_value": result_cascade.pvalues.values,
        }
    )

    cascade_results.to_csv(OUTPUT_DIR / "cascade_glmm_results.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DIR / 'cascade_glmm_results.csv'}")
    print()

except Exception as e:
    print(f"  ERROR fitting cascade GLMM: {e}")
    print("  Skipping cascade model")

print()

# ---------------------------------------------------------------------------
# 5. MODEL 2: POST-ERROR SLOWING (Linear Mixed Model)
# ---------------------------------------------------------------------------
print("=" * 80)
print("MODEL 2: POST-ERROR SLOWING (Linear GLMM)")
print("=" * 80)
print()

print("Fitting linear mixed model for T2 RT...")
print("Formula: t2_rt ~ t2_error_prev × ucla × gender + SOA + DASS + (1|participant)")
print()

pes_results = None
try:
    formula_pes = (
        "t2_rt ~ t2_error_prev * z_ucla * C(gender_male) + "
        "soa_scaled + z_dass_dep + z_dass_anx + z_dass_str"
        + (" + z_age" if "z_age" in prp_clean.columns else "")
    )

    if len(prp_clean) > 3000:
        prp_sample_pes = prp_clean.sample(n=3000, random_state=43)
        print(f"  Subsampling to {len(prp_sample_pes)} trials")
    else:
        prp_sample_pes = prp_clean

    model_pes = smf.mixedlm(
        formula_pes,
        data=prp_sample_pes,
        groups=prp_sample_pes["participant_id"],
        re_formula="~1",
    )

    result_pes = model_pes.fit(method="lbfgs", maxiter=100)

    print(result_pes.summary())
    print()

    pes_results = pd.DataFrame(
        {
            "parameter": result_pes.params.index,
            "coefficient": result_pes.params.values,
            "std_err": result_pes.bse.values,
            "t_value": result_pes.tvalues.values,
            "p_value": result_pes.pvalues.values,
        }
    )

    pes_results.to_csv(OUTPUT_DIR / "pes_glmm_results.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DIR / 'pes_glmm_results.csv'}")
    print()

except Exception as e:
    print(f"  ERROR fitting PES GLMM: {e}")
    print("  Skipping PES model")

print()

# ---------------------------------------------------------------------------
# 6. VISUALIZATION (if models succeeded)
# ---------------------------------------------------------------------------
if cascade_results is not None:
    print("Creating cascade visualization...")

    fig, ax = plt.subplots(figsize=(10, 6))
    cascade_by_prev = (
        prp_clean.groupby(["t2_error_prev", "gender_male"])["error_cascade_int"].mean().reset_index()
    )

    sns.barplot(
        data=cascade_by_prev,
        x="t2_error_prev",
        y="error_cascade_int",
        hue="gender_male",
        palette={0: "coral", 1: "steelblue"},
        ax=ax,
    )

    ax.set_xlabel("Previous T2 Error (0=Correct, 1=Error)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Cascade Rate", fontsize=12, fontweight="bold")
    ax.set_title("Error Cascade Rate by Previous Trial Status & Gender", fontsize=14, fontweight="bold")
    ax.legend(title="Gender", labels=["Female", "Male"], fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cascade_by_previous_error.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'cascade_by_previous_error.png'}")
    plt.close()

print()
print("=" * 80)
print("TRIAL-LEVEL CASCADE & PES ANALYSIS COMPLETE")
print("=" * 80)
print()
print("KEY FILES:")
if cascade_results is not None:
    print(f"  - {OUTPUT_DIR / 'cascade_glmm_results.csv'}")
    print(f"  - {OUTPUT_DIR / 'cascade_by_previous_error.png'}")
if pes_results is not None:
    print(f"  - {OUTPUT_DIR / 'pes_glmm_results.csv'}")
print()
