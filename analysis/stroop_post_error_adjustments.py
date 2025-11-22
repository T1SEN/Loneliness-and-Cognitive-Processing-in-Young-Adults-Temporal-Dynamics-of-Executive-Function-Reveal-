"""
Stroop Post-Error Adjustments Analysis
======================================
Tests adaptive control after errors.

Analyses:
1. Post-error slowing (RT_n+1 after error vs correct)
2. Post-error accuracy
3. Post-error interference (incongruent - congruent after errors)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials


OUTPUT_DIR = Path("results/analysis_outputs/stroop_deep_dive/post_error_adjustments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("STROOP POST-ERROR ADJUSTMENTS")
print("=" * 80)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\n[1] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

demo = master[
    [
        "participant_id",
        "gender_normalized",
        "ucla_total",
        "dass_depression",
        "dass_anxiety",
        "dass_stress",
    ]
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

if "trial" not in trials.columns and "trialIndex" in trials.columns:
    trials = trials.rename(columns={"trialIndex": "trial"})
if "trial" not in trials.columns:
    raise KeyError("Stroop trials missing trial index column")

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

print(f"  Valid trials: {len(trials_clean):,} (n_participants={trials_summary.get('n_participants', trials_clean['participant_id'].nunique())})")


# ---------------------------------------------------------------------------
# Post-error metrics
# ---------------------------------------------------------------------------
def compute_post_error_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("trial").reset_index(drop=True)

    group["prev_correct"] = group["correct"].shift(1)
    group["prev_type"] = group["type"].shift(1)

    post_error = group[group["prev_correct"] == False]
    post_correct = group[group["prev_correct"] == True]

    metrics = {}

    if len(post_error) >= 5 and len(post_correct) >= 5:
        metrics["rt_post_error"] = post_error["rt"].mean()
        metrics["rt_post_correct"] = post_correct["rt"].mean()
        metrics["post_error_slowing"] = post_error["rt"].mean() - post_correct["rt"].mean()
    else:
        metrics["rt_post_error"] = np.nan
        metrics["rt_post_correct"] = np.nan
        metrics["post_error_slowing"] = np.nan

    metrics["accuracy_post_error"] = post_error["correct"].mean() if len(post_error) >= 5 else np.nan
    metrics["accuracy_post_correct"] = post_correct["correct"].mean() if len(post_correct) >= 5 else np.nan

    post_error_incong = group[(group["prev_correct"] == False) & (group["type"] == "incongruent")]
    post_error_cong = group[(group["prev_correct"] == False) & (group["type"] == "congruent")]

    if len(post_error_incong) >= 3 and len(post_error_cong) >= 3:
        metrics["interference_post_error"] = post_error_incong["rt"].mean() - post_error_cong["rt"].mean()
    else:
        metrics["interference_post_error"] = np.nan

    return pd.Series(metrics)


post_error_df = (
    trials_clean.groupby("participant_id")
    .apply(compute_post_error_metrics)
    .reset_index()
    .merge(
        demo[
            [
                "participant_id",
                "gender",
                "ucla_total",
                "dass_depression",
                "dass_anxiety",
                "dass_stress",
            ]
        ],
        on="participant_id",
        how="inner",
    )
)

print(f"\n[2] Post-error metrics computed for N={len(post_error_df)}")
post_error_df.to_csv(OUTPUT_DIR / "post_error_metrics.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Gender-stratified correlations
# ---------------------------------------------------------------------------
results = []
for gender in ["male", "female"]:
    data = post_error_df[post_error_df["gender"] == gender]

    for metric in ["post_error_slowing", "accuracy_post_error", "interference_post_error"]:
        valid = data.dropna(subset=["ucla_total", metric])
        if len(valid) >= 10:
            r, p = pearsonr(valid["ucla_total"], valid[metric])
            results.append({"gender": gender, "metric": metric, "n": len(valid), "r": r, "p": p})

gender_corr_df = pd.DataFrame(results)
print(f"\n[3] Gender correlations:")
print(gender_corr_df)
gender_corr_df.to_csv(OUTPUT_DIR / "gender_stratified_correlations.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STROOP POST-ERROR ADJUSTMENTS - KEY FINDINGS")
print("=" * 80)

print(
    f"""
1. Post-Error Slowing:
   - Mean: {post_error_df['post_error_slowing'].mean():.1f} ms
   - Males: {post_error_df[post_error_df['gender']=='male']['post_error_slowing'].mean():.1f} ms
   - Females: {post_error_df[post_error_df['gender']=='female']['post_error_slowing'].mean():.1f} ms

2. Post-Error Accuracy:
   - Mean: {post_error_df['accuracy_post_error'].mean():.3f}
   - Post-Correct Accuracy: {post_error_df['accuracy_post_correct'].mean():.3f}

3. Gender Correlations:
{gender_corr_df.to_string(index=False)}

4. Files Generated:
   - post_error_metrics.csv
   - gender_stratified_correlations.csv
"""
)

print("\n" + "=" * 80)
print("Stroop post-error adjustments complete!")
print("=" * 80)
