"""
derive_additional_trial_features.py

Computes additional trial-level features needed for advanced analyses:
1. PRP error cascade metrics (cascade rate, cascade inflation)
2. PRP post-error slowing (PES)
3. WCST post-error slowing (PES)
4. WCST post-switch error rate

Outputs: master_comprehensive_features.csv
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials, load_wcst_trials

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/analysis_outputs/advanced_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DERIVE ADDITIONAL TRIAL-LEVEL FEATURES")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# Load trial data
# ---------------------------------------------------------------------------
print("Loading trial-level data...")
prp_trials, prp_summary = load_prp_trials(use_cache=True, rt_min=200, rt_max=5000, require_t1_correct=False, enforce_short_long_only=False)
wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)

print(f"  PRP trials: {len(prp_trials):,} rows (n_participants={prp_summary.get('n_participants', prp_trials['participant_id'].nunique())})")
print(f"  WCST trials: {len(wcst_trials):,} rows (n_participants={wcst_summary.get('n_participants', wcst_trials['participant_id'].nunique())})")
print()

# ---------------------------------------------------------------------------
# PRP error cascade metrics
# ---------------------------------------------------------------------------
print("Computing PRP error cascade metrics...")

prp_clean = prp_trials[
    (prp_trials["t1_correct"].notna())
    & (prp_trials["t2_correct"].notna())
    & (prp_trials["t2_rt"].notna())
    & (prp_trials["t2_rt"].between(200, 5000))
].copy()

if prp_clean["t1_correct"].dtype == "object":
    prp_clean["t1_correct"] = prp_clean["t1_correct"].map({True: True, "True": True, False: False, "False": False, 1: True, 0: False})
if prp_clean["t2_correct"].dtype == "object":
    prp_clean["t2_correct"] = prp_clean["t2_correct"].map({True: True, "True": True, False: False, "False": False, 1: True, 0: False})

prp_clean["error_cascade"] = (~prp_clean["t1_correct"]) & (~prp_clean["t2_correct"])
prp_cascade = prp_clean.groupby("participant_id").agg(
    prp_cascade_rate=("error_cascade", "mean"),
    prp_cascade_n=("error_cascade", "size"),
    prp_pe_rate=("t2_correct", lambda x: 1 - x.mean()),
).reset_index()
prp_cascade["prp_cascade_inflation"] = prp_cascade["prp_cascade_rate"] / prp_cascade["prp_pe_rate"].replace(0, np.nan)

# PRP post-error slowing (T2 RT after error vs correct)
prp_clean = prp_clean.sort_values(["participant_id", "trial_index" if "trial_index" in prp_clean.columns else "trial"])
prp_clean["t2_error_prev"] = prp_clean.groupby("participant_id")["t2_correct"].shift(1).fillna(True)
prp_clean["t2_error_prev"] = (~prp_clean["t2_error_prev"]).astype(bool)

pes = (
    prp_clean.groupby("participant_id")
    .apply(
        lambda g: pd.Series(
            {
                "prp_pes": g.loc[g["t2_error_prev"] == True, "t2_rt"].mean() - g.loc[g["t2_error_prev"] == False, "t2_rt"].mean()
                if (g["t2_error_prev"] == True).any() and (g["t2_error_prev"] == False).any()
                else np.nan
            }
        )
    )
    .reset_index()
)

prp_features = prp_cascade.merge(pes, on="participant_id", how="left")

# ---------------------------------------------------------------------------
# WCST metrics
# ---------------------------------------------------------------------------
print("Computing WCST post-error slowing and post-switch error rate...")

rt_col_wcst = "reactionTimeMs" if "reactionTimeMs" in wcst_trials.columns else "rt_ms" if "rt_ms" in wcst_trials.columns else None
if not rt_col_wcst:
    raise KeyError("WCST trials missing reaction time column")

wcst_trials = wcst_trials.sort_values(["participant_id", "trialIndex" if "trialIndex" in wcst_trials.columns else "trial"]).copy()

if "isPE" in wcst_trials.columns:
    wcst_trials["is_pe"] = wcst_trials["isPE"].astype(bool)
elif "is_pe" not in wcst_trials.columns:
    wcst_trials["is_pe"] = False

# Post-error slowing: RT on trial n+1 after PE vs after correct
wcst_pes_metrics = []
for pid, grp in wcst_trials.groupby("participant_id"):
    grp = grp.reset_index(drop=True)
    if "correct" not in grp.columns:
        continue
    rt_vals = grp[rt_col_wcst].values
    correct = grp["correct"].values
    post_pe_rts = []
    post_correct_rts = []
    for i in range(len(grp) - 1):
        if correct[i] == False:
            post_pe_rts.append(rt_vals[i + 1])
        elif correct[i] == True:
            post_correct_rts.append(rt_vals[i + 1])
    wcst_pes_metrics.append(
        {
            "participant_id": pid,
            "wcst_pes": (np.mean(post_pe_rts) - np.mean(post_correct_rts)) if post_pe_rts and post_correct_rts else np.nan,
        }
    )
wcst_pes_df = pd.DataFrame(wcst_pes_metrics)

# Post-switch error rate (using rule changes in ruleAtThatTime)
rule_col = None
for cand in ["ruleAtThatTime", "rule_at_that_time", "rule_at_time"]:
    if cand in wcst_trials.columns:
        rule_col = cand
        break
wcst_switch_df = pd.DataFrame(columns=["participant_id", "wcst_post_switch_error_rate"])
if rule_col:
    switch_metrics = []
    for pid, grp in wcst_trials.groupby("participant_id"):
        rules = grp[rule_col].values
        is_correct = grp["correct"].values if "correct" in grp.columns else np.array([])
        rule_changes = [0]
        for i in range(1, len(rules)):
            if rules[i] != rules[i - 1]:
                rule_changes.append(i)
        post_switch_errors = []
        for change_idx in rule_changes[1:]:
            window_end = min(change_idx + 10, len(grp))
            post_switch_trials = is_correct[change_idx:window_end]
            if len(post_switch_trials) >= 5:
                post_switch_errors.append(1 - post_switch_trials[:5].mean())
        switch_metrics.append(
            {
                "participant_id": pid,
                "wcst_post_switch_error_rate": np.nanmean(post_switch_errors) if post_switch_errors else np.nan,
            }
        )
    wcst_switch_df = pd.DataFrame(switch_metrics)

wcst_features = wcst_pes_df.merge(wcst_switch_df, on="participant_id", how="outer")

# ---------------------------------------------------------------------------
# Merge with master and save
# ---------------------------------------------------------------------------
print("Merging features with master dataset...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

features = master.merge(prp_features, on="participant_id", how="left").merge(wcst_features, on="participant_id", how="left")
features.to_csv(OUTPUT_DIR / "master_comprehensive_features.csv", index=False, encoding="utf-8-sig")

print(f"Saved: {OUTPUT_DIR / 'master_comprehensive_features.csv'}")

print("\n" + "=" * 80)
print("Feature derivation complete!")
print("=" * 80)
