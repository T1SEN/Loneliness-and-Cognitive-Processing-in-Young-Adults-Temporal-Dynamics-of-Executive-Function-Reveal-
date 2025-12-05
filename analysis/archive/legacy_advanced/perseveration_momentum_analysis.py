"""
Perseveration Momentum Analysis - Consecutive PE Runs
=====================================================
Distinguishes true perseverative rigidity (stuck in set) from random errors.

Questions:
1) Do lonely males show longer runs of consecutive PEs?
2) Is there autocorrelation in PE sequences (PE[t] -> PE[t+1])?
3) Which metric is stronger: run length vs PE rate?
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/analysis_outputs/perseveration_momentum")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("PERSEVERATION MOMENTUM ANALYSIS")
print("=" * 80)
print("\nResearch Question: Do PEs cluster into consecutive runs (true rigidity)?")
print("  - Analyze run lengths of consecutive PEs")
print("  - Compute lag-1 autocorrelation")
print("  - Compare run metrics vs PE rate\n")

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
print("Loading data...")

wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)

rt_col = "reactionTimeMs" if "reactionTimeMs" in wcst_trials.columns else "rt_ms" if "rt_ms" in wcst_trials.columns else None
if not rt_col:
    raise KeyError("WCST trials missing reaction time column")

if "isPE" in wcst_trials.columns:
    wcst_trials["is_pe"] = wcst_trials["isPE"].fillna(0).astype(int)
elif "is_pe" not in wcst_trials.columns:
    wcst_trials["is_pe"] = 0

wcst_trials = wcst_trials.sort_values(["participant_id", "trialIndex" if "trialIndex" in wcst_trials.columns else rt_col]).reset_index(drop=True)

print(f"  Loaded {len(wcst_trials)} WCST trials (n_participants={wcst_summary.get('n_participants', wcst_trials['participant_id'].nunique())})")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

wcst_trials = wcst_trials.merge(master[["participant_id", "ucla_total", "gender_male", "age"]], on="participant_id", how="left")

# ---------------------------------------------------------------------------
# Run-length metrics
# ---------------------------------------------------------------------------
def compute_run_metrics(pe_series: pd.Series) -> dict:
    arr = pe_series.astype(int).values
    if len(arr) == 0:
        return {"pe_rate": np.nan, "mean_run_length": np.nan, "max_run_length": np.nan, "lag1_autocorr": np.nan}

    pe_rate = arr.mean()

    runs = []
    current = 0
    for v in arr:
        if v == 1:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    mean_run = np.mean(runs) if runs else 0
    max_run = np.max(runs) if runs else 0

    lag1 = np.nan
    if len(arr) > 1 and np.std(arr) > 0:
        lag1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]

    return {
        "pe_rate": pe_rate,
        "mean_run_length": mean_run,
        "max_run_length": max_run,
        "lag1_autocorr": lag1,
    }


metrics = []
for pid, grp in wcst_trials.groupby("participant_id"):
    if "is_pe" not in grp.columns:
        continue
    stats_dict = compute_run_metrics(grp["is_pe"])
    stats_dict["participant_id"] = pid
    metrics.append(stats_dict)

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.merge(master[["participant_id", "ucla_total", "gender_male", "age"]], on="participant_id", how="left")

print(f"  Metrics computed for {len(metrics_df)} participants")

# ---------------------------------------------------------------------------
# Correlations and group differences
# ---------------------------------------------------------------------------
print("\nTesting associations with UCLA and gender...")

results = []
for metric in ["pe_rate", "mean_run_length", "max_run_length", "lag1_autocorr"]:
    valid = metrics_df.dropna(subset=[metric, "ucla_total"])
    if len(valid) >= 10:
        r, p = stats.pearsonr(valid["ucla_total"], valid[metric])
        results.append({"metric": metric, "n": len(valid), "r_ucla": r, "p_ucla": p})

    for gender_flag, label in [(1, "Male"), (0, "Female")]:
        subset = valid[valid["gender_male"] == gender_flag]
        if len(subset) >= 10:
            r, p = stats.pearsonr(subset["ucla_total"], subset[metric])
            results.append({"metric": metric, "gender": label, "n": len(subset), "r_ucla": r, "p_ucla": p})

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "pe_run_correlations.csv", index=False, encoding="utf-8-sig")

metrics_df.to_csv(OUTPUT_DIR / "pe_run_metrics.csv", index=False, encoding="utf-8-sig")

print("\nSaved:")
print(f"  - {OUTPUT_DIR / 'pe_run_metrics.csv'}")
print(f"  - {OUTPUT_DIR / 'pe_run_correlations.csv'}")

print("\n" + "=" * 80)
print("Perseveration momentum analysis complete!")
print("=" * 80)
