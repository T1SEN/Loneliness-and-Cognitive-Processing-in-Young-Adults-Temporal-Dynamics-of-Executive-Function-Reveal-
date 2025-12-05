"""
WCST Rule-Switching Dynamics (Quick Analysis)

Research Question:
Do lonely males show slower learning after rule changes (delayed switching)?

Metrics:
- Trials-to-criterion after each rule change
- Post-switch error rate (first 5 trials)
- UCLA × Gender moderation of learning rate
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials


OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/wcst_switching")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("WCST RULE-SWITCHING DYNAMICS")
print("=" * 80)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\n[1/4] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

demo = master[["participant_id", "gender_normalized", "ucla_total"]].copy()
demo["gender"] = demo["gender_normalized"].fillna("").astype(str).str.strip().str.lower()
demo["gender_male"] = (demo["gender"] == "male").astype(int)
demo = demo.dropna(subset=["ucla_total"])

wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)

rt_col = (
    "reactionTimeMs"
    if "reactionTimeMs" in wcst_trials.columns
    else "rt_ms"
    if "rt_ms" in wcst_trials.columns
    else None
)
if not rt_col:
    raise KeyError("WCST trials missing reaction time column (reactionTimeMs/rt_ms)")

trial_col = "trialIndex" if "trialIndex" in wcst_trials.columns else "trial" if "trial" in wcst_trials.columns else None
if not trial_col:
    raise KeyError("WCST trials missing trial index column")

rule_col = None
for cand in ["ruleAtThatTime", "rule_at_that_time", "rule_at_time"]:
    if cand in wcst_trials.columns:
        rule_col = cand
        break
if rule_col is None:
    raise KeyError("WCST trials missing rule column (ruleAtThatTime)")

if "correct" not in wcst_trials.columns:
    raise KeyError("WCST trials missing correct column")

wcst_trials = wcst_trials.sort_values(["participant_id", trial_col])

print(
    f"  Loaded {len(wcst_trials):,} trials "
    f"(n_participants={wcst_summary.get('n_participants', wcst_trials['participant_id'].nunique())})"
)

# ---------------------------------------------------------------------------
# Detect rule changes and compute metrics
# ---------------------------------------------------------------------------
print("\n[2/4] Detecting rule changes...")

switching_metrics = []

for pid, grp in wcst_trials.groupby("participant_id"):
    rules = grp[rule_col].values
    is_correct = grp["correct"].values

    rule_changes = [0]
    for i in range(1, len(rules)):
        if rules[i] != rules[i - 1]:
            rule_changes.append(i)

    if len(rule_changes) < 2:
        continue

    post_switch_errors = []
    trials_to_criterion = []

    for change_idx in rule_changes[1:]:
        window_end = min(change_idx + 10, len(grp))
        post_switch_trials = is_correct[change_idx:window_end]

        if len(post_switch_trials) >= 5:
            post_switch_errors.append(1 - post_switch_trials[:5].mean())

            criterion_met = False
            for j in range(change_idx, len(is_correct) - 4):
                if is_correct[j : j + 5].sum() == 5:
                    trials_to_criterion.append(j - change_idx)
                    criterion_met = True
                    break
            if not criterion_met:
                trials_to_criterion.append(np.nan)

    switching_metrics.append(
        {
            "participant_id": pid,
            "n_rule_changes": len(rule_changes) - 1,
            "post_switch_error_rate": np.nanmean(post_switch_errors) if post_switch_errors else np.nan,
            "avg_trials_to_criterion": np.nanmean(trials_to_criterion) if trials_to_criterion else np.nan,
        }
    )

switching_df = pd.DataFrame(switching_metrics)
print(f"  Computed switching metrics for {len(switching_df)} participants")

# ---------------------------------------------------------------------------
# Merge with UCLA and gender
# ---------------------------------------------------------------------------
analysis_df = switching_df.merge(demo, on="participant_id", how="inner")
analysis_df = analysis_df.dropna(subset=["post_switch_error_rate"])

print(f"\n  Complete cases: N={len(analysis_df)}")

# ---------------------------------------------------------------------------
# Test correlations
# ---------------------------------------------------------------------------
print("\n[3/4] Testing UCLA × switching correlations...")

for gender_flag, label in [(1, "Male"), (0, "Female")]:
    subset = analysis_df[analysis_df["gender_male"] == gender_flag]
    print(f"\n  {label} (N={len(subset)}):")

    if len(subset) >= 10:
        for metric in ["post_switch_error_rate", "avg_trials_to_criterion"]:
            valid = subset[["ucla_total", metric]].dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid["ucla_total"], valid[metric])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    UCLA × {metric}: r={r:.3f}, p={p:.3f} {sig}")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gender_flag, label, marker, color in [
    (1, "Male", "s", "#3498DB"),
    (0, "Female", "o", "#E74C3C"),
]:
    subset = analysis_df[analysis_df["gender_male"] == gender_flag]

    axes[0].scatter(
        subset["ucla_total"],
        subset["post_switch_error_rate"],
        alpha=0.6,
        label=label,
        marker=marker,
        s=80,
        color=color,
    )

    valid = subset.dropna(subset=["avg_trials_to_criterion"])
    axes[1].scatter(
        valid["ucla_total"],
        valid["avg_trials_to_criterion"],
        alpha=0.6,
        label=label,
        marker=marker,
        s=80,
        color=color,
    )

axes[0].set_xlabel("UCLA Loneliness")
axes[0].set_ylabel("Post-Switch Error Rate")
axes[0].set_title("Errors After Rule Changes")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("UCLA Loneliness")
axes[1].set_ylabel("Trials to Criterion")
axes[1].set_title("Learning Speed After Rule Changes")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "switching_dynamics.png", dpi=300, bbox_inches="tight")
plt.close()

# Save results
analysis_df.to_csv(OUTPUT_DIR / "switching_metrics.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("WCST Switching Dynamics Complete!")
print("=" * 80)
