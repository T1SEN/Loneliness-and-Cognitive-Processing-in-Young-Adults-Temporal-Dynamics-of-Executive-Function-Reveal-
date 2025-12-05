"""
WCST Post-Error Adaptation (Quick Analysis)

Research Question:
Do lonely males show impaired post-error adaptation (reduced slowing, more errors after errors)?

Metrics:
- Post-error slowing: RT(n+1|error_n) - RT(n+1|correct_n)
- Post-error accuracy: Accuracy on trial n+1 after error vs correct
- UCLA × Gender moderation
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


OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/post_error_adaptation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("WCST POST-ERROR ADAPTATION")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

demo = master[["participant_id", "gender_normalized", "ucla_total"]].copy()
demo["gender"] = demo["gender_normalized"].fillna("").astype(str).str.strip().str.lower()
demo["gender_male"] = (demo["gender"] == "male").astype(int)
demo = demo.dropna(subset=["ucla_total"])
demo = demo[["participant_id", "gender", "gender_male", "ucla_total"]]

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

wcst_trials["rt_valid"] = wcst_trials[rt_col] > 0
sort_cols = [col for col in ["participant_id", "trialIndex"] if col in wcst_trials.columns]
if sort_cols:
    wcst_trials = wcst_trials.sort_values(sort_cols)

print(
    f"  Loaded {len(wcst_trials):,} trials "
    f"(n_participants={wcst_summary.get('n_participants', wcst_trials['participant_id'].nunique())})"
)

# Compute post-error metrics
print("\n[2/4] Computing post-error adaptation...")

adaptation_metrics = []

for pid, grp in wcst_trials.groupby("participant_id"):
    grp = grp.reset_index(drop=True)
    if "correct" not in grp.columns:
        continue

    is_correct = grp["correct"].values
    rts = grp[rt_col].values
    rt_valid = grp["rt_valid"].values

    post_error_rts = []
    post_correct_rts = []
    post_error_accuracy = []
    post_correct_accuracy = []

    for i in range(len(grp) - 1):
        if not rt_valid[i] or not rt_valid[i + 1]:
            continue

        if is_correct[i] is False:  # Error on trial i
            post_error_rts.append(rts[i + 1])
            post_error_accuracy.append(is_correct[i + 1])
        elif is_correct[i] is True:  # Correct on trial i
            post_correct_rts.append(rts[i + 1])
            post_correct_accuracy.append(is_correct[i + 1])

    if len(post_error_rts) >= 3 and len(post_correct_rts) >= 3:
        adaptation_metrics.append(
            {
                "participant_id": pid,
                "post_error_slowing": float(np.mean(post_error_rts) - np.mean(post_correct_rts)),
                "post_error_accuracy": float(np.mean(post_error_accuracy)) if post_error_accuracy else np.nan,
                "post_correct_accuracy": float(np.mean(post_correct_accuracy)) if post_correct_accuracy else np.nan,
                "post_error_n": len(post_error_rts),
                "post_correct_n": len(post_correct_rts),
            }
        )

adaptation_df = pd.DataFrame(adaptation_metrics)
print(f"  Computed adaptation metrics for {len(adaptation_df)} participants")

# Merge with demographics/UCLA
analysis_df = adaptation_df.merge(demo, on="participant_id", how="inner")
print(f"\n  Complete cases: N={len(analysis_df)}")

# Test correlations
print("\n[3/4] Testing UCLA × adaptation correlations...")

for gender_flag, label in [(1, "Male"), (0, "Female")]:
    subset = analysis_df[analysis_df["gender_male"] == gender_flag]
    print(f"\n  {label} (N={len(subset)}):")

    if len(subset) >= 10:
        for metric in ["post_error_slowing", "post_error_accuracy"]:
            valid = subset[["ucla_total", metric]].dropna()
            if len(valid) >= 10:
                r, p = stats.pearsonr(valid["ucla_total"], valid[metric])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    UCLA × {metric}: r={r:.3f}, p={p:.3f} {sig}")

# Visualization
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for gender_flag, label, marker, color in [
    (1, "Male", "s", "#3498DB"),
    (0, "Female", "o", "#E74C3C"),
]:
    subset = analysis_df[analysis_df["gender_male"] == gender_flag]

    axes[0].scatter(
        subset["ucla_total"],
        subset["post_error_slowing"],
        alpha=0.6,
        label=label,
        marker=marker,
        s=80,
        color=color,
    )

    axes[1].scatter(
        subset["ucla_total"],
        subset["post_error_accuracy"] * 100,
        alpha=0.6,
        label=label,
        marker=marker,
        s=80,
        color=color,
    )

axes[0].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="No slowing")
axes[0].set_xlabel("UCLA Loneliness")
axes[0].set_ylabel("Post-Error Slowing (ms)")
axes[0].set_title("Post-Error RT Adaptation")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("UCLA Loneliness")
axes[1].set_ylabel("Post-Error Accuracy (%)")
axes[1].set_title("Accuracy After Errors")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "post_error_adaptation.png", dpi=300, bbox_inches="tight")
plt.close()

# Save results
analysis_df.to_csv(OUTPUT_DIR / "post_error_metrics.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("WCST Post-Error Adaptation Complete!")
print("=" * 80)
