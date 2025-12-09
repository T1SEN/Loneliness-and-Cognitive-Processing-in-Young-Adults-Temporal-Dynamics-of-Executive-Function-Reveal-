"""
Stroop Congruency Sequence Effect (CSE) - Conflict Adaptation Analysis

Objective:
Test whether lonely males show impaired trial-to-trial cognitive control adjustments.

Outputs:
- Participant-level CSE scores
- Gender-stratified UCLA correlations with CSE
- Visualization of 2×2 interaction patterns and UCLA × CSE scatterplots
- Text report and CSV summaries
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import seaborn as sns

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_stroop_trials


OUTPUT_DIR = Path("results/analysis_outputs/stroop_cse")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
np.random.seed(42)

print("=" * 80)
print("STROOP CONGRUENCY SEQUENCE EFFECT (CSE) - CONFLICT ADAPTATION")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1/5] Loading data...")

master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

demo_cols = [
    "participant_id",
    "gender_normalized",
    "ucla_total",
    "dass_depression",
    "dass_anxiety",
    "dass_stress",
]
if "age" in master.columns:
    demo_cols.append("age")

demo = master[demo_cols].copy()
demo["gender"] = demo["gender_normalized"].fillna("").astype(str).str.strip().str.lower()
demo["gender_male"] = (demo["gender"] == "male").astype(int)
demo = demo.dropna(subset=["ucla_total"])

stroop_trials, trials_summary = load_stroop_trials(use_cache=True)

rt_col = "rt" if "rt" in stroop_trials.columns else "rt_ms" if "rt_ms" in stroop_trials.columns else None
if not rt_col:
    raise KeyError("Stroop trials missing rt/rt_ms column")
if rt_col != "rt":
    stroop_trials["rt"] = stroop_trials[rt_col]

if "type" not in stroop_trials.columns:
    for cand in ["condition", "cond"]:
        if cand in stroop_trials.columns:
            stroop_trials = stroop_trials.rename(columns={cand: "type"})
            break
if "type" not in stroop_trials.columns:
    raise KeyError("Stroop trials missing condition/type column")

if "correct" not in stroop_trials.columns:
    if "isCorrect" in stroop_trials.columns:
        stroop_trials["correct"] = stroop_trials["isCorrect"]
    else:
        raise KeyError("Stroop trials missing correct/isCorrect column")

if "timeout" not in stroop_trials.columns:
    stroop_trials["timeout"] = False

trial_idx_col = None
for cand in ["trialIndex", "trial"]:
    if cand in stroop_trials.columns:
        trial_idx_col = cand
        break
if not trial_idx_col:
    raise KeyError("Stroop trials missing trial index column")

stroop_trials = stroop_trials.sort_values(["participant_id", trial_idx_col])

valid_stroop = stroop_trials[
    (stroop_trials["correct"] == True)
    & (stroop_trials["timeout"] == False)
    & (stroop_trials["rt"] > 200)
    & (stroop_trials["rt"] < 3000)
    & (stroop_trials["type"].isin(["congruent", "incongruent"]))
].copy()

print(
    f"  Valid trials after filtering: {len(valid_stroop):,} "
    f"(n_participants={trials_summary.get('n_participants', valid_stroop['participant_id'].nunique())})"
)

# ---------------------------------------------------------------------------
# 2. PREPARE TRIAL-LEVEL DATA FOR CSE
# ---------------------------------------------------------------------------
print("\n[2/5] Computing trial N-1 congruency sequences...")

valid_stroop["is_incongruent"] = (valid_stroop["type"] == "incongruent").astype(int)
valid_stroop["prev_is_incongruent"] = valid_stroop.groupby("participant_id")["is_incongruent"].shift(1)

cse_data = valid_stroop.dropna(subset=["prev_is_incongruent"]).copy()
print(f"  Trials with N-1 congruency coded: {len(cse_data):,}")

# ---------------------------------------------------------------------------
# 3. CALCULATE CSE PER PARTICIPANT
# ---------------------------------------------------------------------------
print("\n[3/5] Calculating CSE scores per participant...")

cse_results = []
for pid, group in cse_data.groupby("participant_id"):
    cC = group[(group["prev_is_incongruent"] == 0) & (group["is_incongruent"] == 0)]["rt"].mean()
    cI = group[(group["prev_is_incongruent"] == 0) & (group["is_incongruent"] == 1)]["rt"].mean()
    iC = group[(group["prev_is_incongruent"] == 1) & (group["is_incongruent"] == 0)]["rt"].mean()
    iI = group[(group["prev_is_incongruent"] == 1) & (group["is_incongruent"] == 1)]["rt"].mean()

    if pd.notna([cC, cI, iC, iI]).all():
        interference_after_congruent = cI - cC
        interference_after_incongruent = iI - iC
        cse = interference_after_congruent - interference_after_incongruent

        cse_results.append(
            {
                "participant_id": pid,
                "cC": cC,
                "cI": cI,
                "iC": iC,
                "iI": iI,
                "interference_after_congruent": interference_after_congruent,
                "interference_after_incongruent": interference_after_incongruent,
                "cse": cse,
                "n_trials": len(group),
            }
        )

cse_df = pd.DataFrame(cse_results)

print(f"  CSE computed for {len(cse_df)} participants")
print(f"  Mean CSE: {cse_df['cse'].mean():.1f} ms (SD={cse_df['cse'].std():.1f})")

# ---------------------------------------------------------------------------
# 4. MERGE WITH UCLA AND TEST CORRELATIONS
# ---------------------------------------------------------------------------
print("\n[4/5] Testing UCLA × Gender on CSE...")

analysis_data = cse_df.merge(
    demo[
        [
            col
            for col in [
                "participant_id",
                "ucla_total",
                "gender",
                "gender_male",
                "dass_depression",
                "dass_anxiety",
                "dass_stress",
                "age",
            ]
            if col in demo.columns
        ]
    ],
    on="participant_id",
    how="inner",
)

analysis_data = analysis_data.dropna(subset=["ucla_total", "gender_male", "dass_depression", "dass_anxiety", "dass_stress"])

print(f"  Final N={len(analysis_data)} (males={(analysis_data['gender_male'] == 1).sum()}, females={(analysis_data['gender_male'] == 0).sum()})")

r_overall, p_overall = stats.pearsonr(analysis_data["ucla_total"], analysis_data["cse"])
print(f"\n  Overall UCLA × CSE: r={r_overall:.3f}, p={p_overall:.4f}")

males = analysis_data[analysis_data["gender_male"] == 1]
females = analysis_data[analysis_data["gender_male"] == 0]

if len(males) >= 10:
    r_male, p_male = stats.pearsonr(males["ucla_total"], males["cse"])
    print(f"  Males (N={len(males)}): r={r_male:.3f}, p={p_male:.4f}")
else:
    r_male, p_male = np.nan, np.nan
    print("  Males: Insufficient N")

if len(females) >= 10:
    r_female, p_female = stats.pearsonr(females["ucla_total"], females["cse"])
    print(f"  Females (N={len(females)}): r={r_female:.3f}, p={p_female:.4f}")
else:
    r_female, p_female = np.nan, np.nan
    print("  Females: Insufficient N")

analysis_data["z_ucla"] = (analysis_data["ucla_total"] - analysis_data["ucla_total"].mean()) / analysis_data["ucla_total"].std()
analysis_data["z_dass_dep"] = (analysis_data["dass_depression"] - analysis_data["dass_depression"].mean()) / analysis_data["dass_depression"].std()
analysis_data["z_dass_anx"] = (analysis_data["dass_anxiety"] - analysis_data["dass_anxiety"].mean()) / analysis_data["dass_anxiety"].std()
analysis_data["z_dass_str"] = (analysis_data["dass_stress"] - analysis_data["dass_stress"].mean()) / analysis_data["dass_stress"].std()

formula = "cse ~ z_ucla * gender_male + z_dass_dep + z_dass_anx + z_dass_str"
model = smf.ols(formula, data=analysis_data).fit()

print("\n  Interaction Model: CSE ~ UCLA × Gender (DASS-controlled)")
print(f"    UCLA main effect: β={model.params['z_ucla']:.2f}, p={model.pvalues['z_ucla']:.4f}")
print(f"    Gender main effect: β={model.params['gender_male']:.2f}, p={model.pvalues['gender_male']:.4f}")
print(f"    UCLA × Gender: β={model.params['z_ucla:gender_male']:.2f}, p={model.pvalues['z_ucla:gender_male']:.4f}")

# ---------------------------------------------------------------------------
# 5. VISUALIZATIONS
# ---------------------------------------------------------------------------
print("\n[5/5] Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, gender_label, subset in [
    (axes[0], "Males", males),
    (axes[1], "Females", females),
]:
    if len(subset) < 5:
        ax.text(0.5, 0.5, "Insufficient N", ha="center", va="center", fontsize=12)
        ax.set_title(f"{gender_label} (N={len(subset)})", fontweight="bold")
        continue

    mean_cC = subset["cC"].mean()
    mean_cI = subset["cI"].mean()
    mean_iC = subset["iC"].mean()
    mean_iI = subset["iI"].mean()

    se_cC = subset["cC"].sem()
    se_cI = subset["cI"].sem()
    se_iC = subset["iC"].sem()
    se_iI = subset["iI"].sem()

    x = [1, 2]
    after_congruent = [mean_cC, mean_cI]
    after_incongruent = [mean_iC, mean_iI]
    se_congruent = [se_cC, se_cI]
    se_incongruent = [se_iC, se_iI]

    ax.errorbar(
        x,
        after_congruent,
        yerr=se_congruent,
        marker="o",
        linewidth=2,
        label="After Congruent",
        color="#3498DB",
        markersize=8,
        capsize=5,
    )
    ax.errorbar(
        x,
        after_incongruent,
        yerr=se_incongruent,
        marker="s",
        linewidth=2,
        label="After Incongruent",
        color="#E74C3C",
        markersize=8,
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Congruent\nTrial (N)", "Incongruent\nTrial (N)"])
    ax.set_ylabel("Mean RT (ms)", fontweight="bold")
    ax.set_title(f"{gender_label} (N={len(subset)})", fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.3)

    mean_cse = subset["cse"].mean()
    ax.text(
        0.98,
        0.02,
        f"Mean CSE = {mean_cse:.1f} ms",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

plt.suptitle("Congruency Sequence Effect: Trial N-1 × Trial N Interaction", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "cse_interaction_pattern.png", dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, gender_label, subset, r_val, p_val in [
    (axes[0], "Males", males, r_male, p_male),
    (axes[1], "Females", females, r_female, p_female),
]:
    if len(subset) < 10:
        ax.text(0.5, 0.5, "Insufficient N", ha="center", va="center", fontsize=12)
        ax.set_title(f"{gender_label} (N={len(subset)})", fontweight="bold")
        continue

    ax.scatter(subset["ucla_total"], subset["cse"], alpha=0.7, s=80, edgecolors="black")

    z = np.polyfit(subset["ucla_total"], subset["cse"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(subset["ucla_total"].min(), subset["ucla_total"].max(), 100)
    ax.plot(x_line, p_line(x_line), "r--", linewidth=2, alpha=0.7)

    ax.set_xlabel("UCLA Loneliness Score", fontweight="bold")
    ax.set_ylabel("CSE (ms)", fontweight="bold")
    ax.set_title(f"{gender_label} (N={len(subset)})", fontweight="bold", pad=10)
    ax.grid(alpha=0.3)

    if pd.notna(r_val) and pd.notna(p_val):
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(
            0.05,
            0.95,
            f"r = {r_val:.3f}\np = {p_val:.4f} {sig_marker}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, label="CSE = 0 (no adaptation)")

plt.suptitle("UCLA Loneliness × Congruency Sequence Effect", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "ucla_cse_scatterplots.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

if len(males) >= 5:
    ax.hist(males["cse"], bins=15, alpha=0.6, label=f"Males (N={len(males)})", color="#3498DB", edgecolor="black")
if len(females) >= 5:
    ax.hist(females["cse"], bins=15, alpha=0.6, label=f"Females (N={len(females)})", color="#E74C3C", edgecolor="black")

ax.axvline(0, color="black", linestyle="--", linewidth=2, label="CSE = 0 (no adaptation)")
ax.set_xlabel("CSE (ms)", fontweight="bold")
ax.set_ylabel("Frequency", fontweight="bold")
ax.set_title("Distribution of Congruency Sequence Effect by Gender", fontweight="bold", pad=15)
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cse_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("  CSE plots saved")

# ---------------------------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------------------------
analysis_data.to_csv(OUTPUT_DIR / "cse_participant_scores.csv", index=False, encoding="utf-8-sig")

summary_stats = pd.DataFrame(
    {
        "Group": ["Overall", "Males", "Females"],
        "N": [len(analysis_data), len(males), len(females)],
        "Mean_CSE": [
            analysis_data["cse"].mean(),
            males["cse"].mean() if len(males) > 0 else np.nan,
            females["cse"].mean() if len(females) > 0 else np.nan,
        ],
        "SD_CSE": [
            analysis_data["cse"].std(),
            males["cse"].std() if len(males) > 0 else np.nan,
            females["cse"].std() if len(females) > 0 else np.nan,
        ],
        "UCLA_CSE_r": [r_overall, r_male, r_female],
        "UCLA_CSE_p": [p_overall, p_male, p_female],
    }
)
summary_stats.to_csv(OUTPUT_DIR / "cse_summary_stats.csv", index=False, encoding="utf-8-sig")

model_results = pd.DataFrame(
    {
        "Predictor": ["Intercept", "UCLA (z)", "Gender (male)", "UCLA × Gender"],
        "Coefficient": [
            model.params["Intercept"],
            model.params["z_ucla"],
            model.params["gender_male"],
            model.params["z_ucla:gender_male"],
        ],
        "SE": [
            model.bse["Intercept"],
            model.bse["z_ucla"],
            model.bse["gender_male"],
            model.bse["z_ucla:gender_male"],
        ],
        "p_value": [
            model.pvalues["Intercept"],
            model.pvalues["z_ucla"],
            model.pvalues["gender_male"],
            model.pvalues["z_ucla:gender_male"],
        ],
        "CI_lower": model.conf_int()[0].values,
        "CI_upper": model.conf_int()[1].values,
    }
)
model_results.to_csv(OUTPUT_DIR / "cse_regression_model.csv", index=False, encoding="utf-8-sig")

with open(OUTPUT_DIR / "CSE_CONFLICT_ADAPTATION_REPORT.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("STROOP CONGRUENCY SEQUENCE EFFECT (CSE) - CONFLICT ADAPTATION ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write("OBJECTIVE\n" + "-" * 80 + "\n")
    f.write("Test whether lonely individuals (especially males) show impaired trial-to-trial\n")
    f.write("cognitive control adjustments using the Congruency Sequence Effect (CSE).\n\n")
    f.write("SUMMARY STATISTICS\n" + "-" * 80 + "\n")
    f.write(summary_stats.to_string(index=False) + "\n\n")
    f.write("INTERACTION MODEL (DASS-CONTROLLED)\n" + "-" * 80 + "\n")
    f.write(model.summary().as_text())

print("\n" + "=" * 80)
print("Stroop CSE analysis complete!")
print("=" * 80)
