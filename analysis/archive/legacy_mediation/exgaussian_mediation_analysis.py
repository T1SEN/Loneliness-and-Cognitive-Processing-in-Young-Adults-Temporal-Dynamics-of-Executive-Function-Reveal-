"""
Ex-Gaussian Mediation Analysis
=================================
Tests mediation pathways: UCLA -> Ex-Gaussian parameters -> Behavioral outcomes.

Key hypotheses (examples):
1. PRP: UCLA -> τ (long SOA) -> Bottleneck effect (males only)
2. PRP: UCLA -> τ (bottleneck) -> T2 RT variability (males only)
3. WCST: UCLA -> Stroop RT variability -> WCST PE rate (exploratory)

Method: Bootstrap mediation (10,000 iterations) with 95% CI.
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_prp_trials, load_wcst_trials

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/mediation_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_BOOTSTRAP = 10000
ALPHA = 0.05

print("=" * 80)
print("EX-GAUSSIAN MEDIATION ANALYSIS")
print("=" * 80)
print(f"\nBootstrap iterations: {N_BOOTSTRAP}")
print(f"Confidence level: {(1-ALPHA)*100}%\n")

# ---------------------------------------------------------------------------
# Load master covariates (UCLA, gender, age)
# ---------------------------------------------------------------------------
master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)
if "ucla_total" not in master.columns and "ucla_score" in master.columns:
    master["ucla_total"] = master["ucla_score"]

# Use gender_normalized if available
if 'gender_normalized' in master.columns:
    master['gender'] = master['gender_normalized'].fillna('').astype(str).str.strip().str.lower()
else:
    master['gender'] = master['gender'].fillna('').astype(str).str.strip().str.lower()
master["gender_male"] = (master["gender"] == "male").astype(int)

# ---------------------------------------------------------------------------
# Load Ex-Gaussian parameter files
# ---------------------------------------------------------------------------
stroop_exg = pd.read_csv(RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/exgaussian_parameters.csv")
if stroop_exg["participant_id"].dtype == "O" and stroop_exg["participant_id"].iloc[0].startswith("\ufeff"):
    stroop_exg["participant_id"] = stroop_exg["participant_id"].str.replace("\ufeff", "")
stroop_exg = stroop_exg[["participant_id", "mu", "sigma", "tau"]].rename(
    columns={"mu": "stroop_mu", "sigma": "stroop_sigma", "tau": "stroop_tau"}
)

prp_exg = pd.read_csv(RESULTS_DIR / "analysis_outputs/mechanism_analysis/exgaussian/prp_exgaussian_parameters.csv")
if prp_exg["participant_id"].dtype == "O" and prp_exg["participant_id"].iloc[0].startswith("\ufeff"):
    prp_exg["participant_id"] = prp_exg["participant_id"].str.replace("\ufeff", "")

prp_exg["prp_tau_long"] = prp_exg["long_tau"]
prp_exg["prp_sigma_long"] = prp_exg["long_sigma"]
prp_exg["prp_mu_long"] = prp_exg["long_mu"]
prp_exg["prp_tau_short"] = prp_exg["short_tau"]
prp_exg["prp_sigma_short"] = prp_exg["short_sigma"]
prp_exg["prp_mu_short"] = prp_exg["short_mu"]
prp_exg["prp_tau_bottleneck"] = prp_exg["short_tau"] - prp_exg["long_tau"]
prp_exg["prp_sigma_bottleneck"] = prp_exg["short_sigma"] - prp_exg["long_sigma"]
prp_exg["prp_mu_bottleneck"] = prp_exg["short_mu"] - prp_exg["long_mu"]
prp_exg = prp_exg[
    [
        "participant_id",
        "prp_tau_long",
        "prp_sigma_long",
        "prp_mu_long",
        "prp_tau_short",
        "prp_sigma_short",
        "prp_mu_short",
        "prp_tau_bottleneck",
        "prp_sigma_bottleneck",
        "prp_mu_bottleneck",
    ]
].copy()

# ---------------------------------------------------------------------------
# WCST PE rate and RT variability
# ---------------------------------------------------------------------------
wcst_trials, wcst_summary = load_wcst_trials(use_cache=True)
rt_col_wcst = "reactionTimeMs" if "reactionTimeMs" in wcst_trials.columns else "rt_ms" if "rt_ms" in wcst_trials.columns else None
if not rt_col_wcst:
    raise KeyError("WCST trials missing reaction time column")

if "isPE" not in wcst_trials.columns and "is_pe" not in wcst_trials.columns:
    # parse extra as fallback
    def _parse_wcst_extra(extra_str):
        if not isinstance(extra_str, str):
            return {}
        try:
            return ast.literal_eval(extra_str)
        except Exception:
            return {}
    wcst_trials["extra_dict"] = wcst_trials["extra"].apply(_parse_wcst_extra) if "extra" in wcst_trials.columns else {}
    wcst_trials["isPE"] = wcst_trials.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

wcst_trials["is_pe"] = wcst_trials["isPE"] if "isPE" in wcst_trials.columns else wcst_trials["is_pe"]

wcst_summary_df = wcst_trials.groupby("participant_id").agg(
    pe_count=("is_pe", "sum"),
    total_trials=("is_pe", "count"),
    mean_rt=(rt_col_wcst, "mean"),
    sd_rt=(rt_col_wcst, "std"),
).reset_index()
wcst_summary_df["pe_rate"] = (wcst_summary_df["pe_count"] / wcst_summary_df["total_trials"]) * 100

# ---------------------------------------------------------------------------
# PRP trials for bottleneck effect and variability
# ---------------------------------------------------------------------------
prp_trials, prp_summary = load_prp_trials(use_cache=True, rt_min=0, rt_max=10000, require_t1_correct=False, enforce_short_long_only=True)

def bin_soa(soa):
    if soa <= 150:
        return "short"
    elif soa >= 1200:
        return "long"
    elif 300 <= soa <= 600:
        return "medium"
    return "other"

prp_trials["soa_bin"] = prp_trials["soa"].apply(bin_soa)
prp_trials = prp_trials[prp_trials["soa_bin"].isin(["short", "long"])].copy()

prp_summary_rt = (
    prp_trials.groupby(["participant_id", "soa_bin"])
    .agg(t2_rt_mean=("t2_rt", "mean"), t2_rt_sd=("t2_rt", "std"), n_trials=("t2_rt", "count"))
    .reset_index()
)
prp_wide = prp_summary_rt.pivot(index="participant_id", columns="soa_bin", values=["t2_rt_mean", "t2_rt_sd"]).reset_index()
prp_wide.columns = ["_".join(col).strip("_") for col in prp_wide.columns.values]

if "t2_rt_mean_short" in prp_wide.columns and "t2_rt_mean_long" in prp_wide.columns:
    prp_wide["bottleneck_effect"] = prp_wide["t2_rt_mean_short"] - prp_wide["t2_rt_mean_long"]

merge_cols = ["participant_id"]
for col in ["bottleneck_effect", "t2_rt_sd_long", "t2_rt_sd_short"]:
    if col in prp_wide.columns:
        merge_cols.append(col)
prp_wide = prp_wide[merge_cols]

# ---------------------------------------------------------------------------
# Merge master with derived features
# ---------------------------------------------------------------------------
master = (
    master.merge(stroop_exg, on="participant_id", how="left")
    .merge(prp_exg, on="participant_id", how="left")
    .merge(wcst_summary_df[["participant_id", "pe_rate", "sd_rt"]], on="participant_id", how="left")
    .merge(prp_wide, on="participant_id", how="left")
)

master = master.dropna(subset=["ucla_total", "gender_male"]).copy()

print(f"  Final N = {len(master)} (with complete UCLA and gender)")
print(f"  Males: {master['gender_male'].sum()}, Females: {(1 - master['gender_male']).sum()}")
print(f"\n  Available Ex-Gaussian columns:")
exg_cols = [col for col in master.columns if "tau" in col or "sigma" in col or "bottleneck" in col or "t2_" in col]
print(f"    {', '.join(exg_cols)}\n")

# ---------------------------------------------------------------------------
# Bootstrap Mediation Function
# ---------------------------------------------------------------------------
def bootstrap_mediation(X, M, Y, n_bootstrap=10000, alpha=0.05):
    """
    Bootstrap mediation analysis returning paths and CI for indirect effect.
    """
    X = np.array(X, dtype=float)
    M = np.array(M, dtype=float)
    Y = np.array(Y, dtype=float)

    valid = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
    X = X[valid]
    M = M[valid]
    Y = Y[valid]

    n = len(X)
    if n < 10:
        return None

    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(X, M)

    X_mat = np.column_stack([np.ones(n), M, X])
    try:
        beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
        b_path = beta[1]
        c_prime_path = beta[2]
    except Exception:
        return None

    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(X, Y)

    indirect_effect = slope_a * b_path

    indirect_boots = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(n), n_samples=n, replace=True)
        X_boot = X[indices]
        M_boot = M[indices]
        Y_boot = Y[indices]

        slope_a_boot, _, _, _, _ = stats.linregress(X_boot, M_boot)
        X_mat_boot = np.column_stack([np.ones(n), M_boot, X_boot])
        try:
            beta_boot = np.linalg.lstsq(X_mat_boot, Y_boot, rcond=None)[0]
            b_path_boot = beta_boot[1]
            indirect_boots.append(slope_a_boot * b_path_boot)
        except Exception:
            continue

    if len(indirect_boots) < 100:
        return None

    ci_lower = np.percentile(indirect_boots, alpha / 2 * 100)
    ci_upper = np.percentile(indirect_boots, (1 - alpha / 2) * 100)

    proportion_mediated = indirect_effect / slope_c if abs(slope_c) > 1e-6 else np.nan

    return {
        "n": n,
        "a_path": slope_a,
        "a_path_p": p_a,
        "b_path": b_path,
        "c_path": slope_c,
        "c_path_p": p_c,
        "c_prime_path": c_prime_path,
        "indirect_effect": indirect_effect,
        "indirect_ci_lower": ci_lower,
        "indirect_ci_upper": ci_upper,
        "indirect_sig": "Yes" if ci_lower > 0 or ci_upper < 0 else "No",
        "proportion_mediated": proportion_mediated,
    }

# ---------------------------------------------------------------------------
# Mediation analyses
# ---------------------------------------------------------------------------
results_list = []

print("\nPathway 1: PRP τ (long) -> Bottleneck effect (by gender)")
for gender_flag, gender_name in [(1, "Male"), (0, "Female")]:
    df_sub = master[master["gender_male"] == gender_flag].copy()
    if df_sub.empty:
        continue
    res = bootstrap_mediation(
        df_sub["ucla_total"],
        df_sub["prp_tau_long"],
        df_sub["bottleneck_effect"] if "bottleneck_effect" in df_sub.columns else np.nan,
        n_bootstrap=N_BOOTSTRAP,
        alpha=ALPHA,
    )
    if res:
        print(f"  {gender_name}: n={res['n']}, indirect={res['indirect_effect']:.3f} [{res['indirect_ci_lower']:.3f}, {res['indirect_ci_upper']:.3f}] (sig={res['indirect_sig']})")
        results_list.append({"pathway": "UCLA -> PRP τ_long -> Bottleneck effect", "gender": gender_name, **res})

print("\nPathway 2: PRP τ (bottleneck) -> T2 RT variability (by gender)")
for gender_flag, gender_name in [(1, "Male"), (0, "Female")]:
    df_sub = master[master["gender_male"] == gender_flag].copy()
    var_col = "t2_rt_sd_long" if "t2_rt_sd_long" in master.columns else None
    if df_sub.empty or not var_col:
        continue
    res = bootstrap_mediation(
        df_sub["ucla_total"],
        df_sub["prp_tau_bottleneck"],
        df_sub[var_col],
        n_bootstrap=N_BOOTSTRAP,
        alpha=ALPHA,
    )
    if res:
        print(f"  {gender_name}: n={res['n']}, indirect={res['indirect_effect']:.3f} [{res['indirect_ci_lower']:.3f}, {res['indirect_ci_upper']:.3f}] (sig={res['indirect_sig']})")
        results_list.append({"pathway": "UCLA -> PRP τ_bottleneck -> PRP T2 RT SD (long)", "gender": gender_name, **res})

print("\nPathway 3: WCST RT SD -> PE rate (exploratory, by gender)")
for gender_flag, gender_name in [(1, "Male"), (0, "Female")]:
    df_sub = master[master["gender_male"] == gender_flag].copy()
    if df_sub.empty:
        continue
    res = bootstrap_mediation(
        df_sub["ucla_total"],
        df_sub["sd_rt"],
        df_sub["pe_rate"],
        n_bootstrap=N_BOOTSTRAP,
        alpha=ALPHA,
    )
    if res:
        print(f"  {gender_name}: n={res['n']}, indirect={res['indirect_effect']:.3f} [{res['indirect_ci_lower']:.3f}, {res['indirect_ci_upper']:.3f}] (sig={res['indirect_sig']})")
        results_list.append({"pathway": "UCLA -> WCST_RT_SD -> WCST PE rate", "gender": gender_name, **res})

# ---------------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------------
if results_list:
    results_df = pd.DataFrame(results_list)
    output_file = OUTPUT_DIR / "exgaussian_mediation_detailed.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_file}")

    summary_cols = [
        "pathway",
        "gender",
        "n",
        "a_path",
        "a_path_p",
        "b_path",
        "c_path",
        "c_path_p",
        "indirect_effect",
        "indirect_ci_lower",
        "indirect_ci_upper",
        "indirect_sig",
        "proportion_mediated",
    ]
    summary_df = results_df[summary_cols].copy()
    summary_file = OUTPUT_DIR / "exgaussian_mediation_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Summary saved to: {summary_file}")

print("\n" + "=" * 80)
print("MEDIATION ANALYSIS COMPLETE")
print("=" * 80)
