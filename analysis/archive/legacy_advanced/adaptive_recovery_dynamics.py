"""
Adaptive Recovery Dynamics Analysis
====================================

Research Question: Do lonely individuals show impaired dynamic adjustment
after control-demanding events? Track the time-course of recovery.

This analysis examines:
  - Post-error RT time-course (trials 1, 2, 3... after error)
  - Post-conflict decay in Stroop (trials since last incongruent)
  - Proactive vs reactive control indices
  - Recovery speed after setbacks

Tests UCLA associations with DASS-21 control.
Includes Bayesian analysis with ROPE.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import curve_fit

# Bayesian imports
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available; Bayesian analysis will be skipped.")

# Windows encoding fix
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from analysis.utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR
from analysis.utils.trial_data_loader import load_prp_trials, load_stroop_trials, load_wcst_trials

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "adaptive_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_post_event_rt(trial_df: pd.DataFrame, event_col: str, rt_col: str,
                          max_lag: int = 5) -> pd.DataFrame:
    """
    Compute RT at different lags after an event (error, conflict, etc.)
    Returns per-participant post-event RT profile.
    """
    results = []

    for pid, grp in trial_df.groupby("participant_id"):
        grp = grp.sort_values(grp.columns[0]).reset_index(drop=True)

        # Find event positions
        event_positions = np.where(grp[event_col].fillna(False).values)[0]

        if len(event_positions) < 3:  # Need minimum events
            continue

        # Get baseline RT (all correct/non-event trials)
        baseline_rt = grp[~grp[event_col].fillna(False)][rt_col].median()

        # Collect RT at each lag after event
        lag_rts = {lag: [] for lag in range(1, max_lag + 1)}

        for pos in event_positions:
            for lag in range(1, max_lag + 1):
                if pos + lag < len(grp):
                    rt = grp.iloc[pos + lag][rt_col]
                    if pd.notna(rt) and rt > 0:
                        lag_rts[lag].append(rt)

        # Compute mean RT at each lag
        result = {
            "participant_id": pid,
            "baseline_rt": baseline_rt,
            "n_events": len(event_positions),
        }

        for lag in range(1, max_lag + 1):
            if len(lag_rts[lag]) >= 3:
                mean_rt = np.mean(lag_rts[lag])
                result[f"rt_lag{lag}"] = mean_rt
                result[f"rt_lag{lag}_vs_baseline"] = mean_rt - baseline_rt
            else:
                result[f"rt_lag{lag}"] = np.nan
                result[f"rt_lag{lag}_vs_baseline"] = np.nan

        results.append(result)

    return pd.DataFrame(results)


def fit_exponential_recovery(lag_rts: list) -> dict:
    """
    Fit exponential decay to post-event RT recovery.
    RT(t) = baseline + delta * exp(-t/tau)

    Returns tau (recovery time constant) and other parameters.
    """
    if len(lag_rts) < 3 or any(pd.isna(lag_rts)):
        return {"tau": np.nan, "delta": np.nan, "r_squared": np.nan}

    lags = np.arange(1, len(lag_rts) + 1)

    try:
        # Exponential decay function
        def exp_decay(t, delta, tau):
            return delta * np.exp(-t / tau)

        # Initial guess
        delta_init = lag_rts[0] if lag_rts[0] > 0 else 50
        tau_init = 2.0

        popt, _ = curve_fit(exp_decay, lags, lag_rts, p0=[delta_init, tau_init],
                           bounds=([0, 0.5], [500, 10]), maxfev=1000)

        # R-squared
        predicted = exp_decay(lags, *popt)
        ss_res = np.sum((np.array(lag_rts) - predicted) ** 2)
        ss_tot = np.sum((np.array(lag_rts) - np.mean(lag_rts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            "delta": popt[0],  # Initial slowing magnitude
            "tau": popt[1],    # Recovery time constant (higher = slower recovery)
            "r_squared": r_squared,
        }

    except Exception:
        return {"tau": np.nan, "delta": np.nan, "r_squared": np.nan}


def extract_recovery_parameters(post_event_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """
    For each participant, fit exponential recovery curve and extract parameters.
    """
    results = []

    for _, row in post_event_df.iterrows():
        lag_rts = [row.get(f"rt_lag{lag}_vs_baseline", np.nan) for lag in range(1, 6)]

        # Skip if not enough data
        valid_lags = [x for x in lag_rts if pd.notna(x)]
        if len(valid_lags) < 3:
            continue

        recovery_params = fit_exponential_recovery(lag_rts[:len(valid_lags)])

        result = {
            "participant_id": row["participant_id"],
            f"{task_name}_baseline_rt": row["baseline_rt"],
            f"{task_name}_n_events": row["n_events"],
            f"{task_name}_initial_slowing": lag_rts[0] if pd.notna(lag_rts[0]) else np.nan,
            f"{task_name}_recovery_tau": recovery_params["tau"],
            f"{task_name}_recovery_delta": recovery_params["delta"],
            f"{task_name}_recovery_r2": recovery_params["r_squared"],
        }

        # Also add raw lag RTs
        for i, rt in enumerate(lag_rts, 1):
            result[f"{task_name}_rt_lag{i}"] = rt

        results.append(result)

    return pd.DataFrame(results)


def compute_proactive_reactive_index(trial_df: pd.DataFrame, cond_col: str, rt_col: str) -> pd.DataFrame:
    """
    Compute proactive vs reactive control index from Stroop task.
    Proactive: Sustained slowing on all trials (overall mean RT)
    Reactive: Transient slowing after incongruent (CSE effect)
    """
    results = []

    for pid, grp in trial_df.groupby("participant_id"):
        grp = grp.sort_values(grp.columns[0]).reset_index(drop=True)

        if len(grp) < 30:
            continue

        # Add previous trial condition
        grp["prev_cond"] = grp[cond_col].shift(1)

        # CSE: (incongruent after incongruent) - (incongruent after congruent)
        ii = grp[(grp[cond_col] == "incongruent") & (grp["prev_cond"] == "incongruent")][rt_col]
        ci = grp[(grp[cond_col] == "incongruent") & (grp["prev_cond"] == "congruent")][rt_col]

        if len(ii) >= 5 and len(ci) >= 5:
            cse = ii.mean() - ci.mean()  # Negative = conflict adaptation
        else:
            cse = np.nan

        # Overall slowing (proactive)
        mean_rt = grp[rt_col].mean()

        # Reactive index: magnitude of immediate post-conflict slowing
        reactive = grp[(grp["prev_cond"] == "incongruent")][rt_col].mean() - \
                   grp[(grp["prev_cond"] == "congruent")][rt_col].mean()

        results.append({
            "participant_id": pid,
            "stroop_cse": cse,
            "stroop_mean_rt": mean_rt,
            "stroop_reactive_index": reactive if pd.notna(reactive) else np.nan,
        })

    return pd.DataFrame(results)


def run_frequentist_regressions(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Run hierarchical regressions for recovery dynamics outcomes.
    """
    results = []
    formula_template = "{outcome} ~ z_ucla * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age"

    for outcome in outcomes:
        if outcome not in master_df.columns:
            continue

        df_clean = master_df.dropna(subset=[outcome, "z_ucla", "gender_male",
                                            "z_dass_depression", "z_dass_anxiety", "z_dass_stress", "z_age"])

        if len(df_clean) < 30:
            print(f"  Skipping {outcome}: N={len(df_clean)} < 30")
            continue

        formula = formula_template.format(outcome=outcome)

        try:
            model = smf.ols(formula, data=df_clean).fit()

            for param in model.params.index:
                if param == "Intercept":
                    continue
                results.append({
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "parameter": param,
                    "coefficient": model.params[param],
                    "std_error": model.bse[param],
                    "t_value": model.tvalues[param],
                    "p_value": model.pvalues[param],
                    "ci_lower": model.conf_int().loc[param, 0],
                    "ci_upper": model.conf_int().loc[param, 1],
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                })
        except Exception as e:
            print(f"  Error fitting {outcome}: {e}")
            continue

    return pd.DataFrame(results)


def run_bayesian_analysis(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Bayesian regression with ROPE for key outcomes.
    """
    if not HAS_PYMC:
        print("  Skipping Bayesian analysis (PyMC not available)")
        return pd.DataFrame()

    results = []
    rope_interval = (-0.1, 0.1)

    for outcome in outcomes:
        if outcome not in master_df.columns:
            continue

        df_clean = master_df.dropna(subset=[outcome, "z_ucla", "gender_male",
                                            "z_dass_depression", "z_dass_anxiety", "z_dass_stress", "z_age"])

        if len(df_clean) < 30:
            continue

        y = df_clean[outcome].values
        y_std = (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else y

        X_ucla = df_clean["z_ucla"].values
        X_gender = df_clean["gender_male"].values
        X_dass_d = df_clean["z_dass_depression"].values
        X_dass_a = df_clean["z_dass_anxiety"].values
        X_dass_s = df_clean["z_dass_stress"].values
        X_age = df_clean["z_age"].values
        X_interaction = X_ucla * X_gender

        try:
            with pm.Model() as model:
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                b_ucla = pm.Normal("b_ucla", mu=0, sigma=0.5)
                b_gender = pm.Normal("b_gender", mu=0, sigma=0.5)
                b_interaction = pm.Normal("b_ucla_x_gender", mu=0, sigma=0.5)
                b_dass_d = pm.Normal("b_dass_d", mu=0, sigma=0.5)
                b_dass_a = pm.Normal("b_dass_a", mu=0, sigma=0.5)
                b_dass_s = pm.Normal("b_dass_s", mu=0, sigma=0.5)
                b_age = pm.Normal("b_age", mu=0, sigma=0.5)
                sigma = pm.HalfNormal("sigma", sigma=1)

                mu = (intercept + b_ucla * X_ucla + b_gender * X_gender +
                      b_interaction * X_interaction +
                      b_dass_d * X_dass_d + b_dass_a * X_dass_a +
                      b_dass_s * X_dass_s + b_age * X_age)

                likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y_std)

                trace = pm.sample(1000, tune=1000, cores=1, random_seed=42,
                                 progressbar=False, return_inferencedata=True)

            for param in ["b_ucla", "b_ucla_x_gender"]:
                posterior = trace.posterior[param].values.flatten()

                in_rope = np.mean((posterior >= rope_interval[0]) & (posterior <= rope_interval[1]))
                prob_positive = np.mean(posterior > 0)
                prob_direction = max(prob_positive, 1 - prob_positive)

                results.append({
                    "outcome": outcome,
                    "outcome_label": outcome_labels.get(outcome, outcome),
                    "parameter": param,
                    "posterior_mean": np.mean(posterior),
                    "posterior_sd": np.std(posterior),
                    "hdi_3%": np.percentile(posterior, 3),
                    "hdi_97%": np.percentile(posterior, 97),
                    "rope_in": in_rope,
                    "prob_direction": prob_direction,
                    "n_obs": len(df_clean),
                })

        except Exception as e:
            print(f"  Bayesian error for {outcome}: {e}")
            continue

    return pd.DataFrame(results)


def create_summary_report(freq_results: pd.DataFrame, bayes_results: pd.DataFrame) -> str:
    """Generate text summary of key findings."""
    lines = [
        "=" * 80,
        "ADAPTIVE RECOVERY DYNAMICS: SUMMARY REPORT",
        "=" * 80,
        "",
        "RESEARCH QUESTION:",
        "Do lonely individuals show impaired dynamic adjustment after setbacks?",
        "",
        "METHODOLOGY:",
        "- Post-error RT time-course: RT at lags 1-5 after error",
        "- Exponential recovery fit: RT(t) = baseline + delta * exp(-t/tau)",
        "- tau = recovery time constant (higher = slower recovery)",
        "- Proactive vs reactive control indices from Stroop CSE",
        "",
    ]

    lines.extend([
        "-" * 80,
        "KEY FREQUENTIST RESULTS (p < 0.05):",
        "-" * 80,
    ])

    if not freq_results.empty:
        sig_results = freq_results[freq_results["p_value"] < 0.05]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                lines.append(f"  {row['outcome_label']}: {row['parameter']}")
                lines.append(f"    beta = {row['coefficient']:.4f}, p = {row['p_value']:.4f}")
                lines.append(f"    95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
                lines.append("")
        else:
            lines.append("  No significant effects at p < 0.05")

    lines.extend([
        "",
        "-" * 80,
        "BAYESIAN RESULTS (ROPE [-0.1, 0.1]):",
        "-" * 80,
    ])

    if not bayes_results.empty:
        for _, row in bayes_results.iterrows():
            decision = "EQUIVALENT" if row["rope_in"] > 0.95 else (
                "EFFECT EXISTS" if row["rope_in"] < 0.05 else "UNDECIDED"
            )
            lines.append(f"  {row['outcome_label']}: {row['parameter']}")
            lines.append(f"    Posterior mean = {row['posterior_mean']:.4f}")
            lines.append(f"    HDI: [{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")
            lines.append(f"    ROPE: {row['rope_in']*100:.1f}% in, Decision: {decision}")
            lines.append("")

    lines.extend([
        "",
        "=" * 80,
        "INTERPRETATION:",
        "=" * 80,
        "",
        "If UCLA predicts RECOVERY TAU:",
        "  -> Loneliness leads to slower recovery after setbacks",
        "",
        "If UCLA predicts INITIAL SLOWING (delta):",
        "  -> Loneliness increases magnitude of post-error RT increase",
        "",
        "If UCLA predicts CSE (conflict adaptation):",
        "  -> Loneliness impairs trial-by-trial control adjustment",
        "",
        "Connection to variance effects:",
        "  - Slow recovery = prolonged periods of impaired performance",
        "  - Explains variance-without-mean: occasional slow recovery episodes",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ADAPTIVE RECOVERY DYNAMICS ANALYSIS")
    print("=" * 60)

    # Load master dataset
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(force_rebuild=False)
    print(f"  Master dataset: N={len(master)}")

    # Load trial-level data
    print("\n[2] Computing post-event RT profiles...")

    # WCST: Post-error recovery (using isPE as event)
    print("  Processing WCST post-error recovery...")
    wcst_trials, _ = load_wcst_trials(force_rebuild=False)
    rt_col_wcst = "rt_ms" if "rt_ms" in wcst_trials.columns else "reactionTimeMs"
    wcst_trials = wcst_trials[wcst_trials[rt_col_wcst].between(100, 5000)]

    # Use isPE as error event
    wcst_post_error = compute_post_event_rt(wcst_trials, "isPE", rt_col_wcst)
    wcst_recovery = extract_recovery_parameters(wcst_post_error, "wcst")
    print(f"    WCST: {len(wcst_recovery)} participants with recovery parameters")

    # Stroop: Proactive/reactive control
    print("  Processing Stroop proactive/reactive control...")
    stroop_trials, _ = load_stroop_trials(force_rebuild=False)
    cond_col = "type" if "type" in stroop_trials.columns else "condition"

    stroop_control = compute_proactive_reactive_index(stroop_trials, cond_col, "rt")
    print(f"    Stroop: {len(stroop_control)} participants with control indices")

    # Merge with master dataset
    print("\n[3] Merging parameters with master dataset...")
    analysis_df = master.copy()
    for param_df in [wcst_recovery, stroop_control]:
        if not param_df.empty:
            analysis_df = analysis_df.merge(param_df, on="participant_id", how="left")

    print(f"  Final analysis dataset: N={len(analysis_df)}")

    # Save parameters
    recovery_cols = [c for c in analysis_df.columns if any(x in c for x in
                     ["_recovery_", "_initial_slowing", "_rt_lag", "_cse", "_reactive"])]
    recovery_export = analysis_df[["participant_id"] + recovery_cols].copy()
    recovery_export.to_csv(OUTPUT_DIR / "recovery_parameters.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'recovery_parameters.csv'}")

    # Define outcomes
    outcomes = [
        "wcst_recovery_tau", "wcst_recovery_delta", "wcst_initial_slowing",
        "stroop_cse", "stroop_reactive_index",
    ]

    outcome_labels = {
        "wcst_recovery_tau": "WCST Recovery Time Constant",
        "wcst_recovery_delta": "WCST Initial Slowing Magnitude",
        "wcst_initial_slowing": "WCST Post-Error Slowing (lag 1)",
        "stroop_cse": "Stroop Conflict Adaptation (CSE)",
        "stroop_reactive_index": "Stroop Reactive Control Index",
    }

    # Run frequentist regressions
    print("\n[4] Running frequentist regressions (DASS-controlled)...")
    freq_results = run_frequentist_regressions(analysis_df, outcomes, outcome_labels)
    freq_results.to_csv(OUTPUT_DIR / "frequentist_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'frequentist_results.csv'}")

    # Print significant UCLA results
    sig_ucla = freq_results[(freq_results["p_value"] < 0.05) &
                            (freq_results["parameter"].str.contains("ucla", case=False))]
    if len(sig_ucla) > 0:
        print("\n  Significant UCLA effects (p < 0.05):")
        for _, row in sig_ucla.iterrows():
            print(f"    {row['outcome_label']}: {row['parameter']}, beta={row['coefficient']:.3f}, p={row['p_value']:.4f}")
    else:
        print("\n  No significant UCLA effects at p < 0.05")

    # Run Bayesian analysis
    print("\n[5] Running Bayesian analysis (ROPE)...")
    bayes_outcomes = ["wcst_recovery_tau", "stroop_cse"]
    bayes_results = run_bayesian_analysis(analysis_df, bayes_outcomes, outcome_labels)
    if not bayes_results.empty:
        bayes_results.to_csv(OUTPUT_DIR / "bayesian_results.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'bayesian_results.csv'}")

    # Generate summary report
    print("\n[6] Generating summary report...")
    report = create_summary_report(freq_results, bayes_results)
    with open(OUTPUT_DIR / "RECOVERY_DYNAMICS_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {OUTPUT_DIR / 'RECOVERY_DYNAMICS_REPORT.txt'}")

    print("\n" + "=" * 60)
    print("ADAPTIVE RECOVERY DYNAMICS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
