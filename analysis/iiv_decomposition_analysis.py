"""
IIV Decomposition Analysis (Intra-Individual Variability)
=========================================================

Research Question: Does UCLA loneliness predict learning rate vs random noise?

This analysis decomposes RT variability into:
  - Intercept (beta_0): Baseline speed
  - Slope (beta_1): Learning/adaptation rate across trials
  - Residual SD (sigma_epsilon): Pure noise after removing learning trend

Tests UCLA associations with each component, controlling for DASS-21.
Includes Bayesian analysis with ROPE/LOO-CV.
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

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "iiv_decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_iiv_parameters(trial_df: pd.DataFrame, rt_col: str, task_name: str) -> pd.DataFrame:
    """
    For each participant, fit RT ~ trial_index and extract:
      - intercept: baseline RT
      - slope: learning rate (negative = improvement)
      - residual_sd: noise after removing linear trend
    """
    results = []

    for pid, grp in trial_df.groupby("participant_id"):
        grp = grp.sort_values("trial_index").reset_index(drop=True)
        n_trials = len(grp)

        if n_trials < 10:
            continue

        # Standardize trial index (0 to 1) for comparable slopes
        grp["trial_std"] = grp["trial_index"] / grp["trial_index"].max()

        try:
            # Fit linear model: RT ~ trial_std
            X = grp["trial_std"].values
            y = grp[rt_col].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            # Residual SD
            predicted = intercept + slope * X
            residuals = y - predicted
            residual_sd = np.std(residuals, ddof=2)

            # Also compute raw CV for comparison
            raw_cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else np.nan

            results.append({
                "participant_id": pid,
                f"{task_name}_intercept": intercept,
                f"{task_name}_slope": slope,
                f"{task_name}_slope_p": p_value,
                f"{task_name}_residual_sd": residual_sd,
                f"{task_name}_raw_cv": raw_cv,
                f"{task_name}_n_trials": n_trials,
                f"{task_name}_r_squared": r_value**2,
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def add_trial_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add trial_index column if not present, ordered within participant."""
    if "trial_index" not in df.columns:
        df = df.sort_values(["participant_id", df.columns[0]])
        df["trial_index"] = df.groupby("participant_id").cumcount() + 1
    return df


def run_frequentist_regressions(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Run hierarchical regressions for each IIV outcome.
    Model: outcome ~ z_ucla * C(gender_male) + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age
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
                    "r_squared_adj": model.rsquared_adj,
                })
        except Exception as e:
            print(f"  Error fitting {outcome}: {e}")
            continue

    return pd.DataFrame(results)


def run_bayesian_analysis(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Bayesian regression with ROPE and LOO-CV for key IIV outcomes.
    ROPE for standardized effects: [-0.1, 0.1]
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

        # Standardize outcome for ROPE interpretation
        y = df_clean[outcome].values
        y_std = (y - np.mean(y)) / np.std(y)

        X_ucla = df_clean["z_ucla"].values
        X_gender = df_clean["gender_male"].values
        X_dass_d = df_clean["z_dass_depression"].values
        X_dass_a = df_clean["z_dass_anxiety"].values
        X_dass_s = df_clean["z_dass_stress"].values
        X_age = df_clean["z_age"].values
        X_interaction = X_ucla * X_gender

        try:
            with pm.Model() as model:
                # Priors
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                b_ucla = pm.Normal("b_ucla", mu=0, sigma=0.5)
                b_gender = pm.Normal("b_gender", mu=0, sigma=0.5)
                b_interaction = pm.Normal("b_ucla_x_gender", mu=0, sigma=0.5)
                b_dass_d = pm.Normal("b_dass_d", mu=0, sigma=0.5)
                b_dass_a = pm.Normal("b_dass_a", mu=0, sigma=0.5)
                b_dass_s = pm.Normal("b_dass_s", mu=0, sigma=0.5)
                b_age = pm.Normal("b_age", mu=0, sigma=0.5)
                sigma = pm.HalfNormal("sigma", sigma=1)

                # Linear model
                mu = (intercept + b_ucla * X_ucla + b_gender * X_gender +
                      b_interaction * X_interaction +
                      b_dass_d * X_dass_d + b_dass_a * X_dass_a +
                      b_dass_s * X_dass_s + b_age * X_age)

                # Likelihood
                likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y_std)

                # Sampling with log likelihood for LOO-CV
                trace = pm.sample(1000, tune=1000, cores=1, random_seed=42,
                                 progressbar=False, return_inferencedata=True,
                                 idata_kwargs={"log_likelihood": True})

                # LOO-CV (requires log_likelihood in trace)
                try:
                    loo = az.loo(trace, pointwise=True)
                    loo_elpd = loo.elpd_loo
                    loo_se = loo.se
                except Exception as loo_err:
                    print(f"    LOO-CV failed: {loo_err}")
                    loo_elpd = np.nan
                    loo_se = np.nan

            # Extract posteriors for key parameters
            for param in ["b_ucla", "b_ucla_x_gender"]:
                posterior = trace.posterior[param].values.flatten()

                # ROPE analysis
                in_rope = np.mean((posterior >= rope_interval[0]) & (posterior <= rope_interval[1]))
                below_rope = np.mean(posterior < rope_interval[0])
                above_rope = np.mean(posterior > rope_interval[1])

                # Probability of direction (effect exists)
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
                    "rope_below": below_rope,
                    "rope_above": above_rope,
                    "prob_direction": prob_direction,
                    "loo_elpd": loo_elpd,
                    "loo_se": loo_se,
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
        "IIV DECOMPOSITION ANALYSIS: SUMMARY REPORT",
        "=" * 80,
        "",
        "RESEARCH QUESTION:",
        "Does UCLA loneliness predict learning rate (slope) vs random noise (residual SD)?",
        "",
        "METHODOLOGY:",
        "- Decomposed RT into: Intercept (baseline), Slope (learning), Residual SD (noise)",
        "- Tested UCLA associations controlling for DASS-21 (depression, anxiety, stress)",
        "- Both Frequentist (OLS) and Bayesian (PyMC with ROPE) approaches",
        "",
        "-" * 80,
        "KEY FREQUENTIST RESULTS (p < 0.05):",
        "-" * 80,
    ]

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
        "If UCLA predicts SLOPE but not RESIDUAL_SD:",
        "  -> Loneliness impairs learning/adaptation capacity",
        "",
        "If UCLA predicts RESIDUAL_SD but not SLOPE:",
        "  -> Loneliness increases random noise (attentional lapses)",
        "",
        "If UCLA predicts BOTH:",
        "  -> Multiple mechanisms (learning + noise)",
        "",
        "If UCLA predicts NEITHER:",
        "  -> Variance effects may be through other pathways (tau, error clustering)",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("IIV DECOMPOSITION ANALYSIS")
    print("=" * 60)

    # Load master dataset
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(force_rebuild=True)
    print(f"  Master dataset: N={len(master)}")

    # Load trial-level data and extract IIV parameters
    print("\n[2] Extracting IIV parameters from trial data...")

    # PRP trials
    print("  Processing PRP trials...")
    prp_trials, prp_info = load_prp_trials(force_rebuild=True, enforce_short_long_only=False)
    prp_trials = add_trial_index(prp_trials)
    prp_iiv = extract_iiv_parameters(prp_trials, "t2_rt", "prp")
    print(f"    PRP: {len(prp_iiv)} participants with IIV parameters")

    # Stroop trials
    print("  Processing Stroop trials...")
    stroop_trials, stroop_info = load_stroop_trials(force_rebuild=True)
    stroop_trials = add_trial_index(stroop_trials)
    stroop_iiv = extract_iiv_parameters(stroop_trials, "rt", "stroop")
    print(f"    Stroop: {len(stroop_iiv)} participants with IIV parameters")

    # WCST trials
    print("  Processing WCST trials...")
    wcst_trials, wcst_info = load_wcst_trials(force_rebuild=True)
    # WCST RT column
    rt_col_wcst = "rt_ms" if "rt_ms" in wcst_trials.columns else "reactionTimeMs"
    wcst_trials = add_trial_index(wcst_trials)
    # Filter valid RT for WCST
    wcst_trials = wcst_trials[wcst_trials[rt_col_wcst].between(100, 5000)]
    wcst_iiv = extract_iiv_parameters(wcst_trials, rt_col_wcst, "wcst")
    print(f"    WCST: {len(wcst_iiv)} participants with IIV parameters")

    # Merge IIV parameters with master dataset
    print("\n[3] Merging IIV parameters with master dataset...")
    analysis_df = master.copy()
    for iiv_df in [prp_iiv, stroop_iiv, wcst_iiv]:
        if not iiv_df.empty:
            analysis_df = analysis_df.merge(iiv_df, on="participant_id", how="left")

    print(f"  Final analysis dataset: N={len(analysis_df)}")

    # Save IIV parameters
    iiv_cols = [c for c in analysis_df.columns if any(x in c for x in ["_intercept", "_slope", "_residual", "_raw_cv", "_n_trials", "_r_squared"])]
    iiv_export = analysis_df[["participant_id"] + iiv_cols].copy()
    iiv_export.to_csv(OUTPUT_DIR / "iiv_parameters.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved IIV parameters: {OUTPUT_DIR / 'iiv_parameters.csv'}")

    # Define outcomes for regression
    outcomes = [
        "prp_intercept", "prp_slope", "prp_residual_sd",
        "stroop_intercept", "stroop_slope", "stroop_residual_sd",
        "wcst_intercept", "wcst_slope", "wcst_residual_sd",
    ]

    outcome_labels = {
        "prp_intercept": "PRP Baseline RT",
        "prp_slope": "PRP Learning Rate",
        "prp_residual_sd": "PRP Residual Noise",
        "stroop_intercept": "Stroop Baseline RT",
        "stroop_slope": "Stroop Learning Rate",
        "stroop_residual_sd": "Stroop Residual Noise",
        "wcst_intercept": "WCST Baseline RT",
        "wcst_slope": "WCST Learning Rate",
        "wcst_residual_sd": "WCST Residual Noise",
    }

    # Run frequentist regressions
    print("\n[4] Running frequentist regressions (DASS-controlled)...")
    freq_results = run_frequentist_regressions(analysis_df, outcomes, outcome_labels)
    freq_results.to_csv(OUTPUT_DIR / "frequentist_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'frequentist_results.csv'}")

    # Print significant frequentist results
    sig_freq = freq_results[(freq_results["p_value"] < 0.05) &
                            (freq_results["parameter"].str.contains("ucla", case=False))]
    if len(sig_freq) > 0:
        print("\n  Significant UCLA effects (p < 0.05):")
        for _, row in sig_freq.iterrows():
            print(f"    {row['outcome_label']}: {row['parameter']}, beta={row['coefficient']:.3f}, p={row['p_value']:.4f}")
    else:
        print("\n  No significant UCLA effects at p < 0.05")

    # Run Bayesian analysis
    print("\n[5] Running Bayesian analysis (ROPE + LOO-CV)...")
    # Focus on key outcomes for Bayesian (computationally intensive)
    bayes_outcomes = ["prp_slope", "prp_residual_sd", "stroop_slope", "stroop_residual_sd",
                      "wcst_slope", "wcst_residual_sd"]
    bayes_results = run_bayesian_analysis(analysis_df, bayes_outcomes, outcome_labels)
    if not bayes_results.empty:
        bayes_results.to_csv(OUTPUT_DIR / "bayesian_results.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'bayesian_results.csv'}")

    # Generate summary report
    print("\n[6] Generating summary report...")
    report = create_summary_report(freq_results, bayes_results)
    with open(OUTPUT_DIR / "IIV_DECOMPOSITION_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {OUTPUT_DIR / 'IIV_DECOMPOSITION_REPORT.txt'}")

    print("\n" + "=" * 60)
    print("IIV DECOMPOSITION ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
