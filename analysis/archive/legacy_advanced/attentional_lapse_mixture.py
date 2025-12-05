"""
Attentional Lapse Mixture Modeling
==================================

Research Question: Can we identify discrete "lapse states" in RT distributions,
and does loneliness increase the frequency of these lapses?

This analysis fits a Gaussian Mixture Model (2 components) to RT distributions:
  - Component 1: Engaged state (faster, lower variance)
  - Component 2: Lapse state (slower, higher variance)

Extracts per-participant:
  - Lapse probability (pi_2)
  - Engaged mean RT (mu_1)
  - Lapse mean RT (mu_2)
  - State separation (mu_2 - mu_1)

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
from sklearn.mixture import GaussianMixture

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

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "attentional_lapse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRIALS = 30  # Minimum trials for reliable mixture fitting


def fit_gaussian_mixture(rt_values: np.ndarray, n_components: int = 2) -> dict:
    """
    Fit a 2-component GMM to RT values.
    Returns parameters for engaged (faster) and lapse (slower) states.
    """
    rt_values = rt_values[~np.isnan(rt_values)]

    if len(rt_values) < MIN_TRIALS:
        return None

    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        gmm.fit(rt_values.reshape(-1, 1))

        # Identify engaged (faster) vs lapse (slower) component
        means = gmm.means_.flatten()
        idx_engaged = np.argmin(means)
        idx_lapse = np.argmax(means)

        # Extract parameters
        result = {
            "engaged_mean": means[idx_engaged],
            "lapse_mean": means[idx_lapse],
            "engaged_sd": np.sqrt(gmm.covariances_[idx_engaged, 0, 0]),
            "lapse_sd": np.sqrt(gmm.covariances_[idx_lapse, 0, 0]),
            "engaged_prob": gmm.weights_[idx_engaged],
            "lapse_prob": gmm.weights_[idx_lapse],
            "state_separation": means[idx_lapse] - means[idx_engaged],
            "bic": gmm.bic(rt_values.reshape(-1, 1)),
            "aic": gmm.aic(rt_values.reshape(-1, 1)),
            "n_trials": len(rt_values),
            "converged": gmm.converged_,
        }

        # Compare with single-component model
        gmm_single = GaussianMixture(n_components=1, random_state=42)
        gmm_single.fit(rt_values.reshape(-1, 1))
        result["bic_single"] = gmm_single.bic(rt_values.reshape(-1, 1))
        result["bic_diff"] = result["bic_single"] - result["bic"]  # Positive = 2-component better

        return result

    except Exception:
        return None


def extract_mixture_parameters(trial_df: pd.DataFrame, rt_col: str, task_name: str) -> pd.DataFrame:
    """
    For each participant, fit GMM and extract lapse parameters.
    """
    results = []

    for pid, grp in trial_df.groupby("participant_id"):
        rt_values = grp[rt_col].dropna().values

        params = fit_gaussian_mixture(rt_values)

        if params is None:
            continue

        result = {"participant_id": pid}
        for key, value in params.items():
            result[f"{task_name}_{key}"] = value

        results.append(result)

    return pd.DataFrame(results)


def run_frequentist_regressions(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Run hierarchical regressions for lapse probability outcomes.
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

        # Standardize outcome
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
                below_rope = np.mean(posterior < rope_interval[0])
                above_rope = np.mean(posterior > rope_interval[1])
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
                    "n_obs": len(df_clean),
                })

        except Exception as e:
            print(f"  Bayesian error for {outcome}: {e}")
            continue

    return pd.DataFrame(results)


def create_summary_report(freq_results: pd.DataFrame, bayes_results: pd.DataFrame,
                         model_comparison: pd.DataFrame) -> str:
    """Generate text summary of key findings."""
    lines = [
        "=" * 80,
        "ATTENTIONAL LAPSE MIXTURE MODEL: SUMMARY REPORT",
        "=" * 80,
        "",
        "RESEARCH QUESTION:",
        "Can we identify discrete 'lapse states' in RT distributions, and does",
        "loneliness increase the frequency of these lapses?",
        "",
        "METHODOLOGY:",
        "- Fit 2-component Gaussian Mixture Model to RT distributions per participant",
        "- Component 1: Engaged state (faster, lower variance)",
        "- Component 2: Lapse state (slower, higher variance)",
        "- Extracted: Lapse probability, state separation, engaged/lapse means",
        "- Tested UCLA associations controlling for DASS-21",
        "",
    ]

    # Model comparison summary
    lines.extend([
        "-" * 80,
        "MODEL COMPARISON (2-component vs 1-component GMM):",
        "-" * 80,
    ])

    if not model_comparison.empty:
        for task in model_comparison["task"].unique():
            task_data = model_comparison[model_comparison["task"] == task]
            pct_better = task_data["pct_2comp_better"].values[0]
            mean_bic_diff = task_data["mean_bic_diff"].values[0]
            lines.append(f"  {task}: {pct_better:.1f}% favor 2-component (mean BIC diff = {mean_bic_diff:.1f})")

    lines.extend([
        "",
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
        "If UCLA predicts LAPSE PROBABILITY:",
        "  -> Loneliness increases discrete attentional disengagement episodes",
        "",
        "If UCLA predicts STATE SEPARATION:",
        "  -> Loneliness increases magnitude of lapses when they occur",
        "",
        "Comparison with Ex-Gaussian tau:",
        "  - Lapse probability = discrete state transitions",
        "  - Tau = continuous distribution tail",
        "  - If lapse_prob predicts UCLA better than tau: discrete model preferred",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ATTENTIONAL LAPSE MIXTURE MODEL ANALYSIS")
    print("=" * 60)

    # Load master dataset
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(force_rebuild=False)
    print(f"  Master dataset: N={len(master)}")

    # Load trial-level data and fit mixture models
    print("\n[2] Fitting Gaussian Mixture Models to RT distributions...")

    # PRP trials
    print("  Processing PRP trials...")
    prp_trials, _ = load_prp_trials(force_rebuild=False, enforce_short_long_only=False)
    prp_mixture = extract_mixture_parameters(prp_trials, "t2_rt", "prp")
    print(f"    PRP: {len(prp_mixture)} participants with mixture parameters")

    # Stroop trials
    print("  Processing Stroop trials...")
    stroop_trials, _ = load_stroop_trials(force_rebuild=False)
    stroop_mixture = extract_mixture_parameters(stroop_trials, "rt", "stroop")
    print(f"    Stroop: {len(stroop_mixture)} participants with mixture parameters")

    # WCST trials
    print("  Processing WCST trials...")
    wcst_trials, _ = load_wcst_trials(force_rebuild=False)
    rt_col_wcst = "rt_ms" if "rt_ms" in wcst_trials.columns else "reactionTimeMs"
    wcst_trials = wcst_trials[wcst_trials[rt_col_wcst].between(100, 5000)]
    wcst_mixture = extract_mixture_parameters(wcst_trials, rt_col_wcst, "wcst")
    print(f"    WCST: {len(wcst_mixture)} participants with mixture parameters")

    # Merge mixture parameters with master dataset
    print("\n[3] Merging mixture parameters with master dataset...")
    analysis_df = master.copy()
    for mix_df in [prp_mixture, stroop_mixture, wcst_mixture]:
        if not mix_df.empty:
            analysis_df = analysis_df.merge(mix_df, on="participant_id", how="left")

    print(f"  Final analysis dataset: N={len(analysis_df)}")

    # Save mixture parameters
    mix_cols = [c for c in analysis_df.columns if any(x in c for x in
                ["_lapse_prob", "_engaged_", "_lapse_mean", "_state_sep", "_bic"])]
    mix_export = analysis_df[["participant_id"] + mix_cols].copy()
    mix_export.to_csv(OUTPUT_DIR / "mixture_parameters.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'mixture_parameters.csv'}")

    # Model comparison (2-component vs 1-component)
    print("\n[4] Model comparison summary...")
    model_comparison = []
    for task, df in [("PRP", prp_mixture), ("Stroop", stroop_mixture), ("WCST", wcst_mixture)]:
        if df.empty:
            continue
        bic_diff_col = f"{task.lower()}_bic_diff"
        if bic_diff_col in df.columns:
            bic_diffs = df[bic_diff_col].dropna()
            pct_better = (bic_diffs > 0).mean() * 100  # Positive = 2-component better
            model_comparison.append({
                "task": task,
                "n_participants": len(bic_diffs),
                "pct_2comp_better": pct_better,
                "mean_bic_diff": bic_diffs.mean(),
                "median_bic_diff": bic_diffs.median(),
            })

    model_comparison_df = pd.DataFrame(model_comparison)
    if not model_comparison_df.empty:
        model_comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'model_comparison.csv'}")
        for _, row in model_comparison_df.iterrows():
            print(f"    {row['task']}: {row['pct_2comp_better']:.1f}% favor 2-component model")

    # Define outcomes for regression
    outcomes = [
        "prp_lapse_prob", "prp_state_separation", "prp_engaged_mean", "prp_lapse_mean",
        "stroop_lapse_prob", "stroop_state_separation", "stroop_engaged_mean", "stroop_lapse_mean",
        "wcst_lapse_prob", "wcst_state_separation", "wcst_engaged_mean", "wcst_lapse_mean",
    ]

    outcome_labels = {
        "prp_lapse_prob": "PRP Lapse Probability",
        "prp_state_separation": "PRP State Separation",
        "prp_engaged_mean": "PRP Engaged Mean RT",
        "prp_lapse_mean": "PRP Lapse Mean RT",
        "stroop_lapse_prob": "Stroop Lapse Probability",
        "stroop_state_separation": "Stroop State Separation",
        "stroop_engaged_mean": "Stroop Engaged Mean RT",
        "stroop_lapse_mean": "Stroop Lapse Mean RT",
        "wcst_lapse_prob": "WCST Lapse Probability",
        "wcst_state_separation": "WCST State Separation",
        "wcst_engaged_mean": "WCST Engaged Mean RT",
        "wcst_lapse_mean": "WCST Lapse Mean RT",
    }

    # Run frequentist regressions
    print("\n[5] Running frequentist regressions (DASS-controlled)...")
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
    print("\n[6] Running Bayesian analysis (ROPE)...")
    bayes_outcomes = [o for o in outcomes if "lapse_prob" in o or "state_sep" in o]
    bayes_results = run_bayesian_analysis(analysis_df, bayes_outcomes, outcome_labels)
    if not bayes_results.empty:
        bayes_results.to_csv(OUTPUT_DIR / "bayesian_results.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'bayesian_results.csv'}")

    # Generate summary report
    print("\n[7] Generating summary report...")
    report = create_summary_report(freq_results, bayes_results, model_comparison_df)
    with open(OUTPUT_DIR / "LAPSE_MIXTURE_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {OUTPUT_DIR / 'LAPSE_MIXTURE_REPORT.txt'}")

    print("\n" + "=" * 60)
    print("ATTENTIONAL LAPSE MIXTURE MODEL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
