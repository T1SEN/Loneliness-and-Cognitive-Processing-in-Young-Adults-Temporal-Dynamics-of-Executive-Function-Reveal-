"""
Error Burst Clustering Analysis
===============================

Research Question: Do lonely individuals show temporally clustered errors
(error bursts) rather than randomly distributed errors?

This analysis examines error temporal patterns:
  - Runs Test for error clustering
  - Markov transition probabilities: P(Error_t | Error_{t-1}) vs P(Error_t | Correct_{t-1})
  - Error burst metrics: run lengths, clustering coefficient

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

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "error_burst"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TRIALS = 20
MIN_ERRORS = 3


def compute_error_burst_metrics(error_sequence: np.ndarray) -> dict:
    """
    Compute error clustering metrics from a binary error sequence.
    error_sequence: 1 = error, 0 = correct
    """
    n_trials = len(error_sequence)
    n_errors = np.sum(error_sequence)

    if n_trials < MIN_TRIALS or n_errors < MIN_ERRORS:
        return None

    # Compute runs (consecutive sequences of same value)
    runs = []
    current_run = 1
    for i in range(1, len(error_sequence)):
        if error_sequence[i] == error_sequence[i-1]:
            current_run += 1
        else:
            runs.append((error_sequence[i-1], current_run))
            current_run = 1
    runs.append((error_sequence[-1], current_run))

    # Error runs only
    error_runs = [r[1] for r in runs if r[0] == 1]
    n_error_runs = len(error_runs)

    # Runs test for randomness
    n_correct = n_trials - n_errors
    n_runs = len(runs)

    # Expected runs under randomness
    if n_errors > 0 and n_correct > 0:
        expected_runs = 1 + (2 * n_errors * n_correct) / n_trials
        var_runs = (2 * n_errors * n_correct * (2 * n_errors * n_correct - n_trials)) / \
                   (n_trials ** 2 * (n_trials - 1))

        if var_runs > 0:
            z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))  # Two-tailed
        else:
            z_runs = np.nan
            p_runs = np.nan
    else:
        expected_runs = np.nan
        z_runs = np.nan
        p_runs = np.nan

    # Markov transition probabilities
    # P(Error_t | Error_{t-1}) and P(Error_t | Correct_{t-1})
    error_after_error = 0
    error_after_correct = 0
    n_post_error = 0
    n_post_correct = 0

    for i in range(1, len(error_sequence)):
        if error_sequence[i-1] == 1:  # Previous was error
            n_post_error += 1
            if error_sequence[i] == 1:
                error_after_error += 1
        else:  # Previous was correct
            n_post_correct += 1
            if error_sequence[i] == 1:
                error_after_correct += 1

    p_error_given_error = error_after_error / n_post_error if n_post_error > 0 else np.nan
    p_error_given_correct = error_after_correct / n_post_correct if n_post_correct > 0 else np.nan

    # Error transition ratio (clustering measure)
    # > 1 means errors cluster, < 1 means errors are dispersed
    if p_error_given_correct > 0:
        error_transition_ratio = p_error_given_error / p_error_given_correct
    else:
        error_transition_ratio = np.nan

    # Clustering coefficient: observed/expected runs
    if expected_runs > 0:
        clustering_coef = expected_runs / n_runs  # > 1 means more clustered
    else:
        clustering_coef = np.nan

    return {
        "n_trials": n_trials,
        "n_errors": n_errors,
        "error_rate": n_errors / n_trials,
        "n_runs": n_runs,
        "expected_runs": expected_runs,
        "z_runs": z_runs,
        "p_runs": p_runs,
        "n_error_runs": n_error_runs,
        "mean_error_run_length": np.mean(error_runs) if error_runs else np.nan,
        "max_error_run_length": np.max(error_runs) if error_runs else 0,
        "p_error_given_error": p_error_given_error,
        "p_error_given_correct": p_error_given_correct,
        "error_transition_ratio": error_transition_ratio,
        "clustering_coef": clustering_coef,
    }


def extract_error_burst_params(trial_df: pd.DataFrame, error_col: str, task_name: str) -> pd.DataFrame:
    """
    For each participant, compute error burst metrics.
    """
    results = []

    for pid, grp in trial_df.groupby("participant_id"):
        grp = grp.sort_values(grp.columns[0]).reset_index(drop=True)

        # Create error sequence (1 = error, 0 = correct)
        if error_col == "isPE":  # WCST perseverative errors
            error_seq = grp[error_col].fillna(False).astype(int).values
        else:
            error_seq = (~grp[error_col].fillna(True)).astype(int).values  # Invert correct to error

        metrics = compute_error_burst_metrics(error_seq)

        if metrics is None:
            continue

        result = {"participant_id": pid}
        for key, value in metrics.items():
            result[f"{task_name}_{key}"] = value

        results.append(result)

    return pd.DataFrame(results)


def run_frequentist_regressions(master_df: pd.DataFrame, outcomes: list, outcome_labels: dict) -> pd.DataFrame:
    """
    Run hierarchical regressions for error burst outcomes.
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
                         burst_summary: pd.DataFrame) -> str:
    """Generate text summary of key findings."""
    lines = [
        "=" * 80,
        "ERROR BURST CLUSTERING ANALYSIS: SUMMARY REPORT",
        "=" * 80,
        "",
        "RESEARCH QUESTION:",
        "Do lonely individuals show temporally clustered errors (error bursts)?",
        "",
        "METHODOLOGY:",
        "- Runs Test: Tests whether error positions are random or clustered",
        "- Markov Transitions: P(Error_t | Error_{t-1}) vs P(Error_t | Correct_{t-1})",
        "- Error Transition Ratio: >1 means errors cluster together",
        "- Clustering Coefficient: >1 means more clustered than expected",
        "",
    ]

    # Burst summary by task
    lines.extend([
        "-" * 80,
        "ERROR CLUSTERING SUMMARY BY TASK:",
        "-" * 80,
    ])

    if not burst_summary.empty:
        for _, row in burst_summary.iterrows():
            lines.append(f"\n  {row['task']}:")
            lines.append(f"    Mean error rate: {row['mean_error_rate']:.3f}")
            lines.append(f"    Mean P(Error|Error): {row['mean_p_error_given_error']:.3f}")
            lines.append(f"    Mean P(Error|Correct): {row['mean_p_error_given_correct']:.3f}")
            lines.append(f"    Mean transition ratio: {row['mean_transition_ratio']:.2f}")
            lines.append(f"    % with significant clustering (runs test p<.05): {row['pct_significant_clustering']:.1f}%")

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
        "If UCLA predicts ERROR TRANSITION RATIO:",
        "  -> Loneliness increases error-to-error transitions (perseveration)",
        "",
        "If UCLA predicts CLUSTERING COEFFICIENT:",
        "  -> Loneliness leads to bursts of momentary disengagement",
        "",
        "If UCLA predicts MAX ERROR RUN LENGTH:",
        "  -> Loneliness leads to extended periods of cognitive failure",
        "",
        "Connection to tau (Ex-Gaussian tail):",
        "  - Tau captures slow responses generally",
        "  - Error bursts capture temporal clustering specifically",
        "  - If both predict UCLA: confirms attentional lapse mechanism",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("ERROR BURST CLUSTERING ANALYSIS")
    print("=" * 60)

    # Load master dataset
    print("\n[1] Loading master dataset...")
    master = load_master_dataset(force_rebuild=False)
    print(f"  Master dataset: N={len(master)}")

    # Load trial-level data and compute error burst metrics
    print("\n[2] Computing error burst metrics...")

    # PRP trials (T2 errors)
    print("  Processing PRP trials...")
    prp_trials, _ = load_prp_trials(force_rebuild=False, require_t2_correct_for_rt=False,
                                    enforce_short_long_only=False)
    if "t2_correct" in prp_trials.columns:
        prp_burst = extract_error_burst_params(prp_trials, "t2_correct", "prp")
    else:
        prp_burst = pd.DataFrame()
    print(f"    PRP: {len(prp_burst)} participants with burst metrics")

    # Stroop trials
    print("  Processing Stroop trials...")
    stroop_trials, _ = load_stroop_trials(force_rebuild=False, require_correct_for_rt=False)
    stroop_burst = extract_error_burst_params(stroop_trials, "correct", "stroop")
    print(f"    Stroop: {len(stroop_burst)} participants with burst metrics")

    # WCST trials (perseverative errors)
    print("  Processing WCST trials (perseverative errors)...")
    wcst_trials, _ = load_wcst_trials(force_rebuild=False)
    wcst_burst = extract_error_burst_params(wcst_trials, "isPE", "wcst")
    print(f"    WCST: {len(wcst_burst)} participants with burst metrics")

    # Merge burst parameters with master dataset
    print("\n[3] Merging burst parameters with master dataset...")
    analysis_df = master.copy()
    for burst_df in [prp_burst, stroop_burst, wcst_burst]:
        if not burst_df.empty:
            analysis_df = analysis_df.merge(burst_df, on="participant_id", how="left")

    print(f"  Final analysis dataset: N={len(analysis_df)}")

    # Save burst parameters
    burst_cols = [c for c in analysis_df.columns if any(x in c for x in
                  ["_n_trials", "_n_errors", "_error_rate", "_n_runs", "_z_runs", "_p_runs",
                   "_error_run", "_p_error_", "_transition_ratio", "_clustering_coef"])]
    burst_export = analysis_df[["participant_id"] + burst_cols].copy()
    burst_export.to_csv(OUTPUT_DIR / "error_burst_parameters.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_DIR / 'error_burst_parameters.csv'}")

    # Summary statistics by task
    print("\n[4] Computing summary statistics...")
    burst_summary = []
    for task, task_name in [("prp", "PRP"), ("stroop", "Stroop"), ("wcst", "WCST")]:
        cols = {
            "error_rate": f"{task}_error_rate",
            "p_error_given_error": f"{task}_p_error_given_error",
            "p_error_given_correct": f"{task}_p_error_given_correct",
            "transition_ratio": f"{task}_error_transition_ratio",
            "p_runs": f"{task}_p_runs",
        }

        if all(c in analysis_df.columns for c in cols.values()):
            n_sig = (analysis_df[cols["p_runs"]] < 0.05).sum()
            n_total = analysis_df[cols["p_runs"]].notna().sum()

            burst_summary.append({
                "task": task_name,
                "n_participants": n_total,
                "mean_error_rate": analysis_df[cols["error_rate"]].mean(),
                "mean_p_error_given_error": analysis_df[cols["p_error_given_error"]].mean(),
                "mean_p_error_given_correct": analysis_df[cols["p_error_given_correct"]].mean(),
                "mean_transition_ratio": analysis_df[cols["transition_ratio"]].mean(),
                "pct_significant_clustering": (n_sig / n_total * 100) if n_total > 0 else 0,
            })

    burst_summary_df = pd.DataFrame(burst_summary)
    if not burst_summary_df.empty:
        burst_summary_df.to_csv(OUTPUT_DIR / "burst_summary_by_task.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'burst_summary_by_task.csv'}")
        for _, row in burst_summary_df.iterrows():
            print(f"    {row['task']}: Mean transition ratio = {row['mean_transition_ratio']:.2f}, "
                  f"{row['pct_significant_clustering']:.1f}% with sig. clustering")

    # Define outcomes for regression
    outcomes = [
        "prp_error_transition_ratio", "prp_clustering_coef", "prp_max_error_run_length",
        "stroop_error_transition_ratio", "stroop_clustering_coef", "stroop_max_error_run_length",
        "wcst_error_transition_ratio", "wcst_clustering_coef", "wcst_max_error_run_length",
    ]

    outcome_labels = {
        "prp_error_transition_ratio": "PRP Error Transition Ratio",
        "prp_clustering_coef": "PRP Clustering Coefficient",
        "prp_max_error_run_length": "PRP Max Error Run",
        "stroop_error_transition_ratio": "Stroop Error Transition Ratio",
        "stroop_clustering_coef": "Stroop Clustering Coefficient",
        "stroop_max_error_run_length": "Stroop Max Error Run",
        "wcst_error_transition_ratio": "WCST Error Transition Ratio",
        "wcst_clustering_coef": "WCST Clustering Coefficient",
        "wcst_max_error_run_length": "WCST Max Error Run (PE)",
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
    bayes_outcomes = [o for o in outcomes if "transition_ratio" in o or "clustering" in o]
    bayes_results = run_bayesian_analysis(analysis_df, bayes_outcomes, outcome_labels)
    if not bayes_results.empty:
        bayes_results.to_csv(OUTPUT_DIR / "bayesian_results.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved: {OUTPUT_DIR / 'bayesian_results.csv'}")

    # Generate summary report
    print("\n[7] Generating summary report...")
    report = create_summary_report(freq_results, bayes_results, burst_summary_df)
    with open(OUTPUT_DIR / "ERROR_BURST_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {OUTPUT_DIR / 'ERROR_BURST_REPORT.txt'}")

    print("\n" + "=" * 60)
    print("ERROR BURST CLUSTERING ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
