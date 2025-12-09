"""
PRP Bottleneck Shape Analysis
=============================
Analyzes the shape of the PRP effect curve (T2 RT across SOA levels)
to model individual differences in dual-task coordination.

Traditional PRP analysis uses only extreme SOAs (short vs long).
This analysis examines the full SOA-RT function shape:
- Linear slope (rate of RT decrease)
- Asymptotic recovery (final RT level)
- Curve shape (exponential vs linear)

Hypotheses:
- UCLA may affect recovery rate (slope) but not asymptote
- Individual differences in curve shape may reveal processing capacity
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader_utils import (
    load_master_dataset, ensure_participant_id,
    RESULTS_DIR, ANALYSIS_OUTPUT_DIR,
    DEFAULT_RT_MIN, PRP_RT_MAX
)

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "prp_bottleneck_shape"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOA_VALUES = [50, 150, 300, 600, 1200]  # Standard SOA levels


def load_prp_summary_by_soa():
    """Load PRP summary with RT by each SOA level."""
    df = pd.read_csv(RESULTS_DIR / "3_cognitive_tests_summary.csv", encoding='utf-8')
    df = ensure_participant_id(df)

    # Filter PRP tests
    prp = df[df['testName'] == 'prp'].copy()

    # Extract RT by SOA columns
    soa_cols = [f'rt2_soa_{soa}' for soa in SOA_VALUES]
    available_cols = [c for c in soa_cols if c in prp.columns]

    if len(available_cols) < 3:
        raise KeyError(f"Insufficient SOA columns. Found: {available_cols}")

    prp = prp[['participant_id'] + available_cols].dropna()
    print(f"PRP summary loaded: N={len(prp)}, SOA columns: {available_cols}")

    return prp


def exponential_decay(x, a, b, c):
    """Exponential decay function: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c


def linear_function(x, m, b):
    """Linear function: y = m * x + b"""
    return m * x + b


def fit_individual_curves(prp_df):
    """Fit curve parameters for each participant."""
    results = []

    for pid in prp_df['participant_id'].unique():
        pdata = prp_df[prp_df['participant_id'] == pid].iloc[0]

        # Get RT values for each SOA
        rt_values = []
        valid_soas = []
        for soa in SOA_VALUES:
            col = f'rt2_soa_{soa}'
            if col in pdata.index and not pd.isna(pdata[col]):
                rt_values.append(pdata[col])
                valid_soas.append(soa)

        if len(valid_soas) < 3:
            continue

        rt_values = np.array(rt_values)
        soa_array = np.array(valid_soas)

        # Linear fit
        try:
            slope, intercept, r_linear, p_linear, _ = stats.linregress(soa_array, rt_values)
        except (ValueError, TypeError, RuntimeError) as e:
            import warnings
            warnings.warn(f"Linear fit failed for {pid}: {e}")
            slope, intercept, r_linear, p_linear = np.nan, np.nan, np.nan, np.nan

        # Exponential fit (with bounds)
        try:
            # Initial guesses: a=500, b=0.001, c=700
            popt, _ = curve_fit(
                exponential_decay, soa_array, rt_values,
                p0=[500, 0.002, 700],
                bounds=([0, 0, 0], [2000, 0.1, 2000]),
                maxfev=5000
            )
            a_exp, b_exp, c_exp = popt
            # R-squared for exponential
            y_pred = exponential_decay(soa_array, *popt)
            ss_res = np.sum((rt_values - y_pred) ** 2)
            ss_tot = np.sum((rt_values - np.mean(rt_values)) ** 2)
            r_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except (ValueError, TypeError, RuntimeError) as e:
            import warnings
            warnings.warn(f"Exponential fit failed for {pid}: {e}")
            a_exp, b_exp, c_exp, r_exp = np.nan, np.nan, np.nan, np.nan

        # Calculate derived metrics
        rt_short = rt_values[0] if len(rt_values) > 0 else np.nan
        rt_long = rt_values[-1] if len(rt_values) > 0 else np.nan
        bottleneck_traditional = rt_short - rt_long

        # Recovery half-life (if exponential fit succeeded)
        half_life = np.log(2) / b_exp if not np.isnan(b_exp) and b_exp > 0 else np.nan

        results.append({
            'participant_id': pid,
            # Raw values
            'rt_soa_50': pdata.get('rt2_soa_50', np.nan),
            'rt_soa_150': pdata.get('rt2_soa_150', np.nan),
            'rt_soa_300': pdata.get('rt2_soa_300', np.nan),
            'rt_soa_600': pdata.get('rt2_soa_600', np.nan),
            'rt_soa_1200': pdata.get('rt2_soa_1200', np.nan),
            # Traditional bottleneck
            'bottleneck_traditional': bottleneck_traditional,
            # Linear fit
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_r2': r_linear ** 2 if not np.isnan(r_linear) else np.nan,
            # Exponential fit
            'exp_amplitude': a_exp,  # Initial RT elevation
            'exp_decay_rate': b_exp,  # Recovery speed
            'exp_asymptote': c_exp,   # Final RT level
            'exp_r2': r_exp,
            'recovery_half_life': half_life,
            # Fit comparison
            'better_fit': 'exponential' if (not np.isnan(r_exp) and r_exp > r_linear**2) else 'linear'
        })

    return pd.DataFrame(results)


def analyze_group_curve_shape(curve_df):
    """Analyze group-level curve shape characteristics."""
    valid = curve_df.dropna(subset=['linear_slope', 'exp_asymptote'])

    results = {
        'n': len(valid),
        # Linear parameters
        'mean_slope': valid['linear_slope'].mean(),
        'std_slope': valid['linear_slope'].std(),
        'mean_intercept': valid['linear_intercept'].mean(),
        # Exponential parameters
        'mean_amplitude': valid['exp_amplitude'].mean(),
        'std_amplitude': valid['exp_amplitude'].std(),
        'mean_decay_rate': valid['exp_decay_rate'].mean(),
        'std_decay_rate': valid['exp_decay_rate'].std(),
        'mean_asymptote': valid['exp_asymptote'].mean(),
        'std_asymptote': valid['exp_asymptote'].std(),
        'mean_half_life': valid['recovery_half_life'].mean(),
        # Fit comparison
        'pct_exponential_better': (valid['better_fit'] == 'exponential').mean() * 100,
        # Traditional
        'mean_bottleneck': valid['bottleneck_traditional'].mean(),
        'std_bottleneck': valid['bottleneck_traditional'].std()
    }

    # Test if slope significantly < 0
    t_slope, p_slope = stats.ttest_1samp(valid['linear_slope'].dropna(), 0)
    results['slope_t'] = t_slope
    results['slope_p'] = p_slope

    return results


def analyze_ucla_curve_parameters(master_df, curve_df):
    """Test whether UCLA predicts curve parameters."""
    merged = master_df.merge(curve_df, on='participant_id', how='inner')

    required = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress',
                'age', 'gender_male', 'linear_slope', 'exp_asymptote']
    merged = merged.dropna(subset=required)

    if len(merged) < 30:
        return None

    # Standardize
    for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
        merged[f'z_{col}'] = (merged[col] - merged[col].mean()) / merged[col].std()

    outcomes = {
        'linear_slope': 'Recovery rate (linear)',
        'exp_asymptote': 'Asymptotic RT',
        'exp_decay_rate': 'Recovery speed (exp)',
        'bottleneck_traditional': 'Traditional bottleneck',
        'recovery_half_life': 'Recovery half-life'
    }

    results = []
    for outcome, label in outcomes.items():
        if outcome not in merged.columns:
            continue

        outcome_data = merged[[outcome, 'z_ucla_score', 'z_dass_depression',
                               'z_dass_anxiety', 'z_dass_stress', 'z_age',
                               'gender_male']].dropna()

        if len(outcome_data) < 30:
            continue

        formula = f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
        try:
            model = smf.ols(formula, data=outcome_data).fit()
        except (ValueError, np.linalg.LinAlgError) as e:
            import warnings
            warnings.warn(f"OLS fitting failed for {outcome}: {e}")
            continue

        results.append({
            'outcome': outcome,
            'outcome_label': label,
            'n': len(outcome_data),
            'ucla_coef': model.params.get('z_ucla_score', np.nan),
            'ucla_se': model.bse.get('z_ucla_score', np.nan),
            'ucla_p': model.pvalues.get('z_ucla_score', np.nan),
            'model_r2': model.rsquared
        })

    return pd.DataFrame(results)


def analyze_parameter_correlations(curve_df):
    """Analyze correlations between curve parameters."""
    params = ['linear_slope', 'exp_asymptote', 'exp_decay_rate',
              'bottleneck_traditional', 'recovery_half_life']
    available = [p for p in params if p in curve_df.columns]

    corr_matrix = curve_df[available].corr()
    return corr_matrix


def main():
    print("=" * 60)
    print("PRP Bottleneck Shape Analysis")
    print("=" * 60)

    # Load data
    prp_df = load_prp_summary_by_soa()

    # Fit individual curves
    print("\n[1] Fitting individual SOA-RT curves")
    print("-" * 40)
    curve_df = fit_individual_curves(prp_df)
    print(f"  Valid curve fits: N={len(curve_df)}")
    curve_df.to_csv(OUTPUT_DIR / "individual_curve_parameters.csv", index=False, encoding='utf-8-sig')

    # Group statistics
    print("\n[2] Group-level curve characteristics")
    print("-" * 40)
    group_results = analyze_group_curve_shape(curve_df)
    print(f"  Mean linear slope: {group_results['mean_slope']:.3f} ms/ms SOA")
    print(f"  Slope significantly < 0: t={group_results['slope_t']:.2f}, p={group_results['slope_p']:.4f}")
    print(f"  Mean traditional bottleneck: {group_results['mean_bottleneck']:.1f} ms")
    print(f"  Mean asymptotic RT: {group_results['mean_asymptote']:.1f} ms")
    print(f"  Mean recovery half-life: {group_results['mean_half_life']:.1f} ms")
    print(f"  Exponential fit better: {group_results['pct_exponential_better']:.1f}%")

    pd.DataFrame([group_results]).to_csv(
        OUTPUT_DIR / "group_curve_statistics.csv", index=False, encoding='utf-8-sig'
    )

    # Parameter correlations
    print("\n[3] Curve parameter correlations")
    print("-" * 40)
    corr_matrix = analyze_parameter_correlations(curve_df)
    print(corr_matrix.round(3).to_string())
    corr_matrix.to_csv(OUTPUT_DIR / "parameter_correlations.csv", encoding='utf-8-sig')

    # UCLA → curve parameters
    print("\n[4] UCLA → Curve parameters (DASS-controlled)")
    print("-" * 40)

    master = load_master_dataset(use_cache=True)
    ucla_results = analyze_ucla_curve_parameters(master, curve_df)

    if ucla_results is not None and len(ucla_results) > 0:
        print("\n  UCLA effects on curve parameters:")
        for _, row in ucla_results.iterrows():
            sig = "*" if row['ucla_p'] < 0.05 else ""
            print(f"    {row['outcome_label']}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f}{sig}")

        ucla_results.to_csv(
            OUTPUT_DIR / "ucla_curve_parameters.csv", index=False, encoding='utf-8-sig'
        )
    else:
        print("  Insufficient data for UCLA analysis")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
PRP Curve Shape Analysis Results:
- Valid participants: N={len(curve_df)}
- Mean linear slope: {group_results['mean_slope']:.3f} (negative = normal PRP effect)
- Exponential fit preferred: {group_results['pct_exponential_better']:.1f}% of participants

Curve Parameters:
- Asymptotic RT (final level): {group_results['mean_asymptote']:.0f} ms
- Recovery half-life: {group_results['mean_half_life']:.0f} ms

UCLA Effects (DASS-controlled):
""")
    if ucla_results is not None and len(ucla_results) > 0:
        for _, row in ucla_results.iterrows():
            sig = "SIGNIFICANT" if row['ucla_p'] < 0.05 else "n.s."
            print(f"  - {row['outcome_label']}: b={row['ucla_coef']:.3f}, p={row['ucla_p']:.4f} ({sig})")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
