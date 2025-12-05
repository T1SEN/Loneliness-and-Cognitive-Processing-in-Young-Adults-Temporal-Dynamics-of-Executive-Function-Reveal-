"""
Extended Causal DAG Analysis
============================
인과 구조 추정 및 민감도 분석

Features:
1. 여러 경쟁 DAG 모델 비교
2. 인과 효과 추정 (backdoor adjustment)
3. 미관측 교란변수 민감도 분석
4. 성별별 인과 경로 분석

Note: DoWhy 패키지가 없을 경우 수동 구현으로 대체
"""

import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Add parent to path for imports
_this_file = Path(__file__) if '__file__' in dir() else Path('analysis/causal_dag_extended.py')
sys.path.insert(0, str(_this_file.parent))
from utils.data_loader_utils import load_master_dataset, ANALYSIS_OUTPUT_DIR

OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "extended_analyses" / "causal_dag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXTENDED CAUSAL DAG ANALYSIS")
print("=" * 70)


# ============================================================================
# Competing causal models
# ============================================================================
"""
Competing DAG Models:

Model 1: Mood-Driven (DASS → EF, UCLA spurious)
   UCLA ← U → DASS → EF
   (UCLA effect is confounded by unmeasured U)

Model 2: Full Mediation (UCLA → DASS → EF)
   UCLA → DASS → EF
   (DASS fully mediates UCLA effect)

Model 3: Parallel Paths (UCLA → EF + DASS → EF)
   UCLA → EF
   DASS → EF
   UCLA → DASS
   (Both have direct effects on EF)

Model 4: Moderated Mediation (UCLA → DASS → EF varies by Gender)
   UCLA → DASS → EF
   Gender moderates UCLA → EF and/or DASS → EF paths
"""


# ============================================================================
# Helper functions
# ============================================================================

def backdoor_adjustment(df, treatment, outcome, confounders, formula_base=None):
    """
    Estimate causal effect using backdoor adjustment (covariate adjustment).
    Returns ATE and confidence interval.
    """
    if formula_base is None:
        confounder_str = ' + '.join(confounders) if confounders else '1'
        formula = f"{outcome} ~ {treatment} + {confounder_str}"
    else:
        formula = formula_base

    valid_df = df.dropna(subset=[treatment, outcome] + confounders)

    if len(valid_df) < 20:
        return {'ate': np.nan, 'se': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p': np.nan, 'n': len(valid_df)}

    model = smf.ols(formula, data=valid_df).fit()

    ate = model.params[treatment]
    se = model.bse[treatment]
    ci = model.conf_int().loc[treatment]
    p = model.pvalues[treatment]

    return {
        'ate': ate,
        'se': se,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'p': p,
        'n': len(valid_df),
        'r2': model.rsquared
    }


def mediation_decomposition(df, treatment, mediator, outcome, confounders):
    """
    Decompose total effect into direct and indirect (mediated) effects.
    Uses product-of-coefficients method.
    """
    valid_df = df.dropna(subset=[treatment, mediator, outcome] + confounders)

    if len(valid_df) < 20:
        return {'total': np.nan, 'direct': np.nan, 'indirect': np.nan, 'mediation_pct': np.nan}

    # a path: Treatment → Mediator
    confounder_str = ' + '.join(confounders) if confounders else '1'
    formula_a = f"{mediator} ~ {treatment} + {confounder_str}"
    model_a = smf.ols(formula_a, data=valid_df).fit()
    a = model_a.params[treatment]

    # b path: Mediator → Outcome (controlling for treatment)
    formula_b = f"{outcome} ~ {mediator} + {treatment} + {confounder_str}"
    model_b = smf.ols(formula_b, data=valid_df).fit()
    b = model_b.params[mediator]
    direct = model_b.params[treatment]

    # Total effect
    formula_total = f"{outcome} ~ {treatment} + {confounder_str}"
    model_total = smf.ols(formula_total, data=valid_df).fit()
    total = model_total.params[treatment]

    indirect = a * b
    mediation_pct = (indirect / total * 100) if total != 0 else np.nan

    return {
        'a_path': a,
        'b_path': b,
        'indirect': indirect,
        'direct': direct,
        'total': total,
        'mediation_pct': mediation_pct,
        'n': len(valid_df)
    }


def sensitivity_analysis_unmeasured_confounder(df, treatment, outcome, confounders, gamma_range=[1.0, 1.5, 2.0, 2.5]):
    """
    Simplified sensitivity analysis for unmeasured confounding.
    Tests how strong an unmeasured confounder would need to be to nullify the effect.
    """
    valid_df = df.dropna(subset=[treatment, outcome] + confounders)

    if len(valid_df) < 20:
        return []

    # Get unadjusted effect
    confounder_str = ' + '.join(confounders) if confounders else '1'
    formula = f"{outcome} ~ {treatment} + {confounder_str}"
    model = smf.ols(formula, data=valid_df).fit()

    observed_effect = model.params[treatment]
    observed_se = model.bse[treatment]

    results = []
    for gamma in gamma_range:
        # Sensitivity: How much would effect change if there's unmeasured confounding?
        # Simplified: adjusted_effect = observed_effect / gamma
        adjusted_effect = observed_effect / gamma
        adjusted_se = observed_se * gamma  # Conservative

        results.append({
            'gamma': gamma,
            'observed_effect': observed_effect,
            'adjusted_effect': adjusted_effect,
            'still_significant': abs(adjusted_effect) > 1.96 * adjusted_se
        })

    return results


def compare_causal_models(df, treatment, outcome, mediators, confounders):
    """
    Compare different causal model specifications.
    """
    valid_df = df.dropna(subset=[treatment, outcome] + mediators + confounders)

    if len(valid_df) < 20:
        return []

    results = []

    # Model 1: Treatment → Outcome (no mediators, no confounders)
    try:
        model1 = smf.ols(f"{outcome} ~ {treatment}", data=valid_df).fit()
        results.append({
            'model': 'Unadjusted',
            'treatment_effect': model1.params[treatment],
            'p_value': model1.pvalues[treatment],
            'r2': model1.rsquared,
            'aic': model1.aic,
            'n': len(valid_df)
        })
    except Exception:
        pass

    # Model 2: Treatment → Outcome + Confounders
    try:
        confounder_str = ' + '.join(confounders)
        model2 = smf.ols(f"{outcome} ~ {treatment} + {confounder_str}", data=valid_df).fit()
        results.append({
            'model': 'Covariate-Adjusted',
            'treatment_effect': model2.params[treatment],
            'p_value': model2.pvalues[treatment],
            'r2': model2.rsquared,
            'aic': model2.aic,
            'n': len(valid_df)
        })
    except Exception:
        pass

    # Model 3: Treatment → Outcome + Confounders + Mediators
    try:
        mediator_str = ' + '.join(mediators)
        model3 = smf.ols(f"{outcome} ~ {treatment} + {confounder_str} + {mediator_str}", data=valid_df).fit()
        results.append({
            'model': 'Fully-Adjusted',
            'treatment_effect': model3.params[treatment],
            'p_value': model3.pvalues[treatment],
            'r2': model3.rsquared,
            'aic': model3.aic,
            'n': len(valid_df)
        })
    except Exception:
        pass

    # Model 4: With interaction (Gender moderation)
    try:
        formula4 = f"{outcome} ~ {treatment} * C(gender_male) + {confounder_str}"
        model4 = smf.ols(formula4, data=valid_df).fit()
        results.append({
            'model': 'Gender-Moderated',
            'treatment_effect': model4.params[treatment],
            'p_value': model4.pvalues[treatment],
            'r2': model4.rsquared,
            'aic': model4.aic,
            'n': len(valid_df)
        })
    except Exception:
        pass

    return results


# ============================================================================
# Load data
# ============================================================================
print("\n[1] Loading data...")

df = load_master_dataset()

# Standardize key variables
for col in ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']:
    if col in df.columns:
        df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std()

# Create gender indicator
df['gender_male'] = (df['gender'] == 'male').astype(int)

# Define outcomes
outcomes = {
    'pe_rate': 'WCST PE Rate',
    'prp_bottleneck': 'PRP Bottleneck',
    'wcst_accuracy': 'WCST Accuracy'
}
outcomes = {k: v for k, v in outcomes.items() if k in df.columns}

print(f"   N = {len(df)}")
print(f"   Outcomes: {list(outcomes.keys())}")


# ============================================================================
# 2. Backdoor Adjustment Analysis
# ============================================================================
print("\n[2] Causal Effect Estimation (Backdoor Adjustment)...")

backdoor_results = []
confounders = ['z_age']
mediators = ['z_dass_depression', 'z_dass_anxiety', 'z_dass_stress']

for outcome_col, outcome_label in outcomes.items():
    # Unadjusted
    result = backdoor_adjustment(df, 'z_ucla_score', outcome_col, [])
    result['outcome'] = outcome_label
    result['adjustment'] = 'None'
    backdoor_results.append(result)

    # Age-adjusted
    result = backdoor_adjustment(df, 'z_ucla_score', outcome_col, ['z_age'])
    result['outcome'] = outcome_label
    result['adjustment'] = 'Age'
    backdoor_results.append(result)

    # DASS-adjusted (if DASS is confounder, not mediator)
    result = backdoor_adjustment(df, 'z_ucla_score', outcome_col, ['z_age'] + mediators)
    result['outcome'] = outcome_label
    result['adjustment'] = 'Age + DASS'
    backdoor_results.append(result)

backdoor_df = pd.DataFrame(backdoor_results)
backdoor_df.to_csv(OUTPUT_DIR / "backdoor_adjustment_results.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: backdoor_adjustment_results.csv")

print("\n   UCLA → EF Causal Effect Estimates:")
for outcome in outcomes.values():
    print(f"   {outcome}:")
    outcome_data = backdoor_df[backdoor_df['outcome'] == outcome]
    for _, row in outcome_data.iterrows():
        sig = "*" if row['p'] < 0.05 else ""
        print(f"     {row['adjustment']}: ATE = {row['ate']:.3f}, p = {row['p']:.4f} {sig}")


# ============================================================================
# 3. Mediation Decomposition
# ============================================================================
print("\n[3] Mediation Decomposition (UCLA → DASS → EF)...")

mediation_results = []

for outcome_col, outcome_label in outcomes.items():
    # Through depression
    result = mediation_decomposition(df, 'z_ucla_score', 'z_dass_depression', outcome_col, ['z_age'])
    result['outcome'] = outcome_label
    result['mediator'] = 'DASS Depression'
    mediation_results.append(result)

    # Through anxiety
    result = mediation_decomposition(df, 'z_ucla_score', 'z_dass_anxiety', outcome_col, ['z_age'])
    result['outcome'] = outcome_label
    result['mediator'] = 'DASS Anxiety'
    mediation_results.append(result)

    # Through stress
    result = mediation_decomposition(df, 'z_ucla_score', 'z_dass_stress', outcome_col, ['z_age'])
    result['outcome'] = outcome_label
    result['mediator'] = 'DASS Stress'
    mediation_results.append(result)

mediation_df = pd.DataFrame(mediation_results)
mediation_df.to_csv(OUTPUT_DIR / "mediation_decomposition.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: mediation_decomposition.csv")

print("\n   Mediation Summary:")
for outcome in outcomes.values():
    print(f"   {outcome}:")
    outcome_data = mediation_df[mediation_df['outcome'] == outcome]
    for _, row in outcome_data.iterrows():
        if pd.notna(row['mediation_pct']):
            print(f"     via {row['mediator']}: {row['mediation_pct']:.1f}% mediated")


# ============================================================================
# 4. Model Comparison
# ============================================================================
print("\n[4] Comparing Competing Causal Models...")

model_comparison_results = []

for outcome_col, outcome_label in outcomes.items():
    results = compare_causal_models(
        df,
        treatment='z_ucla_score',
        outcome=outcome_col,
        mediators=mediators,
        confounders=['z_age']
    )

    for r in results:
        r['outcome'] = outcome_label
        model_comparison_results.append(r)

model_comparison_df = pd.DataFrame(model_comparison_results)
model_comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: model_comparison.csv")

print("\n   Model Comparison by AIC (lower is better):")
for outcome in outcomes.values():
    print(f"   {outcome}:")
    outcome_data = model_comparison_df[model_comparison_df['outcome'] == outcome].sort_values('aic')
    for _, row in outcome_data.iterrows():
        print(f"     {row['model']}: AIC = {row['aic']:.1f}, R² = {row['r2']:.3f}")


# ============================================================================
# 5. Sensitivity Analysis for Unmeasured Confounding
# ============================================================================
print("\n[5] Sensitivity Analysis for Unmeasured Confounding...")

sensitivity_results = []

for outcome_col, outcome_label in outcomes.items():
    results = sensitivity_analysis_unmeasured_confounder(
        df, 'z_ucla_score', outcome_col, ['z_age']
    )

    for r in results:
        r['outcome'] = outcome_label
        sensitivity_results.append(r)

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_unmeasured_confounding.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: sensitivity_unmeasured_confounding.csv")

print("\n   Sensitivity to Unmeasured Confounding:")
print("   (γ = multiplicative bias from unmeasured confounder)")
for outcome in outcomes.values():
    print(f"   {outcome}:")
    outcome_data = sensitivity_df[sensitivity_df['outcome'] == outcome]
    for _, row in outcome_data.iterrows():
        status = "Still significant" if row['still_significant'] else "Not significant"
        print(f"     γ = {row['gamma']}: {status}")


# ============================================================================
# 6. Gender-Stratified Causal Analysis
# ============================================================================
print("\n[6] Gender-Stratified Causal Analysis...")

gender_causal_results = []

for gender in ['male', 'female']:
    gender_df = df[df['gender'] == gender]

    for outcome_col, outcome_label in outcomes.items():
        # Unadjusted
        result = backdoor_adjustment(gender_df, 'z_ucla_score', outcome_col, [])
        result['outcome'] = outcome_label
        result['gender'] = gender
        result['adjustment'] = 'None'
        gender_causal_results.append(result)

        # DASS-adjusted
        result = backdoor_adjustment(gender_df, 'z_ucla_score', outcome_col, ['z_age'] + mediators)
        result['outcome'] = outcome_label
        result['gender'] = gender
        result['adjustment'] = 'Age + DASS'
        gender_causal_results.append(result)

gender_causal_df = pd.DataFrame(gender_causal_results)
gender_causal_df.to_csv(OUTPUT_DIR / "gender_stratified_causal.csv", index=False, encoding='utf-8-sig')
print(f"   Saved: gender_stratified_causal.csv")

print("\n   Gender-Stratified Causal Effects (DASS-adjusted):")
dass_adjusted = gender_causal_df[gender_causal_df['adjustment'] == 'Age + DASS']
for outcome in outcomes.values():
    print(f"   {outcome}:")
    outcome_data = dass_adjusted[dass_adjusted['outcome'] == outcome]
    for _, row in outcome_data.iterrows():
        sig = "*" if row['p'] < 0.05 else ""
        print(f"     {row['gender'].capitalize()}: ATE = {row['ate']:.3f}, p = {row['p']:.4f} {sig}")


# ============================================================================
# 7. Visualization
# ============================================================================
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 7a. Backdoor adjustment comparison
ax = axes[0, 0]
backdoor_pivot = backdoor_df.pivot(index='outcome', columns='adjustment', values='ate')
backdoor_pivot.plot(kind='bar', ax=ax, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('')
ax.set_ylabel('Average Treatment Effect')
ax.set_title('UCLA → EF Effect by Adjustment Strategy', fontweight='bold')
ax.legend(title='Adjustment')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 7b. Mediation paths
ax = axes[0, 1]
med_pivot = mediation_df.pivot(index='outcome', columns='mediator', values='mediation_pct')
med_pivot.plot(kind='bar', ax=ax, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('')
ax.set_ylabel('% Effect Mediated')
ax.set_title('Mediation Through DASS Components', fontweight='bold')
ax.legend(title='Mediator', fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 7c. Model comparison (AIC)
ax = axes[1, 0]
model_pivot = model_comparison_df.pivot(index='outcome', columns='model', values='aic')
model_pivot.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_xlabel('')
ax.set_ylabel('AIC (lower is better)')
ax.set_title('Model Comparison by AIC', fontweight='bold')
ax.legend(title='Model', fontsize=8)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 7d. Gender-stratified effects
ax = axes[1, 1]
gender_pivot = gender_causal_df[gender_causal_df['adjustment'] == 'Age + DASS'].pivot(
    index='outcome', columns='gender', values='ate'
)
if len(gender_pivot) > 0:
    gender_pivot.plot(kind='bar', ax=ax, alpha=0.7, color=['blue', 'red'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Causal Effect (DASS-adjusted)')
    ax.set_title('Gender-Stratified UCLA → EF Effects', fontweight='bold')
    ax.legend(title='Gender')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "causal_dag_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: causal_dag_plots.png")


# ============================================================================
# 8. DAG Visualization (Text-based)
# ============================================================================
print("\n[8] Creating DAG summary...")

dag_summary = """
COMPETING CAUSAL MODELS
========================

Model 1: Mood-Driven (DASS → EF, UCLA spurious)
  UCLA ← U → DASS → EF
  Implication: UCLA effect disappears when controlling DASS

Model 2: Full Mediation (UCLA → DASS → EF)
  UCLA → DASS → EF
  Implication: Direct effect = 0, only indirect effect remains

Model 3: Parallel Paths
  UCLA → EF
  UCLA → DASS → EF
  Implication: Both direct and indirect effects present

Model 4: Moderated Mediation
  UCLA → DASS → EF (paths moderated by Gender)
  Implication: Different effects for males vs females

EVIDENCE FROM THIS ANALYSIS:
"""

# Determine which model is best supported
for outcome in outcomes.values():
    dag_summary += f"\n{outcome}:\n"

    # Check if effect disappears with DASS control (Model 1 support)
    none_effect = backdoor_df[(backdoor_df['outcome'] == outcome) & (backdoor_df['adjustment'] == 'None')]['ate'].values
    dass_effect = backdoor_df[(backdoor_df['outcome'] == outcome) & (backdoor_df['adjustment'] == 'Age + DASS')]['ate'].values

    if len(none_effect) > 0 and len(dass_effect) > 0:
        reduction = (1 - abs(dass_effect[0] / none_effect[0])) * 100 if none_effect[0] != 0 else 0
        dag_summary += f"  - Effect reduction with DASS control: {reduction:.1f}%\n"

        if reduction > 80:
            dag_summary += f"  → Supports Model 1 (Mood-Driven) or Model 2 (Full Mediation)\n"
        elif reduction > 50:
            dag_summary += f"  → Supports Model 2/3 (Partial Mediation)\n"
        else:
            dag_summary += f"  → Supports Model 3 (Parallel Paths - direct effect remains)\n"

    # Check gender moderation
    gender_data = gender_causal_df[(gender_causal_df['outcome'] == outcome) & (gender_causal_df['adjustment'] == 'Age + DASS')]
    if len(gender_data) == 2:
        male_ate = gender_data[gender_data['gender'] == 'male']['ate'].values[0]
        female_ate = gender_data[gender_data['gender'] == 'female']['ate'].values[0]
        dag_summary += f"  - Male ATE: {male_ate:.3f}, Female ATE: {female_ate:.3f}\n"

        if np.sign(male_ate) != np.sign(female_ate) or abs(male_ate - female_ate) > 1:
            dag_summary += f"  → Supports Model 4 (Gender Moderation)\n"

with open(OUTPUT_DIR / "dag_summary.txt", 'w', encoding='utf-8') as f:
    f.write(dag_summary)
print(f"   Saved: dag_summary.txt")


# ============================================================================
# 9. Summary Report
# ============================================================================
print("\n" + "=" * 70)
print("CAUSAL DAG ANALYSIS SUMMARY")
print("=" * 70)

print("\n1. Causal Effect Estimates (UCLA → EF):")
print("   (Backdoor-adjusted for Age + DASS)")
dass_results = backdoor_df[backdoor_df['adjustment'] == 'Age + DASS']
for _, row in dass_results.iterrows():
    sig = "SIGNIFICANT" if row['p'] < 0.05 else "not significant"
    print(f"   {row['outcome']}: ATE = {row['ate']:.3f}, p = {row['p']:.4f} ({sig})")

print("\n2. Mediation Analysis:")
print("   Strongest mediation paths:")
for outcome in outcomes.values():
    outcome_med = mediation_df[mediation_df['outcome'] == outcome].sort_values('mediation_pct', ascending=False)
    if len(outcome_med) > 0:
        top = outcome_med.iloc[0]
        print(f"   {outcome}: {top['mediation_pct']:.1f}% via {top['mediator']}")

print("\n3. Model Comparison:")
print("   Best fitting models (lowest AIC):")
for outcome in outcomes.values():
    outcome_models = model_comparison_df[model_comparison_df['outcome'] == outcome].sort_values('aic')
    if len(outcome_models) > 0:
        best = outcome_models.iloc[0]
        print(f"   {outcome}: {best['model']} (AIC = {best['aic']:.1f})")

print("\n4. Gender Differences:")
for outcome in outcomes.values():
    gender_data = gender_causal_df[(gender_causal_df['outcome'] == outcome) & (gender_causal_df['adjustment'] == 'Age + DASS')]
    if len(gender_data) == 2:
        male = gender_data[gender_data['gender'] == 'male'].iloc[0]
        female = gender_data[gender_data['gender'] == 'female'].iloc[0]
        print(f"   {outcome}:")
        print(f"     Male: ATE = {male['ate']:.3f}, p = {male['p']:.4f}")
        print(f"     Female: ATE = {female['ate']:.3f}, p = {female['p']:.4f}")

print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("=" * 70)

print("\nCausal DAG analysis complete!")
