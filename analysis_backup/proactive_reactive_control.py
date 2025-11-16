"""
Proactive vs Reactive Control Analysis

Research Question:
Do lonely individuals differ in cognitive control strategy (proactive vs reactive),
and does control strategy moderate UCLA→EF relationships?

Theory:
- Proactive control: Sustained preparatory attention (slower, stable RT)
- Reactive control: Just-in-time stimulus-driven attention (faster, variable RT)

Metrics:
- Task duration: Longer duration → more proactive (cautious, preparatory)
- RT variability (CV): Higher CV → more reactive (inconsistent, stimulus-driven)

Hypotheses:
1. UCLA predicts reactive strategy (higher CV, shorter duration due to disengagement)
2. Reactive strategy moderates UCLA→PE relationship (vulnerable group)
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/advanced_analyses/proactive_reactive_control")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("PROACTIVE VS REACTIVE CONTROL ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

# Demographics
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
gender_map = {'남성': 'male', '여성': 'female'}
participants['gender'] = participants['gender'].map(gender_map)
participants['gender_male'] = (participants['gender'] == 'male').astype(int)
participants = participants.rename(columns={'participantId': 'participant_id'})

# UCLA & DASS
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')
surveys = surveys.rename(columns={'participantId': 'participant_id'})

ucla_data = surveys[surveys['surveyName'] == 'ucla'].copy()
ucla_data['ucla_total'] = pd.to_numeric(ucla_data['score'], errors='coerce')
ucla_data = ucla_data[['participant_id', 'ucla_total']].dropna()

dass_data = surveys[surveys['surveyName'] == 'dass'].copy()
dass_data['score_A'] = pd.to_numeric(dass_data['score_A'], errors='coerce')
dass_data['score_S'] = pd.to_numeric(dass_data['score_S'], errors='coerce')
dass_data['score_D'] = pd.to_numeric(dass_data['score_D'], errors='coerce')
dass_data['dass_total'] = dass_data[['score_A', 'score_S', 'score_D']].sum(axis=1)
dass_data = dass_data[['participant_id', 'dass_total']].dropna()

# Trial-level data
wcst_trials = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding='utf-8-sig')
stroop_trials = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding='utf-8-sig')
prp_trials = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding='utf-8-sig')

# Normalize column names
for df in [wcst_trials, stroop_trials, prp_trials]:
    if 'participantId' in df.columns and 'participant_id' not in df.columns:
        df.rename(columns={'participantId': 'participant_id'}, inplace=True)
    elif 'participantId' in df.columns and 'participant_id' in df.columns:
        df.drop(columns=['participantId'], inplace=True)

print(f"  Data loaded for {len(participants)} participants")

# ============================================================================
# 2. COMPUTE CONTROL STRATEGY METRICS
# ============================================================================
print("\n[2/6] Computing control strategy metrics...")

# WCST
def _parse_wcst_extra(extra_str):
    if not isinstance(extra_str, str):
        return False
    try:
        extra_dict = ast.literal_eval(extra_str)
        return extra_dict.get('isPE', False)
    except (ValueError, SyntaxError):
        return False

wcst_trials['is_pe'] = wcst_trials['extra'].apply(_parse_wcst_extra)
wcst_trials['rt_valid'] = wcst_trials['reactionTimeMs'] > 0

wcst_metrics = wcst_trials[wcst_trials['rt_valid']].groupby('participant_id').agg({
    'reactionTimeMs': ['mean', 'std', 'count'],
    'is_pe': 'mean'
}).reset_index()
wcst_metrics.columns = ['participant_id', 'wcst_mean_rt', 'wcst_sd_rt', 'wcst_n_trials', 'wcst_pe_rate']
wcst_metrics['wcst_cv'] = wcst_metrics['wcst_sd_rt'] / wcst_metrics['wcst_mean_rt']
# Approximate total duration as n_trials × mean_rt (in seconds)
wcst_metrics['wcst_duration'] = wcst_metrics['wcst_n_trials'] * wcst_metrics['wcst_mean_rt'] / 1000

# Stroop
stroop_trials['rt_valid'] = stroop_trials['rt'] > 0
stroop_metrics = stroop_trials[stroop_trials['rt_valid']].groupby('participant_id').agg({
    'rt': ['mean', 'std', 'count'],
    'trial': lambda x: len(x),  # use trial count as proxy
    'correct': lambda x: 1 - x.mean()  # error rate
}).reset_index()
stroop_metrics.columns = ['participant_id', 'stroop_mean_rt', 'stroop_sd_rt', 'stroop_n_trials', 'stroop_trial_count', 'stroop_error_rate']
stroop_metrics['stroop_cv'] = stroop_metrics['stroop_sd_rt'] / stroop_metrics['stroop_mean_rt']

# PRP
prp_trials['t2_rt_valid'] = prp_trials['t2_rt'] > 0
prp_metrics = prp_trials[prp_trials['t2_rt_valid']].groupby('participant_id').agg({
    't2_rt': ['mean', 'std', 'count'],
    'idx': lambda x: len(x)
}).reset_index()
prp_metrics.columns = ['participant_id', 'prp_mean_rt', 'prp_sd_rt', 'prp_n_trials', 'prp_trial_count']
prp_metrics['prp_cv'] = prp_metrics['prp_sd_rt'] / prp_metrics['prp_mean_rt']

print(f"  WCST: Mean duration = {wcst_metrics['wcst_duration'].mean():.1f}s, Mean CV = {wcst_metrics['wcst_cv'].mean():.3f}")
print(f"  Stroop: Mean CV = {stroop_metrics['stroop_cv'].mean():.3f}")
print(f"  PRP: Mean CV = {prp_metrics['prp_cv'].mean():.3f}")

# ============================================================================
# 3. MERGE AND CREATE MASTER DATASET
# ============================================================================
print("\n[3/6] Merging datasets...")

master = wcst_metrics.merge(stroop_metrics, on='participant_id', how='outer')
master = master.merge(prp_metrics, on='participant_id', how='outer')
master = master.merge(ucla_data, on='participant_id', how='left')
master = master.merge(dass_data, on='participant_id', how='left')
master = master.merge(participants[['participant_id', 'gender_male', 'age']], on='participant_id', how='left')
master = master.dropna(subset=['ucla_total'])

print(f"  Complete cases: {len(master)}")

# ============================================================================
# 4. TEST UCLA → CONTROL STRATEGY
# ============================================================================
print("\n[4/6] Testing UCLA → control strategy relationships...")

# Correlations
strategy_vars = ['wcst_duration', 'wcst_cv', 'stroop_cv', 'prp_cv']
strategy_results = []

for var in strategy_vars:
    if var in master.columns:
        valid = master[[var, 'ucla_total']].dropna()
        if len(valid) >= 20:
            r, p = stats.pearsonr(valid['ucla_total'], valid[var])
            strategy_results.append({
                'Variable': var,
                'r': r,
                'p': p,
                'N': len(valid),
                'Sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            })

strategy_df = pd.DataFrame(strategy_results)
print("\n  UCLA × Control Strategy Correlations:")
print(strategy_df.to_string(index=False))

# ============================================================================
# 5. MODERATION ANALYSIS: CONTROL STRATEGY × UCLA → PE
# ============================================================================
print("\n[5/6] Testing moderation: Does control strategy moderate UCLA→PE?")

# Z-score predictors for interpretability
for var in ['ucla_total', 'wcst_cv', 'wcst_duration']:
    if var in master.columns:
        master[f'z_{var}'] = (master[var] - master[var].mean()) / master[var].std()

moderation_results = []

# Test WCST CV moderation
if 'z_wcst_cv' in master.columns and 'wcst_pe_rate' in master.columns:
    data_cv = master[['wcst_pe_rate', 'z_ucla_total', 'z_wcst_cv', 'gender_male', 'age']].dropna()
    if len(data_cv) >= 30:
        model_cv = smf.ols('wcst_pe_rate ~ z_ucla_total * z_wcst_cv + gender_male + age', data=data_cv).fit()
        interaction_coef = model_cv.params['z_ucla_total:z_wcst_cv']
        interaction_p = model_cv.pvalues['z_ucla_total:z_wcst_cv']

        moderation_results.append({
            'Moderator': 'WCST CV (Reactive)',
            'Outcome': 'WCST PE Rate',
            'Interaction β': interaction_coef,
            'p': interaction_p,
            'N': len(data_cv),
            'Sig': '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else ''
        })

        print(f"\n  WCST CV × UCLA → PE:")
        print(f"    Interaction: β={interaction_coef:.3f}, p={interaction_p:.3f}")
        print(f"    N={len(data_cv)}")

# Test WCST duration moderation
if 'z_wcst_duration' in master.columns and 'wcst_pe_rate' in master.columns:
    data_dur = master[['wcst_pe_rate', 'z_ucla_total', 'z_wcst_duration', 'gender_male', 'age']].dropna()
    if len(data_dur) >= 30:
        model_dur = smf.ols('wcst_pe_rate ~ z_ucla_total * z_wcst_duration + gender_male + age', data=data_dur).fit()
        interaction_coef = model_dur.params['z_ucla_total:z_wcst_duration']
        interaction_p = model_dur.pvalues['z_ucla_total:z_wcst_duration']

        moderation_results.append({
            'Moderator': 'WCST Duration (Proactive)',
            'Outcome': 'WCST PE Rate',
            'Interaction β': interaction_coef,
            'p': interaction_p,
            'N': len(data_dur),
            'Sig': '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else ''
        })

        print(f"\n  WCST Duration × UCLA → PE:")
        print(f"    Interaction: β={interaction_coef:.3f}, p={interaction_p:.3f}")
        print(f"    N={len(data_dur)}")

moderation_df = pd.DataFrame(moderation_results)

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

# Plot 1: UCLA × Control Strategy
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (var, title) in enumerate([
    ('wcst_duration', 'WCST Task Duration'),
    ('wcst_cv', 'WCST RT Variability (CV)'),
    ('stroop_cv', 'Stroop RT Variability (CV)'),
    ('prp_cv', 'PRP RT Variability (CV)')
]):
    ax = axes[idx // 2, idx % 2]
    valid = master[['ucla_total', var, 'gender_male']].dropna()

    if len(valid) >= 20:
        # Scatter by gender
        for gender, label, marker, color in [(0, 'Female', 'o', '#E74C3C'), (1, 'Male', 's', '#3498DB')]:
            subset = valid[valid['gender_male'] == gender]
            ax.scatter(subset['ucla_total'], subset[var], alpha=0.6,
                      label=label, marker=marker, s=80, color=color)

        # Regression line
        z = np.polyfit(valid['ucla_total'], valid[var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['ucla_total'].min(), valid['ucla_total'].max(), 100)

        # Get correlation
        r, pval = stats.pearsonr(valid['ucla_total'], valid[var])
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2,
                label=f'r={r:.2f} ({sig})')

        ax.set_xlabel('UCLA Loneliness')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ucla_control_strategy.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Moderation plots (if significant)
if len(moderation_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CV moderation
    if 'z_wcst_cv' in master.columns:
        data_plot = master[['wcst_pe_rate', 'ucla_total', 'wcst_cv']].dropna()

        # Median split on CV
        cv_median = data_plot['wcst_cv'].median()
        low_cv = data_plot[data_plot['wcst_cv'] <= cv_median]
        high_cv = data_plot[data_plot['wcst_cv'] > cv_median]

        axes[0].scatter(low_cv['ucla_total'], low_cv['wcst_pe_rate'],
                       alpha=0.6, label=f'Low CV (Proactive)', s=80, color='#2ECC71')
        axes[0].scatter(high_cv['ucla_total'], high_cv['wcst_pe_rate'],
                       alpha=0.6, label=f'High CV (Reactive)', s=80, color='#E67E22')

        # Regression lines
        for data, color, label in [(low_cv, '#2ECC71', 'Low CV'), (high_cv, '#E67E22', 'High CV')]:
            if len(data) >= 10:
                z = np.polyfit(data['ucla_total'], data['wcst_pe_rate'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
                axes[0].plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.7)

        axes[0].set_xlabel('UCLA Loneliness')
        axes[0].set_ylabel('WCST PE Rate')
        axes[0].set_title('CV Moderation (Reactive Strategy)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Duration moderation
    if 'z_wcst_duration' in master.columns:
        data_plot = master[['wcst_pe_rate', 'ucla_total', 'wcst_duration']].dropna()

        # Median split on duration
        dur_median = data_plot['wcst_duration'].median()
        short_dur = data_plot[data_plot['wcst_duration'] <= dur_median]
        long_dur = data_plot[data_plot['wcst_duration'] > dur_median]

        axes[1].scatter(short_dur['ucla_total'], short_dur['wcst_pe_rate'],
                       alpha=0.6, label=f'Short Duration', s=80, color='#9B59B6')
        axes[1].scatter(long_dur['ucla_total'], long_dur['wcst_pe_rate'],
                       alpha=0.6, label=f'Long Duration (Proactive)', s=80, color='#3498DB')

        # Regression lines
        for data, color in [(short_dur, '#9B59B6'), (long_dur, '#3498DB')]:
            if len(data) >= 10:
                z = np.polyfit(data['ucla_total'], data['wcst_pe_rate'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(data['ucla_total'].min(), data['ucla_total'].max(), 100)
                axes[1].plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.7)

        axes[1].set_xlabel('UCLA Loneliness')
        axes[1].set_ylabel('WCST PE Rate')
        axes[1].set_title('Duration Moderation (Proactive Strategy)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "control_strategy_moderation.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\nSaving results...")

master.to_csv(OUTPUT_DIR / "proactive_reactive_master.csv", index=False, encoding='utf-8-sig')
strategy_df.to_csv(OUTPUT_DIR / "ucla_control_strategy_correlations.csv", index=False, encoding='utf-8-sig')
if len(moderation_results) > 0:
    moderation_df.to_csv(OUTPUT_DIR / "control_strategy_moderation.csv", index=False, encoding='utf-8-sig')

# Summary report
with open(OUTPUT_DIR / "PROACTIVE_REACTIVE_REPORT.txt", 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("PROACTIVE VS REACTIVE CONTROL ANALYSIS - SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write("RESEARCH QUESTION\n")
    f.write("-"*80 + "\n")
    f.write("Do lonely individuals differ in cognitive control strategy?\n")
    f.write("Does control strategy (proactive vs reactive) moderate UCLA→EF effects?\n\n")

    f.write("THEORY\n")
    f.write("-"*80 + "\n")
    f.write("Proactive control: Sustained, preparatory (longer duration, low CV)\n")
    f.write("Reactive control: Stimulus-driven, just-in-time (shorter, high CV)\n\n")

    f.write("KEY FINDINGS\n")
    f.write("-"*80 + "\n\n")

    f.write("UCLA × CONTROL STRATEGY:\n")
    f.write(strategy_df.to_string(index=False))
    f.write("\n\n")

    if len(moderation_results) > 0:
        f.write("MODERATION EFFECTS:\n")
        f.write("-"*80 + "\n")
        f.write(moderation_df.to_string(index=False))
        f.write("\n\n")

    f.write("="*80 + "\n")
    f.write("Full results saved to: " + str(OUTPUT_DIR) + "\n")

print("\n" + "="*80)
print("✓ Proactive vs Reactive Control Analysis complete!")
print("="*80)
