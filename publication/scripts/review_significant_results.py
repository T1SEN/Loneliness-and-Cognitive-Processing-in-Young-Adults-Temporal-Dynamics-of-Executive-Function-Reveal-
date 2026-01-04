"""Review all significant results across all categories."""
import pandas as pd
import sys
from pathlib import Path

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

base = Path('C:/Users/ansel/my_research_exporter/publication/data')

print('=' * 80)
print('정밀 검토: 모든 유의한 결과 (p < 0.05)')
print('=' * 80)

all_sig = []

for task in ['stroop', 'prp', 'wcst']:
    hr = pd.read_csv(base / 'outputs' / 'basic_analysis' / task / 'hierarchical_results.csv', encoding='utf-8-sig')

    # Filter significant results
    sig = hr[hr['p_ucla_wald'] < 0.05].copy()

    for _, row in sig.iterrows():
        col = row['outcome_column']
        outcome = row['outcome']
        p = row['p_ucla_wald']
        dr2 = row['delta_r2_ucla']
        beta = row['ucla_beta']
        n = row['n']

        # Determine category
        col_lower = col.lower()
        if 'exg' in col_lower or 'ex_gauss' in col_lower:
            cat = 'Ex-Gaussian'
        elif 'lba' in col_lower:
            cat = 'LBA'
        elif 'hmm' in col_lower:
            cat = 'HMM'
        elif 'rl_' in col_lower or 'wsls' in col_lower or 'bayes' in col_lower:
            cat = 'RL/WSLS/Bayes'
        elif 'bottleneck' in col_lower or '_cb_' in col_lower or '_cs_' in col_lower:
            cat = 'Bottleneck/CB'
        elif any(x in col_lower for x in ['_slope', 'drift', '_cv', 'dispersion', 'recovery']):
            cat = 'Temporal Dynamics'
        else:
            cat = 'Traditional'

        all_sig.append({
            'task': task.upper(),
            'category': cat,
            'outcome': outcome,
            'outcome_col': col,
            'n': n,
            'beta': beta,
            'p': p,
            'delta_R2': dr2
        })

df = pd.DataFrame(all_sig)
df = df.sort_values(['category', 'task', 'p'])

print(f'\n총 유의한 결과 수: {len(df)}')
print()

# Summary by category
print('=== 카테고리별 유의한 결과 수 ===')
cat_counts = df.groupby('category').size().sort_values(ascending=False)
for cat, count in cat_counts.items():
    print(f'  {cat}: {count}')

print('\n' + '=' * 80)
print('상세 결과 (카테고리별)')
print('=' * 80)

for cat in ['Traditional', 'Temporal Dynamics', 'Ex-Gaussian', 'Bottleneck/CB', 'LBA', 'HMM', 'RL/WSLS/Bayes']:
    subset = df[df['category'] == cat]
    if len(subset) == 0:
        continue

    print(f'\n--- {cat} ({len(subset)}개) ---')
    for _, row in subset.iterrows():
        print(f"  {row['task']:6s} N={row['n']:3.0f}  {row['outcome'][:45]:45s}  p={row['p']:.4f}  ΔR²={row['delta_R2']:.4f}  β={row['beta']:.3f}")

# Save to CSV for review
output_path = base / 'outputs' / 'paper_tables' / 'ALL_significant_results.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'\n\n전체 결과 저장: {output_path}')

# Check sample sizes
print('\n' + '=' * 80)
print('표본 수 확인')
print('=' * 80)

# Raw data
raw = pd.read_csv(base / 'raw' / '1_participants_info.csv', encoding='utf-8-sig')
print(f'Raw 참가자: {len(raw)}')

# Survey complete
surveys = pd.read_csv(base / 'raw' / '2_surveys_results.csv', encoding='utf-8-sig')
ucla_ids = set(surveys[surveys['surveyName'].str.lower() == 'ucla']['participantId'])
dass_ids = set(surveys[surveys['surveyName'].str.lower().str.contains('dass', na=False)]['participantId'])
survey_ids = ucla_ids & dass_ids
print(f'설문 완료 (UCLA + DASS): {len(survey_ids)}')

# Task-specific
for task in ['stroop', 'prp', 'wcst']:
    task_df = pd.read_csv(base / f'complete_{task}' / '1_participants_info.csv', encoding='utf-8-sig')
    print(f'{task.upper()} 완료: {len(task_df)}')

# Overall
overall_df = pd.read_csv(base / 'complete_overall' / '1_participants_info.csv', encoding='utf-8-sig')
print(f'Overall (모든 과제 완료): {len(overall_df)}')
