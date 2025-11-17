import pandas as pd

df = pd.read_csv('results/4a_prp_trials.csv')

print("="*60)
print("PRP DATA STRUCTURE ANALYSIS")
print("="*60)

print(f"\nTotal rows: {len(df)}")

# Check participantId
print("\n[participantId column]")
print(f"  Total non-null: {df['participantId'].notna().sum()}")
pid_with_rt = df[(df['t2_rt'].notna()) & (df['t2_rt'] > 0) & (df['t2_rt'] < 5000)]
print(f"  With valid t2_rt: {len(pid_with_rt)}")
print(f"  Unique participants: {pid_with_rt['participantId'].nunique()}")
if len(pid_with_rt) > 0:
    print(f"  Sample IDs: {pid_with_rt['participantId'].unique()[:3].tolist()}")
    print(f"  SOA distribution:")
    if 'soa_nominal_ms' in pid_with_rt.columns:
        print(pid_with_rt['soa_nominal_ms'].value_counts())
    else:
        print(pid_with_rt['soa'].value_counts())

# Check participant_id
if 'participant_id' in df.columns:
    print("\n[participant_id column]")
    print(f"  Total non-null: {df['participant_id'].notna().sum()}")
    pid2_with_rt = df[(df['participant_id'].notna()) & (df['t2_rt'].notna()) & (df['t2_rt'] > 0) & (df['t2_rt'] < 5000)]
    print(f"  With valid t2_rt: {len(pid2_with_rt)}")
    if len(pid2_with_rt) > 0:
        print(f"  Unique participants: {pid2_with_rt['participant_id'].nunique()}")

# Check t1_correct filtering
print("\n[t1_correct filtering impact]")
t1_correct_data = df[(df['t1_correct'] == True) & (df['t2_rt'].notna()) & (df['t2_rt'] > 0) & (df['t2_rt'] < 5000)]
print(f"  Rows after t1_correct==True: {len(t1_correct_data)}")
print(f"  Unique participants: {t1_correct_data['participantId'].nunique()}")

print("\n"+"="*60)
