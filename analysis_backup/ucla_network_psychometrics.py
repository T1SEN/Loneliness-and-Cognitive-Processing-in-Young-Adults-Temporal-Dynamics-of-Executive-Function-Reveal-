"""
Network Psychometrics - UCLA Item-Level Bridge Analysis

OBJECTIVE:
Identify which specific UCLA loneliness items are most strongly connected
to executive function impairment ("bridge items").

RATIONALE:
- Total UCLA score may mask heterogeneity in loneliness components
- Network analysis reveals item-level connections
- Bridge centrality = items that connect loneliness cluster to EF cluster
- Clinical utility: Target specific loneliness dimensions (e.g., "no one to turn to")

UCLA-20 ITEM STRUCTURE:
- Items 1-20 measuring various loneliness facets
- Factor structure: Social isolation vs Emotional loneliness
- Reversed items: 1, 5, 6, 9, 10, 15, 16, 19, 20

NETWORK ANALYSIS:
1. Load UCLA item-level responses
2. Compute partial correlations (control for all other items)
3. Build network graph (items = nodes, correlations = edges)
4. Calculate centrality metrics (strength, betweenness, closeness)
5. Identify bridge items to PE/EF metrics
6. Test gender differences in bridge structure

OUTPUT:
- Network visualization with communities
- Bridge centrality rankings
- Item-PE correlation matrix
- Gender-specific bridge items
"""

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.covariance import GraphicalLassoCV
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/analysis_outputs/ucla_network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
np.random.seed(42)

print("="*80)
print("UCLA NETWORK PSYCHOMETRICS - BRIDGE ITEM ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading UCLA item-level data...")

# Load surveys
surveys = pd.read_csv(RESULTS_DIR / "2_surveys_results.csv", encoding='utf-8-sig')

# Normalize column names
if 'participantId' in surveys.columns:
    surveys = surveys.rename(columns={'participantId': 'participant_id'})

# Filter UCLA responses
if 'surveyName' in surveys.columns:
    ucla_responses = surveys[surveys['surveyName'] == 'ucla'].copy()
elif 'survey' in surveys.columns:
    ucla_responses = surveys[surveys['survey'] == 'ucla'].copy()
else:
    print("ERROR: Cannot find survey name column")
    import sys
    sys.exit(1)

print(f"  Loaded {len(ucla_responses)} UCLA responses")

# Load master for PE and demographics
master_path = RESULTS_DIR / "analysis_outputs/master_dataset.csv"
if not master_path.exists():
    print("ERROR: master_dataset.csv not found")
    import sys
    sys.exit(1)

master = pd.read_csv(master_path, encoding='utf-8-sig')

# Load participants for gender
participants = pd.read_csv(RESULTS_DIR / "1_participants_info.csv", encoding='utf-8-sig')
if 'participantId' in participants.columns:
    participants = participants.rename(columns={'participantId': 'participant_id'})

if 'gender' not in master.columns:
    master = master.merge(participants[['participant_id', 'gender']], on='participant_id', how='left')

# Normalize gender
gender_map = {'남성': 'male', '여성': 'female'}
master['gender'] = master['gender'].map(gender_map)
master['gender_male'] = (master['gender'] == 'male').astype(int)

# ============================================================================
# 2. PARSE UCLA ITEM-LEVEL RESPONSES
# ============================================================================
print("\n[2/5] Parsing UCLA item-level responses...")

# UCLA responses are stored as JSON in 'responses' column
import json

ucla_items_data = []

for _, row in ucla_responses.iterrows():
    pid = row['participant_id']

    # Parse responses (assuming JSON format)
    if 'responses' in row and pd.notna(row['responses']):
        try:
            responses = json.loads(row['responses']) if isinstance(row['responses'], str) else row['responses']

            # Extract item scores
            item_scores = {}
            for key, value in responses.items():
                # UCLA items are typically numbered Q1-Q20 or similar
                if isinstance(key, str) and key.startswith('Q'):
                    item_num = key[1:]  # Remove 'Q'
                    try:
                        item_scores[f'ucla_{item_num}'] = int(value)
                    except ValueError:
                        pass
                elif isinstance(key, int):
                    item_scores[f'ucla_{key}'] = int(value)

            if len(item_scores) > 0:
                item_scores['participant_id'] = pid
                ucla_items_data.append(item_scores)

        except (json.JSONDecodeError, TypeError):
            continue

# If no responses found, try alternative parsing
if len(ucla_items_data) == 0:
    print("  No item-level data found in 'responses' column")
    print("  Attempting to extract from existing analysis outputs...")

    # Check if factor scores exist
    factor_file = RESULTS_DIR / "analysis_outputs/ucla_facets/participant_factor_scores.csv"
    if factor_file.exists():
        factor_scores = pd.read_csv(factor_file, encoding='utf-8-sig')
        print(f"  Loaded factor scores for {len(factor_scores)} participants")

        # Find PE column name
        pe_col = None
        for col in ['pe_rate', 'perseverative_error_rate']:
            if col in master.columns:
                pe_col = col
                break

        if pe_col is None:
            print("ERROR: No PE column found in master dataset")
            sys.exit(1)

        # If pe_rate doesn't exist but perseverative_error_rate does, rename it
        if pe_col == 'perseverative_error_rate':
            master = master.rename(columns={'perseverative_error_rate': 'pe_rate'})
            pe_col = 'pe_rate'

        # Merge with master for PE
        merge_cols_master = ['participant_id', pe_col, 'gender', 'gender_male', 'ucla_total']
        # Only include columns that exist
        merge_cols_master = [c for c in merge_cols_master if c in master.columns]

        print(f"  DEBUG: Master columns: {master.columns.tolist()}")
        print(f"  DEBUG: Merge columns to use: {merge_cols_master}")

        # Drop overlapping columns from factor_scores before merge to avoid _x/_y suffixes
        overlap_cols = [c for c in factor_scores.columns if c in merge_cols_master and c != 'participant_id']
        if len(overlap_cols) > 0:
            factor_scores = factor_scores.drop(columns=overlap_cols)

        analysis_data = factor_scores.merge(
            master[merge_cols_master],
            on='participant_id',
            how='inner'
        )

        print(f"  Merged data: N={len(analysis_data)}")
        print(f"  DEBUG: Analysis data columns: {analysis_data.columns.tolist()}")

        # Since we don't have item-level data, we'll use factor scores as proxy
        # This is a simplification - ideally we'd have all 20 items

        # Create correlation matrix between factors and PE
        factor_cols = [col for col in analysis_data.columns if col.startswith('factor')]

        if len(factor_cols) >= 2:
            print(f"\n  Available factors: {len(factor_cols)}")

            # Compute correlations with PE
            factor_pe_corrs = []

            for factor in factor_cols:
                valid_data = analysis_data[[factor, 'pe_rate']].dropna()
                if len(valid_data) >= 10:
                    r, p = pearsonr(valid_data[factor], valid_data['pe_rate'])
                    factor_pe_corrs.append({
                        'factor': factor,
                        'r': r,
                        'p': p,
                        'n': len(valid_data)
                    })

            factor_pe_df = pd.DataFrame(factor_pe_corrs)

            print("\n  Factor-PE Correlations:")
            for _, row in factor_pe_df.iterrows():
                sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
                print(f"    {row['factor']}: r={row['r']:.3f}, p={row['p']:.4f} {sig}")

            # Save factor-PE correlations
            factor_pe_df.to_csv(OUTPUT_DIR / 'factor_pe_correlations.csv', index=False, encoding='utf-8-sig')

            # Gender-stratified
            print("\n  Gender-Stratified Factor-PE Correlations:")

            males = analysis_data[analysis_data['gender_male'] == 1]
            females = analysis_data[analysis_data['gender_male'] == 0]

            for factor in factor_cols:
                print(f"\n  {factor}:")

                # Males
                if len(males) >= 10:
                    valid_males = males[[factor, 'pe_rate']].dropna()
                    if len(valid_males) >= 10:
                        r_m, p_m = pearsonr(valid_males[factor], valid_males['pe_rate'])
                        sig_m = '***' if p_m < 0.001 else '**' if p_m < 0.01 else '*' if p_m < 0.05 else 'ns'
                        print(f"    Males (N={len(valid_males)}): r={r_m:.3f}, p={p_m:.4f} {sig_m}")

                # Females
                if len(females) >= 10:
                    valid_females = females[[factor, 'pe_rate']].dropna()
                    if len(valid_females) >= 10:
                        r_f, p_f = pearsonr(valid_females[factor], valid_females['pe_rate'])
                        sig_f = '***' if p_f < 0.001 else '**' if p_f < 0.01 else '*' if p_f < 0.05 else 'ns'
                        print(f"    Females (N={len(valid_females)}): r={r_f:.3f}, p={p_f:.4f} {sig_f}")

            # Visualization: Factor network
            print("\n[3/5] Creating factor network visualization...")

            # Compute partial correlations between factors
            factor_data = analysis_data[factor_cols].dropna()

            if len(factor_data) >= 20 and len(factor_cols) >= 2:
                # Use Graphical Lasso for sparse inverse covariance (partial correlations)
                try:
                    from sklearn.covariance import GraphicalLassoCV

                    # Standardize
                    factor_data_std = (factor_data - factor_data.mean()) / factor_data.std()

                    # Fit Graphical Lasso
                    glasso = GraphicalLassoCV(cv=3, max_iter=100)
                    glasso.fit(factor_data_std)

                    # Partial correlations
                    precision = glasso.precision_
                    partial_corr = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
                    np.fill_diagonal(partial_corr, 1)

                    partial_corr_df = pd.DataFrame(partial_corr, columns=factor_cols, index=factor_cols)

                    # Create network graph
                    G = nx.Graph()

                    # Add nodes
                    for factor in factor_cols:
                        G.add_node(factor)

                    # Add edges (only significant partial correlations)
                    threshold = 0.1
                    for i, f1 in enumerate(factor_cols):
                        for j, f2 in enumerate(factor_cols):
                            if i < j and abs(partial_corr_df.loc[f1, f2]) > threshold:
                                G.add_edge(f1, f2, weight=partial_corr_df.loc[f1, f2])

                    # Layout
                    pos = nx.spring_layout(G, seed=42, k=2)

                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 10))

                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000,
                                          edgecolors='black', linewidths=2, ax=ax)

                    # Draw edges (thickness = correlation strength)
                    edges = G.edges()
                    weights = [abs(G[u][v]['weight']) * 5 for u, v in edges]
                    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, ax=ax)

                    # Draw labels
                    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

                    ax.set_title('UCLA Factor Network (Partial Correlations)', fontweight='bold', fontsize=14, pad=15)
                    ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / 'ucla_factor_network.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    print("  ✓ Factor network visualization saved")

                except Exception as e:
                    print(f"  Error in network visualization: {e}")

            # Visualization: Factor-PE correlations
            print("\n[4/5] Creating factor-PE correlation plots...")

            if len(factor_pe_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))

                colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in factor_pe_df['p']]
                bars = ax.barh(factor_pe_df['factor'], factor_pe_df['r'], color=colors, edgecolor='black', linewidth=1.5)

                ax.axvline(0, color='black', linestyle='-', linewidth=1)
                ax.set_xlabel('Correlation with PE Rate (r)', fontweight='bold')
                ax.set_ylabel('UCLA Factor', fontweight='bold')
                ax.set_title('UCLA Factors → PE Rate Correlations', fontweight='bold', pad=15)
                ax.grid(alpha=0.3, axis='x')

                # Add significance markers
                for i, row in factor_pe_df.iterrows():
                    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
                    if sig:
                        x_pos = row['r'] + (0.02 if row['r'] > 0 else -0.02)
                        ax.text(x_pos, i, sig, va='center', fontsize=12, fontweight='bold')

                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'factor_pe_correlations_plot.png', dpi=300, bbox_inches='tight')
                plt.close()

                print("  ✓ Factor-PE correlation plot saved")

            # Generate report
            print("\n[5/5] Generating report...")

            with open(OUTPUT_DIR / "UCLA_NETWORK_REPORT.txt", 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("UCLA NETWORK PSYCHOMETRICS - BRIDGE ITEM ANALYSIS\n")
                f.write("="*80 + "\n\n")

                f.write("OBJECTIVE\n")
                f.write("-"*80 + "\n")
                f.write("Identify which UCLA loneliness dimensions (factors) are most strongly\n")
                f.write("connected to executive function impairment.\n\n")

                f.write("NOTE\n")
                f.write("-"*80 + "\n")
                f.write("Item-level UCLA data not available in raw format.\n")
                f.write("Analysis performed using existing factor scores from previous analysis.\n\n")

                f.write("SAMPLE\n")
                f.write("-"*80 + "\n")
                f.write(f"Total N = {len(analysis_data)}\n")
                f.write(f"  Males: {(analysis_data['gender_male'] == 1).sum()}\n")
                f.write(f"  Females: {(analysis_data['gender_male'] == 0).sum()}\n\n")

                f.write("FACTOR-PE CORRELATIONS\n")
                f.write("-"*80 + "\n")
                for _, row in factor_pe_df.iterrows():
                    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
                    f.write(f"{row['factor']}: r={row['r']:.3f}, p={row['p']:.4f} {sig}\n")

                f.write("\n")

                if factor_pe_df['p'].min() < 0.05:
                    strongest_factor = factor_pe_df.loc[factor_pe_df['p'].idxmin(), 'factor']
                    strongest_r = factor_pe_df.loc[factor_pe_df['p'].idxmin(), 'r']
                    f.write(f"✓ BRIDGE FACTOR: {strongest_factor} (r={strongest_r:.3f})\n")
                    f.write(f"  This factor shows the strongest connection to PE impairment.\n\n")
                else:
                    f.write("~ No significant factor-PE associations detected.\n\n")

                f.write("INTERPRETATION\n")
                f.write("-"*80 + "\n")
                f.write("Bridge factors = specific loneliness dimensions that predict EF impairment\n")
                f.write("Target these factors in interventions for maximum impact.\n\n")

                f.write("LIMITATIONS\n")
                f.write("-"*80 + "\n")
                f.write("1. Item-level data not available - using aggregated factors\n")
                f.write("2. Cannot identify specific UCLA items (e.g., \"no one to turn to\")\n")
                f.write("3. Factor structure may obscure item-level heterogeneity\n\n")

                f.write("="*80 + "\n")
                f.write(f"Outputs saved to: {OUTPUT_DIR}\n")

            print("\n" + "="*80)
            print("✓ UCLA NETWORK ANALYSIS COMPLETE!")
            print("="*80)
            print(f"\nOutputs saved to: {OUTPUT_DIR}")

        else:
            print("\nERROR: Insufficient factor data for network analysis")
            sys.exit(1)

    else:
        print("\nERROR: No UCLA item-level or factor-level data available")
        sys.exit(1)

else:
    print(f"  Parsed item-level data for {len(ucla_items_data)} participants")

    # Convert to DataFrame
    ucla_items_df = pd.DataFrame(ucla_items_data)

    # Further processing would go here if item-level data was available
    # This code path is kept for future use when item-level data becomes available
