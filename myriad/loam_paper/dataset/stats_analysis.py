#!/usr/bin/env python3
"""
Statistical analysis for RQ2: Task-based contributor roles.

Analyzes the distribution of task types (source, test, documentation, config, other)
between core and peripheral contributors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path):
    """Load and validate the CSV data"""
    df = pd.read_csv(csv_path)
    
    required_cols = ['repo', 'contributor_id', 'task_type', 'file_count', 'commit_count', 'core_status']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    return df


def summary_stats(df):
    """Print descriptive statistics"""
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)
    
    # Unique counts
    n_repos = df['repo'].nunique()
    n_contributors = df['contributor_id'].nunique()
    
    # Count contributor-repo pairs
    contrib_repo_pairs = df.groupby(['contributor_id', 'repo']).size().reset_index()
    n_pairs = len(contrib_repo_pairs)
    
    print(f"\nDataset overview:")
    print(f"  Repositories: {n_repos}")
    print(f"  Unique contributors: {n_contributors}")
    print(f"  Contributor-repo pairs: {n_pairs}")
    
    # Contributors by core status (counting contributor-repo pairs)
    core_pairs = df[df['core_status'] == 'core'].groupby(['contributor_id', 'repo']).ngroups
    periph_pairs = df[df['core_status'] == 'peripheral'].groupby(['contributor_id', 'repo']).ngroups
    
    print(f"\nContributor-repo pairs by status:")
    print(f"  Core: {core_pairs}")
    print(f"  Peripheral: {periph_pairs}")
    
    # Total files and commits by core status
    print("\nTotal files modified by core status:")
    files_by_status = df.groupby('core_status')['file_count'].sum()
    for status, count in files_by_status.items():
        print(f"  {status}: {count:,}")
    
    print("\nTotal commits by core status:")
    commits_by_status = df.groupby('core_status')['commit_count'].sum()
    for status, count in commits_by_status.items():
        print(f"  {status}: {count:,}")
    
    # Files by task type
    print("\nFiles by task type:")
    files_by_task = df.groupby('task_type')['file_count'].sum().sort_values(ascending=False)
    for task, count in files_by_task.items():
        print(f"  {task}: {count:,}")
    
    # Cross-tabulation: core status x task type (file counts)
    print("\nFile counts by core status and task type:")
    pivot = pd.crosstab(
        df['core_status'], 
        df['task_type'], 
        values=df['file_count'], 
        aggfunc='sum',
        margins=True
    )
    print(pivot.to_string())
    
    return pivot


def compute_proportions(df):
    """
    Compute task type proportions for each contributor-repo pair.
    Returns a DataFrame with one row per contributor per repo.
    
    Note: We analyze at the contributor-repo level because core/peripheral 
    status is defined per-repository. A contributor can be core in one repo
    and peripheral in another.
    """
    # Aggregate files per contributor per repo per task type
    contrib_tasks = df.groupby(['contributor_id', 'repo', 'task_type', 'core_status'])['file_count'].sum().reset_index()
    
    # Pivot to get task types as columns
    contrib_pivot = contrib_tasks.pivot_table(
        index=['contributor_id', 'repo', 'core_status'],
        columns='task_type',
        values='file_count',
        fill_value=0
    ).reset_index()
    
    # Get task columns (excluding contributor_id, repo, and core_status)
    task_cols = [c for c in contrib_pivot.columns if c not in ['contributor_id', 'repo', 'core_status']]
    
    # Compute total files per contributor-repo
    contrib_pivot['total_files'] = contrib_pivot[task_cols].sum(axis=1)
    
    # Compute proportions
    for task in task_cols:
        contrib_pivot[f'{task}_prop'] = contrib_pivot[task] / contrib_pivot['total_files']
        contrib_pivot[f'{task}_prop'] = contrib_pivot[f'{task}_prop'].fillna(0)
    
    return contrib_pivot, task_cols


def compare_proportions(df):
    """
    Compare task type proportions between core and peripheral contributors.
    Uses Mann-Whitney U test for comparing distributions with Bonferroni correction.
    
    Unit of analysis: contributor-repo pairs
    """
    print("\n" + "="*70)
    print("PROPORTION COMPARISON: CORE VS PERIPHERAL")
    print("="*70)
    print("\nUnit of analysis: contributor-repo pairs")
    
    contrib_df, task_cols = compute_proportions(df)
    
    print("\nMean proportion of each task type by core status:")
    print("-" * 50)
    
    prop_cols = [f'{t}_prop' for t in task_cols]
    
    # Group by core status and compute means
    means = contrib_df.groupby('core_status')[prop_cols].mean()
    
    # Print formatted table
    print(f"{'Task Type':<20} {'Core':<15} {'Peripheral':<15}")
    print("-" * 50)
    for task in task_cols:
        core_mean = means.loc['core', f'{task}_prop'] if 'core' in means.index else 0
        periph_mean = means.loc['peripheral', f'{task}_prop'] if 'peripheral' in means.index else 0
        print(f"{task:<20} {core_mean:<15.4f} {periph_mean:<15.4f}")
    
    # Mann-Whitney U tests for each task type
    print("\n" + "="*70)
    print("MANN-WHITNEY U TESTS (with Bonferroni correction)")
    print("="*70)
    print("\nComparing proportion distributions between core and peripheral contributors:")
    print("-" * 70)
    
    core_df = contrib_df[contrib_df['core_status'] == 'core']
    periph_df = contrib_df[contrib_df['core_status'] == 'peripheral']
    
    n_tests = len(task_cols) 
    
    results = []
    for task in task_cols:
        prop_col = f'{task}_prop'
        
        core_vals = core_df[prop_col].values
        periph_vals = periph_df[prop_col].values
        
        if len(core_vals) > 0 and len(periph_vals) > 0:
            stat, p_val = mannwhitneyu(core_vals, periph_vals, alternative='two-sided')
            
            # Bonferroni correction
            p_val_corrected = min(p_val * n_tests, 1.0)
            
            # Rank-biserial correlation (effect size for Mann-Whitney U)
            n1, n2 = len(core_vals), len(periph_vals)
            effect_size = 2 * stat / (n1 * n2) - 1
            
            results.append({
                'task': task,
                'statistic': stat,
                'p_value': p_val,
                'p_value_corrected': p_val_corrected,
                'effect_size': effect_size,
                'core_median': np.median(core_vals),
                'periph_median': np.median(periph_vals)
            })
    
    print(f"{'Task':<12} {'U':<14} {'p':<12} {'p (corr.)':<12} {'r':<10} {'Core Med':<10} {'Periph Med':<12}")
    print("-" * 95)
    
    for r in results:
        sig = "***" if r['p_value_corrected'] < 0.001 else "**" if r['p_value_corrected'] < 0.01 else "*" if r['p_value_corrected'] < 0.05 else ""
        print(f"{r['task']:<12} {r['statistic']:<14.2f} {r['p_value']:<12.6f} {r['p_value_corrected']:<12.6f} {r['effect_size']:<10.3f} {r['core_median']:<10.4f} {r['periph_median']:<12.4f} {sig}")
    
    print(f"\nBonferroni correction applied (n = {n_tests} tests)")
    print("Effect size r: rank-biserial correlation")
    print("  |r| < 0.1: negligible, 0.1-0.3: small, 0.3-0.5: medium, > 0.5: large")
    print("\nSignificance (corrected): *** p<0.001, ** p<0.01, * p<0.05")
    
    return results


def contingency_analysis(df):
    """
    Chi-square tests for task engagement by core status.
    Tests whether core vs peripheral contributors differ in whether they engage with each task type.
    
    Unit of analysis: contributor-repo pairs
    """
    print("\n" + "="*70)
    print("CHI-SQUARE TESTS: TASK ENGAGEMENT BY CORE STATUS")
    print("="*70)
    print("\nDoes the proportion of contributor-repo pairs engaging with each task type differ by core status?")
    
    # Get all unique contributor-repo pairs with their core status
    contrib_repo_pairs = df.groupby(['contributor_id', 'repo', 'core_status']).size().reset_index()[['contributor_id', 'repo', 'core_status']]
    
    task_types = df['task_type'].unique()
    
    # For each contributor-repo pair, determine which task types they engaged with
    engagement_data = []
    for _, row in contrib_repo_pairs.iterrows():
        contrib_id = row['contributor_id']
        repo = row['repo']
        core_status = row['core_status']
        
        pair_df = df[(df['contributor_id'] == contrib_id) & (df['repo'] == repo)]
        
        record = {
            'contributor_id': contrib_id,
            'repo': repo,
            'core_status': core_status
        }
        
        for task in task_types:
            task_files = pair_df[pair_df['task_type'] == task]['file_count'].sum()
            record[f'engaged_{task}'] = task_files > 0
        
        engagement_data.append(record)
    
    engagement_df = pd.DataFrame(engagement_data)
    
    results = []
    print(f"\n{'Task Type':<15} {'Chi-square':<12} {'p-value':<12} {'Core Rate':<12} {'Periph Rate':<12}")
    print("-" * 65)
    
    for task in sorted(task_types):
        engaged_col = f'engaged_{task}'
        
        # Create contingency table
        contingency = pd.crosstab(engagement_df['core_status'], engagement_df[engaged_col])
        
        # Ensure both True and False columns exist
        if True not in contingency.columns:
            contingency[True] = 0
        if False not in contingency.columns:
            contingency[False] = 0
        
        if len(contingency) < 2:
            continue
        
        try:
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            
            # Compute engagement rates
            core_row = contingency.loc['core'] if 'core' in contingency.index else pd.Series({True: 0, False: 0})
            periph_row = contingency.loc['peripheral'] if 'peripheral' in contingency.index else pd.Series({True: 0, False: 0})
            
            core_engaged = core_row[True] if True in core_row.index else 0
            core_total = core_row.sum()
            periph_engaged = periph_row[True] if True in periph_row.index else 0
            periph_total = periph_row.sum()
            
            core_rate = core_engaged / core_total if core_total > 0 else 0
            periph_rate = periph_engaged / periph_total if periph_total > 0 else 0
            
            results.append({
                'task': task,
                'chi2': chi2,
                'p_value': p_val,
                'dof': dof,
                'core_engagement_rate': core_rate,
                'periph_engagement_rate': periph_rate
            })
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{task:<15} {chi2:<12.4f} {p_val:<12.6f} {core_rate:<12.2%} {periph_rate:<12.2%} {sig}")
            
        except Exception as e:
            print(f"{task:<15} Error: {e}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
    print("\nInterpretation: A significant result means core and peripheral contributors")
    print("differ in how likely they are to engage with that task type.")
    
    return results


def logistic_regression_analysis(df):
    """
    Logistic regression: Does task engagement predict core status?
    
    Model: P(core) ~ engaged_source + engaged_test + engaged_documentation + ...
    
    Unit of analysis: contributor-repo pairs
    """
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION: TASK ENGAGEMENT PREDICTING CORE STATUS")
    print("="*70)
    
    try:
        from statsmodels.formula.api import logit
    except ImportError:
        print("statsmodels not available, skipping logistic regression")
        return None
    
    # Create contributor-repo level dataset with binary engagement indicators
    contrib_repo_pairs = df.groupby(['contributor_id', 'repo', 'core_status']).size().reset_index()[['contributor_id', 'repo', 'core_status']]
    
    task_types = df['task_type'].unique()
    
    contrib_data = []
    for _, row in contrib_repo_pairs.iterrows():
        contrib_id = row['contributor_id']
        repo = row['repo']
        core_status = row['core_status']
        
        pair_df = df[(df['contributor_id'] == contrib_id) & (df['repo'] == repo)]
        
        record = {
            'contributor_id': contrib_id,
            'repo': repo,
            'is_core': 1 if core_status == 'core' else 0,
        }
        
        for task in task_types:
            task_files = pair_df[pair_df['task_type'] == task]['file_count'].sum()
            record[f'engaged_{task}'] = 1 if task_files > 0 else 0
        
        contrib_data.append(record)
    
    reg_df = pd.DataFrame(contrib_data)
    
    # Build formula dynamically based on available task types
    task_vars = [c for c in reg_df.columns if c.startswith('engaged_')]
    
    if len(task_vars) == 0:
        print("No task variables found")
        return None
    
    formula = 'is_core ~ ' + ' + '.join(task_vars)
    
    print(f"\nModel: {formula}")
    print(f"N = {len(reg_df)} contributor-repo pairs")
    print(f"  (from {reg_df['contributor_id'].nunique()} unique contributors across {reg_df['repo'].nunique()} repos)")
    
    try:
        model = logit(formula, data=reg_df).fit(disp=0)
        
        print("\n" + "-"*70)
        print("LOGISTIC REGRESSION RESULTS")
        print("-"*70)
        
        # Print coefficients table
        print(f"\n{'Variable':<25} {'Coef':<12} {'Std Err':<12} {'z':<10} {'p-value':<12} {'Odds Ratio':<12}")
        print("-"*85)
        
        for var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            z = model.tvalues[var]
            p = model.pvalues[var]
            odds_ratio = np.exp(coef)
            
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{var:<25} {coef:<12.4f} {se:<12.4f} {z:<10.4f} {p:<12.6f} {odds_ratio:<12.4f} {sig}")
        
        print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"\nPseudo R-squared: {model.prsquared:.4f}")
        print(f"Log-Likelihood: {model.llf:.2f}")
        print(f"AIC: {model.aic:.2f}")
        
        print("\nInterpretation:")
        print("- Odds ratio > 1: Engaging with this task type increases odds of being core")
        print("- Odds ratio < 1: Engaging with this task type decreases odds of being core")
        print("- Odds ratio = 1: No effect")
        
        return model
        
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        return None


def generate_figure(df, output_path):
    """
    Generate grouped box plots comparing proportion distributions
    for all task types between core and peripheral contributors.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE")
    print("="*70)
    
    # Task types in display order
    task_types = ['source', 'other', 'documentation', 'config', 'test']
    task_labels = ['Source', 'Other', 'Docs', 'Config', 'Test']
    
    # Pivot to get file counts per task type
    contributor_tasks = df.pivot_table(
        index=['repo', 'contributor_id', 'core_status'],
        columns='task_type',
        values='file_count',
        fill_value=0
    ).reset_index()
    
    # Calculate total files per contributor-repo pair
    contributor_tasks['total'] = contributor_tasks[task_types].sum(axis=1)
    
    # Calculate proportions for each task type
    for task in task_types:
        contributor_tasks[f'{task}_prop'] = (
            contributor_tasks[task] / contributor_tasks['total'] * 100
        )
    
    # Split by core status
    core_df = contributor_tasks[contributor_tasks['core_status'] == 'core']
    periph_df = contributor_tasks[contributor_tasks['core_status'] == 'peripheral']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Position settings
    n_tasks = len(task_types)
    positions_core = np.arange(n_tasks) * 2.5
    positions_periph = positions_core + 0.8
    width = 0.7
    
    # Colors
    color_core = '#2ecc71'
    color_periph = '#3498db'
    
    # Collect data for box plots
    core_data = [core_df[f'{task}_prop'] for task in task_types]
    periph_data = [periph_df[f'{task}_prop'] for task in task_types]
    
    # Create box plots
    bp_core = ax.boxplot(
        core_data,
        positions=positions_core,
        widths=width,
        patch_artist=True,
        showfliers=False
    )
    
    bp_periph = ax.boxplot(
        periph_data,
        positions=positions_periph,
        widths=width,
        patch_artist=True,
        showfliers=False
    )
    
    # Style core boxes
    for patch in bp_core['boxes']:
        patch.set_facecolor(color_core)
        patch.set_alpha(0.7)
    for median in bp_core['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Style peripheral boxes
    for patch in bp_periph['boxes']:
        patch.set_facecolor(color_periph)
        patch.set_alpha(0.7)
    for median in bp_periph['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    # Add median annotations above boxes
    for i, task in enumerate(task_types):
        core_median = core_df[f'{task}_prop'].median()
        periph_median = periph_df[f'{task}_prop'].median()
        
        # Get box heights for positioning
        core_q75 = core_df[f'{task}_prop'].quantile(0.75)
        periph_q75 = periph_df[f'{task}_prop'].quantile(0.75)
        
        ax.annotate(f'{core_median:.1f}%', 
                    xy=(positions_core[i], core_q75 + 3),
                    ha='center', va='bottom', fontsize=9, color='#1a8a4c')
        ax.annotate(f'{periph_median:.1f}%', 
                    xy=(positions_periph[i], periph_q75 + 3),
                    ha='center', va='bottom', fontsize=9, color='#2074a8')

    # X-axis
    ax.set_xticks((positions_core + positions_periph) / 2)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_xlabel('Task Type', fontsize=12)
    
    # Y-axis
    ax.set_ylabel('Proportion of File Modifications (%)', fontsize=12)
    ax.set_ylim(-5, 110)
    
    # Legend
    ax.legend(
        [bp_core['boxes'][0], bp_periph['boxes'][0]], 
        [f'Core (n={len(core_df)})', f'Peripheral (n={len(periph_df)})'],
        loc='upper right',
        fontsize=10
    )
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to contributors_tasks.csv')
    parser.add_argument('--figure', default='rq2_boxplot.png', help='Output path for figure')
    args = parser.parse_args()
    
    print("="*70)
    print("STATISTICAL ANALYSIS: RQ2")
    print("Task-based roles in core vs peripheral contributors")
    print("="*70)
    
    df = load_data(args.csv)
    
    summary_stats(df)
    compare_proportions(df)
    contingency_analysis(df)
    logistic_regression_analysis(df)
    generate_figure(df, args.figure)
    
    print("\n" + "="*70)
    print("Analysis complete.")
    print("="*70)


if __name__ == '__main__':
    main()