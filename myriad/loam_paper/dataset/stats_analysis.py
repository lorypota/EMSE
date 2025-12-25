#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from statsmodels.genmod import generalized_estimating_equations as gee
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.formula.api import logit
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    return pd.read_csv(csv_path)

def summary_stats(df):
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nContributors by core status:")
    print(df.groupby('core_status')['contributor_id'].nunique())
    
    print("\nTotal files modified by core status and task type:")
    pivot = pd.crosstab(df['core_status'], df['task_type'], values=df['file_count'], aggfunc='sum')
    print(pivot)
    
    print("\nAverage files per contributor by core status and task type:")
    gb = df.groupby(['core_status', 'task_type'])['file_count'].agg(['count', 'mean', 'std', 'median'])
    print(gb)

def contributor_participation(df):
    print("\n" + "="*70)
    print("CONTRIBUTOR-LEVEL TASK ENGAGEMENT ANALYSIS")
    print("="*70)
    
    #agg per contributor across repos
    contrib_summary = df.groupby('contributor_id').agg({
        'file_count': 'sum',
        'commit_count': 'sum',
        'core_status': 'first',  #this holds for 99% for a contributors
        'repo': 'nunique'
    }).reset_index()
    contrib_summary.columns = ['contributor_id', 'total_files', 'total_commits', 'core_status', 'num_repos']
    
    #recomputing core status (top 10%))
    sorted_files = sorted(contrib_summary['total_files'].values, reverse=True)
    global_threshold = sorted_files[len(sorted_files) // 10] if len(sorted_files) > 0 else 0
    contrib_summary['global_core'] = contrib_summary['total_files'] >= global_threshold
    
    print(f"\nGlobal core/peripheral split (top 10% by total files):")
    print(contrib_summary['global_core'].value_counts())
    
    #compute per-contributor proportions by task type
    contrib_props = []
    for contrib_id in df['contributor_id'].unique():
        contrib_df = df[df['contributor_id'] == contrib_id]
        total_files = contrib_df['file_count'].sum()
        
        if total_files == 0:
            continue
        
        props = {
            'contributor_id': contrib_id,
            'total_files': total_files,
            'global_core': contrib_summary[contrib_summary['contributor_id'] == contrib_id]['global_core'].values[0],
        }
        
        for task in ['source', 'test', 'documentation', 'other']:
            task_files = contrib_df[contrib_df['task_type'] == task]['file_count'].sum()
            props[f'{task}_prop'] = task_files / total_files if total_files > 0 else 0
        
        contrib_props.append(props)
    
    contrib_props_df = pd.DataFrame(contrib_props)
    
    print("\nMean task proportions by global core status:")
    print(contrib_props_df.groupby('global_core')[['source_prop', 'test_prop', 'documentation_prop', 'other_prop']].mean())
    
    return contrib_props_df

def contingency_tables(df):
    """Chi-square test: did contributors engage in each task type?"""
    print("\n" + "="*70)
    print("CONTINGENCY TABLES & CHI-SQUARE TESTS")
    print("="*70)
    
    #check if contributor modify files in task_type
    df_binary = df.copy()
    df_binary['engaged'] = df_binary['file_count'] > 0
    
    for task in ['source', 'test', 'documentation']:
        df_task = df_binary[df_binary['task_type'] == task]
        if len(df_task) == 0:
            continue
        
        contingency = pd.crosstab(df_task['core_status'], df_task['engaged'])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        
        print(f"\n{task.upper()} engagement by core status:")
        print(contingency)
        print(f"Chi-square: {chi2:.4f}, p-value: {p_val:.6f}, dof: {dof}")

def logistic_regression(df):
    """Logistic regression: core status ~ task type engagement (binary)."""
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION: CORE STATUS ~ TASK TYPE")
    print("="*70)
    
    # dataset for regression: each row = contributor task pair
    contrib_task_data = []
    
    for contrib_id in df['contributor_id'].unique():
        contrib_df = df[df['contributor_id'] == contrib_id]
        
        #global core status
        total_files = contrib_df['file_count'].sum()
        
        for task in ['source', 'test', 'documentation']:
            task_files = contrib_df[contrib_df['task_type'] == task]['file_count'].sum()
            engaged = 1 if task_files > 0 else 0
            
            contrib_task_data.append({
                'contributor_id': contrib_id,
                'task': task,
                'engaged': engaged,
                'total_files': total_files,
            })
    
    reg_df = pd.DataFrame(contrib_task_data)
    
    #determine global core status
    sorted_files = sorted(reg_df['total_files'].unique(), reverse=True)
    threshold = sorted_files[len(sorted_files) // 10] if len(sorted_files) > 0 else 0
    reg_df['is_core'] = (reg_df['total_files'] >= threshold).astype(int)
    
    try:
        formula = 'engaged ~ C(is_core) + C(task)'
        from statsmodels.formula.api import ols
        model = ols(formula, data=reg_df).fit()
        print("\nOLS model summary:")
        print(model.summary())
    except Exception as e:
        print(f"Regression fit error: {e}")
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to contributors_tasks.csv')
    args = parser.parse_args()
    
    df = load_data(args.csv)
    
    summary_stats(df)
    contrib_props = contributor_participation(df)
    contingency_tables(df)
    logistic_regression(df)
    
    print("\n" + "="*70)
    print("Analysis complete.")
    print("="*70)

if __name__ == '__main__':
    main()
