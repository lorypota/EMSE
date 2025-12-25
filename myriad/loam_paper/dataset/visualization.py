#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_data(csv_path):
    return pd.read_csv(csv_path)

def figure_task_engagement(df, output_dir):
    print("Creating figure: task engagement by core status...")
    
    contrib_engagement = []
    for contrib_id in df['contributor_id'].unique():
        contrib_df = df[df['contributor_id'] == contrib_id]
        total_files = contrib_df['file_count'].sum()
        
        core_status_val = contrib_df['core_status'].mode()
        if len(core_status_val) == 0:
            continue
        core_status = core_status_val[0]
        
        engagement = {
            'contributor_id': contrib_id,
            'core_status': core_status,
            'source': 1 if (contrib_df[contrib_df['task_type'] == 'source']['file_count'].sum() > 0) else 0,
            'test': 1 if (contrib_df[contrib_df['task_type'] == 'test']['file_count'].sum() > 0) else 0,
            'documentation': 1 if (contrib_df[contrib_df['task_type'] == 'documentation']['file_count'].sum() > 0) else 0,
        }
        contrib_engagement.append(engagement)
    
    eng_df = pd.DataFrame(contrib_engagement)
    
    #aggr
    engagement_summary = eng_df.groupby('core_status')[['source', 'test', 'documentation']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    engagement_summary.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c', '#3498db'])
    ax.set_title('Task-Based Contributor Roles: Core vs Peripheral\n(Proportion of contributors engaging in each task type)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Contributor Status', fontsize=12)
    ax.set_ylabel('Proportion of Contributors', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(['Source Code', 'Tests', 'Documentation'], loc='upper right')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_task_engagement.png', dpi=300)
    plt.close()
    print(f"  Saved: {output_dir / 'fig_task_engagement.png'}")

def figure_task_proportions(df, output_dir):
    """Violin plots: distribution of proportion of work in each task by core status."""
    print("Creating figure: task proportion distributions...")
    
    #per-contributor proportions
    contrib_props = []
    for contrib_id in df['contributor_id'].unique():
        contrib_df = df[df['contributor_id'] == contrib_id]
        total_files = contrib_df['file_count'].sum()
        
        if total_files == 0:
            continue
        
        core_status_val = contrib_df['core_status'].mode()
        if len(core_status_val) == 0:
            continue
        core_status = core_status_val[0]
        
        for task in ['source', 'test', 'documentation', 'other']:
            task_files = contrib_df[contrib_df['task_type'] == task]['file_count'].sum()
            contrib_props.append({
                'contributor_id': contrib_id,
                'core_status': core_status,
                'task': task,
                'proportion': task_files / total_files,
            })
    
    props_df = pd.DataFrame(contrib_props)
    
    #violin plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tasks = ['source', 'test', 'documentation']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, task in enumerate(tasks):
        task_df = props_df[props_df['task'] == task]
        sns.violinplot(data=task_df, x='core_status', y='proportion', ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{task.capitalize()} Work', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Contributor Status', fontsize=11)
        axes[idx].set_ylabel('Proportion of Work', fontsize=11)
        axes[idx].set_ylim([0, 1])
    
    fig.suptitle('Distribution of Task-Type Engagement by Contributor Status', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_task_proportions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'fig_task_proportions.png'}")

def figure_task_totals_heatmap(df, output_dir):
    """Heatmap: total files by task type and core status."""
    print("Creating figure: task totals heatmap...")
    
    pivot = pd.crosstab(df['core_status'], df['task_type'], values=df['file_count'], aggfunc='sum', normalize='index')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Proportion of Files'})
    ax.set_title('Task-Type Distribution by Contributor Status\n(Row-normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Contributor Status', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_task_heatmap.png', dpi=300)
    plt.close()
    print(f"  Saved: {output_dir / 'fig_task_heatmap.png'}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to contributors_tasks.csv')
    parser.add_argument('--output-dir', default='.', help='Output directory for figures')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data(args.csv)
    
    figure_task_engagement(df, output_dir)
    figure_task_proportions(df, output_dir)
    figure_task_totals_heatmap(df, output_dir)
    
    print(f"\nAll figures saved to {output_dir}")

if __name__ == '__main__':
    main()
