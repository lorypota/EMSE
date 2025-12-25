#!/usr/bin/env python3
"""
Read fetched commits, classify files into task types (source, test, documentation),
and aggregate per-contributor activity. Produces a canonical CSV for statistical analysis.

Output: contributors_tasks.csv with columns:
  repo, contributor_id, task_type, file_count, commit_count
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import csv

#classf. rules
TEST_PATTERNS = [
    r'[/\\](test|tests|spec)[/\\]',
    r'_test\.',
    r'\.test\.',
    r'\.spec\.',
    r'^test_',
]

DOC_PATTERNS = [
    r'\.md$',
    r'\.rst$',
    r'\.txt$',
    r'[/\\]docs?[/\\]',
    r'README',
    r'CHANGELOG',
    r'CONTRIBUTING',
]

SOURCE_EXTS = {
    '.js', '.py', '.cpp', '.c', '.h', '.java', '.ts', '.tsx', '.jsx',
    '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.vue', '.svelte', '.html', '.css', '.scss', '.less', '.json',
    '.yaml', '.yml', '.xml', '.toml', '.ini', '.gradle', '.maven',
}

def classify_file(filepath):
    """
    Classify a file path into: source | test | documentation | other
    """
    if not filepath:
        return 'other'
    
    filepath_lower = filepath.lower()
    
    #checks patterns  
    for pattern in TEST_PATTERNS:
        if re.search(pattern, filepath_lower):
            return 'test'
    
    for pattern in DOC_PATTERNS:
        if re.search(pattern, filepath_lower):
            return 'documentation'
    
    for ext in SOURCE_EXTS:
        if filepath_lower.endswith(ext):
            return 'source'
    return 'other'

def process_fetched_commits(fetched_dir, repo_name):
    """
    Read fetched commits JSON for a repo and return aggregated contributor task data.
    Returns dict: {contributor_id: {task_type: {files: set, commits: set}}}
    """
    filepath = fetched_dir / f"{repo_name.replace('/', '&')}.json"
    
    if not filepath.exists():
        return {}
    
    contributor_data = defaultdict(lambda: defaultdict(lambda: {'files': set(), 'commits': set()}))
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            commits = data.get('commits', [])
            
            for commit in commits:
                author_login = commit.get('author_login')
                sha = commit.get('sha', '')
                files = commit.get('files', [])
                
                if not author_login or not files:
                    continue
                
                for file_path in files:
                    task_type = classify_file(file_path)
                    contributor_data[author_login][task_type]['files'].add(file_path)
                    contributor_data[author_login][task_type]['commits'].add(sha)
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}
    
    return contributor_data

def aggregate_repos(dataset_dir, repos_info_path, categories_path=None, target_category=None):
    """
    Read repos and aggç contributor task engagement
    Returns list of dicts with columns for CSV output
    """
    fetched_dir = dataset_dir / "fetched_commits"
    
    if not fetched_dir.exists():
        raise SystemExit(f"fetched_commits directory not found: {fetched_dir}")
    
    #load repos_info to map repo names
    with open(repos_info_path, 'r', encoding='utf-8') as f:
        repos_list = json.load(f)
    repo_names = [r['name'] for r in repos_list]
    
    #filter category
    if target_category and categories_path:
        with open(categories_path, 'r', encoding='utf-8') as f:
            cats = json.load(f)
        for c in cats:
            if c.get('category') == target_category:
                repo_names = c.get('repos', [])
                break
    
    rows = []
    
    for repo_name in repo_names:
        print(f"Processing {repo_name}...")
        contributor_data = process_fetched_commits(fetched_dir, repo_name)
        
        for contributor_id, task_dict in contributor_data.items():
            for task_type, file_data in task_dict.items():
                if file_data['files'] or file_data['commits']:
                    rows.append({
                        'repo': repo_name,
                        'contributor_id': contributor_id,
                        'task_type': task_type,
                        'file_count': len(file_data['files']),
                        'commit_count': len(file_data['commits']),
                    })
    
    return rows

def add_core_status(rows, repos_info_path):
    """
    Add core/peripheral status to rows
    Uses top 10% by total files modified as core contributors
    """
    with open(repos_info_path, 'r', encoding='utf-8') as f:
        repos_meta = {r['name']: r for r in json.load(f)}
    #group
    repo_contrib_totals = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        repo = row['repo']
        contrib = row['contributor_id']
        repo_contrib_totals[repo][contrib] += row['file_count']
    
    # for each repo, compute threshold
    core_thresholds = {}
    for repo, contrib_files in repo_contrib_totals.items():
        sorted_contribs = sorted(contrib_files.values(), reverse=True)
        top_10_pct_idx = max(0, len(sorted_contribs) // 10)
        threshold = sorted_contribs[top_10_pct_idx] if top_10_pct_idx < len(sorted_contribs) else sorted_contribs[0]
        core_thresholds[repo] = threshold
    for row in rows:
        repo = row['repo']
        contrib = row['contributor_id']
        total_files = repo_contrib_totals[repo][contrib]
        threshold = core_thresholds.get(repo, 0)
        row['core_status'] = 'core' if total_files >= threshold else 'peripheral'
    
    return rows

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--category', default=None, help='Filter by category')
    parser.add_argument('--output', default='contributors_tasks.csv')
    args = parser.parse_args()
    
    dataset = Path(args.dataset_dir)
    repos_info = dataset / 'repos_info.json'
    categories_info = dataset / 'categories_info.json'
    
    print(f"Aggregating from {dataset}...")
    rows = aggregate_repos(dataset, repos_info, categories_info, args.category)
    
    print(f"Adding core/peripheral status...")
    rows = add_core_status(rows, repos_info)
    
    # Write CSV
    output_path = dataset / args.output
    if rows:
        fieldnames = rows[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output_path}")
    else:
        print("No data rows generated.")

if __name__ == '__main__':
    main()
