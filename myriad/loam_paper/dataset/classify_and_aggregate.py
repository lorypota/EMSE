#!/usr/bin/env python3
"""
Read fetched commits, classify files into task types (source, test, documentation, config, other),
and aggregate per-contributor activity. Produces a canonical CSV for statistical analysis.

Output: contributors_tasks.csv with columns:
  repo, contributor_id, task_type, file_count, commit_count, core_status
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import csv

#classf. rules
TEST_PATTERNS = [
    r'[/\\](test|tests|spec|__tests__)[/\\]',
    r'[/\\](test|tests|spec|__tests__)$',
    r'_test\.',
    r'\.test\.',
    r'\.spec\.',
    r'^test_',
    r'_spec\.',
]

DOC_PATTERNS = [
    r'\.md$',
    r'\.rst$',
    r'\.txt$',
    r'[/\\]docs?[/\\]',
    r'^docs?[/\\]',
    r'README',
    r'CHANGELOG',
    r'CONTRIBUTING',
    r'LICENSE',
    r'AUTHORS',
    r'\.adoc$',
]

# Actual implementation code
SOURCE_EXTS = {
    '.js', '.jsx', '.ts', '.tsx',  # JavaScript/TypeScript
    '.py', '.pyx', '.pxd',          # Python
    '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx',  # C/C++
    '.java', '.kt', '.scala',       # JVM languages
    '.rs',                          # Rust
    '.go',                          # Go
    '.rb',                          # Ruby
    '.php',                         # PHP
    '.swift',                       # Swift
    '.r', '.R',                     # R
    '.vue', '.svelte',              # Frontend frameworks
    '.lua',                         # Lua
    '.sh', '.bash', '.zsh',         # Shell scripts
    '.pl', '.pm',                   # Perl
    '.ex', '.exs',                  # Elixir
    '.clj', '.cljs',                # Clojure
    '.hs',                          # Haskell
    '.ml', '.mli',                  # OCaml
    '.fs', '.fsx',                  # F#
    '.cs',                          # C#
    '.m', '.mm',                    # Objective-C
    '.asm', '.s',                   # Assembly
    '.glsl', '.hlsl', '.shader',    # Shader languages
    '.pd',                          # Pure Data
    '.scd',                         # SuperCollider
}

# Configuration and data files (separate from source)
CONFIG_EXTS = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
    '.xml', '.plist',
    '.env', '.properties',
    '.gradle', '.maven',
    '.eslintrc', '.prettierrc', '.babelrc',
    '.editorconfig', '.gitignore', '.gitattributes',
    '.dockerignore', '.npmignore',
}

# Build and CI files
BUILD_PATTERNS = [
    r'Makefile',
    r'CMakeLists\.txt',
    r'Dockerfile',
    r'docker-compose',
    r'\.github[/\\]',
    r'\.gitlab-ci',
    r'\.travis',
    r'\.circleci',
    r'Jenkinsfile',
    r'azure-pipelines',
    r'package\.json$',
    r'package-lock\.json$',
    r'yarn\.lock$',
    r'Cargo\.toml$',
    r'Cargo\.lock$',
    r'requirements\.txt$',
    r'setup\.py$',
    r'pyproject\.toml$',
]

# Web assets (styling, not implementation logic)
WEB_ASSET_EXTS = {
    '.css', '.scss', '.sass', '.less', '.styl',
    '.html', '.htm',
}


def classify_file(filepath):
    """
    Classify a file path into: source | test | documentation | config | other
    
    Priority order:
    1. Test files (by pattern)
    2. Documentation files (by pattern/extension)
    3. Source code files (by extension)
    4. Config files (by extension)
    5. Other (everything else including web assets, images, etc.)
    """
    if not filepath:
        return 'other'
    
    filepath_lower = filepath.lower()
    
    # Check test patterns first (tests can have source extensions)
    for pattern in TEST_PATTERNS:
        if re.search(pattern, filepath_lower):
            return 'test'
    
    # Check documentation patterns
    for pattern in DOC_PATTERNS:
        if re.search(pattern, filepath_lower):
            return 'documentation'
    
    # Check source code extensions
    for ext in SOURCE_EXTS:
        if filepath_lower.endswith(ext):
            return 'source'
    
    # Check config extensions
    for ext in CONFIG_EXTS:
        if filepath_lower.endswith(ext):
            return 'config'
    
    # Check build patterns (classify as config)
    for pattern in BUILD_PATTERNS:
        if re.search(pattern, filepath_lower):
            return 'config'
    
    # Web assets go to other
    for ext in WEB_ASSET_EXTS:
        if filepath_lower.endswith(ext):
            return 'other'
    
    return 'other'


def process_fetched_commits(fetched_dir, repo_name):
    """
    Read fetched commits JSON for a repo and return aggregated contributor task data.
    Returns dict: {contributor_id: {task_type: {files: set, commits: set}}}
    """
    filepath = fetched_dir / f"{repo_name.replace('/', '&')}.json"
    
    if not filepath.exists():
        return {}, 0
    
    contributor_data = defaultdict(lambda: defaultdict(lambda: {'files': set(), 'commits': set()}))
    total_commits = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            commits = data.get('commits', [])
            total_commits = len(commits)
            
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
        return {}, 0
    
    return contributor_data, total_commits


def aggregate_repos(dataset_dir, repos_info_path, categories_path=None, target_categories=None):
    """
    Read repos and aggregate contributor task engagement.
    Returns list of dicts with columns for CSV output.
    """
    fetched_dir = dataset_dir / "fetched_commits"
    
    if not fetched_dir.exists():
        raise SystemExit(f"fetched_commits directory not found: {fetched_dir}")
    
    # Load repos_info to map repo names
    with open(repos_info_path, 'r', encoding='utf-8') as f:
        repos_list = json.load(f)
    repo_names = [r['name'] for r in repos_list]
    
    # Filter by categories if specified
    if target_categories and categories_path:
        with open(categories_path, 'r', encoding='utf-8') as f:
            cats = json.load(f)
        
        allowed_repos = set()
        for category_name in target_categories:
            for c in cats:
                if c.get('category') == category_name:
                    allowed_repos.update(c.get('repos', []))
                    break
        
        repo_names = [r for r in repo_names if r in allowed_repos]
    
    rows = []
    repos_processed = 0
    
    for repo_name in repo_names:
        contributor_data, total_commits = process_fetched_commits(fetched_dir, repo_name)
        
        if not contributor_data:
            continue
        
        repos_processed += 1
        print(f"Processing {repo_name}... ({len(contributor_data)} contributors, {total_commits} commits)")
        
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
    
    print(f"\nProcessed {repos_processed} repositories")
    return rows


def add_core_status(rows):
    """
    Add core/peripheral status to rows.
    
    Core contributors are defined as the top 20% by COMMIT COUNT within each repository.
    This follows Mockus et al. (2002) and Bock et al. (2023).
    """
    # First, compute total commits per contributor per repo
    repo_contrib_commits = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        repo = row['repo']
        contrib = row['contributor_id']
        repo_contrib_commits[repo][contrib] += row['commit_count']
    
    # For each repo, compute the top 20% threshold
    core_thresholds = {}
    for repo, contrib_commits in repo_contrib_commits.items():
        sorted_commits = sorted(contrib_commits.values(), reverse=True)
        n_contributors = len(sorted_commits)
        
        # Top 20% means the cutoff is at index n/5
        # Example: 10 contributors -> top 2 are core (indices 0, 1), threshold at index 1
        top_20_pct_idx = max(0, n_contributors // 5 - 1)
        if top_20_pct_idx >= len(sorted_commits):
            top_20_pct_idx = 0
        
        threshold = sorted_commits[top_20_pct_idx] if sorted_commits else 0
        core_thresholds[repo] = threshold
    
    # Assign core status
    for row in rows:
        repo = row['repo']
        contrib = row['contributor_id']
        total_commits = repo_contrib_commits[repo][contrib]
        threshold = core_thresholds.get(repo, 0)
        row['core_status'] = 'core' if total_commits >= threshold else 'peripheral'
    
    return rows


def print_summary(rows):
    """Print summary statistics about the processed data"""
    repos = set(r['repo'] for r in rows)
    contributors = set(r['contributor_id'] for r in rows)
    
    core_contributors = set(r['contributor_id'] for r in rows if r['core_status'] == 'core')
    peripheral_contributors = set(r['contributor_id'] for r in rows if r['core_status'] == 'peripheral')
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Repositories: {len(repos)}")
    print(f"Total contributors: {len(contributors)}")
    print(f"Core contributors: {len(core_contributors)}")
    print(f"Peripheral contributors: {len(peripheral_contributors)}")
    
    # Task type breakdown
    task_counts = defaultdict(int)
    for row in rows:
        task_counts[row['task_type']] += row['file_count']
    
    print(f"\nFiles by task type:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--categories', nargs='+', default=None, 
                        help='Filter by categories (e.g., --categories visual sound/music)')
    parser.add_argument('--output', default='contributors_tasks.csv', help='Output CSV filename')
    args = parser.parse_args()
    
    dataset = Path(args.dataset_dir)
    repos_info = dataset / 'repos_info.json'
    categories_info = dataset / 'categories_info.json'
    
    print(f"Aggregating from {dataset}...")
    if args.categories:
        print(f"Filtering to categories: {', '.join(args.categories)}")
    
    rows = aggregate_repos(dataset, repos_info, categories_info, args.categories)
    
    print(f"\nAdding core/peripheral status (top 20% by commit count)...")
    rows = add_core_status(rows)
    
    # Print summary
    print_summary(rows)
    
    # Write CSV
    output_path = dataset / args.output
    if rows:
        fieldnames = ['repo', 'contributor_id', 'task_type', 'file_count', 'commit_count', 'core_status']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {output_path}")
    else:
        print("\nNo data rows generated.")


if __name__ == '__main__':
    main()