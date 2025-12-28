#!/usr/bin/env python3
"""
Fetch commit file lists for repos in repos_info.json.
Output: dataset/fetched_commits/<owner>&<repo>.json
"""
import os
import time
import json
import argparse
import requests
from pathlib import Path

API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

def github_get(url, params=None):
    """Make authenticated GitHub API request"""
    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  Network error: {e}, retrying in 10s...")
            time.sleep(10)
            continue
            
        if r.status_code == 200:
            remaining = int(r.headers.get("X-RateLimit-Remaining", "0"))
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            if remaining and remaining < 10:
                sleep_for = max(1, reset - int(time.time()) + 5)
                print(f"  Low remaining ({remaining}), sleeping {sleep_for}s until reset")
                time.sleep(sleep_for)
            return r.json(), r.headers
        elif r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            sleep_for = max(1, reset - int(time.time()) + 5)
            print(f"  Rate limit hit. Sleeping {sleep_for}s...")
            time.sleep(sleep_for)
            continue
        elif r.status_code == 404:
            print(f"  Repository not found (404)")
            return None, r.headers
        elif r.status_code == 409:
            # Empty repository
            print(f"  Empty repository (409)")
            return None, r.headers
        else:
            print(f"  GitHub API error {r.status_code}: {r.text[:200]}")
            return None, r.headers

def fetch_commits_for_repo(owner, repo, out_path, max_commits=None):
    """Fetch all commits and their file lists for a single repository"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists():
        print(f"  Skipping {owner}/{repo} (output exists)")
        return "skipped"
    
    print(f"Fetching commits for {owner}/{repo}")
    commits = []
    per_page = 100
    page = 1
    total_fetched = 0
    
    while True:
        url = f"{API}/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        page_data, headers = github_get(url, params=params)
        
        if page_data is None:
            print(f"  Failed to fetch page {page}, stopping")
            break
        
        if not page_data:
            break
        
        for c in page_data:
            sha = c.get("sha")
            if not sha:
                continue
            commit_url = f"{API}/repos/{owner}/{repo}/commits/{sha}"
            detail, _ = github_get(commit_url)
            
            if detail is None:
                continue
            
            files = [f.get("filename") for f in detail.get("files", []) if f.get("filename")]
            author = detail.get("author")
            
            commit_obj = {
                "sha": sha,
                "date": detail.get("commit", {}).get("author", {}).get("date"),
                "author_login": author.get("login") if author else None,
                "author_name": detail.get("commit", {}).get("author", {}).get("name"),
                "author_email": detail.get("commit", {}).get("author", {}).get("email"),
                "files": files
            }
            commits.append(commit_obj)
            total_fetched += 1
            
            # Progress update every 50 commits
            if total_fetched % 50 == 0:
                print(f"    Fetched {total_fetched} commits...")
            
            if max_commits and total_fetched >= max_commits:
                break
        
        if max_commits and total_fetched >= max_commits:
            break
        
        if len(page_data) < per_page:
            break
        
        page += 1
        time.sleep(0.2)
    
    if commits:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"repo": f"{owner}/{repo}", "commits": commits}, f, indent=2)
        print(f"  Wrote {len(commits)} commits to {out_path}")
        return "fetched"
    else:
        print(f"  No commits fetched for {owner}/{repo}")
        return "empty"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--categories", nargs="+", required=False, 
                        help="Filter by categories (e.g., --categories visual sound/music)")
    parser.add_argument("--max-commits-per-repo", type=int, default=None,
                        help="Limit commits per repo (for testing)")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    repos_info_path = dataset / "repos_info.json"
    categories_path = dataset / "categories_info.json"
    
    if not repos_info_path.exists():
        raise SystemExit(f"repos_info.json not found: {repos_info_path}")

    with open(repos_info_path, "r", encoding="utf-8") as f:
        repos = json.load(f)

    # Filter by categories if specified
    allowed = None
    if args.categories:
        if not categories_path.exists():
            raise SystemExit(f"categories_info.json not found: {categories_path}")
        with open(categories_path, "r", encoding="utf-8") as f:
            cats = json.load(f)
        
        allowed = set()
        for category_name in args.categories:
            found = False
            for c in cats:
                if c.get("category") == category_name:
                    allowed.update(c.get("repos", []))
                    found = True
                    break
            if not found:
                raise SystemExit(f"Category '{category_name}' not found")
        
        print(f"Filtering to {len(allowed)} repos in categories: {', '.join(args.categories)}")

    out_dir = dataset / "fetched_commits"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count repos to process
    repos_to_process = []
    for r in repos:
        name = r.get("name")
        if not name:
            continue
        if allowed and name not in allowed:
            continue
        repos_to_process.append(r)
    
    print(f"\nWill process {len(repos_to_process)} repos\n")
    
    stats = {"fetched": 0, "skipped": 0, "empty": 0, "error": 0}
    
    for i, r in enumerate(repos_to_process, 1):
        name = r.get("name")
        owner, repo = name.split("/", 1)
        out_name = f"{owner}&{repo}.json"
        out_path = out_dir / out_name
        
        print(f"[{i}/{len(repos_to_process)}] ", end="")
        
        try:
            result = fetch_commits_for_repo(owner, repo, out_path, max_commits=args.max_commits_per_repo)
            stats[result] += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error fetching {name}: {e}")
            stats["error"] += 1
    
    print(f"\n{'='*50}")
    print(f"Fetch complete!")
    print(f"  Fetched: {stats['fetched']}")
    print(f"  Skipped (already exists): {stats['skipped']}")
    print(f"  Empty repos: {stats['empty']}")
    print(f"  Errors: {stats['error']}")

if __name__ == "__main__":
    if not TOKEN:
        print("WARNING: GITHUB_TOKEN not set. You will hit rate limits quickly.\n")
    main()