# myriad_collect_core_periph.py
from github import Github, Auth
import json
import math
import os
import sys
import csv
from datetime import datetime
from collections import defaultdict

#config 
GH_TOKEN_FILE = "./github_access_token.txt"
REPO_LIST_FILE = "../get_all_contributors/gh_repos_lists/all_gh_repos.txt"
CATEGORIES_FILE = "./categories_info.json"  
OUT_DIR = "./gh_contributors"
SUMMARY_PATH_JSONL = os.path.join(OUT_DIR, "myriad_repo_summary.jsonl")
SUMMARY_PATH_CSV = os.path.join(OUT_DIR, "myriad_repo_summary.csv")
ALL_IDS_PATH = os.path.join(OUT_DIR, "all_gh_ids.txt")

# top 20% contributors by commit count
CORE_TOP_FRACTION = 0.20  

CREATIVE_CATEGORIES = {
    "visual",
    "sound/music",
    "text/speech" ,
    "electronics",
}

INFRA_CATEGORIES = {
    "AI/ML", 
    "CLI", 
    "GUI", 
    "engine",
    "other",
}


def print_progress_bar(iteration, total, length=50):
    if total <= 0:
        return
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r|{bar}| {iteration}/{total}")
    sys.stdout.flush()

def load_github_client():
    with open(GH_TOKEN_FILE, "r") as tok:
        tok_str = tok.readline().rstrip("\n")
    auth = Auth.Token(tok_str)
    return Github(auth=auth)

def load_repo_categories():
    """
    Reads categories_info.json and returns repo -> set(categories).
    """
    with open(CATEGORIES_FILE, "r", encoding="utf8") as f:
        blocks = json.load(f)

    repo_to_categories = defaultdict(set)

    for block in blocks:
        cat = block.get("category")
        repos = block.get("repos", [])
        if not cat or not isinstance(repos, list):
            continue

        for repo_name in repos:
            if isinstance(repo_name, str) and repo_name.strip():
                repo_to_categories[repo_name.strip()].add(cat)

    return repo_to_categories

def keep_repo(repo, cats):
    """
    Inclusion rule:
      - exclude if any infra category
      - include if any creative category
      - otherwise exclude
    """
    if cats & INFRA_CATEGORIES:
        return False
    if cats & CREATIVE_CATEGORIES:
        return True
    return False

def get_gh_contributors(repo_obj):
    contributors = repo_obj.get_contributors(anon=True)
    total_contributors = contributors.totalCount
    contributors_list = []
    identified_contributors = []
    total_contributions = 0

    for i in range(total_contributors):
        c = contributors[i]
        info = {
            "type": c.type,
            "contributions": int(getattr(c, "contributions", 0)),
        }

        if c.type == "Anonymous":
            # for Anonymous, PyGitHub uses .email sometimes, not always present
            email = getattr(c, "email", None) or f"anonymous_{i}"
            hexhash = hex(abs(hash(email)))
            info["id"] = hexhash[2:].rjust(16, "0")
        else:
            info["id"] = c.login
            identified_contributors.append(c.login)

        total_contributions += info["contributions"]
        contributors_list.append(info)
        print_progress_bar(i + 1, total_contributors)

    print("")
    return total_contributions, contributors_list, identified_contributors

def compute_core_peripheral(contributors_list, top_fraction=CORE_TOP_FRACTION):
    """
    Core = top 20% fraction by commit count.
    Peripheral = rest.
    """
    n = len(contributors_list)
    if n == 0:
        return {
            "n_contributors_total": 0,
            "n_identified": 0,
            "core_n": 0,
            "peripheral_n": 0,
            "core_peripheral_ratio": None,
        }

    # sort descending by contributions
    sorted_contribs = sorted(contributors_list, key=lambda x: x.get("contributions", 0), reverse=True)

    core_n = max(1, int(math.ceil(top_fraction * n)))
    peripheral_n = max(0, n - core_n)

    n_identified = sum(1 for c in contributors_list if c.get("type") != "Anonymous")

    ratio = None
    if peripheral_n > 0:
        ratio = core_n / peripheral_n
    else:
        # all contributors classified as core in tiny repos
        ratio = float("inf") 

    return {
        "n_contributors_total": n,
        "n_identified": n_identified,
        "core_n": core_n,
        "peripheral_n": peripheral_n,
        "core_peripheral_ratio": ratio,
        "core_top_fraction": top_fraction,
    }

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def save_repo_files(repo_name, repo_data):
    filename_base = os.path.join(OUT_DIR, repo_name.replace("/", "&"))
    with open(filename_base + ".json", "w", encoding="utf8") as fp:
        json.dump(repo_data, fp, indent=1)

    # keep .txt ids list for compatibility
    with open(filename_base + ".txt", "w", encoding="utf8") as fp:
        fp.write("\n".join(c["id"] for c in repo_data.get("contributors", [])))

def append_summary_jsonl(summary_row):
    with open(SUMMARY_PATH_JSONL, "a", encoding="utf8") as f:
        f.write(json.dumps(summary_row) + "\n")

def write_summary_csv(rows):
    # minimal CSV writer 
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(SUMMARY_PATH_CSV, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

def main():
    ensure_out_dir()

    g = load_github_client()
    repo_to_categories = load_repo_categories()

    with open(REPO_LIST_FILE, "r", encoding="utf8") as f:
        repos = [r.strip() for r in f.readlines() if r.strip()]

    # reset summary files
    if os.path.exists(SUMMARY_PATH_JSONL):
        os.remove(SUMMARY_PATH_JSONL)

    summary_rows = []
    all_identified_contributors = set()

    kept = 0
    skipped = 0

    for repo_name in repos:
        cats = repo_to_categories.get(repo_name, set())

        if not keep_repo(repo_name, cats):
            skipped += 1
            print(f"[SKIP] {repo_name} categories={sorted(cats)}")
            continue

        kept += 1
        print(f"[KEEP] {repo_name} categories={sorted(cats)}")

        try:
            repo_obj = g.get_repo(repo_name)
            total_contributions, contributors_list, identified = get_gh_contributors(repo_obj)

            metrics = compute_core_peripheral(contributors_list, CORE_TOP_FRACTION)

            repo_data = {
                "repo_name": repo_name,
                "collected_at": datetime.now().astimezone().isoformat(),
                "categories": sorted(cats),
                "total_contributions": total_contributions,
                "contributors": contributors_list,
                "derived_metrics": metrics,
            }

            save_repo_files(repo_name, repo_data)

            all_identified_contributors.update(identified)

            summary_row = {
                "repo_name": repo_name,
                "categories": "|".join(sorted(cats)),
                "total_contributions": total_contributions,
                **metrics,
            }
            append_summary_jsonl(summary_row)
            summary_rows.append(summary_row)

        except Exception as e:
            print(f"[ERROR] {repo_name}: {e}")
            continue

    # save all identified GH ids
    with open(ALL_IDS_PATH, "w", encoding="utf8") as f:
        for login in sorted(all_identified_contributors):
            f.write(login + "\n")

    # save CSV summary
    write_summary_csv(summary_rows)

    print("")
    print(f"Done. Kept repos: {kept}, Skipped repos: {skipped}")

if __name__ == "__main__":
    main()
