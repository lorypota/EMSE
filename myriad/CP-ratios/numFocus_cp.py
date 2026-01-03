import json
import math
import csv
from pathlib import Path
from collections import defaultdict


# config
NUMFOCUS_JSONL = "../../NumFocus_Jan22-Dec24_GH_Actions/NumFocus_Jan22-Dec24_GH_Actions.jsonl"
OUT_CSV = "./numfocus_repo_summary.csv"
CORE_TOP_FRACTION = 0.20

def core_peripheral_from_commit_counts(commit_counts, top_fraction=CORE_TOP_FRACTION):
    n = len(commit_counts)
    if n == 0:
        return 0, 0, None

    core_n = max(1, int(math.ceil(top_fraction * n)))
    peripheral_n = max(0, n - core_n)
    ratio = float("inf") if peripheral_n == 0 else core_n / peripheral_n
    return core_n, peripheral_n, ratio

def main():
    repo_actor_commits = defaultdict(lambda: defaultdict(int))

    # counter initiation
    n_records = 0
    n_push_actions = 0
    n_records = 0
    n_push_actions = 0

    path = Path(NUMFOCUS_JSONL)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {NUMFOCUS_JSONL}")

    print("Reading:", path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)
            n_records += 1

            # flattened action records
            if rec.get("action") != "PushCommits":
                continue

            repo = (rec.get("repository") or {}).get("name")
            actor = (rec.get("actor") or {}).get("login")
            if not repo or not actor:
                continue

            details = rec.get("details") or {}
            push = details.get("push") or {}
            commits = push.get("commits")

            if isinstance(commits, int) and commits >= 0:
                repo_actor_commits[repo][actor] += commits
                n_push_actions += 1

    rows = []
    for repo, actor_counts in repo_actor_commits.items():
        commit_counts = sorted(actor_counts.values(), reverse=True)

        total_contributions = sum(commit_counts)
        n_contributors = len(commit_counts)

        core_n, peripheral_n, ratio = core_peripheral_from_commit_counts(commit_counts)

        rows.append({
            "repo_name": repo,
            "total_contributions": total_contributions,
            "n_contributors_total": n_contributors,
            "core_n": core_n,
            "peripheral_n": peripheral_n,
            "core_peripheral_ratio": ratio,
            "core_top_fraction": CORE_TOP_FRACTION,
        })

    rows.sort(key=lambda r: r["n_contributors_total"], reverse=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("Activity records processed:", n_records)
    print("PushCommits actions used:", n_push_actions)
    print("Repositories with commits:", len(rows))
    print("Output written to:", OUT_CSV)

if __name__ == "__main__":
    main()
