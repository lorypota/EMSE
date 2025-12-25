# RQ2 Analysis: Task-Based Contributor Roles in Creative-Coding OSS

Complete pipeline to answer: **How are task-based contributor roles distributed between core and peripheral contributors in open-source projects used for new media art?**

## Overview

This pipeline:
1. Fetches commit-level file lists from GitHub for repos in the Myriad dataset
2. Classifies modified files into task types (Source Code, Tests, Documentation)
3. Aggregates per-contributor activity and defines core/peripheral status
4. Runs statistical analysis (chi-square, contingency tables, regression)
5. Produces publication-quality visualizations

## Prerequisites

```powershell
pip install requests pandas scipy statsmodels matplotlib seaborn numpy
```

Create a GitHub Personal Access Token (PAT):
- Go to https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Grant `repo` scope (read-only)
- Copy the token

## Step-by-Step Execution

### Step 1: Fetch Commits (fetch_commits_files.py)

Fetches commit-level file lists from GitHub API for specified repos.

**Command:**
```powershell
$env:GITHUB_TOKEN = "<your_token>"

python fetch_commits_files.py `
  --dataset-dir "c:\path\to\dataset" `
  --category visual `
  --max-commits-per-repo 1500
```

**What it does:**
- Reads `repos_info.json` and `categories_info.json`
- For each repo, fetches all commits (up to `--max-commits-per-repo` per repo)
- For each commit, extracts: commit SHA, author login, date, modified file paths
- Saves JSON files to `fetched_commits/<owner>&<repo>.json`

**Output:** `fetched_commits/` directory with one JSON file per repo

**Typical runtime:** 10–60 minutes depending on repo size and rate limits

---

### Step 2: Classify Files & Aggregate Contributor Activity (classify_and_aggregate.py)

Reads fetched commits, classifies files, and produces the canonical CSV for analysis.

**Command:**
```powershell
python classify_and_aggregate.py `
  --dataset-dir "c:\path\to\dataset" `
  --category visual `
  --output contributors_tasks.csv
```

**What it does:**
- Reads each `fetched_commits/<repo>.json`
- For each modified file, classifies into:
  - **Source**: `.js`, `.py`, `.cpp`, etc.; typical code extensions
  - **Test**: files in `test/`, `tests/`, `spec/` directories; files matching `*_test.*`, `*.spec.*`
  - **Documentation**: `.md`, `.rst`, `.txt`; files in `docs/` directories; `README*`, `CHANGELOG`, etc.
  - **Other**: remaining files
- Aggregates per contributor:
  - File count per task type
  - Commit count per task type
  - Core/peripheral status: **top 10% by total files modified per repo = core; rest = peripheral**
- Outputs canonical CSV

**Output:** `contributors_tasks.csv` with columns:
```
repo, contributor_id, task_type, file_count, commit_count, core_status
```

---

### Step 3: Statistical Analysis (stats_analysis.py)

Runs tests to compare core vs peripheral contributor engagement.

**Command:**
```powershell
python stats_analysis.py `
  --csv "c:\path\to\contributors_tasks.csv"
```

**What it does:**

1. **Summary Statistics:**
   - Number of contributors by core status
   - Total files modified by status and task type
   - Average files per contributor

2. **Contributor-Level Analysis:**
   - Computes per-contributor proportions of work in each task type
   - Compares mean proportions: core vs peripheral

3. **Contingency Tables & Chi-Square Tests:**
   - For each task type (source, test, documentation):
     - Binary: did contributor engage (>0 files) vs not?
     - Tests: does core/peripheral status predict engagement?
     - Reports chi-square statistic and p-value

4. **Logistic Regression:**
   - Model: `engaged (binary) ~ core_status + task_type`
   - Interpretation: effect of core status on likelihood of engaging in each task

**Output:** Printed summary table with test statistics

---

### Step 4: Visualizations (visualization.py)

Creates three publication-quality figures.

**Command:**
```powershell
python visualization.py `
  --csv "c:\path\to\contributors_tasks.csv" `
  --output-dir "c:\path\to\figures"
```

**What it produces:**

1. **fig_task_engagement.png**
   - Stacked bar chart
   - X-axis: core vs peripheral
   - Y-axis: proportion of contributors engaging in each task
   - Interpretation: do more core contributors work on tests/docs than peripheral?

2. **fig_task_proportions.png**
   - Three violin plots (one per task type)
   - Shows distribution of task proportions by contributor status
   - Interpretation: variance in contributor specialization

3. **fig_task_heatmap.png**
   - Heatmap: rows=contributor status, columns=task types
   - Values: proportion of work in each task (row-normalized)
   - Interpretation: task-type specialization by status

**Output:** PNG files in specified output directory

---

## Complete Workflow Example

```powershell
# 1. Set token
$env:GITHUB_TOKEN = "github_pat_..."

# 2. Fetch commits for visual category (1500 commits/repo max)
#    Approx 15 mins for 13 visual repos
python fetch_commits_files.py `
  --dataset-dir "c:\Users\..\dataset" `
  --category visual `
  --max-commits-per-repo 1500

# 3. Classify and aggregate
#    Approx 2 mins
python classify_and_aggregate.py `
  --dataset-dir "c:\Users\..\dataset" `
  --category visual `
  --output contributors_tasks.csv

# 4. Run statistics
#    Approx 1 min
python stats_analysis.py `
  --csv "c:\Users..\contributors_tasks.csv"

# 5. Create visualizations
#    Approx 1 min
python visualization.py `
  --csv "c:\Users\..\contributors_tasks.csv" `
  --output-dir "c:\Users\..\figures"
```

---

## File Classification Rules

### Source Code Extensions
`.js`, `.py`, `.cpp`, `.c`, `.h`, `.java`, `.ts`, `.tsx`, `.jsx`, `.rs`, `.go`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.r`, `.vue`, `.svelte`, `.html`, `.css`, `.scss`, `.less`, `.json`, `.yaml`, `.yml`, `.xml`, `.toml`, `.ini`, `.gradle`, `.maven`

### Test File Patterns
- Path contains `/test/`, `/tests/`, `/spec/`
- Filename matches: `*_test.*`, `test_*.py`, `*.spec.*`

### Documentation Patterns
- Extensions: `.md`, `.rst`, `.txt`
- Path contains: `/doc/`, `/docs/`
- Filename: `README*`, `CHANGELOG`, `CONTRIBUTING`

### Core vs Peripheral Definition
**Per-repository:**
- **Core**: Top 10% of contributors by total files modified
- **Peripheral**: Remaining contributors

(Can be tuned in `classify_and_aggregate.py` for sensitivity analysis)

---

## Output Interpretation

### Key Findings to Report

1. **Task Specialization:**
   - Do core contributors engage in more diverse tasks (source + tests + docs)?
   - Do peripheral contributors focus on a single task type?

2. **Task-Type Distribution:**
   - What proportion of core vs peripheral work is documentation?
   - Is testing concentrated among core contributors?

3. **Statistical Significance:**
   - Chi-square p-value < 0.05 → core/peripheral status significantly predicts task engagement
   - Logistic regression coefficients → quantify effect size

### Relating to Onion Model

- **Onion Model prediction**: Different task types occupy different structural positions
  - Core = bug fixing + core development
  - Middle = features, maintenance
  - Peripheral = documentation
---


1. **Alternative core definitions:**
   - Top 5%, Top 15%
   - Contributors covering 80% of contributions

2. **Time windows:**
   - Last 5 years only
   - First 5 years after repo creation

3. **Per-category analysis:**
   - Compare "visual" vs "other" categories
   - Do results hold across different project types?

Edit `classify_and_aggregate.py` to implement these.

---

## Troubleshooting

**Rate limit hit during fetch:**
- Script sleeps automatically; monitor console for "sleeping" messages
- If repeated: GitHub API rate limit is 5000 req/hr for authenticated requests
- Try reducing `--max-commits-per-repo` or use `--category` to limit scope

**Missing output file:**
- Ensure paths have no trailing backslashes
- Check that `fetched_commits/` directory exists after step 1
- Verify `repos_info.json` and `categories_info.json` exist

**No rows in CSV:**
- Fetched repos may not have commits with file data
- Check one `fetched_commits/*.json` manually to verify format
- Rerun fetch with smaller category or different repos

---

## Files in This Analysis

- `fetch_commits_files.py` — GitHub API fetcher
- `classify_and_aggregate.py` — File classification & aggregation
- `stats_analysis.py` — Statistical tests
- `visualization.py` — Figure generation
- `contributors_tasks.csv` — Canonical dataset (output)
- `figures/` — Generated PNG plots (output)

