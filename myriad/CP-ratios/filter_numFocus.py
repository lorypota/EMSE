import pandas as pd

NUMFOCUS_CSV = "numfocus_repo_summary.csv"
OUT_FILTERED = "numfocus_repo_summary_matched_p10_p90.csv"

p10 = 3.0   
p90 = 327.5

numfocus = pd.read_csv(NUMFOCUS_CSV)

numfocus_matched = numfocus[
    (numfocus["n_contributors_total"] >= p10) &
    (numfocus["n_contributors_total"] <= p90)
].copy()

numfocus_matched.to_csv(OUT_FILTERED, index=False)

print("NumFOCUS before:", len(numfocus))
print("NumFOCUS after:", len(numfocus_matched))
# NumFOCUS before: 2216
# NumFOCUS after: 950