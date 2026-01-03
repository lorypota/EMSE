import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt

myriad = pd.read_csv("./gh_contributors/myriad_repo_summary.csv")
numfocus = pd.read_csv("numfocus_repo_summary_matched_p10_p90.csv")

myriad_ratios = myriad["core_peripheral_ratio"].dropna()
numfocus_ratios = numfocus["core_peripheral_ratio"].dropna()

u, p = mannwhitneyu(myriad_ratios, numfocus_ratios, alternative="two-sided")

print("MYRIAD median:", np.median(myriad_ratios))
print("NumFOCUS median:", np.median(numfocus_ratios))
print("Mann–Whitney U p-value:", p)

# MYRIAD median: 0.2631578947368421
# NumFOCUS median: 0.3333333333333333
# Mann–Whitney U p-value: 1.640995996132447e-07

plt.boxplot(
    [myriad_ratios, numfocus_ratios],
    labels=["MYRIAD (Art OSS)", "NumFOCUS (Sci OSS)"],
    showfliers=False
)
plt.ylabel("Core / Peripheral Contributor Ratio")
plt.show()
