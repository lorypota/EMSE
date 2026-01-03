import pandas as pd

myriad = pd.read_csv("./gh_contributors/myriad_repo_summary.csv")

size_col = "n_identified" 

p10 = myriad[size_col].quantile(0.10)
p90 = myriad[size_col].quantile(0.90)

print("p10:", p10, "p90:", p90)
# p10: 3.0 p90: 327.5