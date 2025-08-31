import pandas as pd

ALL = pd.read_csv("outputs_reports/_summary_all_disparities.csv")
sens = {"gender_cat","citizen_cat","age_group","german_speaking_cat","religious_cat"}

# 1) PPR violation counts (total)
ppr = ALL[ALL["metric"].str.upper().eq("PPR")].copy()
grp1 = (ppr.groupby(["prefix_len","encoding"])["violation_80pct"]
           .sum().reset_index().sort_values(["prefix_len","encoding"]))
print("\n[PPR 80% violation counts]\n", grp1.to_string(index=False))

# 2) PPR – sensitive attributes vs. dynamic attributes
ppr["is_sensitive"] = ppr["attribute"].isin(sens)
grp2 = (ppr.groupby(["prefix_len","encoding","is_sensitive"])["violation_80pct"]
           .sum().reset_index())
print("\n[PPR violations: sensitive vs dynamic]\n", grp2.to_string(index=False))

# 3) Worst PPR (n>=30), top 10
ppr["dev"] = (ppr["disparity"] - 1).abs()
stable = ppr.copy()
if "group_n" in stable.columns:
    stable = stable[stable["group_n"].fillna(0) >= 30]
top = (stable.sort_values("dev", ascending=False)
               .head(10)[["prefix_len","encoding","attribute","group","disparity","group_n"]])
print("\n[Top 10 worst PPR (n>=30)]\n", top.to_string(index=False))
