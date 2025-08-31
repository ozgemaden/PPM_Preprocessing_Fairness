import pandas as pd, os
RAW = "hiring_log_medium.csv"
out = "outputs/activity_map.csv"
df = pd.read_csv(RAW)
acts = sorted(df["Activity"].dropna().unique())
rows = [{"ActTok": f"A{str(i+1).zfill(2)}", "Activity": a} for i, a in enumerate(acts)]
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("✓ saved:", out)
