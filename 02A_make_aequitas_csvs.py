# 02A_make_aequitas_csvs.py  (UPDATED)
import os, pandas as pd

IN_DIR  = "outputs"
OUT_DIR = "outputs_aeq"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    "prefix3_simple.csv","prefix3_complex.csv",
    "prefix5_simple.csv","prefix5_complex.csv",
    "prefix10_simple.csv","prefix10_complex.csv",
]

# Demographic column names (include if available)
SENS_ALL = ["gender_cat","citizen_cat","age_group","german_speaking_cat","religious_cat"]

for fn in FILES:
    df = pd.read_csv(os.path.join(IN_DIR, fn))

    # Attribute columns: activities/encoding (Act*, Res*) + demographics (if available)
    act_cols = [c for c in df.columns if c.startswith("Act")]
    res_cols = [c for c in df.columns if c.startswith("Res")]  # may be empty in simple
    sens_cols = [c for c in SENS_ALL if c in df.columns]

    attr_cols = act_cols + res_cols + sens_cols

    # AEQ format: label -> label_value, score remains the same
    if "label" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{fn}: label/score columns not found.")

    aeq = df.rename(columns={"label":"label_value"})[["label_value","score"] + attr_cols].copy()

    # Aequitas requirement: convert all features except label_value/score to string
    for c in attr_cols:
        aeq[c] = aeq[c].astype(str).fillna("Unknown")

    # Type safety (0/1)
    aeq["label_value"] = aeq["label_value"].astype(int)
    aeq["score"] = aeq["score"].astype(int)

    out = os.path.join(OUT_DIR, fn.replace(".csv","_aeq.csv"))
    aeq.to_csv(out, index=False)
    print(f"✓ saved: {out} | attrs={len(attr_cols)} ({', '.join(attr_cols[:6])}{' ...' if len(attr_cols)>6 else ''})")

print("✓ completed: outputs_aeq")
