# 02B_run_aequitas_audit.py  (UPDATED)
import os, warnings, pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias

warnings.filterwarnings("ignore", category=FutureWarning)

IN_DIR  = "outputs_aeq"
OUT_DIR = "outputs_reports"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    "prefix3_simple_aeq.csv","prefix3_complex_aeq.csv",
    "prefix5_simple_aeq.csv","prefix5_complex_aeq.csv",
    "prefix10_simple_aeq.csv","prefix10_complex_aeq.csv",
]

def choose_ref_value(series: pd.Series, colname: str) -> str:
    """Reference group: prioritize T/45-64 for demographics; otherwise mode."""
    vc = series.astype(str).value_counts(dropna=False)
    if colname.endswith("_cat") and "T" in vc.index:
        return "T"
    if colname == "age_group" and "45-64" in vc.index:
        return "45-64"
    return str(vc.index[0])  # most frequent

def audit_one(path: str):
    df = pd.read_csv(path)
    # AEQ default names
    if "label_value" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path}: label_value/score not found.")

    df["label_value"] = df["label_value"].astype(int)
    df["score"] = df["score"].astype(int)

    # Attributes: everything except label_value/score (02A already removes case/PrefixLen)
    attr_cols = [c for c in df.columns if c not in ["label_value","score"]]

    # References: one value for each attr
    refs = {c: choose_ref_value(df[c], c) for c in attr_cols}

    # Aequitas pipeline
    g = Group()
    # Not using unsupported parameters; providing attr_cols
    xtab, _ = g.get_crosstabs(df, attr_cols=attr_cols)

    b = Bias()
    bdf = b.get_disparity_predefined_groups(
        xtab, original_df=df, ref_groups_dict=refs,
        alpha=0.05, mask_significance=True
    )

    # Save
    base = os.path.basename(path).replace(".csv","")
    out_path = os.path.join(OUT_DIR, f"{base}_disparities.csv")
    bdf.to_csv(out_path, index=False)

    print("get_disparity_predefined_group()")
    print(f"✓ {base} | attrs={attr_cols[:6]}{'...' if len(attr_cols)>6 else ''} | refs(sample)={ {k:refs[k] for k in list(refs)[:5]} }")
    print(f"→ saved: {out_path}")

for fn in FILES:
    audit_one(os.path.join(IN_DIR, fn))

print(f"✓ reports -> {OUT_DIR}")
