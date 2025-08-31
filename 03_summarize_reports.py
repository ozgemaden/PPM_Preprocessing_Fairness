import os, re, glob
import numpy as np
import pandas as pd

IN_DIR = "outputs_reports"
OUT_ALL = os.path.join(IN_DIR, "_summary_all_disparities.csv")
OUT_VIO = os.path.join(IN_DIR, "_violations_only.csv")

# ---- Helpers ----------------------------------------------------------------

def find_first(dcols, candidates):
    """Find first match in lowercase column name."""
    for c in candidates:
        if c in dcols:
            return c
    return None

# Common disparity columns in Aequitas -> readable names
METRIC_NAME_MAP = {
    "ppr_disparity": "PPR",            # Predicted Positive Rate (selection rate)
    "pprev_disparity": "PPR",
    "precision_disparity": "PPV",      # Positive Predictive Value (precision)
    "fpr_disparity": "FPR",
    "fnr_disparity": "FNR",
    "tpr_disparity": "TPR",
    "fdr_disparity": "FDR",            # False Discovery Rate
    "for_disparity": "FOR",            # False Omission Rate
    # add more here if needed
}

# Lower bound for small support warning
MIN_N = 30

def parse_fname(path):
    """Extract prefix and encoding from filename."""
    fn = os.path.basename(path)
    m = re.search(r"prefix(\d+)_(simple|complex).*?_disparities\.csv$", fn, flags=re.I)
    if m:
        return int(m.group(1)), m.group(2).lower()
    # if aeq filename is different, still return something
    return None, None

def severity_from_disp(x):
    """Severity label for deviation (optional, looks good in report)."""
    if pd.isna(x):
        return ""
    d = abs(x - 1.0)
    if d < 0.10:   return "low"
    if d < 0.25:   return "moderate"
    if d < 0.50:   return "high"
    return "severe"

def harm_direction(metric, disp):
    """Harm direction (for easier interpretation)."""
    if pd.isna(disp): return ""
    m = metric.upper()
    if m in ["PPR","PPV","TPR"]:     # expected to be 'good'
        return "under" if disp < 1 else "over"
    if m in ["FPR","FNR","FDR","FOR"]:  # error rates (high is bad)
        return "higher" if disp > 1 else "lower"
    return "diff"

# ---- Main flow --------------------------------------------------------------

long_rows = []

files = sorted(glob.glob(os.path.join(IN_DIR, "*_disparities.csv")))
if not files:
    print(f"Warning: No *_disparities.csv found in {IN_DIR}.")
    raise SystemExit(1)

for path in files:
    k, enc = parse_fname(path)
    df = pd.read_csv(path)
    # normalize columns
    original_cols = list(df.columns)
    df.columns = [c.lower() for c in df.columns]
    cols = set(df.columns)

    # attribute and group columns
    col_attr = find_first(cols, ["attribute_name","attribute","attr"])
    col_grp  = find_first(cols, ["attribute_value","group_value","group","value"])

    if not col_attr or not col_grp:
        print(f"[!] {os.path.basename(path)}: attribute/group columns not found."
              f" Columns: {original_cols}")
        continue

    # try to capture support (group size)
    col_n = find_first(cols, ["group_size","count","n","group_count","group_total"])

    # collect available disparity columns
    disp_cols = [c for c in df.columns if c.endswith("_disparity")]
    # prefer known ones; but take all if none known
    known_disp = [c for c in disp_cols if c in METRIC_NAME_MAP]
    use_cols = known_disp if known_disp else disp_cols

    if not use_cols:
        print(f"[!] {os.path.basename(path)}: no disparity column found.")
        continue

    sub = df[[col_attr, col_grp] + ([col_n] if col_n else []) + use_cols].copy()

    # long form
    long = sub.melt(id_vars=[col_attr, col_grp] + ([col_n] if col_n else []),
                    value_vars=use_cols,
                    var_name="metric_raw", value_name="disparity")

    # readable metric name
    long["metric"] = long["metric_raw"].map(METRIC_NAME_MAP).fillna(long["metric_raw"].str.replace("_disparity","", regex=False).str.upper())

    # prefix & encoding
    long["prefix_len"] = k
    long["encoding"]   = enc

    # violation (80% rule: outside 0.8–1.25 band)
    long["violation_80pct"] = (long["disparity"] < 0.8) | (long["disparity"] > 1.25)

    # severity and direction
    long["severity"] = long["disparity"].apply(severity_from_disp)
    long["harm_dir"] = [harm_direction(m, d) for m, d in zip(long["metric"], long["disparity"])]

    # small support warning
    if col_n:
        long["small_support"] = long[col_n].fillna(0).astype(float) < MIN_N
    else:
        long["small_support"] = False

    # arrange columns
    keep_cols = ["prefix_len","encoding", col_attr, col_grp, (col_n if col_n else None),
                 "metric","disparity","violation_80pct","severity","harm_dir","small_support"]
    keep_cols = [c for c in keep_cols if c is not None]
    long = long[keep_cols].rename(columns={
        col_attr: "attribute",
        col_grp: "group",
        (col_n if col_n else ""): "group_n"
    })

    long_rows.append(long)

# combine
ALL = pd.concat(long_rows, ignore_index=True)

# fill NaN group_n if missing
if "group_n" not in ALL.columns:
    ALL["group_n"] = np.nan

# Save
os.makedirs(IN_DIR, exist_ok=True)
ALL.to_csv(OUT_ALL, index=False)

VIOS = ALL.loc[ALL["violation_80pct"]].copy()
VIOS.to_csv(OUT_VIO, index=False)

# ---- Console summary --------------------------------------------------------

def dev_from_1(x):
    try:
        return abs(float(x) - 1.0)
    except Exception:
        return np.nan

print(f"\n✓ Summary saved:\n  - {OUT_ALL}\n  - {OUT_VIO}")

# Top 3 worst deviations for each (prefix, encoding)
for (k, enc), grp in ALL.groupby(["prefix_len","encoding"], dropna=False):
    g2 = grp.copy()
    g2["dev"] = g2["disparity"].apply(dev_from_1)
    worst = g2.sort_values("dev", ascending=False).head(3)
    total_v = int(g2["violation_80pct"].sum())
    print(f"\n[{k}, {enc}]  violation count = {total_v}")
    for _, r in worst.iterrows():
        n_txt = "" if (pd.isna(r["group_n"])) else f", n={int(r['group_n'])}"
        flag  = " ⚠️" if r["violation_80pct"] else ""
        print(f"  • {r['attribute']}={r['group']} | {r['metric']}: disp={r['disparity']:.3f}{n_txt} "
              f"| {r['harm_dir']} ({r['severity']}){flag}")
