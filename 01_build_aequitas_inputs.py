import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RAW = "hiring_log_medium.csv"
OUTDIR = "outputs"   # format for Aequitas input
os.makedirs(OUTDIR, exist_ok=True)

# ---------- helpers ----------
def bin_age(a):
    try: a = float(a)
    except Exception: return "unk"
    if a < 30: return "18-29"
    if a < 45: return "30-44"
    if a < 65: return "45-64"
    return "65+"

def to_cat(x):
    s = str(x).lower()
    if s in ["true","t","1","yes","y"]:  return "T"
    if s in ["false","f","0","no","n"]:  return "F"
    return str(x)

def one_hot(df, cols):
    # Convert Act/Res to one-hot for Decision Tree
    return pd.get_dummies(df[cols].astype(str), dummy_na=False)

# ---------- read data + sort by time ----------
df = pd.read_csv(RAW)
time_col = "Complete Timestamp" if "Complete Timestamp" in df.columns else "time"
df["_ts"] = pd.to_datetime(df[time_col], errors="coerce")
df["__row"] = np.arange(len(df))
df = df.sort_values(["case","_ts","__row"])

# ---------- label: case contains "Make Job Offer" = 1 ----------
label_by_case = (
    df.groupby("case")["Activity"]
      .apply(lambda s: 1 if (s == "Make Job Offer").any() else 0)
      .rename("label")
      .reset_index()
)

# ---------- reduce sensitive attributes to case-level ----------
static_raw = ["gender","citizen","german speaking","religious","age"]
have = [c for c in static_raw if c in df.columns]
statics = df.groupby("case").first().reset_index()[["case"]+have]

if "age" in statics.columns:
    statics["age_group"] = statics["age"].apply(bin_age)

for c in ["gender","citizen","german speaking","religious"]:
    if c in statics.columns:
        statics[c+"_cat"] = statics[c].apply(to_cat)

# simplify column name with space
if "german speaking_cat" in statics.columns:
    statics = statics.rename(columns={"german speaking_cat":"german_speaking_cat"})

SENS_COLS = [c for c in ["gender_cat","citizen_cat","age_group","german_speaking_cat","religious_cat"] if c in statics.columns]

# ---------- map Activities to short tokens (for readable headers) ----------
acts_sorted = sorted(df["Activity"].dropna().unique())
act2tok = {a: f"A{str(i+1).zfill(2)}" for i,a in enumerate(acts_sorted)}

# ---------- prefix table builder ----------
def build_prefix_table(k: int, complex_enc: bool):
    # take first k events for each case; skip traces shorter than k
    f = df.groupby("case").head(k)
    counts = f.groupby("case").size().rename("cnt").reset_index()
    valid_cases = counts.loc[counts["cnt"]==k, "case"]
    f = f[f["case"].isin(valid_cases)].copy()
    f["ord"] = f.groupby("case").cumcount() + 1

    # activity token
    f["ActTok"] = f["Activity"].map(act2tok)

    acts = f.pivot(index="case", columns="ord", values="ActTok")
    acts.columns = [f"Act{c}" for c in acts.columns]
    out = acts

    if complex_enc:
        if "resource" not in f.columns:
            raise ValueError("complex encoding requested but 'resource' column not found.")
        ress = f.pivot(index="case", columns="ord", values="resource")
        ress.columns = [f"Res{c}" for c in ress.columns]
        out = out.join(ress, how="left")

    out = (out.reset_index()
              .merge(label_by_case, on="case", how="left")
              .merge(statics[["case"]+SENS_COLS], on="case", how="left"))
    out["PrefixLen"] = k
    return out

def build_predict_and_save(k: int, tag: str):
    """
    - Produces CSV format for Aequitas:
      case, PrefixLen, Act1..Actk, [Res1..Resk], sens..., label, score
    - score: DecisionTree.predict (argmax) -> binary 0/1
    """
    enc = build_prefix_table(k, complex_enc=(tag=="complex"))

    feat_cols = [c for c in enc.columns if c.startswith("Act") or c.startswith("Res")]
    # one-hot for DT
    X_all = one_hot(enc, feat_cols)
    y_all = enc["label"].astype(int)
    cases = enc["case"]

    # case-based split (no data leakage)
    X_tr, X_te, y_tr, y_te, cases_tr, cases_te = train_test_split(
        X_all, y_all, cases, test_size=0.2, random_state=42, stratify=y_all
    )

    # model: DT + class_weight=balanced (reduces bias towards 0)
    clf = DecisionTreeClassifier(random_state=42, class_weight="balanced",
                                 max_depth=None, min_samples_leaf=1)
    clf.fit(X_tr, y_tr)

    # "highest probability value" -> argmax ==> predict (0/1)
    yhat_te = clf.predict(X_te).astype(int)

    # OUTPUT (Aequitas format)
    cols_output = ["case","PrefixLen"] + feat_cols + SENS_COLS
    out = enc.loc[X_te.index, cols_output].copy()
    out["label"] = y_te.values.astype(int)
    out["score"] = yhat_te

    # health check
    lp, sp = float(out["label"].mean()), float(out["score"].mean())
    print(f"[k={k} {tag}] n={len(out)}  label_pos={lp:.3f}  score_pos={sp:.3f}")

    # save
    path = os.path.join(OUTDIR, f"prefix{k}_{tag}.csv")
    out.to_csv(path, index=False)

for k in [3,5,10]:
    build_predict_and_save(k, "simple")
    build_predict_and_save(k, "complex")

print(f"✓ completed -> {OUTDIR}")
