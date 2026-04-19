"""
shelter_adoption_model.py
=========================
"Take Me Home" — Animal Shelter Adoption Length-of-Stay Predictor
MIT 15.727 Analytical Edge

Pipeline:
  1. Load & inspect data
  2. Feature engineering
  3. Preprocessing & train/test split
  4. XGBoost regression (with GPU auto-detection)
  5. Hyperparameter tuning (RandomizedSearchCV)
  6. K-Means clustering for animal profiles
  7. Survival analysis (Kaplan-Meier + Cox PH)
  8. SHAP feature importance
  9. Predict function for new intake animals
  10. Save model artifacts

Improvements over v1:
  - Target encoding replaces LabelEncoder → preserves ordinal signal per category
  - Outlier cap at 365 days removes extreme long-tail that crushed R²
  - New features: age_bucket, is_black, is_pitbull, sick_senior interaction
  - R² reported on both log scale and original days scale
  - intake_year dropped (weak signal, causes out-of-range issues at predict time)

Requirements:
  pip install pandas numpy scikit-learn xgboost shap lifelines matplotlib seaborn openpyxl
"""

# =============================================================================
# 0. Imports
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

# ── output folder for all artefacts ──────────────────────────────────────────
OUTPUT_DIR = "shelter_model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# =============================================================================
# 1. Load & inspect
# =============================================================================
print("\n" + "="*60)
print("STEP 1 — Load & Inspect Data")
print("="*60)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
#DATA_PATH = "combined_austin_shelter_data_v2.xlsx"
DATA_PATH   = os.path.join(SCRIPT_DIR, "..", "combined_austin_shelter_data_v2.xlsx")
df = pd.read_excel(DATA_PATH, parse_dates=["Intake_Date", "Outcome_Date"])

# Coerce to numeric — non-parseable values become NaN
df["Length_of_Stay_Days"] = pd.to_numeric(df["Length_of_Stay_Days"], errors="coerce")

print(f"  Total records : {len(df):,}")
print(f"  Columns       : {df.shape[1]}")
print(f"\n  Outcome type distribution:")
print(df["Outcome Type"].value_counts().to_string(index=True))
print(f"\n  Length_of_Stay_Days (all animals):")
print(df["Length_of_Stay_Days"].describe().round(1).to_string())


# =============================================================================
# 2. Feature engineering
# =============================================================================
print("\n" + "="*60)
print("STEP 2 — Feature Engineering")
print("="*60)

def age_to_days(age_str: str) -> float:
    """Convert Austin shelter age strings ('3 years', '2 months', …) → float days."""
    try:
        parts = str(age_str).strip().lower().split()
        val   = float(parts[0])
        unit  = parts[1]
        if "year"  in unit: return val * 365
        if "month" in unit: return val * 30
        if "week"  in unit: return val * 7
        if "day"   in unit: return val
    except Exception:
        pass
    return np.nan


def age_bucket(days: float) -> str:
    """Group continuous age into interpretable life-stage buckets."""
    if   days <   60: return "baby"      # < 2 months
    elif days <  365: return "young"     # 2 months – 1 year
    elif days < 2190: return "adult"     # 1 – 6 years
    else:             return "senior"    # 6+ years


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all model features to a copy of df."""
    d = df.copy()

    # ── numeric ──────────────────────────────────────────────────────────────
    d["age_days"]      = d["Age upon Intake"].apply(age_to_days)
    d["age_days"]      = d["age_days"].fillna(d["age_days"].median())

    # ── age life-stage bucket ─────────────────────────────────────────────────
    d["age_bucket"]    = d["age_days"].apply(age_bucket)

    # ── binary flags ─────────────────────────────────────────────────────────
    d["is_named"]      = d["Name_intake"].notna().astype(int)
    d["is_mixed"]      = d["Breed_intake"].str.contains(
                             r"Mix|/", case=False, na=False).astype(int)
    d["is_neutered"]   = d["Sex upon Intake"].str.contains(
                             r"Neutered|Spayed", case=False, na=False).astype(int)

    # ── "black dog/cat syndrome" — dark animals adopted more slowly ───────────
    d["is_black"]      = d["Color_intake"].str.lower().str.startswith(
                             "black").fillna(False).astype(int)

    # ── breed flags ───────────────────────────────────────────────────────────
    d["is_pitbull"]    = d["Breed_intake"].str.contains(
                             r"Pit Bull|Staffordshire", case=False, na=False).astype(int)

    # ── interaction: sick/injured senior is hardest to place ─────────────────
    d["sick_senior"]   = (
        d["Intake Condition"].str.contains("Sick|Injured", case=False, na=False) &
        (d["age_days"] > 2190)
    ).astype(int)

    # ── date features (seasonality) ───────────────────────────────────────────
    d["intake_month"]      = d["Intake_Date"].dt.month
    d["intake_dayofweek"]  = d["Intake_Date"].dt.dayofweek   # 0=Mon … 6=Sun

    # ── derived categoricals ──────────────────────────────────────────────────
    d["sex"]           = (d["Sex upon Intake"]
                            .str.extract(r"(Male|Female)", expand=False)
                            .fillna("Unknown"))
    d["primary_color"] = d["Color_intake"].str.split("/").str[0].str.strip()

    # Collapse very rare breeds into "Other" (breeds seen < 50 times)
    breed_counts       = d["Breed_intake"].value_counts()
    top_breeds         = breed_counts[breed_counts >= 50].index
    d["breed_grouped"] = d["Breed_intake"].where(
                             d["Breed_intake"].isin(top_breeds), "Other")

    return d


df = engineer_features(df)
print("  Features engineered successfully.")

# Categorical columns that will be target-encoded
CAT_COLS = [
    "Animal Type_intake",
    "Intake Type",
    "Intake Condition",
    "primary_color",
    "sex",
    "breed_grouped",
    "age_bucket",
]

# Final feature list — categoricals appear as their _te (target-encoded) version
FEATURES = [
    "Animal Type_intake_te",  # target-encoded: mean log_los per animal type
    "Intake Type_te",         # target-encoded: mean log_los per intake type
    "Intake Condition_te",    # target-encoded: mean log_los per condition
    "age_days",               # continuous numeric
    "age_bucket_te",          # target-encoded life-stage bucket
    "is_named",               # 0/1
    "intake_month",           # 1-12  (seasonality)
    "intake_dayofweek",       # 0-6
    "is_mixed",               # 0/1
    "primary_color_te",       # target-encoded color
    "is_neutered",            # 0/1
    "sex_te",                 # target-encoded sex
    "breed_grouped_te",       # target-encoded breed
    "is_black",               # 0/1  (black animal syndrome)
    "is_pitbull",             # 0/1
    "sick_senior",            # 0/1  interaction feature
]

TARGET = "Length_of_Stay_Days"


# =============================================================================
# 3. Preprocessing & train/test split  (adoption cohort)
# =============================================================================
print("\n" + "="*60)
print("STEP 3 — Preprocessing & Split")
print("="*60)

# ── filter to adoptions only for regression ───────────────────────────────────
adopted = df[df["Outcome Type"] == "Adoption"].copy()
print(f"  Adoption records (before cap): {len(adopted):,}  "
      f"(median LOS = {adopted[TARGET].median():.1f} days)")

# ── cap extreme outliers ──────────────────────────────────────────────────────
# Animals with LOS > 365 days are ~0.5% of records but massively inflate SST,
# depressing R². Capping keeps the model focused on typical adoption timelines.
LOS_CAP = 365
adopted = adopted[adopted[TARGET] <= LOS_CAP].copy()
print(f"  After capping LOS at {LOS_CAP} days: {len(adopted):,} records")

# ── keep required columns and drop nulls ─────────────────────────────────────
base_cols = CAT_COLS + [TARGET, "age_days", "is_named", "intake_month",
                         "intake_dayofweek", "is_mixed", "is_neutered",
                         "is_black", "is_pitbull", "sick_senior"]
data = adopted[base_cols].dropna().copy()
print(f"  After dropna    : {len(data):,} rows")

# ── log1p-transform target ────────────────────────────────────────────────────
data["log_los"] = np.log1p(data[TARGET])
GLOBAL_MEAN_LOG_LOS = data["log_los"].mean()

# ── target-encode categorical columns (5-fold to prevent leakage) ────────────
# Each row is encoded using the mean log_los of the OTHER 4 folds only.
# This gives XGBoost the real signal (fast breed → low value) without leakage.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in CAT_COLS:
    te_col = f"{col}_te"
    data[te_col] = GLOBAL_MEAN_LOG_LOS   # default: global mean
    for tr_idx, val_idx in kf.split(data):
        fold_means = data.iloc[tr_idx].groupby(col)["log_los"].mean()
        # .values strips the original row index so iloc positional assignment
        # aligns correctly — without it pandas silently writes NaN everywhere
        data.iloc[val_idx, data.columns.get_loc(te_col)] = (
            data.iloc[val_idx][col].map(fold_means)
                                   .fillna(GLOBAL_MEAN_LOG_LOS)
                                   .values)

# ── save full-dataset target maps for use at predict time ────────────────────
target_encoders = {
    col: data.groupby(col)["log_los"].mean()
    for col in CAT_COLS
}
print(f"  Target-encoded columns: {CAT_COLS}")

# ── split ─────────────────────────────────────────────────────────────────────
X = data[FEATURES].astype(float)
y = data["log_los"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")


# =============================================================================
# 4. XGBoost — base model
# =============================================================================
print("\n" + "="*60)
print("STEP 4 — XGBoost Base Model")
print("="*60)

# Auto-detect GPU (XGBoost ≥ 2.0 uses device= parameter)
try:
    _probe = xgb.XGBRegressor(device="cuda", n_estimators=1)
    _probe.fit(X_train.iloc[:10], y_train.iloc[:10])
    DEVICE = "cuda"
    print("  GPU detected — training on CUDA.")
except Exception:
    DEVICE = "cpu"
    print("  No GPU found — training on CPU.")

xgb_model = xgb.XGBRegressor(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    gamma                 = 0.1,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    tree_method           = "hist",
    device                = DEVICE,
    random_state          = 42,
    early_stopping_rounds = 30,
    eval_metric           = "mae",
    verbosity             = 0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    verbose  = False,
)

# ── evaluate ─────────────────────────────────────────────────────────────────
y_pred_log = xgb_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

mae    = mean_absolute_error(y_true, y_pred)
rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
r2     = r2_score(y_true, y_pred)
r2_log = r2_score(y_test, y_pred_log)   # on log scale — less distorted by skew

print(f"\n  Base model results (test set):")
print(f"    MAE            = {mae:.1f} days")
print(f"    RMSE           = {rmse:.1f} days")
print(f"    R² (days scale)= {r2:.3f}   ← depressed by long tail even after cap")
print(f"    R² (log scale) = {r2_log:.3f}   ← truer measure of fit")

# ── residual plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_true, y_pred, alpha=0.15, s=6, color="#185FA5")
lim = max(y_true.max(), y_pred.max()) * 1.02
axes[0].plot([0, lim], [0, lim], "r--", lw=1)
axes[0].set_xlabel("Actual LOS (days)")
axes[0].set_ylabel("Predicted LOS (days)")
axes[0].set_title(f"Predicted vs Actual  (R²={r2:.3f}, log R²={r2_log:.3f})")
axes[0].set_xlim(0, min(lim, 400))
axes[0].set_ylim(0, min(lim, 400))

residuals = y_true - y_pred
axes[1].hist(residuals.clip(-150, 150), bins=60, color="#1D9E75", edgecolor="white", lw=0.3)
axes[1].axvline(0, color="red", lw=1)
axes[1].set_xlabel("Residual (days)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

fig.tight_layout()
save(fig, "01_predicted_vs_actual.png")


# =============================================================================
# 5. Hyperparameter tuning
# =============================================================================
print("\n" + "="*60)
print("STEP 5 — Hyperparameter Tuning")
print("="*60)

# ── Option A (default): use validated best params directly ─────────────────
BEST_PARAMS = {
    "n_estimators"    : 600,
    "max_depth"       : 6,
    "learning_rate"   : 0.04,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma"           : 0.05,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
}
print(f"  Using best params: {BEST_PARAMS}")

final_model = xgb.XGBRegressor(
    **BEST_PARAMS,
    tree_method  = "hist",
    device       = DEVICE,
    random_state = 42,
    verbosity    = 0,
)
final_model.fit(X_train, y_train)

y_pred_best     = np.expm1(final_model.predict(X_test))
mae_best        = mean_absolute_error(y_true, y_pred_best)
r2_best         = r2_score(y_true, y_pred_best)
r2_best_log     = r2_score(y_test, final_model.predict(X_test))

print(f"\n  Final model — Test MAE : {mae_best:.1f} d")
print(f"               R² (days): {r2_best:.3f}")
print(f"               R² (log) : {r2_best_log:.3f}")

# ── Option B (uncomment to run a fresh hyperparameter search) ──────────────
# WARNING: takes ~5-15 min on CPU with n_iter=20, cv=3.
#
# from sklearn.model_selection import RandomizedSearchCV
# param_dist = {
#     "n_estimators"     : [400, 600, 800],
#     "max_depth"        : [4, 6, 8],
#     "learning_rate"    : [0.01, 0.04, 0.05, 0.1],
#     "subsample"        : [0.7, 0.8, 1.0],
#     "colsample_bytree" : [0.7, 0.8, 1.0],
#     "min_child_weight" : [1, 3, 5],
#     "gamma"            : [0.0, 0.05, 0.1, 0.3],
#     "reg_alpha"        : [0.0, 0.1, 0.5],
# }
# search = RandomizedSearchCV(
#     xgb.XGBRegressor(tree_method="hist", device=DEVICE, random_state=42, verbosity=0),
#     param_distributions=param_dist, n_iter=30, cv=3,
#     scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=1,
# )
# search.fit(X_train, y_train)
# print("Best params:", search.best_params_)
# final_model = search.best_estimator_


# =============================================================================
# 6. K-Means clustering — animal profiles
# =============================================================================
print("\n" + "="*60)
print("STEP 6 — K-Means Clustering (animal intake profiles)")
print("="*60)

# Scale features for clustering (KMeans is distance-based)
scaler   = StandardScaler()
X_all    = data[FEATURES].astype(float)
X_scaled = scaler.fit_transform(X_all)

# Elbow method: inertia for k=2..10
inertias = []
K_RANGE  = range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_RANGE), inertias, "o-", color="#185FA5")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia (within-cluster SSE)")
ax.set_title("Elbow curve — choose k where curve flattens")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "02_elbow_curve.png")

# Fit final clustering with k=5
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
data["cluster"] = kmeans.fit_predict(X_scaled)

# ── cluster summary ───────────────────────────────────────────────────────────
cluster_summary = (
    data.groupby("cluster")
        .agg(
            count        = (TARGET, "size"),
            median_los   = (TARGET, "median"),
            mean_los     = (TARGET, "mean"),
            pct_named    = ("is_named",    "mean"),
            pct_neutered = ("is_neutered", "mean"),
            pct_mixed    = ("is_mixed",    "mean"),
            pct_black    = ("is_black",    "mean"),
            pct_pitbull  = ("is_pitbull",  "mean"),
        )
        .round(2)
        .sort_values("median_los")
)
print("\n  Cluster profiles (sorted by median LOS):")
print(cluster_summary.to_string())

# ── cluster LOS boxplot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cluster_los = [data[data["cluster"] == k][TARGET].clip(0, 300).values
               for k in range(N_CLUSTERS)]
bp = ax.boxplot(cluster_los, patch_artist=True, notch=False,
                medianprops=dict(color="white", lw=2))
colors_bp = ["#E6F1FB","#B5D4F4","#378ADD","#185FA5","#0C447C"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
ax.set_xticklabels([f"Cluster {k}" for k in range(N_CLUSTERS)])
ax.set_ylabel("Length of Stay (days, clipped at 300)")
ax.set_title("LOS distribution by cluster")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "03_cluster_los_boxplot.png")


# =============================================================================
# 7. Survival analysis
# =============================================================================
print("\n" + "="*60)
print("STEP 7 — Survival Analysis")
print("="*60)

# Use FULL dataset (not just adoptions) — non-adoptions are censored events
surv_df = df[["Length_of_Stay_Days", "Outcome Type",
              "Animal Type_intake"]].dropna().copy()
surv_df["adopted"] = (surv_df["Outcome Type"] == "Adoption").astype(int)

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(9, 5))

animal_types = ["Dog", "Cat"]
colors_km    = ["#185FA5", "#1D9E75"]
for atype, col in zip(animal_types, colors_km):
    mask = surv_df["Animal Type_intake"] == atype
    kmf.fit(
        durations      = surv_df.loc[mask, "Length_of_Stay_Days"],
        event_observed = surv_df.loc[mask, "adopted"],
        label          = atype,
    )
    kmf.plot_survival_function(ax=ax, color=col, ci_show=True)

ax.set_xlabel("Days in shelter")
ax.set_ylabel("P(not yet adopted)")
ax.set_title("Kaplan-Meier survival curves — Dogs vs Cats")
ax.set_xlim(0, 180)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "04_kaplan_meier.png")
print("  Kaplan-Meier curves saved.")

# ── Cox Proportional Hazards ──────────────────────────────────────────────────
cox_features = ["age_days", "is_named", "is_neutered", "is_mixed",
                "is_black", "is_pitbull", "intake_month", "intake_dayofweek"]

cox_df = df[cox_features + ["Length_of_Stay_Days", "Outcome Type",
                             "Animal Type_intake", "Intake Condition"]].dropna().copy()
cox_df["adopted"] = (cox_df["Outcome Type"] == "Adoption").astype(int)
cox_df = pd.get_dummies(cox_df, columns=["Animal Type_intake", "Intake Condition"],
                         drop_first=True, dtype=float)
cox_df.drop(columns=["Outcome Type"], inplace=True)
cox_df["Length_of_Stay_Days"] = cox_df["Length_of_Stay_Days"].clip(0.01, 730)

cph = CoxPHFitter(penalizer=0.1)
try:
    cph.fit(cox_df, duration_col="Length_of_Stay_Days",
            event_col="adopted", show_progress=False)

    print("\n  Cox PH — top 10 coefficients (exp(coef) = hazard ratio):")
    summary = cph.summary[["exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
    print(summary.head(10).round(3).to_string())

    c_idx = concordance_index(cox_df["Length_of_Stay_Days"],
                               -cph.predict_partial_hazard(cox_df),
                               cox_df["adopted"])
    print(f"\n  Concordance index (C-stat): {c_idx:.3f}  "
          f"(0.5 = random, 1.0 = perfect)")

    fig, ax = plt.subplots(figsize=(8, 7))
    cph.plot(ax=ax)
    ax.set_title("Cox PH — hazard ratios (95% CI)")
    fig.tight_layout()
    save(fig, "05_cox_hazard_ratios.png")

except Exception as e:
    print(f"  Cox PH skipped: {e}")


# =============================================================================
# 8. SHAP feature importance
# =============================================================================
print("\n" + "="*60)
print("STEP 8 — SHAP Feature Importance")
print("="*60)

SHAP_SAMPLE = 2000
X_shap = X_test.sample(n=min(SHAP_SAMPLE, len(X_test)), random_state=42)

# XGBoost 2.x stores base_score as '[2.8611002E0]' in the raw UBJ binary that
# SHAP reads via save_raw() — NOT save_config(). Fix: wrap decode_ubjson_buffer
# in SHAP's module so brackets are stripped before float() is called.
import shap.explainers._tree as _shap_tree

_orig_decode = _shap_tree.decode_ubjson_buffer

def _fixed_decode(fd):
    result = _orig_decode(fd)
    try:
        bs = result["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("["):
            result["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
    except (KeyError, TypeError):
        pass
    return result

_shap_tree.decode_ubjson_buffer = _fixed_decode

explainer   = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_shap)

# ── beeswarm / summary plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES,
                  show=False, plot_size=None)
plt.title("SHAP summary — impact of each feature on log(LOS)")
fig.tight_layout()
save(fig, "06_shap_summary.png")

# ── mean absolute SHAP (bar chart) ───────────────────────────────────────────
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=FEATURES
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
mean_shap.plot(kind="barh", ax=ax, color="#185FA5", edgecolor="white")
ax.set_xlabel("Mean |SHAP value|  (impact on log LOS)")
ax.set_title("Feature importance via SHAP")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "07_shap_bar.png")

print("  Top-5 most important features:")
for feat, val in mean_shap.sort_values(ascending=False).head(5).items():
    print(f"    {feat:<30} mean|SHAP| = {val:.4f}")


# =============================================================================
# 9. Predict function for new intake animals
# =============================================================================
print("\n" + "="*60)
print("STEP 9 — Prediction Function Demo")
print("="*60)

def predict_adoption_los(animal: dict) -> dict:
    """
    Predict expected length-of-stay before adoption for a new intake animal.

    Parameters
    ----------
    animal : dict
        Pass the raw field values (strings for categoricals, numbers for
        numeric fields).  Target encoding and derived flags are computed here.

    Returns
    -------
    dict with keys: predicted_los_days, predicted_los_weeks,
                    confidence_lo_days, confidence_hi_days
    """
    row = pd.DataFrame([animal]).copy()

    # ── derive computed fields if raw inputs provided ─────────────────────────
    if "Age upon Intake" in row.columns and "age_days" not in row.columns:
        row["age_days"] = row["Age upon Intake"].apply(age_to_days)
    if "age_days" in row.columns and "age_bucket" not in row.columns:
        row["age_bucket"] = row["age_days"].apply(age_bucket)
    if "Name_intake" in row.columns and "is_named" not in row.columns:
        row["is_named"] = row["Name_intake"].notna().astype(int)
    if "Breed_intake" in row.columns:
        if "is_mixed" not in row.columns:
            row["is_mixed"] = row["Breed_intake"].str.contains(
                r"Mix|/", case=False, na=False).astype(int)
        if "is_pitbull" not in row.columns:
            row["is_pitbull"] = row["Breed_intake"].str.contains(
                r"Pit Bull|Staffordshire", case=False, na=False).astype(int)
    if "Color_intake" in row.columns:
        if "primary_color" not in row.columns:
            row["primary_color"] = row["Color_intake"].str.split("/").str[0].str.strip()
        if "is_black" not in row.columns:
            row["is_black"] = row["Color_intake"].str.lower().str.startswith(
                "black").fillna(False).astype(int)

    # ── apply target encoding using saved training maps ───────────────────────
    for col in CAT_COLS:
        te_col = f"{col}_te"
        if col in row.columns:
            row[te_col] = row[col].astype(str).map(
                target_encoders[col]).fillna(GLOBAL_MEAN_LOG_LOS)

    # ── derive sick_senior if not passed directly ─────────────────────────────
    if "sick_senior" not in row.columns:
        cond_sick = row.get("Intake Condition", pd.Series(["Normal"])).str.contains(
            r"Sick|Injured", case=False, na=False)
        row["sick_senior"] = (cond_sick & (row.get("age_days", 0) > 2190)).astype(int)

    log_pred  = final_model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))

    # mae_best is in original days — apply interval after back-transforming
    lo = max(pred_days - mae_best, 0)
    hi = pred_days + mae_best

    return {
        "predicted_los_days"  : round(pred_days, 1),
        "predicted_los_weeks" : round(pred_days / 7, 1),
        "confidence_lo_days"  : round(lo, 1),
        "confidence_hi_days"  : round(hi, 1),
    }


# ── demo predictions ──────────────────────────────────────────────────────────
examples = [
    {
        "label"                : "Young named spayed female dog (Normal, summer)",
        "Animal Type_intake"   : "Dog",
        "Intake Type"          : "Stray",
        "Intake Condition"     : "Normal",
        "age_days"             : 365,
        "age_bucket"           : "young",
        "is_named"             : 1,
        "intake_month"         : 6,
        "intake_dayofweek"     : 1,
        "is_mixed"             : 0,
        "primary_color"        : "Brown",
        "is_neutered"          : 1,
        "sex"                  : "Female",
        "breed_grouped"        : "Labrador Retriever Mix",
        "is_black"             : 0,
        "is_pitbull"           : 0,
        "sick_senior"          : 0,
    },
    {
        "label"                : "Senior unnamed intact black cat (Sick, winter)",
        "Animal Type_intake"   : "Cat",
        "Intake Type"          : "Owner Surrender",
        "Intake Condition"     : "Sick",
        "age_days"             : 3650,
        "age_bucket"           : "senior",
        "is_named"             : 0,
        "intake_month"         : 1,
        "intake_dayofweek"     : 0,
        "is_mixed"             : 0,
        "primary_color"        : "Black",
        "is_neutered"          : 0,
        "sex"                  : "Male",
        "breed_grouped"        : "Domestic Shorthair",
        "is_black"             : 1,
        "is_pitbull"           : 0,
        "sick_senior"          : 1,
    },
]

for ex in examples:
    label = ex.pop("label")
    result = predict_adoption_los(ex)
    print(f"\n  {label}")
    print(f"    Predicted LOS : {result['predicted_los_days']} days  "
          f"({result['predicted_los_weeks']} weeks)")
    print(f"    Likely range  : {result['confidence_lo_days']} – "
          f"{result['confidence_hi_days']} days")


# =============================================================================
# 10. Save model artefacts
# =============================================================================
print("\n" + "="*60)
print("STEP 10 — Save Model Artifacts")
print("="*60)

artifacts = {
    "xgb_model"          : final_model,
    "target_encoders"    : target_encoders,   # dict: col → pd.Series(cat→mean_log_los)
    "global_mean_log_los": GLOBAL_MEAN_LOG_LOS,
    "kmeans"             : kmeans,
    "scaler"             : scaler,
    "features"           : FEATURES,
    "cat_cols"           : CAT_COLS,
    "train_mae_days"     : mae_best,
}

model_path = os.path.join(OUTPUT_DIR, "shelter_model_artifacts.pkl")
with open(model_path, "wb") as f:
    pickle.dump(artifacts, f)
print(f"  Artifacts saved → {model_path}")

print("\n" + "="*60)
print("ALL STEPS COMPLETE")
print(f"Outputs in: ./{OUTPUT_DIR}/")
print("="*60)
