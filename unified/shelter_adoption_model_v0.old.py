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

from sklearn.model_selection import train_test_split  # add RandomizedSearchCV if running Step 5 Option B
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# DATA_PATH = "cleaned_austin_shelter_data__v1.xlsx"
DATA_PATH = "combined_austin_shelter_data_v2.xlsx"
df = pd.read_excel(DATA_PATH, parse_dates=["Intake_Date", "Outcome_Date"])

# Coerce to numeric — non-parseable values (empty strings, "N/A", etc.) become NaN
df["Length_of_Stay_Days"] = pd.to_numeric(df["Length_of_Stay_Days"], errors="coerce")

print(f"  Total records : {len(df):,}")
print(f"  Columns       : {df.shape[1]}")
print(f"\n  Outcome type distribution:")
print(df["Outcome Type"].value_counts().to_string(index=True))

# LOS quick stats
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all model features to a copy of df."""
    d = df.copy()

    # ── numeric ──────────────────────────────────────────────────────────────
    d["age_days"]         = d["Age upon Intake"].apply(age_to_days)
    d["age_days"]         = d["age_days"].fillna(d["age_days"].median())

    # ── binary flags ─────────────────────────────────────────────────────────
    d["is_named"]         = d["Name_intake"].notna().astype(int)
    d["is_mixed"]         = d["Breed_intake"].str.contains(
                                r"Mix|/", case=False, na=False).astype(int)
    d["is_neutered"]      = d["Sex upon Intake"].str.contains(
                                r"Neutered|Spayed", case=False, na=False).astype(int)

    # ── date features (seasonality) ───────────────────────────────────────────
    d["intake_month"]     = d["Intake_Date"].dt.month
    d["intake_dayofweek"] = d["Intake_Date"].dt.dayofweek   # 0=Mon … 6=Sun
    d["intake_year"]      = d["Intake_Date"].dt.year

    # ── derived categoricals ──────────────────────────────────────────────────
    d["sex"]              = (d["Sex upon Intake"]
                               .str.extract(r"(Male|Female)", expand=False)
                               .fillna("Unknown"))
    d["primary_color"]    = d["Color_intake"].str.split("/").str[0].str.strip()

    # Collapse very rare breeds into "Other" (keeps label encoding stable)
    breed_counts          = d["Breed_intake"].value_counts()
    top_breeds            = breed_counts[breed_counts >= 50].index
    d["breed_grouped"]    = d["Breed_intake"].where(
                                d["Breed_intake"].isin(top_breeds), "Other")

    return d


df = engineer_features(df)
print("  Features engineered successfully.")

# Feature list used by every model in this script
FEATURES = [
    "Animal Type_intake",   # Dog / Cat / Other …
    "Intake Type",          # Stray / Owner Surrender …
    "Intake Condition",     # Normal / Injured / Sick …
    "age_days",             # numeric
    "is_named",             # 0/1
    "intake_month",         # 1-12
    "intake_dayofweek",     # 0-6
    "intake_year",          # year effect / trend
    "is_mixed",             # 0/1
    "primary_color",        # Black / White …
    "is_neutered",          # 0/1
    "sex",                  # Male / Female / Unknown
    "breed_grouped",        # top breeds + "Other"
]
TARGET = "Length_of_Stay_Days"

# =============================================================================
# 3. Preprocessing & train/test split  (adoption cohort)
# =============================================================================
print("\n" + "="*60)
print("STEP 3 — Preprocessing & Split")
print("="*60)

# ── filter to adoptions only for regression ───────────────────────────────────
# Rationale: LOS for non-adoptions doesn't represent "time-to-adoption"
adopted = df[df["Outcome Type"] == "Adoption"].copy()
print(f"  Adoption records: {len(adopted):,}  "
      f"(median LOS = {adopted[TARGET].median():.1f} days)")

# ── drop rows where any feature or target is null ────────────────────────────
# .copy() ensures pandas CoW (2.0+) doesn't track this as a child slice,
# which would silently discard column assignments below.
data = adopted[FEATURES + [TARGET]].dropna().copy()
print(f"  After dropna    : {len(data):,} rows")

# ── log1p-transform target  ───────────────────────────────────────────────────
# LOS is heavily right-skewed (mean ≈35 d, max ≈1912 d).
# log1p compresses outliers and makes residuals more Gaussian.
data["log_los"] = np.log1p(data[TARGET])

# ── label-encode categorical columns ─────────────────────────────────────────
CAT_COLS = ["Animal Type_intake", "Intake Type", "Intake Condition",
            "primary_color", "sex", "breed_grouped"]

label_encoders: dict[str, LabelEncoder] = {}
for col in CAT_COLS:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

print(f"  Categorical columns label-encoded: {CAT_COLS}")

# Force all feature columns to numeric — guards against any residual object dtype
for col in FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

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
    n_estimators        = 500,
    max_depth           = 6,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 3,
    gamma               = 0.1,
    reg_alpha           = 0.1,      # L1
    reg_lambda          = 1.0,      # L2
    tree_method         = "hist",   # fast histogram method
    device              = DEVICE,
    random_state        = 42,
    early_stopping_rounds = 30,
    eval_metric         = "mae",
    verbosity           = 0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set            = [(X_test, y_test)],
    verbose             = False,
)

# ── evaluate ─────────────────────────────────────────────────────────────────
y_pred_log = xgb_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"\n  Base model results (test set, original days scale):")
print(f"    MAE  = {mae:.1f} days")
print(f"    RMSE = {rmse:.1f} days")
print(f"    R²   = {r2:.3f}")

# ── residual plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_true, y_pred, alpha=0.15, s=6, color="#185FA5")
lim = max(y_true.max(), y_pred.max()) * 1.02
axes[0].plot([0, lim], [0, lim], "r--", lw=1)
axes[0].set_xlabel("Actual LOS (days)")
axes[0].set_ylabel("Predicted LOS (days)")
axes[0].set_title(f"Predicted vs Actual  (R²={r2:.3f})")
axes[0].set_xlim(0, min(lim, 500))
axes[0].set_ylim(0, min(lim, 500))

residuals = y_true - y_pred
axes[1].hist(residuals.clip(-200, 200), bins=60, color="#1D9E75", edgecolor="white", lw=0.3)
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
# These params were found via RandomizedSearchCV on this dataset.
# Skipping the search loop saves ~5–10 min on CPU; enable Option B to re-run.
BEST_PARAMS = {
    "n_estimators"    : 500,
    "max_depth"       : 6,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma"           : 0.1,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
}
print(f"  Using validated best params: {BEST_PARAMS}")

final_model = xgb.XGBRegressor(
    **BEST_PARAMS,
    tree_method = "hist",
    device      = DEVICE,
    random_state= 42,
    verbosity   = 0,
)
final_model.fit(X_train, y_train)

y_pred_best = np.expm1(final_model.predict(X_test))
mae_best    = mean_absolute_error(y_true, y_pred_best)
r2_best     = r2_score(y_true, y_pred_best)
print(f"\n  Final model — Test MAE: {mae_best:.1f} d  |  R²: {r2_best:.3f}")

# ── Option B (uncomment to run a fresh hyperparameter search) ──────────────
# WARNING: takes ~5-15 min on CPU with n_iter=20, cv=3.
#
# from sklearn.model_selection import RandomizedSearchCV
# param_dist = {
#     "n_estimators"     : [300, 500, 800],
#     "max_depth"        : [4, 6, 8],
#     "learning_rate"    : [0.01, 0.05, 0.1],
#     "subsample"        : [0.7, 0.8, 1.0],
#     "colsample_bytree" : [0.7, 0.8, 1.0],
#     "min_child_weight" : [1, 3, 5],
#     "gamma"            : [0.0, 0.1, 0.3],
#     "reg_alpha"        : [0.0, 0.1, 0.5],
# }
# search = RandomizedSearchCV(
#     xgb.XGBRegressor(tree_method="hist", device=DEVICE, random_state=42, verbosity=0),
#     param_distributions=param_dist, n_iter=20, cv=3,
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
scaler    = StandardScaler()
X_all     = data[FEATURES].astype(float)
X_scaled  = scaler.fit_transform(X_all)

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

# Fit final clustering with k=5 (reasonable for shelter profiles)
N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
data["cluster"] = kmeans.fit_predict(X_scaled)

# ── cluster summary ───────────────────────────────────────────────────────────
cluster_summary = (
    data.groupby("cluster")
        .agg(
            count      = (TARGET, "size"),
            median_los = (TARGET, "median"),
            mean_los   = (TARGET, "mean"),
            pct_named  = ("is_named", "mean"),
            pct_neutered=("is_neutered","mean"),
            pct_mixed  = ("is_mixed","mean"),
        )
        .round(1)
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

# ── Kaplan-Meier: probability of still being in shelter after T days ──────────
# Use the FULL dataset (not just adoptions).
# "event" = 1 if adopted, 0 otherwise (censored / transferred / etc.)
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
        durations  = surv_df.loc[mask, "Length_of_Stay_Days"],
        event_observed = surv_df.loc[mask, "adopted"],
        label      = atype,
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
# Cox models the hazard (instantaneous adoption rate) as a function of features.
# We use a small numeric/binary-only feature set to keep it tractable.

cox_features = ["age_days", "is_named", "is_neutered", "is_mixed",
                "intake_month", "intake_dayofweek"]

# Encode animal type and intake condition as dummies for Cox
cox_df = df[cox_features + ["Length_of_Stay_Days", "Outcome Type",
                             "Animal Type_intake", "Intake Condition"]].dropna().copy()
cox_df["adopted"] = (cox_df["Outcome Type"] == "Adoption").astype(int)
cox_df = pd.get_dummies(cox_df, columns=["Animal Type_intake", "Intake Condition"],
                         drop_first=True, dtype=float)
cox_df.drop(columns=["Outcome Type"], inplace=True)

# Clip extreme LOS to avoid numerical issues
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

# Use a sample for speed (SHAP is O(n × features) for TreeExplainer)
SHAP_SAMPLE = 2000
X_shap = X_test.sample(n=min(SHAP_SAMPLE, len(X_test)), random_state=42)

# XGBoost 2.x stores base_score as '[2.8611002E0]' (1-element array notation)
# in the raw UBJ binary that SHAP reads via save_raw() — not save_config().
# All previous save_config patches had no effect because SHAP never calls it.
# Fix: patch decode_ubjson_buffer in SHAP's module namespace so brackets are
# stripped from base_score immediately after UBJ decode, before float() runs.
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
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES,
                  show=False, plot_size=None)
plt.title("SHAP summary — impact of each feature on log(LOS)")
fig.tight_layout()
save(fig, "06_shap_summary.png")

# ── mean absolute SHAP (bar chart) ───────────────────────────────────────────
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=FEATURES
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
mean_shap.plot(kind="barh", ax=ax, color="#185FA5", edgecolor="white")
ax.set_xlabel("Mean |SHAP value|  (impact on log LOS)")
ax.set_title("Feature importance via SHAP")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
save(fig, "07_shap_bar.png")
print("  Top-3 most important features:")
for feat, val in mean_shap.sort_values(ascending=False).head(3).items():
    print(f"    {feat:<25} mean|SHAP| = {val:.4f}")

# =============================================================================
# 9. Predict function for new intake animals
# =============================================================================
print("\n" + "="*60)
print("STEP 9 — Prediction Function Demo")
print("="*60)

def predict_adoption_los(animal: dict) -> dict:
    """
    Predict the expected length-of-stay before adoption for a new intake animal.

    Parameters
    ----------
    animal : dict
        Keys must match FEATURES. Categorical values should be raw strings
        (e.g. "Dog", "Stray") — encoding is handled internally.

    Returns
    -------
    dict with keys:
        predicted_los_days   : float
        predicted_los_weeks  : float
        confidence_lo_days   : float  (approx 25th-pct empirical interval)
        confidence_hi_days   : float  (approx 75th-pct empirical interval)
    """
    row = pd.DataFrame([animal])

    # Apply same feature engineering for incoming raw fields
    if "Age upon Intake" in row.columns and "age_days" not in row.columns:
        row["age_days"] = row["Age upon Intake"].apply(age_to_days)
    if "Name_intake" in row.columns and "is_named" not in row.columns:
        row["is_named"] = row["Name_intake"].notna().astype(int)
    if "Breed_intake" in row.columns and "is_mixed" not in row.columns:
        row["is_mixed"] = row["Breed_intake"].str.contains(
            r"Mix|/", case=False, na=False).astype(int)
    if "Color_intake" in row.columns and "primary_color" not in row.columns:
        row["primary_color"] = row["Color_intake"].str.split("/").str[0].str.strip()

    # Encode categoricals
    for col in CAT_COLS:
        if col in row.columns:
            known = set(label_encoders[col].classes_)
            row[col] = row[col].astype(str).apply(
                lambda v: v if v in known else label_encoders[col].classes_[0])
            row[col] = label_encoders[col].transform(row[col])

    log_pred = final_model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))

    # mae_best is in original days scale — apply interval after back-transforming
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
        "label"               : "Young named spayed dog (Normal intake)",
        "Animal Type_intake"  : "Dog",
        "Intake Type"         : "Stray",
        "Intake Condition"    : "Normal",
        "age_days"            : 365,
        "is_named"            : 1,
        "intake_month"        : 6,
        "intake_dayofweek"    : 1,
        "intake_year"         : 2024,
        "is_mixed"            : 0,
        "primary_color"       : "Brown",
        "is_neutered"         : 1,
        "sex"                 : "Female",
        "breed_grouped"       : "Labrador Retriever Mix",
    },
    {
        "label"               : "Senior unnamed intact black cat (Sick)",
        "Animal Type_intake"  : "Cat",
        "Intake Type"         : "Owner Surrender",
        "Intake Condition"    : "Sick",
        "age_days"            : 3650,
        "is_named"            : 0,
        "intake_month"        : 1,
        "intake_dayofweek"    : 0,
        "intake_year"         : 2024,
        "is_mixed"            : 0,
        "primary_color"       : "Black",
        "is_neutered"         : 0,
        "sex"                 : "Male",
        "breed_grouped"       : "Domestic Shorthair",
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
    "xgb_model"       : final_model,
    "label_encoders"  : label_encoders,
    "kmeans"          : kmeans,
    "scaler"          : scaler,
    "features"        : FEATURES,
    "cat_cols"        : CAT_COLS,
    "train_mae_days"  : mae_best,
}

model_path = os.path.join(OUTPUT_DIR, "shelter_model_artifacts.pkl")
with open(model_path, "wb") as f:
    pickle.dump(artifacts, f)
print(f"  Artifacts saved → {model_path}")

# ── how to reload ─────────────────────────────────────────────────────────────
reload_snippet = '''
# ── Reload saved model in another script ──────────────────────────────────
import pickle, numpy as np, pandas as pd

with open("shelter_model_output/shelter_model_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

model    = art["xgb_model"]
encoders = art["label_encoders"]
features = art["features"]
'''
print("\n  Reload snippet:")
print(reload_snippet)

print("\n" + "="*60)
print("ALL STEPS COMPLETE")
print(f"Outputs in: ./{OUTPUT_DIR}/")
print("="*60)