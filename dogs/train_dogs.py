"""
train_dogs.py
=============
"Take Me Home" — Dog-only Adoption Length-of-Stay Predictor
MIT 15.727 Analytical Edge

Dog-specific improvements over the combined model:
  - Filtered to dogs only: breed signal is purely within-species
  - Animal Type removed from features (redundant — all rows are dogs)
  - Added is_large_breed flag (large dogs historically have longer stays)
  - Added is_bully_breed flag (broader than just Pit Bull)

Run from this directory:
    cd dogs
    python train_dogs.py

Data file is read from the parent directory.
Outputs saved to dogs/shelter_model_output_dogs/
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "..", "combined_austin_shelter_data_v2.xlsx")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "shelter_model_output_dogs")
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
print("STEP 1 — Load & Inspect Data (Dogs only)")
print("="*60)

df = pd.read_excel(DATA_PATH, parse_dates=["Intake_Date", "Outcome_Date"])
df["Length_of_Stay_Days"] = pd.to_numeric(df["Length_of_Stay_Days"], errors="coerce")

# ── filter to dogs ────────────────────────────────────────────────────────────
df = df[df["Animal Type_intake"] == "Dog"].copy()
print(f"  Dog records : {len(df):,}")
print(f"\n  Outcome type distribution:")
print(df["Outcome Type"].value_counts().to_string())
print(f"\n  Length_of_Stay_Days:")
print(df["Length_of_Stay_Days"].describe().round(1).to_string())


# =============================================================================
# 2. Feature engineering
# =============================================================================
print("\n" + "="*60)
print("STEP 2 — Feature Engineering")
print("="*60)

# Large breed list — commonly take longer to adopt due to space requirements
LARGE_BREEDS = {
    "labrador retriever", "german shepherd", "golden retriever", "rottweiler",
    "great dane", "saint bernard", "husky", "malamute", "akita", "doberman",
    "weimaraner", "boxer", "bernese mountain", "newfoundland", "great pyrenees",
    "irish wolfhound", "cane corso", "mastiff",
}

BULLY_BREEDS = {
    "pit bull", "staffordshire", "american bulldog", "bull terrier",
    "american bully",
}

def age_to_days(age_str: str) -> float:
    try:
        parts = str(age_str).strip().lower().split()
        val, unit = float(parts[0]), parts[1]
        if "year"  in unit: return val * 365
        if "month" in unit: return val * 30
        if "week"  in unit: return val * 7
        if "day"   in unit: return val
    except Exception:
        pass
    return np.nan

def age_bucket(days: float) -> str:
    if   days <   60: return "puppy"
    elif days <  365: return "young"
    elif days < 2190: return "adult"
    else:             return "senior"

def engineer_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_days"]       = d["Age upon Intake"].apply(age_to_days)
    d["age_days"]       = d["age_days"].fillna(d["age_days"].median())
    d["age_bucket"]     = d["age_days"].apply(age_bucket)

    d["is_named"]       = d["Name_intake"].notna().astype(int)
    d["is_mixed"]       = d["Breed_intake"].str.contains(
                              r"Mix|/", case=False, na=False).astype(int)
    d["is_neutered"]    = d["Sex upon Intake"].str.contains(
                              r"Neutered|Spayed", case=False, na=False).astype(int)
    d["is_black"]       = d["Color_intake"].str.lower().str.startswith(
                              "black").fillna(False).astype(int)

    # Dog-specific: large breeds have longer stays (space/cost concerns)
    breed_lower         = d["Breed_intake"].str.lower().fillna("")
    d["is_large_breed"] = breed_lower.apply(
        lambda b: int(any(lb in b for lb in LARGE_BREEDS)))

    # Dog-specific: bully breeds (broader than just Pit Bull)
    d["is_bully_breed"] = breed_lower.apply(
        lambda b: int(any(bb in b for bb in BULLY_BREEDS)))

    # Sick or injured senior dog — very hard to place
    d["sick_senior"]    = (
        d["Intake Condition"].str.contains("Sick|Injured", case=False, na=False) &
        (d["age_days"] > 2190)
    ).astype(int)

    d["intake_month"]      = d["Intake_Date"].dt.month
    d["intake_dayofweek"]  = d["Intake_Date"].dt.dayofweek

    d["sex"]            = (d["Sex upon Intake"]
                             .str.extract(r"(Male|Female)", expand=False)
                             .fillna("Unknown"))
    d["primary_color"]  = d["Color_intake"].str.split("/").str[0].str.strip()

    breed_counts        = d["Breed_intake"].value_counts()
    top_breeds          = breed_counts[breed_counts >= 30].index   # lower threshold: dogs only
    d["breed_grouped"]  = d["Breed_intake"].where(
                              d["Breed_intake"].isin(top_breeds), "Other")
    return d

df = engineer_features(df)
print("  Features engineered.")

CAT_COLS = [
    "Intake Type",
    "Intake Condition",
    "primary_color",
    "sex",
    "breed_grouped",
    "age_bucket",
]

FEATURES = [
    "Intake Type_te",
    "Intake Condition_te",
    "age_days",
    "age_bucket_te",
    "is_named",
    "intake_month",
    "intake_dayofweek",
    "is_mixed",
    "primary_color_te",
    "is_neutered",
    "sex_te",
    "breed_grouped_te",
    "is_black",
    "is_large_breed",    # dog-specific
    "is_bully_breed",    # dog-specific
    "sick_senior",
]

TARGET = "Length_of_Stay_Days"


# =============================================================================
# 3. Preprocessing & split
# =============================================================================
print("\n" + "="*60)
print("STEP 3 — Preprocessing & Split")
print("="*60)

adopted = df[df["Outcome Type"] == "Adoption"].copy()
print(f"  Dog adoption records (before cap): {len(adopted):,}  "
      f"(median LOS = {adopted[TARGET].median():.1f} days)")

LOS_CAP = 365
adopted = adopted[adopted[TARGET] <= LOS_CAP].copy()
print(f"  After capping at {LOS_CAP} days: {len(adopted):,} records")

base_cols = CAT_COLS + [TARGET, "age_days", "is_named", "intake_month",
                         "intake_dayofweek", "is_mixed", "is_neutered",
                         "is_black", "is_large_breed", "is_bully_breed", "sick_senior"]
data = adopted[base_cols].dropna().copy()
print(f"  After dropna: {len(data):,} rows")

data["log_los"] = np.log1p(data[TARGET])
GLOBAL_MEAN = data["log_los"].mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in CAT_COLS:
    te_col = f"{col}_te"
    data[te_col] = GLOBAL_MEAN
    for tr_idx, val_idx in kf.split(data):
        fold_means = data.iloc[tr_idx].groupby(col)["log_los"].mean()
        data.iloc[val_idx, data.columns.get_loc(te_col)] = (
            data.iloc[val_idx][col].map(fold_means)
                                   .fillna(GLOBAL_MEAN)
                                   .values)

target_encoders = {col: data.groupby(col)["log_los"].mean() for col in CAT_COLS}
print(f"  Target-encoded: {CAT_COLS}")

X = data[FEATURES].astype(float)
y = data["log_los"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# =============================================================================
# 4. XGBoost
# =============================================================================
print("\n" + "="*60)
print("STEP 4 — XGBoost")
print("="*60)

try:
    import json
    _probe = xgb.XGBRegressor(device="cuda", n_estimators=1)
    _probe.fit(X_train.iloc[:10], y_train.iloc[:10])
    _actual = json.loads(_probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    DEVICE = _actual if _actual == "cuda" else "cpu"
except Exception:
    DEVICE = "cpu"
print(f"  Device: {DEVICE.upper()}")

xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", device=DEVICE, random_state=42,
    early_stopping_rounds=30, eval_metric="mae", verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_log = xgb_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

mae    = mean_absolute_error(y_true, y_pred)
rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
r2     = r2_score(y_true, y_pred)
r2_log = r2_score(y_test, y_pred_log)

print(f"\n  Base model (dogs):")
print(f"    MAE            = {mae:.1f} days")
print(f"    RMSE           = {rmse:.1f} days")
print(f"    R² (days)      = {r2:.3f}")
print(f"    R² (log scale) = {r2_log:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_true, y_pred, alpha=0.15, s=6, color="#185FA5")
lim = max(y_true.max(), y_pred.max()) * 1.02
axes[0].plot([0, lim], [0, lim], "r--", lw=1)
axes[0].set_xlabel("Actual LOS (days)")
axes[0].set_ylabel("Predicted LOS (days)")
axes[0].set_title(f"Dogs — Predicted vs Actual  (log R²={r2_log:.3f})")
axes[0].set_xlim(0, min(lim, 400)); axes[0].set_ylim(0, min(lim, 400))
residuals = y_true - y_pred
axes[1].hist(residuals.clip(-150, 150), bins=60, color="#1D9E75", edgecolor="white", lw=0.3)
axes[1].axvline(0, color="red", lw=1)
axes[1].set_xlabel("Residual (days)"); axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
fig.tight_layout(); save(fig, "01_predicted_vs_actual.png")


# =============================================================================
# 5. Hyperparameter tuning
# =============================================================================
print("\n" + "="*60)
print("STEP 5 — Final Model")
print("="*60)

BEST_PARAMS = {
    "n_estimators": 600, "max_depth": 6, "learning_rate": 0.04,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "gamma": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0,
}
final_model = xgb.XGBRegressor(**BEST_PARAMS, tree_method="hist",
                                device=DEVICE, random_state=42, verbosity=0)
final_model.fit(X_train, y_train)

y_pred_best  = np.expm1(final_model.predict(X_test))
mae_best     = mean_absolute_error(y_true, y_pred_best)
r2_best      = r2_score(y_true, y_pred_best)
r2_best_log  = r2_score(y_test, final_model.predict(X_test))
print(f"  Final — MAE: {mae_best:.1f} d  |  R²(days): {r2_best:.3f}  |  R²(log): {r2_best_log:.3f}")


# =============================================================================
# 6. K-Means clustering
# =============================================================================
print("\n" + "="*60)
print("STEP 6 — K-Means Clustering")
print("="*60)

scaler   = StandardScaler()
X_all    = data[FEATURES].astype(float)
X_scaled = scaler.fit_transform(X_all)

inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    inertias.append(km.fit(X_scaled).inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(2, 11), inertias, "o-", color="#185FA5")
ax.set_xlabel("k"); ax.set_ylabel("Inertia")
ax.set_title("Elbow curve — Dogs"); ax.grid(True, alpha=0.3)
fig.tight_layout(); save(fig, "02_elbow_curve.png")

N_CLUSTERS = 5
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
data["cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = (
    data.groupby("cluster")
        .agg(count=(TARGET,"size"), median_los=(TARGET,"median"),
             mean_los=(TARGET,"mean"), pct_named=("is_named","mean"),
             pct_neutered=("is_neutered","mean"), pct_large=("is_large_breed","mean"),
             pct_bully=("is_bully_breed","mean"))
        .round(2).sort_values("median_los")
)
print("\n  Dog cluster profiles:")
print(cluster_summary.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
cluster_los = [data[data["cluster"]==k][TARGET].clip(0,300).values for k in range(N_CLUSTERS)]
bp = ax.boxplot(cluster_los, patch_artist=True, medianprops=dict(color="white", lw=2))
colors_bp = ["#E6F1FB","#B5D4F4","#378ADD","#185FA5","#0C447C"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
ax.set_xticklabels([f"Cluster {k}" for k in range(N_CLUSTERS)])
ax.set_ylabel("LOS (days, clipped 300)"); ax.set_title("Dog LOS by cluster")
ax.grid(True, axis="y", alpha=0.3); fig.tight_layout(); save(fig, "03_cluster_boxplot.png")


# =============================================================================
# 7. Survival analysis (dogs only)
# =============================================================================
print("\n" + "="*60)
print("STEP 7 — Survival Analysis (Dogs)")
print("="*60)

surv_df = df[["Length_of_Stay_Days","Outcome Type","is_bully_breed"]].dropna().copy()
surv_df["adopted"] = (surv_df["Outcome Type"] == "Adoption").astype(int)

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(9, 5))
for flag, label, color in [(0,"Non-bully","#185FA5"), (1,"Bully breed","#E05C2A")]:
    mask = surv_df["is_bully_breed"] == flag
    kmf.fit(surv_df.loc[mask,"Length_of_Stay_Days"],
            surv_df.loc[mask,"adopted"], label=label)
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
ax.set_xlabel("Days in shelter"); ax.set_ylabel("P(not yet adopted)")
ax.set_title("Kaplan-Meier — Bully vs Non-bully dogs")
ax.set_xlim(0, 180); ax.grid(True, alpha=0.3)
fig.tight_layout(); save(fig, "04_kaplan_meier_dogs.png")
print("  Kaplan-Meier saved.")

cox_features = ["age_days","is_named","is_neutered","is_mixed",
                "is_black","is_large_breed","is_bully_breed","intake_month"]
cox_df = df[cox_features + ["Length_of_Stay_Days","Outcome Type","Intake Condition"]].dropna().copy()
cox_df["adopted"] = (cox_df["Outcome Type"] == "Adoption").astype(int)
cox_df = pd.get_dummies(cox_df, columns=["Intake Condition"], drop_first=True, dtype=float)
cox_df.drop(columns=["Outcome Type"], inplace=True)
cox_df["Length_of_Stay_Days"] = cox_df["Length_of_Stay_Days"].clip(0.01, 730)

cph = CoxPHFitter(penalizer=0.1)
try:
    cph.fit(cox_df, duration_col="Length_of_Stay_Days",
            event_col="adopted", show_progress=False)
    print("\n  Cox PH — top coefficients:")
    summary = cph.summary[["exp(coef)","p"]].sort_values("exp(coef)", ascending=False)
    print(summary.head(10).round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    cph.plot(ax=ax); ax.set_title("Cox PH — Dogs hazard ratios")
    fig.tight_layout(); save(fig, "05_cox_dogs.png")
except Exception as e:
    print(f"  Cox PH skipped: {e}")


# =============================================================================
# 8. SHAP
# =============================================================================
print("\n" + "="*60)
print("STEP 8 — SHAP Feature Importance (Dogs)")
print("="*60)

X_shap = X_test.sample(n=min(2000, len(X_test)), random_state=42)

import shap.explainers._tree as _shap_tree, json as _json
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

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_values, X_shap, feature_names=FEATURES, show=False, plot_size=None)
plt.title("SHAP summary — Dogs"); fig.tight_layout(); save(fig, "06_shap_summary.png")

mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
mean_shap.plot(kind="barh", ax=ax, color="#185FA5", edgecolor="white")
ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Feature importance — Dogs")
ax.grid(True, axis="x", alpha=0.3); fig.tight_layout(); save(fig, "07_shap_bar.png")

print("  Top-5 features (dogs):")
for feat, val in mean_shap.sort_values(ascending=False).head(5).items():
    print(f"    {feat:<28} {val:.4f}")


# =============================================================================
# 9. Predict function demo
# =============================================================================
print("\n" + "="*60)
print("STEP 9 — Prediction Demo")
print("="*60)

def predict_dog_los(animal: dict) -> dict:
    row = pd.DataFrame([animal]).copy()
    if "age_days" in row.columns and "age_bucket" not in row.columns:
        row["age_bucket"] = row["age_days"].apply(age_bucket)
    for col in CAT_COLS:
        te_col = f"{col}_te"
        if col in row.columns:
            row[te_col] = row[col].astype(str).map(target_encoders[col]).fillna(GLOBAL_MEAN)
    if "sick_senior" not in row.columns:
        row["sick_senior"] = 0
    log_pred  = final_model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))
    return {
        "predicted_los_days"  : round(pred_days, 1),
        "predicted_los_weeks" : round(pred_days / 7, 1),
        "confidence_lo_days"  : round(max(pred_days - mae_best, 0), 1),
        "confidence_hi_days"  : round(pred_days + mae_best, 1),
    }

examples = [
    {"Intake Type":"Stray","Intake Condition":"Normal","age_days":365,"age_bucket":"young",
     "is_named":1,"intake_month":6,"intake_dayofweek":1,"is_mixed":0,"primary_color":"Brown",
     "is_neutered":1,"sex":"Female","breed_grouped":"Labrador Retriever Mix",
     "is_black":0,"is_large_breed":1,"is_bully_breed":0,"sick_senior":0,
     "_label":"Young named female Labrador (Normal)"},
    {"Intake Type":"Owner Surrender","Intake Condition":"Normal","age_days":2555,"age_bucket":"adult",
     "is_named":0,"intake_month":1,"intake_dayofweek":0,"is_mixed":0,"primary_color":"Brown",
     "is_neutered":0,"sex":"Male","breed_grouped":"Pit Bull",
     "is_black":0,"is_large_breed":0,"is_bully_breed":1,"sick_senior":0,
     "_label":"Adult unnamed intact male Pit Bull (Surrender)"},
]

for ex in examples:
    label = ex.pop("_label")
    result = predict_dog_los(ex)
    print(f"\n  {label}")
    print(f"    Predicted : {result['predicted_los_days']} days ({result['predicted_los_weeks']} weeks)")
    print(f"    Range     : {result['confidence_lo_days']} – {result['confidence_hi_days']} days")


# =============================================================================
# 10. Save artifacts
# =============================================================================
print("\n" + "="*60)
print("STEP 10 — Save Artifacts")
print("="*60)

artifacts = {
    "xgb_model"          : final_model,
    "target_encoders"    : target_encoders,
    "global_mean_log_los": GLOBAL_MEAN,
    "kmeans"             : kmeans,
    "scaler"             : scaler,
    "features"           : FEATURES,
    "cat_cols"           : CAT_COLS,
    "train_mae_days"     : mae_best,
    "species"            : "Dog",
}
model_path = os.path.join(OUTPUT_DIR, "dog_model_artifacts.pkl")
with open(model_path, "wb") as f:
    pickle.dump(artifacts, f)
print(f"  Saved → {model_path}")

print("\n" + "="*60)
print("DOG MODEL COMPLETE")
print(f"Outputs in: {OUTPUT_DIR}")
print("="*60)
