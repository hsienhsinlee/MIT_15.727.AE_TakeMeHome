"""
predict_cats.py
===============
Standalone prediction script for the Cat adoption LOS model.

Usage:
    cd cats
    python predict_cats.py

Requires: cats/shelter_model_output_cats/cat_model_artifacts.pkl
          (produced by running train_cats.py at least once)
"""

import pickle
import numpy as np
import pandas as pd
import os

# =============================================================================
# Load artifacts
# =============================================================================
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(SCRIPT_DIR, "shelter_model_output_cats", "cat_model_artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    art = pickle.load(f)

model               = art["xgb_model"]
target_encoders     = art["target_encoders"]
GLOBAL_MEAN         = art["global_mean_log_los"]
FEATURES            = art["features"]
CAT_COLS            = art["cat_cols"]
mae_best            = art["train_mae_days"]

print(f"Cat model loaded — MAE: {mae_best:.1f} days")
print(f"Features: {FEATURES}\n")


# =============================================================================
# Helpers
# =============================================================================
def age_to_days(age_str) -> float:
    try:
        parts = str(age_str).strip().lower().split()
        val, unit = float(parts[0]), parts[1]
        if "year"  in unit: return val * 365
        if "month" in unit: return val * 30
        if "week"  in unit: return val * 7
        if "day"   in unit: return val
    except Exception:
        pass
    return 30.0

def age_bucket(days: float) -> str:
    if   days <   90: return "kitten"
    elif days <  365: return "young"
    elif days < 2555: return "adult"
    else:             return "senior"


# =============================================================================
# Prediction function
# =============================================================================
def predict_cat_los(animal: dict) -> dict:
    """
    Predict adoption LOS for a cat.

    Pass raw field values. All derived flags and target encoding are computed
    automatically.

    Field reference:
      Intake Type       : "Stray" | "Owner Surrender" | "Public Assist"
      Intake Condition  : "Normal" | "Sick" | "Injured" | "Aged" | "Feral"
      age_days          : numeric (365=1yr, 90=3mo, 2555=7yr)
      age_bucket        : "kitten" | "young" | "adult" | "senior"
      is_named          : 1 / 0
      intake_month      : 1–12
      intake_dayofweek  : 0=Mon … 6=Sun
      is_mixed          : 1 / 0
      primary_color     : "Black" | "White" | "Orange" | "Brown" …
      is_neutered       : 1 / 0
      sex               : "Male" | "Female"
      breed_grouped     : "Domestic Shorthair" | "Domestic Longhair" | "Other" …
      is_black          : 1 / 0
      is_domestic       : 1 = Domestic Shorthair/Medium/Longhair, 0 = other
      is_feral          : 1 / 0
      is_longhair       : 1 = longhair breed, 0 = shorthair
      sick_senior       : 1 = sick/injured AND age > 7 yrs, else 0
    """
    row = pd.DataFrame([animal]).copy()

    if "age_days" in row.columns and "age_bucket" not in row.columns:
        row["age_bucket"] = row["age_days"].apply(age_bucket)

    for col in CAT_COLS:
        te_col = f"{col}_te"
        if col in row.columns:
            row[te_col] = row[col].astype(str).map(
                target_encoders[col]).fillna(GLOBAL_MEAN)

    if "sick_senior" not in row.columns:
        row["sick_senior"] = 0

    log_pred  = model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))

    return {
        "predicted_days"  : round(pred_days, 1),
        "predicted_weeks" : round(pred_days / 7, 1),
        "range_low_days"  : round(max(pred_days - mae_best, 0), 1),
        "range_high_days" : round(pred_days + mae_best, 1),
    }


# =============================================================================
# Edit animals_to_predict and run: python predict_cats.py
# =============================================================================
animals_to_predict = [
    {
        "label"           : "Kitten named male domestic shorthair (Normal, summer)",
        "Intake Type"     : "Stray",
        "Intake Condition": "Normal",
        "age_days"        : 90,
        "age_bucket"      : "kitten",
        "is_named"        : 1,
        "intake_month"    : 6,
        "intake_dayofweek": 2,
        "is_mixed"        : 0,
        "primary_color"   : "Orange",
        "is_neutered"     : 0,
        "sex"             : "Male",
        "breed_grouped"   : "Domestic Shorthair",
        "is_black"        : 0,
        "is_domestic"     : 1,
        "is_feral"        : 0,
        "is_longhair"     : 0,
        "sick_senior"     : 0,
    },
    # {
    #     "label"           : "Senior unnamed sick black female (Stray, winter)",
    #     "Intake Type"     : "Stray",
    #     "Intake Condition": "Sick",
    #     "age_days"        : 3650,
    #     "age_bucket"      : "senior",
    #     "is_named"        : 0,
    #     "intake_month"    : 1,
    #     "intake_dayofweek": 0,
    #     "is_mixed"        : 0,
    #     "primary_color"   : "Black",
    #     "is_neutered"     : 0,
    #     "sex"             : "Female",
    #     "breed_grouped"   : "Domestic Shorthair",
    #     "is_black"        : 1,
    #     "is_domestic"     : 1,
    #     "is_feral"        : 0,
    #     "is_longhair"     : 0,
    #     "sick_senior"     : 1,
    # },
]

print("=" * 55)
print("CAT ADOPTION LENGTH-OF-STAY PREDICTIONS")
print("=" * 55)

for entry in animals_to_predict:
    label  = entry.pop("label")
    result = predict_cat_los(entry)
    print(f"\n  {label}")
    print(f"    Predicted : {result['predicted_days']} days  ({result['predicted_weeks']} weeks)")
    print(f"    Range     : {result['range_low_days']} – {result['range_high_days']} days")

print("\n" + "=" * 55)
