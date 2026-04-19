"""
predict_dogs.py
===============
Standalone prediction script for the Dog adoption LOS model.

Usage:
    cd dogs
    python predict_dogs.py

Requires: dogs/shelter_model_output_dogs/dog_model_artifacts.pkl
          (produced by running train_dogs.py at least once)
"""

import pickle
import numpy as np
import pandas as pd
import os

# =============================================================================
# Load artifacts
# =============================================================================
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(SCRIPT_DIR, "shelter_model_output_dogs", "dog_model_artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    art = pickle.load(f)

model               = art["xgb_model"]
target_encoders     = art["target_encoders"]
GLOBAL_MEAN         = art["global_mean_log_los"]
FEATURES            = art["features"]
CAT_COLS            = art["cat_cols"]
mae_best            = art["train_mae_days"]

print(f"Dog model loaded — MAE: {mae_best:.1f} days")
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
    if   days <   60: return "puppy"
    elif days <  365: return "young"
    elif days < 2190: return "adult"
    else:             return "senior"


# =============================================================================
# Prediction function
# =============================================================================
def predict_dog_los(animal: dict) -> dict:
    """
    Predict adoption LOS for a dog.

    Pass raw field values. All derived flags and target encoding are computed
    automatically. Only fields actually listed in FEATURES need to be provided.

    Field reference:
      Intake Type       : "Stray" | "Owner Surrender" | "Public Assist"
      Intake Condition  : "Normal" | "Sick" | "Injured" | "Aged" | "Feral"
      age_days          : numeric (365=1yr, 730=2yr, 90=3mo)
      age_bucket        : "puppy" | "young" | "adult" | "senior"
      is_named          : 1 / 0
      intake_month      : 1–12
      intake_dayofweek  : 0=Mon … 6=Sun
      is_mixed          : 1 / 0
      primary_color     : "Black" | "Brown" | "White" | "Tan" …
      is_neutered       : 1 / 0
      sex               : "Male" | "Female"
      breed_grouped     : breed name (or "Other" if rare)
      is_black          : 1 / 0
      is_large_breed    : 1 / 0  (Lab, GSD, Rottweiler, etc.)
      is_bully_breed    : 1 / 0  (Pit Bull, Staffordshire, Am. Bulldog, etc.)
      sick_senior       : 1 = sick/injured AND age > 6 yrs, else 0
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
# Edit animals_to_predict and run: python predict_dogs.py
# =============================================================================
animals_to_predict = [
    {
        "label"           : "Young named neutered male Pit Bull (Surrender, summer)",
        "Intake Type"     : "Owner Surrender",
        "Intake Condition": "Normal",
        "age_days"        : 730,
        "age_bucket"      : "young",
        "is_named"        : 1,
        "intake_month"    : 6,
        "intake_dayofweek": 1,
        "is_mixed"        : 0,
        "primary_color"   : "Brown",
        "is_neutered"     : 1,
        "sex"             : "Male",
        "breed_grouped"   : "Pit Bull",
        "is_black"        : 0,
        "is_large_breed"  : 0,
        "is_bully_breed"  : 1,
        "sick_senior"     : 0,
    },
    {
         "label"           : "Senior unnamed intact female Lab (Stray, winter)",
         "Intake Type"     : "Stray",
         "Intake Condition": "Normal",
         "age_days"        : 3650,
         "age_bucket"      : "senior",
         "is_named"        : 0,
         "intake_month"    : 1,
         "intake_dayofweek": 0,
         "is_mixed"        : 0,
         "primary_color"   : "Yellow",
         "is_neutered"     : 0,
         "sex"             : "Female",
         "breed_grouped"   : "Labrador Retriever",
         "is_black"        : 0,
         "is_large_breed"  : 1,
         "is_bully_breed"  : 0,
         "sick_senior"     : 0,
    },
]

print("=" * 55)
print("DOG ADOPTION LENGTH-OF-STAY PREDICTIONS")
print("=" * 55)

for entry in animals_to_predict:
    label  = entry.pop("label")
    result = predict_dog_los(entry)
    print(f"\n  {label}")
    print(f"    Predicted : {result['predicted_days']} days  ({result['predicted_weeks']} weeks)")
    print(f"    Range     : {result['range_low_days']} – {result['range_high_days']} days")

print("\n" + "=" * 55)
