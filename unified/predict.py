"""
predict.py
==========
Standalone prediction script for the Shelter Adoption LOS model (v2).

Usage:
    python predict.py

Edit the `animals_to_predict` list at the bottom to query the model.
Requires: shelter_model_output/shelter_model_artifacts.pkl  (produced by
          running shelter_adoption_model.py at least once).
"""

import pickle
import numpy as np
import pandas as pd

# =============================================================================
# Load saved artifacts
# =============================================================================
ARTIFACTS_PATH = "shelter_model_output/shelter_model_artifacts.pkl"

with open(ARTIFACTS_PATH, "rb") as f:
    art = pickle.load(f)

model              = art["xgb_model"]
target_encoders    = art["target_encoders"]     # dict: col → Series(cat→mean_log_los)
GLOBAL_MEAN_LOG_LOS= art["global_mean_log_los"] # fallback for unseen categories
FEATURES           = art["features"]
CAT_COLS           = art["cat_cols"]
mae_best           = art["train_mae_days"]

print(f"Model loaded from {ARTIFACTS_PATH}")
print(f"  Features : {FEATURES}")
print(f"  Train MAE: {mae_best:.1f} days\n")


# =============================================================================
# Helpers
# =============================================================================
def age_to_days(age_str) -> float:
    """Convert '2 years', '3 months', '1 week', '4 days' → float days."""
    try:
        parts = str(age_str).strip().lower().split()
        val, unit = float(parts[0]), parts[1]
        if "year"  in unit: return val * 365
        if "month" in unit: return val * 30
        if "week"  in unit: return val * 7
        if "day"   in unit: return val
    except Exception:
        pass
    return 30.0  # fallback ~1 month


def age_bucket(days: float) -> str:
    if   days <   60: return "baby"
    elif days <  365: return "young"
    elif days < 2190: return "adult"
    else:             return "senior"


# =============================================================================
# Prediction function
# =============================================================================
def predict_adoption_los(animal: dict) -> dict:
    """
    Predict expected length-of-stay (days) before adoption for a new animal.

    Parameters
    ----------
    animal : dict
        Pass raw field values — strings for categoricals, numbers for numeric
        fields. Derived flags and target encoding are computed here automatically.

    Returns
    -------
    dict with keys:
        predicted_days  : float  — point estimate
        predicted_weeks : float  — same, in weeks
        range_low_days  : float  — lower bound (point – MAE)
        range_high_days : float  — upper bound (point + MAE)
    """
    row = pd.DataFrame([animal]).copy()

    # ── derive computed fields from raw inputs if provided ────────────────────
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

    # ── derive sick_senior interaction if not directly provided ───────────────
    if "sick_senior" not in row.columns:
        cond_col = "Intake Condition" if "Intake Condition" in row.columns else None
        if cond_col:
            is_sick = row[cond_col].str.contains(r"Sick|Injured", case=False, na=False)
        else:
            is_sick = pd.Series([False] * len(row))
        row["sick_senior"] = (is_sick & (row.get("age_days", 0) > 2190)).astype(int)

    log_pred  = model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))

    lo = max(pred_days - mae_best, 0)
    hi = pred_days + mae_best

    return {
        "predicted_days"  : round(pred_days, 1),
        "predicted_weeks" : round(pred_days / 7, 1),
        "range_low_days"  : round(lo, 1),
        "range_high_days" : round(hi, 1),
    }


# =============================================================================
# Query — edit this list to predict for any animal
# =============================================================================
#
# Field reference:
#   Animal Type_intake  : "Dog" | "Cat" | "Bird" | "Rabbit" | "Other"
#   Intake Type         : "Stray" | "Owner Surrender" | "Public Assist"
#   Intake Condition    : "Normal" | "Sick" | "Injured" | "Aged" | "Feral"
#   age_days            : numeric days  (365=1yr, 730=2yrs, 90=3mo)
#   age_bucket          : "baby" | "young" | "adult" | "senior"
#   is_named            : 1 = has a name, 0 = unnamed
#   intake_month        : 1–12
#   intake_dayofweek    : 0=Monday … 6=Sunday
#   is_mixed            : 1 = mixed breed, 0 = purebred
#   primary_color       : "Black" | "White" | "Brown" | "Orange" | "Tan" …
#   is_neutered         : 1 = spayed/neutered, 0 = intact
#   sex                 : "Female" | "Male"
#   breed_grouped       : common breed name or "Other"
#   is_black            : 1 = predominantly black coat, 0 = otherwise
#   is_pitbull          : 1 = Pit Bull / Staffordshire, 0 = otherwise
#   sick_senior         : 1 = sick/injured AND age > 6 years, 0 = otherwise

animals_to_predict = [
    {
        "label"              : "Young neutered male dog (Normal, summer)",
        "Animal Type_intake" : "Dog",
        "Intake Type"        : "Owner Surrender",
        "Intake Condition"   : "Normal",
        "age_days"           : 180,
        "age_bucket"         : "baby",
        "is_named"           : 1,
        "intake_month"       : 6,
        "intake_dayofweek"   : 1,
        "is_mixed"           : 0,
        "primary_color"      : "White",
        "is_neutered"        : 1,
        "sex"                : "Male",
        "breed_grouped"      : "Labrador Retriever",
        "is_black"           : 0,
        "is_pitbull"         : 0,
        "sick_senior"        : 0,
    },
    # {
    #     "label"              : "Senior unnamed intact male black cat (Sick, winter)",
    #     "Animal Type_intake" : "Cat",
    #     "Intake Type"        : "Owner Surrender",
    #     "Intake Condition"   : "Sick",
    #     "age_days"           : 3650,
    #     "age_bucket"         : "senior",
    #     "is_named"           : 0,
    #     "intake_month"       : 1,
    #     "intake_dayofweek"   : 0,
    #     "is_mixed"           : 0,
    #     "primary_color"      : "Black",
    #     "is_neutered"        : 0,
    #     "sex"                : "Male",
    #     "breed_grouped"      : "Domestic Shorthair",
    #     "is_black"           : 1,
    #     "is_pitbull"         : 0,
    #     "sick_senior"        : 1,
    # },
]


# =============================================================================
# Run predictions
# =============================================================================
print("=" * 55)
print("ADOPTION LENGTH-OF-STAY PREDICTIONS")
print("=" * 55)

for entry in animals_to_predict:
    label  = entry.pop("label")
    result = predict_adoption_los(entry)
    print(f"\n  {label}")
    print(f"    Predicted : {result['predicted_days']} days  "
          f"({result['predicted_weeks']} weeks)")
    print(f"    Range     : {result['range_low_days']} – "
          f"{result['range_high_days']} days")

print("\n" + "=" * 55)
print("To predict a different animal, edit 'animals_to_predict'")
print("at the bottom of this file and re-run: python predict.py")
print("=" * 55)
