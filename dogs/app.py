"""
app.py — "Take Me Home" Dog Adoption LOS Predictor
Flask REST API backend

Usage:
    pip install flask flask-cors
    python app.py

Endpoint:
    POST /predict   — JSON body with dog attributes → prediction JSON
    GET  /health    — health check
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(SCRIPT_DIR, "shelter_model_output_dogs", "dog_model_artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    art = pickle.load(f)

model            = art["xgb_model"]
target_encoders  = art["target_encoders"]
GLOBAL_MEAN      = art["global_mean_log_los"]
FEATURES         = art["features"]
CAT_COLS         = art["cat_cols"]
MAE_DAYS         = art["train_mae_days"]

print(f"Dog model loaded — MAE: {MAE_DAYS:.1f} days")

# ---------------------------------------------------------------------------
# Domain knowledge (mirrors train_dogs.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def age_bucket(days: float) -> str:
    if   days <   60: return "puppy"
    elif days <  365: return "young"
    elif days < 2190: return "adult"
    else:             return "senior"


def derive_flags(data: dict) -> dict:
    """Auto-derive binary flags from user-supplied values."""
    d = dict(data)

    breed_lower = str(d.get("breed_grouped", "")).lower()
    color_lower = str(d.get("primary_color", "")).lower()
    condition   = str(d.get("Intake Condition", "")).lower()
    age_days    = float(d.get("age_days", 365))

    d.setdefault("is_black",      int(color_lower.startswith("black")))
    d.setdefault("is_large_breed", int(any(lb in breed_lower for lb in LARGE_BREEDS)))
    d.setdefault("is_bully_breed", int(any(bb in breed_lower for bb in BULLY_BREEDS)))
    d.setdefault("sick_senior",    int(
        any(k in condition for k in ("sick", "injured")) and age_days > 2190
    ))
    return d


def predict(animal: dict) -> dict:
    animal = derive_flags(animal)
    row    = pd.DataFrame([animal]).copy()

    # Derive age_bucket if missing
    if "age_days" in row.columns and "age_bucket" not in row.columns:
        row["age_bucket"] = row["age_days"].apply(age_bucket)

    # Target-encode categorical columns
    for col in CAT_COLS:
        te_col = f"{col}_te"
        if col in row.columns:
            row[te_col] = (
                row[col].astype(str)
                        .map(target_encoders[col])
                        .fillna(GLOBAL_MEAN)
            )

    log_pred  = model.predict(row[FEATURES].astype(float))[0]
    pred_days = float(np.expm1(log_pred))

    return {
        "predicted_days" : round(pred_days, 1),
        "predicted_weeks": round(pred_days / 7, 1),
        "range_low_days" : round(max(pred_days - MAE_DAYS, 0), 1),
        "range_high_days": round(pred_days + MAE_DAYS, 1),
        "mae_days"       : round(MAE_DAYS, 1),
    }

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow calls from GitHub Pages (different origin)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "dog_los_xgb", "mae_days": MAE_DAYS})


@app.route("/predict", methods=["POST"])
def predict_route():
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400
    try:
        result = predict(body)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
