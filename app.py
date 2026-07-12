"""
Bank Customer Churn Prediction — Flask Application
====================================================
Serves a scikit-learn RandomForestClassifier trained to predict whether a
bank customer is likely to churn (leave the bank), based on the classic
"Churn Modelling" dataset (Deep Learning A-Z).

Author: generated with Claude
"""

import os
import logging
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# --------------------------------------------------------------------------
# App & logging setup
# --------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model_random.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "Scaler_ANN.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "Encoders_ANN.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "Columns_ANN.pkl")

# --------------------------------------------------------------------------
# Load model + preprocessing artifacts once, at startup
# --------------------------------------------------------------------------
logger.info("Loading model and preprocessing artifacts...")

model = joblib.load(MODEL_PATH)             # RandomForestClassifier
scaler = joblib.load(SCALER_PATH)           # StandardScaler fitted on EstimatedSalary
encoders = joblib.load(ENCODERS_PATH)       # {"Geography": LabelEncoder, "Gender": LabelEncoder}
FEATURE_COLUMNS = joblib.load(COLUMNS_PATH)   # exact column order the model expects

logger.info("Feature order expected by the model: %s", FEATURE_COLUMNS)
logger.info("Model type: %s", type(model).__name__)
logger.info("Model + artifacts loaded successfully.")

# Values the dropdowns / toggles in the UI are allowed to send
GEOGRAPHY_OPTIONS = list(encoders["Geography"].classes_)
GENDER_OPTIONS = list(encoders["Gender"].classes_)

# Index of the "churned" class (1) inside model.classes_ / predict_proba output
CHURN_CLASS_INDEX = list(model.classes_).index(1)


# --------------------------------------------------------------------------
# Preprocessing helper
# --------------------------------------------------------------------------
def build_feature_vector(payload: dict) -> np.ndarray:
    """
    Turn a raw JSON payload from the form into the exact numeric vector
    the model expects, in the exact column order stored in Columns_ANN.pkl:

        ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
         'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    - Geography / Gender -> encoded with the saved LabelEncoders
    - EstimatedSalary     -> scaled with the saved StandardScaler
    - everything else     -> passed through as-is (this matches how the
      artifacts were fitted: the scaler was fit only on EstimatedSalary)
    """
    row = {
        "CreditScore": float(payload["CreditScore"]),
        "Geography": encoders["Geography"].transform([payload["Geography"]])[0],
        "Gender": encoders["Gender"].transform([payload["Gender"]])[0],
        "Age": float(payload["Age"]),
        "Tenure": float(payload["Tenure"]),
        "Balance": float(payload["Balance"]),
        "NumOfProducts": float(payload["NumOfProducts"]),
        "HasCrCard": float(payload["HasCrCard"]),
        "IsActiveMember": float(payload["IsActiveMember"]),
        "EstimatedSalary": scaler.transform([[float(payload["EstimatedSalary"])]])[0][0],
    }

    ordered_values = [row[col] for col in FEATURE_COLUMNS]
    return np.array(ordered_values, dtype="float64").reshape(1, -1)


def validate_payload(payload: dict):
    """Basic server-side validation. Returns an error message or None."""
    required = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ]
    missing = [f for f in required if f not in payload or payload[f] in ("", None)]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"

    if payload["Geography"] not in GEOGRAPHY_OPTIONS:
        return f"Geography must be one of {GEOGRAPHY_OPTIONS}"
    if payload["Gender"] not in GENDER_OPTIONS:
        return f"Gender must be one of {GENDER_OPTIONS}"

    try:
        credit_score = float(payload["CreditScore"])
        age = float(payload["Age"])
        tenure = float(payload["Tenure"])
        balance = float(payload["Balance"])
        num_products = float(payload["NumOfProducts"])
        salary = float(payload["EstimatedSalary"])
    except (TypeError, ValueError):
        return "Numeric fields must contain valid numbers."

    if not (300 <= credit_score <= 900):
        return "CreditScore should be between 300 and 900."
    if not (18 <= age <= 100):
        return "Age should be between 18 and 100."
    if not (0 <= tenure <= 15):
        return "Tenure should be between 0 and 15."
    if balance < 0:
        return "Balance cannot be negative."
    if not (1 <= num_products <= 4):
        return "NumOfProducts should be between 1 and 4."
    if salary < 0:
        return "EstimatedSalary cannot be negative."

    if str(payload["HasCrCard"]) not in ("0", "1"):
        return "HasCrCard must be 0 or 1."
    if str(payload["IsActiveMember"]) not in ("0", "1"):
        return "IsActiveMember must be 0 or 1."

    return None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        geography_options=GEOGRAPHY_OPTIONS,
        gender_options=GENDER_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()

    error = validate_payload(payload)
    if error:
        return jsonify({"ok": False, "error": error}), 400

    try:
        features = build_feature_vector(payload)
        probability = float(model.predict_proba(features)[0][CHURN_CLASS_INDEX])
        will_churn = probability >= 0.5

        response = {
            "ok": True,
            "churn_probability": round(probability * 100, 2),
            "stay_probability": round((1 - probability) * 100, 2),
            "will_churn": will_churn,
            "risk_level": (
                "High" if probability >= 0.66 else
                "Medium" if probability >= 0.33 else
                "Low"
            ),
        }
        return jsonify(response)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"ok": False, "error": f"Prediction failed: {exc}"}), 500


@app.route("/health")
def health():
    """Simple health-check endpoint for deployment platforms."""
    return jsonify({"status": "ok"})


# --------------------------------------------------------------------------
# Entry point (used only for local `python app.py` runs; production uses
# gunicorn via the Procfile — see README.md)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
