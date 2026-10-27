import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Make the project root importable so "src" can be found
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.features import add_features

# Load the saved model and info ONCE, when the app starts
model = joblib.load(ROOT / "models" / "rf_pipeline.joblib")
with open(ROOT / "models" / "model_info.json") as f:
    info = json.load(f)

THRESHOLD = info["threshold"]
FEATURES = info["feature_columns"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # reject uploads over 2 MB

# Allowed ranges for validation (sanity limits, adjust if needed)
LIMITS = {
    "Air temperature [K]": (250, 350),
    "Process temperature [K]": (250, 400),
    "Rotational speed [rpm]": (500, 5000),
    "Torque [Nm]": (0, 150),
    "Tool wear [min]": (0, 500),
}
INPUT_COLS = ["Type"] + list(LIMITS)

# HTML form field name -> dataset column name
FORM_FIELDS = {
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


def risk_category(p):
    """Low below the chosen threshold, High from 0.5, Medium in between."""
    if p < THRESHOLD:
        return "Low"
    if p < 0.5:
        return "Medium"
    return "High"


def predict_proba(df):
    """Add derived features, order the columns like training, return failure probabilities."""
    X = add_features(df)[FEATURES]
    return model.predict_proba(X)[:, 1]


@app.route("/")
def home():
    return render_template("home.html", info=info)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", error=None, result=None)

    row = {"Type": request.form.get("type", "")}
    if row["Type"] not in ("L", "M", "H"):
        return render_template("predict.html", error="Please choose a product type.", result=None)

    for field, col in FORM_FIELDS.items():
        raw = request.form.get(field, "").strip()
        try:
            value = float(raw)
        except ValueError:
            return render_template("predict.html", error=f"{col} must be a number.", result=None)
        low, high = LIMITS[col]
        if not (low <= value <= high):
            return render_template(
                "predict.html", error=f"{col} must be between {low} and {high}.", result=None
            )
        row[col] = value

    p = float(predict_proba(pd.DataFrame([row]))[0])
    result = {
        "probability": round(p * 100, 1),
        "failure": p >= THRESHOLD,
        "risk": risk_category(p),
    }
    return render_template("predict.html", error=None, result=result)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", error=None)

    def fail(message):
        return render_template("batch.html", error=message)

    file = request.files.get("file")
    if file is None or file.filename == "":
        return fail("Please choose a CSV file.")
    if not file.filename.lower().endswith(".csv"):
        return fail("Only .csv files are allowed.")

    try:
        df = pd.read_csv(file)
    except Exception:
        return fail("Could not read the file. Please upload a valid CSV.")

    if df.empty:
        return fail("The file has no rows.")
    if len(df) > 5000:
        return fail("Please upload at most 5000 rows at a time.")

    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        return fail("Missing required columns: " + ", ".join(missing))

    df = df[INPUT_COLS].copy()
    if df.isnull().any().any():
        return fail("The file has empty cells in the required columns.")

    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    if not df["Type"].isin(["L", "M", "H"]).all():
        return fail("Type must be L, M or H in every row.")

    for col, (low, high) in LIMITS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            return fail(f"{col} must contain only numbers.")
        if not df[col].between(low, high).all():
            return fail(f"{col} has values outside the allowed range {low} to {high}.")

    proba = predict_proba(df)
    out = df.copy()
    out["Failure probability (%)"] = (proba * 100).round(1)
    out["Predicted"] = np.where(proba >= THRESHOLD, "Failure likely", "No failure expected")
    out["Risk"] = [risk_category(p) for p in proba]

    counts = out["Risk"].value_counts().to_dict()
    return render_template(
        "batch_results.html",
        columns=list(out.columns),
        rows=out.to_dict(orient="records"),
        total=len(out),
        low=counts.get("Low", 0),
        medium=counts.get("Medium", 0),
        high=counts.get("High", 0),
    )


if __name__ == "__main__":
    app.run(debug=True)