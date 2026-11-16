"""
Web server for the Credit Card Fraud Detection project.

Serves the custom frontend (templates/ + static/) and a small JSON API
backed by the trained Random Forest in fraud_detection_model.pkl.

Run locally:   python server.py           -> http://localhost:5000
On Render:     gunicorn server:app        (see render.yaml)
"""

import io
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "fraud_detection_model.pkl"
SAMPLE_LABELED = BASE_DIR / "sample_test_data.csv"
SAMPLE_UNLABELED = BASE_DIR / "sample_test_unlabeled.csv"
PLOTS = {"class_distribution.png", "confusion_matrix.png", "transaction_amount.png"}
MAX_ROWS_RETURNED = 2000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB upload limit

_saved = joblib.load(MODEL_FILE)
model = _saved["model"]
if hasattr(model, "n_jobs"):
    model.n_jobs = 1  # parallel threads only slow down small, real-time batches
feature_names = list(_saved["feature_names"])
_samples = pd.read_csv(SAMPLE_LABELED) if SAMPLE_LABELED.exists() else pd.DataFrame()

# ---- Dataset stats for the Model insights page (computed once) -----------
DATASET = BASE_DIR / "creditcard.csv"
_overview = {}
if DATASET.exists():
    _cls = pd.read_csv(DATASET, usecols=["Amount", "Class"])
    _fraud = _cls[_cls["Class"] == 1]
    _overview = {
        "rows": int(len(_cls)),
        "fraud": int(len(_fraud)),
        "fraud_rate": float(len(_fraud) / len(_cls)),
        "avg_fraud_amount": float(_fraud["Amount"].mean()),
    }
    del _cls, _fraud


def _score(frame: pd.DataFrame):
    X = frame[feature_names].astype(float)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def _risk_level(p: float) -> str:
    if p >= 0.75:
        return "critical"
    if p >= 0.5:
        return "high"
    if p >= 0.2:
        return "elevated"
    return "low"


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/plots/<name>")
def plots(name):
    if name not in PLOTS:
        abort(404)
    return send_from_directory(BASE_DIR, name)


@app.route("/download/sample.csv")
def download_sample():
    return send_from_directory(
        BASE_DIR, SAMPLE_UNLABELED.name, as_attachment=True,
        download_name="sample_transactions.csv",
    )


@app.route("/healthz")
def health():
    return {"ok": True}


# ---------------------------------------------------------------- API
@app.route("/api/meta")
def meta():
    importances = getattr(model, "feature_importances_", None)
    top = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:10]
        top = [{"feature": feature_names[i], "importance": float(importances[i])} for i in order]
    return jsonify({
        "algorithm": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "n_features": len(feature_names),
        "features": feature_names,
        "top_features": top,
    })


@app.route("/api/overview")
def overview():
    return jsonify(_overview)


@app.route("/api/sample")
def sample():
    """Return a random labelled transaction (kind = fraud | genuine | any) with its score."""
    if _samples.empty:
        abort(404)
    kind = request.args.get("kind", "any")
    pool = _samples
    if "Class" in pool.columns and kind in ("fraud", "genuine"):
        pool = pool[pool["Class"] == (1 if kind == "fraud" else 0)]
    row = pool.sample(1)
    pred, proba = _score(row)
    record = {k: float(v) for k, v in row[feature_names].iloc[0].items()}
    p = float(proba[0])
    return jsonify({
        "transaction": record,
        "actual": int(row["Class"].iloc[0]) if "Class" in row.columns else None,
        "prediction": int(pred[0]),
        "probability": p,
        "risk": _risk_level(p),
    })


@app.route("/api/predict", methods=["POST"])
def predict_one():
    payload = request.get_json(silent=True) or {}
    missing = [f for f in feature_names if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        row = pd.DataFrame([{f: float(payload[f]) for f in feature_names}])
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numbers."}), 400
    pred, proba = _score(row)
    p = float(proba[0])
    return jsonify({"prediction": int(pred[0]), "probability": p, "risk": _risk_level(p)})


@app.route("/api/predict-csv", methods=["POST"])
def predict_csv():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    try:
        data = pd.read_csv(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read that file as CSV."}), 400

    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {', '.join(missing)}"}), 400
    if data[feature_names].isnull().values.any():
        return jsonify({"error": "The file has empty values in required columns."}), 400

    pred, proba = _score(data)
    has_label = "Class" in data.columns
    total = int(len(data))
    fraud = int(pred.sum())

    summary = {
        "total": total,
        "fraud": fraud,
        "genuine": total - fraud,
        "fraud_amount": float(data.loc[pred == 1, "Amount"].sum()),
        "total_amount": float(data["Amount"].sum()),
        "risk_buckets": {
            lvl: int(sum(1 for p in proba if _risk_level(p) == lvl))
            for lvl in ("low", "elevated", "high", "critical")
        },
    }
    if has_label:
        y = data["Class"].astype(int).to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum())
        summary["labelled"] = {
            "accuracy": float((pred == y).mean()),
            "caught": tp,
            "actual_fraud": int(y.sum()),
            "false_alarms": int(((pred == 1) & (y == 0)).sum()),
        }

    # Most suspicious first, so the interesting rows survive the row cap.
    order = np.argsort(-proba)[:MAX_ROWS_RETURNED]
    rows = []
    for i in order:
        rows.append({
            "index": int(i) + 1,
            "time": float(data["Time"].iloc[i]),
            "amount": float(data["Amount"].iloc[i]),
            "probability": float(proba[i]),
            "prediction": int(pred[i]),
            "risk": _risk_level(float(proba[i])),
            "actual": int(data["Class"].iloc[i]) if has_label else None,
        })
    return jsonify({"summary": summary, "rows": rows, "truncated": total > MAX_ROWS_RETURNED})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File is too large (limit 40 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
