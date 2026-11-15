"""
Credit Card Fraud Detection - Prediction Script using joblib
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

MODEL_FILE = Path("fraud_detection_model.pkl")
DEFAULT_DATA_FILE = Path("sample_test_data.csv")


def predict_csv(file_path: Path):
    # 1. Load the trained model and feature names using joblib
    if not MODEL_FILE.exists():
        print(f"Error: Model file '{MODEL_FILE}' not found. Run main.py first.")
        return

    print(f"Loading model from '{MODEL_FILE}' using joblib...")
    saved_data = joblib.load(MODEL_FILE)
    model = saved_data["model"]
    feature_names = saved_data["feature_names"]

    # 2. Load dataset
    if not file_path.exists():
        print(f"Error: Dataset '{file_path}' not found.")
        return

    print(f"Loading test dataset from '{file_path}'...")
    data = pd.read_csv(file_path)

    # 3. Check features
    missing_cols = [col for col in feature_names if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing required feature columns: {missing_cols}")
        return

    X = data[feature_names]

    # 4. Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # 5. Format results
    results = data.copy()
    results["Prediction"] = predictions
    results["Status"] = results["Prediction"].map({0: "GENUINE", 1: "FRAUD"})
    results["Fraud_Probability"] = probabilities.round(4)

    # Display summary
    total = len(results)
    fraud_count = int((predictions == 1).sum())
    genuine_count = total - fraud_count

    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total Transactions Scanned: {total}")
    print(f"Genuine Transactions:       {genuine_count}")
    print(f"Fraudulent Transactions:    {fraud_count}")
    print("=" * 50)

    # Display preview table
    cols_to_show = ["Time", "Amount"]
    if "Class" in results.columns:
        cols_to_show.append("Class")
    cols_to_show.extend(["Status", "Fraud_Probability"])

    print("\nDetailed Predictions:")
    print(results[cols_to_show].to_string(index=False))

    # Save output
    output_path = Path("predictions_output.csv")
    results.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to '{output_path}'")


if __name__ == "__main__":
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_FILE
    predict_csv(target_file)
