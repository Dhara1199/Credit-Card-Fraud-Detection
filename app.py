import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Detect fraudulent transactions using Machine Learning.")

# Load trained model
@st.cache_resource
def load_model():
    saved = joblib.load("fraud_detection_model.pkl")
    return saved["model"], saved["feature_names"]

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model not found. Please run main.py first.")
    st.stop()

st.subheader("Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Remove target column if present
    X = data.drop(columns=["Class"], errors="ignore")

    missing = [c for c in feature_names if c not in X.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")

    elif st.button("Detect Fraud", type="primary"):

        X = X[feature_names]

        if X.isnull().values.any():
            st.error("Please remove missing values before prediction.")
            st.stop()

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = data.copy()

        results["Prediction"] = predictions
        results["Fraud Probability"] = probabilities

        results["Status"] = results["Prediction"].map({
            0: "Genuine",
            1: "Fraud"
        })

        total = len(results)
        fraud = int((predictions == 1).sum())
        genuine = total - fraud

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Genuine Transactions", genuine)
        col3.metric("Fraud Transactions", fraud)

        st.subheader("Prediction Results")

        st.dataframe(results, use_container_width=True)

        st.subheader("Fraud Detection Chart")

        chart = pd.DataFrame({
            "Transaction": ["Genuine", "Fraud"],
            "Count": [genuine, fraud]
        })

        st.bar_chart(
            chart.set_index("Transaction")
        )

        # Download report
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Report",
            csv,
            "fraud_detection_results.csv",
            "text/csv"
        )

        st.success("Fraud detection completed!")

st.divider()
st.caption("Credit Card Fraud Detection | Random Forest ML Model")
