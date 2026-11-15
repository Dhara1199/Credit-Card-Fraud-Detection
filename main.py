"""
Credit Card Fraud Detection using Python and Machine Learning

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

How to run:
1. Put creditcard.csv in this folder.
2. Install packages: pip install -r requirements.txt
3. Run: python main.py

The target column is Class:
0 = genuine transaction
1 = fraudulent transaction
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

DATA_FILE = Path("creditcard.csv")
MODEL_FILE = Path("fraud_detection_model.pkl")


def load_dataset():
    """Read the CSV dataset and perform basic checks."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Download it from Kaggle and "
            "place it in the same folder as main.py."
        )

    data = pd.read_csv(DATA_FILE)

    if "Class" not in data.columns:
        raise ValueError("The dataset must contain a 'Class' column.")

    print("\nFirst five rows:")
    print(data.head())

    print("\nDataset shape:", data.shape)
    print("\nMissing values:", int(data.isnull().sum().sum()))
    print("\nClass counts (0=genuine, 1=fraud):")
    print(data["Class"].value_counts())

    return data


def visualize_data(data):
    """Create and save class distribution and amount plots."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=data)
    plt.title("Genuine vs Fraudulent Transactions")
    plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(x="Class", y="Amount", data=data, showfliers=False)
    plt.title("Transaction Amount by Class")
    plt.tight_layout()
    plt.savefig("transaction_amount.png")
    plt.close()


def train_model(data):
    """Split the data, train a Random Forest, and evaluate it."""
    X = data.drop("Class", axis=1)
    y = data["Class"].astype(int)

    # Stratification keeps the fraud/non-fraud ratio similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    print("\nTraining the Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Genuine", "Fraud"],
            zero_division=0,
        )
    )
    print("ROC-AUC: {:.4f}".format(roc_auc_score(y_test, probabilities)))
    print(
        "PR-AUC (Average Precision): {:.4f}".format(
            average_precision_score(y_test, probabilities)
        )
    )

    # Confusion matrix
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save model and its feature names together.
    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        MODEL_FILE,
    )
    print(f"\nModel saved as {MODEL_FILE}")

    return model, X_test


def predict_example(model, X_test):
    """Show one example prediction from the held-out test set."""
    sample = X_test.iloc[[0]]
    fraud_probability = model.predict_proba(sample)[0][1]
    predicted_class = int(fraud_probability >= 0.5)

    print("\n--- Example Prediction ---")
    print("Predicted class:", predicted_class)
    print("Result:", "FRAUD" if predicted_class == 1 else "GENUINE")
    print("Estimated fraud probability: {:.2%}".format(fraud_probability))


def main():
    data = load_dataset()
    visualize_data(data)
    model, X_test = train_model(data)
    predict_example(model, X_test)

    print("\nProject completed successfully.")
    print("Generated files: class_distribution.png, transaction_amount.png,")
    print("                 confusion_matrix.png, fraud_detection_model.pkl")


if __name__ == "__main__":
    main()
