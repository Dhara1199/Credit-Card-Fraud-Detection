# Credit Card Fraud Detection Using Python and Machine Learning

## Project details
### Definition:
Credit Card Fraud Detection is a machine learning project that identifies fraudulent credit card transactions by analyzing transaction details and historical patterns. The system is trained using past transaction data and predicts whether a new transaction is genuine or fraudulent.

### Objectives:
1. Detect fraudulent credit card transactions.
2. Classify transactions as genuine or fraudulent.
3. Reduce financial losses caused by fraud.
4. Improve transaction security using machine learning..

## Software and libraries required
<img width="433" height="181" alt="image" src="https://github.com/user-attachments/assets/f863aa91-5925-4ca6-bbb0-2fcfc9b49829" />

## Algorithm
Random Forest Classifier (supervised classification).

## Dataset
Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the CSV file in the same folder as `main.py`.

The dataset includes Time, Amount, anonymized features V1-V28, and Class.
Class 0 means genuine; Class 1 means fraudulent.

## Installation
Install Python 3.9 or later, then run:

    pip install -r requirements.txt

## Execution

    python main.py

## How the project works
1. Load and inspect the transaction dataset.
2. Visualize the number of genuine and fraudulent transactions.
3. Separate the input columns (X) and target column (y).
4. Split data into training (80%) and testing (20%) sets.
5. Train a Random Forest classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score,
   ROC-AUC, PR-AUC, and a confusion matrix.
7. Save the trained model and show an example prediction.

<img width="791" height="535" alt="image" src="https://github.com/user-attachments/assets/436fc25f-9f5c-460d-92a2-e824265de996" />

## Model Evaluation
The project evaluates the model using:

<img width="497" height="161" alt="image" src="https://github.com/user-attachments/assets/595e7ec9-b4a2-40f0-999d-b1bff27ffbde" />

### Important: 
Fraud datasets are highly imbalanced. Therefore, recall, precision and PR-AUC are especially important. High accuracy alone does not mean the model detects fraud effectively.
