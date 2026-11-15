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

| Software / Library | Purpose |
| :--- | :--- |
| Python | Programming |
| Jupyter Notebook / VS Code | Code execution |
| Pandas | Dataset handling |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization |
| Seaborn | Graphs |
| Scikit-learn | Machine learning |
| Joblib | Saving trained model |

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

<img width="784" height="542" alt="image" src="https://github.com/user-attachments/assets/546bd5c5-b7ce-4153-a9d3-470fbba71341" />

## Model Evaluation
The project evaluates the model using:

| Metric | Purpose |
| :--- | :--- |
| Accuracy | Overall correct predictions |
| Precision | How many predicted fraud cases are actually fraud |
| Recall | How many actual fraud cases are detected |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Correct and incorrect predictions |
| ROC-AUC | Ability to distinguish two classes |
| PR-AUC | Performance on imbalanced data |

### Important: 
Fraud datasets are highly imbalanced. Therefore, recall, precision and PR-AUC are especially important. High accuracy alone does not mean the model detects fraud effectively.
