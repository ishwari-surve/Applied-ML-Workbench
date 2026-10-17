# 💳 Credit Card Fraud Detection Using Machine Learning Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine-Learning-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-blue?logo=scikitlearn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green" alt="Imbalanced Learn">
  <img src="https://img.shields.io/badge/Project-09-purple" alt="Project 09">
</p>

## 📌 Project Overview

Credit card fraud is a major challenge in digital payments and financial security. Fraudulent transactions can cause financial losses for customers and financial institutions.

This project develops a **Machine Learning-based Credit Card Fraud Detection System** that classifies transactions as:

- ✅ **Legitimate Transaction (Class 0)**
- 🚨 **Fraudulent Transaction (Class 1)**

The project explores multiple classification algorithms, addresses class imbalance, tunes Logistic Regression hyperparameters, and optimizes the classification threshold to improve fraud detection performance.

## 🎯 Objectives

- Build a machine learning system for credit card fraud detection.
- Perform exploratory data analysis (EDA).
- Clean and validate the transaction dataset.
- Compare multiple classification algorithms.
- Handle imbalanced transaction classes.
- Perform stratified cross-validation and hyperparameter tuning.
- Optimize the classification threshold using training data.
- Evaluate model performance using appropriate metrics.
- Analyze classification errors and important features.
- Save the trained model and evaluation results.

## 🗂️ Project Structure

```text
09-Credit-Card-Fraud-Detection-Classification/
│
├── data/
│   └── creditcard.csv
│
├── src/
│   └── main.py
│
├── models/
│   └── credit_card_fraud_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── final_model_metrics.csv
│   ├── logistic_regression_tuning.csv
│   ├── classification_report.csv
│   └── model_metrics.json
│
├── images/
│   └── README.md
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

*Note: The `models/`, `results/`, and `images/` directories and their files are generated or maintained separately. Some may appear only after execution or after you add them.*

## 📊 Dataset Information

The project uses the **Credit Card Fraud Detection dataset**, containing anonymized transaction features and a binary target.

| Property | Description |
|---|---|
| Dataset file | `data/creditcard.csv` |
| Original dataset size | 284,807 transactions |
| Input features | Time, V1–V28, Amount |
| Target column | Class |
| Class 0 | Legitimate transaction |
| Class 1 | Fraudulent transaction |
| Problem type | Binary classification |

### Feature Description

- **Time:** Seconds elapsed between a transaction and the first transaction in the dataset.
- **V1–V28:** Anonymized numerical features obtained through Principal Component Analysis (PCA).
- **Amount:** Transaction amount.
- **Class:** Target label indicating whether a transaction is legitimate or fraudulent.

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of all transactions.

## 🧠 Machine Learning Algorithms

The project compares the following classifiers:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Baseline and hyperparameter-tuned classification |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble classification using multiple trees |
| Gradient Boosting | Sequential ensemble learning |
| Linear Support Vector Machine | Classification using a separating decision boundary |

The final evaluation focuses on the **tuned Logistic Regression model**.

## ⚙️ Methodology

1. **Dataset Inspection:** Examine dimensions, data types, missing values, duplicates, and descriptive statistics.
2. **Data Cleaning:** Remove duplicate records and validate missing values.
3. **Exploratory Data Analysis:** Visualize class distribution, transaction amounts, and transaction times.
4. **Feature and Target Separation:** Separate the input features from the `Class` target.
5. **Train-Test Split:** Use a stratified 80:20 split.
6. **Preprocessing:** Apply `RobustScaler` to `Amount` and `Time`.
7. **Model Comparison:** Compare additional classifiers using stratified five-fold cross-validation.
8. **Class Imbalance Handling:** Compare class weighting, SMOTE, and random undersampling.
9. **Hyperparameter Tuning:** Tune Logistic Regression using `RandomizedSearchCV` and Average Precision scoring.
10. **Threshold Optimization:** Select a classification threshold using out-of-fold predictions on training data.
11. **Final Evaluation:** Evaluate the selected model on the held-out test set.
12. **Error Analysis and Explainability:** Analyze false positives, false negatives, and Logistic Regression coefficients.
13. **Model and Results Saving:** Save the trained pipeline, threshold, evaluation metrics, tuning results, and classification report.

## 🛠️ Technologies Used

- **Python 3.10**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Exploratory data analysis
- **Scikit-learn** — Machine learning, preprocessing, cross-validation, and evaluation
- **Imbalanced-learn** — SMOTE and random undersampling
- **Joblib** — Model persistence

## 📈 Evaluation Metrics

Because fraud detection involves imbalanced classes, accuracy alone is not sufficient.

- **Average Precision (AUPRC):** Evaluates precision-recall performance across different thresholds.
- **Precision:** Measures how many transactions predicted as fraudulent are actually fraudulent.
- **Recall:** Measures how many actual fraudulent transactions are detected.
- **F1-Score:** Balances precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.

The project also generates a precision-recall curve and a classification report.

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd 09-Credit-Card-Fraud-Detection-Classification
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the dataset location

Ensure the dataset exists at:

```text
data/creditcard.csv
```

### 5. Run the project

From the **main project directory**, execute:

```bash
python src/main.py
```

**Important:** Run the command from the main project directory because the program uses the relative dataset path `data/creditcard.csv`.

The program performs data analysis, cross-validation, model tuning, threshold optimization, final evaluation, and result saving. Execution may take some time because the dataset is large and multiple cross-validation procedures are performed.

## 📁 Output Files

After successful execution, the project is designed to generate:

| File | Purpose |
|---|---|
| `models/credit_card_fraud_model.pkl` | Saved model pipeline and tuned threshold |
| `results/model_comparison.csv` | Cross-validation comparison of classifiers |
| `results/final_model_metrics.csv` | Final test-set metrics |
| `results/logistic_regression_tuning.csv` | Hyperparameter search results |
| `results/classification_report.csv` | Precision, recall, F1-score, and support |
| `results/model_metrics.json` | Final evaluation metrics in JSON format |

## 🔍 Expected Learning Outcomes

Through this project, I aim to demonstrate practical knowledge of:

- Supervised machine learning and binary classification.
- Handling highly imbalanced datasets.
- Data preprocessing and feature scaling.
- Model evaluation using Average Precision and precision-recall analysis.
- Cross-validation and hyperparameter optimization.
- Classification threshold selection.
- Model interpretability and error analysis.
- Saving and loading trained machine learning pipelines.

## 🔮 Future Enhancements

- Develop a real-time transaction prediction interface.
- Add a Streamlit or Flask web application.
- Integrate an API for transaction scoring.
- Explore advanced anomaly detection methods.
- Monitor model performance and data drift.
- Evaluate the system on new, unseen transaction data.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

Machine Learning | Python | Data Analytics

GitHub: [ishwari-surve](https://github.com/ishwari-surve)

---

⭐ If you find this project interesting, consider starring the repository!

*Disclaimer: This project is intended for educational purposes. Its evaluation results should be verified after execution; no specific fraud detection performance is claimed in advance.*

