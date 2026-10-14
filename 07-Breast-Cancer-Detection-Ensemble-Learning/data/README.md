# 📊 Dataset — Breast Cancer Detection Using Ensemble Learning

## 📌 Overview

This directory contains the dataset used for the **Breast Cancer Detection Using Ensemble Learning** project.

The dataset is used to train and evaluate machine learning models that classify breast cancer cases as malignant or benign.

## 📁 Dataset File

| **File Name** | **Description** |
|---|---|
| `breast_cancer.csv` | CSV dataset containing input features and target labels for breast cancer classification. |

## 🎯 Purpose

The dataset supports the following tasks:

- Exploratory data analysis
- Data quality checking and statistical analysis
- Feature and target separation
- Training and testing machine learning models
- Cross-validation and model comparison
- Evaluation of malignant and benign classifications

## 🧾 Dataset Requirements

The CSV file should contain:

- **Feature columns:** Numerical measurements used by the classification models.
- **Target column:** A column named `target` representing the class labels.
- **Two classes:** Malignant and benign.

### Target Classification

| **Target Value** | **Diagnosis** |
|---|---|
| `0` | Malignant |
| `1` | Benign |

## ⚙️ Usage

The dataset is loaded by the Python implementation:

`src/main.py`

Place `breast_cancer.csv` inside the `data` directory. The program expects the following path:

`data/breast_cancer.csv`

## ⚠️ Important Notes

- Verify the target column and class labels before running the program.
- The dataset should contain sufficient records from both classes for stratified 5-fold cross-validation.
- The program displays missing values and duplicate records for data quality analysis.
- The dataset is intended for educational and machine learning experimentation.
- Predictions from this project must not be used as a substitute for professional medical diagnosis.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**
