# 📊 Visualizations — Credit Card Fraud Detection

## 📌 Overview

This directory contains the visualizations generated during Exploratory Data Analysis (EDA), model evaluation, and feature analysis for the **Credit Card Fraud Detection using Machine Learning Classification** project.

These plots help understand the dataset, analyze class imbalance, evaluate the trained model, and interpret important features.

## 📂 Images Description

### 1. `class_distribution.png`

Displays the distribution of legitimate and fraudulent transactions.

**Purpose:**
- Identify class imbalance in the dataset.
- Understand the difference in the number of transactions belonging to each class.

### 2. `transaction_amount_distribution.png`

Displays the distribution of transaction amounts for legitimate and fraudulent transactions.

**Purpose:**
- Examine transaction amount patterns.
- Explore differences between the two transaction classes.

### 3. `transaction_time_distribution.png`

Visualizes transaction frequency across elapsed time, separated by transaction class.

**Purpose:**
- Explore transaction activity over time.
- Observe the distribution of legitimate and fraudulent transactions across the recorded time period.

### 4. `confusion_matrix.png`

Displays the confusion matrix for the evaluated fraud detection model.

The plotted matrix shows:

| Actual Class | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Legitimate | 56,636 | 15 |
| Fraud | 23 | 72 |

**Purpose:**
- Identify correctly classified transactions.
- Examine false positives and false negatives.
- Understand the model's fraud detection performance.

### 5. `precision_recall_curve.png`

Displays the Precision-Recall curve for the evaluated model.

**Reported AUPRC:** `0.6882`

**Purpose:**
- Visualize the relationship between precision and recall at different classification thresholds.
- Evaluate model performance on an imbalanced classification problem.

### 6. `feature_importance.png`

Displays the top 10 features ranked by the absolute values of their Logistic Regression coefficients.

**Purpose:**
- Identify features with the largest coefficient magnitudes.
- Support model interpretation and analysis.

**Note:** Absolute coefficient magnitude does not indicate whether a feature increases or decreases the predicted fraud probability.

## 📁 Folder Structure

```text
images/
├── README.md
├── class_distribution.png
├── transaction_amount_distribution.png
├── transaction_time_distribution.png
├── confusion_matrix.png
├── precision_recall_curve.png
└── feature_importance.png
```

## 🛠️ Technologies Used

- Python
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

## 🎯 Purpose

The visualizations support:

- Exploratory Data Analysis.
- Class imbalance investigation.
- Transaction pattern analysis.
- Model evaluation.
- Feature interpretation.

---

**Project:** Credit Card Fraud Detection using Machine Learning Classification  
**Folder:** `images/`  
**Purpose:** Visual representation of dataset analysis and model results.

