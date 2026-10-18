# 📊 Execution Outputs — Credit Card Fraud Detection

## 📌 Overview

This directory contains screenshots of the Command Prompt outputs generated during the execution of the **Credit Card Fraud Detection using Machine Learning Classification** project.

These screenshots document the complete workflow, including dataset inspection, data cleaning, exploratory data analysis, feature preparation, model comparison, hyperparameter tuning, model evaluation, error analysis, and successful project execution.

## 📂 Files Description

| No. | File Name | Description |
|---|---|---|
| 01 | `01_dataset_inspection.png` | Displays the dataset shape, first five records, and column names. |
| 02 | `02_data_types_missing_values.png` | Shows data types and missing-value information for the dataset. |
| 03 | `03_data_cleaning_eda.png` | Documents duplicate removal, dataset validation, and class distribution. |
| 04 | `04_feature_preparation_and_model_creation.png` | Shows feature preparation, stratified train-test splitting, preprocessing, and model creation. |
| 05 | `05_model_comparison_and_tuning.png` | Displays five-fold cross-validation model comparison and Logistic Regression hyperparameter tuning. |
| 06 | `06_cross_validation_imbalance_threshold_evaluation.png` | Shows cross-validation results, class imbalance technique comparison, threshold tuning, and evaluation metrics. |
| 07 | `07_error_analysis_and_model_explainability.png` | Documents false-positive and false-negative analysis, important features, and trained-model saving. |
| 08 | `08_final_evaluation_and_project_summary.png` | Shows the saved evaluation files, final performance metrics, and successful completion message. |

## 🔍 Key Execution Highlights

- **Original dataset:** 284,807 transactions and 31 columns.
- **Duplicate records removed:** 1,081.
- **Dataset after cleaning:** 283,726 transactions.
- **Input features:** 30.
- **Training records:** 226,980.
- **Testing records:** 56,746.
- **Selected model:** Tuned Logistic Regression.
- **Classification threshold:** 0.0685.

## 📈 Final Model Evaluation

The final evaluation displayed by the project execution is:

| Metric | Value |
|---|---:|
| AUPRC | 0.6882 |
| Precision | 0.8276 |
| Recall | 0.7579 |
| F1-Score | 0.7912 |

These values correspond to the final evaluation reported by the project script. AUPRC measures performance across precision-recall trade-offs, while precision, recall, and F1-score summarize classification performance at the selected threshold.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## 🎯 Purpose

These screenshots provide visual evidence of the project's execution and help document the steps performed during the machine learning workflow.

**Note:** The screenshots are supplementary documentation. The actual graphs are stored in the `images/` directory, and the generated CSV and JSON evaluation files are stored in the `results/` directory.


