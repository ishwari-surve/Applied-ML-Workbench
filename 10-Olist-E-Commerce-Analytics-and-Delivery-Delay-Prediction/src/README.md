# 💻 Source Code — Olist E-Commerce Analytics & Delivery Delay Prediction

## 📌 Overview

The `src` directory contains the Python source code for the **Olist E-Commerce Analytics and Delivery Delay Prediction** project.

The source code implements the data processing pipeline, exploratory data analysis, Recency–Frequency–Monetary (RFM) analysis, machine learning workflow, delivery-delay predictions, and business insight generation.

## 🎯 Objectives

- Load and understand the Olist e-commerce datasets.
- Clean and preprocess data for analysis.
- Merge related datasets to create a master dataset.
- Perform feature engineering and exploratory data analysis (EDA).
- Conduct customer RFM analysis.
- Build and compare machine learning classification models.
- Predict on-time and late deliveries.
- Evaluate model performance using classification metrics.
- Generate actionable business insights.
- Export results for further analysis and Power BI reporting.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Joblib | Saving and loading trained models |
| JSON | Storing structured business insights |

## 🧩 Main Workflow

1. **Data Loading:** Load the raw Olist datasets.
2. **Data Understanding:** Inspect dataset structure, columns, and data types.
3. **Data Quality Checks:** Identify missing values and duplicate records.
4. **Data Cleaning:** Prepare datasets for analysis.
5. **Data Integration:** Merge relevant datasets.
6. **Feature Engineering:** Create useful analytical and predictive features.
7. **Exploratory Data Analysis:** Examine order, customer, product, and delivery patterns.
8. **RFM Analysis:** Analyze customers using recency, frequency, and monetary value.
9. **Target Creation:** Create the `is_late` target variable.
10. **Leakage Prevention:** Remove columns that could reveal the actual delivery outcome.
11. **ML Data Preparation:** Prepare features and chronologically split training and test data.
12. **Model Building and Comparison:** Train and compare classification pipelines.
13. **Model Selection and Training:** Select the best model using cross-validation and train it on the training data.
14. **Model Persistence:** Save and reload the trained model.
15. **Model Evaluation:** Evaluate performance on unseen test data.
16. **Prediction and Visualization:** Generate delivery predictions and performance plots.
17. **Business Insights:** Summarize findings and recommendations.
18. **Output Generation:** Export model comparisons, predictions, evaluation metrics, and business insights.

## 🤖 Machine Learning Models

The project workflow includes the following models:

- Dummy Classifier — baseline comparison
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- HistGradientBoosting Classifier

Models are compared using Average Precision, Recall, and F1-score through time-series cross-validation.

## 📊 Evaluation Metrics

- **Average Precision:** Evaluates how effectively the model ranks late deliveries.
- **Precision:** Measures the proportion of predicted late deliveries that are actually late.
- **Recall:** Measures the proportion of actual late deliveries detected.
- **F1-score:** Balances precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect classifications.

## 📁 Directory Contents

The `src` directory contains the project's Python source file or files implementing the workflow.

The main Python script coordinates the pipeline, while individual functions handle data processing, analytics, model training, evaluation, prediction, and output generation.

> The exact filenames and module structure depend on the files included in this directory.

## 🚀 Execution

Run the main Python script from the project root after configuring the dataset paths and installing the required dependencies.

```bash
python src/main.py
```

If your main script is located elsewhere, update the command to match its actual path.

## 📦 Dependencies

Install the required Python libraries using the project's `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

## 📤 Outputs

Depending on the project configuration, the pipeline generates:

- Model comparison results
- Delivery prediction results
- Model evaluation metrics
- Confusion matrix visualization
- Precision–Recall curve
- Model performance chart
- Business insights in JSON format
- Saved trained model

These outputs can support further analysis and Power BI dashboard development.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

## 📝 Note

This directory contains the implementation of the project's analytical and machine learning workflow. Model performance depends on data quality, feature availability, chronological validation, and the evaluation results obtained during execution.
