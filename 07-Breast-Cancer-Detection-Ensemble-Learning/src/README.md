# 🩺 Breast Cancer Detection Using Ensemble Learning

## 📌 Overview

The `src` folder contains the Python source code for classifying breast cancer cases as malignant or benign using machine learning and ensemble learning algorithms.

The program compares 10 classification models using stratified 5-fold cross-validation, selects the best ensemble based on malignant-class F1-score, evaluates its performance, and generates visualizations.

## 🤖 Algorithms Implemented

- Decision Tree
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Bagging Classifier
- Random Forest
- AdaBoost
- Gradient Boosting
- Hard Voting Classifier
- Soft Voting Classifier

## 🛠️ Technologies Used

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib

## 📂 Source File

| File | Description |
|---|---|
| `main.py` | Implements dataset loading, model creation, cross-validation, model selection, evaluation, visualization, and predictions. |

## ⚙️ Workflow

1. Load and analyze the breast cancer dataset.
2. Split the dataset into training and testing sets.
3. Build and compare 10 classification models.
4. Perform stratified 5-fold cross-validation.
5. Select the best ensemble using malignant-class F1-score.
6. Evaluate the selected model on test data.
7. Generate model comparison, confusion matrix, and feature importance graphs.
8. Save the trained model and display sample predictions.

## 📊 Evaluation Metrics

- Accuracy
- Malignant Recall
- Malignant F1-Score
- Confusion Matrix
- Classification Report

## 📁 Generated Outputs

| File | Description |
|---|---|
| `models/breast_cancer_model.pkl` | Saved trained model. |
| `images/model_comparison.png` | Cross-validation model comparison. |
| `images/confusion_matrix.png` | Confusion matrix of the selected model. |
| `images/feature_importance.png` | Top 10 feature importances from Random Forest. |

## ▶️ Execution

Run the following command from the project root directory:

```bash
python src/main.py
```

**Note:** The dataset must be available at `data/breast_cancer.csv`, with `target` encoded as `0 = malignant` and `1 = benign`.

This project is for educational purposes only and is not intended for clinical diagnosis.

## Author
**Ishwari Surve**
