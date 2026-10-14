# 🤖 Trained Models — Breast Cancer Detection Using Ensemble Learning

## 📌 Overview

This directory stores the trained machine learning model selected by the **Breast Cancer Detection Using Ensemble Learning** project.

The model is selected using stratified 5-fold cross-validation, based on the highest F1-score for the malignant class among the ensemble models.

## 📁 Model File

| **File Name** | **Description** |
|---|---|
| `breast_cancer_model.pkl` | Serialized trained ensemble classifier saved for future use. |

## 🧠 Algorithms Evaluated

The project compares the following ensemble classifiers:

1. Bagging Classifier
2. Random Forest
3. AdaBoost
4. Gradient Boosting
5. Hard Voting Classifier
6. Soft Voting Classifier

## ⚙️ Model Selection

The selection process uses cross-validation F1-score for the malignant class to choose the best ensemble model.

The model is selected using training data, while the held-out test set is reserved for final evaluation.

## 💾 Model Saving

The trained model is saved using the **Joblib** library.

Output path:

`models/breast_cancer_model.pkl`

## 🔄 Loading the Saved Model

Use the following Python example to load the saved model:

```python
import joblib

model = joblib.load("models/breast_cancer_model.pkl")
```

**Note:** This model is intended for educational purposes only and is not a clinically validated diagnostic system.
