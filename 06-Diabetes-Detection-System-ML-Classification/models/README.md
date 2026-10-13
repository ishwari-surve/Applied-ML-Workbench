# 🤖 Trained Model – Diabetes Detection System

## 📌 Overview

This folder stores the trained Machine Learning model for the **Diabetes Detection System** project using Logistic Regression.

The model is trained using the preprocessed diabetes dataset and saved using Joblib so it can be loaded later without training the model again.

## 📂 File Description

| **File** | **Description** |
| -------- | --------------- |
| `diabetes_model.pkl` | Serialized Logistic Regression pipeline saved using Joblib. |

## 🛠️ Technology Used

- Python
- scikit-learn
- Joblib

## 💾 Model Saving

The trained model is saved using:

```python
joblib.dump(model, MODEL_PATH)
```

## 📥 Model Loading

The saved model is loaded using:

```python
model = joblib.load(MODEL_PATH)
```

## 🎯 Purpose

- Reuse the trained model without retraining it every time.
- Support diabetes classification predictions using the saved Logistic Regression pipeline.
- Preserve preprocessing steps, including median imputation and feature scaling, within the saved pipeline.
- Demonstrate model persistence in a Machine Learning workflow.

## 📊 Model Performance

Results from the current run:

- Cross-validation F1-score: 65.45%
- Test accuracy: 70.78%
- Recall for diabetes: 50.00%
- F1-score for diabetes: 54.55%

These results represent the current train-test split and are not a guarantee of future performance.

## ⚠️ Important Note

The `diabetes_model.pkl` file is generated when `main.py` runs successfully. The model file may be excluded from GitHub using `.gitignore` if you choose not to upload serialized model files.

Only load `.pkl` files from trusted sources because loading an untrusted pickle file can execute malicious code.

This project is intended for educational purposes only. The model must not be used as a substitute for professional medical diagnosis.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

GitHub: [https://github.com/ishwari-surve](https://github.com/ishwari-surve)

