# 🤖 Trained Model – Titanic Survival Prediction

## 📌 Overview

This folder stores the trained Machine Learning model for the **Titanic Survival Prediction** project using Logistic Regression.

The model is trained using the preprocessed Titanic dataset and saved using Joblib so it can be loaded later without training the model again.

## 📂 File Description

| File | Description |
|---|---|
| `titanic_model.pkl` | Serialized Logistic Regression model saved using Joblib. |

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
- Support predictions using the saved Logistic Regression model.
- Demonstrate model persistence in a Machine Learning workflow.

## ⚠️ Important Note

The `titanic_model.pkl` file is generated when `main.py` runs successfully. The model file may be excluded from GitHub using `.gitignore` if you choose not to upload serialized model files.

Only load `.pkl` files from trusted sources because loading an untrusted pickle file can execute malicious code.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

GitHub: https://github.com/ishwari-surve

