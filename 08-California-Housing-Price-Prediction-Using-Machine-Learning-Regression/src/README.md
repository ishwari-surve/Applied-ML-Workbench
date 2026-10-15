# 🏡 California Housing Price Prediction
### 💻 Source Code | Machine Learning Regression

Welcome to the **Source Code** folder of the California Housing Price Prediction project! This folder contains the Python implementation for predicting median house values in California using multiple machine learning regression algorithms.

---

## 📂 Source File

| File Name | Description |
|---|---|
| `main.py` | 🐍 Complete machine learning pipeline, including data loading, model training, evaluation, hyperparameter tuning, visualization, and prediction |

---

## 🏠 Project Workflow

The project follows a complete machine learning workflow:

1. 📥 **Dataset Loading** — Load and save the California Housing dataset.
2. 🔍 **Data Exploration** — Examine records, features, missing values, and statistical summaries.
3. 🧩 **Feature Engineering** — Separate input features and the target variable.
4. ✂️ **Train-Test Split** — Divide the dataset into 80% training and 20% testing data.
5. 🤖 **Model Initialization** — Initialize nine regression algorithms.
6. 🔄 **Cross-Validation** — Evaluate models using 5-fold cross-validation.
7. 🎯 **Model Training** — Train the algorithms on the training dataset.
8. 📊 **Performance Evaluation** — Calculate R² Score, RMSE, and MAE.
9. 🏆 **Model Comparison** — Compare and rank the regression models.
10. ⚙️ **Hyperparameter Tuning** — Optimize Random Forest using `GridSearchCV`.
11. 🏡 **House Value Prediction** — Predict median house values from input features.
12. 🔎 **Feature Importance** — Analyze which features contribute most to predictions.
13. 📈 **Data Visualization** — Generate model comparison, prediction, residual, and feature-importance plots.
14. 💾 **Save Outputs** — Save trained models and evaluation results.

---

## 🤖 Machine Learning Algorithms

The following nine regression algorithms are implemented:

- 📉 Linear Regression
- ⚖️ Ridge Regression
- 🧮 Lasso Regression
- 🎯 Support Vector Regression (SVR)
- 🌳 Decision Tree Regressor
- 🌲 Bagging Regressor
- 🌳 Random Forest Regressor
- 🚀 AdaBoost Regressor
- 📈 Gradient Boosting Regressor

---

## 📊 Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| 📈 R² Score | Measures how well the model explains variation in house values |
| 📉 RMSE | Measures prediction error, penalizing larger errors |
| 🎯 MAE | Measures the average absolute prediction error |

---

## 🛠️ Technologies Used

- 🐍 Python
- 🔢 NumPy
- 🐼 Pandas
- 📊 Matplotlib
- 🤖 Scikit-learn
- 💾 Joblib

---

## ▶️ How to Run the Project

**Step 1: Open the project root directory.**

**Step 2: Install the required libraries.**

```bash
pip install -r requirements.txt
```

**Step 3: Execute the main Python file.**

```bash
python src/main.py
```

---

## 📁 Generated Outputs

The program generates or saves files in the corresponding project folders:

- 🗃️ `data/` — California Housing dataset
- 📊 `results/` — Model comparison, feature importance, and prediction comparison CSV files
- 💾 `models/` — Saved regression models
- 📈 `images/` — Project visualizations, when saved separately

---

## 🎯 Project Objective

To develop a machine learning regression system that compares multiple algorithms, tunes a Random Forest model, and predicts median house values using housing, demographic, and geographical features.

---

## 🏡 Key Highlights

- ✅ Nine regression algorithms compared
- ✅ 5-fold cross-validation
- ✅ Multiple model evaluation metrics
- ✅ Random Forest hyperparameter tuning
- ✅ Feature importance analysis
- ✅ Prediction error analysis and visualizations
- ✅ Model and results export

---

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

*Case Study 08 — California Housing Price Prediction Using Machine Learning Regression*


