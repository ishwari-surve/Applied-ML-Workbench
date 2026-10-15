# 🏡 California Housing Price Prediction
### 📊 Results & Performance Analysis

Welcome to the **Results** folder of the California Housing Price Prediction project! This folder stores the evaluation results, prediction comparisons, and feature importance data generated during the machine learning workflow.

---

## 📁 Files in This Folder

### 1. 🏆 `model_comparison.csv`

Contains the performance evaluation results of the nine regression algorithms.

**Includes:**
- Model name
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

**Purpose:** To compare and rank regression models based on their performance on the test dataset.

### 2. 🌳 `feature_importance.csv`

Contains the feature importance scores calculated by the tuned Random Forest Regressor.

**Includes:**
- Feature name
- Importance score

**Purpose:** To identify which input features contribute most to the model's predictions.

### 3. 🎯 `prediction_comparison.csv`

Contains actual and predicted house values for the test dataset.

**Includes:**
- Actual house value
- Predicted house value
- Absolute prediction error

**Purpose:** To analyze prediction accuracy and understand the difference between actual and predicted values.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| 📈 R² Score | Measures how well the model explains variation in house values. Higher is generally better. |
| 📉 RMSE | Measures prediction error and penalizes larger errors. Lower is better. |
| 🎯 MAE | Measures the average absolute prediction error. Lower is better. |

---

## 🏡 How Results Are Generated

The results are produced by `src/main.py` through the following workflow:

1. 🤖 Train multiple regression models.
2. 🔄 Evaluate models using 5-fold cross-validation.
3. 📊 Calculate test-set performance metrics.
4. 🏆 Compare the regression algorithms.
5. ⚙️ Tune the Random Forest Regressor using `GridSearchCV`.
6. 🌳 Calculate feature importance scores.
7. 🎯 Generate actual-versus-predicted values and absolute errors.
8. 💾 Export the results to CSV files.

---

## 📂 Folder Structure

```text
results/
├── README.md
├── model_comparison.csv
├── feature_importance.csv
└── prediction_comparison.csv
```

*Note: The CSV files are generated when the corresponding saving functions in `src/main.py` run successfully.*

---

## 💡 Why This Folder Matters

- 📈 Makes model performance comparisons easy.
- 🏆 Helps identify the best-performing regression algorithm.
- 🌳 Provides insight into feature importance.
- 🎯 Supports detailed prediction error analysis.
- 💾 Preserves results for future analysis and reporting.

---

## 🛠️ Technologies Used

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn

---

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

*Case Study 08 — California Housing Price Prediction Using Machine Learning Regression*


