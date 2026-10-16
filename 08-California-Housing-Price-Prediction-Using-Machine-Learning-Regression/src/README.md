# 💻 Source Code — California Housing Price Prediction

## 📌 Overview

This folder contains the Python source code for the **California Housing Price Prediction Using Ensemble Learning** project.

The main program, `main.py`, implements the complete Machine Learning workflow, from loading and analyzing the dataset to model training, evaluation, visualization, and house price prediction.

## 📂 Files

- `main.py` — Main Python program containing the complete implementation of the project.

## ⚙️ Functionalities Implemented

The source code includes the following functionalities:

1. **Dataset Loading** — Loads the California Housing dataset from a CSV file.
2. **Data Quality Checks** — Checks missing values, duplicate rows, and capped house prices.
3. **Statistical Analysis** — Displays descriptive statistics for the dataset.
4. **Correlation Analysis** — Examines correlations between input features and house prices.
5. **Feature and Target Separation** — Separates independent variables from the target variable.
6. **Train-Test Split** — Divides the dataset into training and testing sets using an 80:20 ratio.
7. **Model Building** — Initializes six regression models.
8. **Cross-Validation** — Compares models using 5-fold cross-validation.
9. **Best Ensemble Selection** — Selects the best ensemble model using cross-validation R² scores.
10. **Model Training** — Trains the selected ensemble model.
11. **Model Persistence** — Saves and reloads the trained model using Joblib.
12. **Performance Evaluation** — Evaluates the selected model using R² Score, RMSE, and MAE.
13. **Model Comparison** — Displays the test performance of all six models.
14. **Data Visualization** — Generates model comparison, actual-versus-predicted, feature importance, and California housing price map plots.
15. **Price Prediction** — Displays actual and predicted prices for sample test records.

## 🤖 Models Implemented

| Model | Category |
|---|---|
| Linear Regression | Single model |
| Decision Tree Regressor | Single model |
| Support Vector Regression (SVR) | Single model |
| Bagging Regressor | Bagging ensemble |
| Random Forest Regressor | Bagging ensemble |
| Gradient Boosting Regressor | Boosting ensemble |

## 🛠️ Technologies and Libraries

- **Python** — Main programming language
- **Pandas** — Dataset handling and analysis
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Regression algorithms, preprocessing, cross-validation, and evaluation metrics
- **Joblib** — Saving and loading the trained model

## ▶️ How to Run

Run the following command from the directory containing `main.py`:

```bash
python main.py
```

Ensure that the dataset is available at `data/california_housing.csv` relative to the project root, and that the required Python libraries are installed.

## 📁 Output

When execution completes successfully, the program generates:

- A saved trained model: `models/california_housing_model.pkl`
- Model comparison graph: `images/model_comparison.png`
- Actual-versus-predicted graph: `images/actual_vs_predicted.png`
- Feature importance graph: `images/feature_importance.png`
- California housing price map: `images/price_map.png`

It also displays model evaluation metrics, dataset analysis, and sample price predictions in the terminal.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

Part of the `Applied-ML-Workbench` Machine Learning portfolio.

