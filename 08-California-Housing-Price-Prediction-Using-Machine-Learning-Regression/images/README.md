# 🏡 California Housing Price Prediction
### 📊 Model Visualizations & Performance Analysis

Welcome to the **Images** folder of the California Housing Price Prediction project! This folder contains visual representations of model performance, prediction accuracy, and feature importance to help understand the machine learning results.

---

## 🖼️ Visualizations

The following plots are used to analyze and compare the regression models.

### 1. 🔄 Cross-Validation R² Comparison

Compares the mean R² scores of nine regression algorithms using 5-fold cross-validation.

**Purpose:** To identify models that perform consistently well across validation folds.

### 2. 🏆 Regression Model Comparison

Compares the test-set R² scores of all regression models.

**Purpose:** To evaluate how well each model explains variation in California housing values.

### 3. 📉 RMSE Comparison

Visualizes the Root Mean Squared Error (RMSE) of each regression model.

**Purpose:** To compare prediction errors, with lower RMSE indicating better performance.

### 4. 🎯 MAE Comparison

Compares the Mean Absolute Error (MAE) across the regression algorithms.

**Purpose:** To identify models with lower average absolute prediction errors.

### 5. 🏡 Actual vs. Predicted House Values

Compares actual house values with the values predicted by the tuned Random Forest model.

**Purpose:** To visually assess prediction accuracy. Predictions closer to the reference line indicate better agreement with actual values.

### 6. 🔍 Residual Plot

Displays residual errors against predicted house values.

**Purpose:** To inspect prediction errors and identify possible patterns in model performance.

### 7. 🌳 Feature Importance

Displays the relative importance of the eight input features in the tuned Random Forest model.

**Purpose:** To understand which housing, demographic, and geographical features contribute most to the model's predictions.

### 8. ⚖️ Original vs. Tuned Random Forest

Compares the test-set R² scores of the original and hyperparameter-tuned Random Forest models.

**Purpose:** To visualize the effect of hyperparameter tuning on model performance.

---

## 📌 Why These Visualizations Matter

- 📊 Make model comparisons easier to understand.
- 🏆 Help identify high-performing regression algorithms.
- 🎯 Provide insight into prediction accuracy and error.
- 🌳 Help interpret feature importance.
- ⚙️ Show the impact of hyperparameter tuning.

---

## 📁 Folder Structure

```text
images/
├── README.md
├── cross_validation_comparison.png
├── model_comparison.png
├── rmse_comparison.png
├── mae_comparison.png
├── actual_vs_predicted.png
├── residual_plot.png
├── feature_importance.png
└── tuned_model_comparison.png
```

*Note: These are suggested filenames. Use the actual filenames of your saved images.*

---

## 🛠️ Tools Used

- 🐍 Python
- 📊 Matplotlib
- 🐼 Pandas
- 🤖 Scikit-learn

---

## 🎯 Project Goal

To present machine learning results visually and make the performance, accuracy, and behavior of California housing price prediction models easier to analyze.

---

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

*Case Study 08 — California Housing Price Prediction Using Machine Learning Regression*


