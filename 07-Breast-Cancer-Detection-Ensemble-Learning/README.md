# 🩺 07 — Breast Cancer Detection Using Ensemble Learning

## 📌 Overview

This case study implements a machine learning pipeline to classify breast cancer cases as **malignant** or **benign** using the Wisconsin Breast Cancer Dataset.

It compares multiple classification algorithms, applies stratified 5-fold cross-validation, selects the best ensemble model, and evaluates its performance on unseen test data.

## 🎯 Objectives

- Analyze the breast cancer dataset.
- Train and compare machine learning classifiers.
- Apply bagging, boosting, and voting techniques.
- Select the best ensemble using malignant-class F1-score.
- Evaluate performance using classification metrics.
- Generate visualizations and save the trained model.

## 🛠️ Technologies Used

- Python
- NumPy and Pandas
- Matplotlib
- Scikit-learn
- Joblib

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

## 📂 Project Structure

```text
07-Breast-Cancer/
├── data/
│   ├── breast_cancer.csv
│   └── README.md
├── src/
│   ├── main.py
│   └── README.md
├── models/
│   ├── breast_cancer_model.pkl
│   └── README.md
├── images/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── README.md
└── README.md
```

## ⚙️ Workflow

1. Load and analyze the dataset.
2. Split the data into training and testing sets.
3. Build and compare 10 classification models.
4. Perform stratified 5-fold cross-validation.
5. Select the best ensemble model.
6. Evaluate the selected model on test data.
7. Generate performance visualizations.
8. Save the trained model and display sample predictions.

## 📊 Results

In the recorded execution:

- **Best ensemble:** Hard Voting
- **Cross-validation F1-score (malignant):** 96.49%
- **Test accuracy:** 98.25%
- **Test recall (malignant):** 95.24%
- **Test F1-score (malignant):** 97.56%

*Results correspond to the recorded run and may vary with dataset changes or different configurations.*

## ▶️ How to Run

Run the following command from the case study's root directory:

```bash
python src/main.py
```

Install dependencies if required:

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

## ⚠️ Disclaimer

This case study is intended for educational and machine learning experimentation. It is not a clinically validated diagnostic system and must not be used for medical decisions.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**
