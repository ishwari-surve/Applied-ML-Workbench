# 🩺 Diabetes Detection System using Machine Learning Classification

A machine learning classification project that predicts whether a person belongs to the **diabetes** or **no diabetes** class using medical attributes. The project compares five classification algorithms using **5-fold cross-validation** and selects the best classifier based on the F1-score.

## 🎯 Problem Statement

Given eight medical attributes, such as glucose level, BMI, blood pressure, insulin level and age, predict whether the outcome belongs to the diabetes class (1) or the no diabetes class (0).

This project demonstrates data preprocessing, classifier comparison, model selection, evaluation and model persistence using Python and scikit-learn.

## 🛠️ Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn
- matplotlib
- Joblib

## 📊 Dataset

- File: `data/diabetes.csv`
- Size: 768 samples, 8 input features and 1 target (`Outcome`)
- Classes: No diabetes (500 samples), Diabetes (268 samples)
- Missing values in the original dataset: none
- Zero values treated as missing: present in selected medical measurement columns

The eight input features are `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction` and `Age`.

The target column is `Outcome`:

- `0` = No diabetes
- `1` = Diabetes

The classes are not balanced. A model that always predicts no diabetes achieves approximately **65.10% accuracy** on the complete dataset, which provides a baseline for comparison.

## 🤖 Algorithms and Techniques

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`): estimates the probability of belonging to the diabetes class.
- **K-Nearest Neighbors (KNN)** (`sklearn.neighbors.KNeighborsClassifier`): classifies a sample using nearby training samples.
- **Support Vector Machine (SVM)** (`sklearn.svm.SVC`): separates classes using a decision boundary.
- **Decision Tree** (`sklearn.tree.DecisionTreeClassifier`): classifies samples using feature-based decision rules.
- **Random Forest** (`sklearn.ensemble.RandomForestClassifier`): combines multiple decision trees.
- **Missing-value handling** (`SimpleImputer`): uses median imputation for missing medical feature values.
- **Feature scaling** (`StandardScaler`): scales features within the machine learning pipelines.
- **5-fold cross-validation** (`StratifiedKFold`): compares the classifiers using accuracy, recall and F1-score.
- **Model selection:** chooses the classifier with the highest mean cross-validation F1-score.
- **Model persistence** (`Joblib`): saves and reloads the trained model pipeline.

## 🔍 Workflow

1. Load the diabetes dataset.
2. Check the dataset shape, missing values, invalid zero values and target class distribution.
3. Replace zero values in selected medical measurement columns with missing values.
4. Separate independent variables (X) and dependent variable (Y).
5. Split the dataset into training (80%) and testing (20%) sets using stratified sampling.
6. Build five classification pipelines with median imputation and feature scaling.
7. Compare the five classifiers using 5-fold cross-validation.
8. Select the classifier with the highest cross-validation F1-score.
9. Train the selected classifier on the training data.
10. Save and reload the trained model using Joblib.
11. Evaluate the model on the held-out test data using accuracy, recall, F1-score, confusion matrix and classification report.
12. Generate the model comparison, confusion matrix and feature importance charts.
13. Predict the outcomes of two example patients and display the final project summary.

## 📂 Project Structure

```text
06-Diabetes-Detection-System-ML-Classification/
├── data/
│   ├── diabetes.csv
│   └── README.md
├── images/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── output_1.png
│   ├── output_2.png
│   ├── output_3.png
│   ├── output_4.png
│   └── README.md
├── models/
│   ├── diabetes_model.pkl
│   └── README.md
├── src/
│   ├── main.py
│   └── README.md
├── requirements.txt
└── README.md
```

## ▶️ How to Run

Run from this folder (the one with `requirements.txt`):

```bash
pip install -r requirements.txt
python src/main.py
```

You can also run the program from inside the `src` folder:

```bash
python main.py
```

The program generates three charts automatically in the `images` folder and saves the trained model in the `models` folder.

## 🖥️ Sample Output

```text
Step 7: Select the best classifier

Best classifier (highest cross validation F1) : Logistic Regression
Cross validation F1 : 65.45 %

Step 10: Evaluate the model on the test data

Accuracy on test data               : 70.78 %
Baseline accuracy (always 'No')     : 64.94 %
Recall for diabetes (patients found): 50.00 %
F1-score for diabetes               : 54.55 %

Confusion matrix :
[[82 18]
 [27 27]]

Step 13: Project summary

Classifiers compared      : 5
Best classifier           : Logistic Regression
Cross validation F1       : 65.45 %
Test accuracy             : 70.78 %
Test recall (diabetes)    : 50.00 %
Test F1-score (diabetes)  : 54.55 %
```

## 📸 Output Screenshots

### Model Comparison

[Model Comparison](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/model_comparison.png)

### Confusion Matrix

[Confusion Matrix](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/confusion_matrix.png)

### Feature Importance

[Feature Importance](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/feature_importance.png)

### Terminal Output

- [Dataset loading and data quality](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/output_1.png)
- [Preprocessing and classifier comparison](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/output_2.png)
- [Model selection and evaluation](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/output_3.png)
- [Predictions and project summary](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/06-Diabetes-Detection-System-ML-Classification/images/output_4.png)

## ✅ Results

| **Measure** | **Value** |
| ----------- | --------: |
| Best classifier | Logistic Regression |
| Cross-validation accuracy | 78.82% |
| Cross-validation recall | 57.48% |
| Cross-validation F1-score | 65.45% |
| Accuracy on test data (154 samples) | 70.78% |
| Test recall for diabetes | 50.00% |
| Test F1-score for diabetes | 54.55% |
| Test baseline accuracy | 64.94% |

- Logistic Regression achieved the highest cross-validation F1-score among the five classifiers.
- The model correctly classified 82 no-diabetes cases and 27 diabetes cases in the test set.
- It incorrectly classified 18 no-diabetes cases as diabetes and missed 27 diabetes cases.
- Glucose had the highest feature importance in the Random Forest analysis, followed by BMI and DiabetesPedigreeFunction.
- Test accuracy was higher than the test-set majority-class baseline of 64.94%.

The dataset contains 154 test samples. The results describe this particular train-test split; performance may vary with different data splits or model settings.

### Model Comparison

The `model_comparison.png` chart compares cross-validation accuracy, recall and F1-score for all five classifiers.

### Confusion Matrix

The `confusion_matrix.png` chart shows the correct and incorrect predictions made by the selected Logistic Regression model on the test data.

### Feature Importance

The `feature_importance.png` chart shows the relative feature importance calculated by a separate Random Forest model. These values describe the model's feature usage and do not establish medical causation.

## 📚 Learning Outcomes

- Data inspection and preprocessing using pandas and NumPy.
- Identifying invalid zero values in medical measurement columns.
- Handling missing values using median imputation.
- Building machine learning pipelines with scikit-learn.
- Comparing five binary classification algorithms.
- Evaluating classifiers using 5-fold cross-validation.
- Selecting a model based on F1-score.
- Evaluating classification performance using accuracy, recall, F1-score and a confusion matrix.
- Saving and loading trained models using Joblib.
- Visualizing model performance and feature importance.
- Generating predictions for new input records.

## 🚀 Future Improvements

- Tune classifier hyperparameters using `GridSearchCV`.
- Explore methods to improve recall for the diabetes class.
- Compare alternative preprocessing and feature-selection techniques.
- Evaluate performance using repeated cross-validation and additional metrics.
- Investigate model calibration and decision thresholds.
- Add a user interface for entering input values and displaying model predictions.

## ⚠️ Important Note

This project is intended for educational machine learning experimentation only. It is not a clinically validated diagnostic system, and its predictions must not be used as a substitute for professional medical advice or diagnosis.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

GitHub: [https://github.com/ishwari-surve](https://github.com/ishwari-surve)

