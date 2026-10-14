# Source Code — Breast Cancer Detection Using Ensemble Learning

## 📌 Overview

The `src` folder contains the Python source code for the **Breast Cancer Detection Using Ensemble Learning** project.

The main program, `main.py`, implements a complete machine learning workflow to classify breast cancer cases as **malignant** or **benign** using the Wisconsin breast cancer dataset. It compares multiple classification algorithms, selects the best-performing ensemble model based on cross-validation F1-score for the malignant class, evaluates the selected model, and generates visualizations of the results.

## 📂 Folder Contents

```text
src/
└── main.py
```

| File | Description |
|---|---|
| `main.py` | Main Python program containing dataset loading, preprocessing checks, model creation, cross-validation, model selection, evaluation, visualization, and prediction functions. |
| `README.md` | Documentation for the source code and instructions for running the program. |

## ⚙️ Machine Learning Workflow

The program follows these steps:

1. **Load the dataset** — Reads `breast_cancer.csv` using Pandas.
2. **Check data quality** — Displays missing values, duplicate rows, class distribution, and baseline accuracy.
3. **Explore the data** — Displays descriptive statistics for the input measurements.
4. **Separate features and target** — Divides the dataset into independent features (`X`) and target labels (`Y`).
5. **Split the dataset** — Uses an 80:20 training-testing split with stratification.
6. **Build classification models** — Creates 10 models for comparison.
7. **Perform cross-validation** — Uses stratified 5-fold cross-validation on the training data.
8. **Select the best ensemble** — Chooses the ensemble with the highest mean cross-validation F1-score for the malignant class.
9. **Train and save the model** — Trains the selected ensemble and saves it using Joblib.
10. **Evaluate performance** — Calculates test accuracy, malignant-class recall and F1-score, the confusion matrix, and the classification report.
11. **Compare all models** — Evaluates the models on the held-out test set for reporting purposes.
12. **Generate visualizations** — Saves the model comparison graph, confusion matrix, and Random Forest feature-importance graph.
13. **Display sample predictions** — Compares predictions with actual labels for five test samples.

## 🤖 Algorithms Implemented

The program compares the following 10 classification models.

### 1. Baseline Models

- Decision Tree
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

These models provide baseline performance for comparison with ensemble techniques.

### 2. Bagging Algorithms

- Bagging Classifier
- Random Forest Classifier

### 3. Boosting Algorithms

- AdaBoost Classifier
- Gradient Boosting Classifier

### 4. Voting Classifiers

- **Hard Voting:** Combines class predictions from Logistic Regression, Decision Tree, and KNN using majority voting.
- **Soft Voting:** Combines predicted class probabilities from the same three classifiers.

Feature scaling using `StandardScaler` is applied within the pipelines for Logistic Regression, KNN, and SVM.

## 📊 Evaluation Metrics

The following metrics are used to evaluate model performance:

- **Accuracy:** Percentage of correctly classified cases.
- **Recall (Malignant):** Proportion of actual malignant cases correctly identified.
- **F1-score (Malignant):** Harmonic mean of malignant-class precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications.
- **Classification Report:** Provides precision, recall, F1-score, and support for each class.

The best ensemble is selected using cross-validation F1-score for the malignant class. This prioritizes performance on identifying malignant cases rather than relying on accuracy alone.

## 📦 Dependencies

The source code uses the following Python libraries:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

Install the required packages using:

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

## ▶️ How to Run

### Step 1: Open the project

Open the complete project folder in Visual Studio Code or your preferred Python IDE.

### Step 2: Verify the folder structure

The program expects the dataset in the project's root-level `data` directory.

```text
Breast-Cancer-Detection/
├── data/
│   └── breast_cancer.csv
├── src/
│   ├── main.py
│   └── README.md
├── images/
├── models/
├── requirements.txt
├── LICENSE
└── README.md
```

The `images` and `models` directories are created automatically when needed. The root `README.md`, `requirements.txt`, and `LICENSE` are project-level files.

### Step 3: Install dependencies

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

### Step 4: Run the program

From the project root directory, execute:

```bash
python src/main.py
```

Alternatively, open `src/main.py` in VS Code and run it using the Python interpreter configured for your environment.

**Dataset requirement:** The CSV file must contain a column named `target`, with `0` representing malignant and `1` representing benign cases, as expected by the program.

## 📁 Generated Outputs

After successful execution, the program saves the following files:

| Output | Purpose |
|---|---|
| `models/breast_cancer_model.pkl` | Trained best ensemble model saved using Joblib. |
| `images/model_comparison.png` | Comparison of cross-validation accuracy, malignant recall, and malignant F1-score. |
| `images/confusion_matrix.png` | Confusion matrix for the selected model on the test set. |
| `images/feature_importance.png` | Top 10 feature-importance measurements from a Random Forest model. |

The program also displays model comparisons, evaluation metrics, and sample predictions in the terminal.

## 🧠 Technologies Used

- **Programming Language:** Python
- **Data Handling:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Development Environment:** Visual Studio Code

## ⚠️ Disclaimer

This project is developed for educational purposes to demonstrate machine learning classification and ensemble learning techniques. It is **not a clinically validated diagnostic system** and must not be used to make medical decisions.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

*Breast Cancer Detection Using Ensemble Learning*


