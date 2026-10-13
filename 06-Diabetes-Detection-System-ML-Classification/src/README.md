# Source Code

This folder has the source code of the Diabetes Detection System case study.

## File

- `main.py`

## What the code does

- Loads the diabetes dataset from `data/diabetes.csv` (768 patients)
- Checks missing values, invalid zero values and the distribution of the target classes
- Replaces zero values in selected medical measurement columns with missing values
- Separates the 8 input features (X) and the target `Outcome` (Y)
- Splits the dataset into training (80%) and testing (20%) sets
- Builds five classification pipelines using median imputation and feature scaling
- Compares Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree and Random Forest using 5-fold cross-validation
- Evaluates each classifier using accuracy, recall and F1-score
- Selects the classifier with the highest cross-validation F1-score
- Trains the selected classifier on the training data
- Saves and reloads the trained model using `joblib`
- Evaluates the selected model on the test data using accuracy, recall, F1-score, confusion matrix and classification report
- Generates and saves model comparison, confusion matrix and feature importance charts
- Predicts diabetes outcomes for two example patients
- Displays the final project summary

## Functions

| **Function** | **Purpose** |
| ------------ | ----------- |
| `display_title` | Prints the project title |
| `display_step` | Displays the title of each processing step |
| `load_data` | Loads the CSV dataset and validates required columns |
| `check_data` | Displays missing values, zero counts and target class distribution |
| `replace_zeros_with_missing` | Replaces selected zero values with missing values |
| `split_features_target` | Separates input features (X) and target (Y) |
| `split_dataset` | Splits the dataset into training and testing sets |
| `build_models` | Creates the five classification pipelines |
| `compare_models` | Compares classifiers using 5-fold cross-validation |
| `select_best_model` | Selects the classifier with the highest cross-validation F1-score |
| `train_best_model` | Trains the selected classifier |
| `save_model` | Saves the trained model as a pickle file using `joblib` |
| `load_model` | Loads the saved model from storage |
| `evaluate_model` | Calculates test accuracy, recall, F1-score and classification report |
| `plot_model_comparison` | Generates and saves the classifier comparison chart |
| `plot_confusion_matrix` | Generates and saves the confusion matrix chart |
| `plot_feature_importance` | Trains a Random Forest and displays feature importance values |
| `detect_diabetes` | Predicts the outcome and estimated diabetes probability for a new patient |
| `display_summary` | Displays the final model performance summary |
| `display_footer` | Prints the project completion message |
| `main` | Executes the complete workflow in order |

## Features and Target

- **Features:** 8 medical attributes — Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction and Age
- **Target:** `Outcome` (0 = no diabetes, 1 = diabetes)

## Algorithms Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

The classifier with the highest 5-fold cross-validation F1-score is selected as the best model.

## How to Run

Run from the case study folder (the one with `requirements.txt`):

```bash
pip install -r requirements.txt
python src/main.py
```

You can also run it from inside the `src` folder:

```bash
python main.py
```

## Output Files

The code creates these images inside the `images` folder:

- `model_comparison.png`
- `confusion_matrix.png`
- `feature_importance.png`

The trained model is saved inside the `models` folder:

- `diabetes_model.pkl`

## Expected Results

The following results were obtained in the current run:

- Best classifier: Logistic Regression
- Cross-validation F1-score: 65.45%
- Accuracy on test data: 70.78%
- Recall for diabetes: 50.00%
- F1-score for diabetes: 54.55%

Results may vary if the dataset, random state, preprocessing or model parameters change.

## Note

This is an educational machine learning project. Its predictions are experimental and must not be used as a substitute for professional medical diagnosis.

## Author
**Ishwari Surve**

