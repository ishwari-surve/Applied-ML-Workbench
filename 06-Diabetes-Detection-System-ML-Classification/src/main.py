"""
------------------------------------------------------------
Project Name        : Diabetes Detection System

Dataset Information :

Dataset File        : data/diabetes.csv (768 patients)

Features            : Pregnancies              -> Number of pregnancies
                      Glucose                  -> Plasma glucose level
                      BloodPressure            -> Diastolic blood pressure
                      SkinThickness            -> Skin fold thickness
                      Insulin                  -> Insulin level
                      BMI                      -> Body mass index
                      DiabetesPedigreeFunction -> Family history score
                      Age                      -> Age in years

Target              : Outcome -> 0 = no diabetes, 1 = diabetes

Machine Learning Information :

Algorithms Compared : Logistic Regression
                      K-Nearest Neighbors (KNN)
                      Support Vector Machine (SVM)
                      Decision Tree
                      Random Forest

Techniques Used     : Median imputation of missing medical values
                      Feature Scaling (StandardScaler)
                      5-fold Cross Validation
                      Model selection by cross validation F1-score
                      Model saving and loading (joblib)

Library             : Scikit-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Accuracy, Recall, F1-score
                      Confusion Matrix

Author              : Ishwari Vijaykumar Surve

Date                : 13/10/2026
------------------------------------------------------------
"""


############################################################
# Required Python Packages
############################################################

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


############################################################
# File Paths
############################################################

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "diabetes.csv"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_model.pkl"
)


############################################################
# Headers
############################################################

TARGET_HEADER = "Outcome"

FEATURE_HEADERS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# In these columns a value of 0 is not possible for a living
# patient, so a 0 means that the value is missing.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

CROSS_VALIDATION_FOLDS = 5


############################################################
# Display Configuration
############################################################

BORDER = "-" * 60


############################################################
# Function Name : display_title
# Description   : Display project title
# Input         : Nothing
# Output        : Displays project title
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("            DIABETES DETECTION SYSTEM")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the diabetes dataset. It checks that the
#                 file and the required columns exist.
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def load_data(path):

    # Check that the dataset file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at : {path}\n"
            "Please place the CSV file at data/diabetes.csv"
        )

    df = pd.read_csv(path)

    # Check that the columns needed by the program exist
    for column in FEATURE_HEADERS + [TARGET_HEADER]:
        if column not in df.columns:
            raise ValueError(f"The dataset must contain the '{column}' column")

    print()
    print("First five rows of the dataset :")
    print(df.head())

    print()
    print("Shape of the dataset :", df.shape)

    return df


############################################################
# Function Name : check_data
# Description   : Display missing values, the number of impossible
#                 zero values and the class distribution
# Input         : df - Input pandas dataframe
# Output        : Displays data quality information
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def check_data(df):

    print()
    print("Missing values (empty cells) :", int(df.isnull().sum().sum()))

    print()
    print("Zero values in columns where 0 is not possible :")
    for column in ZERO_AS_MISSING_COLUMNS:
        print(f"{column:<15}: {int((df[column] == 0).sum())}")

    counts = df[TARGET_HEADER].value_counts()

    print()
    print("Outcome count :")
    print(f"0 (no diabetes) : {counts[0]}")
    print(f"1 (diabetes)    : {counts[1]}")

    print()
    print(f"Baseline accuracy (always 'no diabetes') : "
          f"{counts[0] / len(df) * 100:.2f} %")


############################################################
# Function Name : replace_zeros_with_missing
# Description   : Replace the impossible zero values with NaN.
#                 They are filled with the median later, inside
#                 the model pipeline.
# Input         : df - Input pandas dataframe
# Output        : Returns dataframe with NaN instead of zeros
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def replace_zeros_with_missing(df):

    df = df.copy()

    df[ZERO_AS_MISSING_COLUMNS] = df[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)

    print()
    print("Missing values after replacing the zeros :")
    print(df[ZERO_AS_MISSING_COLUMNS].isnull().sum())

    return df


############################################################
# Function Name : split_features_target
# Description   : Separate independent and dependent variables
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def split_features_target(df):

    X = df[FEATURE_HEADERS]

    Y = df[TARGET_HEADER]

    print()
    print("Shape of X :", X.shape)
    print("Shape of Y :", Y.shape)

    return X, Y


############################################################
# Function Name : split_dataset
# Description   : Split dataset into training and testing data
# Input         : X - Features
#                 Y - Target
# Output        : Returns training and testing datasets
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def split_dataset(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )

    print()
    print("X_train shape :", X_train.shape)
    print("X_test shape  :", X_test.shape)
    print("Y_train shape :", Y_train.shape)
    print("Y_test shape  :", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


############################################################
# Function Name : build_models
# Description   : Build the 5 classifiers. Every classifier is a
#                 pipeline: median imputation, scaling, model.
# Input         : Nothing
# Output        : Returns dictionary of model name and pipeline
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def build_models():

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    }

    models = {}

    for name, classifier in classifiers.items():

        models[name] = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            classifier
        )

    print()
    print("Classifiers to compare :", list(models.keys()))

    return models


############################################################
# Function Name : compare_models
# Description   : Compare all classifiers with 5-fold cross
#                 validation on the training data only.
#                 The test data is not used here.
# Input         : models - Dictionary of model name and pipeline
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns dataframe of cross validation scores
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def compare_models(models, X_train, Y_train):

    folds = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    rows = []

    for name, pipeline in models.items():

        scores = cross_validate(
            pipeline,
            X_train,
            Y_train,
            cv=folds,
            scoring=["accuracy", "recall", "f1"]
        )

        rows.append({
            "Model": name,
            "CV Accuracy": scores["test_accuracy"].mean() * 100,
            "CV Recall": scores["test_recall"].mean() * 100,
            "CV F1": scores["test_f1"].mean() * 100
        })

    results = pd.DataFrame(rows).set_index("Model").round(2)

    print()
    print("Cross validation scores (%) :")
    print(results)

    return results


############################################################
# Function Name : select_best_model
# Description   : Select the classifier with the highest cross
#                 validation F1-score. F1 balances the patients
#                 that are missed and the false alarms.
# Input         : results - Cross validation scores
# Output        : Returns name of the best classifier
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def select_best_model(results):

    best_name = results["CV F1"].idxmax()

    print()
    print("Best classifier (highest cross validation F1) :", best_name)
    print(f"Cross validation F1 : {results.loc[best_name, 'CV F1']:.2f} %")

    return best_name


############################################################
# Function Name : train_best_model
# Description   : Train the best classifier on the training data
# Input         : model - Best classifier pipeline
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def train_best_model(model, X_train, Y_train):

    model.fit(X_train, Y_train)

    print()
    print("Best classifier trained successfully")

    return model


############################################################
# Function Name : save_model
# Description   : Save the trained model on secondary storage
# Input         : model - Trained model
#                 path - Path of the model file
# Output        : Saves the model file in models folder
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def save_model(model, path):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(model, path)

    print()
    print("Model saved successfully : models/diabetes_model.pkl")


############################################################
# Function Name : load_model
# Description   : Load the saved model from secondary storage
# Input         : path - Path of the model file
# Output        : Returns loaded model
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def load_model(path):

    model = joblib.load(path)

    print("Model loaded successfully : models/diabetes_model.pkl")

    return model


############################################################
# Function Name : evaluate_model
# Description   : Test the loaded model once on the test data.
#                 It shows accuracy, baseline accuracy, recall,
#                 F1-score, confusion matrix and report.
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
# Output        : Returns predictions, accuracy, recall and F1
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def evaluate_model(model, X_test, Y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)
    baseline = Y_test.value_counts(normalize=True).max()

    print()
    print(f"Accuracy on test data               : {accuracy * 100:.2f} %")
    print(f"Baseline accuracy (always 'No')     : {baseline * 100:.2f} %")
    print(f"Recall for diabetes (patients found): {recall * 100:.2f} %")
    print(f"F1-score for diabetes               : {f1 * 100:.2f} %")

    print()
    print("Confusion matrix :")
    print(confusion_matrix(Y_test, predictions))

    print()
    print(classification_report(
        Y_test,
        predictions,
        target_names=["No diabetes", "Diabetes"]
    ))

    return predictions, accuracy, recall, f1


############################################################
# Function Name : plot_model_comparison
# Description   : Draw and save the comparison of the 5
#                 classifiers (cross validation scores)
# Input         : results - Cross validation scores
#                 best_name - Name of the best classifier
# Output        : Saves model comparison image
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def plot_model_comparison(results, best_name):

    positions = np.arange(len(results))

    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(positions - width, results["CV Accuracy"], width, label="Accuracy")
    plt.bar(positions, results["CV Recall"], width, label="Recall")
    plt.bar(positions + width, results["CV F1"], width, label="F1-score")

    plt.xticks(positions, results.index)
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.title(f"Model Comparison (5-fold cross validation) - Best: {best_name}")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "model_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/model_comparison.png")


############################################################
# Function Name : plot_confusion_matrix
# Description   : Draw and save the confusion matrix image
# Input         : Y_test - Actual target values
#                 predictions - Predicted target values
#                 best_name - Name of the best classifier
# Output        : Saves confusion matrix image
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def plot_confusion_matrix(Y_test, predictions, best_name):

    matrix = confusion_matrix(Y_test, predictions)

    class_names = ["No diabetes", "Diabetes"]

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(2), class_names)
    plt.yticks(range(2), class_names)

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=14
            )

    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/confusion_matrix.png")


############################################################
# Function Name : plot_feature_importance
# Description   : Train a Random Forest on the training data and
#                 save the feature importance graph. It shows
#                 which medical values matter most.
# Input         : X_train - Training features
#                 Y_train - Training target
# Output        : Saves feature importance image
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def plot_feature_importance(X_train, Y_train):

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    )

    pipeline.fit(X_train, Y_train)

    importance = pd.Series(
        pipeline[-1].feature_importances_,
        index=FEATURE_HEADERS
    ).sort_values()

    print()
    print("Feature importance (Random Forest) :")
    print(importance.sort_values(ascending=False).round(3))

    plt.figure(figsize=(8, 5))

    plt.barh(importance.index, importance.values)

    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/feature_importance.png")


############################################################
# Function Name : detect_diabetes
# Description   : Detect diabetes for a new patient using the
#                 loaded model. A missing value can be given
#                 as None and it is filled with the median.
# Input         : model - Loaded model
#                 patient - Dictionary of the patient values
# Output        : Displays the result and the probability
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def detect_diabetes(model, patient):

    sample = pd.DataFrame(
        [patient],
        columns=FEATURE_HEADERS
    ).astype(float)

    result = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    label = "Diabetes detected" if result == 1 else "No diabetes detected"

    print()
    print(patient)
    print(f"-> {label} (probability of diabetes : {probability:.2f})")


############################################################
# Function Name : display_summary
# Description   : Display the final project summary
# Input         : best_name - Name of the best classifier
#                 results - Cross validation scores
#                 accuracy - Accuracy on test data
#                 recall - Recall on test data
#                 f1 - F1-score on test data
# Output        : Displays project summary
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def display_summary(best_name, results, accuracy, recall, f1):

    print()
    print("Classifiers compared      : 5")
    print("Best classifier           :", best_name)
    print(f"Cross validation F1       : {results.loc[best_name, 'CV F1']:.2f} %")
    print(f"Test accuracy             : {accuracy * 100:.2f} %")
    print(f"Test recall (diabetes)    : {recall * 100:.2f} %")
    print(f"Test F1-score (diabetes)  : {f1 * 100:.2f} %")
    print()
    print("Note : this is a learning project, not a medical diagnosis.")


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("        DIABETES DETECTION SYSTEM COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Diabetes Detection System
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 13/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load the dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Check the data quality")
    check_data(df)

    display_step("Step 3: Replace the impossible zero values with missing values")
    df = replace_zeros_with_missing(df)

    display_step("Step 4: Separate independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 5: Split the dataset for training and testing")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    display_step("Step 6: Compare the 5 classifiers with cross validation")
    models = build_models()
    results = compare_models(models, X_train, Y_train)

    display_step("Step 7: Select the best classifier")
    best_name = select_best_model(results)

    display_step("Step 8: Train the best classifier")
    best_model = train_best_model(models[best_name], X_train, Y_train)

    display_step("Step 9: Save and load the model")
    save_model(best_model, MODEL_PATH)
    loaded_model = load_model(MODEL_PATH)

    display_step("Step 10: Evaluate the model on the test data")
    predictions, accuracy, recall, f1 = evaluate_model(
        loaded_model, X_test, Y_test
    )

    display_step("Step 11: Plot the graphs")
    plot_model_comparison(results, best_name)
    plot_confusion_matrix(Y_test, predictions, best_name)
    plot_feature_importance(X_train, Y_train)

    display_step("Step 12: Detect diabetes for new patients")
    detect_diabetes(loaded_model, {
        "Pregnancies": 5, "Glucose": 160, "BloodPressure": 80,
        "SkinThickness": 30, "Insulin": 150, "BMI": 36.5,
        "DiabetesPedigreeFunction": 0.8, "Age": 52
    })
    detect_diabetes(loaded_model, {
        "Pregnancies": 1, "Glucose": 90, "BloodPressure": 70,
        "SkinThickness": 20, "Insulin": None, "BMI": 24.0,
        "DiabetesPedigreeFunction": 0.2, "Age": 25
    })

    display_step("Step 13: Project summary")
    display_summary(best_name, results, accuracy, recall, f1)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()