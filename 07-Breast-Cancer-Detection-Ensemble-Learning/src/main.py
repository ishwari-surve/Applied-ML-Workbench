"""
------------------------------------------------------------
Project Name        : Breast Cancer Detection using Ensemble Learning

Dataset Information :

Dataset File        : data/breast_cancer.csv (569 patients)

Features            : 30 measurements of the cell nuclei
                      (mean, error and worst values of radius,
                       texture, perimeter, area, smoothness,
                       compactness, concavity, symmetry, etc.)

Target              : target -> 0 = malignant (cancer)
                                1 = benign (not cancer)

Machine Learning Information :

Models Compared     : Single models   -> Decision Tree, Logistic
                                         Regression, KNN, SVM
                                         (baseline for comparison)
                      Bagging         -> Bagging Classifier,
                                         Random Forest
                      Boosting        -> AdaBoost, Gradient Boosting
                      Voting          -> Hard Voting, Soft Voting

Techniques Used     : Feature Scaling (StandardScaler)
                      5-fold Cross Validation
                      Best ensemble selected by cross validation
                      F1-score of the malignant class
                      Model saving and loading (joblib)

Library             : Scikit-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Accuracy, Recall, F1-score
                      Confusion Matrix

Author              : Ishwari Vijaykumar Surve

Date                : 14/10/2026
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    make_scorer,
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
    "breast_cancer.csv"
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
    "breast_cancer_model.pkl"
)


############################################################
# Headers
############################################################

TARGET_HEADER = "target"

# target 0 = malignant (cancer), target 1 = benign (not cancer)
MALIGNANT = 0

CLASS_NAMES = ["Malignant", "Benign"]


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

CROSS_VALIDATION_FOLDS = 5

TREES = 100


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
# Date          : 14/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("     BREAST CANCER DETECTION - ENSEMBLE LEARNING")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the breast cancer dataset. It checks that
#                 the file and the target column exist.
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def load_data(path):

    # Check that the dataset file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at : {path}\n"
            "Please place the CSV file at data/breast_cancer.csv"
        )

    df = pd.read_csv(path)

    # Check that the target column exists
    if TARGET_HEADER not in df.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_HEADER}' column")

    print()
    print("First five rows of the dataset (first 5 columns) :")
    print(df.iloc[:, :5].head())

    print()
    print("Shape of the dataset :", df.shape)

    return df


############################################################
# Function Name : check_data
# Description   : Display missing values, duplicate rows and
#                 the class distribution
# Input         : df - Input pandas dataframe
# Output        : Displays data quality information
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def check_data(df):

    counts = df[TARGET_HEADER].value_counts()

    print()
    print("Missing values (empty cells) :", int(df.isnull().sum().sum()))
    print("Duplicate rows               :", int(df.duplicated().sum()))

    print()
    print("Target count :")
    print(f"0 (malignant) : {counts[0]}")
    print(f"1 (benign)    : {counts[1]}")

    print()
    print(f"Baseline accuracy (always 'benign') : "
          f"{counts.max() / len(df) * 100:.2f} %")


############################################################
# Function Name : show_statistics
# Description   : Display the statistical summary of the first
#                 ten measurements of the dataset
# Input         : df - Input pandas dataframe
# Output        : Displays count, mean, std, min and max
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def show_statistics(df):

    summary = df.drop(columns=[TARGET_HEADER]).describe().T

    print()
    print("Statistical summary of the first 10 measurements :")
    print(summary[["count", "mean", "std", "min", "max"]].head(10).round(2))


############################################################
# Function Name : split_features_target
# Description   : Separate independent and dependent variables
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def split_features_target(df):

    X = df.drop(columns=[TARGET_HEADER])

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
# Date          : 14/10/2026
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
# Description   : Build the 10 models.
#                 Models that use distance or gradients
#                 (Logistic Regression, KNN, SVM) are scaled.
#                 The 4 single models are the baseline. The
#                 voting models are made of Logistic Regression,
#                 Decision Tree and KNN.
# Input         : Nothing
# Output        : Returns dictionary of model name and
#                 (model type, model)
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def build_models():

    def logistic_regression():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000)
        )

    def decision_tree():
        return DecisionTreeClassifier(random_state=RANDOM_STATE)

    def knn():
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        )

    def voting(kind):
        return VotingClassifier(
            estimators=[
                ("lr", logistic_regression()),
                ("dt", decision_tree()),
                ("knn", knn())
            ],
            voting=kind
        )

    models = {
        "Decision Tree": ("Single", decision_tree()),
        "Logistic Regression": ("Single", logistic_regression()),
        "KNN": ("Single", knn()),
        "SVM": ("Single", make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1, gamma="scale")
        )),
        "Bagging": ("Bagging", BaggingClassifier(
            estimator=decision_tree(),
            n_estimators=TREES,
            random_state=RANDOM_STATE
        )),
        "Random Forest": ("Bagging", RandomForestClassifier(
            n_estimators=TREES,
            random_state=RANDOM_STATE
        )),
        "AdaBoost": ("Boosting", AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=RANDOM_STATE
        )),
        "Gradient Boosting": ("Boosting", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )),
        "Hard Voting": ("Voting", voting("hard")),
        "Soft Voting": ("Voting", voting("soft"))
    }

    print()
    print("Models to compare :", len(models))

    for name, (kind, model) in models.items():
        print(f"{name:<22}: {kind}")

    return models


############################################################
# Function Name : compare_models
# Description   : Compare all models with 5-fold cross
#                 validation on the training data only.
#                 The test data is not used here.
#                 Malignant is the class we must not miss, so
#                 recall and F1 are calculated for malignant.
# Input         : models - Dictionary of models
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns dataframe of cross validation scores
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def compare_models(models, X_train, Y_train):

    folds = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "recall": make_scorer(recall_score, pos_label=MALIGNANT),
        "f1": make_scorer(f1_score, pos_label=MALIGNANT)
    }

    rows = []

    for name, (kind, model) in models.items():

        scores = cross_validate(
            model,
            X_train,
            Y_train,
            cv=folds,
            scoring=scoring
        )

        rows.append({
            "Model": name,
            "Type": kind,
            "CV Accuracy": scores["test_accuracy"].mean() * 100,
            "CV Recall": scores["test_recall"].mean() * 100,
            "CV F1": scores["test_f1"].mean() * 100
        })

    results = pd.DataFrame(rows).set_index("Model").round(2)

    print()
    print("Cross validation scores (%) :")
    print("(Recall and F1 are for the malignant class)")
    print(results)

    return results


############################################################
# Function Name : select_best_model
# Description   : Select the best ENSEMBLE model (Bagging,
#                 Boosting or Voting) with the highest cross
#                 validation F1-score for the malignant class.
#                 The 4 single models are only the baseline.
# Input         : results - Cross validation scores
# Output        : Returns name of the best ensemble model
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def select_best_model(results):

    ensembles = results[results["Type"] != "Single"]

    best_name = ensembles["CV F1"].idxmax()

    best_single = results[results["Type"] == "Single"]["CV F1"].idxmax()

    print()
    print("Best ensemble model                      :", best_name)
    print("Type                                     :", results.loc[best_name, "Type"])
    print(f"Cross validation F1                      : {results.loc[best_name, 'CV F1']:.2f} %")

    print()
    print("Best single model (baseline)             :", best_single)
    print(f"Cross validation F1                      : {results.loc[best_single, 'CV F1']:.2f} %")

    worst_single = results[results["Type"] == "Single"]["CV F1"].min()

    print()
    print(f"The best ensemble is {results.loc[best_name, 'CV F1'] - worst_single:.2f} "
          "points better than the weakest single model.")

    return best_name


############################################################
# Function Name : train_best_model
# Description   : Train the best model on the training data
# Input         : model - Best model
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def train_best_model(model, X_train, Y_train):

    model.fit(X_train, Y_train)

    print()
    print("Best model trained successfully")

    return model


############################################################
# Function Name : save_model
# Description   : Save the trained model on secondary storage
# Input         : model - Trained model
#                 path - Path of the model file
# Output        : Saves the model file in models folder
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def save_model(model, path):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(model, path)

    print()
    print("Model saved successfully : models/breast_cancer_model.pkl")


############################################################
# Function Name : load_model
# Description   : Load the saved model from secondary storage
# Input         : path - Path of the model file
# Output        : Returns loaded model
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def load_model(path):

    model = joblib.load(path)

    print("Model loaded successfully : models/breast_cancer_model.pkl")

    return model


############################################################
# Function Name : evaluate_model
# Description   : Test the loaded model once on the test data.
#                 It shows accuracy, baseline accuracy, recall
#                 and F1-score for malignant, confusion matrix
#                 and classification report.
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
# Output        : Returns predictions, accuracy, recall and F1
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def evaluate_model(model, X_test, Y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions, pos_label=MALIGNANT)
    f1 = f1_score(Y_test, predictions, pos_label=MALIGNANT)
    baseline = Y_test.value_counts(normalize=True).max()

    print()
    print(f"Accuracy on test data                  : {accuracy * 100:.2f} %")
    print(f"Baseline accuracy (always 'benign')    : {baseline * 100:.2f} %")
    print(f"Recall for malignant (cancers found)   : {recall * 100:.2f} %")
    print(f"F1-score for malignant                 : {f1 * 100:.2f} %")

    print()
    print("Confusion matrix :")
    print(confusion_matrix(Y_test, predictions))

    print()
    print(classification_report(
        Y_test,
        predictions,
        target_names=CLASS_NAMES
    ))

    return predictions, accuracy, recall, f1


############################################################
# Function Name : evaluate_all_models
# Description   : Train every model on the training data and test
#                 it on the test data. This table is only for
#                 reporting. The test data is NOT used to select
#                 the best model.
# Input         : models - Dictionary of models
#                 X_train - Training features
#                 Y_train - Training target
#                 X_test - Testing features
#                 Y_test - Actual testing target
# Output        : Displays test scores of all models
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def evaluate_all_models(models, X_train, Y_train, X_test, Y_test):

    rows = []

    for name, (kind, model) in models.items():

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        rows.append({
            "Model": name,
            "Type": kind,
            "Test Accuracy": accuracy_score(Y_test, predictions) * 100,
            "Test Recall": recall_score(Y_test, predictions, pos_label=MALIGNANT) * 100,
            "Test F1": f1_score(Y_test, predictions, pos_label=MALIGNANT) * 100
        })

    table = pd.DataFrame(rows).set_index("Model").round(2)

    print()
    print("Test scores of all models (%) - for reporting only :")
    print("(Recall and F1 are for the malignant class)")
    print(table)


############################################################
# Function Name : plot_model_comparison
# Description   : Draw and save the comparison of the 10 models
#                 (cross validation scores)
# Input         : results - Cross validation scores
#                 best_name - Name of the best model
# Output        : Saves model comparison image
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def plot_model_comparison(results, best_name):

    positions = np.arange(len(results))

    width = 0.27

    plt.figure(figsize=(12, 6))

    plt.bar(positions - width, results["CV Accuracy"], width, label="Accuracy")
    plt.bar(positions, results["CV Recall"], width, label="Recall (malignant)")
    plt.bar(positions + width, results["CV F1"], width, label="F1-score (malignant)")

    plt.xticks(positions, results.index, rotation=30, ha="right")
    plt.ylabel("Score (%)")
    plt.ylim(80, 100)
    plt.title(f"Model Comparison (5-fold cross validation) - Best ensemble: {best_name}")
    plt.legend(loc="lower right")
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
#                 best_name - Name of the best model
# Output        : Saves confusion matrix image
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def plot_confusion_matrix(Y_test, predictions, best_name):

    matrix = confusion_matrix(Y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(2), CLASS_NAMES)
    plt.yticks(range(2), CLASS_NAMES)

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
#                 save the graph of the 10 most important
#                 measurements.
# Input         : X_train - Training features
#                 Y_train - Training target
# Output        : Saves feature importance image
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def plot_feature_importance(X_train, Y_train):

    forest = RandomForestClassifier(
        n_estimators=TREES,
        random_state=RANDOM_STATE
    )

    forest.fit(X_train, Y_train)

    importance = pd.Series(
        forest.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    top = importance.head(10)

    print()
    print("Top 10 important measurements (Random Forest) :")
    print(top.round(3))

    plt.figure(figsize=(8, 6))

    plt.barh(top.index[::-1], top.values[::-1])

    plt.title("Top 10 Feature Importance (Random Forest)")
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
# Function Name : detect_cancer
# Description   : Detect cancer for some patients of the test
#                 data using the loaded model and compare the
#                 result with the real diagnosis
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
#                 count - Number of patients to check
# Output        : Displays predicted and actual diagnosis
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def detect_cancer(model, X_test, Y_test, count):

    samples = X_test.iloc[:count]

    predictions = model.predict(samples)

    print()
    for index, prediction, actual in zip(samples.index, predictions, Y_test.iloc[:count]):

        result = "Correct" if prediction == actual else "Wrong"

        print(
            f"Patient {index:<4}: Predicted = {CLASS_NAMES[prediction]:<10} "
            f"Actual = {CLASS_NAMES[actual]:<10} -> {result}"
        )


############################################################
# Function Name : display_summary
# Description   : Display the final project summary
# Input         : best_name - Name of the best model
#                 results - Cross validation scores
#                 accuracy - Accuracy on test data
#                 recall - Recall for malignant on test data
#                 f1 - F1-score for malignant on test data
# Output        : Displays project summary
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def display_summary(best_name, results, accuracy, recall, f1):

    print()
    print("Models compared               : 10")
    print("Best ensemble model           :", best_name)
    print(f"Cross validation F1           : {results.loc[best_name, 'CV F1']:.2f} %")
    print(f"Test accuracy                 : {accuracy * 100:.2f} %")
    print(f"Test recall (malignant)       : {recall * 100:.2f} %")
    print(f"Test F1-score (malignant)     : {f1 * 100:.2f} %")
    print()
    print("Note : this is a learning project, not a medical diagnosis.")


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("     BREAST CANCER DETECTION COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Breast Cancer Detection
#                 project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 14/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load the dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Check the data quality")
    check_data(df)

    display_step("Step 3: Display the statistical summary")
    show_statistics(df)

    display_step("Step 4: Separate independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 5: Split the dataset for training and testing")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    display_step("Step 6: Build the 10 models")
    models = build_models()

    display_step("Step 7: Compare the models with cross validation")
    results = compare_models(models, X_train, Y_train)

    display_step("Step 8: Select the best ensemble model")
    best_name = select_best_model(results)

    display_step("Step 9: Train the best ensemble model")
    best_model = train_best_model(models[best_name][1], X_train, Y_train)

    display_step("Step 10: Save and load the model")
    save_model(best_model, MODEL_PATH)
    loaded_model = load_model(MODEL_PATH)

    display_step("Step 11: Evaluate the best model on the test data")
    predictions, accuracy, recall, f1 = evaluate_model(
        loaded_model, X_test, Y_test
    )

    display_step("Step 12: Test scores of all 10 models")
    evaluate_all_models(models, X_train, Y_train, X_test, Y_test)

    display_step("Step 13: Plot the graphs")
    plot_model_comparison(results, best_name)
    plot_confusion_matrix(Y_test, predictions, best_name)
    plot_feature_importance(X_train, Y_train)

    display_step("Step 14: Detect cancer for patients of the test data")
    detect_cancer(loaded_model, X_test, Y_test, 5)

    display_step("Step 15: Project summary")
    display_summary(best_name, results, accuracy, recall, f1)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()