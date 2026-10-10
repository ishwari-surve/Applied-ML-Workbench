"""
------------------------------------------------------------
Project Name        : Titanic Survival Prediction

Dataset Information :

Dataset File        : data/titanic.csv (1309 passengers,
                      891 passengers with a real survival label
                      are used for the model)

Features            : Age         -> Age of the passenger
                      Fare        -> Ticket fare
                      Sex         -> Sex of the passenger (0 / 1)
                      sibsp       -> Siblings or spouses on board
                      Parch       -> Parents or children on board
                      Pclass      -> Ticket class (1, 2, 3)
                      Embarked    -> Port of embarkation (0, 1, 2)

Target              : Survived    -> 0 = did not survive, 1 = survived

Machine Learning Information :

Algorithm Used      : Logistic Regression

Techniques Used     : Missing value handling (median / mode)
                      One-hot encoding of Embarked
                      Model saving and loading (joblib)

Library             : Scikit-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Accuracy
                      Confusion Matrix
                      Classification Report

Author              : Ishwari Vijaykumar Surve

Date                : 10/10/2026
------------------------------------------------------------
"""


############################################################
# Required Python Packages
############################################################

import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
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
    "titanic.csv"
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
    "titanic_model.pkl"
)


############################################################
# Headers
############################################################

TARGET_HEADER = "Survived"

DROP_COLUMNS = ["Passengerid", "zero", "Name", "Cabin"]

# Passengers with Passengerid above 891 have Survived = 0 for every
# row (no real label), so only the first 891 passengers are used.
LAST_LABELED_PASSENGER = 891


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

MAX_ITERATIONS = 1000


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
# Date          : 10/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("            TITANIC SURVIVAL PREDICTION")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the Titanic dataset and show basic
#                 information about it.
#                 It checks that the file and the required
#                 columns (Passengerid and Survived) exist.
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def load_data(path):

    # Check that the dataset file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at : {path}\n"
            "Please place the CSV file at data/titanic.csv"
        )

    df = pd.read_csv(path)

    # Check that the columns needed by the program exist
    for column in ["Passengerid", TARGET_HEADER]:
        if column not in df.columns:
            raise ValueError(f"The dataset must contain the '{column}' column")

    print()
    print("First five rows of the dataset :")
    print(df.head())

    print()
    print("Shape of the dataset :", df.shape)

    print()
    print("Column names :", df.columns.tolist())

    print()
    print("Missing values in each column :")
    print(df.isnull().sum())

    return df


############################################################
# Function Name : remove_unlabeled_passengers
# Description   : Keep only the passengers that have a real
#                 survival label.
#                 In this file every passenger after Passengerid
#                 891 (418 passengers) has Survived = 0, which is
#                 a placeholder and not a real result.
# Input         : df - Input pandas dataframe
# Output        : Returns dataframe of labeled passengers
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def remove_unlabeled_passengers(df):

    after_id = df[df["Passengerid"] > LAST_LABELED_PASSENGER]

    print()
    print("Passengers after Passengerid", LAST_LABELED_PASSENGER, ":", len(after_id))
    print("Their Survived values :", after_id[TARGET_HEADER].unique().tolist())

    df = df[df["Passengerid"] <= LAST_LABELED_PASSENGER].copy()

    print()
    print("Passengers kept for the model :", len(df))

    print()
    print("Survived count after removal :")
    print(df[TARGET_HEADER].value_counts())

    return df


############################################################
# Function Name : remove_columns
# Description   : Remove the unnecessary columns
# Input         : df - Input pandas dataframe
# Output        : Returns dataframe without unnecessary columns
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def remove_columns(df):

    existing_columns = [
        column for column in DROP_COLUMNS
        if column in df.columns
    ]

    print()
    print("Columns to be dropped :", existing_columns)

    df = df.drop(columns=existing_columns)

    print()
    print("Data after column removal :")
    print(df.head())

    return df


############################################################
# Function Name : handle_missing_values
# Description   : Fill missing values.
#                 Age and Fare  -> median value
#                 Embarked      -> most frequent value
# Input         : df - Input pandas dataframe
# Output        : Returns dataframe without missing values
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def handle_missing_values(df):

    # Age and Fare : invalid values become NaN, then median is used
    for column in ["Age", "Fare"]:

        if column in df.columns:

            df[column] = pd.to_numeric(df[column], errors="coerce")

            median_value = df[column].median()

            df[column] = df[column].fillna(median_value)

            print()
            print(f"{column} : missing values filled with median {median_value}")

    # Embarked : missing values are filled with the most frequent value
    if "Embarked" in df.columns:

        mode_value = df["Embarked"].mode()[0]

        df["Embarked"] = df["Embarked"].fillna(mode_value)

        df["Embarked"] = df["Embarked"].astype(int)

        print()
        print(f"Embarked : missing values filled with mode {mode_value}")

    print()
    print("Missing values after handling :")
    print(df.isnull().sum())

    return df


############################################################
# Function Name : encode_data
# Description   : One-hot encode the Embarked column
# Input         : df - Input pandas dataframe
# Output        : Returns encoded dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def encode_data(df):

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    print()
    print("Shape of the dataset :", df.shape)

    print()
    print("Data after encoding :")
    print(df.head())

    return df


############################################################
# Function Name : split_features_target
# Description   : Separate independent and dependent variables
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def split_features_target(df):

    X = df.drop(columns=[TARGET_HEADER])

    Y = df[TARGET_HEADER]

    print()
    print("Shape of X :", X.shape)
    print("Shape of Y :", Y.shape)

    print()
    print("Input columns :", X.columns.tolist())
    print("Output column :", TARGET_HEADER)

    return X, Y


############################################################
# Function Name : split_dataset
# Description   : Split dataset into training and testing data
# Input         : X - Features
#                 Y - Target
# Output        : Returns training and testing datasets
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
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
# Function Name : train_model
# Description   : Create and train the Logistic Regression model
# Input         : X_train - Training features
#                 Y_train - Training target
# Output        : Returns trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def train_model(X_train, Y_train):

    model = LogisticRegression(max_iter=MAX_ITERATIONS)

    model.fit(X_train, Y_train)

    print()
    print("Model trained successfully")

    return model


############################################################
# Function Name : show_coefficients
# Description   : Display model coefficients and intercept
# Input         : model - Trained model
#                 feature_names - Names of the features
# Output        : Displays coefficient of every feature
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def show_coefficients(model, feature_names):

    print()
    print("Model coefficients :")

    for feature, coefficient in zip(feature_names, model.coef_[0]):
        print(f"{feature:<12}: {coefficient:.4f}")

    print()
    print(f"Intercept : {model.intercept_[0]:.4f}")


############################################################
# Function Name : save_model
# Description   : Save the trained model on secondary storage
# Input         : model - Trained model
#                 path - Path of the model file
# Output        : Saves the model file in models folder
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def save_model(model, path):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(model, path)

    print()
    print("Model saved successfully : models/titanic_model.pkl")


############################################################
# Function Name : load_model
# Description   : Load the saved model from secondary storage
# Input         : path - Path of the model file
# Output        : Returns loaded model
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def load_model(path):

    model = joblib.load(path)

    print()
    print("Model loaded successfully : models/titanic_model.pkl")

    return model


############################################################
# Function Name : evaluate_model
# Description   : Test the loaded model on the test data.
#                 It shows accuracy, the baseline accuracy
#                 (always predicting the most common class),
#                 confusion matrix and classification report.
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
# Output        : Returns accuracy, baseline accuracy and predictions
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def evaluate_model(model, X_test, Y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)

    baseline = Y_test.value_counts(normalize=True).max()

    print()
    print(f"Accuracy of model on test data      : {accuracy * 100:.2f} %")
    print(f"Baseline accuracy (always 'No')     : {baseline * 100:.2f} %")

    print()
    print("Confusion matrix :")
    print(confusion_matrix(Y_test, predictions))

    print()
    print(classification_report(
        Y_test,
        predictions,
        target_names=["Not survived", "Survived"]
    ))

    return accuracy, baseline, predictions


############################################################
# Function Name : plot_survival_count
# Description   : Draw and save the survival count graph
# Input         : df - Cleaned pandas dataframe
# Output        : Saves survival count image
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def plot_survival_count(df):

    plt.figure(figsize=(6, 4))

    sns.countplot(x=TARGET_HEADER, data=df)

    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "survival_count.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/survival_count.png")


############################################################
# Function Name : plot_age_distribution
# Description   : Draw and save the age distribution graph
# Input         : df - Cleaned pandas dataframe
# Output        : Saves age distribution image
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def plot_age_distribution(df):

    plt.figure(figsize=(6, 4))

    sns.histplot(df["Age"], bins=20, kde=True)

    plt.title("Age Distribution of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "age_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/age_distribution.png")


############################################################
# Function Name : plot_confusion_matrix
# Description   : Draw and save the confusion matrix image
# Input         : Y_test - Actual target values
#                 predictions - Predicted target values
# Output        : Saves confusion matrix image
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def plot_confusion_matrix(Y_test, predictions):

    matrix = confusion_matrix(Y_test, predictions)

    class_names = ["Not survived", "Survived"]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix - Logistic Regression")
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
# Function Name : display_summary
# Description   : Display the final project summary
# Input         : accuracy - Accuracy on test data
#                 baseline - Baseline accuracy
# Output        : Displays project summary
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def display_summary(accuracy, baseline):

    print()
    print("Algorithm Used     : Logistic Regression")
    print(f"Test accuracy      : {accuracy * 100:.2f} %")
    print(f"Baseline accuracy  : {baseline * 100:.2f} %")
    print("Problem Type       : Binary Classification")


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("       TITANIC SURVIVAL PREDICTION COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Titanic Survival project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 10/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load the dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Remove passengers without a real survival label")
    df = remove_unlabeled_passengers(df)

    display_step("Step 3: Remove unnecessary columns")
    df = remove_columns(df)

    display_step("Step 4: Handle missing values")
    df = handle_missing_values(df)

    display_step("Step 5: Encode the categorical column")
    df = encode_data(df)

    display_step("Step 6: Separate independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 7: Split the dataset for training and testing")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    display_step("Step 8: Train the Logistic Regression model")
    model = train_model(X_train, Y_train)

    display_step("Step 9: Show model coefficients")
    show_coefficients(model, X.columns)

    display_step("Step 10: Save the model")
    save_model(model, MODEL_PATH)

    display_step("Step 11: Load the saved model")
    loaded_model = load_model(MODEL_PATH)

    display_step("Step 12: Evaluate the model")
    accuracy, baseline, predictions = evaluate_model(
        loaded_model, X_test, Y_test
    )

    display_step("Step 13: Plot the graphs")
    plot_survival_count(df)
    plot_age_distribution(df)
    plot_confusion_matrix(Y_test, predictions)

    display_step("Step 14: Project summary")
    display_summary(accuracy, baseline)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()