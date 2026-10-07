"""
------------------------------------------------------------
Project Name        : Wine Classification using KNN

Dataset Information :

Dataset File        : data/wine.csv (178 samples)

Features            : 13 chemical measurements of the wine
                      (Alcohol, Malic acid, Ash, Magnesium,
                       Flavanoids, Color intensity, Proline, etc.)

Target              : Class -> Grape cultivar (variety) of the wine
                      Class 1, Class 2 and Class 3

Machine Learning Information :

Algorithm Used      : K-Nearest Neighbors (KNN)

Techniques Used     : Feature Scaling (StandardScaler)
                      Hyperparameter Tuning of K
                      (5-fold Cross Validation)

Library             : Scikit-Learn

Problem Type        : Multi-class Classification

Author              : Ishwari Vijaykumar Surve

Date                : 07/10/2026
------------------------------------------------------------
"""


############################################################
# Required Python Packages
############################################################

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

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
    "wine.csv"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)


############################################################
# Headers
############################################################

TARGET_HEADER = "Class"


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

K_VALUES = range(1, 21)

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
# Date          : 07/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("            WINE CLASSIFICATION USING KNN")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the Wine dataset into pandas dataframe
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def load_data(path):

    df = pd.read_csv(path)

    print()
    print("Some entries from the dataset :")
    print(df.head())

    return df


############################################################
# Function Name : clean_data
# Description   : Remove empty rows from the dataset
# Input         : df - Input pandas dataframe
# Output        : Returns cleaned dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def clean_data(df):

    df = df.dropna()

    print()
    print("Total records :", df.shape[0])
    print("Total columns :", df.shape[1])

    return df


############################################################
# Function Name : split_features_target
# Description   : Separate independent and dependent variables
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
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
# Date          : 07/10/2026
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
# Function Name : scale_features
# Description   : Scale features using StandardScaler.
#                 The scaler learns only from the training data
#                 and the same scaling is applied to test data.
# Input         : X_train - Training features
#                 X_test - Testing features
# Output        : Returns scaled training and testing features
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print()
    print("Feature scaling is done")

    return X_train_scaled, X_test_scaled


############################################################
# Function Name : tune_k_value
# Description   : Find accuracy of every K value (1 to 20)
#                 using 5-fold cross validation on the
#                 training data only. The test data is not
#                 used for choosing K.
# Input         : X_train - Training features
#                 Y_train - Training target
# Output        : Returns list of accuracy for every K
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def tune_k_value(X_train, Y_train):

    folds = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    accuracy_scores = []

    print()
    print("K value  :  Cross Validation Accuracy")

    for k in K_VALUES:

        # Scaling is done inside every fold, so no data leaks
        pipeline = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k)
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            Y_train,
            cv=folds,
            scoring="accuracy"
        )

        accuracy_scores.append(scores.mean())

        print(f"K = {k:<4} :  {scores.mean():.4f}")

    return accuracy_scores


############################################################
# Function Name : plot_k_vs_accuracy
# Description   : Draw and save the K vs accuracy graph
# Input         : accuracy_scores - Accuracy of every K
#                 best_k - Best value of K
# Output        : Saves K vs accuracy image
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def plot_k_vs_accuracy(accuracy_scores, best_k):

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(K_VALUES),
        accuracy_scores,
        marker="o"
    )

    plt.axvline(
        best_k,
        color="red",
        linestyle="--",
        label=f"Best K = {best_k}"
    )

    plt.title("K value vs Cross Validation Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.xticks(list(K_VALUES))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "k_vs_accuracy.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/k_vs_accuracy.png")


############################################################
# Function Name : find_best_k
# Description   : Find the K value with the highest accuracy
# Input         : accuracy_scores - Accuracy of every K
# Output        : Returns best value of K
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def find_best_k(accuracy_scores):

    best_k = list(K_VALUES)[accuracy_scores.index(max(accuracy_scores))]

    print()
    print("Best value of K is :", best_k)
    print(f"Cross validation accuracy : {max(accuracy_scores):.4f}")

    return best_k


############################################################
# Function Name : train_final_model
# Description   : Build and train the final KNN model using
#                 the best value of K
# Input         : X_train_scaled - Scaled training features
#                 Y_train - Training target
#                 best_k - Best value of K
# Output        : Returns trained KNN model
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def train_final_model(X_train_scaled, Y_train, best_k):

    model = KNeighborsClassifier(n_neighbors=best_k)

    model.fit(X_train_scaled, Y_train)

    print()
    print("Final model trained with K =", best_k)

    return model


############################################################
# Function Name : evaluate_model
# Description   : Test the final model and calculate accuracy
# Input         : model - Trained KNN model
#                 X_test_scaled - Scaled testing features
#                 Y_test - Actual testing target
# Output        : Returns accuracy and predicted values
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def evaluate_model(model, X_test_scaled, Y_test):

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(Y_test, predictions)

    print()
    print(f"Accuracy of model on test data : {accuracy * 100:.2f} %")

    return accuracy, predictions


############################################################
# Function Name : plot_confusion_matrix
# Description   : Print and save the confusion matrix image
# Input         : Y_test - Actual target values
#                 predictions - Predicted target values
# Output        : Saves confusion matrix image
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def plot_confusion_matrix(Y_test, predictions):

    matrix = confusion_matrix(Y_test, predictions)

    class_names = [f"Class {label}" for label in sorted(Y_test.unique())]

    print()
    print(matrix)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix - KNN")
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
# Function Name : show_classification_report
# Description   : Display precision, recall and f1-score
# Input         : Y_test - Actual target values
#                 predictions - Predicted target values
# Output        : Displays classification report
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def show_classification_report(Y_test, predictions):

    print()
    print(classification_report(Y_test, predictions))


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("       WINE CLASSIFICATION USING KNN COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Wine Classification project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 07/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load the dataset from CSV file")
    df = load_data(DATA_PATH)

    display_step("Step 2: Clean the dataset by removing empty rows")
    df = clean_data(df)

    display_step("Step 3: Separate independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 4: Split the dataset for training and testing")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    display_step("Step 5: Feature scaling")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    display_step("Step 6: Hyperparameter tuning of K (5-fold cross validation)")
    accuracy_scores = tune_k_value(X_train, Y_train)

    display_step("Step 7: Find the best value of K")
    best_k = find_best_k(accuracy_scores)

    display_step("Step 8: Plot graph of K vs accuracy")
    plot_k_vs_accuracy(accuracy_scores, best_k)

    display_step("Step 9: Build the final model using the best value of K")
    model = train_final_model(X_train_scaled, Y_train, best_k)

    display_step("Step 10: Calculate the final accuracy")
    accuracy, predictions = evaluate_model(model, X_test_scaled, Y_test)

    display_step("Step 11: Display the confusion matrix")
    plot_confusion_matrix(Y_test, predictions)

    display_step("Step 12: Display the classification report")
    show_classification_report(Y_test, predictions)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()