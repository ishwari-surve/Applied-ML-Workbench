"""
------------------------------------------------------------
Project Name        : Iris Classification

Dataset Information :

Dataset File        : data/iris.csv (150 samples)

Species Encoding    : Setosa     = 0
                      Versicolor = 1
                      Virginica  = 2

Features            : Sepal Length (cm)
                      Sepal Width (cm)
                      Petal Length (cm)
                      Petal Width (cm)

Target              : Species of the Iris flower

Machine Learning Information :

Algorithms Used     : K-Nearest Neighbors (KNN)
                      Decision Tree Classifier

Library             : Scikit-Learn

Problem Type        : Multi-class Classification

Author              : Ishwari Vijaykumar Surve

Date                : 05/10/2026
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
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
    "iris.csv"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)


############################################################
# Headers
############################################################

FEATURE_HEADERS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

TARGET_HEADER = "species"

ENCODED_TARGET_HEADER = "species_encoded"


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

KNN_NEIGHBORS = 5


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
# Date          : 05/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("             IRIS CLASSIFICATION")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the Iris dataset into pandas dataframe
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def load_data(path):

    df = pd.read_csv(path)

    print()
    print("First Five Records :")
    print(df.head())

    return df


############################################################
# Function Name : clean_data
# Description   : Clean and standardize dataset columns
# Input         : df - Input pandas dataframe
# Output        : Returns cleaned dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def clean_data(df):

    # Remove unnecessary columns if they exist
    df = df.drop(
        columns=["Unnamed: 0", "Id"],
        errors="ignore"
    )

    # Remove extra spaces from column names
    df.columns = [
        " ".join(column.split())
        for column in df.columns
    ]

    return df


############################################################
# Function Name : check_missing_values
# Description   : Check missing values in the dataset
# Input         : df - Input pandas dataframe
# Output        : Displays missing values for each column
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def check_missing_values(df):

    print()
    print("Missing Values :")
    print(df.isnull().sum())


############################################################
# Function Name : show_statistics
# Description   : Display statistical information
# Input         : df - Input pandas dataframe
# Output        : Displays descriptive statistics
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def show_statistics(df):

    print()
    print("Dataset Statistics :")
    print(df.describe())


############################################################
# Function Name : show_class_distribution
# Description   : Display number of records for each species
# Input         : df - Input pandas dataframe
# Output        : Displays class distribution
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def show_class_distribution(df):

    print()
    print("Class Distribution :")
    print(df[TARGET_HEADER].value_counts())


############################################################
# Function Name : encode_target
# Description   : Encode species names into numerical labels
# Input         : df - Input pandas dataframe
# Output        : Returns dataframe and label encoder
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def encode_target(df):

    encoder = LabelEncoder()

    df[ENCODED_TARGET_HEADER] = encoder.fit_transform(
        df[TARGET_HEADER]
    )

    print()
    print("Species Classes :")
    print(encoder.classes_)

    print()
    print("Encoded Species :")
    print(df[
        [TARGET_HEADER, ENCODED_TARGET_HEADER]
    ].head())

    return df, encoder


############################################################
# Function Name : split_features_target
# Description   : Separate input features and target variable
# Input         : df - Input pandas dataframe
# Output        : Returns feature dataframe and target series
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def split_features_target(df):

    X = df[FEATURE_HEADERS]

    Y = df[ENCODED_TARGET_HEADER]

    return X, Y


############################################################
# Function Name : split_dataset
# Description   : Split dataset into training and testing data
# Input         : X - Features
#                 Y - Target
# Output        : Returns training and testing datasets
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def split_dataset(X, Y):

    train_x, test_x, train_y, test_y = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )

    print()
    print("Training Data Size :", len(train_x))
    print("Testing Data Size  :", len(test_x))

    return train_x, test_x, train_y, test_y


############################################################
# Function Name : train_knn
# Description   : Train K-Nearest Neighbors classification model
# Input         : train_x - Training features
#                 train_y - Training target
# Output        : Returns trained KNN model
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def train_knn(train_x, train_y):

    model = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS
    )

    model.fit(train_x, train_y)

    return model


############################################################
# Function Name : train_decision_tree
# Description   : Train Decision Tree classification model
# Input         : train_x - Training features
#                 train_y - Training target
# Output        : Returns trained Decision Tree model
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def train_decision_tree(train_x, train_y):

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(train_x, train_y)

    return model


############################################################
# Function Name : evaluate_model
# Description   : Evaluate trained model using test data
# Input         : model - Trained machine learning model
#                 test_x - Testing features
#                 test_y - Actual testing target
# Output        : Returns accuracy and predicted values
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def evaluate_model(model, test_x, test_y):

    predictions = model.predict(test_x)

    accuracy = accuracy_score(
        test_y,
        predictions
    )

    return accuracy, predictions


############################################################
# Function Name : compare_models
# Description   : Compare accuracy of KNN and Decision Tree
# Input         : knn_accuracy - KNN accuracy
#                 dt_accuracy - Decision Tree accuracy
# Output        : Displays model comparison
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def compare_models(knn_accuracy, dt_accuracy):

    print()
    print("Model Accuracy Comparison")
    print(BORDER)

    print(
        "KNN Accuracy          : "
        f"{knn_accuracy:.4f}"
    )

    print(
        "Decision Tree Accuracy: "
        f"{dt_accuracy:.4f}"
    )

    if knn_accuracy > dt_accuracy:

        print()
        print("Best Model : K-Nearest Neighbors (KNN)")

    elif dt_accuracy > knn_accuracy:

        print()
        print("Best Model : Decision Tree")

    else:

        print()
        print("Best Model : Both models have same accuracy")


############################################################
# Function Name : show_classification_report
# Description   : Display classification report
# Input         : test_y - Actual target values
#                 predictions - Predicted target values
#                 class_names - Names of target classes
# Output        : Displays precision, recall and F1-score
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def show_classification_report(
    test_y,
    predictions,
    class_names
):

    print()
    print("Classification Report")
    print(BORDER)

    print(
        classification_report(
            test_y,
            predictions,
            target_names=class_names
        )
    )


############################################################
# Function Name : plot_confusion_matrix
# Description   : Generate and save confusion matrix
# Input         : test_y - Actual target values
#                 predictions - Predicted target values
#                 class_names - Names of target classes
#                 title - Plot title
#                 color_map - Seaborn color map
#                 file_name - Output image file name
# Output        : Saves confusion matrix image
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def plot_confusion_matrix(
    test_y,
    predictions,
    class_names,
    title,
    color_map,
    file_name
):

    matrix = confusion_matrix(
        test_y,
        predictions
    )

    print()
    print(title)
    print(matrix)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=color_map,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        IMAGES_DIR,
        file_name
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/" + file_name)


############################################################
# Function Name : plot_accuracy_comparison
# Description   : Generate and save model accuracy comparison
# Input         : knn_accuracy - KNN accuracy
#                 dt_accuracy - Decision Tree accuracy
# Output        : Saves accuracy comparison image
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def plot_accuracy_comparison(
    knn_accuracy,
    dt_accuracy
):

    models = [
        "KNN",
        "Decision Tree"
    ]

    accuracies = [
        knn_accuracy,
        dt_accuracy
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        models,
        accuracies
    )

    plt.title(
        "Model Accuracy Comparison"
    )

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.ylim(
        0,
        1
    )

    for index, accuracy in enumerate(accuracies):

        plt.text(
            index,
            accuracy + 0.02,
            f"{accuracy:.2f}",
            ha="center"
        )

    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        IMAGES_DIR,
        "accuracy_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/accuracy_comparison.png")


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("       IRIS CLASSIFICATION COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Iris Classification project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 05/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Remove unwanted columns")
    df = clean_data(df)

    display_step("Step 3: Check missing values")
    check_missing_values(df)

    display_step("Step 4: Statistical summary")
    show_statistics(df)

    display_step("Step 5: Class distribution")
    show_class_distribution(df)

    display_step("Step 6: Encode target variable")
    df, encoder = encode_target(df)

    display_step("Step 7: Split into independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 8: Split into training and testing sets (80/20)")
    train_x, test_x, train_y, test_y = split_dataset(X, Y)

    display_step("Step 9: Build and train KNN model")
    knn_model = train_knn(train_x, train_y)
    knn_accuracy, knn_predictions = evaluate_model(knn_model, test_x, test_y)
    print(f"KNN Accuracy : {knn_accuracy * 100:.2f} %")

    display_step("Step 10: Build and train Decision Tree model")
    dt_model = train_decision_tree(train_x, train_y)
    dt_accuracy, dt_predictions = evaluate_model(dt_model, test_x, test_y)
    print(f"Decision Tree Accuracy : {dt_accuracy * 100:.2f} %")

    display_step("Step 11: Model comparison")
    compare_models(knn_accuracy, dt_accuracy)

    display_step("Step 12: Classification report (KNN)")
    show_classification_report(test_y, knn_predictions, encoder.classes_)

    display_step("Step 13: Confusion matrix - KNN")
    plot_confusion_matrix(
        test_y, knn_predictions, encoder.classes_,
        "Confusion Matrix - KNN", "Blues", "confusion_matrix_knn.png"
    )

    display_step("Step 14: Confusion matrix - Decision Tree")
    plot_confusion_matrix(
        test_y, dt_predictions, encoder.classes_,
        "Confusion Matrix - Decision Tree", "Greens",
        "confusion_matrix_decision_tree.png"
    )

    display_step("Step 15: Model accuracy comparison")
    plot_accuracy_comparison(knn_accuracy, dt_accuracy)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()
