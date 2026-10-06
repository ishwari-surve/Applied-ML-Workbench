"""
------------------------------------------------------------
Project Name        : Advertising Sales Prediction

Dataset Information :

Dataset File        : data/advertising.csv (200 samples)

Features            : TV        -> Advertising budget spent on TV
                      radio     -> Advertising budget spent on radio
                      newspaper -> Advertising budget spent on newspaper

Target              : sales     -> Sales of the product

Machine Learning Information :

Algorithm Used      : Multiple Linear Regression

Library             : Scikit-Learn

Problem Type        : Regression

Evaluation Metrics  : Mean Squared Error (MSE)
                      Root Mean Squared Error (RMSE)
                      R Square (R2)

Author              : Ishwari Vijaykumar Surve

Date                : 06/10/2026
------------------------------------------------------------
"""


############################################################
# Required Python Packages
############################################################

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


############################################################
# File Paths
############################################################

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "advertising.csv"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)


############################################################
# Headers
############################################################

FEATURE_HEADERS = ["TV", "radio", "newspaper"]

TARGET_HEADER = "sales"


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42


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
# Date          : 06/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("         ADVERTISING SALES PREDICTION")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the advertising dataset into dataframe
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def load_data(path):

    df = pd.read_csv(path)

    print()
    print("Few records from the dataset :")
    print(df.head())

    print()
    print("Last records from the dataset :")
    print(df.tail())

    return df


############################################################
# Function Name : clean_data
# Description   : Remove the unwanted index column
# Input         : df - Input pandas dataframe
# Output        : Returns cleaned dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def clean_data(df):

    print()
    print("Shape of dataset before removal :", df.shape)

    # Remove the unnamed index column if it exists
    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

    print("Shape of dataset after removal  :", df.shape)

    print()
    print("Clean dataset is :")
    print(df.head())

    print()
    print(df.tail())

    return df


############################################################
# Function Name : check_missing_values
# Description   : Check missing values in the dataset
# Input         : df - Input pandas dataframe
# Output        : Displays missing values for each column
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def check_missing_values(df):

    print()
    print("Missing values count :")
    print(df.isnull().sum())


############################################################
# Function Name : show_statistics
# Description   : Display statistical information
# Input         : df - Input pandas dataframe
# Output        : Displays descriptive statistics
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def show_statistics(df):

    print()
    print("Dataset statistics :")
    print(df.describe())


############################################################
# Function Name : show_correlation
# Description   : Display correlation matrix and save heatmap
# Input         : df - Input pandas dataframe
# Output        : Displays matrix, saves heatmap image
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def show_correlation(df):

    correlation = df.corr()

    print()
    print("Correlation matrix :")
    print(correlation)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/correlation_heatmap.png")


############################################################
# Function Name : split_features_target
# Description   : Separate input features and target variable
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def split_features_target(df):

    X = df[FEATURE_HEADERS]

    Y = df[TARGET_HEADER]

    print()
    print("Shape of independent variables :", X.shape)
    print("Shape of dependent variable    :", Y.shape)

    return X, Y


############################################################
# Function Name : split_dataset
# Description   : Split dataset into training and testing data
# Input         : X - Features
#                 Y - Target
# Output        : Returns training and testing datasets
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def split_dataset(X, Y):

    train_x, test_x, train_y, test_y = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print()
    print("X_train shape :", train_x.shape)
    print("X_test shape  :", test_x.shape)
    print("Y_train shape :", train_y.shape)
    print("Y_test shape  :", test_y.shape)

    return train_x, test_x, train_y, test_y


############################################################
# Function Name : train_model
# Description   : Create and train the Multiple Linear Regression model
# Input         : train_x - Training features
#                 train_y - Training target
# Output        : Returns trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def train_model(train_x, train_y):

    model = LinearRegression()

    model.fit(train_x, train_y)

    print()
    print("Model training completed")

    return model


############################################################
# Function Name : predict_sales
# Description   : Predict sales for the testing data
# Input         : model - Trained model
#                 test_x - Testing features
# Output        : Returns predicted sales
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def predict_sales(model, test_x):

    predictions = model.predict(test_x)

    print()
    print("Prediction completed for", len(predictions), "test samples")

    return predictions


############################################################
# Function Name : evaluate_model
# Description   : Calculate MSE, RMSE and R Square
# Input         : test_y - Actual sales
#                 predictions - Predicted sales
# Output        : Returns MSE, RMSE and R2
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def evaluate_model(test_y, predictions):

    mse = mean_squared_error(test_y, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(test_y, predictions)

    print()
    print("Mean Squared Error      :", round(mse, 4))
    print("Root Mean Squared Error :", round(rmse, 4))
    print("R Square Value          :", round(r2, 4))

    return mse, rmse, r2


############################################################
# Function Name : show_coefficients
# Description   : Display model coefficients and intercept
# Input         : model - Trained model
#                 feature_names - Names of the features
# Output        : Displays coefficient of every feature
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def show_coefficients(model, feature_names):

    print()
    print("Model coefficients :")

    for column, value in zip(feature_names, model.coef_):
        print(f"{column:<10}: {value:.4f}")

    print()
    print(f"Intercept : {model.intercept_:.4f}")


############################################################
# Function Name : compare_actual_predicted
# Description   : Display actual and predicted sales together
# Input         : test_y - Actual sales
#                 predictions - Predicted sales
# Output        : Displays first five records
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def compare_actual_predicted(test_y, predictions):

    result = pd.DataFrame({
        "Actual sale": test_y.values,
        "Predicted sale": predictions
    })

    print()
    print(result.head())


############################################################
# Function Name : plot_actual_vs_predicted
# Description   : Draw and save actual vs predicted sales plot
# Input         : test_y - Actual sales
#                 predictions - Predicted sales
# Output        : Saves actual vs predicted image
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def plot_actual_vs_predicted(test_y, predictions):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        test_y,
        predictions
    )

    # Ideal line: predicted value equals actual value
    low = min(test_y.min(), predictions.min())
    high = max(test_y.max(), predictions.max())

    plt.plot(
        [low, high],
        [low, high],
        color="red",
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Actual sales vs Predicted sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "actual_vs_predicted.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/actual_vs_predicted.png")


############################################################
# Function Name : predict_new_sales
# Description   : Predict sales for a new advertising budget
# Input         : model - Trained model
#                 tv - Budget spent on TV
#                 radio - Budget spent on radio
#                 newspaper - Budget spent on newspaper
# Output        : Returns predicted sales
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def predict_new_sales(model, tv, radio, newspaper):

    sample = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=FEATURE_HEADERS
    )

    prediction = model.predict(sample)[0]

    print()
    print(
        f"TV = {tv}, radio = {radio}, newspaper = {newspaper}"
        f"  ->  Predicted sales : {prediction:.2f}"
    )

    return prediction


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("     ADVERTISING SALES PREDICTION COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete Advertising Sales project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 06/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Remove unwanted columns")
    df = clean_data(df)

    display_step("Step 3: Check missing values")
    check_missing_values(df)

    display_step("Step 4: Display statistical summary")
    show_statistics(df)

    display_step("Step 5: Correlation between columns")
    show_correlation(df)

    display_step("Step 6: Split data into independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 7: Split dataset for training and testing")
    train_x, test_x, train_y, test_y = split_dataset(X, Y)

    display_step("Step 8: Create and train the Multiple Linear Regression model")
    model = train_model(train_x, train_y)

    display_step("Step 9: Test the model")
    predictions = predict_sales(model, test_x)

    display_step("Step 10: Evaluate the model")
    evaluate_model(test_y, predictions)

    display_step("Step 11: Calculate model coefficients")
    show_coefficients(model, X.columns)

    display_step("Step 12: Compare the actual and predicted values")
    compare_actual_predicted(test_y, predictions)

    display_step("Step 13: Plot actual vs predicted")
    plot_actual_vs_predicted(test_y, predictions)

    display_step("Step 14: Predict sales for new advertising budgets")
    predict_new_sales(model, 200, 40, 30)
    predict_new_sales(model, 50, 10, 20)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()