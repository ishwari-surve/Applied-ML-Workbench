"""
------------------------------------------------------------
Project Name        : California Housing Price Prediction
                      using Ensemble Learning

Dataset Information :

Dataset File        : data/california_housing.csv
                      (20640 neighborhoods)

Features            : MedInc     -> Median income of the area
                      HouseAge   -> Median age of the houses
                      AveRooms   -> Average number of rooms
                      AveBedrms  -> Average number of bedrooms
                      Population -> People living in the area
                      AveOccup   -> Average people per house
                      Latitude   -> Latitude of the area
                      Longitude  -> Longitude of the area

Target              : target -> Median house value
                                (in units of 100,000 dollars)

Machine Learning Information :

Models Compared     : Single models -> Linear Regression,
                                       Decision Tree, SVR
                                       (baseline for comparison)
                      Bagging       -> Bagging Regressor,
                                       Random Forest
                      Boosting      -> Gradient Boosting

Techniques Used     : Feature Scaling (StandardScaler)
                      5-fold Cross Validation
                      Best ensemble selected by cross validation R2
                      Model saving and loading (joblib)

Library             : Scikit-Learn

Problem Type        : Regression

Evaluation Metrics  : R2 Score, RMSE, MAE

Author              : Ishwari Vijaykumar Surve

Date                : 15/10/2026
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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
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
    "california_housing.csv"
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
    "california_housing_model.pkl"
)


############################################################
# Headers
############################################################

TARGET_HEADER = "target"

# The price is given in units of 100,000 dollars
PRICE_UNIT = 100000

# Prices are cut off at 5.0 (500,000 dollars) in this dataset
PRICE_CAP = 5.0


############################################################
# Machine Learning Parameters
############################################################

TEST_SIZE = 0.2

RANDOM_STATE = 42

CROSS_VALIDATION_FOLDS = 5

BAGGING_TREES = 50

FOREST_TREES = 100


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
# Date          : 15/10/2026
############################################################

def display_title():

    print()
    print("=" * 60)
    print("   CALIFORNIA HOUSING PRICE PREDICTION - ENSEMBLE")
    print("=" * 60)
    print()


############################################################
# Function Name : display_step
# Description   : Display processing step title
# Input         : title - Name of processing step
# Output        : Displays processing step
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def display_step(title):

    print()
    print(BORDER)
    print(title)
    print(BORDER)


############################################################
# Function Name : load_data
# Description   : Read the California housing dataset. It checks
#                 that the file and the target column exist.
# Input         : path - Path of CSV file
# Output        : Returns pandas dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def load_data(path):

    # Check that the dataset file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at : {path}\n"
            "Please place the CSV file at data/california_housing.csv"
        )

    df = pd.read_csv(path)

    # Check that the target column exists
    if TARGET_HEADER not in df.columns:
        raise ValueError(f"The dataset must contain the '{TARGET_HEADER}' column")

    print()
    print("First five rows of the dataset :")
    print(df.head())

    print()
    print("Shape of the dataset :", df.shape)

    return df


############################################################
# Function Name : check_data
# Description   : Display missing values, duplicate rows and
#                 the number of prices that are cut off at 5.0
# Input         : df - Input pandas dataframe
# Output        : Displays data quality information
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def check_data(df):

    capped = int((df[TARGET_HEADER] >= PRICE_CAP).sum())

    print()
    print("Missing values (empty cells) :", int(df.isnull().sum().sum()))
    print("Duplicate rows               :", int(df.duplicated().sum()))

    print()
    print(f"Areas with the price cut off at {PRICE_CAP} (500,000 dollars) : "
          f"{capped} ({capped / len(df) * 100:.1f} %)")
    print("The real price of these areas may be higher.")


############################################################
# Function Name : show_statistics
# Description   : Display the statistical summary of the dataset
# Input         : df - Input pandas dataframe
# Output        : Displays count, mean, std, min and max
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def show_statistics(df):

    print()
    print("Statistical summary :")
    print(df.describe().T[["count", "mean", "std", "min", "max"]].round(2))


############################################################
# Function Name : show_correlation
# Description   : Display the correlation of every column
#                 with the house price
# Input         : df - Input pandas dataframe
# Output        : Displays correlation with the price
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def show_correlation(df):

    correlation = df.corr()[TARGET_HEADER].drop(TARGET_HEADER)

    print()
    print("Correlation with the house price :")
    print(correlation.sort_values(ascending=False).round(2))


############################################################
# Function Name : split_features_target
# Description   : Separate independent and dependent variables
# Input         : df - Input pandas dataframe
# Output        : Returns features (X) and target (Y)
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
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
# Date          : 15/10/2026
############################################################

def split_dataset(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print()
    print("X_train shape :", X_train.shape)
    print("X_test shape  :", X_test.shape)
    print("Y_train shape :", Y_train.shape)
    print("Y_test shape  :", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


############################################################
# Function Name : build_models
# Description   : Build the 6 models.
#                 Linear Regression and SVR are scaled.
#                 The 3 single models are the baseline.
# Input         : Nothing
# Output        : Returns dictionary of model name and
#                 (model type, model)
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def build_models():

    models = {
        "Linear Regression": ("Single", make_pipeline(
            StandardScaler(),
            LinearRegression()
        )),
        "Decision Tree": ("Single", DecisionTreeRegressor(
            random_state=RANDOM_STATE
        )),
        "SVR": ("Single", make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=1.0)
        )),
        "Bagging": ("Bagging", BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
            n_estimators=BAGGING_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        "Random Forest": ("Bagging", RandomForestRegressor(
            n_estimators=FOREST_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        "Gradient Boosting": ("Boosting", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        ))
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
# Input         : models - Dictionary of models
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns dataframe of cross validation scores
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def compare_models(models, X_train, Y_train):

    folds = KFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    }

    rows = []

    for name, (kind, model) in models.items():

        print(f"Cross validation running : {name}")

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
            "CV R2": scores["test_r2"].mean(),
            "CV RMSE": -scores["test_rmse"].mean(),
            "CV MAE": -scores["test_mae"].mean()
        })

    results = pd.DataFrame(rows).set_index("Model").round(4)

    print()
    print("Cross validation scores (RMSE and MAE are in 100,000 dollars) :")
    print(results)

    return results


############################################################
# Function Name : select_best_model
# Description   : Select the best ENSEMBLE model (Bagging or
#                 Boosting) with the highest cross validation
#                 R2 score. The single models are the baseline.
# Input         : results - Cross validation scores
# Output        : Returns name of the best ensemble model
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def select_best_model(results):

    ensembles = results[results["Type"] != "Single"]

    best_name = ensembles["CV R2"].idxmax()

    singles = results[results["Type"] == "Single"]

    best_single = singles["CV R2"].idxmax()

    print()
    print("Best ensemble model            :", best_name)
    print("Type                           :", results.loc[best_name, "Type"])
    print(f"Cross validation R2            : {results.loc[best_name, 'CV R2']:.4f}")

    print()
    print("Best single model (baseline)   :", best_single)
    print(f"Cross validation R2            : {results.loc[best_single, 'CV R2']:.4f}")

    gain = results.loc[best_name, "CV R2"] - singles["CV R2"].min()

    print()
    print(f"The best ensemble has an R2 score {gain:.4f} higher "
          "than the weakest single model.")

    return best_name


############################################################
# Function Name : train_best_model
# Description   : Train the best model on the training data
# Input         : model - Best model
#                 X_train - Training features
#                 Y_train - Training target
# Output        : Returns trained model
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
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
# Date          : 15/10/2026
############################################################

def save_model(model, path):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    joblib.dump(model, path, compress=3)

    print()
    print("Model saved successfully : models/california_housing_model.pkl")


############################################################
# Function Name : load_model
# Description   : Load the saved model from secondary storage
# Input         : path - Path of the model file
# Output        : Returns loaded model
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def load_model(path):

    model = joblib.load(path)

    print("Model loaded successfully : models/california_housing_model.pkl")

    return model


############################################################
# Function Name : evaluate_model
# Description   : Test the loaded model once on the test data.
#                 It shows R2, RMSE and MAE and the baseline
#                 (always predicting the average price).
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
# Output        : Returns predictions, R2, RMSE and MAE
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def evaluate_model(model, X_test, Y_test):

    predictions = model.predict(X_test)

    r2 = r2_score(Y_test, predictions)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    mae = mean_absolute_error(Y_test, predictions)
    baseline_mae = mean_absolute_error(
        Y_test,
        np.full(len(Y_test), Y_test.mean())
    )

    print()
    print(f"R2 score on test data              : {r2:.4f}")
    print(f"RMSE on test data                  : {rmse:.4f}  "
          f"(about {rmse * PRICE_UNIT:,.0f} dollars)")
    print(f"MAE on test data                   : {mae:.4f}  "
          f"(about {mae * PRICE_UNIT:,.0f} dollars)")
    print(f"Baseline MAE (always the average)  : {baseline_mae:.4f}  "
          f"(about {baseline_mae * PRICE_UNIT:,.0f} dollars)")

    return predictions, r2, rmse, mae


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
# Date          : 15/10/2026
############################################################

def evaluate_all_models(models, X_train, Y_train, X_test, Y_test):

    rows = []

    for name, (kind, model) in models.items():

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        rows.append({
            "Model": name,
            "Type": kind,
            "Test R2": r2_score(Y_test, predictions),
            "Test RMSE": np.sqrt(mean_squared_error(Y_test, predictions)),
            "Test MAE": mean_absolute_error(Y_test, predictions)
        })

    table = pd.DataFrame(rows).set_index("Model").round(4)

    print()
    print("Test scores of all models - for reporting only :")
    print(table)


############################################################
# Function Name : plot_model_comparison
# Description   : Draw and save the comparison of the 6 models
#                 (cross validation R2 and RMSE)
# Input         : results - Cross validation scores
#                 best_name - Name of the best model
# Output        : Saves model comparison image
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def plot_model_comparison(results, best_name):

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(results.index, results["CV R2"])
    axes[0].set_title("R2 score (higher is better)")
    axes[0].set_xlabel("Cross validation R2")
    axes[0].set_xlim(0, 1)

    axes[1].barh(results.index, results["CV RMSE"], color="tab:orange")
    axes[1].set_title("RMSE (lower is better)")
    axes[1].set_xlabel("Cross validation RMSE (100,000 dollars)")

    figure.suptitle(f"Model Comparison (5-fold cross validation) - "
                    f"Best ensemble: {best_name}")
    figure.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    figure.savefig(
        os.path.join(IMAGES_DIR, "model_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print()
    print("Image Saved : images/model_comparison.png")


############################################################
# Function Name : plot_actual_vs_predicted
# Description   : Draw and save actual vs predicted price plot
# Input         : Y_test - Actual prices
#                 predictions - Predicted prices
#                 best_name - Name of the best model
# Output        : Saves actual vs predicted image
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def plot_actual_vs_predicted(Y_test, predictions, best_name):

    plt.figure(figsize=(7, 6))

    plt.scatter(
        Y_test,
        predictions,
        s=6,
        alpha=0.4
    )

    plt.plot(
        [0, PRICE_CAP],
        [0, PRICE_CAP],
        color="red",
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual price (100,000 dollars)")
    plt.ylabel("Predicted price (100,000 dollars)")
    plt.title(f"Actual vs Predicted Price - {best_name}")
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
# Function Name : plot_feature_importance
# Description   : Train a Random Forest on the training data and
#                 save the graph of the feature importance.
# Input         : X_train - Training features
#                 Y_train - Training target
# Output        : Saves feature importance image
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def plot_feature_importance(X_train, Y_train):

    forest = RandomForestRegressor(
        n_estimators=FOREST_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    forest.fit(X_train, Y_train)

    importance = pd.Series(
        forest.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print()
    print("Feature importance (Random Forest) :")
    print(importance.round(3))

    plt.figure(figsize=(8, 5))

    plt.barh(importance.index[::-1], importance.values[::-1])

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
# Function Name : plot_price_map
# Description   : Draw and save a map of California where the
#                 colour of every point is the house price
# Input         : df - Input pandas dataframe
# Output        : Saves price map image
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def plot_price_map(df):

    plt.figure(figsize=(8, 7))

    points = plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df[TARGET_HEADER],
        s=4,
        cmap="viridis"
    )

    plt.colorbar(points, label="House price (100,000 dollars)")

    plt.title("House Prices in California")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    os.makedirs(
        IMAGES_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(IMAGES_DIR, "price_map.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Image Saved : images/price_map.png")


############################################################
# Function Name : predict_prices
# Description   : Predict the price of some areas of the test
#                 data using the loaded model and compare the
#                 result with the real price
# Input         : model - Loaded model
#                 X_test - Testing features
#                 Y_test - Actual testing target
#                 count - Number of areas to check
# Output        : Displays actual and predicted prices
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def predict_prices(model, X_test, Y_test, count):

    samples = X_test.iloc[:count]

    predictions = model.predict(samples)

    print()
    for index, prediction, actual in zip(samples.index, predictions, Y_test.iloc[:count]):

        print(
            f"Area {index:<6}: Actual = {actual * PRICE_UNIT:>9,.0f} dollars   "
            f"Predicted = {prediction * PRICE_UNIT:>9,.0f} dollars   "
            f"Difference = {abs(actual - prediction) * PRICE_UNIT:>8,.0f} dollars"
        )


############################################################
# Function Name : display_summary
# Description   : Display the final project summary
# Input         : best_name - Name of the best model
#                 results - Cross validation scores
#                 r2 - R2 score on test data
#                 rmse - RMSE on test data
#                 mae - MAE on test data
# Output        : Displays project summary
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def display_summary(best_name, results, r2, rmse, mae):

    print()
    print("Models compared          : 6")
    print("Best ensemble model      :", best_name)
    print(f"Cross validation R2      : {results.loc[best_name, 'CV R2']:.4f}")
    print(f"Test R2 score            : {r2:.4f}")
    print(f"Test RMSE                : {rmse:.4f}  (about {rmse * PRICE_UNIT:,.0f} dollars)")
    print(f"Test MAE                 : {mae:.4f}  (about {mae * PRICE_UNIT:,.0f} dollars)")
    print()
    print("Note : prices are cut off at 500,000 dollars in this dataset.")


############################################################
# Function Name : display_footer
# Description   : Display project completion message
# Input         : Nothing
# Output        : Displays completion message
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def display_footer():

    print()
    print("=" * 60)
    print("   CALIFORNIA HOUSING PRICE PREDICTION COMPLETED")
    print("=" * 60)
    print()


############################################################
# Function Name : main
# Description   : Execute complete California Housing project
# Input         : Nothing
# Output        : Executes complete machine learning workflow
# Author        : Ishwari Vijaykumar Surve
# Date          : 15/10/2026
############################################################

def main():

    display_title()

    display_step("Step 1: Load the dataset")
    df = load_data(DATA_PATH)

    display_step("Step 2: Check the data quality")
    check_data(df)

    display_step("Step 3: Display the statistical summary")
    show_statistics(df)

    display_step("Step 4: Check the correlation with the price")
    show_correlation(df)

    display_step("Step 5: Separate independent and dependent variables")
    X, Y = split_features_target(df)

    display_step("Step 6: Split the dataset for training and testing")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    display_step("Step 7: Build the 6 models")
    models = build_models()

    display_step("Step 8: Compare the models with cross validation")
    results = compare_models(models, X_train, Y_train)

    display_step("Step 9: Select the best ensemble model")
    best_name = select_best_model(results)

    display_step("Step 10: Train the best ensemble model")
    best_model = train_best_model(models[best_name][1], X_train, Y_train)

    display_step("Step 11: Save and load the model")
    save_model(best_model, MODEL_PATH)
    loaded_model = load_model(MODEL_PATH)

    display_step("Step 12: Evaluate the best model on the test data")
    predictions, r2, rmse, mae = evaluate_model(
        loaded_model, X_test, Y_test
    )

    display_step("Step 13: Test scores of all 6 models")
    evaluate_all_models(models, X_train, Y_train, X_test, Y_test)

    display_step("Step 14: Plot the graphs")
    plot_model_comparison(results, best_name)
    plot_actual_vs_predicted(Y_test, predictions, best_name)
    plot_feature_importance(X_train, Y_train)
    plot_price_map(df)

    display_step("Step 15: Predict the price of some areas")
    predict_prices(loaded_model, X_test, Y_test, 5)

    display_step("Step 16: Project summary")
    display_summary(best_name, results, r2, rmse, mae)

    display_footer()


############################################################
# Application Starter
############################################################

if __name__ == "__main__":

    main()