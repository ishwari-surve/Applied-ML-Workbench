"""
------------------------------------------------------------
Project Name        : Credit Card Fraud Detection
                      using Machine Learning Classification

Dataset Information :

Dataset File        : data/creditcard.csv
                      (284807 transactions)

Features            : Time       -> Seconds elapsed between
                                     each transaction and the
                                     first transaction
                      V1 - V28   -> Anonymized features obtained
                                     using PCA transformation
                      Amount     -> Transaction amount

Target              : Class      -> 0 = Legitimate transaction
                                     1 = Fraudulent transaction

Machine Learning Information :

Models Compared     : Logistic Regression
                      Decision Tree
                      Random Forest
                      Gradient Boosting
                      Linear SVM

Techniques Used     : Feature Scaling (RobustScaler)
                      Stratified Train-Test Split
                      Class Imbalance Handling
                      (Class Weighting, SMOTE,
                       Random Undersampling)
                      Stratified 5-fold Cross Validation
                      Hyperparameter Tuning
                      Classification Threshold Tuning
                      Model Explainability
                      Model Saving and Loading (joblib)

Library             : Scikit-Learn, Imbalanced-Learn

Problem Type        : Binary Classification

Evaluation Metrics  : Average Precision (AUPRC),
                      Precision, Recall, F1-Score,
                      Confusion Matrix

Author              : Ishwari Vijaykumar Surve

Date                : 17/10/2026
------------------------------------------------------------
"""


############################################################
# Required Python Packages
############################################################

import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict


############################################################
# Function 1: Display Dataset Information
############################################################

def display_dataset_info(data):
    """
    Display basic information about the credit card fraud
    detection dataset.
    """

    print("\n" + "-" * 60)
    print("CREDIT CARD FRAUD DETECTION - DATASET INFORMATION")
    print("-" * 60)

    # Display dataset dimensions
    print("\nDataset Shape:")
    print(f"Rows    : {data.shape[0]}")
    print(f"Columns : {data.shape[1]}")

    # Display first five records
    print("\nFirst Five Records:")
    print(data.head())

    # Display column names
    print("\nColumn Names:")
    print(data.columns.tolist())

    # Display data types
    print("\nData Types:")
    print(data.dtypes)

    # Display missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Display duplicate rows
    print("\nDuplicate Rows:")
    print(data.duplicated().sum())

    # Display statistical summary
    print("\nStatistical Summary:")
    print(data.describe())

    print("\n" + "-" * 60)

############################################################
# Function : Clean Dataset
############################################################

def clean_dataset(data):

    print("\n" + "=" * 70)
    print("DATASET CLEANING")
    print("=" * 70)

    DuplicateCount = data.duplicated().sum()
    data = data.drop_duplicates().copy()

    print("Duplicate Rows Removed:", DuplicateCount)
    print("Dataset Shape After Cleaning:", data.shape)

    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    return data


############################################################
# Function 2: Validate Dataset
############################################################

def validate_dataset(data):
    """
    Validate the dataset before model development.
    """

    print("\n" + "-" * 60)
    print("DATASET VALIDATION")
    print("-" * 60)

    # Check for empty dataset
    if data.empty:
        raise ValueError("Dataset is empty!")

    # Check required columns
    required_columns = (
        ["Time", "Amount"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Class"]
    )

    if not set(required_columns).issubset(data.columns):
        raise ValueError("Required dataset columns are missing.")

    # Check missing values
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Check target values
    if set(data["Class"].unique()) != {0, 1}:
        raise ValueError("Target must contain both 0 and 1.")

    print("Dataset validation successful.")
    print("Target: 0 = Legitimate, 1 = Fraud")
    print("-" * 60)

    return True


############################################################
# Function 3: Exploratory Data Analysis (EDA)
############################################################

def perform_eda(data):
    """
    Analyze class imbalance and visualize transaction patterns.
    """

    print("\n" + "-" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("-" * 60)

    # Class distribution
    class_counts = data["Class"].value_counts().sort_index()
    fraud_percentage = data["Class"].mean() * 100

    print("\nTransaction Distribution:")
    print(f"Legitimate Transactions : {class_counts.get(0, 0)}")
    print(f"Fraudulent Transactions  : {class_counts.get(1, 0)}")
    print(f"Fraud Percentage        : {fraud_percentage:.4f}%")

    # Visualize class imbalance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=data, x="Class")
    
    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()

    plt.savefig(
        "images/class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # Transaction amount distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=data,
        x="Amount",
        hue="Class",
        bins=50,
        element="step",
        stat="count",
        common_norm=False
    )
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Number of Transactions")
    
    plt.tight_layout()
    plt.savefig(
    "images/transaction_amount_distribution.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    # Transaction time distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=data,
        x="Time",
        hue="Class",
        bins=50,
        element="step",
        stat="count",
        common_norm=False
    )
    plt.title("Transaction Time Distribution")
    plt.xlabel("Time (seconds from first transaction)")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(
    "images/transaction_time_distribution.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    print("\nEDA completed.")
    print("-" * 60)

############################################################
# Function 4: Prepare Data
############################################################

def PrepareData(dfDataset):

    print("\n" + "=" * 70)
    print("STEP 4: PREPARE FEATURES AND TARGET")
    print("=" * 70)

    # Separate features and target
    X = dfDataset.drop(columns="Class")
    y = dfDataset["Class"]

    print("Feature Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nTarget Distribution:")
    print(y.value_counts())

    return X, y


############################################################
# Function 5: Split Data
############################################################

def split_data(X, y):

    print("\n" + "=" * 70)
    print("STEP 5: SPLIT TRAINING AND TESTING DATA")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape :", X_test.shape)
    print("Training Target Shape  :", y_train.shape)
    print("Testing Target Shape   :", y_test.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


############################################################
# Function 6: Create Preprocessing Pipeline
############################################################

def CreatePreprocessingPipeline():

    print("\n" + "=" * 70)
    print("STEP 6: CREATE PREPROCESSING PIPELINE")
    print("=" * 70)

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), ["Amount", "Time"])
        ],
        remainder="passthrough"
    )

    print("RobustScaler applied to: Amount, Time")
    print("PCA features V1-V28: Unchanged")

    return preprocessor


############################################################
# Function 7: Create Baseline Model
############################################################

def CreateBaselineModel(preprocessor):

    print("\n" + "=" * 70)
    print("STEP 7: CREATE BASELINE MODEL")
    print("=" * 70)

    model = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    print("Model      : Logistic Regression")
    print("Class Weight: Balanced")
    print("Preprocessing: Included in pipeline")

    return model


############################################################
# Function 8: Train Model
############################################################

def train_model(model, X_train, y_train):

    print("\n" + "=" * 70)
    print("STEP 8: TRAIN BASELINE MODEL")
    print("=" * 70)

    model.fit(X_train, y_train)

    print("Model trained successfully.")
    print("Training Samples:", X_train.shape[0])
    print("Features:", X_train.shape[1])

    return model


############################################################
# Function 9: Evaluate Model
############################################################


def evaluate_model(model, X_test, y_test, Threshold=0.5):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get prediction scores
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    # Apply the selected threshold
    y_pred = (y_prob >= Threshold).astype(int)

    # Calculate evaluation metrics
    auprc = average_precision_score(y_test, y_prob)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    print(f"\nThreshold : {Threshold:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(values_format="d")
    plt.title("Credit Card Fraud Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    return {
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


############################################################
# Function 10: Create Additional Models
############################################################

def CreateModels():

    print("\n" + "=" * 70)
    print("STEP 10: CREATE CLASSIFICATION MODELS")
    print("=" * 70)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=5000
        )
    }

    for name in models:
        print(f"Model Created: {name}")

    return models


############################################################
# Function 11: Train and Compare Models
############################################################


def TrainAndCompareModels(models, preprocessor, X_train, y_train):

    print("\n" + "=" * 70)
    print("STEP 11: TRAIN AND COMPARE MODELS")
    print("=" * 70)

    CV = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    trained_models = {}

    for name, classifier in models.items():

        print(f"\nEvaluating {name}...")

        pipeline = ImbPipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(classifier))
        ])

        Scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            scoring={
                "AUPRC": "average_precision",
                "Precision": "precision",
                "Recall": "recall",
                "F1": "f1"
            },
            n_jobs=-1
        )

        results.append({
            "Model": name,
            "AUPRC": Scores["test_AUPRC"].mean(),
            "Precision": Scores["test_Precision"].mean(),
            "Recall": Scores["test_Recall"].mean(),
            "F1-Score": Scores["test_F1"].mean()
        })

        # Fit a model on the complete training set
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="AUPRC",
        ascending=False
    ).reset_index(drop=True)

    print("\n5-FOLD CROSS-VALIDATION MODEL COMPARISON:")
    print(results_df.to_string(index=False))

    return trained_models, results_df


############################################################
# Function 12: Tune Logistic Regression
############################################################

def TuneLogisticRegression(Preprocessor, X_train, y_train):
    """Tune Logistic Regression hyperparameters."""

    print("\n" + "=" * 70)
    print("STEP 12: TUNE LOGISTIC REGRESSION")
    print("=" * 70)

    Model = Pipeline([
        ("Preprocessor", clone(Preprocessor)),
        ("Classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    ParamDistribution = {
        "Classifier__C": [0.01, 0.1, 1, 10, 100],
        "Classifier__class_weight": [None, "balanced"],
        "Classifier__solver": ["lbfgs"]
    }

    Search = RandomizedSearchCV(
        estimator=Model,
        param_distributions=ParamDistribution,
        n_iter=10,
        scoring="average_precision",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    StartTime = time.perf_counter()
    Search.fit(X_train, y_train)
    TuningTime = time.perf_counter() - StartTime

    print("\nBest Parameters:")
    print(Search.best_params_)

    print(
        f"\nBest Cross-Validation AUPRC: "
        f"{Search.best_score_:.4f}"
    )
    print(f"Tuning Time: {TuningTime:.2f} seconds")

    return Search.best_estimator_, Search



############################################################
# Function 13: Perform Cross Validation
############################################################

def PerformCrossValidation(Model, X_train, y_train):

    print("\n" + "=" * 70)
    print("STEP 13: PERFORM CROSS VALIDATION")
    print("=" * 70)

    CV = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    Scores = cross_validate(
        Model,
        X_train,
        y_train,
        cv=CV,
        scoring="average_precision",
        n_jobs=-1
    )

    AUPRCScores = Scores["test_score"]

    for Fold, Score in enumerate(AUPRCScores, start=1):
        print(f"Fold {Fold} AUPRC: {Score:.4f}")

    print(f"\nMean AUPRC: {AUPRCScores.mean():.4f}")
    print(f"Standard Deviation: {AUPRCScores.std():.4f}")

    return AUPRCScores


############################################################
# Function 14: Compare Imbalance Techniques
############################################################

def CompareImbalanceTechniques(Preprocessor, X_train, y_train):

    print("\n" + "=" * 70)
    print("STEP 14: COMPARE IMBALANCE TECHNIQUES")
    print("=" * 70)

    Techniques = {
        "Class Weighting": {
            "sampler": None,
            "class_weight": "balanced"
        },
        "SMOTE": {
            "sampler": SMOTE(random_state=42),
            "class_weight": None
        },
        "Random Undersampling": {
            "sampler": RandomUnderSampler(random_state=42),
            "class_weight": None
        }
    }

    CV = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    Results = []

    for Name, Settings in Techniques.items():

        Steps = [
            ("Preprocessor", clone(Preprocessor))
        ]

        if Settings["sampler"] is not None:
            Steps.append(("Sampler", Settings["sampler"]))

        Steps.append((
            "Classifier",
            LogisticRegression(
                class_weight=Settings["class_weight"],
                max_iter=2000,
                random_state=42
            )
        ))

        Model = ImbPipeline(steps=Steps)

        Scores = cross_validate(
            Model,
            X_train,
            y_train,
            cv=CV,
            scoring="average_precision",
            n_jobs=-1
        )

        Results.append({
            "Technique": Name,
            "Mean AUPRC": Scores["test_score"].mean()
        })

        print(
            f"{Name}: "
            f"{Scores['test_score'].mean():.4f}"
        )

    ResultsDF = pd.DataFrame(Results).sort_values(
        by="Mean AUPRC",
        ascending=False
    ).reset_index(drop=True)

    print("\nIMBALANCE TECHNIQUE COMPARISON:")
    print(ResultsDF.to_string(index=False))

    return ResultsDF


############################################################
# Function 15: Plot Precision-Recall Curve
############################################################

def PlotPrecisionRecallCurve(Model, X_test, y_test):

    print("\n" + "=" * 70)
    print("STEP 15: PLOT PRECISION-RECALL CURVE")
    print("=" * 70)

    # Get prediction scores
    if hasattr(Model, "predict_proba"):
        y_scores = Model.predict_proba(X_test)[:, 1]
    else:
        y_scores = Model.decision_function(X_test)

    # Calculate precision and recall
    Precision, Recall, _ = precision_recall_curve(
        y_test, y_scores
    )

    AUPRC = average_precision_score(y_test, y_scores)

    # Plot curve
    plt.figure(figsize=(8, 6))
    plt.plot(Recall, Precision, label=f"AUPRC = {AUPRC:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Credit Card Fraud Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
    "images/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
   )
    plt.show()

    print(f"AUPRC: {AUPRC:.4f}")


############################################################
# Function 16: Tune Classification Threshold
############################################################

def TuneClassificationThreshold(Model, X_train, y_train):

    print("\n" + "=" * 70)
    print("STEP 16: TUNE CLASSIFICATION THRESHOLD")
    print("=" * 70)

    # Generate out-of-fold prediction scores on training data
    CV = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if hasattr(Model, "predict_proba"):
        Scores = cross_val_predict(
            Model,
            X_train,
            y_train,
            cv=CV,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]
    else:
        Scores = cross_val_predict(
            Model,
            X_train,
            y_train,
            cv=CV,
            method="decision_function",
            n_jobs=-1
        )

    Precision, Recall, Thresholds = precision_recall_curve(
        y_train, Scores
    )

    # Exclude the final precision/recall point (no threshold)
    F1Scores = (
        2 * Precision[:-1] * Recall[:-1]
        / (Precision[:-1] + Recall[:-1] + 1e-10)
    )

    BestIndex = np.argmax(F1Scores)
    BestThreshold = Thresholds[BestIndex]

    print(f"Best Threshold: {BestThreshold:.4f}")
    print(f"Precision      : {Precision[BestIndex]:.4f}")
    print(f"Recall         : {Recall[BestIndex]:.4f}")
    print(f"F1-Score       : {F1Scores[BestIndex]:.4f}")

    return BestThreshold


############################################################
# Function 17: Perform Error Analysis
############################################################

def PerformErrorAnalysis(Model, X_test, y_test, Threshold=0.5):

    print("\n" + "=" * 70)
    print("STEP 17: PERFORM ERROR ANALYSIS")
    print("=" * 70)

    # Get prediction scores
    if hasattr(Model, "predict_proba"):
        Scores = Model.predict_proba(X_test)[:, 1]
    else:
        Scores = Model.decision_function(X_test)

    # Apply threshold
    if hasattr(Model, "predict_proba"):
        y_pred = (Scores >= Threshold).astype(int)
    else:
        y_pred = (Scores >= Threshold).astype(int)

    # Identify errors
    FalsePositives = X_test[(y_test == 0) & (y_pred == 1)]
    FalseNegatives = X_test[(y_test == 1) & (y_pred == 0)]

    print(f"\nFalse Positives: {len(FalsePositives)}")
    print(f"False Negatives: {len(FalseNegatives)}")

    print("\nFalse Positives (Legitimate flagged as fraud):")
    print(FalsePositives.head())

    print("\nFalse Negatives (Fraud missed by model):")
    print(FalseNegatives.head())

    return FalsePositives, FalseNegatives


############################################################
# Function 18: Explain Model
############################################################

def ExplainModel(Model, FeatureNames):

    print("\n" + "=" * 70)
    print("STEP 18: MODEL EXPLAINABILITY")
    print("=" * 70)

    # Extract Logistic Regression coefficients
    Classifier = Model.named_steps["Classifier"]
    Coefficients = Classifier.coef_[0]

    # Create feature importance table
    ImportanceDF = pd.DataFrame({
        "Feature": FeatureNames,
        "Importance": np.abs(Coefficients)
    })

    ImportanceDF = ImportanceDF.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features:")
    print(ImportanceDF.head(10).to_string(index=False))

    # Plot top 10 features
    TopFeatures = ImportanceDF.head(10).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(8, 6))
    plt.barh(
        TopFeatures["Feature"],
        TopFeatures["Importance"]
    )
    plt.xlabel("Absolute Coefficient")
    plt.title("Top 10 Features - Logistic Regression")
    plt.tight_layout()
    plt.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
    )

    plt.show()

    return ImportanceDF


############################################################
# Function 19: Save Model
############################################################

def SaveModel(
    Model,
    Threshold=0.5,
    FilePath="models/credit_card_fraud_model.pkl"
):

    print("\n" + "=" * 70)
    print("STEP 19: SAVE TRAINED MODEL")
    print("=" * 70)

    # Create models directory
    os.makedirs(os.path.dirname(FilePath), exist_ok=True)

    # Save model and tuned threshold together
    ModelData = {
        "model": Model,
        "threshold": Threshold
    }

    joblib.dump(ModelData, FilePath)

    print(f"Model and threshold saved successfully: {FilePath}")
    print(f"Saved threshold: {Threshold:.4f}")



############################################################
# Function 20: Load Model
############################################################

def LoadModel(FilePath="models/credit_card_fraud_model.pkl"):

    print("\n" + "=" * 70)
    print("STEP 20: LOAD TRAINED MODEL")
    print("=" * 70)

    if not os.path.exists(FilePath):
        raise FileNotFoundError(
            f"Model file not found: {FilePath}"
        )

    # Load saved model data
    ModelData = joblib.load(FilePath)

    Model = ModelData["model"]
    Threshold = ModelData["threshold"]

    print("Model loaded successfully.")
    print(f"Loaded threshold: {Threshold:.4f}")

    return Model, Threshold


############################################################
# Function 21: Save Evaluation Results
############################################################

def SaveEvaluationResults(ResultsDF, FilePath="results/model_comparison.csv"):

    print("\n" + "=" * 70)
    print("STEP 21: SAVE EVALUATION RESULTS")
    print("=" * 70)

    # Create results directory
    os.makedirs(os.path.dirname(FilePath), exist_ok=True)

    # Save results to CSV
    ResultsDF.to_csv(FilePath, index=False)

    print(f"Evaluation results saved: {FilePath}")
    


############################################################
# Function 22: Load Evaluation Results
############################################################

def LoadEvaluationResults(
    FilePath="results/model_comparison.csv"
):

    print("\n" + "=" * 70)
    print("STEP 22: LOAD EVALUATION RESULTS")
    print("=" * 70)

    if not os.path.exists(FilePath):
        raise FileNotFoundError(
            f"Results file not found: {FilePath}"
        )

    ResultsDF = pd.read_csv(FilePath)

    print("\nSaved Evaluation Results:")
    print(ResultsDF.to_string(index=False))

    return ResultsDF


############################################################
# Function 23: Save Tuning Results
############################################################

def SaveTuningResults(Search, FilePath="results/logistic_regression_tuning.csv"):

    print("\n" + "=" * 70)
    print("STEP 23: SAVE TUNING RESULTS")
    print("=" * 70)

    os.makedirs(os.path.dirname(FilePath), exist_ok=True)

    TuningResults = pd.DataFrame(Search.cv_results_)

    TuningResults.to_csv(FilePath, index=False)

    print(f"Tuning results saved: {FilePath}")

    return TuningResults



############################################################
# Function 24: Save Classification Report
############################################################


def SaveClassificationReport(
    Model, X_test, y_test, Threshold=0.5,
    FilePath="results/classification_report.csv"
):

    print("\n" + "=" * 70)
    print("STEP 24: SAVE CLASSIFICATION REPORT")
    print("=" * 70)

    # Get prediction scores
    if hasattr(Model, "predict_proba"):
        Scores = Model.predict_proba(X_test)[:, 1]
    else:
        Scores = Model.decision_function(X_test)

    # Apply the selected threshold
    y_pred = (Scores >= Threshold).astype(int)

    # Generate classification report
    Report = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    ReportDF = pd.DataFrame(Report).transpose()

    # Create results directory
    os.makedirs(os.path.dirname(FilePath), exist_ok=True)

    # Save report
    ReportDF.to_csv(FilePath)

    print(f"Threshold used: {Threshold:.4f}")
    print(f"Classification report saved: {FilePath}")

    return ReportDF


############################################################
# Function 25: Save Model Metrics
############################################################

def SaveModelMetrics(Metrics, FilePath="results/model_metrics.json"):

    print("\n" + "=" * 70)
    print("STEP 25: SAVE MODEL METRICS")
    print("=" * 70)

    # Create results directory
    os.makedirs(os.path.dirname(FilePath), exist_ok=True)

    # Convert NumPy values to standard Python types
    Metrics = {
        Key: float(Value)
        for Key, Value in Metrics.items()
    }

    # Save metrics as JSON
    with open(FilePath, "w", encoding="utf-8") as File:
        json.dump(Metrics, File, indent=4)

    print(f"Model metrics saved: {FilePath}")
    


############################################################
# Function 26: Print Final Summary
############################################################

def PrintFinalSummary(ModelName, Metrics):

    print("\n" + "=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)

    print(f"\nSelected Model: {ModelName}")

    print("\nFinal Evaluation Metrics:")

    for Metric, Value in Metrics.items():
        print(f"{Metric:<15}: {Value:.4f}")

    print("\nProject: Credit Card Fraud Detection")
    print("Problem Type: Binary Classification")
    print("=" * 70)


############################################################
# Function 27: Predict Transaction
############################################################

def PredictTransaction(Model, Transaction, Threshold=0.5):

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD PREDICTION")
    print("=" * 70)

    # Get prediction score
    if hasattr(Model, "predict_proba"):
        Score = Model.predict_proba(Transaction)[0, 1]
    else:
        Score = Model.decision_function(Transaction)[0]

    # Apply the selected threshold
    Prediction = int(Score >= Threshold)

    if Prediction == 1:
        Result = "FRAUDULENT TRANSACTION"
    else:
        Result = "LEGITIMATE TRANSACTION"

    print(f"Fraud Score : {Score:.4f}")
    print(f"Threshold   : {Threshold:.4f}")
    print(f"Prediction  : {Result}")

    return Prediction



############################################################
# Function 28: Main
############################################################

def main():

    print("\n" + "=" * 70)
    print("CREDIT CARD FRAUD DETECTION")
    print("=" * 70)

    # Load dataset
    DatasetPath = "data/creditcard.csv"
    dfDataset = pd.read_csv(DatasetPath)

    # Inspect, clean, and validate dataset
    display_dataset_info(dfDataset)
    dfDataset = clean_dataset(dfDataset)
    validate_dataset(dfDataset)

    # Exploratory data analysis
    perform_eda(dfDataset)

    # Prepare and split data
    X, y = PrepareData(dfDataset)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create preprocessing pipeline
    Preprocessor = CreatePreprocessingPipeline()

    
    # Create and compare additional models
    Models = CreateModels()

    TrainedModels, ComparisonResults = TrainAndCompareModels(
        Models,
        Preprocessor,
        X_train,
        y_train
    )

    # Save model comparison results
    SaveEvaluationResults(
        ComparisonResults,
        "results/model_comparison.csv"
    )


    # Train baseline model
    BaselineModel = CreateBaselineModel(Preprocessor)
    BaselineModel = train_model(
        BaselineModel, X_train, y_train
    )

    # Tune Logistic Regression
    BestModel, Search = TuneLogisticRegression(
        Preprocessor, X_train, y_train
    )

    # Cross-validation
    PerformCrossValidation(
        BestModel, X_train, y_train
    )

    # Compare imbalance techniques
    CompareImbalanceTechniques(
        Preprocessor, X_train, y_train
    )

    
    # Tune threshold using training data
    Threshold = TuneClassificationThreshold(
        BestModel, X_train, y_train
    )

    # Visualize precision-recall curve
    PlotPrecisionRecallCurve(
        BestModel, X_test, y_test
    )

    # Final evaluation
    FinalMetrics = evaluate_model(
    BestModel, X_test, y_test, Threshold
    ) 


    # Error analysis
    PerformErrorAnalysis(
        BestModel, X_test, y_test, Threshold
    )

    FeatureNames = (
    ["Amount", "Time"]
    + [f"V{i}" for i in range(1, 29)]
    )

    ExplainModel(BestModel, FeatureNames) 

    # Save model and results
    SaveModel(BestModel, Threshold)
    SaveEvaluationResults(
    pd.DataFrame([FinalMetrics]),
    "results/final_model_metrics.csv"
    )

    SaveTuningResults(Search)

    SaveClassificationReport(
    BestModel, X_test, y_test, Threshold
   )
    SaveModelMetrics(FinalMetrics)

    # Final summary
    PrintFinalSummary(
        "Tuned Logistic Regression",
        FinalMetrics
    )

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()







