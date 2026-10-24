"""
------------------------------------------------------------
Project Name        : IMDb Movie Review Sentiment Analysis
                      (NLP Classification)

Dataset Information :

Dataset File        : data/raw/IMDB_Dataset.csv
                      (50,000 movie reviews)

Columns             : review     -> Text of the movie review
                      sentiment  -> positive or negative

Target              : sentiment -> 1 = positive, 0 = negative
                      (balanced: 25,000 each)

Project Parts       : A. Data understanding and cleaning
                      B. Exploratory analysis of sentiment
                      C. Text preprocessing and feature creation
                      D. Model training, comparison and tuning
                      E. Evaluation, error analysis and prediction
                      F. Saving outputs and project summary

Machine Learning Information :

Models Compared     : Baseline (most frequent class)
                      Multinomial Naive Bayes
                      Logistic Regression
                      Linear SVM
                      Random Forest

Techniques Used     : Text cleaning (HTML removal, normalisation)
                      Stopword removal (keeps negations)
                      Lemmatisation
                      Bag of Words and TF-IDF vectorisation
                      Stratified train-test split
                      5-fold stratified cross validation
                      Hyperparameter tuning (training data only)
                      Error analysis
                      Model saving and loading (joblib)

Library             : Pandas, NumPy, Matplotlib, Seaborn,
                      Scikit-Learn, NLTK, Joblib

Problem Type        : Binary Classification (balanced)

Evaluation Metrics  : Accuracy, Precision, Recall, F1-score
                      (per class), ROC-AUC, Confusion Matrix

Author              : Ishwari Vijaykumar Surve

Date                : 23/10/2026
------------------------------------------------------------
"""

############################################################
# Required Python Packages
############################################################

import os
import re
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import clone

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore", category=UserWarning)


############################################################
# Project Paths and Constants
############################################################

BASE_DIR = r"C:\Users\Ishwari Surve\OneDrive\Desktop\Marvellous Python -AIML\11-IMDB Dataset case study"

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

DATA_PATH = os.path.join(RAW_DIR, "IMDB_Dataset.csv")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

REQUIRED_COLUMNS = ["review", "sentiment"]

BORDER = "=" * 60

############################################################
# Function Name : setup_project
# Description   : Create the output folders and download the
#                 NLTK resources needed for text processing
#                 (stopwords and WordNet). Downloads happen only
#                 when the resources are missing.
# Input         : None
# Output        : Returns True when the setup is complete
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def setup_project():

    print("\n" + BORDER)
    print("PROJECT SETUP")
    print(BORDER)

    # Create the output folders if they do not exist
    for folder in [IMAGES_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(folder, exist_ok=True)
        print(f"Folder ready: {folder}")

    # Download NLTK resources only when they are missing
    nltk_resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }

    for name, path in nltk_resources.items():
        try:
            nltk.data.find(path)
            print(f"NLTK resource found : {name}")
        except LookupError:
            print(f"Downloading NLTK resource : {name}")
            nltk.download(name, quiet=True)

    print("\nProject setup completed successfully.")

    return True

############################################################
# Function Name : display_header
# Description   : Display the project title, author information,
#                 and a short introduction to the IMDb sentiment
#                 analysis project.
# Input         : None
# Output        : Prints a formatted project header in the console
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def display_header():

    print("\n" + BORDER)
    print("       IMDb MOVIE REVIEW SENTIMENT ANALYSIS")
    print("          USING NLP AND MACHINE LEARNING")
    print(BORDER)

    print("Project  : Sentiment Classification")
    print("Dataset  : IMDb Dataset of 50K Movie Reviews")
    print("Author   : Ishwari Vijaykumar Surve")
    print("Language : Python")
    print(BORDER)

############################################################
# Function Name : load_dataset
# Description   : Check that the IMDb CSV file exists, read it,
#                 and confirm that the required columns are present.
#                 Shows the shape of the dataset after loading.
# Input         : path - Path of the IMDb CSV file
# Output        : Returns the dataset as a pandas DataFrame
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def load_dataset(path):

    print("\n" + BORDER)
    print("STEP 1: LOAD THE IMDb DATASET")
    print(BORDER)

    # Check that the file exists before reading
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download IMDB_Dataset.csv from Kaggle and place it in data/raw/."
        )

    df = pd.read_csv(path)

    # Check that the required columns are present
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    print(f"Dataset path      : {path}")
    print(f"Dataset loaded    : successfully")
    print(f"Number of rows    : {df.shape[0]:,}")
    print(f"Number of columns : {df.shape[1]}")

    return df

############################################################
# Function Name : understand_dataset
# Description   : Display the structure of the dataset: first and
#                 last records, column names, data types, class
#                 counts, sample reviews and word-count statistics.
# Input         : df - DataFrame returned by load_dataset()
# Output        : Displays dataset information. Does not change data.
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def understand_dataset(df):

    print("\n" + BORDER)
    print("STEP 2: UNDERSTAND THE DATASET")
    print(BORDER)

    # First and last records
    print("\nFirst five records:")
    print(df.head().to_string(max_colwidth=60))

    print("\nLast five records:")
    print(df.tail().to_string(max_colwidth=60))

    # Column names and data types
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    # Dimensions
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Class counts
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts().to_string())

    # Word count statistics (computed on a temporary column)
    word_counts = df["review"].astype(str).str.split().str.len()

    print("\nWord count statistics per review:")
    print(word_counts.describe().round(1).to_string())

    # Sample reviews from each class
    print("\nSample positive review:")
    print(df[df["sentiment"] == "positive"]["review"].iloc[0][:300])

    print("\nSample negative review:")
    print(df[df["sentiment"] == "negative"]["review"].iloc[0][:300])

    print("\nDataset understanding completed.")

############################################################
# Function Name : check_structure
# Description   : Check the structure of the dataset: missing
#                 values, duplicate reviews, and whether the
#                 sentiment labels contain only the expected values.
# Input         : df - DataFrame returned by load_dataset()
# Output        : Returns a dictionary with the check results
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def check_structure(df):

    print("\n" + BORDER)
    print("STEP 3: STRUCTURAL CHECKS")
    print(BORDER)

    # Missing values per column
    missing = df.isnull().sum()

    print("\nMissing values per column:")
    print(missing.to_string())

    # Duplicate rows (identical review and label)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percent = duplicate_rows / len(df) * 100

    print(f"\nDuplicate rows : {duplicate_rows:,} ({duplicate_percent:.2f}%)")

    # Duplicate review text with different labels (conflicting labels)
    conflicting = int(
        df.groupby("review")["sentiment"].nunique().gt(1).sum()
    )

    print(f"Reviews with conflicting labels : {conflicting:,}")

    # Label validity
    valid_labels = {"positive", "negative"}
    found_labels = set(df["sentiment"].dropna().unique())
    invalid_labels = found_labels - valid_labels

    print(f"Labels found : {sorted(found_labels)}")

    if invalid_labels:
        print(f"Invalid labels found : {sorted(invalid_labels)}")
    else:
        print("Label check : all labels are valid")

    print("\nStructural checks completed.")

    return {
        "missing_total": int(missing.sum()),
        "duplicate_rows": duplicate_rows,
        "conflicting_labels": conflicting,
        "invalid_labels": sorted(invalid_labels)
    }

############################################################
# Function Name : clean_dataset
# Description   : Clean the dataset by removing duplicate reviews
#                 and rows with missing values, then confirm that
#                 the sentiment labels are valid. The original
#                 DataFrame is not changed.
# Input         : df - DataFrame returned by load_dataset()
# Output        : Returns the cleaned DataFrame
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def clean_dataset(df):

    print("\n" + BORDER)
    print("STEP 4: CLEAN THE DATASET")
    print(BORDER)

    # Work on a copy to keep the original data unchanged
    clean = df.copy()

    rows_before = len(clean)

    # Remove rows with missing review text or label
    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    missing_removed = rows_before - len(clean)

    # Remove exact duplicate rows (same review and same label)
    duplicates_before = len(clean)
    clean = clean.drop_duplicates()
    duplicates_removed = duplicates_before - len(clean)

    # Remove reviews that appear with conflicting labels
    conflicting_reviews = clean.groupby("review")["sentiment"].nunique()
    conflicting_reviews = conflicting_reviews[conflicting_reviews > 1].index

    conflicts_removed = int(clean["review"].isin(conflicting_reviews).sum())
    clean = clean[~clean["review"].isin(conflicting_reviews)]

    # Keep only the valid sentiment labels
    valid_labels = ["positive", "negative"]
    invalid_removed = int((~clean["sentiment"].isin(valid_labels)).sum())
    clean = clean[clean["sentiment"].isin(valid_labels)]

    # Reset the row index after removing rows
    clean = clean.reset_index(drop=True)

    print(f"Rows before cleaning      : {rows_before:,}")
    print(f"Missing rows removed      : {missing_removed:,}")
    print(f"Duplicate rows removed    : {duplicates_removed:,}")
    print(f"Conflicting rows removed  : {conflicts_removed:,}")
    print(f"Invalid label rows removed: {invalid_removed:,}")
    print(f"Rows after cleaning       : {len(clean):,}")

    print("\nSentiment counts after cleaning:")
    print(clean["sentiment"].value_counts().to_string())

    print("\nDataset cleaning completed.")

    return clean

############################################################
# Function Name : analyse_sentiment_distribution
# Description   : Count the positive and negative reviews, show
#                 their percentages, and save a bar chart of the
#                 class balance to the images folder.
# Input         : df - Cleaned DataFrame returned by clean_dataset()
# Output        : Saves images/01_sentiment_distribution.png and
#                 returns the class counts
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def analyse_sentiment_distribution(df):

    print("\n" + BORDER)
    print("STEP 5: SENTIMENT DISTRIBUTION")
    print(BORDER)

    counts = df["sentiment"].value_counts()
    percentages = df["sentiment"].value_counts(normalize=True) * 100

    print(f"{'Sentiment':<12}{'Count':>10}{'Percent':>10}")
    print("-" * 32)

    for label in counts.index:
        print(f"{label:<12}{counts[label]:>10,}{percentages[label]:>9.2f}%")

    # Check whether the classes are balanced (within 5 percentage points)
    gap = abs(percentages.max() - percentages.min())

    if gap <= 5:
        print("\nClass balance : balanced (gap under 5%)")
    else:
        print(f"\nClass balance : imbalanced (gap of {gap:.2f}%)")

    # Bar chart of class counts
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", legend=False)

    for index, value in enumerate(counts.values):
        plt.text(index, value, f"{value:,}", ha="center", va="bottom")

    plt.title("Sentiment Distribution of IMDb Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    chart_path = os.path.join(IMAGES_DIR, "01_sentiment_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"\nChart saved: {chart_path}")
    print("Sentiment distribution analysis completed.")

    return counts

############################################################
# Function Name : analyse_review_lengths
# Description   : Calculate the word and character counts of
#                 every review, compare them by sentiment, and
#                 save a histogram of word counts.
# Input         : df - Cleaned DataFrame returned by clean_dataset()
# Output        : Saves images/02_review_length_distribution.png and
#                 returns the summary table
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def analyse_review_lengths(df):

    print("\n" + BORDER)
    print("STEP 6: REVIEW LENGTH ANALYSIS")
    print(BORDER)

    lengths = pd.DataFrame({
        "sentiment": df["sentiment"],
        "words": df["review"].str.split().str.len(),
        "characters": df["review"].str.len()
    })

    summary = (
        lengths.groupby("sentiment")[["words", "characters"]]
        .agg(["mean", "median"])
        .round(1)
    )

    print("Review length by sentiment:")
    print(summary.to_string())

    print(f"\nOverall average words      : {lengths['words'].mean():.1f}")
    print(f"Overall average characters : {lengths['characters'].mean():.1f}")

    # Histogram of word counts, capped at the 99th percentile
    # so a few very long reviews do not squeeze the chart
    upper_limit = lengths["words"].quantile(0.99)

    plt.figure(figsize=(9, 5))

    for label, color in [("positive", "tab:green"), ("negative", "tab:red")]:
        subset = lengths.loc[lengths["sentiment"] == label, "words"]
        plt.hist(
            subset.clip(upper=upper_limit),
            bins=50,
            alpha=0.6,
            label=label.capitalize(),
            color=color
        )

    plt.title("Distribution of Review Length by Sentiment")
    plt.xlabel("Number of Words per Review")
    plt.ylabel("Number of Reviews")
    plt.legend()
    plt.tight_layout()

    chart_path = os.path.join(IMAGES_DIR, "02_review_length_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"\nChart saved: {chart_path}")
    print("Review length analysis completed.")

    return summary


############################################################
# Function Name : explore_common_words
# Description   : Find the most frequent words in positive and
#                 negative reviews, save a bar chart for each
#                 class, and list the words that appear in the
#                 top words of both classes.
# Input         : df - DataFrame with the clean_review column
#                      (created by preprocess_reviews())
# Output        : Saves the two word charts in the images folder
#                 and returns the top words for each class
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def explore_common_words(df, top_n=20):

    print("\n" + BORDER)
    print("STEP 7: COMMON WORDS BY SENTIMENT")
    print(BORDER)

    top_words = {}

    for label, color in [("positive", "tab:green"), ("negative", "tab:red")]:

        words = (
            df.loc[df["sentiment"] == label, "clean_review"]
            .str.split()
            .explode()
            .value_counts()
        )

        top = words.head(top_n)
        top_words[label] = top

        print(f"\nTop {top_n} words in {label} reviews:")
        print(top.to_string())

        plt.figure(figsize=(9, 6))
        top.sort_values().plot(kind="barh", color=color)

        plt.title(f"Most Common Words in {label.capitalize()} Reviews")
        plt.xlabel("Number of Occurrences")
        plt.ylabel("Word")
        plt.tight_layout()

        chart_path = os.path.join(IMAGES_DIR, f"03_top_words_{label}.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()

        print(f"Chart saved: {chart_path}")

    # Words that appear in the top list of both classes
    shared = sorted(
        set(top_words["positive"].index) & set(top_words["negative"].index)
    )

    print(f"\nWords in the top {top_n} of both classes : {len(shared)}")
    print(", ".join(shared) if shared else "None")

    print("\nCommon words analysis completed.")

    return top_words


############################################################
# Function Name : clean_review_text
# Description   : Clean the raw text of one review: remove HTML
#                 tags and URLs, lowercase it, convert "n't"
#                 contractions to "not" (so negations survive),
#                 remove punctuation and digits, and collapse
#                 extra spaces.
# Input         : text - One raw review (string)
# Output        : Returns the cleaned string
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def clean_review_text(text):

    text = str(text)

    # Remove HTML tags such as <br />
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove web addresses
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = text.lower()

    # Keep negations: "don't" becomes "do not", "won't" becomes "will not"
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = text.replace("n't", " not")

    # Keep letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


############################################################
# Function Name : preprocess_reviews
# Description   : Apply the full text preprocessing to every review:
#                 1. Clean the text (clean_review_text)
#                 2. Tokenise on spaces
#                 3. Remove stopwords, but keep negation words
#                    such as "not", "no" and "never"
#                 4. Lemmatise each word (for example "movies" to
#                    "movie")
#                 Adds a clean_review column and a word count.
# Input         : df - DataFrame with the review column
# Output        : Returns a copy with clean_review and
#                 clean_word_count columns
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def preprocess_reviews(df):

    print("\n" + BORDER)
    print("STEP 8: TEXT PREPROCESSING")
    print(BORDER)

    # Words that carry negative meaning and must not be removed
    negation_words = {"not", "no", "nor", "never", "none", "nothing", "neither"}

    # Built here, after setup_project() has downloaded the NLTK data
    stop_words = set(stopwords.words("english")) - negation_words
    lemmatizer = WordNetLemmatizer()

    def remove_stopwords_and_lemmatise(text):

        tokens = [
            token for token in text.split()
            if token not in stop_words and len(token) > 1
        ]

        return " ".join(lemmatizer.lemmatize(token) for token in tokens)

    processed = df.copy()

    processed["clean_review"] = (
        processed["review"]
        .apply(clean_review_text)
        .apply(remove_stopwords_and_lemmatise)
    )

    processed["clean_word_count"] = (
        processed["clean_review"].str.split().str.len()
    )

    empty_reviews = int((processed["clean_word_count"] == 0).sum())

    print("Example before cleaning:")
    print(df["review"].iloc[0][:200])
    print("\nExample after cleaning:")
    print(processed["clean_review"].iloc[0][:200])

    print(f"\nAverage words after cleaning : {processed['clean_word_count'].mean():.1f}")
    print(f"Empty reviews after cleaning : {empty_reviews:,}")
    print("Negation words kept          : " + ", ".join(sorted(negation_words)))

    print("\nText preprocessing completed.")

    return processed

############################################################
# Function Name : encode_labels
# Description   : Convert the sentiment text into a binary label:
#                 positive = 1 and negative = 0.
# Input         : df - DataFrame with the sentiment column
# Output        : Returns a copy with a label column
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def encode_labels(df):

    print("\n" + BORDER)
    print("STEP 9: ENCODE TARGET LABELS")
    print(BORDER)

    encoded = df.copy()

    encoded["label"] = encoded["sentiment"].map({"positive": 1, "negative": 0})

    if encoded["label"].isnull().any():
        raise ValueError("Some sentiment values could not be encoded.")

    print("Mapping : positive = 1, negative = 0")
    print("\nLabel counts:")
    print(encoded["label"].value_counts().sort_index().to_string())

    print("\nLabel encoding completed.")

    return encoded


############################################################
# Function Name : create_features
# Description   : Define the two text representations used in the
#                 project: Bag of Words (for Naive Bayes) and
#                 TF-IDF (for the other models). The vectorizers
#                 are fitted inside the model pipelines on the
#                 training data only, which prevents data leakage.
# Input         : None
# Output        : Returns a dictionary of unfitted vectorizers
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def create_features():

    print("\n" + BORDER)
    print("STEP 10: CREATE TEXT FEATURES")
    print(BORDER)

    features = {
        "bag_of_words": CountVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2
        ),
        "tfidf": TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    }

    print("Bag of Words : word counts, 1 to 2 word phrases, max 50,000 features")
    print("TF-IDF       : weighted word counts, 1 to 2 word phrases, max 50,000 features")
    print("Vectorizers are fitted inside each pipeline on training data only.")

    print("\nFeature creation completed.")

    return features

############################################################
# Function Name : split_data
# Description   : Split the processed reviews into training and
#                 testing sets (80/20). The split is stratified so
#                 both sets keep the same positive/negative balance.
#                 The original review text of the test set is kept
#                 for error analysis.
# Input         : df - DataFrame with clean_review, label and review
# Output        : Returns a dictionary with the training and testing data
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def split_data(df):

    print("\n" + BORDER)
    print("STEP 11: SPLIT DATA")
    print(BORDER)

    X = df["clean_review"]
    y = df["label"]
    raw = df["review"]

    X_train, X_test, y_train, y_test, _, raw_test = train_test_split(
        X,
        y,
        raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training reviews : {len(X_train):,}")
    print(f"Testing reviews  : {len(X_test):,}")

    print("\nTraining label counts:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTesting label counts:")
    print(y_test.value_counts().sort_index().to_string())

    print("\nData split completed.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "raw_test": raw_test
    }

############################################################
# Function Name : build_baseline_model
# Description   : Build the baseline pipeline. It always predicts
#                 the most frequent class, so every real model must
#                 beat its score to be useful.
# Input         : features - Dictionary returned by create_features()
# Output        : Returns the baseline pipeline (not yet trained)
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def build_baseline_model(features):

    print("\n" + BORDER)
    print("STEP 12: BASELINE MODEL")
    print(BORDER)

    baseline = Pipeline([
        ("vectorizer", clone(features["tfidf"])),
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])

    print("Baseline : always predicts the most frequent class")
    print("Baseline pipeline created.")

    return baseline

############################################################
# Function Name : create_model_pipelines
# Description   : Build one pipeline for each classifier:
#                 Multinomial Naive Bayes (Bag of Words),
#                 Logistic Regression (TF-IDF),
#                 Linear SVM (TF-IDF), and
#                 Random Forest (TF-IDF).
# Input         : features - Dictionary returned by create_features()
# Output        : Returns a dictionary of model pipelines
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def create_model_pipelines(features):

    print("\n" + BORDER)
    print("STEP 13: CREATE MODEL PIPELINES")
    print(BORDER)

    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("vectorizer", clone(features["bag_of_words"])),
            ("classifier", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            ("vectorizer", clone(features["tfidf"])),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Linear SVM": Pipeline([
            ("vectorizer", clone(features["tfidf"])),
            ("classifier", LinearSVC(
                max_iter=5000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("vectorizer", clone(features["tfidf"])),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    for name in models:
        print(f"Created pipeline: {name}")

    print("\nModel pipelines created:", len(models))

    return models

############################################################
# Function Name : compare_models
# Description   : Compare all models with stratified 5-fold cross
#                 validation on the TRAINING data only. The test
#                 data is not used here. Saves the comparison table.
# Input         : models - Dictionary of pipelines (including baseline)
#                 split  - Dictionary returned by split_data()
# Output        : Returns the comparison table, sorted by F1 macro
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def compare_models(models, split):

    print("\n" + BORDER)
    print("STEP 14: MODEL COMPARISON")
    print(BORDER)

    X_train = split["X_train"]
    y_train = split["y_train"]

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc"
    }

    rows = []

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        row = {
            "Model": name,
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV F1 Macro": scores["test_f1_macro"].mean(),
            "CV ROC-AUC": scores["test_roc_auc"].mean()
        }

        rows.append(row)

        print(f"CV Accuracy : {row['CV Accuracy']:.4f}")
        print(f"CV F1 Macro : {row['CV F1 Macro']:.4f}")
        print(f"CV ROC-AUC  : {row['CV ROC-AUC']:.4f}")

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("CV F1 Macro", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "-" * 60)
    print("MODEL COMPARISON SUMMARY (5-fold CV on training data)")
    print("-" * 60)
    print(comparison_df.round(4).to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.round(4).to_csv(comparison_path, index=False)

    print(f"\nComparison saved: {comparison_path}")

    return comparison_df

############################################################
# Function Name : select_best_model
# Description   : Choose the best model from the cross-validation
#                 results. The baseline is excluded, since it is
#                 the reference, not a candidate. The test data is
#                 not used.
# Input         : comparison_df - Table returned by compare_models()
# Output        : Returns the name of the best model
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def select_best_model(comparison_df):

    print("\n" + BORDER)
    print("STEP 15: SELECT BEST MODEL")
    print(BORDER)

    candidates = comparison_df[comparison_df["Model"] != "Baseline"]

    best_row = candidates.loc[candidates["CV F1 Macro"].idxmax()]
    best_name = best_row["Model"]

    baseline_row = comparison_df[comparison_df["Model"] == "Baseline"].iloc[0]

    print(f"Selected Model      : {best_name}")
    print(f"CV Accuracy         : {best_row['CV Accuracy']:.4f}")
    print(f"CV F1 Macro         : {best_row['CV F1 Macro']:.4f}")
    print(f"CV ROC-AUC          : {best_row['CV ROC-AUC']:.4f}")
    print(f"Baseline CV F1 Macro: {baseline_row['CV F1 Macro']:.4f}")
    print("\nThe model is selected using cross-validation results on the training data only.")

    return best_name

############################################################
# Function Name : tune_best_model
# Description   : Search a small set of hyperparameters for the
#                 selected model using GridSearchCV on the TRAINING
#                 data only. The tuned settings are kept only if they
#                 beat the untuned model's cross-validation score.
#                 Saves the search results to results/tuning_results.csv.
# Input         : best_name     - Name of the selected model
#                 models        - Dictionary of pipelines
#                 split         - Dictionary returned by split_data()
#                 comparison_df - Table returned by compare_models()
# Output        : Returns (unfitted pipeline, parameters used).
#                 Parameters are an empty dictionary when the
#                 original settings are kept.
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def tune_best_model(best_name, models, split, comparison_df):

    print("\n" + BORDER)
    print("STEP 16: TUNE BEST MODEL")
    print(BORDER)

    param_grids = {
        "Multinomial Naive Bayes": {
            "vectorizer__max_features": [20000, 50000],
            "classifier__alpha": [0.1, 0.5, 1.0]
        },
        "Logistic Regression": {
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.5, 1.0, 4.0]
        },
        "Linear SVM": {
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 0.5, 1.0]
        },
        "Random Forest": {
            "classifier__max_depth": [None, 30],
            "classifier__min_samples_leaf": [1, 5]
        }
    }

    untuned_model = models[best_name]

    untuned_score = comparison_df.loc[
        comparison_df["Model"] == best_name, "CV F1 Macro"
    ].iloc[0]

    print(f"Model to tune        : {best_name}")
    print(f"Untuned CV F1 Macro  : {untuned_score:.4f}")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=clone(untuned_model),
        param_grid=param_grids[best_name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=False
    )

    search.fit(split["X_train"], split["y_train"])

    tuning_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    tuning_path = os.path.join(RESULTS_DIR, "tuning_results.csv")
    tuning_results.to_csv(tuning_path, index=False)

    print(f"\nBest tuned CV F1 Macro : {search.best_score_:.4f}")
    print("Best parameters        :")
    for key, value in search.best_params_.items():
        print(f"  {key} = {value}")
    print(f"Tuning results saved   : {tuning_path}")

    if search.best_score_ > untuned_score:
        tuned_model = clone(untuned_model).set_params(**search.best_params_)
        used_params = search.best_params_
        print("\nTuned settings are better and will be used.")
    else:
        tuned_model = clone(untuned_model)
        used_params = {}
        print("\nTuning did not improve the score. Original settings are kept.")

    print("\nModel tuning completed.")

    return tuned_model, used_params

############################################################
# Function Name : train_final_model
# Description   : Train the selected pipeline on the complete
#                 training set. The test set is not used.
# Input         : model - Unfitted pipeline from tune_best_model()
#                 split - Dictionary returned by split_data()
# Output        : Returns the trained pipeline
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def train_final_model(model, split):

    print("\n" + BORDER)
    print("STEP 17: TRAIN FINAL MODEL")
    print(BORDER)

    model.fit(split["X_train"], split["y_train"])

    print(f"Training reviews used : {len(split['X_train']):,}")
    print("Final model trained successfully.")

    return model

############################################################
# Function Name : save_model
# Description   : Save the trained pipeline (text vectorizer and
#                 classifier together) to the models folder, then
#                 reload it to confirm the saved file works.
# Input         : model - Trained pipeline from train_final_model()
# Output        : Returns the path of the saved model
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def save_model(model):

    print("\n" + BORDER)
    print("STEP 18: SAVE MODEL")
    print(BORDER)

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    # Reload and test a prediction to confirm the file works
    reloaded = joblib.load(MODEL_PATH)
    test_prediction = reloaded.predict(pd.Series(["a great and enjoyable movie"]))

    print(f"Reload check passed (test prediction: {int(test_prediction[0])})")

    return MODEL_PATH

############################################################
# Function Name : get_positive_scores
# Description   : Helper function. Returns a score between 0 and 1
#                 for the positive class. Uses predict_proba when
#                 the model has it. Otherwise it converts the SVM
#                 decision values with a sigmoid. The sigmoid is
#                 monotonic, so ROC-AUC is unchanged.
# Input         : model - Trained pipeline
#                 X     - Text data
# Output        : Returns a numpy array of scores
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def get_positive_scores(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    decision = model.decision_function(X)

    return 1 / (1 + np.exp(-decision))


############################################################
# Function Name : evaluate_model
# Description   : Evaluate the trained model on the unseen test set:
#                 accuracy, precision, recall, F1 for each class,
#                 macro F1, ROC-AUC, the classification report,
#                 and the confusion matrix.
# Input         : model - Trained pipeline
#                 split - Dictionary returned by split_data()
# Output        : Returns a dictionary of metrics and predictions
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def evaluate_model(model, split):

    print("\n" + BORDER)
    print("STEP 19: EVALUATE FINAL MODEL ON TEST DATA")
    print(BORDER)

    X_test = split["X_test"]
    y_test = split["y_test"]

    y_pred = model.predict(X_test)
    y_score = get_positive_scores(model, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_positive = f1_score(y_test, y_pred, pos_label=1)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_score)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["negative", "positive"],
        digits=4
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"Test reviews     : {len(y_test):,}")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision (pos)  : {precision:.4f}")
    print(f"Recall (pos)     : {recall:.4f}")
    print(f"F1 (pos)         : {f1_positive:.4f}")
    print(f"F1 negative      : {f1_per_class[0]:.4f}")
    print(f"F1 positive      : {f1_per_class[1]:.4f}")
    print(f"F1 macro         : {f1_macro:.4f}")
    print(f"ROC-AUC          : {roc_auc:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 Pred negative   Pred positive")
    print(f"Actual negative {cm[0, 0]:>14,}{cm[0, 1]:>16,}")
    print(f"Actual positive {cm[1, 0]:>14,}{cm[1, 1]:>16,}")

    print("\nFinal evaluation completed.")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_positive": float(f1_positive),
        "f1_negative": float(f1_per_class[0]),
        "f1_positive_class": float(f1_per_class[1]),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_score": y_score
    }

############################################################
# Function Name : analyse_errors
# Description   : Study the reviews the model got wrong. Count the
#                 false positives and false negatives, check how
#                 often negation and contrast words appear in the
#                 errors compared with the correct predictions,
#                 and save the misclassified reviews to a CSV file.
# Input         : evaluation - Dictionary returned by evaluate_model()
#                 split      - Dictionary returned by split_data()
# Output        : Saves results/misclassified_reviews.csv and
#                 returns the table of misclassified reviews
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def analyse_errors(evaluation, split):

    print("\n" + BORDER)
    print("STEP 20: ERROR ANALYSIS")
    print(BORDER)

    errors = pd.DataFrame({
        "review": split["raw_test"].values,
        "actual": evaluation["y_test"].values,
        "predicted": evaluation["y_pred"],
        "positive_score": np.round(evaluation["y_score"], 4)
    })

    errors["actual_sentiment"] = errors["actual"].map({1: "positive", 0: "negative"})
    errors["predicted_sentiment"] = errors["predicted"].map({1: "positive", 0: "negative"})

    false_positives = errors[(errors["actual"] == 0) & (errors["predicted"] == 1)]
    false_negatives = errors[(errors["actual"] == 1) & (errors["predicted"] == 0)]
    misclassified = errors[errors["actual"] != errors["predicted"]].copy()
    correct = errors[errors["actual"] == errors["predicted"]]

    print(f"Total misclassified reviews : {len(misclassified):,}")
    print(f"False positives (negative predicted as positive) : {len(false_positives):,}")
    print(f"False negatives (positive predicted as negative) : {len(false_negatives):,}")

    # Check whether certain word types appear more often in errors
    patterns = {
        "negation (not, no, never, n't)": r"\b(?:not|no|never)\b|n't",
        "contrast (but, however, although, though)": r"\b(?:but|however|although|though)\b"
    }

    print("\nWord patterns in misclassified vs correct reviews:")
    print(f"{'Pattern':<42}{'In errors':>12}{'In correct':>13}")
    print("-" * 67)

    for name, pattern in patterns.items():

        error_rate = misclassified["review"].str.lower().str.contains(pattern, regex=True).mean() * 100
        correct_rate = correct["review"].str.lower().str.contains(pattern, regex=True).mean() * 100

        print(f"{name:<42}{error_rate:>11.2f}%{correct_rate:>12.2f}%")

    print("\nExample false positive (negative review predicted as positive):")
    if not false_positives.empty:
        print(false_positives["review"].iloc[0][:250])

    print("\nExample false negative (positive review predicted as negative):")
    if not false_negatives.empty:
        print(false_negatives["review"].iloc[0][:250])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    errors_path = os.path.join(RESULTS_DIR, "misclassified_reviews.csv")

    misclassified[[
        "review", "actual_sentiment", "predicted_sentiment", "positive_score"
    ]].to_csv(errors_path, index=False)

    print(f"\nMisclassified reviews saved: {errors_path}")
    print("Error analysis completed.")

    return misclassified

############################################################
# Function Name : plot_model_results
# Description   : Create the project charts:
#                 04 model comparison (cross-validation scores),
#                 confusion matrix, ROC curve, and the top words
#                 that push predictions towards each sentiment.
# Input         : comparison_df - Table returned by compare_models()
#                 evaluation    - Dictionary returned by evaluate_model()
#                 model         - Trained pipeline
# Output        : Saves the charts in the images folder
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def plot_model_results(comparison_df, evaluation, model):

    print("\n" + BORDER)
    print("STEP 21: PLOT MODEL RESULTS")
    print(BORDER)

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. Model comparison chart
    chart_data = comparison_df.set_index("Model")[
        ["CV Accuracy", "CV F1 Macro", "CV ROC-AUC"]
    ]

    ax = chart_data.plot(kind="bar", figsize=(11, 6), colormap="Set2")
    ax.set_title("Model Comparison (5-fold cross validation on training data)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, "04_model_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Chart saved: {path}")

    # 2. Confusion matrix
    cm = evaluation["confusion_matrix"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"]
    )
    plt.title("Confusion Matrix (Test Data)")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Chart saved: {path}")

    # 3. ROC curve
    fpr, tpr, _ = roc_curve(evaluation["y_test"], evaluation["y_score"])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {evaluation['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    plt.title("ROC Curve (Test Data)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, "roc_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Chart saved: {path}")

    # 4. Top words for the final model (linear models and Naive Bayes only)
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
    elif hasattr(classifier, "feature_log_prob_"):
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:
        weights = None

    if weights is not None:

        order = np.argsort(weights)

        negative_words = feature_names[order[:15]]
        negative_weights = weights[order[:15]]

        positive_words = feature_names[order[-15:]]
        positive_weights = weights[order[-15:]]

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        axes[0].barh(positive_words, positive_weights, color="tab:green")
        axes[0].set_title("Words pushing towards POSITIVE")
        axes[0].set_xlabel("Weight")

        axes[1].barh(negative_words, negative_weights, color="tab:red")
        axes[1].set_title("Words pushing towards NEGATIVE")
        axes[1].set_xlabel("Weight")

        plt.tight_layout()

        path = os.path.join(IMAGES_DIR, "top_features.png")
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"Chart saved: {path}")
        print("\nTop positive words:", ", ".join(positive_words[::-1]))
        print("Top negative words:", ", ".join(negative_words))

    else:
        print("Top words chart skipped: this model does not give word weights.")

    print("\nModel result charts completed.")

############################################################
# Function Name : predict_sentiment
# Description   : Predict the sentiment of new movie reviews typed
#                 by the user. Each text goes through the same
#                 preprocessing as the training data before prediction.
# Input         : model - Trained pipeline
#                 texts - List of review strings (optional)
# Output        : Returns a list of dictionaries with the prediction
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def predict_sentiment(model, texts=None):

    print("\n" + BORDER)
    print("STEP 22: PREDICT NEW REVIEWS")
    print(BORDER)

    if texts is None:
        texts = [
            "The acting was superb and the story kept me hooked from start to finish.",
            "I was bored to tears and the plot made no sense at all.",
            "Not a bad movie, but not a good one either.",
            "This film is not good at all, I would never watch it again.",
            "A beautiful, moving film with a brilliant ending."
        ]

    new_reviews = pd.DataFrame({"review": texts})
    processed = preprocess_reviews(new_reviews)["clean_review"]

    predictions = model.predict(processed)
    scores = get_positive_scores(model, processed)

    results = []

    print()
    for text, prediction, score in zip(texts, predictions, scores):

        sentiment = "positive" if prediction == 1 else "negative"
        confidence = score if prediction == 1 else 1 - score

        results.append({
            "review": text,
            "sentiment": sentiment,
            "confidence": float(confidence)
        })

        print(f"Review     : {text}")
        print(f"Sentiment  : {sentiment.upper()} (confidence {confidence:.2%})")
        print("-" * 60)

    print("Prediction completed.")

    return results

############################################################
# Function Name : save_outputs
# Description   : Save the final test metrics and the project
#                 insights (best model, parameters, scores, and
#                 recommendations) to the results folder.
# Input         : comparison_df - Table returned by compare_models()
#                 evaluation    - Dictionary returned by evaluate_model()
#                 split         - Dictionary returned by split_data()
#                 best_name     - Name of the selected model
#                 best_params   - Parameters used (may be empty)
# Output        : Saves the metrics CSV and insights JSON
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def save_outputs(comparison_df, evaluation, split, best_name, best_params):

    print("\n" + BORDER)
    print("STEP 22: SAVE OUTPUTS")
    print(BORDER)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Test metrics as CSV
    metrics_df = pd.DataFrame([
        ("Accuracy", evaluation["accuracy"]),
        ("Precision (positive)", evaluation["precision"]),
        ("Recall (positive)", evaluation["recall"]),
        ("F1 (positive)", evaluation["f1_positive"]),
        ("F1 (negative)", evaluation["f1_negative"]),
        ("F1 macro", evaluation["f1_macro"]),
        ("ROC-AUC", evaluation["roc_auc"])
    ], columns=["Metric", "Score"])

    metrics_path = os.path.join(RESULTS_DIR, "model_evaluation_metrics.csv")
    metrics_df.round(4).to_csv(metrics_path, index=False)

    print(f"Evaluation metrics saved: {metrics_path}")

    # 2. Insights as JSON
    best_row = comparison_df[comparison_df["Model"] == best_name].iloc[0]

    recommendations = []

    if evaluation["accuracy"] >= 0.85:
        recommendations.append(
            "The model classifies most reviews correctly on unseen data."
        )
    else:
        recommendations.append(
            "Accuracy is below 85%. Review the preprocessing and features."
        )

    if abs(evaluation["f1_positive"] - evaluation["f1_negative"]) > 0.05:
        recommendations.append(
            "Performance differs between the two classes. Check the per-class F1 scores."
        )

    recommendations.append(
        "Review the misclassified reviews to find patterns such as sarcasm or mixed opinions."
    )

    insights = {
        "best_model": best_name,
        "tuned_parameters": {key: str(value) for key, value in best_params.items()}
        if best_params else "original parameters kept",
        "dataset_reviews": int(len(split["X_train"]) + len(split["X_test"])),
        "training_reviews": int(len(split["X_train"])),
        "testing_reviews": int(len(split["X_test"])),
        "cross_validation": {
            "accuracy": float(best_row["CV Accuracy"]),
            "f1_macro": float(best_row["CV F1 Macro"]),
            "roc_auc": float(best_row["CV ROC-AUC"])
        },
        "test_metrics": {
            "accuracy": evaluation["accuracy"],
            "precision_positive": evaluation["precision"],
            "recall_positive": evaluation["recall"],
            "f1_positive": evaluation["f1_positive"],
            "f1_negative": evaluation["f1_negative"],
            "f1_macro": evaluation["f1_macro"],
            "roc_auc": evaluation["roc_auc"]
        },
        "recommendations": recommendations
    }

    insights_path = os.path.join(RESULTS_DIR, "sentiment_insights.json")

    with open(insights_path, "w", encoding="utf-8") as file:
        json.dump(insights, file, indent=4, ensure_ascii=False)

    print(f"Project insights saved: {insights_path}")
    print("\nAll outputs saved successfully.")

    return {
        "metrics": metrics_path,
        "insights": insights_path
    }

############################################################
# Function Name : display_summary
# Description   : Print the final project summary: the selected
#                 model, the tuned settings, the cross-validation
#                 and test results, and the saved files.
# Input         : best_name, best_params, comparison_df,
#                 evaluation, split, output_paths
# Output        : Displays the project summary
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def display_summary(best_name, best_params, comparison_df,
                    evaluation, split, output_paths):

    print("\n" + "=" * 70)
    print("       IMDb SENTIMENT ANALYSIS - PROJECT SUMMARY")
    print("=" * 70)

    print("\n1. PROJECT OVERVIEW")
    print("-" * 70)
    print("Problem        : Binary sentiment classification (positive / negative)")
    print("Dataset        : IMDb movie reviews")
    print(f"Training set   : {len(split['X_train']):,} reviews")
    print(f"Testing set    : {len(split['X_test']):,} reviews")

    print("\n2. SELECTED MODEL")
    print("-" * 70)
    print(f"Best model     : {best_name}")

    if best_params:
        print("Tuned settings :")
        for key, value in best_params.items():
            print(f"  {key} = {value}")
    else:
        print("Tuned settings : original parameters kept")

    print("\n3. CROSS-VALIDATION RESULTS (training data)")
    print("-" * 70)
    best_row = comparison_df[comparison_df["Model"] == best_name].iloc[0]
    print(f"Accuracy       : {best_row['CV Accuracy']:.4f}")
    print(f"F1 macro       : {best_row['CV F1 Macro']:.4f}")
    print(f"ROC-AUC        : {best_row['CV ROC-AUC']:.4f}")

    print("\n4. FINAL TEST RESULTS (unseen data)")
    print("-" * 70)
    print(f"Accuracy       : {evaluation['accuracy']:.4f}")
    print(f"Precision (pos): {evaluation['precision']:.4f}")
    print(f"Recall (pos)   : {evaluation['recall']:.4f}")
    print(f"F1 positive    : {evaluation['f1_positive']:.4f}")
    print(f"F1 negative    : {evaluation['f1_negative']:.4f}")
    print(f"F1 macro       : {evaluation['f1_macro']:.4f}")
    print(f"ROC-AUC        : {evaluation['roc_auc']:.4f}")

    print("\n5. GENERATED OUTPUT FILES")
    print("-" * 70)
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print("\n" + "=" * 70)
    print("       PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


############################################################
# Function Name : display_footer
# Description   : Display the final completion message.
# Input         : None
# Output        : Displays the project footer
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def display_footer():

    print("\n" + "=" * 60)
    print("   IMDb SENTIMENT ANALYSIS - NLP CLASSIFICATION PROJECT")
    print("=" * 60)
    print("Dataset Loading     : Completed")
    print("Data Cleaning       : Completed")
    print("Exploratory Analysis: Completed")
    print("Text Preprocessing  : Completed")
    print("Model Training      : Completed")
    print("Model Evaluation    : Completed")
    print("Error Analysis      : Completed")
    print("Predictions         : Completed")
    print("Output Files        : Saved")
    print("-" * 60)
    print("   THANK YOU FOR USING THIS PROJECT!")
    print("=" * 60)

############################################################
# Main Function
# Description   : Run the complete pipeline in order. Every step
#                 calls one of the functions above.
# Author        : Ishwari Vijaykumar Surve
# Date          : 23/10/2026
############################################################

def main():

    display_header()

    # Setup and loading
    setup_project()

    print("Dataset path:", DATA_PATH)
    print("File exists:", os.path.isfile(DATA_PATH))

     # Load the dataset
    df = load_dataset(DATA_PATH)

    # Understanding the dataset
    understand_dataset(df)
    check_structure(df)

    # Cleaning and EDA
    clean = clean_dataset(df)
    analyse_sentiment_distribution(clean)
    analyse_review_lengths(clean)

    # Text preprocessing (encode_labels runs after cleaning)
    processed = preprocess_reviews(clean)
    processed = encode_labels(processed)
    explore_common_words(processed)

    # Features and splitting
    features = create_features()
    split = split_data(processed)

    # Model development
    models = {
        "Baseline": build_baseline_model(features),
        **create_model_pipelines(features)
    }

    comparison_df = compare_models(models, split)
    best_name = select_best_model(comparison_df)

    tuned_model, best_params = tune_best_model(
        best_name, models, split, comparison_df
    )
    final_model = train_final_model(tuned_model, split)
    save_model(final_model)

    # Evaluation and analysis
    evaluation = evaluate_model(final_model, split)
    analyse_errors(evaluation, split)
    plot_model_results(comparison_df, evaluation, final_model)

    # Predictions and outputs
    predict_sentiment(final_model)

    output_paths = save_outputs(
        comparison_df, evaluation, split, best_name, best_params
    )

    display_summary(
        best_name, best_params, comparison_df,
        evaluation, split, output_paths
    )

    display_footer()


if __name__ == "__main__":
    main()
