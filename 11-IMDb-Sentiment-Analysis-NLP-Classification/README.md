# IMDb Movie Review Sentiment Analysis (NLP Classification)

An end-to-end NLP project that classifies movie reviews as **positive** or **negative**. It covers data cleaning, text preprocessing, feature creation with TF-IDF, comparison of classical machine learning models, hyperparameter tuning, error analysis, and saving a reusable prediction pipeline.

## Project Overview

- **Dataset:** IMDB Dataset of 50K Movie Reviews (50,000 reviews, 25,000 positive and 25,000 negative)
- **Problem type:** Binary classification
- **Target:** `sentiment` (positive = 1, negative = 0)
- **Main approach:** Text cleaning, stopword removal (negations kept), lemmatisation, and TF-IDF features

## Models Compared

| Model | Text Features |
|---|---|
| Baseline (most frequent class) | TF-IDF |
| Multinomial Naive Bayes | Bag of Words |
| Logistic Regression | TF-IDF |
| Linear SVM | TF-IDF |
| Random Forest | TF-IDF |

All models are compared using stratified 5-fold cross-validation on the training data. The test set is used only for the final evaluation.

## Workflow

1. Set up the project folders and NLTK resources
2. Load and inspect the dataset
3. Check structure (missing values, duplicates, label validity)
4. Clean the dataset and analyse sentiment distribution
5. Analyse review lengths and common words by sentiment
6. Preprocess text and encode labels
7. Create text features and split the data (80/20, stratified)
8. Compare models with cross-validation and select the best one
9. Tune the selected model on training data only
10. Train the final model, save it, and evaluate it on the test set
11. Analyse errors and plot results
12. Predict sentiment for new reviews and save the outputs

## Project Structure
11-IMDb-Sentiment-Analysis-NLP-Classification/
├── data/
│ ├── README.md
│ └── raw/
│ └── IMDB_Dataset.csv (not included, see data/README.md)
├── images/ charts saved by the program
├── models/
│ └── sentiment_model.joblib trained pipeline (created on run)
├── results/ metrics, comparison and insight files
├── src/
│ └── main.py full pipeline
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md


## Setup

**1. Clone the repository**

```bash
git clone https://github.com/ishwari-surve/11-IMDb-Sentiment-Analysis-NLP-Classification.git
cd 11-IMDb-Sentiment-Analysis-NLP-Classification
```

**2. Install the dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Download the dataset**

Download `IMDB_Dataset.csv` from Kaggle and place it in `data/raw/`. See [data/README.md](data/README.md) for the link and details.

## How to Run

From the project root:

```bash
python src/main.py
```

The first run downloads the NLTK resources (stopwords, WordNet, and omw-1.4), so it needs an internet connection. The full run takes several minutes, mainly because of Random Forest and the tuning step.

## Outputs

| Folder | File | Description |
|---|---|---|
| `images/` | `01_sentiment_distribution.png` | Class balance chart |
| `images/` | `02_review_length_distribution.png` | Review length by sentiment |
| `images/` | `03_top_words_positive.png`, `03_top_words_negative.png` | Most common words per sentiment |
| `images/` | `04_model_comparison.png` | Cross-validation comparison |
| `images/` | `confusion_matrix.png` | Test set confusion matrix |
| `images/` | `roc_curve.png` | ROC curve on test data |
| `images/` | `top_features.png` | Words that push predictions towards each sentiment |
| `models/` | `sentiment_model.joblib` | Saved pipeline (vectorizer and classifier together) |
| `results/` | `model_comparison.csv` | Cross-validation results for all models |
| `results/` | `tuning_results.csv` | Hyperparameter search results |
| `results/` | `model_evaluation_metrics.csv` | Final test metrics |
| `results/` | `misclassified_reviews.csv` | Reviews the model got wrong |
| `results/` | `sentiment_insights.json` | Summary of the best model and recommendations |

## Results

| Metric (test set) | Score |
|---|---|
| Accuracy | *to be added after run* |
| F1 (positive) | *to be added after run* |
| F1 (negative) | *to be added after run* |
| ROC-AUC | *to be added after run* |

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, Joblib

## Notes

- This is a learning project. It is not intended for commercial sentiment monitoring.
- The dataset is not included in the repository. Use it only under the terms stated on its Kaggle page.

## Author

**Ishwari Vijaykumar Surve**
GitHub: [ishwari-surve](https://github.com/ishwari-surve)

## Licence

This project is licensed under the terms in the [LICENSE](LICENSE) file.
