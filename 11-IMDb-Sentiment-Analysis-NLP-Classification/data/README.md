# IMDb Movie Review Dataset

## 📌 Dataset Overview

This project uses the **IMDB Dataset of 50K Movie Reviews** for sentiment analysis with Natural Language Processing (NLP) and Machine Learning.

The dataset contains 50,000 movie reviews, each labelled as either **positive** or **negative**.
 
## 📥 Dataset Download

The dataset is not included in this repository. Download it from Kaggle:

🔗 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

**Dataset Name:** IMDB Dataset of 50K Movie Reviews
**Source:** Kaggle
**File Name:** `IMDB_Dataset.csv`

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Total Records | 50,000 movie reviews |
| Input Feature | `review` |
| Target Column | `sentiment` |
| Sentiment Classes | Positive and Negative (25,000 each) |
| Problem Type | Binary Classification |
| Application | NLP and Sentiment Analysis |

## 📂 Dataset Placement

After downloading, place the CSV file in this folder:
data/raw/IMDB_Dataset.csv


Expected project structure:

11-IMDb-Sentiment-Analysis-NLP-Classification/
├── data/
│ ├── README.md
│ └── raw/
│ └── IMDB_Dataset.csv
├── images/
├── models/
├── results/
├── src/
│ └── main.py
└── README.md


**Important:** The filename must match `IMDB_Dataset.csv` exactly. If Kaggle downloads it with a space (`IMDB Dataset.csv`), or as an Excel file, rename it or save it as CSV before running the code.

## 🎯 Purpose

The dataset is used to:

- Explore movie review text and sentiment labels.
- Clean and preprocess textual data.
- Convert text into features using Bag of Words and TF-IDF.
- Train and compare classification models.
- Evaluate model performance.
- Predict the sentiment of new movie reviews.

## 📝 Attribution

- **Kaggle dataset:** Lakshmipathi N (Kaggle)
- **Original dataset:** Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011. https://ai.stanford.edu/~amaas/data/sentiment/

Please refer to the Kaggle dataset page for its licence and usage conditions.

## Author
**Ishwari Surve**
