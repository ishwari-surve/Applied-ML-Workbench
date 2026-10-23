# IMDb Movie Review Sentiment Analysis using NLP and Machine Learning

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies IMDb movie reviews as **positive** or **negative**. The project covers data cleaning, text preprocessing, feature extraction, model comparison, hyperparameter tuning, error analysis, and saving a reusable prediction pipeline.

## Project Overview

- **Project Type:** Natural Language Processing and Binary Classification
- **Dataset:** IMDb Dataset of 50K Movie Reviews
- **Dataset Size:** 50,000 reviews
- **Target Variable:** `sentiment`
- **Classes:** Positive and Negative
- **Feature Extraction:** Bag of Words and TF-IDF
- **Model Evaluation:** Stratified 5-fold cross-validation and held-out test-set evaluation

The objective is to build and evaluate machine learning models that can identify the sentiment expressed in a movie review based on its textual content.

## Dataset Information

The dataset contains movie reviews and their corresponding sentiment labels.

| Attribute | Description |
|---|---|
| Total reviews | 50,000 |
| Positive reviews | 25,000 |
| Negative reviews | 25,000 |
| Input feature | `review` |
| Target column | `sentiment` |
| Classification type | Binary classification |

The sentiment labels are encoded as follows:

- `positive` → `1`
- `negative` → `0`

**Dataset location:** `data/raw/IMDB_Dataset.csv`

The dataset is not included in this repository. Refer to [`data/README.md`](data/README.md) for download instructions.

## Machine Learning Models

The following models are compared to identify the most suitable classifier for sentiment prediction.

| Model | Text Features |
|---|---|
| Dummy Classifier (Baseline) | TF-IDF |
| Multinomial Naive Bayes | Bag of Words |
| Logistic Regression | TF-IDF |
| Linear Support Vector Machine (Linear SVM) | TF-IDF |
| Random Forest Classifier | TF-IDF |

All models are compared using **stratified 5-fold cross-validation on the training data**. The best model is selected using the F1-score, followed by hyperparameter tuning. The held-out test set is reserved for final model evaluation.

## Project Workflow

1. **Project Setup:** Create the required directories and download the necessary NLTK resources.
2. **Data Loading:** Load the IMDb dataset and inspect its dimensions and columns.
3. **Data Validation:** Examine missing values, duplicate records, data types, and sentiment labels.
4. **Data Cleaning:** Handle invalid records and prepare the dataset for analysis.
5. **Exploratory Data Analysis:** Analyze sentiment distribution, review lengths, and common words.
6. **Text Preprocessing:** Clean review text, remove stopwords while preserving negations, and apply lemmatization.
7. **Label Encoding:** Convert sentiment labels into numerical values.
8. **Feature Extraction:** Generate text features using Bag of Words and TF-IDF.
9. **Train-Test Split:** Divide the dataset into training and testing sets using an 80:20 stratified split.
10. **Model Comparison:** Compare the baseline and machine learning models using cross-validation.
11. **Hyperparameter Tuning:** Optimize the selected model using GridSearchCV.
12. **Final Model Training:** Train the selected model using the training data.
13. **Model Evaluation:** Evaluate performance on the held-out test set.
14. **Error Analysis:** Inspect misclassified reviews to understand prediction errors.
15. **Visualization:** Generate charts for sentiment distribution, model performance, and evaluation results.
16. **Prediction and Persistence:** Predict sentiment for new reviews and save the trained model and analysis outputs.

## Project Structure

```text
11-IMDb-Sentiment-Analysis-NLP-Classification/
│
├── data/
│   ├── README.md
│   └── raw/
│       └── IMDB_Dataset.csv
│
├── images/
│   ├── 01_sentiment_distribution.png
│   ├── 02_review_length_distribution.png
│   ├── 03_top_words_positive.png
│   ├── 03_top_words_negative.png
│   ├── 04_model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── top_features.png
│
├── models/
│   └── sentiment_model.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── tuning_results.csv
│   ├── model_evaluation_metrics.csv
│   ├── misclassified_reviews.csv
│   └── sentiment_insights.json
│
├── src/
│   └── main.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

*Note: The model, image, and results files are generated when the program runs. The dataset is not committed to the repository. Actual generated filenames depend on the implementation.*

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Dataset manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **NLTK** — Text preprocessing, stopwords, and lemmatization
- **Scikit-learn** — Feature extraction, machine learning, cross-validation, tuning, and evaluation
- **Joblib** — Saving and loading the trained model pipeline

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ishwari-surve/11-IMDb-Sentiment-Analysis-NLP-Classification.git
cd 11-IMDb-Sentiment-Analysis-NLP-Classification
```

### 2. Install Dependencies

Ensure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 3. Download the Dataset

Download the IMDb movie review dataset from Kaggle and place the CSV file at:

```text
data/raw/IMDB_Dataset.csv
```

Refer to [`data/README.md`](data/README.md) for the dataset download link and instructions.

Ensure that the filename is exactly `IMDB_Dataset.csv`.

### 4. Run the Project

From the project root directory, execute:

```bash
python src/main.py
```

The program will run the configured data analysis, preprocessing, model training, evaluation, prediction, and output-generation workflow.

**First-run note:** NLTK may need to download resources such as `stopwords`, `wordnet`, and `omw-1.4`. An internet connection may be required if these resources are not already installed. Execution time depends on your hardware and model-tuning configuration.

## Generated Outputs

### Visualizations — `images/`

| File | Description |
|---|---|
| `01_sentiment_distribution.png` | Distribution of positive and negative reviews |
| `02_review_length_distribution.png` | Review length distribution by sentiment |
| `03_top_words_positive.png` | Common words in positive reviews |
| `03_top_words_negative.png` | Common words in negative reviews |
| `04_model_comparison.png` | Comparison of model performance |
| `confusion_matrix.png` | Final model's test-set classification results |
| `roc_curve.png` | ROC curve on the test set |
| `top_features.png` | Important text features, where supported by the selected model |

### Trained Model — `models/`

| File | Description |
|---|---|
| `sentiment_model.joblib` | Saved prediction pipeline containing the fitted text vectorizer and classifier |

### Analysis Results — `results/`

| File | Description |
|---|---|
| `model_comparison.csv` | Cross-validation results for the compared models |
| `tuning_results.csv` | Hyperparameter tuning results |
| `model_evaluation_metrics.csv` | Final test-set evaluation metrics |
| `misclassified_reviews.csv` | Reviews incorrectly classified by the model |
| `sentiment_insights.json` | Summary of model results and sentiment analysis insights |

*The filenames above describe the intended outputs; confirm them against the files generated by your current implementation.*

## Model Evaluation

The final model is evaluated on the held-out test set using the following metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of correctly classified reviews |
| Precision | Proportion of predicted positive cases that are actually positive |
| Recall | Proportion of actual positive cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Breakdown of correct and incorrect classifications |
| ROC-AUC | Ability to distinguish between positive and negative reviews using prediction scores |

### Test Results

| Metric | Score |
|---|---|
| Accuracy | To be updated after execution |
| Positive-class F1-score | To be updated after execution |
| Negative-class F1-score | To be updated after execution |
| ROC-AUC | To be updated after execution |

Actual results will be added after the complete pipeline runs successfully. No performance scores are assumed in advance.

## Example Sentiment Prediction

**Input review:**

```text
The movie was excellent, with a compelling story and brilliant acting.
```

**Expected sentiment:**

```text
Positive
```

**Input review:**

```text
The story was boring, and the acting was disappointing.
```

**Expected sentiment:**

```text
Negative
```

These are illustrative examples. Actual predictions depend on the trained model.

## Key Learning Outcomes

- Applying NLP techniques to real-world text classification.
- Cleaning and validating a large text dataset.
- Understanding Bag of Words and TF-IDF representations.
- Building and comparing classical machine learning classifiers.
- Using stratified cross-validation and hyperparameter tuning.
- Evaluating classification models with multiple performance metrics.
- Performing error analysis and interpreting model behavior.
- Saving and reusing a complete text-classification pipeline.

## Limitations

- The model learns from the patterns and labels present in the training dataset.
- Sarcasm, subtle context, mixed sentiment, and ambiguous expressions can lead to incorrect predictions.
- Performance depends on the preprocessing strategy, feature representation, and selected classifier.
- The project is intended for learning and experimentation, not as a guarantee of accurate sentiment detection in every real-world scenario.

## Future Enhancements

- Compare TF-IDF with word embeddings and transformer-based models.
- Experiment with deep learning approaches such as LSTM and BERT.
- Develop an interactive sentiment analysis application using Streamlit.
- Deploy the model through a REST API.
- Add batch prediction and support for new review datasets.
- Monitor performance on additional datasets to assess generalization.

## Author

**Ishwari Vijaykumar Surve**

- GitHub: [ishwari-surve](https://github.com/ishwari-surve)
- Project Repository: [IMDb Movie Review Sentiment Analysis](https://github.com/ishwari-surve/11-IMDb-Sentiment-Analysis-NLP-Classification)

## License

This project is licensed under the terms specified in the [`LICENSE`](LICENSE) file.

The dataset is not included in this repository. Please follow the dataset provider's terms and conditions when downloading or using it.
