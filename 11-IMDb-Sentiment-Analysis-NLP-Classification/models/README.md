# Models

This folder holds the trained sentiment model saved by `src/main.py`.

## Files

| File | Description |
|---|---|
| `sentiment_model.joblib` | Trained pipeline: TF-IDF vectorizer and Linear SVM classifier, saved together |

## Model Details

| Item | Value |
|---|---|
| Algorithm | Linear Support Vector Machine (`LinearSVC`) |
| Text features | TF-IDF (`TfidfVectorizer`) |
| Ngram range | (1, 2), single words and two-word phrases |
| Max features | 50,000 |
| Min document frequency | 2 |
| Sublinear TF | Yes |
| Regularisation (C) | 0.5 (selected by tuning on training data) |
| Training data | 39,665 reviews (80% of the cleaned dataset) |
| Labels | 1 = positive, 0 = negative |

## Test Performance

Measured on 9,917 unseen reviews:

| Metric | Score |
|---|---|
| Accuracy | 0.9090 |
| F1 (positive) | 0.9101 |
| F1 (negative) | 0.9080 |
| ROC-AUC | 0.9701 |

## How the Model Was Chosen

Five models were compared with 5-fold stratified cross-validation on the training data. Linear SVM scored highest on CV F1 macro (0.9043). After tuning, the score rose to 0.9066. The test set was not used for selection.

## Loading the Model

The file contains the full pipeline, so the vectorizer and classifier load together.

```python
import joblib

model = joblib.load("models/sentiment_model.joblib")
```

## Using the Model on New Text

The pipeline does not include the text cleaning steps. New reviews must go through the same preprocessing used in training before prediction: `clean_review_text()` and `preprocess_reviews()` in `src/main.py`. The code in `predict_sentiment()` already does this.

## Compatibility

The model was saved with the library versions listed in `requirements.txt`. Load it with the same or a newer scikit-learn version. Loading with a much older version may fail.

## Regenerating the Model

Running `python src/main.py` retrains the model from the dataset and overwrites this file.
