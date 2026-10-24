# Results

This folder holds the numerical results saved by `src/main.py`. Each file is written when the program runs, so rerunning the script regenerates them.

## Files

| File | Description |
|---|---|
| `model_comparison.csv` | 5-fold cross-validation scores on the training data for all five models |
| `tuning_results.csv` | Every hyperparameter combination tried during tuning, with its CV F1 macro score and rank |
| `model_evaluation_metrics.csv` | Final scores of the selected model on the unseen test set |
| `misclassified_reviews.csv` | Test reviews the model labelled incorrectly, with the actual and predicted sentiment and the positive score |
| `sentiment_insights.json` | Summary of the selected model, tuned settings, cross-validation and test scores, and recommendations |

## Key Results

**Model comparison (5-fold CV, training data):**

| Model | CV Accuracy | CV F1 Macro | CV ROC-AUC |
|---|---|---|---|
| Linear SVM | 0.9043 | 0.9043 | 0.9656 |
| Logistic Regression | 0.9016 | 0.9016 | 0.9644 |
| Multinomial Naive Bayes | 0.8748 | 0.8748 | 0.9366 |
| Random Forest | 0.8588 | 0.8588 | 0.9351 |
| Baseline | 0.5019 | 0.3342 | 0.5000 |

**Tuning:** Linear SVM was tuned with C = 0.5 and bigram features (1, 2). CV F1 macro rose from 0.9043 to 0.9066.

**Final test results (9,917 unseen reviews):**

| Metric | Score |
|---|---|
| Accuracy | 0.9090 |
| Precision (positive) | 0.9032 |
| Recall (positive) | 0.9170 |
| F1 (positive) | 0.9101 |
| F1 (negative) | 0.9080 |
| F1 macro | 0.9090 |
| ROC-AUC | 0.9701 |

## Notes

- The cross-validation and test scores are close (0.9043 and 0.9090), so the model is not overfitting.
- Multinomial Naive Bayes used Bag of Words features, while the other models used TF-IDF. The comparison is not like-for-like.
- The misclassified file shows that errors often involve contrast words such as "but" and "however".

## Reproducing the Results

```bash
python src/main.py
```
