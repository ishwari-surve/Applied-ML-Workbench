# Images

This folder holds the charts produced by `src/main.py`. The program saves them automatically when it runs, so you can regenerate them at any time.

## Files

| File | What it shows | Key point |
|---|---|---|
| `01_sentiment_distribution.png` | Count of positive and negative reviews | The dataset is balanced after cleaning: 24,884 positive and 24,698 negative |
| `02_review_length_distribution.png` | Histogram of word counts for each sentiment | Both classes have similar length profiles. Word counts are capped at the 99th percentile so long reviews do not distort the chart |
| `03_top_words_positive.png` | Most frequent words in positive reviews | "not" appears most often, which shows negations are kept during preprocessing. Other common words are "film", "movie", and "great" |
| `03_top_words_negative.png` | Most frequent words in negative reviews | "not" is again the most frequent word, followed by "movie", "film", and "no" |
| `04_model_comparison.png` | CV accuracy, macro F1, and ROC-AUC for each model | Linear SVM and Logistic Regression score highest. The baseline scores about 0.50, which is the reference level |
| `confusion_matrix.png` | Test set confusion matrix | 4,451 negatives and 4,564 positives correct. 489 negatives were predicted as positive, and 413 positives as negative |
| `roc_curve.png` | ROC curve on the test set | ROC-AUC of 0.9701 on unseen data |
| `top_features.png` | Words with the strongest weight for each sentiment | Strong positive words include "great", "excellent", and "perfect". Strong negative words include "worst", "awful", and "bad" |

## Notes

- The top-words charts count words after cleaning and lemmatisation, so word forms are merged (for example "movies" and "movie").
- `top_features.png` comes from the Linear SVM, which gives each word a weight. Some phrases such as "not funny" and "not recommend" appear because the model uses 1 to 2 word phrases.
- A few weaker items appear in the top lists, such as "today" and "one best". Mention this honestly in interviews rather than hiding it.
- All charts are saved at 300 dpi.

## Regenerating the Charts

From the project root:

```bash
python src/main.py
```
