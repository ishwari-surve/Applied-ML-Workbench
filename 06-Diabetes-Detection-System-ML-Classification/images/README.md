# Images

This folder has the output images and terminal screenshots of the Diabetes Detection System case study.

## Files

| **File** | **Description** |
| -------- | --------------- |
| `model_comparison.png` | Compares the accuracy, recall and F1-score of five classifiers using 5-fold cross-validation |
| `confusion_matrix.png` | Confusion matrix of the final Logistic Regression model on the test data |
| `feature_importance.png` | Feature importance chart generated using Random Forest |
| `output_1.png` | Terminal screenshot: Steps 1 to 2, dataset loading, dataset shape, data quality checks and class distribution |
| `output_2.png` | Terminal screenshot: Steps 3 to 6, zero-value preprocessing, feature-target separation, train-test split and cross-validation comparison |
| `output_3.png` | Terminal screenshot: Steps 7 to 10, best classifier selection, model training, saving and loading, and test evaluation |
| `output_4.png` | Terminal screenshot: Steps 11 to 13, graph generation, feature importance, example patient predictions and project summary |

## How the images are created

The following three images are created automatically when you run the code.

Run from the case study folder (the one with `requirements.txt`):

```bash
python src/main.py
```

The `output_` images are screenshots of the terminal output, added manually.

## Results shown

- Best classifier: Logistic Regression
- Cross-validation F1-score: 65.45%
- Accuracy on test data: 70.78%
- Recall for diabetes: 50.00%
- F1-score for diabetes: 54.55%

**Note:** This is an educational machine learning project, not a medical diagnosis.
