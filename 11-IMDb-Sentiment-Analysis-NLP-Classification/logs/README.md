# Logs

This folder holds the console output from a complete run of `src/main.py`.

## Files

| File | Description |
|---|---|
| `run_log.txt` | Full terminal output of one successful run, from the project header to the completion footer |

## What the Log Shows

The log records each step in order:

- Setup of folders and NLTK resources
- Dataset loading: 50,000 rows, 2 columns
- Structural checks: 418 duplicate rows, no missing values, no conflicting labels
- Cleaning: 49,582 reviews remain (24,884 positive, 24,698 negative)
- Text preprocessing and feature creation
- Split: 39,665 training and 9,917 testing reviews
- Cross-validation of five models, selection of Linear SVM, and tuning
- Final test results and error analysis
- Chart and output file paths

Key final result: **accuracy 0.9090, ROC-AUC 0.9701** on the 9,917 unseen test reviews.

## Notes

- The log is from one run only. Earlier attempts that failed on the dataset path are not included.
- Some step labels appear out of order. For example, STEP 8 and STEP 9 print before STEP 7, and the prediction step prints a second STEP 8 header. The code and results are correct; only the labels are misleading.
- File paths in the log show the author's local Windows folder. This is expected for a local run.

## Reproducing the Log

From the `src` folder:

```bash
python main.py
```

Results will differ slightly if the dataset, package versions, or random seed change.
