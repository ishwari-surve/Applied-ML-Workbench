# src

This folder contains the source code for the IMDb Sentiment Analysis project.

## Files

| File | Description |
|---|---|
| `main.py` | Runs the complete pipeline: loading, cleaning, text preprocessing, model comparison, tuning, evaluation, prediction, and saving outputs |

## How the Code Is Organised

`main.py` is grouped into sections, one function for each step:

1. **Setup and loading:** `setup_project()`, `load_dataset()`
2. **Understanding the data:** `understand_dataset()`, `check_structure()`
3. **Cleaning and EDA:** `clean_dataset()`, `analyse_sentiment_distribution()`, `analyse_review_lengths()`, `explore_common_words()`
4. **Text preprocessing:** `clean_review_text()`, `preprocess_reviews()`, `encode_labels()`
5. **Features and splitting:** `create_features()`, `split_data()`
6. **Model development:** `build_baseline_model()`, `create_model_pipelines()`, `compare_models()`, `select_best_model()`, `tune_best_model()`, `train_final_model()`
7. **Evaluation and analysis:** `save_model()`, `evaluate_model()`, `analyse_errors()`, `plot_model_results()`
8. **Prediction and outputs:** `predict_sentiment()`, `save_outputs()`, `display_summary()`, `display_footer()`

All functions are called from `main()` in the order shown above.

## Running the Code

Run from the `src` folder or from the project root:

```bash
python src/main.py
```

The script finds the project folders automatically, so it works from either location as long as `data/raw/IMDB_Dataset.csv` is in place.

## Key Design Choices

- **No data leakage:** the vectorizers are fitted inside each pipeline on the training data only, and the test set is used only once, in the final evaluation.
- **Cross-validation for model selection:** the best model is chosen from training data, not from the test set.
- **Negations kept:** words such as "not", "no", and "never" are kept during stopword removal, because they change the meaning of a review.
- **Saved pipeline:** the text vectorizer and classifier are saved together, so new text can be predicted directly from the saved file.
