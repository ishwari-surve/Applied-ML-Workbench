# src

This folder contains the source code for the IMDb Sentiment Analysis project.

## Files

| File | Description |
|---|---|
| `main.py` | Runs the complete pipeline, from loading the dataset to saving the model, results, and charts |

## Pipeline Functions

All functions are defined in `main.py` and called from `main()` in this order.

**Setup and loading**
- `display_header()`: prints the project title
- `setup_project()`: creates the output folders and downloads the NLTK resources
- `load_dataset()`: checks the file exists and the required columns are present

**Understanding the data**
- `understand_dataset()`: shows the first and last records, data types, class counts, and word statistics
- `check_structure()`: checks missing values, duplicates, and label validity

**Cleaning and exploratory analysis**
- `clean_dataset()`: removes missing rows, duplicates, and conflicting labels
- `analyse_sentiment_distribution()`: counts each class and saves a chart
- `analyse_review_lengths()`: compares word and character counts by sentiment
- `explore_common_words()`: finds the most frequent words in each class

**Text preprocessing**
- `clean_review_text()`: removes HTML tags, URLs, and punctuation, and converts "n't" to "not"
- `preprocess_reviews()`: applies cleaning, removes stopwords except negations, and lemmatises
- `encode_labels()`: maps positive to 1 and negative to 0

**Features and splitting**
- `create_features()`: defines the Bag of Words and TF-IDF vectorizers
- `split_data()`: makes a stratified 80/20 train-test split

**Model development**
- `build_baseline_model()`: the most-frequent-class reference model
- `create_model_pipelines()`: one pipeline each for Naive Bayes, Logistic Regression, Linear SVM, and Random Forest
- `compare_models()`: 5-fold stratified cross-validation on training data
- `select_best_model()`: picks the best candidate by CV F1 macro, excluding the baseline
- `tune_best_model()`: GridSearchCV on training data, keeps the tuned settings only if they improve the score
- `train_final_model()`: fits the chosen pipeline on the full training set
- `save_model()`: saves the pipeline and reloads it to check it works

**Evaluation and analysis**
- `get_positive_scores()`: returns positive-class scores (uses a sigmoid for SVM)
- `evaluate_model()`: test-set metrics, classification report, and confusion matrix
- `analyse_errors()`: saves misclassified reviews and checks negation and contrast words
- `plot_model_results()`: saves the comparison, confusion matrix, ROC curve, and top-features charts

**Prediction and outputs**
- `predict_sentiment()`: predicts sentiment for new reviews with a confidence score
- `save_outputs()`: saves the metrics CSV and the insights JSON
- `display_summary()`: prints the final summary
- `display_footer()`: prints the completion message

## Running the Code

From the project root:

```bash
python src/main.py
```

The script works from any folder, because paths are built from the location of `main.py`.

## Key Design Choices

- **No data leakage:** vectorizers are fitted inside each pipeline on training data only, and the test set is used once, in the final evaluation.
- **Model selection on training data:** the best model and its tuned settings are chosen with cross-validation, not with the test set.
- **Negations kept:** words such as "not", "no", and "never" are not removed during stopword removal, because they change a review's meaning.
- **Saved pipeline:** the vectorizer and classifier are saved together, so new text can be predicted from the saved file.

## Author 
**Ishwari Surve**
