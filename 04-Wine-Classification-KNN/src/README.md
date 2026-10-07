# Source Code

This folder has the source code of the Wine Classification case study.

## File
- `main.py`

## What the code does
- Loads the Wine dataset from `data/wine.csv` (178 samples)
- Removes empty rows
- Separates the 13 features (X) and the target `Class` (Y)
- Splits the data into training (80%) and testing (20%) sets
- Scales the features with StandardScaler
- Tests K from 1 to 20 with 5-fold cross validation and finds the best K
- Trains the final KNN model with the best K
- Evaluates the model with accuracy, confusion matrix and classification report
- Saves charts inside the `images` folder

## Functions

| Function                 | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| display_title            | Prints the project title                            |
| display_step             | Prints the title of each step                       |
| load_data                | Reads the CSV file                                  |
| clean_data               | Removes empty rows                                  |
| split_features_target    | Separates features (X) and target (Y)               |
| split_dataset            | Splits data into train and test sets                |
| scale_features           | Scales the features (learns from training data only)|
| tune_k_value             | Finds the accuracy of K = 1 to 20 (cross validation)|
| find_best_k              | Finds the K with the highest accuracy               |
| plot_k_vs_accuracy       | Draws and saves the K vs accuracy graph             |
| train_final_model        | Trains the final KNN model with the best K          |
| evaluate_model           | Calculates accuracy on the test data                |
| plot_confusion_matrix    | Draws and saves the confusion matrix                |
| show_classification_report | Shows precision, recall and f1-score              |
| display_footer           | Prints the completion message                       |
| main                     | Runs the complete workflow in order                 |

## Features and Target
- Features: 13 chemical measurements of the wine
- Target: Class (the grape cultivar, 1, 2 or 3)

## How to Run
Run from the case study folder (the one with requirements.txt):

    pip install -r requirements.txt
    python src/main.py

You can also run it from inside the src folder:

    python main.py

## Output Files
The code creates these images inside the `images` folder:
- `k_vs_accuracy.png`
- `confusion_matrix.png`

## Expected Results
- Best value of K: 13
- Cross validation accuracy: 0.9650
- Accuracy on test data: 100.00 %
