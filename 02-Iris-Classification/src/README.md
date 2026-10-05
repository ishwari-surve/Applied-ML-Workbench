# Source Code

This folder has the source code of the Iris Classification case study.

## File
- `iris_classification.py`

## What the code does
- Loads the Iris dataset from `data/iris.csv` (150 samples)
- Cleans the column names and checks missing values
- Shows statistics and class distribution
- Encodes the species names into numbers
- Splits the data into training (80%) and testing (20%) sets
- Trains two models: KNN and Decision Tree
- Compares accuracy of both models
- Shows the classification report and confusion matrices
- Saves charts inside the `images` folder

## Functions

| Function                  | Purpose                                      |
|---------------------------|----------------------------------------------|
| display_title             | Prints the project title                     |
| display_step              | Prints the title of each step                |
| load_data                 | Reads the CSV file                           |
| clean_data                | Removes extra columns and extra spaces       |
| check_missing_values      | Shows missing values for every column        |
| show_statistics           | Shows mean, std, min, max, etc.              |
| show_class_distribution   | Shows the number of samples per species      |
| encode_target             | Converts species names into numbers          |
| split_features_target     | Separates features (X) and target (Y)        |
| split_dataset             | Splits data into train and test sets         |
| train_knn                 | Trains the KNN model (5 neighbors)           |
| train_decision_tree       | Trains the Decision Tree model               |
| evaluate_model            | Calculates accuracy and predictions          |
| compare_models            | Compares both models and shows the best one  |
| show_classification_report| Shows precision, recall and f1-score         |
| plot_confusion_matrix     | Draws and saves the confusion matrix         |
| plot_accuracy_comparison  | Draws and saves the accuracy bar chart       |
| display_footer            | Prints the completion message                |
| main                      | Runs the complete workflow in order          |

## Encoding
- Setosa = 0
- Versicolor = 1
- Virginica = 2

## How to Run
Run from the case study folder (the one with requirements.txt):

    pip install -r requirements.txt
    python src/main.py

You can also run it from inside the src folder:

    python main.py

## Output Files
The code creates these images inside the `images` folder:
- `confusion_matrix_knn.png`
- `confusion_matrix_decision_tree.png`
- `accuracy_comparison.png`

## Expected Results
- KNN accuracy: 1.0000
- Decision Tree accuracy: 0.9333
- Best model: KNN
