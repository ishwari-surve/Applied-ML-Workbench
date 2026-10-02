# Source Code

This folder has the source code of the Ball Classification case study.

## File
- `main.py`

## What the code does
- Loads the encoded dataset (15 samples)
- Splits the data into training and testing sets
- Trains a Decision Tree Classifier
- Evaluates the model with accuracy
- Predicts the type of new balls

## Functions

| Function       | Purpose                                  |
|----------------|------------------------------------------|
| display_data   | Prints the case study heading            |
| load_data      | Returns features and labels              |
| split_dataset  | Splits data into train and test sets     |
| build_model    | Creates the Decision Tree model          |
| train_model    | Trains the model                         |
| evaluate_model | Calculates accuracy on test data         |
| predict_ball   | Predicts the type of a new ball          |
| display_footer | Prints the completion message            |
| main           | Runs the complete workflow in order      |

## Encoding
- Surface: Rough = 1, Smooth = 0
- Label: Tennis = 1, Cricket = 2

## How to Run
Run from the case study folder (the one with requirements.txt):

    pip install -r requirements.txt
    python src/main.py

## Expected Output
    ------------------------------------------------------------
    ---------- Ball Classification Case Study ------------------
    ------------------------------------------------------------
    Model Accuracy :: 100.00 %
    Object looks like Tennis Ball
    Object looks like Cricket Ball
    ------------------------------------------------------------
    Ball Classification Case Study Completed Successfully
    ------------------------------------------------------------
