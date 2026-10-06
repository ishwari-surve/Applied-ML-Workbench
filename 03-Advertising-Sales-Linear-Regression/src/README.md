# Source Code

This folder has the source code of the Advertising Sales Prediction case study.

## File
- `main.py`

## What the code does
- Loads the advertising dataset from `data/advertising.csv` (200 samples)
- Removes the unwanted index column and checks missing values
- Shows statistics and the correlation between columns
- Splits the data into independent (TV, radio, newspaper) and dependent (sales) variables
- Splits the data into training (80%) and testing (20%) sets
- Trains a Multiple Linear Regression model
- Predicts sales for the test data
- Evaluates the model with MSE, RMSE and R2
- Shows the model coefficients and compares actual and predicted sales
- Predicts sales for new advertising budgets
- Saves charts inside the `images` folder

## Functions

| Function                 | Purpose                                         |
|--------------------------|-------------------------------------------------|
| display_title            | Prints the project title                        |
| display_step             | Prints the title of each step                   |
| load_data                | Reads the CSV file                              |
| clean_data               | Removes the unwanted index column               |
| check_missing_values     | Shows missing values for every column           |
| show_statistics          | Shows mean, std, min, max, etc.                 |
| show_correlation         | Shows the correlation matrix and saves heatmap  |
| split_features_target    | Separates features (X) and target (Y)           |
| split_dataset            | Splits data into train and test sets            |
| train_model              | Creates and trains the model                    |
| predict_sales            | Predicts sales for the test data                |
| evaluate_model           | Calculates MSE, RMSE and R2                     |
| show_coefficients        | Shows the coefficients and the intercept        |
| compare_actual_predicted | Shows actual and predicted sales side by side   |
| plot_actual_vs_predicted | Draws and saves the actual vs predicted plot    |
| predict_new_sales        | Predicts sales for a new advertising budget     |
| display_footer           | Prints the completion message                   |
| main                     | Runs the complete workflow in order             |

## Features and Target
- Features: TV, radio, newspaper
- Target: sales

## How to Run
Run from the case study folder (the one with requirements.txt):

    pip install -r requirements.txt
    python src/main.py

You can also run it from inside the src folder:

    python main.py

## Output Files
The code creates these images inside the `images` folder:
- `correlation_heatmap.png`
- `actual_vs_predicted.png`

## Expected Results
- Mean Squared Error: 3.1741
- Root Mean Squared Error: 1.7816
- R Square: 0.8994
- Predicted sales for TV = 200, radio = 40, newspaper = 30: 19.58
- Predicted sales for TV = 50, radio = 10, newspaper = 20: 7.16
