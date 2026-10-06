# Images

This folder has the output images of the Advertising Sales Prediction case study.

## Files

| File                    | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| correlation_heatmap.png | Correlation between TV, radio, newspaper and sales                       |
| actual_vs_predicted.png | Actual sales vs predicted sales (the red line shows perfect prediction)  |
| output_1.png            | Terminal output: Step 1, loading the dataset (first and last records)    |
| output_2.png            | Terminal output: Steps 2 to 3, cleaning the data and checking missing values |
| output_3.png            | Terminal output: Steps 4 to 6, statistics, correlation and variable split |
| output_4.png            | Terminal output: Steps 7 to 11, data split, training, evaluation and coefficients |
| output_5.png            | Terminal output: Steps 12 to 14, comparison, plot saved and new predictions |

## How the images are created
The first two images are created automatically when you run the code.
Run from the case study folder (the one with requirements.txt):

    python src/main.py

The `output_` images are screenshots of the terminal output, added manually.

## Results shown
- Mean Squared Error: 3.1741
- Root Mean Squared Error: 1.7816
- R Square: 0.8994
- TV has the strongest correlation with sales (0.78).
