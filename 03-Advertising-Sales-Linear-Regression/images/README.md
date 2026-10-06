# Images

This folder has the output images of the Advertising Sales Prediction case study.

## Files

| File                    | Description                                                             |
|-------------------------|-------------------------------------------------------------------------|
| correlation_heatmap.png | Correlation between TV, radio, newspaper and sales                      |
| actual_vs_predicted.png | Actual sales vs predicted sales (the red line shows perfect prediction) |
| output_1.png            | Terminal output: data split, model training, evaluation and coefficients |
| output_2.png            | Terminal output: actual vs predicted, plot saved and new predictions    |

## How the images are created
The first two images are created automatically when you run the code.
Run from the case study folder (the one with requirements.txt):

    python src/main.py

`output_1.png` and `output_2.png` are screenshots of the terminal output, added manually.

## Results shown
- Mean Squared Error: 3.1741
- Root Mean Squared Error: 1.7816
- R Square: 0.8994
- TV has the strongest correlation with sales (0.78).
