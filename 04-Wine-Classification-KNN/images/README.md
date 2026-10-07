# Images

This folder has the output images of the Wine Classification case study.

## Files

| File                 | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| k_vs_accuracy.png    | K value vs cross validation accuracy (the red line shows best K)     |
| confusion_matrix.png | Confusion matrix of the final KNN model on the test data             |
| output_1.png         | Terminal output: Steps 1 to 4, loading, cleaning, missing values and statistics |
| output_2.png         | Terminal output: Steps 5 to 9, data split, scaling, K tuning and best K |
| output_3.png         | Terminal output: Steps 10 to 15, final model, accuracy, confusion matrix, report and summary |

## How the images are created
The first two images are created automatically when you run the code.
Run from the case study folder (the one with requirements.txt):

    python src/main.py

The `output_` images are screenshots of the terminal output, added manually.

## Results shown
- Best value of K: 13
- Cross validation accuracy: 0.9650
- Accuracy on test data: 100.00 %
