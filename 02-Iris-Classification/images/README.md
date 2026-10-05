# Images

This folder has the output images of the Iris Classification case study.

## Files

| File                               | Description                                                   |
|------------------------------------|---------------------------------------------------------------|
| confusion_matrix_knn.png           | Confusion matrix of the KNN model                             |
| confusion_matrix_decision_tree.png | Confusion matrix of the Decision Tree model                   |
| accuracy_comparison.png            | Accuracy comparison of both models                            |
| output_1.png                       | Terminal output: model training and comparison                |
| output_2.png                       | Terminal output: classification report and confusion matrices |

## How the images are created
The first three images are created automatically when you run the code.
Run from the case study folder (the one with requirements.txt):

    python src/main.py

`output_1.png` and `output_2.png` are screenshots of the terminal output, added manually.

## Results shown
- KNN accuracy: 1.0000
- Decision Tree accuracy: 0.9333
- KNN is the best model.
- The Decision Tree confuses 2 samples (1 Versicolor and 1 Virginica).
