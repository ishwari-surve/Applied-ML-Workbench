# Dataset

This folder has the dataset used in the Iris Classification case study.

## File
- `iris.csv`

## Details
- Total samples: 150
- Features: 4 (all measurements in cm)
- Target: species of the Iris flower
- Classes: 3 (Setosa, Versicolor, Virginica) with 50 samples each
- Missing values: none

## Columns

| Column            | Description                    | Type    |
|-------------------|--------------------------------|---------|
| sepal length (cm) | Length of the sepal            | Number  |
| sepal width (cm)  | Width of the sepal             | Number  |
| petal length (cm) | Length of the petal            | Number  |
| petal width (cm)  | Width of the petal             | Number  |
| species           | Species of the Iris flower     | Text    |

## Encoding used in code

| Species    | Encoded value |
|------------|---------------|
| Setosa     | 0             |
| Versicolor | 1             |
| Virginica  | 2             |

## Note
- The column names in the CSV file have extra spaces. The code removes them while cleaning the data.
- `iris_classification.py` reads this file from `data/iris.csv`. Do not rename the file.
