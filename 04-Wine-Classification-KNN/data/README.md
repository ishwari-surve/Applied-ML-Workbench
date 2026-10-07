# Dataset

This folder has the dataset used in the Wine Classification case study.

## File
- `wine.csv`

## Details
- Total samples: 178
- Features: 13 (chemical measurements of the wine)
- Target: Class (the grape cultivar of the wine)
- Classes: 3
- Missing values: none
- Duplicate rows: none

## Class distribution

| Class   | Samples |
|---------|---------|
| Class 1 | 59      |
| Class 2 | 71      |
| Class 3 | 48      |

## Columns

| Column                       | Type   |
|------------------------------|--------|
| Class (target)               | Number (1, 2 or 3) |
| Alcohol                      | Number |
| Malic acid                   | Number |
| Ash                          | Number |
| Alcalinity of ash            | Number |
| Magnesium                    | Number |
| Total phenols                | Number |
| Flavanoids                   | Number |
| Nonflavanoid phenols         | Number |
| Proanthocyanins              | Number |
| Color intensity              | Number |
| Hue                          | Number |
| OD280/OD315 of diluted wines | Number |
| Proline                      | Number |

## Note
- The `Class` column is the first column in the file.
- The features have very different ranges, so the code scales them before using KNN.
- `main.py` reads this file from `data/wine.csv`. Do not rename the file.
