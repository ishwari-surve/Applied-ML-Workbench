# Dataset

This folder has the dataset used in the Advertising Sales Prediction case study.

## File
- `advertising.csv`

## Details
- Total samples: 200
- Features: 3 (TV, radio, newspaper)
- Target: sales
- Missing values: none
- Duplicate rows: none

## Columns

| Column    | Description                           | Type   |
|-----------|---------------------------------------|--------|
| TV        | Advertising budget spent on TV        | Number |
| radio     | Advertising budget spent on radio     | Number |
| newspaper | Advertising budget spent on newspaper | Number |
| sales     | Sales of the product (target)         | Number |

## Correlation with sales

| Feature   | Correlation |
|-----------|-------------|
| TV        | 0.78        |
| radio     | 0.58        |
| newspaper | 0.23        |

## Note
- The original file has an extra first column with no name (an index from 1 to 200). The code removes it while cleaning the data.
- The file does not give the units of the budgets and sales.
- `main.py` reads this file from `data/advertising.csv`. Do not rename the file.
