# Dataset

This folder has a readable copy of the dataset used in the Ball Classification case study.

## File
- `dataset.csv`

## Details
- Total samples: 15
- Classes: Tennis (9 samples), Cricket (6 samples)

## Columns

| Column  | Description         | Encoding used in code  |
|---------|---------------------|------------------------|
| Weight  | Weight of the ball  | Number (used as is)    |
| Surface | Rough or Smooth     | Rough = 1, Smooth = 0  |
| Label   | Tennis or Cricket   | Tennis = 1, Cricket = 2|

## Note
- The same data is hard-coded inside `main.py` in encoded form.
- `main.py` does not read this CSV file. It is kept only for easy viewing.
