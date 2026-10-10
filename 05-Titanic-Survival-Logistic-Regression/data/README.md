# Dataset

This folder has the dataset used in the Titanic Survival Prediction case study.

## File
- `titanic.csv`

## Details
- Total samples: 1309 passengers
- Columns: 10 (9 input columns and the target `Survived`)
- Target: Survived (0 = did not survive, 1 = survived)
- Duplicate rows: none
- Missing values: 2, only in the `Embarked` column

## Class distribution

| Survived        | Passengers | Share  |
|-----------------|------------|--------|
| 0 (No)          | 967        | 73.9 % |
| 1 (Yes)         | 342        | 26.1 % |

The classes are not balanced. A model that always predicts "did not survive" is already 73.9 % accurate, so this is the baseline to beat.

## Columns

| Column      | Description                                  | Values                         |
|-------------|----------------------------------------------|--------------------------------|
| Passengerid | Passenger number                             | 1 to 1309                      |
| Age         | Age of the passenger in years                | 0.17 to 80                     |
| Fare        | Ticket fare                                  | 0 to 512.33                    |
| Sex         | Sex of the passenger (already encoded)       | 0 and 1                        |
| sibsp       | Number of siblings or spouses on board       | 0 to 8                         |
| Parch       | Number of parents or children on board       | 0 to 9                         |
| zero        | Column with the value 0 in every row         | 0                              |
| Pclass      | Ticket class                                 | 1, 2, 3                        |
| Embarked    | Port of embarkation (already encoded)        | 0, 1, 2 (2 missing values)     |
| Survived    | Target: survived or not                      | 0, 1                           |

## Note
- The file does not explain the numbers in `Sex` and `Embarked`. They match the standard Titanic dataset: Sex 0 = male and 1 = female, and Embarked 0 = Cherbourg, 1 = Queenstown and 2 = Southampton. The counts in this file (843 and 466 for Sex, 270, 123 and 914 for Embarked) agree with that.
- The `Passengerid` and `zero` columns carry no information for the model, so the code drops them.
- `main.py` reads this file from `data/titanic.csv`. Do not rename the file.
