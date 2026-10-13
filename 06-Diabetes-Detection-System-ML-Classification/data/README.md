# Dataset

This folder has the dataset used in the Diabetes Detection System using Machine Learning Classification case study.

## File

- `diabetes.csv`

## Details

- Total samples: 768 individuals
- Columns: 9 (8 input features and the target `Outcome`)
- Target: `Outcome` (0 = no diabetes, 1 = diabetes)
- Missing values in the original dataset: 0
- Zero values treated as missing: present in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI`

## Class distribution

| **Outcome** | **Individuals** | **Share** |
| ----------- | ---------------: | --------: |
| 0 (No diabetes) | 500 | 65.1 % |
| 1 (Diabetes) | 268 | 34.9 % |

The classes are not balanced. A model that always predicts "no diabetes" achieves approximately 65.1 % accuracy on the complete dataset, so this is the baseline to beat.

## Columns

| **Column** | **Description** | **Values / Details** |
| ---------- | ---------------- | -------------------- |
| `Pregnancies` | Number of pregnancies | Non-negative integer |
| `Glucose` | Plasma glucose concentration | 5 zero values treated as missing |
| `BloodPressure` | Diastolic blood pressure | 35 zero values treated as missing |
| `SkinThickness` | Triceps skinfold thickness | 227 zero values treated as missing |
| `Insulin` | Insulin level | 374 zero values treated as missing |
| `BMI` | Body Mass Index | 11 zero values treated as missing |
| `DiabetesPedigreeFunction` | Diabetes pedigree function value | Numeric |
| `Age` | Age of the individual | Numeric, in years |
| `Outcome` | Target: diabetes or no diabetes | 0, 1 |

## Note

- The original CSV file contains no empty cells, but some medical measurement columns contain zero values that the project treats as missing measurements.
- The code replaces zero values with missing values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI`.
- Missing feature values are handled using median imputation in the machine learning pipelines.
- The dataset contains 500 records with `Outcome = 0` and 268 records with `Outcome = 1`.
- `Outcome` is the target column. The other eight columns are used as input features.
- `main.py` reads this file from `data/diabetes.csv`. Do not rename the file.
- This dataset is used for educational machine learning experimentation and is not intended for medical diagnosis.
