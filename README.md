# 🤖 Applied Machine Learning Workbench

A collection of hands-on machine learning case studies built in Python. Each case study follows a structured workflow covering data loading, preparation, model training, evaluation, and prediction, with dedicated source code, dependencies, documentation, and outputs.

## 📚 Case Studies

| **No.** | **Case Study** | **Type** | **Algorithm** | **Result** |
|---|---|---|---|---|
| 01 | [Ball Classification](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/01-Ball-Classification-Decision-Tree) | Classification | Decision Tree | 100% accuracy (2 test samples) |
| 02 | [Iris Classification](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/02-Iris-Classification) | Classification | KNN, Decision Tree | KNN 100%, Decision Tree 93.33% |
| 03 | [Advertising Sales Prediction](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/03-Advertising-Sales-Linear-Regression) | Regression | Multiple Linear Regression | R² 0.8994, RMSE 1.78 |
| 04 | [Wine Classification](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/04-Wine-Classification-KNN) | Classification | KNN with feature scaling and hyperparameter tuning | 96.50% cross-validation, 100% test accuracy (best K = 13) |
| 05 | [Titanic Survival Prediction](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/05-Titanic-Survival-Logistic-Regression) | Classification | Logistic Regression | 80.45% accuracy (baseline 61.45%) |
| 06 | [Diabetes Prediction](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/06-Diabetes-Prediction) | Classification | Refer to the case study implementation | Refer to the case study README |
| 07 | [Breast Cancer Detection](https://github.com/ishwari-surve/Applied-ML-Workbench/tree/main/07-Breast-Cancer) | Classification | Decision Tree, Logistic Regression, KNN, SVM, Bagging, Random Forest, AdaBoost, Gradient Boosting, Hard Voting, Soft Voting | 98.25% test accuracy in the recorded run |

*Results reflect recorded executions and may vary depending on the dataset and configuration. The breast cancer case study is for educational purposes only and is not intended for clinical diagnosis.*

More case studies will be added.

## 🛠️ Tech Stack

- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

## 📂 Repository Structure

```text
Applied-ML-Workbench/
├── 01-Ball-Classification-Decision-Tree/
│   ├── data/
│   ├── images/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 02-Iris-Classification/
│   ├── data/
│   ├── images/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 03-Advertising-Sales-Linear-Regression/
│   ├── data/
│   ├── images/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 04-Wine-Classification-KNN/
│   ├── data/
│   ├── images/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 05-Titanic-Survival-Logistic-Regression/
│   ├── data/
│   ├── images/
│   ├── models/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 06-Diabetes-Prediction/
│   ├── data/
│   ├── images/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── 07-Breast-Cancer/
│   ├── data/
│   ├── images/
│   ├── models/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── .gitignore
├── LICENSE
└── README.md
```

## ▶️ How to Run a Case Study

### 1. Clone the repository

```bash
git clone https://github.com/ishwari-surve/Applied-ML-Workbench.git
```

### 2. Navigate to a case study

For example, to run the breast cancer project:

```bash
cd Applied-ML-Workbench/07-Breast-Cancer
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the program

```bash
python src/main.py
```

Each case study contains its own README with project details, workflow, dependencies, and outputs.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/ishwari-surve/Applied-ML-Workbench/blob/main/LICENSE) file for details.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

GitHub: [ishwari-surve](https://github.com/ishwari-surve)
