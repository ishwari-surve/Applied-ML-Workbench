# 🌸 Iris Classification using KNN and Decision Tree

A machine learning classification project that predicts the species of an Iris flower from its sepal and petal measurements. Two models, **K-Nearest Neighbors (KNN)** and **Decision Tree**, are trained and compared.

## 🎯 Problem Statement
Given the sepal length, sepal width, petal length and petal width of an Iris flower, predict its species: Setosa, Versicolor or Virginica.

## 🛠️ Tech Stack
- Python 3
- pandas
- scikit-learn
- matplotlib
- seaborn

## 📊 Dataset
- File: `data/iris.csv`
- Size: 150 samples, 4 features, 3 classes (50 samples each)
- Missing values: none

| Column            | Description                | Type   |
|-------------------|----------------------------|--------|
| sepal length (cm) | Length of the sepal        | Number |
| sepal width (cm)  | Width of the sepal         | Number |
| petal length (cm) | Length of the petal        | Number |
| petal width (cm)  | Width of the petal         | Number |
| species           | Species of the Iris flower | Text   |

Encoding used in code: Setosa = 0, Versicolor = 1, Virginica = 2

## 🌳 Algorithms
- **KNN:** `KNeighborsClassifier` with 5 neighbors. It predicts the class of a flower from its 5 closest flowers in the training data.
- **Decision Tree:** `DecisionTreeClassifier`. It learns simple if-else rules from the measurements.

## 🔍 Workflow
1. Load the dataset
2. Remove unwanted columns and clean the column names
3. Check missing values
4. Show the statistical summary
5. Show the class distribution
6. Encode the target variable
7. Split into independent and dependent variables
8. Split into training (80%) and testing (20%) sets
9. Train the KNN model
10. Train the Decision Tree model
11. Compare both models
12. Show the classification report
13. Save the confusion matrix of both models
14. Save the accuracy comparison chart

## 📂 Project Structure
```
02-Iris-Classification/
├── data/
│   ├── iris.csv
│   └── README.md
├── images/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_knn.png
│   ├── output_1.png
│   ├── output_2.png
|   ├── output_3.png
|   ├── output_4.png
|   ├── output_5.png
|   ├── output_6.png
│   └── README.md
├── src/
│   ├── iris_classification.py
│   └── README.md
├── requirements.txt
└── README.md
```

## ▶️ How to Run
Run from this folder (the one with `requirements.txt`):

```bash
pip install -r requirements.txt
python src/iris_classification.py
```

The charts are saved automatically in the `images` folder.

## 🖥️ Sample Output
```
Step 9: Build and train KNN model
KNN Accuracy : 100.00 %

Step 10: Build and train Decision Tree model
Decision Tree Accuracy : 93.33 %

Step 11: Model comparison
KNN Accuracy          : 1.0000
Decision Tree Accuracy: 0.9333

Best Model : K-Nearest Neighbors (KNN)
```

## ✅ Results

| Model         | Accuracy |
|---------------|----------|
| KNN           | 1.0000   |
| Decision Tree | 0.9333   |

**Best model: KNN**

- KNN classified all 30 test samples correctly.
- Decision Tree made 2 mistakes: 1 Versicolor was predicted as Virginica, and 1 Virginica was predicted as Versicolor.
- Setosa was classified correctly by both models.

**Note:** The test set has only 30 samples and the result comes from one train/test split, so the accuracy can change with a different split. This project is for learning the ML workflow.

## 📚 Learning Outcomes
- Data cleaning and exploration with pandas
- Label encoding of a multi-class target
- Stratified train/test split
- Training and comparing two classification models
- Model evaluation with accuracy, classification report and confusion matrix
- Saving charts with matplotlib and seaborn

## 🚀 Future Improvements
- Use cross-validation for a more reliable accuracy
- Scale the features before KNN
- Tune the number of neighbors and tree depth
- Add more models (Logistic Regression, Random Forest, SVM)

## 👩‍💻 Author
**Ishwari Surve**


GitHub: https://github.com/ishwari-surve
