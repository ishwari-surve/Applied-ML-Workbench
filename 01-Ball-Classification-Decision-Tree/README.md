# 🏏🎾 Ball Classification using Decision Tree

A machine learning classification project that predicts whether a ball is a **Tennis Ball** or a **Cricket Ball** using its weight and surface type.

This case study shows the complete ML workflow with a Decision Tree Classifier in scikit-learn: load data, split, train, evaluate and predict.

## 🎯 Problem Statement
Given the weight and surface (Rough / Smooth) of a ball, predict its category.

## 🛠️ Tech Stack
- Python 3
- scikit-learn

## 📊 Dataset
- Size: 15 samples (small custom dataset created for this case study)
- Classes: Tennis (9 samples), Cricket (6 samples)
- The data is hard-coded in `main.py` in encoded form.
- A readable copy is available in `data/dataset.csv`. The code does not read this file.

| Column  | Description        | Encoding in code        |
|---------|--------------------|-------------------------|
| Weight  | Weight of the ball | Number (used as is)     |
| Surface | Rough or Smooth    | Rough = 1, Smooth = 0   |
| Label   | Tennis or Cricket  | Tennis = 1, Cricket = 2 |

## 🌳 Algorithm
- Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- The tree learns simple rules from the training data, such as "heavy and smooth means cricket".

## 🔍 Workflow
1. Load the dataset
2. Split into train and test sets
3. Build and train the model
4. Evaluate with accuracy
5. Predict new balls

## 📂 Project Structure
```
01-Ball-Classification-Decision-Tree/
├── data/
│   ├── dataset.csv
│   └── README.md
├── images/
│   └── output.png
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
```

## 📈 Sample Prediction
Input: `[35, 1]` (weight = 35, surface = Rough)

Output:
```
Object looks like tennis ball
```

## 🖥️ Sample Output
```
------------------------------------------------------------
---------- Ball Classification Case Study ------------------
------------------------------------------------------------
Model accuracy :: 1.0
Object looks like tennis ball
Object looks like cricket ball
```

## ✅ Results
- Test accuracy: 100% on 2 test samples
- Weight 37 with rough surface gives Tennis. Weight 94 with smooth surface gives Cricket.
- Note: the dataset is very small, so accuracy is not a reliable measure. This project is for learning the ML workflow.

## 📚 Learning Outcomes
- Supervised learning and classification
- Feature and label encoding
- Train/test split
- Decision Tree algorithm
- Model evaluation and prediction with scikit-learn

## 🚀 Future Improvements
- Use a larger dataset
- Add confusion matrix
- Visualize the decision tree
- Add more ball types

## 👩‍💻 Author
**Ishwari Surve**


GitHub: https://github.com/ishwari-surve
