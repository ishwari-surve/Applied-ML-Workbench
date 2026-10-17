# 📂 Credit Card Fraud Detection — Dataset

## 📌 Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the Machine Learning Group at Université Libre de Bruxelles (ULB) and hosted on Kaggle.

The dataset contains anonymized credit card transactions used to develop and evaluate Machine Learning classification models for identifying fraudulent transactions.

### 📥 Download Dataset

The dataset file is too large to upload directly to this GitHub repository. Download it from the Kaggle dataset page below.

**Dataset Link:**  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 📊 Dataset Information

| **Property** | **Description** |
|---|---|
| Dataset Name | Credit Card Fraud Detection |
| Source | Kaggle – MLG-ULB |
| Total Transactions | 284,807 |
| Total Input Features | 30 |
| Total Columns | 31 |
| Fraudulent Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Target Column | `Class` |
| File Name | `creditcard.csv` |
| File Size | Approximately 150 MB |

### 🧾 Dataset Features

- **`Time`** — Seconds elapsed between each transaction and the first transaction in the dataset.
- **`V1` to `V28`** — Anonymized numerical features obtained using Principal Component Analysis (PCA).
- **`Amount`** — Transaction amount.
- **`Class`** — Target variable indicating whether a transaction is legitimate or fraudulent.

### 🎯 Target Variable

| **Class Value** | **Description** |
|---|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

### ⚖️ Class Imbalance

The dataset is highly imbalanced because fraudulent transactions account for approximately **0.172%** of all transactions.

Therefore, this project evaluates models using metrics such as Precision, Recall, F1-Score, and Average Precision to better assess fraud detection performance.

### 📁 Folder Structure

```text
data/
├── README.md
└── creditcard.csv  # Download separately; not uploaded to GitHub
```

### ⚙️ Setup Instructions

1. Download the dataset using the Kaggle link provided above.
2. Extract the downloaded ZIP file.
3. Place `creditcard.csv` inside the `data/` folder.
4. Run the project from the main project directory.

**Note:** The CSV file is excluded from the GitHub repository because of its large size. Downloading the dataset is required to execute the project locally.

---

**Project:** Credit Card Fraud Detection using Machine Learning Classification  
**Purpose:** Exploratory Data Analysis, model training, and fraud classification.

