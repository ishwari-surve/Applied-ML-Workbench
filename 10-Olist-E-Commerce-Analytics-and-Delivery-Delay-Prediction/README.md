# 🛒 Olist E-Commerce Analytics & Delivery Delay Prediction

<p align="center">

**Exploratory Data Analysis | Customer Segmentation | Machine Learning | Business Intelligence**

</p>

## 📌 Project Overview

The **Olist E-Commerce Analytics and Delivery Delay Prediction** project focuses on analyzing Brazilian e-commerce data to understand customer purchasing behavior, explore delivery performance, and predict whether delivered orders arrive later than their estimated delivery dates.

Using Python, data analytics, and machine learning, this project transforms raw transactional data into meaningful business insights that can support operational decision-making and improve delivery monitoring.

## 🎯 Project Objectives

- Analyze e-commerce orders, customers, sellers, products, and payments.
- Clean and integrate multiple related datasets.
- Perform Exploratory Data Analysis (EDA) to identify business patterns.
- Conduct customer segmentation using Recency, Frequency, and Monetary (RFM) analysis.
- Engineer meaningful features for delivery-delay prediction.
- Develop machine learning classification models.
- Compare models using cross-validation and select a suitable candidate.
- Evaluate the selected model on unseen test data.
- Generate delivery predictions and risk categories.
- Export analytical results for further reporting and Power BI visualization.

## 📊 Dataset Information

**Dataset:** Olist Brazilian E-Commerce Public Dataset

**Source:** Kaggle

🔗 **Dataset Link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset contains anonymized Brazilian e-commerce marketplace information, including orders, customers, sellers, products, payments, reviews, and geographic data.

The project uses nine CSV datasets:

1. `olist_customers_dataset.csv`
2. `olist_geolocation_dataset.csv`
3. `olist_order_items_dataset.csv`
4. `olist_order_payments_dataset.csv`
5. `olist_order_reviews_dataset.csv`
6. `olist_orders_dataset.csv`
7. `olist_products_dataset.csv`
8. `olist_sellers_dataset.csv`
9. `product_category_name_translation.csv`

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Data preprocessing, model building, and evaluation |
| Joblib | Saving and loading trained models |
| JSON | Storing structured business insights |
| Power BI | Further business intelligence reporting |

## 🔍 Project Workflow

### 1. Data Loading and Understanding
Load the raw CSV datasets and inspect their structures, columns, data types, and record counts.

### 2. Data Quality Assessment
Identify missing values and duplicate records and prepare the datasets for reliable analysis.

### 3. Data Cleaning and Integration
Clean the datasets and combine relevant customer, seller, order, product, and payment information.

### 4. Feature Engineering
Create useful features related to purchase timing, order value, freight costs, product attributes, and delivery timelines.

### 5. Exploratory Data Analysis (EDA)
Explore order patterns, customer behavior, product performance, and delivery-related trends.

### 6. Customer RFM Analysis
Segment customers using recency, frequency, and monetary value to understand purchasing behavior.

### 7. Delivery Delay Prediction
Create the binary target variable `is_late` for eligible delivered orders:

- `0` — Delivered on time or earlier.
- `1` — Delivered after the estimated delivery date.

### 8. Machine Learning
Build and compare the following classification models:

- Dummy Classifier
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- HistGradientBoosting Classifier

### 9. Model Evaluation
Evaluate the selected model on the held-out test dataset using Average Precision, Precision, Recall, F1-score, and a confusion matrix.

### 10. Business Insights and Reporting
Generate delivery prediction summaries, identify high-risk orders based on configured probability thresholds, and export results for further analysis.

## 📈 Model Evaluation Metrics

| Metric | Purpose |
|---|---|
| Average Precision | Evaluates the model's ability to rank late deliveries above on-time deliveries |
| Precision | Measures how many predicted late deliveries were actually late |
| Recall | Measures how many actual late deliveries were detected |
| F1-score | Balances precision and recall |
| Confusion Matrix | Shows correct predictions and classification errors |

Because late deliveries may be less common than on-time deliveries, evaluating multiple metrics is important.

## 📂 Repository Structure

```text
10-Olist-E-Commerce-Analytics-and-Delivery-Delay-Prediction/
│
├── data/
│   ├── README.md
│   └── raw/
│       ├── olist_customers_dataset.csv
│       ├── olist_geolocation_dataset.csv
│       ├── olist_order_items_dataset.csv
│       ├── olist_order_payments_dataset.csv
│       ├── olist_order_reviews_dataset.csv
│       ├── olist_orders_dataset.csv
│       ├── olist_products_dataset.csv
│       ├── olist_sellers_dataset.csv
│       └── product_category_name_translation.csv
│
├── src/
│   ├── README.md
│   └── main.py
│
├── images/
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── model_performance_metrics.png
│
├── models/
│   └── best_delivery_delay_model.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── delivery_predictions.csv
│   ├── model_evaluation_metrics.csv
│   └── business_insights.json
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

*Note: This structure reflects the intended organization of the project. Keep only the files and directories that actually exist in your repository, and update the script filename if you use a different name.*

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Folder

```bash
cd 10-Olist-E-Commerce-Analytics-and-Delivery-Delay-Prediction
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Project

If your entry-point script is `src/main.py`, execute:

```bash
python src/main.py
```

Ensure the dataset paths and output directory configurations are correct before running the pipeline.

## 📤 Generated Outputs

The project is designed to generate the following artifacts:

- **Model comparison:** Cross-validation results for the classification models.
- **Delivery predictions:** Actual outcomes, predicted outcomes, probabilities, and risk categories for test-set orders.
- **Evaluation metrics:** Performance scores for the selected model.
- **Conf
