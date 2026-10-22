# 📈 Visualizations — Olist E-Commerce Analytics & Delivery Delay Prediction

## 📌 Overview

The `images` directory contains visualizations generated during Exploratory Data Analysis (EDA), customer segmentation, and machine learning model evaluation for the **Olist E-Commerce Analytics and Delivery Delay Prediction** project.

These charts help explain order trends, product sales, delivery performance, customer reviews, freight costs, customer segments, and model prediction quality.

## 📂 Directory Structure

```text
images/
├── README.md
├── 01_order_status_distribution.png
├── 02_monthly_orders_sales.png
├── 03_top_categories_by_sales.png
├── 04_delivery_performance.png
├── 05_review_score_distribution.png
├── 06_freight_cost_distribution.png
├── 07_rfm_customer_segments.png
├── confusion_matrix.png
├── model_performance_metrics.png
└── precision_recall_curve.png
```

## 📊 Exploratory Data Analysis Visualizations

### 1. Order Status Distribution

**File:** `01_order_status_distribution.png`

Displays the distribution of orders across statuses such as delivered, shipped, canceled, unavailable, invoiced, processing, created, and approved.

**Purpose:** Understand the distribution of order statuses and identify the dominant order outcome.

### 2. Monthly Orders and Product Sales

**File:** `02_monthly_orders_sales.png`

Shows monthly order volumes alongside product sales values over the dataset's observed period.

**Purpose:** Identify monthly trends, changes in order activity, and variations in product sales.

### 3. Top 10 Product Categories by Sales

**File:** `03_top_categories_by_sales.png`

Compares the top 10 product categories by total product sales.

**Purpose:** Identify categories contributing the most to product sales and support category-level business analysis.

### 4. Delivery Performance Distribution

**File:** `04_delivery_performance.png`

Shows the distribution of delivery differences relative to estimated delivery dates. Negative values indicate early deliveries, while positive values indicate late deliveries.

**Purpose:** Understand delivery timing patterns and the variation between actual and estimated delivery dates.

### 5. Customer Review Score Distribution

**File:** `05_review_score_distribution.png`

Displays the distribution of customer review scores from 1 to 5.

**Purpose:** Understand customer rating patterns and the distribution of review scores.

### 6. Freight Cost Distribution

**File:** `06_freight_cost_distribution.png`

Shows the distribution of total freight costs, with the visualization capped at the 99th percentile.

**Purpose:** Examine common freight cost ranges and identify the presence of high-cost observations.

### 7. RFM Customer Segmentation

**File:** `07_rfm_customer_segments.png`

Displays the number of customers in each RFM segment, including Loyal Customers, Hibernating, Potential Loyalists, Regular Customers, At Risk, Recent Customers, and Champions.

**Purpose:** Understand customer segment sizes and support customer retention and engagement strategies.

## 🤖 Machine Learning Evaluation Visualizations

### 8. Confusion Matrix

**File:** `confusion_matrix.png`

Displays the model's classification results for on-time and late deliveries.

The matrix represents:

- **True negatives:** On-time orders correctly predicted as on-time.
- **False positives:** On-time orders incorrectly predicted as late.
- **False negatives:** Late orders incorrectly predicted as on-time.
- **True positives:** Late orders correctly predicted as late.

**Purpose:** Understand the types and frequency of classification errors.

### 9. Model Performance Metrics

**File:** `model_performance_metrics.png`

Compares the selected model's Average Precision, Precision, Recall, and F1-score on the test dataset.

**Purpose:** Provide a compact overview of model performance across multiple evaluation metrics.

### 10. Precision–Recall Curve

**File:** `precision_recall_curve.png`

Shows the relationship between precision and recall across different classification thresholds.

**Purpose:** Evaluate the trade-off between identifying late deliveries and limiting incorrect late-delivery alerts, particularly when the target classes are imbalanced.

## 🎯 Business Applications

These visualizations support:

- E-commerce order and sales trend analysis.
- Product category performance assessment.
- Delivery performance monitoring.
- Customer satisfaction analysis.
- Freight cost distribution analysis.
- Customer segmentation and retention planning.
- Machine learning model evaluation.
- Further reporting and dashboard development using Power BI.

## 🛠️ Tools Used

- **Python** — data analysis and visualization workflow.
- **Pandas** — data manipulation and aggregation.
- **Matplotlib** — chart generation.
- **Scikit-learn** — machine learning evaluation metrics.

## 📝 Note

The charts represent the data and model results produced by the project pipeline. Machine learning evaluation charts reflect the test dataset used during execution. Actual values and interpretations should be confirmed against the underlying data and generated outputs.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

