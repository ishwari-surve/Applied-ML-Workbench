# 📊 Data — Olist E-Commerce Analytics & Delivery Delay Prediction

## 📌 Overview

The `data` directory contains the datasets used in the **Olist E-Commerce Analytics and Delivery Delay Prediction** project.

These datasets provide information about customers, sellers, orders, products, payments, reviews, and delivery timelines. They are used for data cleaning, exploratory data analysis (EDA), RFM analysis, feature engineering, machine learning, and business intelligence reporting.

## 📥 Dataset Source

**Dataset:** Olist Brazilian E-Commerce Public Dataset

**Source:** Kaggle

🔗 **Dataset Link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The dataset contains anonymized information about orders placed through the Olist Brazilian e-commerce marketplace.

Please refer to Kaggle's dataset page for source details and applicable usage terms.

## 📂 Directory Structure

```text
data/
├── README.md
└── raw/
    ├── olist_customers_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

## 🗃️ Dataset Files Description

| CSV File | Description |
|---|---|
| `olist_customers_dataset.csv` | Contains customer IDs, ZIP codes, cities, and states. |
| `olist_geolocation_dataset.csv` | Contains Brazilian ZIP-code prefixes and geographic coordinates. |
| `olist_order_items_dataset.csv` | Provides order item details, product and seller IDs, prices, freight values, and shipping limits. |
| `olist_order_payments_dataset.csv` | Contains payment methods, installment counts, and payment values. |
| `olist_order_reviews_dataset.csv` | Contains customer review scores, comments, and review-related timestamps. |
| `olist_orders_dataset.csv` | Includes order status and purchase, approval, carrier, estimated delivery, and actual delivery timestamps. |
| `olist_products_dataset.csv` | Contains product categories, dimensions, weight, and other product attributes. |
| `olist_sellers_dataset.csv` | Contains seller IDs, ZIP codes, cities, and states. |
| `product_category_name_translation.csv` | Maps Portuguese product category names to English translations. |

## 🎯 Purpose of the Data

The datasets support the following project objectives:

- Analyze e-commerce order and delivery patterns.
- Identify missing values, duplicate records, and data quality issues.
- Combine related datasets into a master analytical dataset.
- Perform exploratory data analysis and customer RFM segmentation.
- Engineer features for machine learning.
- Predict late deliveries using binary classification.
- Evaluate model performance and generate business insights.
- Export results for reporting and Power BI dashboards.

## 🔄 Data Processing Workflow

1. Load the raw CSV files.
2. Inspect dataset structures, data types, missing values, and duplicates.
3. Clean the datasets and prepare them for analysis.
4. Merge related datasets using appropriate keys.
5. Create analytical and predictive features.
6. Define the `is_late` target variable for eligible delivered orders.
7. Prepare training and test datasets using a chronological split.
8. Use the processed data for analysis, model training, and reporting.

## ⚠️ Data Handling Guidelines

- Preserve the original CSV files in the `raw/` directory.
- Perform cleaning and transformations through the Python source code.
- Keep generated outputs separate from the original datasets.
- Avoid uploading unnecessary duplicate datasets or large generated files.
- Document any additional data sources and transformations.
- Follow the dataset provider's usage terms.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

## 📝 Note

This directory stores the source datasets used by the project's data analytics and machine learning pipeline. Processed datasets and generated outputs depend on the implementation and configuration of the project.
