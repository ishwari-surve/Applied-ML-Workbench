# 📊 Data — Olist E-Commerce Analytics & Delivery Delay Prediction

## 📌 Overview

The `data` directory contains the datasets used for analyzing the Olist Brazilian e-commerce marketplace and developing a machine learning model to predict delivery delays.

The datasets provide information about customer orders, customers, sellers, products, payments, reviews, and order delivery timelines.

## 📂 Directory Structure

```text
data/
├── README.md
├── raw/
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
└── processed/
    └── (processed datasets, if generated)
```

*Note: The `processed/` directory is optional and should only be included if it exists in your project.*

## 🗃️ Dataset Description

| Dataset | Description |
|---|---|
| `olist_customers_dataset.csv` | Customer identifiers, ZIP codes, cities, and states |
| `olist_geolocation_dataset.csv` | Geographic coordinates associated with Brazilian ZIP codes |
| `olist_order_items_dataset.csv` | Order items, products, sellers, prices, freight, and shipping limits |
| `olist_order_payments_dataset.csv` | Payment methods, installments, and payment values |
| `olist_order_reviews_dataset.csv` | Customer review scores and review comments |
| `olist_orders_dataset.csv` | Order status and purchase, approval, shipping, and delivery timestamps |
| `olist_products_dataset.csv` | Product categories, dimensions, weight, and other product attributes |
| `olist_sellers_dataset.csv` | Seller identifiers, ZIP codes, cities, and states |
| `product_category_name_translation.csv` | Translation of Portuguese product category names into English |

## 🎯 Purpose of the Data

The datasets support the following analytical tasks:

- Data quality assessment and cleaning.
- Integration of customer, seller, order, product, and payment information.
- Exploratory data analysis (EDA).
- Customer segmentation using RFM analysis.
- Feature engineering for delivery-delay prediction.
- Binary classification of delivered orders as on-time or late.
- Business insight generation and Power BI reporting.

## 🔄 Data Processing Workflow

1. Load the raw CSV datasets.
2. Inspect columns, data types, missing values, and duplicate records.
3. Clean and prepare the individual datasets.
4. Merge related datasets using appropriate identifiers.
5. Engineer analytical and machine learning features.
6. Create the `is_late` target variable for eligible delivered orders.
7. Prepare the final data for chronological training and testing.

## ⚠️ Data Handling Guidelines

- Keep the original raw datasets unchanged.
- Perform cleaning and transformations through the source code.
- Avoid committing large generated files unless required by the project.
- Do not include credentials, personal secrets, or unnecessary sensitive data.
- Document the source and any transformations applied to datasets.
- Ensure generated datasets and model inputs are consistent with the project pipeline.

## 📚 Data Source

**Olist Brazilian E-Commerce Public Dataset**

The dataset is publicly available through Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Refer to the dataset provider's terms and licensing information before redistributing the data.

## 👩‍💻 Author

**Ishwari Vijaykumar Surve**

## 📝 Note

This directory stores the source data used by the project's analytics and machine learning pipeline. The exact processed files depend on the execution and configuration of the project.

