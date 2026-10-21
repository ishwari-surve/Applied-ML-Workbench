"""
------------------------------------------------------------
Project Name        : Olist E-Commerce Analytics and
                      Delivery Delay Prediction

Dataset Information :

Dataset Files       : 9 CSV files of the Olist Brazilian
                      E-Commerce Public Dataset (data/raw/)
                      customers, geolocation, order items,
                      payments, reviews, orders, products,
                      sellers, category translation

Orders              : 99,441 orders (Sep 2016 to Oct 2018)

Target              : is_late -> 1 = delivered after the
                                 estimated date, 0 = on time
                      (delivered orders only, 8.11 % are late)

Project Parts       : A. Data integration and cleaning
                      B. Exploratory analytics and RFM analysis
                      C. Delivery delay prediction
                      D. Insights and Power BI export

Machine Learning Information :

Models Compared     : Baseline (always on time)
                      Single models -> Logistic Regression,
                                       Decision Tree
                      Ensemble      -> Random Forest,
                                       Gradient Boosting

Techniques Used     : Multi-table joins, feature engineering,
                      seller-customer distance, time-based split,
                      class weights for the imbalance,
                      5-fold cross validation,
                      model saving and loading (joblib)

Library             : Pandas, NumPy, Matplotlib, Scikit-Learn

Problem Type        : Binary Classification (imbalanced)

Evaluation Metrics  : AUPRC, Precision, Recall, F1-score
                      (for the late class), Confusion Matrix

Author              : Ishwari Vijaykumar Surve

Date                : 19/10/2026
------------------------------------------------------------
"""

############################################################
# Required Python Packages
############################################################


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Preprocessing and pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Machine learning models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Model selection
from sklearn.model_selection import TimeSeriesSplit, cross_validate

# Evaluation metrics
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

# Approximate geographical boundaries of Brazil
BRAZIL_LATITUDE = (-34.0, 6.0)
BRAZIL_LONGITUDE = (-74.0, -34.0)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# Date columns to convert into datetime format
DATE_COLUMNS = {
    "orders": (
        "%Y-%m-%d %H:%M:%S",
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    ),
    "items": (
        "%Y-%m-%d %H:%M:%S",
        ["shipping_limit_date"]
    ),
    "reviews": (
        "%Y-%m-%d %H:%M:%S",
        [
            "review_creation_date",
            "review_answer_timestamp"
        ]
    )
}


# Relationships between the Olist datasets
TABLE_LINKS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("items", "order_id", "orders", "order_id"),
    ("items", "product_id", "products", "product_id"),
    ("items", "seller_id", "sellers", "seller_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("reviews", "order_id", "orders", "order_id"),
    (
        "translation",
        "product_category_name",
        "products",
        "product_category_name"
    )
]


# Unique key columns for each dataset
UNIQUE_KEYS = {
    "customers": ["customer_id"],
    "geolocation": [],
    "items": [],
    "payments": [],
    "reviews": ["review_id"],
    "orders": ["order_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "translation": ["product_category_name"]
}



############################################################
# File Paths  : BASE_DIR, RAW_DIR (data/raw), IMAGES_DIR,
#               MODELS_DIR, OUTPUTS_DIR, MODEL_PATH
# Constants   : TARGET_HEADER, DATE_FORMAT = "%d-%m-%Y %H:%M",
#               TEST_SIZE, RANDOM_STATE, CV_FOLDS, BORDER
############################################################
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# The 9 CSV files of the Olist dataset (table name : file name)
DATASET_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv"
}

############################################################
# Function Name : load_datasets
# Description   : Load the 9 Olist CSV files from the raw data
#                 folder. It checks that every file exists
#                 before reading, and it shows the number of
#                 rows and columns of every table.
#                 The dates are not converted here. The
#                 cleaning step converts them.
# Input         : raw_dir - Path of the folder with the CSV files
# Output        : Returns dictionary of table name and dataframe
# Author        : Ishwari Vijaykumar Surve
# Date          : 19/10/2026
############################################################

def load_datasets(raw_dir):

    # Check that every file exists
    missing_files = [
        file_name for file_name in DATASET_FILES.values()
        if not os.path.exists(os.path.join(raw_dir, file_name))
    ]

    if missing_files:
        raise FileNotFoundError(
            "These files are missing in data/raw :\n"
            + "\n".join(missing_files)
            + "\nDownload them from the Olist dataset on Kaggle."
        )

    datasets = {}

    print()
    print(f"{'Table':<14}{'Rows':>12}{'Columns':>10}")
    print("-" * 36)

    for name, file_name in DATASET_FILES.items():

        df = pd.read_csv(os.path.join(raw_dir, file_name))

        datasets[name] = df

        print(f"{name:<14}{df.shape[0]:>12,}{df.shape[1]:>10}")

    print("-" * 36)
    print("Total tables loaded :", len(datasets))

    return datasets

############################################################
# Function Name : understand_datasets
# Description   : Display the structure of all nine Olist tables:
#                 dimensions, column names, data types and the
#                 statistical information of the numeric and the
#                 text columns. At the end it checks the links
#                 between the tables (for example every order of
#                 the items table must exist in the orders table).
# Input         : datasets - Dictionary of table name and dataframe
#                            returned by load_datasets()
# Output        : Displays dataset information for understanding
#                 the structure and the contents of every table
# Author        : Ishwari Vijaykumar Surve
# Date          : 21/10/2026
############################################################

def understand_datasets(datasets):

    for name, df in datasets.items():

        print()
        print("=" * 60)
        print("Dataset Name :", name.capitalize())
        print("=" * 60)

        print("Number of Rows    :", f"{df.shape[0]:,}")
        print("Number of Columns :", df.shape[1])

        print()
        print("Column Names :")
        print(df.columns.tolist())

        print()
        print("Data Types :")
        print(df.dtypes)

        # Numeric columns: count, mean, std, min and max
        numeric = df.select_dtypes(include="number")

        if not numeric.empty:

            print()
            print("Statistical Information (numeric columns) :")
            print(numeric.describe().T[["count", "mean", "std", "min", "max"]].round(2))

        # Text columns: count, unique values, most common value
        text = df.select_dtypes(exclude="number")

        if not text.empty:

            print()
            print("Statistical Information (text columns) :")

            with pd.option_context("display.max_columns", None,
                                   "display.width", 200):
                print(text.describe().T)

    # Check the links between the tables
    print()
    print("=" * 60)
    print("Links between the tables (values not found in the parent table)")
    print("=" * 60)

    for child, child_column, parent, parent_column in TABLE_LINKS:

        child_values = datasets[child][child_column].dropna().unique()

        parent_values = set(datasets[parent][parent_column].unique())

        not_found = sum(1 for value in child_values if value not in parent_values)

        print(f"{child}.{child_column} -> {parent}.{parent_column} : "
              f"{not_found:,} of {len(child_values):,} not found")

############################################################
# Function Name : check_missing_values
# Description   : Display the missing values of every table:
#                 the columns that have empty cells, the number
#                 of empty cells and their percentage.
#                 It also shows why the delivery date of some
#                 orders is empty (the order is not delivered).
# Input         : datasets - Dictionary of table name and dataframe
# Output        : Displays the missing values of every table
# Author        : Ishwari Vijaykumar Surve
# Date          : 21/10/2026
############################################################

def check_missing_values(datasets):

    summary = []

    for name, df in datasets.items():

        missing = df.isnull().sum()

        missing = missing[missing > 0]

        print()
        print("=" * 60)
        print("Dataset Name :", name.capitalize())
        print("=" * 60)

        if missing.empty:

            print("No missing values")

        else:

            print(f"{'Column':<34}{'Missing':>10}{'% of rows':>12}")

            for column, count in missing.items():

                print(f"{column:<34}{count:>10,}{count / len(df) * 100:>11.2f}%")

        summary.append({
            "Table": name,
            "Columns with missing values": len(missing),
            "Missing cells": int(missing.sum())
        })

    # Summary of all tables
    print()
    print("=" * 60)
    print("Summary of the missing values")
    print("=" * 60)

    print(pd.DataFrame(summary).to_string(index=False))

    # Why is the delivery date empty? Check it with the order status
    orders = datasets["orders"]

    no_delivery_date = orders[orders["order_delivered_customer_date"].isnull()]

    print()
    print("=" * 60)
    print("Orders without a delivery date, by order status")
    print("=" * 60)

    print(no_delivery_date["order_status"].value_counts().to_string())

    delivered_without_date = int(
        (no_delivery_date["order_status"] == "delivered").sum()
    )

    print()
    print("Delivered orders without a delivery date :", delivered_without_date)
    print("These orders cannot be used for the delay target.")

############################################################
# Function Name : check_duplicates
# Description   : Display the duplicate rows of every table and
#                 check that the key columns are unique
#                 (for example one row for every order in the
#                 orders table). It also shows the tables that
#                 need special care in the next steps.
# Input         : datasets - Dictionary of table name and dataframe
# Output        : Displays duplicate rows and duplicate keys
# Author        : Ishwari Vijaykumar Surve
# Date          : 21/10/2026
############################################################

def check_duplicates(datasets):

    summary = []

    for name, df in datasets.items():

        duplicate_rows = int(df.duplicated().sum())

        print()
        print("=" * 60)
        print("Dataset Name :", name.capitalize())
        print("=" * 60)

        print(f"Duplicate rows (all columns the same) : {duplicate_rows:,} "
              f"({duplicate_rows / len(df) * 100:.2f}%)")

        duplicate_keys = 0

        for key in UNIQUE_KEYS.get(name, []):

            repeated = int(df.duplicated(subset=key).sum())

            duplicate_keys += repeated

            print(f"Repeated values in {key} : {repeated:,}")

        if name == "geolocation":

            zip_codes = df["geolocation_zip_code_prefix"].nunique()

            print(f"Rows per zip code : {len(df):,} rows for {zip_codes:,} zip codes")

        summary.append({
            "Table": name,
            "Duplicate rows": duplicate_rows,
            "Repeated key values": duplicate_keys
        })

    # Summary of all tables
    print()
    print("=" * 60)
    print("Summary of the duplicates")
    print("=" * 60)

    print(pd.DataFrame(summary).to_string(index=False))

    # Information for the next steps
    customers = datasets["customers"]

    repeat_customers = int(
        (customers["customer_unique_id"].value_counts() > 1).sum()
    )

    reviews = datasets["reviews"]

    orders_with_many_reviews = int(
        (reviews["order_id"].value_counts() > 1).sum()
    )

    print()
    print("=" * 60)
    print("What the next steps must do")
    print("=" * 60)

    print("geolocation : many rows for every zip code -> keep one row "
          "for every zip code (Step 5)")
    print(f"reviews     : {orders_with_many_reviews:,} orders have more than "
          "one review -> keep one review for every order (Step 6)")
    print(f"customers   : {repeat_customers:,} real customers ordered more than "
          "once -> this is correct, it is used for the RFM analysis")

############################################################
# Function Name : clean_datasets
# Description   : Clean the tables:
#                 1. convert the date columns to datetime with
#                    the exact format (the orders file uses
#                    day-month-year)
#                 2. translate the product categories to English
#                 3. remove the points outside Brazil and keep one
#                    row for every zip code in the geolocation table
#                 The original dictionary is not changed.
# Input         : datasets - Dictionary of table name and dataframe
# Output        : Returns the dictionary of clean tables
# Author        : Ishwari Vijaykumar Surve
# Date          : 21/10/2026
############################################################

def clean_datasets(datasets):

    clean = dict(datasets)

    # 1. Dates : text to datetime
    print()
    print("Converting the dates :")

    for name, (date_format, columns) in DATE_COLUMNS.items():

        df = datasets[name].copy()

        missing_before = df[columns].isnull().sum().sum()

        for column in columns:
            df[column] = pd.to_datetime(df[column], format=date_format, errors="coerce")

        new_missing = df[columns].isnull().sum().sum() - missing_before

        clean[name] = df

        print(f"{name:<8}: {len(columns)} date columns converted, "
              f"new missing values = {new_missing}")

    # 2. Product categories : English names
    print()
    print("Translating the product categories :")

    products = datasets["products"].rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length"
    })

    english = datasets["translation"].set_index(
        "product_category_name"
    )["product_category_name_english"]

    category = products["product_category_name"]

    translated = category.map(english)

    products["product_category_english"] = translated.fillna(category).fillna("unknown")

    clean["products"] = products

    print(f"{int(translated.notnull().sum()):,} products translated, "
          f"{int((translated.isnull() & category.notnull()).sum())} keep the "
          f"Portuguese name, {int(category.isnull().sum())} are 'unknown'")

    # 3. Geolocation : inside Brazil, one row for every zip code
    print()
    print("Cleaning the geolocation table :")

    geolocation = datasets["geolocation"]

    inside_brazil = (
        geolocation["geolocation_lat"].between(*BRAZIL_LATITUDE)
        & geolocation["geolocation_lng"].between(*BRAZIL_LONGITUDE)
    )

    clean["geolocation"] = (
        geolocation[inside_brazil]
        .groupby("geolocation_zip_code_prefix", as_index=False)
        [["geolocation_lat", "geolocation_lng"]]
        .median()
    )

    print(f"{int((~inside_brazil).sum())} points outside Brazil removed, "
          f"{len(geolocation):,} rows reduced to {len(clean['geolocation']):,} "
          "rows (one for every zip code)")

    return clean

############################################################
# Function Name : merge_datasets
# Description   : Build the master table with ONE row for every
#                 order. Every table that has many rows for an
#                 order (items, payments, reviews) is reduced to
#                 one row for every order first, then all tables
#                 are joined to the orders table.
#                 It checks that the number of orders did not
#                 change.
# Input         : clean - Dictionary of clean tables
# Output        : Returns the master table
# Author        : Ishwari Vijaykumar Surve
# Date          : 21/10/2026
############################################################

def merge_datasets(clean):

    orders = clean["orders"]

    # Items and products : one row for every order
    items = clean["items"].merge(
        clean["products"][[
            "product_id", "product_category_english", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm"
        ]],
        on="product_id",
        how="left"
    )

    items["volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )

    # The first item of the order gives the main product and the main seller
    first_item = (
        items.sort_values("order_item_id")
        .drop_duplicates("order_id")
        [["order_id", "product_id", "seller_id", "product_category_english"]]
    )

    item_totals = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum")
    )

    item_size = items.groupby("order_id")[["product_weight_g", "volume_cm3"]].sum(min_count=1)

    item_size.columns = ["total_weight_g", "total_volume_cm3"]

    # Payments : one row for every order
    payments = clean["payments"]

    payment_totals = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max")
    )

    main_payment = (
        payments.sort_values("payment_sequential")
        .drop_duplicates("order_id")
        .set_index("order_id")[["payment_type"]]
    )

    # Reviews : the latest review of every order
    review = (
        clean["reviews"].sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="last")
        .set_index("order_id")[["review_score"]]
    )

    # Locations of the customer and of the main seller
    location = clean["geolocation"].set_index("geolocation_zip_code_prefix")

    customers = clean["customers"].join(
        location.rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"}),
        on="customer_zip_code_prefix"
    )

    sellers = clean["sellers"].join(
        location.rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"}),
        on="seller_zip_code_prefix"
    )

    # Join everything to the orders table
    master = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(first_item, on="order_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .join(item_totals, on="order_id")
        .join(item_size, on="order_id")
        .join(payment_totals, on="order_id")
        .join(main_payment, on="order_id")
        .join(review, on="order_id")
    )

    # The number of orders must not change
    if len(master) != len(orders) or master["order_id"].duplicated().any():
        raise ValueError("The master table does not have one row for every order")

    print()
    print(f"Orders table  : {len(orders):,} rows")
    print(f"Master table  : {master.shape[0]:,} rows and {master.shape[1]} columns")
    print("Check passed  : one row for every order")

    print()
    print(f"Orders without items           : {int(master['n_items'].isnull().sum()):,}")
    print(f"Orders without a review        : {int(master['review_score'].isnull().sum()):,}")
    print(f"Orders without customer location : {int(master['customer_lat'].isnull().sum()):,}")
    print(f"Orders without seller location   : {int(master['seller_lat'].isnull().sum()):,}")

    return master


############################################################
# Function Name : feature_engineering
# Description   : Create useful features from the master table:
#                 1. Seller-customer distance
#                 2. Freight ratio
#                 3. Product weight and volume
#                 4. Order approval and delivery time features
#                 5. Estimated and actual delivery durations
# Input         : master - Master table returned by merge_datasets()
# Output        : Returns the master table with new features
############################################################

def feature_engineering(master):

    # Work on a copy to preserve the original master table
    df = master.copy()

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Seller-customer geographical distance
    # --------------------------------------------------------
    # Haversine formula calculates approximate distance in km

    customer_lat = np.radians(df["customer_lat"])
    customer_lng = np.radians(df["customer_lng"])
    seller_lat = np.radians(df["seller_lat"])
    seller_lng = np.radians(df["seller_lng"])

    lat_difference = seller_lat - customer_lat
    lng_difference = seller_lng - customer_lng

    a = (
        np.sin(lat_difference / 2) ** 2
        + np.cos(customer_lat)
        * np.cos(seller_lat)
        * np.sin(lng_difference / 2) ** 2
    )

    a = a.clip(0, 1)

    df["seller_customer_distance_km"] = (
        6371 * 2 * np.arcsin(np.sqrt(a))
    )

    # Missing coordinates produce missing distances
    missing_coordinates = (
        df["customer_lat"].isna()
        | df["customer_lng"].isna()
        | df["seller_lat"].isna()
        | df["seller_lng"].isna()
    )

    df.loc[
        missing_coordinates, "seller_customer_distance_km"
    ] = np.nan

    # --------------------------------------------------------
    # 2. Freight ratio
    # --------------------------------------------------------
    # Share of the item price plus freight that is freight cost

    order_value = df["total_price"] + df["total_freight"]

    df["freight_ratio"] = np.where(
        order_value > 0,
        df["total_freight"] / order_value,
        np.nan
    )

    # --------------------------------------------------------
    # 3. Product weight and volume
    # --------------------------------------------------------

    df["total_weight_kg"] = df["total_weight_g"] / 1000

    # Volume is already calculated in cubic centimetres
    df["total_volume_litres"] = df["total_volume_cm3"] / 1000

    # --------------------------------------------------------
    # 4. Order approval time
    # --------------------------------------------------------

    df["approval_time_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    # --------------------------------------------------------
    # 5. Estimated delivery duration
    # --------------------------------------------------------

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # 6. Actual delivery duration
    # --------------------------------------------------------

    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # 7. Delivery difference
    # --------------------------------------------------------
    # Positive value = late; negative value = early

    df["delivery_difference_days"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # 8. Purchase date/time features
    # --------------------------------------------------------

    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day_of_week"] = (
        df["order_purchase_timestamp"].dt.dayofweek
    )
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # --------------------------------------------------------
    # Display feature engineering summary
    # --------------------------------------------------------

    new_features = [
        "seller_customer_distance_km",
        "freight_ratio",
        "total_weight_kg",
        "total_volume_litres",
        "approval_time_hours",
        "estimated_delivery_days",
        "actual_delivery_days",
        "delivery_difference_days",
        "purchase_year",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour"
    ]

    print("\nNew features created:")
    for feature in new_features:
        print(f"- {feature}")

    print("\nFeature summary:")
    print(df[new_features].describe().round(2).to_string())

    print("\nMaster table shape after feature engineering:")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    print("\nFeature engineering completed.")

    return df


############################################################
# Function Name : perform_eda
# Description   : Perform exploratory data analysis of sales,
#                 orders, product categories, delivery times,
#                 freight costs and customer reviews.
# Input         : master - Master table returned by
#                          feature_engineering()
#                 clean  - Dictionary of cleaned datasets
# Output        : Saves EDA charts in the images folder
############################################################

def perform_eda(master, clean):

    # Create the images directory
    images_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    df = master.copy()

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Order status distribution
    # --------------------------------------------------------

    print("\n1. Order Status Distribution")

    status_counts = df["order_status"].value_counts()

    print(status_counts.to_string())

    plt.figure(figsize=(9, 5))
    status_counts.plot(kind="bar")

    plt.title("Distribution of Order Status")
    plt.xlabel("Order Status")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(images_dir, "01_order_status_distribution.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # 2. Monthly order and sales trends
    # --------------------------------------------------------

    print("\n2. Monthly Order and Sales Trends")

    df["purchase_month_period"] = (
        df["order_purchase_timestamp"].dt.to_period("M")
    )

    monthly = df.groupby("purchase_month_period").agg(
        total_orders=("order_id", "nunique"),
        total_sales=("total_price", "sum")
    ).reset_index()

    monthly["purchase_month_period"] = (
        monthly["purchase_month_period"].astype(str)
    )

    print(monthly.tail(12).to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        monthly["purchase_month_period"],
        monthly["total_orders"],
        marker="o",
        label="Orders"
    )
    ax1.set_xlabel("Purchase Month")
    ax1.set_ylabel("Number of Orders")
    ax1.tick_params(axis="x", rotation=60)

    ax2 = ax1.twinx()
    ax2.plot(
        monthly["purchase_month_period"],
        monthly["total_sales"],
        marker="s",
        linestyle="--",
        label="Sales"
    )
    ax2.set_ylabel("Product Sales Value")

    plt.title("Monthly Orders and Product Sales")
    fig.tight_layout()

    plt.savefig(
        os.path.join(images_dir, "02_monthly_orders_sales.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # 3. Top product categories by sales
    # --------------------------------------------------------
    # Use item-level data so every product contributes to sales.
    # This avoids relying only on the first product in an order.

    print("\n3. Top Product Categories by Sales")

    items = clean["items"].copy()
    products = clean["products"].copy()

    category_items = items.merge(
        products[["product_id", "product_category_english"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    category_items["product_category_english"] = (
        category_items["product_category_english"].fillna("unknown")
    )

    category_sales = (
        category_items.groupby("product_category_english")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print(category_sales.to_string())

    plt.figure(figsize=(11, 6))
    category_sales.sort_values().plot(kind="barh")

    plt.title("Top 10 Product Categories by Sales")
    plt.xlabel("Total Product Sales")
    plt.ylabel("Product Category")
    plt.tight_layout()

    plt.savefig(
        os.path.join(images_dir, "03_top_categories_by_sales.png"),
        dpi=300
    )
    plt.close()

    # --------------------------------------------------------
    # 4. Delivery time and late-delivery analysis
    # --------------------------------------------------------

    print("\n4. Delivery Performance")

    delivered = df[
        (df["order_status"] == "delivered")
        & df["actual_delivery_days"].notna()
        & df["delivery_difference_days"].notna()
    ].copy()

    delivered["is_late"] = (
        delivered["delivery_difference_days"] > 0
    ).astype(int)

    late_count = int(delivered["is_late"].sum())
    delivered_count = len(delivered)

    late_rate = (
        late_count / delivered_count * 100
        if delivered_count else 0
    )

    print(f"Delivered orders analysed : {delivered_count:,}")
    print(f"Late deliveries           : {late_count:,}")
    print(f"Late delivery rate        : {late_rate:.2f}%")

    if not delivered.empty:

        plt.figure(figsize=(9, 5))

        delivered["delivery_difference_days"].clip(
            lower=-60, upper=60
        ).hist(bins=40)

        plt.axvline(0, linestyle="--")

        plt.title("Delivery Difference from Estimated Date")
        plt.xlabel("Days (negative = early, positive = late)")
        plt.ylabel("Number of Orders")
        plt.tight_layout()

        plt.savefig(
            os.path.join(images_dir, "04_delivery_performance.png"),
            dpi=300
        )
        plt.close()

    # --------------------------------------------------------
    # 5. Customer review score distribution
    # --------------------------------------------------------

    print("\n5. Customer Review Scores")

    review_counts = df["review_score"].dropna().value_counts().sort_index()

    print(review_counts.to_string())

    if not review_counts.empty:

        plt.figure(figsize=(8, 5))
        review_counts.plot(kind="bar")

        plt.title("Customer Review Score Distribution")
        plt.xlabel("Review Score")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            os.path.join(images_dir, "05_review_score_distribution.png"),
            dpi=300
        )
        plt.close()

    # --------------------------------------------------------
    # 6. Freight cost analysis
    # --------------------------------------------------------

    print("\n6. Freight Cost Analysis")

    freight_summary = df[
        ["total_freight", "total_price", "freight_ratio"]
    ].describe().round(2)

    print(freight_summary.to_string())

    freight_data = df["total_freight"].dropna()

    if not freight_data.empty:

        plt.figure(figsize=(9, 5))
        freight_data.clip(
            upper=freight_data.quantile(0.99)
        ).hist(bins=40)

        plt.title("Distribution of Freight Costs (99th Percentile Cap)")
        plt.xlabel("Total Freight Cost")
        plt.ylabel("Number of Orders")
        plt.tight_layout()

        plt.savefig(
            os.path.join(images_dir, "06_freight_cost_distribution.png"),
            dpi=300
        )
        plt.close()

    print("\nEDA charts saved in:", images_dir)
    print("Exploratory data analysis completed.")


############################################################
# Function Name : rfm_analysis
# Description   : Calculate Recency, Frequency and Monetary
#                 values, assign RFM scores and segment customers.
# Input         : master - Master table returned by
#                          feature_engineering()
# Output        : Saves customer RFM segments to a CSV file
#                 and saves a customer segmentation chart.
############################################################

def rfm_analysis(master):

    print("\n" + "=" * 60)
    print("CUSTOMER RFM ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Prepare customer order data
    # --------------------------------------------------------

    required_columns = [
        "customer_unique_id",
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "total_price"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in master.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing from master table: {missing_columns}"
        )

    df = master[required_columns].copy()

    # Use delivered orders for meaningful purchase frequency
    df = df[
        (df["order_status"] == "delivered")
        & df["customer_unique_id"].notna()
        & df["order_purchase_timestamp"].notna()
        & df["total_price"].notna()
    ].copy()

    if df.empty:
        raise ValueError(
            "No eligible delivered orders are available for RFM analysis."
        )

    # --------------------------------------------------------
    # 2. Calculate RFM values
    # --------------------------------------------------------

    reference_date = (
        df["order_purchase_timestamp"].max().normalize()
        + pd.Timedelta(days=1)
    )

    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("total_price", "sum")
    )

    rfm["recency"] = (
        reference_date - rfm["last_purchase"].dt.normalize()
    ).dt.days

    rfm = rfm[
        ["recency", "frequency", "monetary"]
    ].reset_index()

    # --------------------------------------------------------
    # 3. Assign scores from 1 to 5
    # --------------------------------------------------------
    # Higher recency score = more recent purchase
    # Higher frequency/monetary score = better customer value

    def assign_score(series, higher_is_better=True):

        ranks = series.rank(
            method="first",
            ascending=higher_is_better
        )

        return pd.qcut(
            ranks,
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

    rfm["R_score"] = assign_score(
        rfm["recency"], higher_is_better=False
    )

    rfm["F_score"] = assign_score(
        rfm["frequency"], higher_is_better=True
    )

    rfm["M_score"] = assign_score(
        rfm["monetary"], higher_is_better=True
    )

    rfm["RFM_score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )

    # --------------------------------------------------------
    # 4. Create customer segments
    # --------------------------------------------------------

    def classify_customer(row):

        r = row["R_score"]
        f = row["F_score"]
        m = row["M_score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        elif f >= 4:
            return "Loyal Customers"

        elif r >= 4 and f >= 2:
            return "Potential Loyalists"

        elif r >= 4:
            return "Recent Customers"

        elif r <= 2 and f >= 3:
            return "At Risk"

        elif r <= 2 and f <= 2:
            return "Hibernating"

        else:
            return "Regular Customers"

    rfm["customer_segment"] = rfm.apply(
        classify_customer, axis=1
    )

    # --------------------------------------------------------
    # 5. Display RFM summary
    # --------------------------------------------------------

    segment_summary = rfm.groupby(
        "customer_segment"
    ).agg(
        customers=("customer_unique_id", "count"),
        average_recency_days=("recency", "mean"),
        average_orders=("frequency", "mean"),
        average_spending=("monetary", "mean"),
        total_spending=("monetary", "sum")
    ).round(2).sort_values(
        "customers", ascending=False
    )

    print("\nCustomer Segment Summary:")
    print(segment_summary.to_string())

    print(f"\nUnique customers analysed : {len(rfm):,}")
    print(f"Reference date             : {reference_date.date()}")

    # --------------------------------------------------------
    # 6. Save the RFM results
    # --------------------------------------------------------

    outputs_dir = os.path.join(BASE_DIR, "outputs")
    images_dir = os.path.join(BASE_DIR, "images")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    rfm.to_csv(
        os.path.join(outputs_dir, "rfm_customer_segments.csv"),
        index=False
    )

    segment_summary.to_csv(
        os.path.join(outputs_dir, "rfm_segment_summary.csv")
    )

    # --------------------------------------------------------
    # 7. Plot customer segment distribution
    # --------------------------------------------------------

    segment_counts = (
        rfm["customer_segment"]
        .value_counts()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    segment_counts.plot(kind="barh")

    plt.title("Customer Segmentation Using RFM Analysis")
    plt.xlabel("Number of Customers")
    plt.ylabel("Customer Segment")
    plt.tight_layout()

    plt.savefig(
        os.path.join(images_dir, "07_rfm_customer_segments.png"),
        dpi=300
    )

    plt.close()

    print("\nRFM customer data saved to:")
    print(os.path.join(outputs_dir, "rfm_customer_segments.csv"))

    print("\nRFM segment summary saved to:")
    print(os.path.join(outputs_dir, "rfm_segment_summary.csv"))

    print("\nRFM chart saved to:")
    print(os.path.join(images_dir, "07_rfm_customer_segments.png"))

    print("\nRFM analysis completed successfully.")

    return rfm


############################################################
# Function Name : create_target
# Description   : Create the binary target variable is_late
#                 for delivery delay prediction.
# Input         : master - Feature-engineered master table
# Output        : Returns delivered orders with the is_late target
############################################################

def create_target(master):

    print("\n" + "=" * 60)
    print("STEP 10: CREATE TARGET VARIABLE")
    print("=" * 60)

    # Work on a copy to preserve the original master table
    df = master.copy()

    # Required columns
    required_columns = [
        "order_status",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    # Check whether required columns exist
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep delivered orders with valid delivery dates
    df = df[
        (df["order_status"] == "delivered")
        & df["order_delivered_customer_date"].notna()
        & df["order_estimated_delivery_date"].notna()
    ].copy()

    if df.empty:
        raise ValueError(
            "No delivered orders with valid delivery dates found."
        )

    # Create target:
    # 1 = delivered late
    # 0 = delivered on time or early
    df["is_late"] = (
        df["order_delivered_customer_date"]
        > df["order_estimated_delivery_date"]
    ).astype(int)

    # Display target distribution
    target_counts = df["is_late"].value_counts().sort_index()

    total_orders = len(df)
    late_orders = int(df["is_late"].sum())
    on_time_orders = total_orders - late_orders

    print(f"\nEligible delivered orders : {total_orders:,}")
    print(f"On-time or early orders   : {on_time_orders:,}")
    print(f"Late orders               : {late_orders:,}")

    print("\nTarget distribution:")
    for target_value, count in target_counts.items():
        label = "Late" if target_value == 1 else "On time / Early"
        percentage = count / total_orders * 100

        print(
            f"{label:<18}: {count:>8,} "
            f"({percentage:.2f}%)"
        )

    print("\nTarget variable 'is_late' created successfully.")

    return df


############################################################
# Function Name : remove_leakage
# Description   : Remove columns that reveal the delivery
#                 outcome or are unsuitable as model inputs.
# Input         : target_data - Dataset returned by create_target()
# Output        : Returns the dataset after removing leakage columns
############################################################

def remove_leakage(target_data):

    print("\n" + "=" * 60)
    print("STEP 11: REMOVE DATA LEAKAGE")
    print("=" * 60)

    # Work on a copy to preserve the original dataset
    df = target_data.copy()

    # Columns that must not be used as model input features
    leakage_columns = [
        # Actual delivery outcome
        "order_delivered_customer_date",
        "actual_delivery_days",
        "delivery_difference_days",

        # Other post-purchase or post-delivery information
        "review_score",
        "order_delivered_carrier_date",

        # Administrative dates not intended as model inputs
        "order_status"
    ]

    # Remove only columns that exist in the dataset
    columns_to_remove = [
        column for column in leakage_columns
        if column in df.columns
    ]

    df = df.drop(columns=columns_to_remove)

    # The target must remain available
    if "is_late" not in df.columns:
        raise ValueError(
            "Target column 'is_late' is missing. "
            "Run create_target() before remove_leakage()."
        )

    print("\nColumns removed:")
    if columns_to_remove:
        for column in columns_to_remove:
            print(f"- {column}")
    else:
        print("No listed leakage columns were present.")

    print("\nTarget column retained: is_late")
    print("Remaining rows   :", f"{len(df):,}")
    print("Remaining columns:", len(df.columns))

    print("\nData leakage removal completed.")

    return df


############################################################
# Function Name : prepare_ml_data
# Description   : Select model features, separate the target,
#                 and split data chronologically into training
#                 and testing sets.
# Input         : model_data - Dataset returned by remove_leakage()
# Output        : Dictionary containing X_train, X_test,
#                 y_train, y_test, and feature information
############################################################

def prepare_ml_data(model_data):

    print("\n" + "=" * 60)
    print("STEP 12: PREPARE MACHINE LEARNING DATA")
    print("=" * 60)

    df = model_data.copy()

    # --------------------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "is_late",
        "order_purchase_timestamp"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. Define features available for prediction
    # --------------------------------------------------------
    # Select features explicitly to prevent identifiers,
    # target leakage, and unwanted columns from entering X.

    candidate_features = [
        # Purchase timing
        "purchase_year",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour",

        # Order and product information
        "n_items",
        "n_sellers",
        "total_price",
        "total_freight",
        "total_weight_kg",
        "total_volume_litres",
        "freight_ratio",
        "product_category_english",

        # Payment information
        "total_payment_value",
        "max_installments",
        "payment_type",

        # Geographic information
        "customer_state",
        "seller_state",
        "seller_customer_distance_km",

        # Information available after approval, if prediction
        # is made at that point
        "approval_time_hours",
        "estimated_delivery_days"
    ]

    feature_columns = [
        column for column in candidate_features
        if column in df.columns
    ]

    if not feature_columns:
        raise ValueError(
            "No candidate features were found. "
            "Check the columns produced by merge_datasets() "
            "and feature_engineering()."
        )

    # --------------------------------------------------------
    # 3. Sort data by purchase timestamp
    # --------------------------------------------------------
    # Earlier orders are used for training.
    # Later orders are reserved for testing.

    df = df.dropna(
        subset=["order_purchase_timestamp", "is_late"]
    )

    df = df.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Separate features (X) and target (y)
    # --------------------------------------------------------

    X = df[feature_columns].copy()
    y = df["is_late"].astype(int).copy()

    # Replace infinite numeric values with missing values.
    # Imputation will be handled later in model pipelines.
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns

    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # --------------------------------------------------------
    # 5. Perform chronological 80/20 split
    # --------------------------------------------------------

    split_index = int(len(df) * 0.80)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Insufficient data for an 80/20 train-test split."
        )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # --------------------------------------------------------
    # 6. Identify numerical and categorical features
    # --------------------------------------------------------

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_features = [
        column for column in feature_columns
        if column not in categorical_features
    ]

    # --------------------------------------------------------
    # 7. Display preparation summary
    # --------------------------------------------------------

    print(f"\nTotal eligible orders : {len(df):,}")
    print(f"Training rows         : {len(X_train):,}")
    print(f"Testing rows          : {len(X_test):,}")
    print(f"Number of features    : {len(feature_columns)}")

    print("\nTraining target distribution:")
    print(
        y_train.value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "On time / Early", 1: "Late"})
        .to_string()
    )

    print("\nTesting target distribution:")
    print(
        y_test.value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "On time / Early", 1: "Late"})
        .to_string()
    )

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    print("\nTraining period:")
    print(
        df["order_purchase_timestamp"].iloc[0],
        "to",
        df["order_purchase_timestamp"].iloc[split_index - 1]
    )

    print("\nTesting period:")
    print(
        df["order_purchase_timestamp"].iloc[split_index],
        "to",
        df["order_purchase_timestamp"].iloc[-1]
    )

    print("\nMachine learning data preparation completed.")

    # Return all data required by the next steps.
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features
    }


############################################################
# Function Name : build_models
# Description   : Build preprocessing and classification
#                 pipelines for the baseline and four ML models.
# Input         : ml_data - Dictionary returned by prepare_ml_data()
# Output        : Dictionary containing the model pipelines
############################################################

def build_models(ml_data):

    print("\n" + "=" * 60)
    print("STEP 13: BUILD MACHINE LEARNING MODELS")
    print("=" * 60)

    # Get feature information from Step 12
    numerical_features = ml_data["numerical_features"]
    categorical_features = ml_data["categorical_features"]

    # --------------------------------------------------------
    # 1. Create numerical preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # --------------------------------------------------------
    # 2. Create categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
])

    # --------------------------------------------------------
    # 3. Combine preprocessing
    # --------------------------------------------------------

    transformers = []

    if numerical_features:
        transformers.append(
            ("numeric", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------------
    # 4. Define the baseline and candidate models
    # --------------------------------------------------------

    model_classifiers = {
        "Baseline": DummyClassifier(
            strategy="constant",
            constant=0
        ),

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    # --------------------------------------------------------
    # 5. Build a separate pipeline for every model
    # --------------------------------------------------------

    models = {}

    for model_name, classifier in model_classifiers.items():

        models[model_name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        print(f"Created pipeline: {model_name}")

    print("\nTotal pipelines created:", len(models))
    print("Preprocessing: imputation, scaling and categorical encoding")
    print("Model construction completed.")

    return models


# ============================================================
# STEP 14: COMPARE MODELS
# ============================================================

def compare_models(models, ml_data):
    """
    Compare classification models using time-series cross-validation.

    Models are evaluated only on the training data.
    The test data is reserved for final evaluation.
    """

    print("\n" + "=" * 60)
    print("STEP 14: MODEL COMPARISON")
    print("=" * 60)

    # Get training data from the prepared ML dataset.
    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]

    # Use chronological splits because the data is time ordered.
    time_cv = TimeSeriesSplit(n_splits=5)

    # Evaluate models using metrics suited to late-delivery detection.
    scoring_metrics = {
        "average_precision": "average_precision",
        "recall": "recall",
        "f1": "f1"
    }

    comparison_results = []

    for model_name, model in models.items():

        print(f"\nEvaluating: {model_name}")

        # Cross-validation uses training data only.
        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=time_cv,
            scoring=scoring_metrics,
            n_jobs=1,
            error_score="raise"
        )

        # Calculate the mean score across chronological folds.
        result = {
            "Model": model_name,
            "Average Precision": scores[
                "test_average_precision"
            ].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-Score": scores["test_f1"].mean()
        }

        comparison_results.append(result)

        print(
            f"Average Precision: "
            f"{result['Average Precision']:.4f}"
        )
        print(f"Recall: {result['Recall']:.4f}")
        print(f"F1-Score: {result['F1-Score']:.4f}")

    # Convert the results into a readable DataFrame.
    comparison_df = pd.DataFrame(comparison_results)

    # Sort by Average Precision for the initial comparison.
    comparison_df = comparison_df.sort_values(
        by="Average Precision",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "-" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("-" * 60)

    print(comparison_df.round(4).to_string(index=False))

    return comparison_df


# ============================================================
# STEP 15: SELECT BEST MODEL
# ============================================================

def select_best_model(comparison_df, models):
    """
    Select the best model based on cross-validation
    Average Precision.

    The test dataset is not used for model selection.
    """

    print("\n" + "=" * 60)
    print("STEP 15: SELECT BEST MODEL")
    print("=" * 60)

    # Validate the comparison results.
    if comparison_df.empty:
        raise ValueError(
            "Model comparison results are empty."
        )

    required_columns = [
        "Model",
        "Average Precision",
        "Recall",
        "F1-Score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in comparison_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing comparison columns: {missing_columns}"
        )

    # Select the model with the highest Average Precision.
    best_row = comparison_df.loc[
        comparison_df["Average Precision"].idxmax()
    ]

    best_model_name = best_row["Model"]

    # Retrieve the corresponding model pipeline.
    best_model = models[best_model_name]

    print(f"\nSelected Model: {best_model_name}")
    print(
        f"Average Precision: "
        f"{best_row['Average Precision']:.4f}"
    )
    print(f"Recall: {best_row['Recall']:.4f}")
    print(f"F1-Score: {best_row['F1-Score']:.4f}")

    print(
        "\nThe model is selected using cross-validation "
        "results from the training data."
    )

    return best_model_name, best_model


# ============================================================
# STEP 16.1: TRAIN BEST MODEL
# ============================================================

def train_best_model(best_model_name, best_model, ml_data):
    """
    Train the selected model using the complete training dataset.

    The test dataset remains untouched for final evaluation.
    """

    print("\n" + "=" * 60)
    print("STEP 16.1: TRAIN BEST MODEL")
    print("=" * 60)

    # Retrieve the training features and target.
    X_train = ml_data["X_train"]
    y_train = ml_data["y_train"]

    print(f"Selected Model : {best_model_name}")
    print(f"Training Rows  : {len(X_train)}")
    print(f"Training Features: {X_train.shape[1]}")

    # Fit the complete pipeline on training data.
    best_model.fit(X_train, y_train)

    print("\nModel training completed successfully.")

    return best_model


# ============================================================
# STEP 16.2: SAVE TRAINED MODEL
# ============================================================

def save_model(trained_model, best_model_name):
    """
    Save the trained model pipeline to the models directory.
    """

    print("\n" + "=" * 60)
    print("STEP 16.2: SAVE TRAINED MODEL")
    print("=" * 60)

    # Create the models directory if it does not exist.
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Define the output path.
    model_path = os.path.join(
        MODELS_DIR,
        "Best_delivery_delay_model.joblib"
    )

    # Save the trained pipeline.
    joblib.dump(trained_model, model_path)

    print(f"Model Name: {best_model_name}")
    print(f"Model saved successfully at:\n{model_path}")

    return model_path


# ============================================================
# STEP 16.3: LOAD SAVED MODEL
# ============================================================

def load_model(model_path):
    """
    Load a previously saved machine learning model pipeline.
    """

    print("\n" + "=" * 60)
    print("STEP 16.3: LOAD SAVED MODEL")
    print("=" * 60)

    # Check whether the model file exists.
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    # Load the saved model pipeline.
    loaded_model = joblib.load(model_path)

    print("Model loaded successfully.")
    print(f"Loaded from: {model_path}")

    return loaded_model


# ============================================================
# STEP 17: EVALUATE MODEL
# ============================================================

def evaluate_model(loaded_model, ml_data):
    """
    Evaluate the trained model on the unseen test dataset.
    """

    print("\n" + "=" * 60)
    print("STEP 17: MODEL EVALUATION")
    print("=" * 60)

    # Retrieve the test features and actual target values.
    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]

    # Generate predictions for the test dataset.
    y_pred = loaded_model.predict(X_test)

    # Calculate prediction probabilities for late deliveries.
    y_prob = loaded_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics.
    average_precision = average_precision_score(
        y_test, y_prob
    )

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # Generate the confusion matrix.
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    # Display evaluation results.
    print("\nTEST DATASET PERFORMANCE")
    print("-" * 60)

    print(f"Test Samples      : {len(y_test)}")
    print(f"Average Precision : {average_precision:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-Score          : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nConfusion Matrix Interpretation:")
    print(f"True On-Time      : {cm[0, 0]}")
    print(f"False Late Alert  : {cm[0, 1]}")
    print(f"Missed Late Order : {cm[1, 0]}")
    print(f"Correct Late Alert: {cm[1, 1]}")

    # Return metrics and predictions for later steps.
    evaluation_results = {
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    return evaluation_results


# ============================================================
# STEP 18: PLOT MODEL RESULTS
# ============================================================

def plot_results(evaluation_results):
    """
    Visualize the trained model's test-set performance.
    """

    print("\n" + "=" * 60)
    print("STEP 18: PLOT MODEL RESULTS")
    print("=" * 60)

    # Create the directory for saving result plots.
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Retrieve evaluation results.
    cm = evaluation_results["confusion_matrix"]
    y_test = evaluation_results["y_test"]
    y_prob = evaluation_results["y_prob"]

    # --------------------------------------------------------
    # Plot 1: Confusion Matrix
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        title="Delivery Delay Prediction - Confusion Matrix",
        xlabel="Predicted Class",
        ylabel="Actual Class",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["On-Time", "Late"],
        yticklabels=["On-Time", "Late"]
    )

    # Display the count inside each cell.
    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=12
            )

    fig.tight_layout()

    confusion_path = os.path.join(
        IMAGES_DIR,
        "confusion_matrix.png"
    )

    fig.savefig(confusion_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Confusion matrix saved: {confusion_path}")

    # --------------------------------------------------------
    # Plot 2: Precision-Recall Curve
    # --------------------------------------------------------

    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        recall_values,
        precision_values,
        label="Precision-Recall Curve"
    )

    ax.set(
        title="Precision-Recall Curve - Late Delivery Detection",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.05)
    )

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    pr_path = os.path.join(
        IMAGES_DIR,
        "precision_recall_curve.png"
    )

    fig.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Precision-recall curve saved: {pr_path}")

    # --------------------------------------------------------
    # Plot 3: Model Performance Metrics
    # --------------------------------------------------------

    metric_names = [
        "Average Precision",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    metric_values = [
        evaluation_results["average_precision"],
        evaluation_results["precision"],
        evaluation_results["recall"],
        evaluation_results["f1_score"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(metric_names, metric_values)

    ax.set(
        title="Delivery Delay Model - Test Performance",
        ylabel="Score",
        ylim=(0, 1.05)
    )

    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)

    # Add metric values above the bars.
    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    metrics_path = os.path.join(
        IMAGES_DIR,
        "model_performance_metrics.png"
    )

    fig.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Performance chart saved: {metrics_path}")

    print("\nAll model evaluation plots generated successfully.")


# ============================================================
# STEP 19: PREDICT ORDERS
# ============================================================

def predict_orders(loaded_model, ml_data):
    """
    Predict on-time or late delivery for test-set orders.

    The input data must contain the same features used
    when training the saved model.
    """

    print("\n" + "=" * 60)
    print("STEP 19: PREDICT ORDERS")
    print("=" * 60)

    # Retrieve test features and actual target values.
    X_test = ml_data["X_test"]
    y_test = ml_data["y_test"]

    if X_test.empty:
        raise ValueError(
            "Test dataset is empty. Cannot generate predictions."
        )

    # Generate predicted classes and late-delivery probabilities.
    predicted_classes = loaded_model.predict(X_test)
    late_probabilities = loaded_model.predict_proba(X_test)[:, 1]

    # Create a results DataFrame.
    prediction_results = pd.DataFrame(index=X_test.index)

    prediction_results["Actual_Status"] = np.asarray(y_test)
    prediction_results["Predicted_Status"] = predicted_classes
    prediction_results["Late_Probability"] = late_probabilities

    # Convert binary predictions into readable labels.
    prediction_results["Actual_Delivery_Status"] = (
        prediction_results["Actual_Status"]
        .map({0: "On-Time", 1: "Late"})
    )

    prediction_results["Predicted_Delivery_Status"] = (
        prediction_results["Predicted_Status"]
        .map({0: "On-Time", 1: "Late"})
    )

    # Assign a simple probability-based risk category.
    prediction_results["Risk_Level"] = np.where(
        late_probabilities >= 0.70,
        "High",
        np.where(
            late_probabilities >= 0.40,
            "Medium",
            "Low"
        )
    )

    # Display a sample of the predictions.
    print("\nSample Delivery Predictions:")
    print("-" * 60)

    display_columns = [
        "Actual_Delivery_Status",
        "Predicted_Delivery_Status",
        "Late_Probability",
        "Risk_Level"
    ]

    print(
        prediction_results[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nPrediction Summary:")
    print("-" * 60)
    print(f"Total Orders Predicted: {len(prediction_results)}")
    print(
        "Predicted Late Orders : "
        f"{(predicted_classes == 1).sum()}"
    )
    print(
        "Predicted On-Time     : "
        f"{(predicted_classes == 0).sum()}"
    )

    return prediction_results


# ============================================================
# STEP 20: GENERATE BUSINESS INSIGHTS
# ============================================================

def generate_insights(evaluation_results, prediction_results):
    """
    Generate business insights from delivery predictions
    and model evaluation results.
    """

    print("\n" + "=" * 60)
    print("STEP 20: BUSINESS INSIGHTS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Calculate delivery statistics
    # --------------------------------------------------------

    total_orders = len(prediction_results)

    if total_orders == 0:
        raise ValueError(
            "No prediction results available to generate insights."
        )

    actual_late_orders = (
        prediction_results["Actual_Status"] == 1
    ).sum()

    actual_ontime_orders = (
        prediction_results["Actual_Status"] == 0
    ).sum()

    predicted_late_orders = (
        prediction_results["Predicted_Status"] == 1
    ).sum()

    predicted_ontime_orders = (
        prediction_results["Predicted_Status"] == 0
    ).sum()

    high_risk_orders = (
        prediction_results["Risk_Level"] == "High"
    ).sum()

    actual_late_percentage = (
        actual_late_orders / total_orders
    ) * 100

    predicted_late_percentage = (
        predicted_late_orders / total_orders
    ) * 100

    high_risk_percentage = (
        high_risk_orders / total_orders
    ) * 100

    # --------------------------------------------------------
    # 2. Retrieve model evaluation metrics
    # --------------------------------------------------------

    average_precision = evaluation_results["average_precision"]
    precision = evaluation_results["precision"]
    recall = evaluation_results["recall"]
    f1 = evaluation_results["f1_score"]

    # --------------------------------------------------------
    # 3. Display business insights
    # --------------------------------------------------------

    print("\n1. DELIVERY PERFORMANCE")
    print("-" * 60)

    print(f"Orders Analyzed          : {total_orders:,}")
    print(
        f"Actually Late Orders     : "
        f"{actual_late_orders:,}"
    )
    print(
        f"Actually On-Time Orders  : "
        f"{actual_ontime_orders:,}"
    )
    print(
        f"Actual Late-Delivery Rate: "
        f"{actual_late_percentage:.2f}%"
    )

    print("\n2. MODEL PREDICTION INSIGHTS")
    print("-" * 60)

    print(
        f"Predicted Late Orders    : "
        f"{predicted_late_orders:,}"
    )
    print(
        f"Predicted On-Time Orders : "
        f"{predicted_ontime_orders:,}"
    )
    print(
        f"Predicted Late Rate      : "
        f"{predicted_late_percentage:.2f}%"
    )
    print(
        f"High-Risk Orders         : "
        f"{high_risk_orders:,}"
    )
    print(
        f"High-Risk Order Rate     : "
        f"{high_risk_percentage:.2f}%"
    )

    print("\n3. MODEL QUALITY")
    print("-" * 60)

    print(f"Average Precision: {average_precision:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1-Score         : {f1:.4f}")

    # --------------------------------------------------------
    # 4. Create actionable recommendations
    # --------------------------------------------------------

    recommendations = []

    if actual_late_percentage > 0:
        recommendations.append(
            "Investigate shipping and fulfillment processes "
            "to identify causes of delivery delays."
        )

    if recall < 0.50:
        recommendations.append(
            "The model misses many late orders. Review features, "
            "class imbalance handling, and the decision threshold."
        )
    else:
        recommendations.append(
            "Review detected late orders to identify opportunities "
            "for proactive customer communication."
        )

    if high_risk_orders > 0:
        recommendations.append(
            "Prioritize high-risk orders for operational review "
            "before promising delivery dates."
        )

    if precision < 0.50:
        recommendations.append(
            "Many late-delivery alerts may be incorrect. "
            "Review alert thresholds before operational deployment."
        )

    print("\n4. BUSINESS RECOMMENDATIONS")
    print("-" * 60)

    for number, recommendation in enumerate(
        recommendations, start=1
    ):
        print(f"{number}. {recommendation}")

    # --------------------------------------------------------
    # 5. Return insights for reporting and export
    # --------------------------------------------------------

    insights = {
        "total_orders": int(total_orders),
        "actual_late_orders": int(actual_late_orders),
        "actual_ontime_orders": int(actual_ontime_orders),
        "actual_late_percentage": float(actual_late_percentage),
        "predicted_late_orders": int(predicted_late_orders),
        "predicted_ontime_orders": int(predicted_ontime_orders),
        "predicted_late_percentage": float(predicted_late_percentage),
        "high_risk_orders": int(high_risk_orders),
        "high_risk_percentage": float(high_risk_percentage),
        "average_precision": float(average_precision),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "recommendations": recommendations
    }

    print("\nBusiness insights generated successfully.")

    return insights


# ============================================================
# STEP 21: SAVE PROJECT OUTPUTS
# ============================================================

def save_outputs(comparison_df, prediction_results,
                 evaluation_results, insights):
    """
    Save model comparison, predictions, evaluation metrics,
    and business insights to the project's results directory.
    """

    print("\n" + "=" * 60)
    print("STEP 21: SAVE PROJECT OUTPUTS")
    print("=" * 60)

    # Create the results directory if it does not exist.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Save model comparison results
    # --------------------------------------------------------

    comparison_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print(f"Model comparison saved: {comparison_path}")

    # --------------------------------------------------------
    # 2. Save order prediction results
    # --------------------------------------------------------

    predictions_path = os.path.join(
        RESULTS_DIR,
        "delivery_predictions.csv"
    )

    prediction_results.to_csv(
        predictions_path,
        index=True,
        index_label="order_index"
    )

    print(f"Delivery predictions saved: {predictions_path}")

    # --------------------------------------------------------
    # 3. Save model evaluation metrics
    # --------------------------------------------------------

    evaluation_metrics = {
        "average_precision": float(
            evaluation_results["average_precision"]
        ),
        "precision": float(
            evaluation_results["precision"]
        ),
        "recall": float(
            evaluation_results["recall"]
        ),
        "f1_score": float(
            evaluation_results["f1_score"]
        )
    }

    evaluation_path = os.path.join(
        RESULTS_DIR,
        "model_evaluation_metrics.csv"
    )

    evaluation_df = pd.DataFrame(
        list(evaluation_metrics.items()),
        columns=["Metric", "Score"]
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False
    )

    print(f"Evaluation metrics saved: {evaluation_path}")

    # --------------------------------------------------------
    # 4. Save business insights as JSON
    # --------------------------------------------------------

    insights_path = os.path.join(
        RESULTS_DIR,
        "business_insights.json"
    )

    with open(insights_path, "w", encoding="utf-8") as file:
        json.dump(
            insights,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Business insights saved: {insights_path}")

    # --------------------------------------------------------
    # 5. Display saved output files
    # --------------------------------------------------------

    print("\nAll output files saved successfully.")

    return {
        "model_comparison": comparison_path,
        "delivery_predictions": predictions_path,
        "evaluation_metrics": evaluation_path,
        "business_insights": insights_path
    }


# ============================================================
# STEP 22: DISPLAY FINAL PROJECT SUMMARY
# ============================================================

def display_summary(
    best_model_name,
    ml_data,
    evaluation_results,
    insights,
    output_paths
):
    """
    Display a final summary of the analytics and
    delivery-delay prediction project.
    """

    print("\n")
    print("=" * 70)
    print("       OLIST E-COMMERCE ANALYTICS & DELIVERY DELAY PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Project overview
    # --------------------------------------------------------

    print("\n1. PROJECT OVERVIEW")
    print("-" * 70)

    print("Project       : Olist E-Commerce Analytics")
    print("Objective     : Analyze orders and predict delivery delays")
    print("ML Task       : Binary Classification")
    print("Target        : is_late")
    print("Class 0       : On-Time Delivery")
    print("Class 1       : Late Delivery")

    # --------------------------------------------------------
    # 2. Dataset summary
    # --------------------------------------------------------

    print("\n2. MACHINE LEARNING DATA")
    print("-" * 70)

    print(f"Training Samples : {len(ml_data['X_train']):,}")
    print(f"Testing Samples  : {len(ml_data['X_test']):,}")
    print(f"Input Features   : {len(ml_data['feature_columns'])}")

    # --------------------------------------------------------
    # 3. Selected model
    # --------------------------------------------------------

    print("\n3. SELECTED MACHINE LEARNING MODEL")
    print("-" * 70)

    print(f"Best Model: {best_model_name}")

    # --------------------------------------------------------
    # 4. Test-set performance
    # --------------------------------------------------------

    print("\n4. FINAL TEST PERFORMANCE")
    print("-" * 70)

    print(
        f"Average Precision : "
        f"{evaluation_results['average_precision']:.4f}"
    )
    print(
        f"Precision         : "
        f"{evaluation_results['precision']:.4f}"
    )
    print(
        f"Recall            : "
        f"{evaluation_results['recall']:.4f}"
    )
    print(
        f"F1-Score          : "
        f"{evaluation_results['f1_score']:.4f}"
    )

    # --------------------------------------------------------
    # 5. Business insights
    # --------------------------------------------------------

    print("\n5. DELIVERY INSIGHTS")
    print("-" * 70)

    print(
        f"Test Orders Analyzed  : "
        f"{insights['total_orders']:,}"
    )
    print(
        f"Actual Late Orders    : "
        f"{insights['actual_late_orders']:,}"
    )
    print(
        f"Actual Late Rate      : "
        f"{insights['actual_late_percentage']:.2f}%"
    )
    print(
        f"Predicted Late Orders : "
        f"{insights['predicted_late_orders']:,}"
    )
    print(
        f"High-Risk Orders      : "
        f"{insights['high_risk_orders']:,}"
    )

    # --------------------------------------------------------
    # 6. Recommendations
    # --------------------------------------------------------

    print("\n6. BUSINESS RECOMMENDATIONS")
    print("-" * 70)

    for number, recommendation in enumerate(
        insights["recommendations"],
        start=1
    ):
        print(f"{number}. {recommendation}")

    # --------------------------------------------------------
    # 7. Saved output files
    # --------------------------------------------------------

    print("\n7. GENERATED OUTPUT FILES")
    print("-" * 70)

    for output_name, output_path in output_paths.items():
        print(f"{output_name}: {output_path}")

    # --------------------------------------------------------
    # 8. Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("       PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Analytics, model evaluation, predictions, and exports completed.")
    print("=" * 70)



#----------------------------------------------------------------------
#Function Name : display_footer
#Description   : Displays the final completion message for the
#                Olist E-Commerce Analytics and Delivery Delay
#                Prediction project.
#Input         : None.
#Output        : Displays the project completion footer.
#Author        : Ishwari Vijaykumar Surve
#Date          : 21/10/2026
#----------------------------------------------------------------------

def display_footer():

    print("\n" + "=" * 60)
    print("   OLIST E-COMMERCE ANALYTICS & DATA SCIENCE PROJECT")
    print("=" * 60)
    print("Data Loading        : Completed")
    print("Data Cleaning       : Completed")
    print("Exploratory Analysis: Completed")
    print("RFM Analysis        : Completed")
    print("Feature Engineering : Completed")
    print("Machine Learning    : Completed")
    print("Model Evaluation    : Completed")
    print("Delivery Prediction : Completed")
    print("Business Insights   : Generated")
    print("Output Files        : Saved")
    print("-" * 60)
    print("   THANK YOU FOR USING THIS PROJECT!")
    print("=" * 60)



# ============================================================
# MAIN FUNCTION: EXECUTE COMPLETE PROJECT PIPELINE
# ============================================================

def main():
    """
    Execute the complete Olist E-Commerce Analytics and
    Delivery Delay Prediction pipeline.
    """

    print("\n" + "=" * 70)
    print("OLIST E-COMMERCE ANALYTICS & DELIVERY DELAY PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Load datasets
    # --------------------------------------------------------

    datasets = load_datasets(RAW_DIR)

    # --------------------------------------------------------
    # STEP 2: Understand datasets
    # --------------------------------------------------------

    understand_datasets(datasets)

    # --------------------------------------------------------
    # STEP 3: Check missing values
    # --------------------------------------------------------

    check_missing_values(datasets)

    # --------------------------------------------------------
    # STEP 4: Check duplicates
    # --------------------------------------------------------

    check_duplicates(datasets)

    # --------------------------------------------------------
    # STEP 5: Clean datasets
    # --------------------------------------------------------

    cleaned_datasets = clean_datasets(datasets)

    # --------------------------------------------------------
    # STEP 6: Merge datasets
    # --------------------------------------------------------

    master = merge_datasets(cleaned_datasets)

    # --------------------------------------------------------
    # STEP 7: Feature engineering
    # --------------------------------------------------------

    master = feature_engineering(master)

    # --------------------------------------------------------
    # STEP 8: Perform exploratory data analysis
    # --------------------------------------------------------

    perform_eda(master, cleaned_datasets)

    # --------------------------------------------------------
    # STEP 9: RFM analysis
    # --------------------------------------------------------

    rfm_results = rfm_analysis(master)

    # --------------------------------------------------------
    # STEP 10: Create target variable
    # --------------------------------------------------------

    target_data = create_target(master)

    # --------------------------------------------------------
    # STEP 11: Remove target leakage
    # --------------------------------------------------------

    model_data = remove_leakage(target_data)

    # --------------------------------------------------------
    # STEP 12: Prepare machine learning data
    # --------------------------------------------------------

    ml_data = prepare_ml_data(model_data)

    # --------------------------------------------------------
    # STEP 13: Build machine learning models
    # --------------------------------------------------------

    models = build_models(ml_data)

    # --------------------------------------------------------
    # STEP 14: Compare models using cross-validation
    # --------------------------------------------------------

    comparison_df = compare_models(models, ml_data)

    # --------------------------------------------------------
    # STEP 15: Select the best model
    # --------------------------------------------------------

    best_model_name, best_model = select_best_model(
        comparison_df,
        models
    )

    # --------------------------------------------------------
    # STEP 16.1: Train the best model
    # --------------------------------------------------------

    trained_model = train_best_model(
        best_model_name,
        best_model,
        ml_data
    )

    # --------------------------------------------------------
    # STEP 16.2: Save the trained model
    # --------------------------------------------------------

    model_path = save_model(
        trained_model,
        best_model_name
    )

    # --------------------------------------------------------
    # STEP 16.3: Load the saved model
    # --------------------------------------------------------

    loaded_model = load_model(model_path)

    # --------------------------------------------------------
    # STEP 17: Evaluate model on unseen test data
    # --------------------------------------------------------

    evaluation_results = evaluate_model(
        loaded_model,
        ml_data
    )

    # --------------------------------------------------------
    # STEP 18: Plot evaluation results
    # --------------------------------------------------------

    plot_results(evaluation_results)

    # --------------------------------------------------------
    # STEP 19: Predict delivery outcomes
    # --------------------------------------------------------

    prediction_results = predict_orders(
        loaded_model,
        ml_data
    )

    # --------------------------------------------------------
    # STEP 20: Generate business insights
    # --------------------------------------------------------

    insights = generate_insights(
        evaluation_results,
        prediction_results
    )

    # --------------------------------------------------------
    # STEP 21: Save project outputs
    # --------------------------------------------------------

    output_paths = save_outputs(
        comparison_df,
        prediction_results,
        evaluation_results,
        insights
    )

    # --------------------------------------------------------
    # STEP 22: Display final project summary
    # --------------------------------------------------------

    display_summary(
        best_model_name,
        ml_data,
        evaluation_results,
        insights,
        output_paths
    )

    # --------------------------------------------------------
    # FINAL STEP: DISPLAY PROJECT COMPLETION FOOTER
    # --------------------------------------------------------

    display_footer()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()














