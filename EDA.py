import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Config
st.set_page_config(page_title="E-Commerce Data Analysis", layout="wide")
sns.set_style("whitegrid")

# Title
st.title("📊 E-Commerce Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload your E-Commerce CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    # Rename columns for consistency
    df.rename(columns={
        'InvoiceNo': 'invoice_num',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'UnitPrice': 'unit_price',
        'CustomerID': 'cust_id',
        'Country': 'country'
    }, inplace=True)

    # Convert date column
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')

    # Standardize description
    df['description'] = df['description'].astype(str).str.lower()

    # Drop missing values
    df_new = df.dropna()
    df_new['cust_id'] = df_new['cust_id'].astype('int64')

    # Add new features
    df_new = df_new[df_new.quantity > 0]
    df_new['amount_spent'] = df_new['quantity'] * df_new['unit_price']
    df_new = df_new[['invoice_num', 'invoice_date', 'stock_code', 'description',
                     'quantity', 'unit_price', 'amount_spent', 'cust_id', 'country']]

    df_new.insert(2, 'year_month', df_new['invoice_date'].dt.strftime('%Y%m'))
    df_new.insert(3, 'month', df_new['invoice_date'].dt.month)
    df_new.insert(4, 'day', df_new['invoice_date'].dt.dayofweek + 1)
    df_new.insert(5, 'hour', df_new['invoice_date'].dt.hour)

    # -------------------------
    # Dataset Preview
    # -------------------------
    st.subheader("📂 Dataset Preview")
    st.write(df_new.head())

    # -------------------------
    # Missing Values
    # -------------------------
    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    # -------------------------
    # Summary Statistics
    # -------------------------
    st.subheader("📈 Summary Statistics")
    st.write(df_new.describe())

    # -------------------------
    # Customer Orders Analysis
    # -------------------------
    st.subheader("🛒 Number of Orders per Customer")
    orders = df_new.groupby(['cust_id'])['invoice_num'].count().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(orders['cust_id'], orders['invoice_num'], color="blue")
    ax.set_title("Orders by Customer")
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Number of Orders")
    st.pyplot(fig)

    st.write("Top 5 Customers by Number of Orders:")
    st.write(orders.sort_values(by='invoice_num', ascending=False).head())

    # -------------------------
    # Money Spent per Customer
    # -------------------------
    st.subheader("💰 Money Spent per Customer")
    money_spent = df_new.groupby('cust_id')['amount_spent'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(money_spent['cust_id'], money_spent['amount_spent'], color="green")
    ax.set_title("Money Spent by Customer")
    ax.set_xlabel("Customer ID")
    ax.set_ylabel("Amount Spent")
    st.pyplot(fig)

    st.write("Top 5 Customers by Money Spent:")
    st.write(money_spent.sort_values(by='amount_spent', ascending=False).head())

    # -------------------------
    # Orders by Month
    # -------------------------
    st.subheader("📅 Orders by Month")
    monthly_orders = df_new.groupby('year_month')['invoice_num'].nunique()

    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_orders.plot(kind='bar', color="orange", ax=ax)
    ax.set_title("Number of Orders per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    st.pyplot(fig)

    # -------------------------
    # Orders by Day of Week
    # -------------------------
    st.subheader("📆 Orders by Day of Week")
    day_orders = df_new.groupby('day')['invoice_num'].nunique()

    fig, ax = plt.subplots(figsize=(10, 5))
    day_orders.plot(kind='bar', color="purple", ax=ax)
    ax.set_title("Orders by Day of Week (Mon=1 ... Sun=7)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Orders")
    st.pyplot(fig)

    # -------------------------
    # Orders by Hour
    # -------------------------
    st.subheader("⏰ Orders by Hour of Day")
    hour_orders = df_new.groupby('hour')['invoice_num'].nunique()

    fig, ax = plt.subplots(figsize=(10, 5))
    hour_orders.plot(kind='bar', color="red", ax=ax)
    ax.set_title("Orders by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Orders")
    st.pyplot(fig)

    # -------------------------
    # Country-wise Orders
    # -------------------------
    st.subheader("🌍 Orders by Country")
    country_orders = df_new.groupby('country')['invoice_num'].count().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    country_orders.plot(kind='barh', color="skyblue", ax=ax)
    ax.set_title("Orders by Country")
    ax.set_xlabel("Orders")
    st.pyplot(fig)

    # -------------------------
    # Country-wise Spending
    # -------------------------
    st.subheader("💵 Spending by Country")
    country_spent = df_new.groupby('country')['amount_spent'].sum().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    country_spent.plot(kind='barh', color="darkgreen", ax=ax)
    ax.set_title("Total Spending by Country")
    ax.set_xlabel("Money Spent")
    st.pyplot(fig)
