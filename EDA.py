import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Streamlit page config
st.set_page_config(page_title="E-Commerce Analysis Dashboard", layout="wide")

# Ignore warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Load dataset
def load_data():
    df = pd.read_csv("ecommerce_dataset.csv", encoding="ISO-8859-1")
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

    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['description'] = df['description'].astype(str).str.lower()
    df = df.dropna()
    df['cust_id'] = df['cust_id'].astype('int64')
    df = df[df['quantity'] > 0]
    df['amount_spent'] = df['quantity'] * df['unit_price']

    # Add time features
    df['year_month'] = df['invoice_date'].dt.to_period('M')
    df['month'] = df['invoice_date'].dt.month
    df['day'] = df['invoice_date'].dt.dayofweek + 1
    df['hour'] = df['invoice_date'].dt.hour
    
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Select Country", options=df['country'].unique(), default=df['country'].unique())
df = df[df['country'].isin(country_filter)]

st.title("📊 E-Commerce Dataset Analysis Dashboard")

# Show data sample
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(20))

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", df['invoice_num'].nunique())
col2.metric("Unique Customers", df['cust_id'].nunique())
col3.metric("Total Revenue ($)", round(df['amount_spent'].sum(), 2))
col4.metric("Unique Countries", df['country'].nunique())

# Orders per customer
st.subheader("Number of Orders by Customers")
orders = df.groupby('cust_id')['invoice_num'].count()
fig, ax = plt.subplots(figsize=(12,6))
orders.plot(ax=ax)
ax.set_xlabel("Customer ID")
ax.set_ylabel("Number of Orders")
st.pyplot(fig)

# Money spent per customer
st.subheader("Money Spent by Customers")
money_spent = df.groupby('cust_id')['amount_spent'].sum()
fig, ax = plt.subplots(figsize=(12,6))
money_spent.plot(ax=ax)
ax.set_xlabel("Customer ID")
ax.set_ylabel("Money Spent ($)")
st.pyplot(fig)

# Monthly orders
st.subheader("Monthly Orders")
monthly_orders = df.groupby('year_month')['invoice_num'].nunique()
fig, ax = plt.subplots(figsize=(12,6))
monthly_orders.plot(kind='bar', ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("Month")
ax.set_ylabel("Number of Orders")
st.pyplot(fig)

# Orders by day of week
st.subheader("Orders by Day of Week")
daily_orders = df.groupby('day')['invoice_num'].nunique()
fig, ax = plt.subplots(figsize=(12,6))
daily_orders.plot(kind='bar', ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("Day of Week (1=Mon ... 7=Sun)")
ax.set_ylabel("Orders")
st.pyplot(fig)

# Orders by hour
st.subheader("Orders by Hour")
hourly_orders = df.groupby('hour')['invoice_num'].nunique()
fig, ax = plt.subplots(figsize=(12,6))
hourly_orders.plot(kind='bar', ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Orders")
st.pyplot(fig)

# Unit price distribution
st.subheader("Unit Price Distribution")
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x=df['unit_price'], ax=ax)
st.pyplot(fig)

# Free items
st.subheader("Free Items Frequency")
df_free = df[df['unit_price'] == 0]
free_items = df_free.groupby('year_month')['invoice_num'].nunique()
fig, ax = plt.subplots(figsize=(12,6))
free_items.plot(kind='bar', ax=ax, color=sns.color_palette()[0])
ax.set_xlabel("Month")
ax.set_ylabel("Free Items Count")
st.pyplot(fig)

# Orders by Country (with UK)
st.subheader("Orders by Country (with UK)")
orders_country = df.groupby('country')['invoice_num'].nunique().sort_values()
fig, ax = plt.subplots(figsize=(12,6))
orders_country.plot(kind='barh', ax=ax, color=sns.color_palette()[0])
st.pyplot(fig)

# Orders by Country (without UK)
st.subheader("Orders by Country (without UK)")
if 'United Kingdom' in orders_country.index:
    orders_country_noUK = orders_country.drop('United Kingdom')
else:
    orders_country_noUK = orders_country
fig, ax = plt.subplots(figsize=(12,6))
orders_country_noUK.plot(kind='barh', ax=ax, color=sns.color_palette()[0])
st.pyplot(fig)

# Revenue by Country
st.subheader("Revenue by Country")
revenue_country = df.groupby('country')['amount_spent'].sum().sort_values()
fig, ax = plt.subplots(figsize=(12,6))
revenue_country.plot(kind='barh', ax=ax, color=sns.color_palette()[0])
st.pyplot(fig)
