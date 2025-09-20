import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime

# Streamlit page config
st.set_page_config(page_title="E-Commerce Analysis Dashboard", layout="wide")

# Ignore warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce_dataset.csv", encoding="ISO-8859-1")
    # rename columns
    df.rename(index=str, columns={
        'InvoiceNo': 'invoice_num',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'UnitPrice': 'unit_price',
        'CustomerID': 'cust_id',
        'Country': 'country'
    }, inplace=True)

    # clean data
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['description'] = df['description'].astype(str).str.lower()
    df = df.dropna()
    df['cust_id'] = df['cust_id'].astype('int64')
    df = df[df['quantity'] > 0]
    df['amount_spent'] = df['quantity'] * df['unit_price']

    # insert time features
    df.insert(loc=2, column='year_month', value=df['invoice_date'].map(lambda x: 100 * x.year + x.month))
    df.insert(loc=3, column='month', value=df.invoice_date.dt.month)
    df.insert(loc=4, column='day', value=(df.invoice_date.dt.dayofweek) + 1)
    df.insert(loc=5, column='hour', value=df.invoice_date.dt.hour)
    
    # reorder columns
    df = df[['invoice_num','invoice_date','year_month','month','day','hour','stock_code','description','quantity','unit_price','amount_spent','cust_id','country']]
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect("Select Country", options=df['country'].unique(), default=df['country'].unique())
df = df[df['country'].isin(country_filter)]

st.title("📊 E-Commerce Dataset Analysis Dashboard")
st.write("This dashboard explores the transactional data for e-commerce analysis.")

# Show data sample
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(20))

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Orders", df['invoice_num'].nunique())
with col2:
    st.metric("Unique Customers", df['cust_id'].nunique())
with col3:
    st.metric("Total Revenue ($)", round(df['amount_spent'].sum(), 2))
with col4:
    st.metric("Unique Countries", df['country'].nunique())

# Orders per customer
st.subheader("Number of Orders by Customers")
orders = df.groupby(by=['cust_id','country'], as_index=False)['invoice_num'].count()
fig, ax = plt.subplots(figsize=(15,6))
plt.plot(orders.cust_id, orders.invoice_num)
plt.xlabel('Customer ID')
plt.ylabel('Number of Orders')
plt.title('Orders per Customer')
st.pyplot(fig)

# Money spent per customer
st.subheader("Money Spent by Customers")
money_spent = df.groupby(by=['cust_id','country'], as_index=False)['amount_spent'].sum()
fig, ax = plt.subplots(figsize=(15,6))
plt.plot(money_spent.cust_id, money_spent.amount_spent)
plt.xlabel('Customer ID')
plt.ylabel('Total Money Spent ($)')
plt.title('Money Spent per Customer')
st.pyplot(fig)

# Monthly orders
st.subheader("Monthly Orders")
monthly_orders = df.groupby('invoice_num')['year_month'].unique().value_counts().sort_index()
fig, ax = plt.subplots(figsize=(15,6))
monthly_orders.plot(kind='bar', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.title('Orders per Month')
st.pyplot(fig)

# Orders by day of week
st.subheader("Orders by Day of Week")
daily_orders = df.groupby('invoice_num')['day'].unique().value_counts().sort_index()
fig, ax = plt.subplots(figsize=(15,6))
daily_orders.plot(kind='bar', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Day of Week')
plt.ylabel('Number of Orders')
plt.title('Orders by Day of Week')
plt.xticks(ticks=range(0,6), labels=['Mon','Tue','Wed','Thu','Fri','Sat'], rotation=0)
st.pyplot(fig)

# Orders by hour
st.subheader("Orders by Hour")
hourly_orders = df.groupby('invoice_num')['hour'].unique().value_counts().sort_index()
fig, ax = plt.subplots(figsize=(15,6))
hourly_orders.plot(kind='bar', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Hour of Day')
plt.ylabel('Number of Orders')
plt.title('Orders by Hour')
st.pyplot(fig)

# Unit price distribution
st.subheader("Unit Price Distribution")
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x=df['unit_price'], ax=ax)
plt.title("Distribution of Unit Price")
st.pyplot(fig)

# Free items
st.subheader("Free Items Frequency")
df_free = df[df['unit_price'] == 0]
fig, ax = plt.subplots(figsize=(12,6))
df_free.year_month.value_counts().sort_index().plot(kind='bar', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.title('Free Items over Months')
st.pyplot(fig)

# Orders by Country (with UK)
st.subheader("Orders by Country (with UK)")
fig, ax = plt.subplots(figsize=(15,8))
df.groupby('country')['invoice_num'].count().sort_values().plot(kind='barh', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Orders by Country (with UK)')
st.pyplot(fig)

# Orders by Country (without UK)
st.subheader("Orders by Country (without UK)")
fig, ax = plt.subplots(figsize=(15,8))
country_orders_noUK = df.groupby('country')['invoice_num'].count().sort_values()
if 'United Kingdom' in country_orders_noUK.index:
    country_orders_noUK = country_orders_noUK.drop('United Kingdom')
country_orders_noUK.plot(kind='barh', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.title('Orders by Country (without UK)')
st.pyplot(fig)

# Revenue by Country
st.subheader("Revenue by Country")
fig, ax = plt.subplots(figsize=(15,8))
df.groupby('country')['amount_spent'].sum().sort_values().plot(kind='barh', color=sns.color_palette()[0], ax=ax)
plt.xlabel('Revenue ($)')
plt.ylabel('Country')
plt.title('Revenue by Country')
st.pyplot(fig)
