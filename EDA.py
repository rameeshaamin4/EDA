import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="📦 E-commerce Data Analysis", layout="wide")

# Title
st.title("📊 Comprehensive E-commerce Data Analysis App")

# Intro
st.markdown("""
Welcome to the **E-commerce Data Analysis App** 🎉  
Here, we’ll explore sales patterns, customer behaviors, and product performance.  
Upload your dataset and let’s begin the analysis!
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert invoice_date if exists
    if "invoicedate" in df.columns:
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    # Show dataset preview
    st.subheader("👀 Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("📌 Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing values
    st.subheader("🔎 Missing Values")
    st.write(df.isnull().sum())

    # ================================================================
    # SALES ANALYSIS
    # ================================================================
    st.markdown("## 📈 Sales Trends Over Time")

    if "invoicedate" in df.columns and "quantity" in df.columns and "unitprice" in df.columns:
        df["sales"] = df["quantity"] * df["unitprice"]
        sales_trend = df.groupby(df["invoicedate"].dt.date)["sales"].sum().reset_index()

        fig = px.line(sales_trend, x="invoicedate", y="sales", title="Total Sales Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # TOP PRODUCTS
    # ================================================================
    st.markdown("## 🏆 Top Selling Products")

    if "description" in df.columns and "quantity" in df.columns:
        top_products = df.groupby("description")["quantity"].sum().reset_index().sort_values(by="quantity", ascending=False).head(10)

        fig = px.bar(top_products, x="description", y="quantity", title="Top 10 Products by Quantity Sold")
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # REVENUE BY COUNTRY
    # ================================================================
    st.markdown("## 🌍 Revenue by Country")

    if "country" in df.columns:
        revenue_country = df.groupby("country")["sales"].sum().reset_index().sort_values(by="sales", ascending=False).head(10)

        fig = px.bar(revenue_country, x="country", y="sales", title="Top 10 Countries by Revenue")
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # CUSTOMER ANALYSIS
    # ================================================================
    st.markdown("## 👥 Customer Analysis")

    if "customerid" in df.columns:
        customer_revenue = df.groupby("customerid")["sales"].sum().reset_index().sort_values(by="sales", ascending=False).head(10)

        fig = px.bar(customer_revenue, x="customerid", y="sales", title="Top 10 Customers by Revenue")
        st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # COLUMN-WISE INTERACTIVE ANALYSIS
    # ================================================================
    st.markdown("## 🔍 Interactive Column-wise Analysis")

    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of **{column}**:")
        st.write(df[column].describe())

        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of **{column}**:")
        st.write(df[column].value_counts().head(20))
        fig = px.bar(df[column].value_counts().head(20), title=f"Top 20 Categories in {column}")
        st.plotly_chart(fig, use_container_width=True)
